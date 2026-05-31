import os, sys
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path: sys.path.insert(0, _root)

"""
live_demo_xsens.py  v3
----------------------
用 Xsens MTw Awinda 6 个 IMU 替换官方 Noitom IMU，复现 TransPose live demo。

传感器佩戴顺序（TransPose 要求）：
  0: left forearm    11837972
  1: right forearm   11837980
  2: left lower leg  11837965
  3: right lower leg 11837973
  4: head            11837964
  5: pelvis          11837969

运行方式：
  python live_demo_xsens.py           # 正式运行（需要 Unity 连接 127.0.0.1:8888）
  python live_demo_xsens.py --scan    # 只扫描传感器 ID，不启动推理
  python live_demo_xsens.py --debug   # 调试：验证数据，不需要 Unity
  python live_demo_xsens.py --record  # 录制一段后用 Python 离线可视化，不需要 Unity
"""

import os
import socket
import sys
import threading
import time
from datetime import datetime

import numpy as np
import torch
from pygame.time import Clock

import xsensdeviceapi as xda

from articulate.math import quaternion_to_rotation_matrix, rotation_matrix_to_axis_angle
import config
from net import TransPoseNet

# ── 传感器 ID（decimal int）──────────────────────────────────────────────────
SENSOR_IDS = {
    0: 11837972,   # left forearm
    1: 11837980,   # right forearm
    2: 11837965,   # left lower leg
    3: 11837973,   # right lower leg
    4: 11837964,   # head
    5: 11837969,   # pelvis
}

# ── acceleration frame 开关 ──────────────────────────────────────────────────
# False = calibratedFreeAcceleration() 已是 global frame（默认）
# True  = sensor-local frame，需要用 ori 转到 world frame
# 跑 --debug 后看 acc norm 是否接近 0 来确认
ACC_IS_SENSOR_LOCAL = False

# ── 模型初始化 ───────────────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
inertial_poser = TransPoseNet(num_past_frame=20, num_future_frame=5).to(device)
inertial_poser.eval()

running = False
start_recording = False


# ── 辅助：quaternion 均值（处理符号歧义）────────────────────────────────────
def mean_quaternion_np(qs: np.ndarray) -> np.ndarray:
    """qs: [T, 4] in [w,x,y,z]，返回 [4] normalized"""
    qs = qs.copy()
    for i in range(1, len(qs)):
        if np.dot(qs[0], qs[i]) < 0:
            qs[i] = -qs[i]
    q = qs.mean(axis=0)
    return q / np.linalg.norm(q)


# ── Xsens 数据回调 ───────────────────────────────────────────────────────────
class XsensCallback(xda.XsCallback):
    def __init__(self, max_buffer=60):
        super().__init__()
        self._data = {}
        self._max  = max_buffer
        self._lock = threading.Lock()

    def onLiveDataAvailable(self, dev, packet):
        if not packet.containsOrientation() or not packet.containsCalibratedData():
            return
        did = dev.deviceId().toInt()
        q   = packet.orientationQuaternion()       # w,x,y,z
        a   = packet.freeAcceleration()  # x,y,z  m/s²  (global frame)
        q_arr = np.array(q, dtype=np.float32)  # [w,x,y,z]
        a_arr = np.array([a[0], a[1], a[2]],           dtype=np.float32)
        with self._lock:
            buf = self._data.setdefault(did, [])
            buf.append((q_arr, a_arr))
            if len(buf) > self._max:
                buf.pop(0)

    def get_latest(self, ordered_ids):
        with self._lock:
            qs, as_ = [], []
            for did in ordered_ids:
                buf = self._data.get(did)
                if not buf:
                    return None, None
                qs.append(buf[-1][0])
                as_.append(buf[-1][1])
            return np.array(qs, np.float32), np.array(as_, np.float32)

    def get_mean_n_frames(self, ordered_ids, n=200):
        with self._lock:
            qs, as_ = [], []
            for did in ordered_ids:
                buf = self._data.get(did, [])
                if not buf:
                    return None, None
                frames = buf[-n:]
                qs.append(mean_quaternion_np(np.array([f[0] for f in frames])))
                as_.append(np.array([f[1] for f in frames]).mean(axis=0))
            return np.array(qs, np.float32), np.array(as_, np.float32)


# ── XsensIMUSet ──────────────────────────────────────────────────────────────
class XsensIMUSet:
    def __init__(self, sensor_ids: dict, buffer_len=26):
        self._ordered_ids = [sensor_ids[i] for i in range(6)]
        self._buffer_len  = buffer_len
        self._quat_buf    = []
        self._acc_buf     = []
        self._reading     = False
        self._thread      = None
        self.clock        = Clock()
        self._ctrl        = xda.XsControl.construct()
        self._cb          = XsensCallback(max_buffer=60)
        self._master      = None
        self._connect()

    def _connect(self):
        print("扫描 Xsens 端口...")
        master_port = next(
            (p for p in xda.XsScanner.scanPorts() if p.deviceId().isWirelessMaster()), None)
        if master_port is None:
            raise RuntimeError("未找到 Awinda Station，请检查 USB。")

        if not self._ctrl.openPort(master_port.portName(), master_port.baudrate()):
            raise RuntimeError(f"无法打开端口 {master_port.portName()}")

        self._master = self._ctrl.device(master_port.deviceId())
        self._master.addCallbackHandler(self._cb)

        if not self._master.gotoConfig():
            raise RuntimeError("Master 无法进入 config 模式")

        # 开启 radio（Awinda 必须）
        if self._master.isRadioEnabled():
            self._master.disableRadio()
        enabled = False
        for ch in [13, 11, 15, 19, 20]:
            if self._master.enableRadio(ch):
                print(f"Radio enabled on channel {ch}")
                enabled = True
                break
        if not enabled:
            raise RuntimeError("无法开启 Awinda radio，请关闭 MT Manager 再试。")

        print("等待 MTw 传感器连接（最多 30 秒）...")
        deadline = time.time() + 30
        while time.time() < deadline:
            n = self._master.childCount()
            print(f"\r  已连接 {n}/6 个 MTw...", end='', flush=True)
            if n >= 6:
                break
            time.sleep(1)
        print()

        mtws = [self._master.children()[i] for i in range(self._master.children().size())]
        if len(mtws) < 6:
            raise RuntimeError(f"只连接了 {len(mtws)} 个 MTw，需要 6 个。")
        for m in mtws:
            m.addCallbackHandler(self._cb)

        online = {m.deviceId().toInt() for m in mtws}
        for idx, did in enumerate(self._ordered_ids):
            if did not in online:
                raise RuntimeError(
                    f"SENSOR_IDS[{idx}]={did} 不在已连接设备中。\n"
                    f"在线 decimal IDs: {sorted(online)}")
        print("所有传感器验证完毕。")

    def start_reading(self):
        if self._thread is not None:
            return
        if not self._master.gotoMeasurement():
            raise RuntimeError("无法进入 Measurement 模式")
        self._reading  = True
        self._quat_buf = []
        self._acc_buf  = []
        self._thread   = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        while self._reading:
            q, a = self._cb.get_latest(self._ordered_ids)
            if q is not None:
                drop = int(len(self._quat_buf) == self._buffer_len)
                self._quat_buf = self._quat_buf[drop:] + [q]
                self._acc_buf  = self._acc_buf[drop:]  + [a]
                self.clock.tick()
            time.sleep(1.0 / 200)

    def stop_reading(self):
        if self._thread:
            self._reading = False
            self._thread.join()
            self._thread = None
        self._master.gotoConfig()

    def get_current_buffer(self):
        """返回 q [buf,6,4], acc [buf,6,3]；空时返回正确 shape 的空 tensor"""
        if not self._quat_buf:
            return (torch.empty((0, 6, 4), dtype=torch.float),
                    torch.empty((0, 6, 3), dtype=torch.float))
        return (torch.tensor(np.array(self._quat_buf), dtype=torch.float),
                torch.tensor(np.array(self._acc_buf),  dtype=torch.float))

    def get_mean_measurement_of_n_second(self, num_seconds=3, buffer_len=200):
        old = self._buffer_len
        self._buffer_len = buffer_len
        self.start_reading()
        time.sleep(num_seconds)
        self.stop_reading()
        q_np, a_np = self._cb.get_mean_n_frames(self._ordered_ids, n=buffer_len)
        self._buffer_len = old
        if q_np is None:
            raise RuntimeError("校准期间未收到数据，请检查传感器。")
        return (torch.tensor(q_np, dtype=torch.float),
                torch.tensor(a_np, dtype=torch.float))

    def close(self):
        self.stop_reading()
        try:
            self._master.disableRadio()
            self._master.removeCallbackHandler(self._cb)
            self._ctrl.closePort(self._master.portName())
        except Exception:
            pass
        self._ctrl.destruct()


# ── 公共校准流程 ─────────────────────────────────────────────────────────────
def do_calibration(imu_set):
    """两步校准，返回 smpl2imu, device2bone, acc_offsets"""
    input('\n把 sensor 0（左前臂，ID=11837972）对齐身体参考系\n'
          '  x=左  y=上  z=前\n静止后按回车...')
    print('保持 3 秒...', end='', flush=True)
    oris0 = imu_set.get_mean_measurement_of_n_second(3, 200)[0][0]
    smpl2imu = quaternion_to_rotation_matrix(oris0).view(3, 3).t()
    print(f'  完成。smpl2imu =\n{smpl2imu.numpy().round(3)}')

    input('\n穿好全部 6 个传感器，按回车...')
    for i in range(3, 0, -1):
        print(f'\r站 T-pose，{i} 秒后开始...', end='', flush=True)
        time.sleep(1)
    print('\r保持 T-pose 3 秒...', end='', flush=True)
    oris_t, accs_t = imu_set.get_mean_measurement_of_n_second(3, 200)
    oris_t = quaternion_to_rotation_matrix(oris_t)

    device2bone = smpl2imu.matmul(oris_t).transpose(1, 2).matmul(torch.eye(3))
    if ACC_IS_SENSOR_LOCAL:
        acc_offsets = smpl2imu.matmul(oris_t.matmul(accs_t.unsqueeze(-1)))
    else:
        acc_offsets = smpl2imu.matmul(accs_t.unsqueeze(-1))
    print(f'  完成。acc_offsets（应接近 0）:\n{acc_offsets.squeeze(-1).numpy().round(4)}')
    print("device2bone[0] left forearm =", device2bone[0].numpy().round(3))
    print("device2bone[1] right forearm =", device2bone[1].numpy().round(3))
    return smpl2imu, device2bone, acc_offsets


def run_one_frame(ori_raw, acc_raw, smpl2imu, device2bone, acc_offsets):
    """单帧校准 + 归一化，返回 data_nn [1,72]"""
    ori_raw = quaternion_to_rotation_matrix(ori_raw).view(1, 6, 3, 3)
    if ACC_IS_SENSOR_LOCAL:
        acc_world = ori_raw.matmul(acc_raw.view(1, 6, 3, 1))
        acc_cal = (smpl2imu.matmul(acc_world) - acc_offsets).view(1, 6, 3)
    else:
        acc_cal = (smpl2imu.matmul(acc_raw.view(1, 6, 3, 1)) - acc_offsets).view(1, 6, 3)
    ori_cal = smpl2imu.matmul(ori_raw).matmul(device2bone)
    acc = torch.cat((acc_cal[:, :5] - acc_cal[:, 5:], acc_cal[:, 5:]), dim=1) \
              .bmm(ori_cal[:, -1]) / config.acc_scale
    ori = torch.cat((ori_cal[:, 5:].transpose(2, 3).matmul(ori_cal[:, :5]),
                     ori_cal[:, 5:]), dim=1)
    return torch.cat((acc.view(-1, 18), ori.view(-1, 54)), dim=1)


# ── 键盘输入线程 ─────────────────────────────────────────────────────────────
def get_input():
    global running, start_recording
    while running:
        c = input()
        if c == 'q':
            running = False
        elif c == 'r':
            start_recording = True
        elif c == 's':
            start_recording = False


# ── 扫描工具 ─────────────────────────────────────────────────────────────────
def scan_sensors():
    ctrl = xda.XsControl.construct()
    for p in xda.XsScanner.scanPorts():
        if not p.deviceId().isWirelessMaster():
            continue
        ctrl.openPort(p.portName(), p.baudrate())
        m = ctrl.device(p.deviceId())
        m.gotoConfig()
        if m.isRadioEnabled():
            m.disableRadio()
        m.enableRadio(13)
        print(f"Master: {p.deviceId().toXsString()}  ({p.portName()})")
        print("等待传感器连接 10 秒...")
        time.sleep(10)
        for i in range(m.childCount()):
            c2 = m.child(i)
            print(f"  [{i}]  decimal={c2.deviceId().toInt()}  "
                  f"hex={c2.deviceId().toXsString()}")
        m.disableRadio()
        ctrl.closePort(p.portName())
    ctrl.destruct()


# ── main ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':

    # ── --scan ───────────────────────────────────────────────────────────────
    if '--scan' in sys.argv:
        scan_sensors()
        sys.exit(0)

    imu_set = XsensIMUSet(SENSOR_IDS, buffer_len=1)

    # ── --debug：只验证数据，不进模型，不等 Unity ─────────────────────────────
    if '--debug' in sys.argv:
        print("\n[DEBUG] 读取 3 秒，打印传感器数据统计...")
        imu_set.start_reading()
        time.sleep(3)
        q, a = imu_set.get_current_buffer()

        print(f"\nordered sensor IDs : {imu_set._ordered_ids}")
        print(f"received IDs       : {sorted(imu_set._cb._data.keys())}")
        print(f"q shape            : {q.shape}   ← 应为 [1, 6, 4]")
        print(f"a shape            : {a.shape}   ← 应为 [1, 6, 3]")
        if q.shape[0] > 0:
            print(f"q norm (各传感器)  : {q[-1].norm(dim=-1).numpy().round(4)}  ← 应全为 1.0")
            print(f"acc (各传感器)     :\n{a[-1].numpy().round(4)}")
            print(f"acc norm           : {a[-1].norm(dim=-1).numpy().round(4)}")
            print("  → free acc 静止时应接近 0，若接近 9.8 说明含重力")
        else:
            print("[警告] buffer 为空，请检查传感器是否正常发送数据。")

        imu_set.close()
        sys.exit(0)

    # ── --record：录制 + 独立文件夹 + 时间戳 + TLV + 渲染视频 ───────────────
    if '--record' in sys.argv:
        import articulate as art
        import json, csv
        import smplx as _smplx
        import cv2
        from smplx.joint_names import SMPLH_JOINT_NAMES
        from config import paths

        smpl2imu, device2bone, acc_offsets = do_calibration(imu_set)
        imu_set.start_reading()

        rec_sec = 180
        session_start = datetime.now()
        session_tag   = session_start.strftime('%Y%m%d_%H-%M-%S')

        # 每次录制建独立文件夹
        save_dir = f'data/imu_recordings/{session_tag}'
        os.makedirs(save_dir, exist_ok=True)
        print(f'\n录制会话：{session_tag}')
        print(f'保存目录：{save_dir}')
        print(f'系统时间：{session_start.isoformat()}')
        print(f'开始录制 {rec_sec} 秒，请做动作...')

        frames      = []
        timestamps  = []
        raw_acc_buf = []

        t_end = time.time() + rec_sec
        while time.time() < t_end:
            ori_raw, acc_raw = imu_set.get_current_buffer()
            if ori_raw.shape[0] == 0:
                time.sleep(0.01)
                continue
            ts = time.time()
            with torch.no_grad():
                data_nn = run_one_frame(ori_raw, acc_raw,
                                        smpl2imu, device2bone, acc_offsets)
            frames.append(data_nn.detach().cpu())
            timestamps.append(ts)
            raw_acc_buf.append(acc_raw[-1, 5].numpy().tolist())
            time.sleep(1.0 / 60)

        imu_set.close()
        T = len(frames)
        print(f'录制完成，共 {T} 帧。推理中...')

        all_data = torch.cat(frames, dim=0).to(device)
        with torch.no_grad():
            pose, tran = inertial_poser.forward_offline(all_data)

        # pose / tran
        torch.save(pose, f'{save_dir}/pose.pt')
        torch.save(tran, f'{save_dir}/tran.pt')

        # 时间戳
        t0 = timestamps[0]
        actual_fps = T / (timestamps[-1] - t0) if T > 1 else 60.0
        frames_ts = []
        for i, t in enumerate(timestamps):
            dt = datetime.fromtimestamp(t)
            ff = int((dt.microsecond / 1e6) * 60)
            frames_ts.append({
                'frame':        i,
                'time_iso':     dt.strftime('%Y-%m-%d %H:%M:%S') + f':{ff:02d}',
                'time_unix':    round(t, 6),
                'time_rel_sec': round(t - t0, 4),
            })
        ts_data = {
            'session_start_iso':  session_start.isoformat(),
            'session_start_unix': session_start.timestamp(),
            'actual_fps':         round(actual_fps, 2),
            'total_frames':       T,
            'frames':             frames_ts,
        }
        with open(f'{save_dir}/timestamps.json', 'w') as f:
            json.dump(ts_data, f, indent=2)
        np.save(f'{save_dir}/pelvis_acc.npy', np.array(raw_acc_buf))

        # TLV zone
        tlv_rows = []
        try:
            with open('/home/yding263/Documents/tlv/dip18-master/data/subject_betas.json') as f:
                _sb = json.load(f)
            SUBJECT_BETAS  = torch.tensor(_sb['betas'], dtype=torch.float32).unsqueeze(0)
            SUBJECT_GENDER = _sb['gender']
            UP_AXIS        = int(_sb['up_axis'])

            _body = _smplx.create(
                '/home/yding263/Documents/tlv/babel/data/SMPL_models/smplh/SMPLH_FEMALE.pkl', model_type='smplh',
                gender=SUBJECT_GENDER, use_pca=False, flat_hand_mean=True,
                batch_size=T).to('cpu')
            from articulate.math import rotation_matrix_to_axis_angle
            _pose_aa = rotation_matrix_to_axis_angle(pose.view(T, 24, 3, 3)).view(T, 72)
            _out = _body(
                body_pose=_pose_aa[:, 3:66],
                global_orient=_pose_aa[:, :3],
                betas=SUBJECT_BETAS.repeat(T, 1),
                transl=torch.zeros(T,3))
            joints = _out.joints.detach().numpy()

            def _jidx(n): return SMPLH_JOINT_NAMES.index(n)
            L_ANK  = _jidx('left_ankle');   R_ANK  = _jidx('right_ankle')
            L_MID1 = _jidx('left_middle1'); R_MID1 = _jidx('right_middle1')

            ankle_min_up = float(np.minimum(
                joints[:, L_ANK, UP_AXIS], joints[:, R_ANK, UP_AXIS]).min())
            floor_u   = ankle_min_up - 0.05
            ankle_mid = 0.5 * (joints[:, L_ANK] + joints[:, R_ANK])
            hand_mid  = 0.5 * (joints[:, L_MID1] + joints[:, R_MID1])
            h_arr     = hand_mid[:, UP_AXIS] - floor_u
            axes      = [i for i in range(3) if i != UP_AXIS]
            diff      = hand_mid - ankle_mid
            reach_arr = np.sqrt(diff[:, axes[0]]**2 + diff[:, axes[1]]**2)

            THR_V = [0.25, 0.81, 1.25, 1.73]
            THR_R = [0.30, 0.60, 0.80]
            V_NAMES = ['V0_floor_to_midshin','V1_midshin_to_knuckle',
                       'V2_knuckle_to_shoulder','V3_shoulder_to_reach','V4_above_reach']
            H_NAMES = {1:'H1_close', 2:'H2_intermediate', 3:'H3_extended', -1:'invalid'}

            def _bv(x):
                for k, thr in enumerate(THR_V):
                    if x < thr: return k
                return 4
            def _br(x):
                if x < THR_R[0]: return 1
                elif x < THR_R[1]: return 2
                elif x < THR_R[2]: return 3
                else: return -1

            for t in range(T):
                vert = float(h_arr[t]); horiz = float(reach_arr[t])
                zv = _bv(vert); zh = _br(horiz)
                tlv_rows.append({
                    'frame':        t,
                    'time_iso':     frames_ts[t]['time_iso'],
                    'time_unix':    frames_ts[t]['time_unix'],
                    'hand_vert_m':  round(vert, 4),
                    'hand_reach_m': round(horiz, 4),
                    'floor_ref_m':  round(floor_u, 4),
                    'V_zone_id':    zv,
                    'V_zone':       V_NAMES[zv],
                    'H_zone_id':    zh,
                    'H_zone':       H_NAMES[zh],
                    'valid':        int(zv <= 3 and zh != -1),
                })
            with open(f'{save_dir}/tlv.csv', 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=tlv_rows[0].keys())
                writer.writeheader()
                writer.writerows(tlv_rows)
            print(f'TLV 已保存  floor_ref={floor_u:.4f}m  '
                  f'有效帧={sum(r["valid"] for r in tlv_rows)}/{T}')
        except Exception as _e:
            import traceback; traceback.print_exc()
            print(f'[警告] TLV 跳过：{_e}')

        # 渲染骨架视频
        print('渲染骨架视频...')
        try:
            import pyrender, trimesh, imageio
            os.environ['PYOPENGL_PLATFORM'] = 'egl'

            _body1 = _smplx.create(
                '/home/yding263/Documents/tlv/babel/data/SMPL_models/smplh/SMPLH_FEMALE.pkl', model_type='smplh',
                gender=_sb.get('gender','neutral'),
                use_pca=False, flat_hand_mean=True, batch_size=1).to('cpu')

            scene = pyrender.Scene(ambient_light=[0.4,0.4,0.4])
            cam   = pyrender.PerspectiveCamera(yfov=np.pi/3)
            cam_pose = np.array([[1,0,0,0],[0,1,0,0.8],[0,0,1,3],[0,0,0,1]], dtype=np.float32)
            scene.add(cam, pose=cam_pose)
            scene.add(pyrender.DirectionalLight(color=[1,1,1], intensity=3.0), pose=cam_pose)
            renderer  = pyrender.OffscreenRenderer(640, 480)
            mesh_node = None

            from articulate.math import rotation_matrix_to_axis_angle as _raa
            _pa = _raa(pose.view(T, 24, 3, 3)).view(T, 72)

            writer = imageio.get_writer(
                f'{save_dir}/skeleton.mp4', fps=30,
                codec='libx264', output_params=['-crf','23'])

            for t in range(0, T, 2):
                _o = _body1(
                    body_pose=_pa[t:t+1, 3:66],
                    global_orient=_pa[t:t+1, :3],
                    betas=SUBJECT_BETAS,
                    transl=torch.zeros(1,3))
                verts = _o.vertices.detach().numpy()[0]
                mesh  = trimesh.Trimesh(verts, _body1.faces)
                pmesh = pyrender.Mesh.from_trimesh(mesh)
                if mesh_node: scene.remove_node(mesh_node)
                mesh_node = scene.add(pmesh)
                color, _ = renderer.render(scene)

                img = color.copy()
                ts_str = frames_ts[t]['time_iso']
                cv2.putText(img, ts_str, (10,30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
                if tlv_rows:
                    zstr = f"V:{tlv_rows[t]['V_zone']}  H:{tlv_rows[t]['H_zone']}"
                    cv2.putText(img, zstr, (10,60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,128), 2)
                writer.append_data(img)

            writer.close()
            renderer.delete()
            print(f'视频已保存：{save_dir}/skeleton.mp4')
        except Exception as _e:
            import traceback; traceback.print_exc()
            print(f'[警告] 视频渲染跳过：{_e}')
            art.ParametricModel(paths.smpl_file).view_motion([pose], [tran])

        print(f'\n文件夹：{save_dir}/')
        print('  pose.pt / tran.pt     骨架姿态')
        print('  timestamps.json       每帧时间戳')
        print('  pelvis_acc.npy        骨盆加速度')
        print('  tlv.csv               逐帧TLV zone')
        print('  skeleton.mp4          骨架渲染视频')
        sys.exit(0)

    # ── 正式 live 模式（需要 Unity）──────────────────────────────────────────
    smpl2imu, device2bone, acc_offsets = do_calibration(imu_set)
    imu_set.start_reading()
    print('\n开始实时估计。按 q 退出 | r 开始录制 | s 停止录制')

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)   # 避免 Address in use
    srv.bind(('127.0.0.1', 8888))
    srv.listen(1)
    print('等待 Unity3D 连接 127.0.0.1:8888 ...')
    conn, addr = srv.accept()
    print(f'Unity 已连接：{addr}')

    running = True
    clock   = Clock()
    is_recording  = False
    record_buffer = None

    t_input = threading.Thread(target=get_input, daemon=True)
    t_input.start()

    while running:
        clock.tick(60)
        ori_raw, acc_raw = imu_set.get_current_buffer()
        if ori_raw.shape[0] == 0:
            continue

        with torch.no_grad():
            data_nn = run_one_frame(ori_raw, acc_raw,
                                    smpl2imu, device2bone, acc_offsets).to(device)
            pose, tran = inertial_poser.forward_online(data_nn)

        pose_np = rotation_matrix_to_axis_angle(pose.view(1, 216)).view(-1).detach().cpu().numpy()
        tran_np = tran.view(-1).detach().cpu().numpy()

        # 录制（存 CPU tensor）
        data_cpu = data_nn.detach().cpu().view(1, -1)
        if not is_recording and start_recording:
            record_buffer = data_cpu
            is_recording  = True
        elif is_recording and start_recording:
            record_buffer = torch.cat([record_buffer, data_cpu], dim=0)
        elif is_recording and not start_recording:
            os.makedirs('data/imu_recordings', exist_ok=True)
            torch.save(record_buffer,
                       'data/imu_recordings/r' +
                       datetime.now().strftime('%T').replace(':', '-') + '.pt')
            is_recording = False

        s = (','.join('%g' % v for v in pose_np) + '#' +
             ','.join('%g' % v for v in tran_np) + '$')
        conn.send(s.encode('utf8'))

        print(f'\r{"(录制中)" if is_recording else "       "}'
              f'  Sensor FPS: {imu_set.clock.get_fps():.1f}'
              f'  Output FPS: {clock.get_fps():.1f}',
              end='', flush=True)

    t_input.join()
    imu_set.close()
    print('\n结束。')
