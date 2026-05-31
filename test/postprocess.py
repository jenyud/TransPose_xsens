import sys, os, json, csv, math
sys.path.insert(0, '/home/yding263/Documents/tlv/TransPose')
import numpy as np
import torch
import smplx
from smplx.joint_names import SMPLH_JOINT_NAMES
from articulate.math import rotation_matrix_to_axis_angle
from datetime import datetime

# ── 输入目录（命令行参数或自动找最新）────────────────────────────────────────
if len(sys.argv) > 1:
    save_dir = sys.argv[1].rstrip('/')
else:
    import glob
    dirs = sorted(glob.glob('data/imu_recordings/*/'))
    save_dir = dirs[-1].rstrip('/')
print(f'处理目录: {save_dir}')

SMPLH_PKL    = '/home/yding263/Documents/tlv/babel/data/SMPL_models/smplh/SMPLH_FEMALE.pkl'
BETAS_JSON   = '/home/yding263/Documents/tlv/dip18-master/data/subject_betas.json'
THR_V = [0.25, 0.81, 1.25, 1.73]
THR_R = [0.30, 0.60, 0.80]
V_NAMES = ['V0_floor_to_midshin','V1_midshin_to_knuckle',
           'V2_knuckle_to_shoulder','V3_shoulder_to_reach','V4_above_reach']
H_NAMES = {1:'H1_close', 2:'H2_intermediate', 3:'H3_extended', -1:'invalid'}

# ── 加载数据 ─────────────────────────────────────────────────────────────────
pose = torch.load(f'{save_dir}/pose.pt')
tran = torch.load(f'{save_dir}/tran.pt')
T    = pose.shape[0]
print(f'pose shape: {pose.shape}')

with open(BETAS_JSON) as f:
    sb = json.load(f)
UP_AXIS = int(sb['up_axis'])

# 加载时间戳（如果有）
ts_file = f'{save_dir}/timestamps.json'
if os.path.exists(ts_file):
    with open(ts_file) as f:
        ts_data = json.load(f)
    frames_ts = ts_data['frames']
    print(f'时间戳: {len(frames_ts)} 帧')
else:
    # 没有时间戳，用 session 目录名推算
    tag = os.path.basename(save_dir)
    try:
        t0 = datetime.strptime(tag, '%Y%m%d_%H-%M-%S').timestamp()
    except:
        t0 = 0.0
    frames_ts = [{'frame': i,
                  'time_iso': datetime.fromtimestamp(t0 + i/60).strftime('%Y-%m-%d %H:%M:%S') + f':{i%60:02d}',
                  'time_unix': round(t0 + i/60, 6),
                  'time_rel_sec': round(i/60, 4)} for i in range(T)]
    print('无时间戳文件，按 60fps 推算')

# ── SMPL-H 前向 ──────────────────────────────────────────────────────────────
print(f'运行 SMPL-H（{T} 帧，分批处理）...')
body = smplx.create(SMPLH_PKL, model_type='smplh', gender=sb['gender'],
                    use_pca=False, flat_hand_mean=True, batch_size=1)
betas = torch.tensor(sb['betas'], dtype=torch.float32).unsqueeze(0)

pose_aa = rotation_matrix_to_axis_angle(pose.view(T, 24, 3, 3)).view(T, 72)

joints_list = []
BATCH = 500
for s in range(0, T, BATCH):
    e = min(T, s + BATCH)
    bs = e - s
    _body = smplx.create(SMPLH_PKL, model_type='smplh', gender=sb['gender'],
                         use_pca=False, flat_hand_mean=True, batch_size=bs)
    out = _body(body_pose=pose_aa[s:e, 3:66],
                global_orient=pose_aa[s:e, :3],
                betas=betas.repeat(bs, 1),
                transl=torch.zeros(bs, 3))
    joints_list.append(out.joints.detach().numpy())
    print(f'  {e}/{T}')
joints = np.concatenate(joints_list, axis=0)
print(f'joints shape: {joints.shape}')

# ── TLV 计算 ─────────────────────────────────────────────────────────────────
def _jidx(n): return SMPLH_JOINT_NAMES.index(n)
L_ANK  = _jidx('left_ankle');   R_ANK  = _jidx('right_ankle')
L_MID1 = _jidx('left_middle1'); R_MID1 = _jidx('right_middle1')

floor_u   = float(np.minimum(joints[:,L_ANK,UP_AXIS], joints[:,R_ANK,UP_AXIS]).min()) - 0.05
ankle_mid = 0.5 * (joints[:,L_ANK] + joints[:,R_ANK])
hand_mid  = 0.5 * (joints[:,L_MID1] + joints[:,R_MID1])
h_arr     = hand_mid[:,UP_AXIS] - floor_u
axes      = [i for i in range(3) if i != UP_AXIS]
diff      = hand_mid - ankle_mid
reach_arr = np.sqrt(diff[:,axes[0]]**2 + diff[:,axes[1]]**2)

def _bv(x):
    for k, thr in enumerate(THR_V):
        if x < thr: return k
    return 4
def _br(x):
    if x < THR_R[0]: return 1
    elif x < THR_R[1]: return 2
    elif x < THR_R[2]: return 3
    else: return -1

tlv_rows = []
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
print(f'TLV 已保存  floor_ref={floor_u:.4f}m  有效帧={sum(r["valid"] for r in tlv_rows)}/{T}')

# ── 渲染视频 ─────────────────────────────────────────────────────────────────
print('渲染骨架视频...')
try:
    import pyrender, trimesh, imageio, cv2
    os.environ['PYOPENGL_PLATFORM'] = 'egl'

    _body1 = smplx.create(SMPLH_PKL, model_type='smplh', gender=sb['gender'],
                          use_pca=False, flat_hand_mean=True, batch_size=1)
    scene = pyrender.Scene(ambient_light=[0.4,0.4,0.4])
    cam   = pyrender.PerspectiveCamera(yfov=np.pi/3)
    cam_pose = np.array([[1,0,0,0],[0,1,0,0.8],[0,0,1,3],[0,0,0,1]], dtype=np.float32)
    scene.add(cam, pose=cam_pose)
    scene.add(pyrender.DirectionalLight(color=[1,1,1], intensity=3.0), pose=cam_pose)
    renderer  = pyrender.OffscreenRenderer(640, 480)
    mesh_node = None

    writer = imageio.get_writer(f'{save_dir}/skeleton.mp4', fps=30,
                                codec='libx264', output_params=['-crf','23'])
    for t in range(0, T, 2):
        _o = _body1(body_pose=pose_aa[t:t+1, 3:66],
                    global_orient=pose_aa[t:t+1, :3],
                    betas=betas, transl=torch.zeros(1,3))
        verts = _o.vertices.detach().numpy()[0]
        mesh  = trimesh.Trimesh(verts, _body1.faces)
        pmesh = pyrender.Mesh.from_trimesh(mesh)
        if mesh_node: scene.remove_node(mesh_node)
        mesh_node = scene.add(pmesh)
        color, _ = renderer.render(scene)
        img = color.copy()
        cv2.putText(img, frames_ts[t]['time_iso'], (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
        zstr = f"V:{tlv_rows[t]['V_zone']}  H:{tlv_rows[t]['H_zone']}"
        cv2.putText(img, zstr, (10,60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,128), 2)
        writer.append_data(img)
        if t % 200 == 0:
            print(f'  渲染 {t}/{T}')
    writer.close()
    renderer.delete()
    print(f'视频已保存: {save_dir}/skeleton.mp4')
except Exception as e:
    import traceback; traceback.print_exc()
    print(f'视频渲染跳过: {e}')

print(f'\n完成！文件在 {save_dir}/')
