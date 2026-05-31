#!/usr/bin/env python3
"""
Xsens MTw → TransPose 输入格式转换脚本

完全按照 TransPose live_demo.py 的校准和 normalization 流程：

Step 1: smpl2imu = R_align.T
        用对齐窗口（传感器平放同方向时）的朝向推导全局坐标系
        只用一个参考传感器（head，方向最稳定）

Step 2: device2bone = (smpl2imu @ R_tpose).T
        T-pose 时每个传感器的 sensor-to-bone 映射

Step 3: acc_offsets = smpl2imu @ acc_tpose_mean
        T-pose 时加速度偏移（在 smpl2imu 空间，不经过 R_raw）

Step 4: 每帧处理
        ori_cal = smpl2imu @ ori_raw @ device2bone
        acc_cal = smpl2imu @ acc_raw - acc_offsets

Step 5: normalization（TransPose 格式，完全对应 live_demo.py）
        acc_norm = cat(acc_cal[:5] - acc_cal[5:], acc_cal[5:]) @ ori_cal[5] / acc_scale
        ori_norm = cat(ori_cal[5].T @ ori_cal[:5], ori_cal[5])
        data_nn = cat(acc_norm.view(18), ori_norm.view(54))  → shape (T, 72)

传感器顺序（TransPose 和 DIP 相同）：
  [left_arm(0), right_arm(1), left_leg(2), right_leg(3), head(4), pelvis(5)]

用法：
  python convert_xsens_to_transpose.py \\
      --data_dir data/006 \\
      --session 006 \\
      --out_pt data/xsens_006_006_transpose.pt \\
      --align_start 5 --align_end 10 \\
      --tpose_start 260 --tpose_end 262 \\
      --data_start 70
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.spatial.transform import Rotation as Rot

# ─────────────────────────────────────────────────────────────────────────────
# 传感器映射
# ─────────────────────────────────────────────────────────────────────────────
DEVICE_TO_SLOT: Dict[str, str] = {
    "00B4A214": "left_arm",
    "00B4A21C": "right_arm",
    "00B4A20D": "left_leg",
    "00B4A215": "right_leg",
    "00B4A20C": "head",
    "00B4A211": "pelvis",
}
BODY_ORDER: List[str] = ["left_arm", "right_arm", "left_leg", "right_leg", "head", "pelvis"]
ROOT_IDX  = BODY_ORDER.index("pelvis")  # = 5

# TransPose acc_scale（来自 config.py）
ACC_SCALE = 30.0


# ─────────────────────────────────────────────────────────────────────────────
# I/O
# ─────────────────────────────────────────────────────────────────────────────

def parse_xsens_txt(filepath: Path) -> pd.DataFrame:
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
    header_idx = 0
    for i, line in enumerate(lines):
        if not line.startswith("//"):
            header_idx = i
            break
    try:
        return pd.read_csv(filepath, skiprows=header_idx, sep="\t",
                           engine="python", on_bad_lines="skip")
    except TypeError:
        return pd.read_csv(filepath, skiprows=header_idx, sep="\t",
                           engine="python", error_bad_lines=False)


def get_device_id(filepath: Path) -> str:
    return filepath.stem.split("_")[-1]


def load_session(data_dir: Path, session: str) -> Dict[str, pd.DataFrame]:
    files = sorted(data_dir.glob(f"MT_*_{session}-000_*.txt"))
    if not files:
        raise FileNotFoundError(f"找不到 session={session}")

    dfs: Dict[str, pd.DataFrame] = {}
    print(f"\n处理 session {session}：")
    for f in files:
        dev_id = get_device_id(f)
        slot   = DEVICE_TO_SLOT.get(dev_id)
        if slot is None:
            continue
        df = parse_xsens_txt(f)
        dfs[slot] = df
        print(f"  {slot:>12s} ({dev_id}): {len(df)} 行")

    missing = [s for s in BODY_ORDER if s not in dfs]
    if missing:
        raise RuntimeError(f"缺少传感器: {missing}")

    pc_sets   = {slot: set(df["PacketCounter"].values) for slot, df in dfs.items()}
    common_pc = np.array(sorted(set.intersection(*pc_sets.values())), dtype=np.int64)
    for slot in dfs:
        df = dfs[slot]
        df = df[df["PacketCounter"].isin(common_pc)].copy()
        df = df.sort_values("PacketCounter").reset_index(drop=True)
        dfs[slot] = df

    print(f"公共帧数: {len(common_pc)}")
    return dfs, common_pc


def extract_raw(dfs: Dict[str, pd.DataFrame]) -> Tuple[np.ndarray, np.ndarray]:
    T = len(next(iter(dfs.values())))
    ori_world  = np.zeros((T, 6, 3, 3), dtype=np.float32)
    acc_sensor = np.zeros((T, 6, 3),    dtype=np.float32)
    for i, slot in enumerate(BODY_ORDER):
        df = dfs[slot]
        q  = df[["Quat_q1","Quat_q2","Quat_q3","Quat_q0"]].values.astype(np.float32)
        ori_world[:, i]  = Rot.from_quat(q).as_matrix().astype(np.float32)
        acc_sensor[:, i] = df[["Acc_X","Acc_Y","Acc_Z"]].values.astype(np.float32)
    return ori_world, acc_sensor


# ─────────────────────────────────────────────────────────────────────────────
# 校准（完全对应 live_demo.py）
# ─────────────────────────────────────────────────────────────────────────────

def calibrate(
    ori_world:  np.ndarray,  # (T, 6, 3, 3)
    acc_sensor: np.ndarray,  # (T, 6, 3)
    align_cf0:  int, align_cf1:  int,   # 对齐窗口（传感器平放时）
    tpose_cf0:  int, tpose_cf1:  int,   # T-pose 窗口
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    返回：
      smpl2imu    (3, 3)
      device2bone (6, 3, 3)
      acc_offsets (6, 3)
    """
    # Step 1: smpl2imu
    # 用 head 传感器在对齐窗口的平均朝向（live_demo 用第一个 IMU，我们用 head 更稳定）
    head_idx  = BODY_ORDER.index("head")
    R_align   = Rot.from_matrix(ori_world[align_cf0:align_cf1, head_idx]).mean().as_matrix()
    smpl2imu  = R_align.T.astype(np.float32)  # (3, 3)

    print(f"\n[校准] smpl2imu（来自 head 对齐窗口）：")
    print(np.round(smpl2imu, 4))

    # Step 2 & 3: device2bone 和 acc_offsets（T-pose）
    device2bone = np.zeros((6, 3, 3), dtype=np.float32)
    acc_offsets = np.zeros((6, 3),    dtype=np.float32)

    print(f"\n[校准] T-pose 校准诊断：")
    for i, slot in enumerate(BODY_ORDER):
        R_tp  = Rot.from_matrix(ori_world[tpose_cf0:tpose_cf1, i]).mean().as_matrix().astype(np.float32)
        # device2bone = (smpl2imu @ R_tpose).T @ I = (smpl2imu @ R_tpose).T
        device2bone[i] = (smpl2imu @ R_tp).T

        # acc_offsets = smpl2imu @ acc_raw_mean（不经过 R_raw，直接用传感器原始加速度）
        acc_mean = acc_sensor[tpose_cf0:tpose_cf1, i].mean(axis=0).astype(np.float32)
        acc_offsets[i] = smpl2imu @ acc_mean

        # 诊断：T-pose 校准后的角度误差（应接近 0°）
        R_cal = smpl2imu @ R_tp @ device2bone[i]
        angle = float(np.degrees(Rot.from_matrix(R_cal).magnitude()))
        print(f"  {slot:>12s}: T-pose → Identity 误差 = {angle:.4f}°  "
              f"acc_offset = {np.round(acc_offsets[i], 3)}")

    return smpl2imu, device2bone, acc_offsets


# ─────────────────────────────────────────────────────────────────────────────
# 变换 + TransPose normalization
# ─────────────────────────────────────────────────────────────────────────────

def transform_and_normalize(
    ori_world:   np.ndarray,  # (T, 6, 3, 3)
    acc_sensor:  np.ndarray,  # (T, 6, 3)
    smpl2imu:    np.ndarray,  # (3, 3)
    device2bone: np.ndarray,  # (6, 3, 3)
    acc_offsets: np.ndarray,  # (6, 3)
    acc_scale:   float = ACC_SCALE,
) -> torch.Tensor:
    """
    完全对应 live_demo.py 的 normalization：

    ori_cal = smpl2imu @ ori_raw @ device2bone
    acc_cal = smpl2imu @ acc_raw - acc_offsets

    acc_norm = cat(acc_cal[:5] - acc_cal[5:], acc_cal[5:]) @ ori_cal[5] / acc_scale
    ori_norm = cat(ori_cal[5].T @ ori_cal[:5], ori_cal[5])
    data_nn  = cat(acc_norm.view(-1, 18), ori_norm.view(-1, 54))  → (T, 72)
    """
    T = ori_world.shape[0]

    # ori_cal: (T, 6, 3, 3)
    ori_cal = np.zeros((T, 6, 3, 3), dtype=np.float32)
    for i in range(6):
        ori_cal[:, i] = smpl2imu @ ori_world[:, i] @ device2bone[i]

    # acc_cal: (T, 6, 3)  smpl2imu @ acc_raw - acc_offsets
    acc_cal = np.zeros((T, 6, 3), dtype=np.float32)
    for i in range(6):
        acc_cal[:, i] = (smpl2imu @ acc_sensor[:, i].T).T - acc_offsets[i]

    # 转为 torch
    ori_t = torch.from_numpy(ori_cal).float()  # (T, 6, 3, 3)
    acc_t = torch.from_numpy(acc_cal).float()  # (T, 6, 3)

    # Normalization（完全对应 live_demo.py 第 70-72 行）
    # acc: relative to pelvis(5), rotated to pelvis frame, scaled
    acc_norm = torch.cat(
        (acc_t[:, :5] - acc_t[:, 5:],   # (T, 5, 3) relative to pelvis
         acc_t[:, 5:]),                  # (T, 1, 3) pelvis
        dim=1                            # (T, 6, 3)
    ).bmm(ori_t[:, ROOT_IDX]) / acc_scale   # rotate by pelvis ori → (T, 6, 3)

    # ori: relative to pelvis(5)
    ori_norm = torch.cat(
        (ori_t[:, ROOT_IDX:ROOT_IDX+1].transpose(2, 3).matmul(ori_t[:, :5]),  # (T, 5, 3, 3)
         ori_t[:, ROOT_IDX:ROOT_IDX+1]),                                       # (T, 1, 3, 3)
        dim=1  # (T, 6, 3, 3)
    )

    # 拼接成 data_nn: (T, 72) = 18 + 54
    data_nn = torch.cat(
        (acc_norm.view(T, -1),   # (T, 18)
         ori_norm.view(T, -1)),  # (T, 54)
        dim=1
    )  # (T, 72)

    return data_nn, ori_t, acc_t


# ─────────────────────────────────────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────────────────────────────────────

def run(
    data_dir:      Path,
    session:       str,
    out_pt:        Path,
    align_start:   float,
    align_end:     float,
    tpose_start:   float,
    tpose_end:     float,
    data_start:    float = 0.0,
    hz:            int   = 60,
) -> None:

    dfs, common_pc = load_session(data_dir, session)
    T_full = len(common_pc)

    align_cf0 = int(round(align_start * hz))
    align_cf1 = int(round(align_end   * hz))
    tpose_cf0 = int(round(tpose_start * hz))
    tpose_cf1 = int(round(tpose_end   * hz))
    ds        = int(round(data_start  * hz))

    print(f"\n对齐窗口:    {align_start}-{align_end}s  (帧 {align_cf0}-{align_cf1})")
    print(f"T-pose 窗口: {tpose_start}-{tpose_end}s  (帧 {tpose_cf0}-{tpose_cf1})")
    print(f"数据起始:    {data_start}s  (跳过前 {ds} 帧)")

    ori_world, acc_sensor = extract_raw(dfs)

    # 校准
    smpl2imu, device2bone, acc_offsets = calibrate(
        ori_world, acc_sensor,
        align_cf0, align_cf1,
        tpose_cf0, tpose_cf1,
    )

    # 裁剪到 data_start 之后
    ori_crop = ori_world[ds:]
    acc_crop = acc_sensor[ds:]
    T = ori_crop.shape[0]
    print(f"\n有效帧数: {T}  ({T/hz:.1f}s)")

    # 变换 + normalization
    data_nn, ori_cal, acc_cal = transform_and_normalize(
        ori_crop, acc_crop, smpl2imu, device2bone, acc_offsets)

    # 诊断
    print(f"\n[诊断] 输出统计：")
    print(f"  data_nn shape: {data_nn.shape}  (应为 (T, 72))")
    acc_part = data_nn[:, :18]
    ori_part = data_nn[:, 18:]
    print(f"  acc part std:      {acc_part.std():.4f}  (TransPose 参考 ~1.0)")
    print(f"  ori part abs mean: {ori_part.abs().mean():.4f}  (参考 ~0.3-0.5)")

    # T-pose 段诊断（如果在裁剪后的数据里）
    tpose_in_crop = tpose_cf0 - ds
    if 0 <= tpose_in_crop < T:
        tpose_end_crop = min(tpose_cf1 - ds, T)
        seg_ori = ori_cal[tpose_in_crop:tpose_end_crop]  # (seg, 6, 3, 3)
        print(f"\n[诊断] T-pose 段校准误差（应接近 0°）：")
        for i, slot in enumerate(BODY_ORDER):
            R_mean = Rot.from_matrix(seg_ori[:, i].numpy()).mean().as_matrix()
            angle  = float(np.degrees(Rot.from_matrix(R_mean).magnitude()))
            print(f"  {slot:>12s}: {angle:.4f}°")

    # 保存
    out_pt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data_nn, out_pt)
    print(f"\n已保存: {out_pt}  shape={data_nn.shape}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Xsens MTw → TransPose 输入格式（完全对应 live_demo.py）")
    ap.add_argument("--data_dir",    required=True)
    ap.add_argument("--session",     required=True)
    ap.add_argument("--out_pt",      required=True, help="输出 .pt 文件路径")
    ap.add_argument("--align_start", type=float, required=True,
                    help="传感器对齐窗口开始（秒），传感器平放同方向时")
    ap.add_argument("--align_end",   type=float, required=True)
    ap.add_argument("--tpose_start", type=float, required=True,
                    help="T-pose 校准窗口开始（秒）")
    ap.add_argument("--tpose_end",   type=float, required=True)
    ap.add_argument("--data_start",  type=float, default=0.0,
                    help="跳过开头多少秒（传感器未穿戴时间，默认 0）")
    ap.add_argument("--hz",          type=int,   default=60)
    args = ap.parse_args()

    run(
        data_dir    = Path(args.data_dir),
        session     = args.session,
        out_pt      = Path(args.out_pt),
        align_start = args.align_start,
        align_end   = args.align_end,
        tpose_start = args.tpose_start,
        tpose_end   = args.tpose_end,
        data_start  = args.data_start,
        hz          = args.hz,
    )