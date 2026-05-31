import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.spatial.transform import Rotation as Rot


DEVICE_TO_SLOT = {
    "00B4A214": "left_arm",
    "00B4A21C": "right_arm",
    "00B4A20D": "left_leg",
    "00B4A215": "right_leg",
    "00B4A20C": "head",
    "00B4A211": "pelvis",
}

# TransPose order:
# left forearm, right forearm, left lower leg, right lower leg, head, pelvis
BODY_ORDER = ["left_arm", "right_arm", "left_leg", "right_leg", "head", "pelvis"]


def read_xsens_txt(filepath: Path) -> pd.DataFrame:
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    header_idx = None
    for i, line in enumerate(lines):
        if not line.startswith("//"):
            header_idx = i
            break

    if header_idx is None:
        raise RuntimeError(f"Cannot find header in {filepath}")

    return pd.read_csv(
        filepath,
        skiprows=header_idx,
        sep="\t",
        engine="python",
        on_bad_lines="skip",
    )


def get_device_id(filepath: Path) -> str:
    return filepath.stem.split("_")[-1]


def acc_cols(df: pd.DataFrame):
    for cols in [
        ["Acc_X", "Acc_Y", "Acc_Z"],
        ["Acceleration_X", "Acceleration_Y", "Acceleration_Z"],
    ]:
        if all(c in df.columns for c in cols):
            return cols
    raise RuntimeError(f"Cannot find Acc columns. Columns:\n{list(df.columns)}")


def load_synced_session(data_dir: Path, session: str):
    files = sorted(data_dir.glob(f"MT_*_{session}-000_*.txt"))
    if not files:
        raise FileNotFoundError(f"No files found for session {session} in {data_dir}")

    dfs = {}
    for f in files:
        dev = get_device_id(f)
        slot = DEVICE_TO_SLOT.get(dev)
        if slot is None:
            continue
        dfs[slot] = read_xsens_txt(f)
        print(f"{slot:>10s} ({dev}): {len(dfs[slot])} rows")

    missing = [s for s in BODY_ORDER if s not in dfs]
    if missing:
        raise RuntimeError(f"Missing sensors: {missing}")

    common_pc = set.intersection(*[set(df["PacketCounter"].values) for df in dfs.values()])
    common_pc = np.array(sorted(common_pc), dtype=np.int64)
    print(f"Common PacketCounter frames: {len(common_pc)}")

    for slot in BODY_ORDER:
        df = dfs[slot]
        df = df[df["PacketCounter"].isin(common_pc)].copy()
        df = df.sort_values("PacketCounter").reset_index(drop=True)
        dfs[slot] = df

    return dfs


def mean_rotation(R_seq):
    return Rot.from_matrix(R_seq).mean().as_matrix().astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--session", required=True)
    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--ref_start", type=float, required=True,
                    help="Raw time when all sensors are aligned/facing same way, before wearing.")
    ap.add_argument("--ref_end", type=float, required=True)

    ap.add_argument("--tpose_start", type=float, required=True,
                    help="Raw time of wearing T-pose calibration.")
    ap.add_argument("--tpose_end", type=float, required=True)

    ap.add_argument("--data_start", type=float, default=0.0)
    ap.add_argument("--data_end", type=float, default=None)
    ap.add_argument("--hz", type=int, default=60)
    ap.add_argument("--gravity", type=float, default=9.8707)
    ap.add_argument("--acc_sign", type=float, default=1.0, help="Try +1 or -1 for Xsens Acc sign.")

    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dfs = load_synced_session(data_dir, args.session)

    T0 = len(dfs[BODY_ORDER[0]])

    ref0 = int(round(args.ref_start * args.hz))
    ref1 = int(round(args.ref_end * args.hz))
    tp0 = int(round(args.tpose_start * args.hz))
    tp1 = int(round(args.tpose_end * args.hz))

    f0 = int(round(args.data_start * args.hz))
    f1 = T0 if args.data_end is None else int(round(args.data_end * args.hz))

    ref0, ref1 = max(0, ref0), min(T0, ref1)
    tp0, tp1 = max(0, tp0), min(T0, tp1)
    f0, f1 = max(0, f0), min(T0, f1)

    print(f"Reference alignment window: {args.ref_start}-{args.ref_end}s frames {ref0}:{ref1}")
    print(f"T-pose calibration window: {args.tpose_start}-{args.tpose_end}s frames {tp0}:{tp1}")
    print(f"Output crop: {args.data_start}-{f1/args.hz:.2f}s frames {f0}:{f1}")

    if ref1 <= ref0 or tp1 <= tp0 or f1 <= f0:
        raise RuntimeError("Invalid time windows.")

    # Load raw orientations and sensor-local accelerations.
    R_raw_all = {}
    A_sensor_all = {}

    for slot in BODY_ORDER:
        df = dfs[slot]

        q = df[["Quat_q1", "Quat_q2", "Quat_q3", "Quat_q0"]].values.astype(np.float32)
        R_raw = Rot.from_quat(q).as_matrix().astype(np.float32)

        a_sensor = df[acc_cols(df)].values.astype(np.float32)

        R_raw_all[slot] = R_raw
        A_sensor_all[slot] = a_sensor

    # ------------------------------------------------------------
    # TransPose live-style calibration
    # Step 1: smpl2imu from first IMU aligned with body reference.
    # Official uses IMU 1. Here IMU 1 = left_arm.
    # smpl2imu is body_from_world / SMPL-frame-from-IMU-world transform.
    # ------------------------------------------------------------
    R_ref = mean_rotation(R_raw_all["left_arm"][ref0:ref1])
    smpl2imu = R_ref.T.astype(np.float32)

    print("\nsmpl2imu/body_from_world from left_arm ref window:")
    print(np.round(smpl2imu, 4))

    # ------------------------------------------------------------
    # Step 2: device2bone from T-pose.
    # In T-pose, target bone orientation is identity in this calibrated frame.
    # device2bone_i = (smpl2imu @ R_i_tpose)^T
    # ------------------------------------------------------------
    device2bone = np.zeros((6, 3, 3), dtype=np.float32)
    acc_offsets = np.zeros((6, 3), dtype=np.float32)

    g_world = np.array([0.0, 0.0, args.gravity], dtype=np.float32)

    print("\nT-pose calibration diagnostics:")
    for i, slot in enumerate(BODY_ORDER):
        R_tp = mean_rotation(R_raw_all[slot][tp0:tp1])
        device2bone[i] = (smpl2imu @ R_tp).T

        # Official TransPose live_demo style:
        # acc_cal = smpl2imu @ acc_raw - acc_offsets
        # No R_raw @ acc_sensor rotation here.
        a_sensor_seq = args.acc_sign * A_sensor_all[slot][tp0:tp1]
        a_body_seq = (smpl2imu @ a_sensor_seq.T).T
        acc_offsets[i] = a_body_seq.mean(axis=0)

        # Check that orientation at T-pose becomes identity after calibration.
        R_cal_tp = smpl2imu @ R_tp @ device2bone[i]
        angle = np.degrees(Rot.from_matrix(R_cal_tp).magnitude())

        print(
            f"{slot:>10s}: "
            f"T-pose calibrated angle-to-I={angle:.4f} deg, "
            f"acc_offset={np.round(acc_offsets[i], 4)}"
        )

    # ------------------------------------------------------------
    # Step 3: apply calibration to the whole output segment.
    # ori_cal = smpl2imu @ R_raw @ device2bone
    # acc_cal = smpl2imu @ acc_world - acc_offsets
    # ------------------------------------------------------------
    T = f1 - f0
    ori_cal = np.zeros((T, 6, 3, 3), dtype=np.float32)
    acc_cal = np.zeros((T, 6, 3), dtype=np.float32)

    for i, slot in enumerate(BODY_ORDER):
        R_seq = R_raw_all[slot][f0:f1]
        a_sensor_seq = args.acc_sign * A_sensor_all[slot][f0:f1]

        # Official TransPose live_demo style: no R_seq @ acc.
        a_body_seq = (smpl2imu @ a_sensor_seq.T).T

        ori_cal[:, i] = np.einsum(
            "ij,tjk,kl->til",
            smpl2imu,
            R_seq,
            device2bone[i],
        )
        acc_cal[:, i] = a_body_seq - acc_offsets[i]

    acc_t = torch.from_numpy(acc_cal).float()
    ori_t = torch.from_numpy(ori_cal).float()

    torch.save(acc_t, out_dir / "acc.pt")
    torch.save(ori_t, out_dir / "ori.pt")

    print("\nSaved:")
    print(" ", out_dir / "acc.pt", acc_t.shape)
    print(" ", out_dir / "ori.pt", ori_t.shape)

    print("\nSanity:")
    print("acc std:", float(acc_t.std()))
    print("ori abs mean:", float(ori_t.abs().mean()))

    # Check T-pose segment after output crop if included.
    tpc0 = int(round((args.tpose_start - args.data_start) * args.hz))
    tpc1 = int(round((args.tpose_end - args.data_start) * args.hz))

    if 0 <= tpc0 < tpc1 <= T:
        print("\nCheck output T-pose segment after calibration:")
        seg_ori = ori_t[tpc0:tpc1]
        seg_acc = acc_t[tpc0:tpc1]
        for i, slot in enumerate(BODY_ORDER):
            R_mean = mean_rotation(seg_ori[:, i].numpy())
            angle = np.degrees(Rot.from_matrix(R_mean).magnitude())
            amean = seg_acc[:, i].mean(dim=0).numpy()
            astd = seg_acc[:, i].std().item()
            print(
                f"{slot:>10s}: "
                f"ori angle-to-I={angle:.4f} deg, "
                f"acc mean={np.round(amean, 4)}, acc std={astd:.4f}"
            )


if __name__ == "__main__":
    main()
