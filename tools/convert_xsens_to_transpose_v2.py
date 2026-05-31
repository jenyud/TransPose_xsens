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

    raise RuntimeError(f"Cannot find Acc columns. Available columns:\n{list(df.columns)}")


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--session", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--data_start", type=float, default=0.0)
    ap.add_argument("--data_end", type=float, default=None)
    ap.add_argument("--hz", type=int, default=60)

    # IMPORTANT:
    # Xsens Acc_X/Y/Z is treated as sensor-frame gravity-included acceleration.
    # We rotate it into world frame, then subtract world gravity.
    ap.add_argument("--gravity", type=float, default=9.8707)

    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dfs = load_synced_session(data_dir, args.session)

    T0 = len(dfs[BODY_ORDER[0]])
    f0 = int(round(args.data_start * args.hz))
    f1 = T0 if args.data_end is None else int(round(args.data_end * args.hz))

    f0 = max(0, min(f0, T0))
    f1 = max(f0, min(f1, T0))

    print(f"Crop: raw frames {f0}:{f1}, seconds {f0/args.hz:.2f}-{f1/args.hz:.2f}")
    print(f"Output frames: {f1 - f0}, duration: {(f1 - f0)/args.hz:.2f}s")

    T = f1 - f0

    ori = np.zeros((T, 6, 3, 3), dtype=np.float32)
    acc = np.zeros((T, 6, 3), dtype=np.float32)

    g_world = np.array([0.0, 0.0, args.gravity], dtype=np.float32)

    for i, slot in enumerate(BODY_ORDER):
        df = dfs[slot]

        q = df[["Quat_q1", "Quat_q2", "Quat_q3", "Quat_q0"]].values.astype(np.float32)
        R = Rot.from_quat(q).as_matrix().astype(np.float32)

        a_sensor = df[acc_cols(df)].values.astype(np.float32)

        # Rotate sensor-frame acceleration into world frame.
        a_world = np.einsum("tij,tj->ti", R, a_sensor)

        # Remove gravity in world frame.
        a_free_world = a_world - g_world

        ori[:, i] = R[f0:f1]
        acc[:, i] = a_free_world[f0:f1]

        # Diagnostics on a known i-pose candidate around raw 261.6-262.6s if inside range.
        print(
            f"{slot:>10s}: "
            f"free_acc mean={acc[:, i].mean(axis=0).round(4)}, "
            f"free_acc std={acc[:, i].std():.4f}"
        )

    acc_t = torch.from_numpy(acc).float()
    ori_t = torch.from_numpy(ori).float()

    torch.save(acc_t, out_dir / "acc.pt")
    torch.save(ori_t, out_dir / "ori.pt")

    print("\nSaved:")
    print(" ", out_dir / "acc.pt", acc_t.shape)
    print(" ", out_dir / "ori.pt", ori_t.shape)

    print("\nSanity:")
    print("acc shape:", tuple(acc_t.shape))
    print("ori shape:", tuple(ori_t.shape))
    print("ori abs mean:", float(ori_t.abs().mean()))
    print("acc std:", float(acc_t.std()))

    # Check the chosen i-pose raw 261.6-262.6s mapped after data_start.
    ip0 = int(round((261.6 - args.data_start) * args.hz))
    ip1 = int(round((262.6 - args.data_start) * args.hz))

    if 0 <= ip0 < ip1 <= T:
        print("\nCheck raw 261.6-262.6s i-pose free_acc mean:")
        seg = acc_t[ip0:ip1]
        for i, slot in enumerate(BODY_ORDER):
            m = seg[:, i].mean(dim=0).numpy()
            s = seg[:, i].std().item()
            print(f"{slot:>10s}: mean={np.round(m, 4)}, std={s:.4f}")


if __name__ == "__main__":
    main()
