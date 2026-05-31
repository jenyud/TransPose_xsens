import itertools
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.spatial.transform import Rotation as Rot

DATA_DIR = Path("/home/yding263/Documents/tlv/dip18-master/data/006")
SESSION = "006"
OUT_ROOT = Path("data/axis_search_006_ipose_93_101")
OUT_ROOT.mkdir(parents=True, exist_ok=True)

HZ = 60
REF = (42, 46)
TPOSE = (102.85, 103.65)

# output only raw 93-101s, because this contains the useful i-pose around 96.95-97.75s
DATA_START = 93
DATA_END = 101

DEVICE_TO_SLOT = {
    "00B4A214": "left_arm",
    "00B4A21C": "right_arm",
    "00B4A20D": "left_leg",
    "00B4A215": "right_leg",
    "00B4A20C": "head",
    "00B4A211": "pelvis",
}
BODY_ORDER = ["left_arm", "right_arm", "left_leg", "right_leg", "head", "pelvis"]

def read_xsens(f):
    with open(f, "r", encoding="utf-8", errors="ignore") as fp:
        lines = fp.readlines()
    h = next(i for i, l in enumerate(lines) if not l.startswith("//"))
    return pd.read_csv(f, skiprows=h, sep="\t", engine="python", on_bad_lines="skip")

def get_device_id(filepath):
    return filepath.stem.split("_")[-1]

def acc_cols(df):
    for cols in [["Acc_X", "Acc_Y", "Acc_Z"], ["Acceleration_X", "Acceleration_Y", "Acceleration_Z"]]:
        if all(c in df.columns for c in cols):
            return cols
    raise RuntimeError("No acceleration columns found.")

def mean_R(Rseq):
    return Rot.from_matrix(Rseq).mean().as_matrix().astype(np.float32)

def frames(w):
    return int(w[0] * HZ), int(w[1] * HZ)

def signed_permutation_mats():
    mats = []
    names = []
    axes = np.eye(3)
    axis_names = ["X", "Y", "Z"]

    for perm in itertools.permutations(range(3)):
        P = axes[:, perm]
        pname = "".join(axis_names[i] for i in perm)
        for signs in itertools.product([-1, 1], repeat=3):
            S = P @ np.diag(signs)
            if np.linalg.det(S) > 0.5:
                mats.append(S.astype(np.float32))
                sname = "".join(["+" if x > 0 else "-" for x in signs])
                names.append(f"{pname}_{sname}")
    return mats, names

# load data
files = sorted(DATA_DIR.glob(f"MT_*_{SESSION}-000_*.txt"))
dfs = {}
for f in files:
    dev = get_device_id(f)
    slot = DEVICE_TO_SLOT.get(dev)
    if slot:
        dfs[slot] = read_xsens(f)

missing = [s for s in BODY_ORDER if s not in dfs]
if missing:
    raise RuntimeError(f"Missing sensors: {missing}")

common = set.intersection(*[set(df["PacketCounter"].values) for df in dfs.values()])
common = np.array(sorted(common), dtype=np.int64)

for slot in BODY_ORDER:
    df = dfs[slot]
    df = df[df["PacketCounter"].isin(common)].copy()
    dfs[slot] = df.sort_values("PacketCounter").reset_index(drop=True)

R_raw_all = {}
A_sensor_all = {}
for slot in BODY_ORDER:
    df = dfs[slot]
    q = df[["Quat_q1", "Quat_q2", "Quat_q3", "Quat_q0"]].values.astype(np.float32)
    R_raw_all[slot] = Rot.from_quat(q).as_matrix().astype(np.float32)
    A_sensor_all[slot] = df[acc_cols(df)].values.astype(np.float32)

ref0, ref1 = frames(REF)
tp0, tp1 = frames(TPOSE)
f0, f1 = frames((DATA_START, DATA_END))

R_ref = mean_R(R_raw_all["left_arm"][ref0:ref1])
base_smpl2imu = R_ref.T.astype(np.float32)

axis_mats, axis_names = signed_permutation_mats()

print("Generating", len(axis_mats), "axis mapping variants")
print("Output root:", OUT_ROOT)
print("Segment raw:", DATA_START, DATA_END, "sec")

for idx, (A, name) in enumerate(zip(axis_mats, axis_names)):
    out_dir = OUT_ROOT / f"map{idx:02d}_{name}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # New body-frame mapping
    smpl2imu = A @ base_smpl2imu

    device2bone = np.zeros((6, 3, 3), dtype=np.float32)
    acc_offsets = np.zeros((6, 3), dtype=np.float32)

    for i, slot in enumerate(BODY_ORDER):
        R_tp = mean_R(R_raw_all[slot][tp0:tp1])
        device2bone[i] = (smpl2imu @ R_tp).T

        # keep same acceleration treatment as the current working converter:
        # sensor Acc -> world by R -> body by smpl2imu -> subtract T-pose offset
        R_seq_tp = R_raw_all[slot][tp0:tp1]
        a_sensor_tp = A_sensor_all[slot][tp0:tp1]
        a_world_tp = np.einsum("tij,tj->ti", R_seq_tp, a_sensor_tp)
        a_body_tp = (smpl2imu @ a_world_tp.T).T
        acc_offsets[i] = a_body_tp.mean(axis=0)

    T = f1 - f0
    ori_cal = np.zeros((T, 6, 3, 3), dtype=np.float32)
    acc_cal = np.zeros((T, 6, 3), dtype=np.float32)

    for i, slot in enumerate(BODY_ORDER):
        R_seq = R_raw_all[slot][f0:f1]
        a_sensor = A_sensor_all[slot][f0:f1]
        a_world = np.einsum("tij,tj->ti", R_seq, a_sensor)
        a_body = (smpl2imu @ a_world.T).T

        ori_cal[:, i] = np.einsum("ij,tjk,kl->til", smpl2imu, R_seq, device2bone[i])
        acc_cal[:, i] = a_body - acc_offsets[i]

    torch.save(torch.from_numpy(acc_cal).float(), out_dir / "acc.pt")
    torch.save(torch.from_numpy(ori_cal).float(), out_dir / "ori.pt")

    with open(out_dir / "axis_map.txt", "w") as f:
        f.write(f"index={idx}\n")
        f.write(f"name={name}\n")
        f.write("A=\n")
        f.write(str(A))
        f.write("\n")

    print(f"map{idx:02d}_{name}: saved {T} frames")

print("Done.")
