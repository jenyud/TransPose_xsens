import argparse
import cv2
import torch
import articulate as art
from config import paths

# Patch OpenCV H264 -> mp4v
orig_fourcc = cv2.VideoWriter_fourcc
orig_videowriter = cv2.VideoWriter

H264_CODES = set()
for code in ["H264", "h264", "X264", "x264", "avc1", "AVC1"]:
    try:
        H264_CODES.add(orig_fourcc(*code))
    except Exception:
        pass

def patched_fourcc(*args):
    code = "".join(args)
    if code.lower() in ["h264", "x264", "avc1"]:
        print(f"[patch] fourcc {code} -> mp4v")
        return orig_fourcc(*"mp4v")
    return orig_fourcc(*args)

def patched_videowriter(*args, **kwargs):
    args = list(args)
    if len(args) >= 2 and args[1] in H264_CODES:
        print("[patch] VideoWriter H264-like codec -> mp4v")
        args[1] = orig_fourcc(*"mp4v")
    vw = orig_videowriter(*args, **kwargs)
    try:
        print("[patch] VideoWriter opened:", vw.isOpened())
    except Exception:
        pass
    return vw

cv2.VideoWriter_fourcc = patched_fourcc
cv2.VideoWriter = patched_videowriter

parser = argparse.ArgumentParser()
parser.add_argument("--pose_pt", required=True)
parser.add_argument("--tran_pt", required=True)
parser.add_argument("--start_sec", type=float, required=True)
parser.add_argument("--end_sec", type=float, required=True)
parser.add_argument("--hz", type=int, default=60)
args = parser.parse_args()

pose = torch.load(args.pose_pt).float()
tran = torch.load(args.tran_pt).float()

f0 = int(args.start_sec * args.hz)
f1 = int(args.end_sec * args.hz)

f0 = max(0, min(f0, pose.shape[0]))
f1 = max(f0, min(f1, pose.shape[0]))

pose_seg = pose[f0:f1]
tran_seg = tran[f0:f1]

print("full pose:", pose.shape)
print("full tran:", tran.shape)
print(f"render segment: {args.start_sec}-{args.end_sec}s, frames {f0}:{f1}")
print("segment pose:", pose_seg.shape)
print("segment tran:", tran_seg.shape)

body_model = art.ParametricModel(paths.smpl_file)
body_model.view_motion([pose_seg], [tran_seg])
