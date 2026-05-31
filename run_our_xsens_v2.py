import argparse
from pathlib import Path

import torch
from net import TransPoseNet
from utils import normalize_and_concat
from config import paths
import articulate as art


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/our_xsens_006_006_v2")
    parser.add_argument("--start_sec", type=float, default=None)
    parser.add_argument("--end_sec", type=float, default=None)
    parser.add_argument("--hz", type=int, default=60)
    parser.add_argument("--out_prefix", default="transpose_xsens_006_006_v2")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    acc = torch.load(data_dir / "acc.pt").float()
    ori = torch.load(data_dir / "ori.pt").float()

    print("original acc:", acc.shape)
    print("original ori:", ori.shape)
    print("original acc std:", acc.std().item())
    print("original ori abs mean:", ori.abs().mean().item())

    f0 = 0 if args.start_sec is None else int(args.start_sec * args.hz)
    f1 = acc.shape[0] if args.end_sec is None else int(args.end_sec * args.hz)

    f0 = max(0, min(f0, acc.shape[0]))
    f1 = max(f0, min(f1, acc.shape[0]))

    acc = acc[f0:f1]
    ori = ori[f0:f1]

    print(f"cropped frames: {f0}:{f1}")
    print("cropped duration:", acc.shape[0] / args.hz, "sec")
    print("cropped acc:", acc.shape)
    print("cropped ori:", ori.shape)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    net = TransPoseNet().to(device)
    print("loading weights:", paths.weights_file)
    net.load_state_dict(torch.load(paths.weights_file, map_location=device))
    net.eval()

    x = normalize_and_concat(acc, ori).to(device)
    print("x:", x.shape)

    with torch.no_grad():
        pose, tran = net.forward_offline(x)

    pose = pose.detach().cpu()
    tran = tran.detach().cpu()

    print("pose:", pose.shape)
    print("tran:", tran.shape)

    pose_out = data_dir / f"{args.out_prefix}_pose.pt"
    tran_out = data_dir / f"{args.out_prefix}_tran.pt"

    torch.save(pose, pose_out)
    torch.save(tran, tran_out)

    print("saved:", pose_out)
    print("saved:", tran_out)

    print("rendering video...")
    body_model = art.ParametricModel(paths.smpl_file)
    body_model.view_motion([pose], [tran])


if __name__ == "__main__":
    main()
