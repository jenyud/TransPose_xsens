import cv2
import runpy

orig_fourcc = cv2.VideoWriter_fourcc
orig_videowriter = cv2.VideoWriter

# possible H264-related fourcc integer values
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

    # args usually: filename, fourcc, fps, frameSize, ...
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

runpy.run_path("example.py", run_name="__main__")
