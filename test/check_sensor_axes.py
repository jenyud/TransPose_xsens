import sys, time, threading
sys.path.insert(0, '/home/yding263/Documents/tlv/TransPose')
import numpy as np
import xsensdeviceapi as xda

TARGET_ID = 11837972
latest_q = None
lock = threading.Lock()

class Cb(xda.XsCallback):
    def onLiveDataAvailable(self, dev, packet):
        global latest_q
        if dev.deviceId().toInt() != TARGET_ID: return
        if packet.containsOrientation():
            with lock:
                latest_q = np.array(packet.orientationQuaternion(), dtype=np.float64)

def q_to_rotmat(q):
    w,x,y,z = q
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-w*z),   2*(x*z+w*y)],
        [2*(x*y+w*z),   1-2*(x*x+z*z), 2*(y*z-w*x)],
        [2*(x*z-w*y),   2*(y*z+w*x),   1-2*(x*x+y*y)]])

cb = Cb()   # 保持引用，防止被 GC

ctrl = xda.XsControl.construct()
for p in xda.XsScanner.scanPorts():
    if not p.deviceId().isWirelessMaster(): continue
    ctrl.openPort(p.portName(), p.baudrate())
    master = ctrl.device(p.deviceId())
    master.gotoConfig()
    if master.isRadioEnabled(): master.disableRadio()
    master.enableRadio(13)
    print("等待传感器连接...")
    while master.childCount() < 1: time.sleep(0.5)
    children = master.children()
    for i in range(children.size()):
        dev = children[i]
        if dev.deviceId().toInt() == TARGET_ID:
            dev.addCallbackHandler(cb)
            print(f"已注册 {TARGET_ID}")
    master.gotoMeasurement()
    break

time.sleep(2)
print("\n按回车打印当前各轴方向，Ctrl-C 退出\n")

poses = [
    "传感器平放桌上，正面(有字/logo)朝上",
    "传感器竖起，长边朝上",
    "传感器平放，正面朝下",
    "传感器横放，长边朝右",
]

for desc in poses:
    input(f"→ {desc}，稳定后按回车...")
    with lock:
        q = latest_q.copy() if latest_q is not None else None
    if q is None:
        print("  无数据！\n"); continue
    R = q_to_rotmat(q)
    print(f"  x轴→世界: {R[:,0].round(2)}")
    print(f"  y轴→世界: {R[:,1].round(2)}")
    print(f"  z轴→世界: {R[:,2].round(2)}\n")

ctrl.destruct()
