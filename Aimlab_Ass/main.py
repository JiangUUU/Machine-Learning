import cv2
import numpy as np
import win32gui
import win32con
import win32api
from ultralytics import YOLO
import mss
import time

# ================= YOLO =================
model = YOLO(r"runs\detect\aimlab_model\weights\best.pt")
model.to("cuda")

# ================= Screen =================
sct = mss.mss()
monitor = sct.monitors[1]
W, H = monitor["width"], monitor["height"]

# ================= Create Window =================
cv2.namedWindow("overlay", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("overlay", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# 👉 必须先 show 一次，窗口才存在
blank = np.zeros((H, W, 3), dtype=np.uint8)
cv2.imshow("overlay", blank)
cv2.waitKey(1)

# ================= Win32 Setup =================
hwnd = win32gui.FindWindow(None, "overlay")
assert hwnd != 0, "找不到 overlay 窗口"

# 去边框
style = win32gui.GetWindowLong(hwnd, win32con.GWL_STYLE)
style &= ~(win32con.WS_CAPTION | win32con.WS_THICKFRAME)
win32gui.SetWindowLong(hwnd, win32con.GWL_STYLE, style)

# 扩展样式（关键）
ex_style = win32gui.GetWindowLong(hwnd, win32con.GWL_EXSTYLE)
win32gui.SetWindowLong(
    hwnd,
    win32con.GWL_EXSTYLE,
    ex_style
    | win32con.WS_EX_TOPMOST
    | win32con.WS_EX_LAYERED
    | win32con.WS_EX_TRANSPARENT
    | win32con.WS_EX_NOACTIVATE
)

# 黑色透明
win32gui.SetLayeredWindowAttributes(
    hwnd,
    win32api.RGB(0, 0, 0),
    0,
    win32con.LWA_COLORKEY
)

# 强制置顶一次
win32gui.SetWindowPos(
    hwnd,
    win32con.HWND_TOPMOST,
    0, 0, W, H,
    win32con.SWP_SHOWWINDOW | win32con.SWP_NOACTIVATE
)

print("Overlay ready")

# ================= Main Loop =================
while True:
    img = np.array(sct.grab(monitor))
    frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    overlay = np.zeros((H, W, 3), dtype=np.uint8)

    results = model(frame, conf=0.4, verbose=False)

    for r in results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])

            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                overlay,
                f"{conf:.2f}",
                (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

    cv2.imshow("overlay", overlay)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyAllWindows()
