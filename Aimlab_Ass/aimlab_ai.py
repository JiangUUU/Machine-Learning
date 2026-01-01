import cv2
import numpy as np
import win32gui
import win32con
import win32api
from ultralytics import YOLO
import mss
import time
import ctypes
from ctypes import wintypes

# ================= SendInput: Relative Mouse =================
user32 = ctypes.WinDLL('user32', use_last_error=True)
INPUT_MOUSE = 0
MOUSEEVENTF_MOVE = 0x0001

class MOUSEINPUT(ctypes.Structure):
    _fields_ = (
        ('dx', wintypes.LONG),
        ('dy', wintypes.LONG),
        ('mouseData', wintypes.DWORD),
        ('dwFlags', wintypes.DWORD),
        ('time', wintypes.DWORD),
        ('dwExtraInfo', ctypes.POINTER(wintypes.ULONG)),
    )

class INPUT(ctypes.Structure):
    class _INPUT(ctypes.Union):
        _fields_ = (('mi', MOUSEINPUT),)
    _anonymous_ = ('_input',)
    _fields_ = (('type', wintypes.DWORD), ('_input', _INPUT))

def move_relative(dx, dy):
    mi = MOUSEINPUT()
    mi.dx = int(dx)
    mi.dy = int(dy)
    mi.dwFlags = MOUSEEVENTF_MOVE
    inputs = INPUT(type=INPUT_MOUSE, mi=mi)
    user32.SendInput(1, ctypes.byref(inputs), ctypes.sizeof(INPUT))

def left_click():
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.WinDLL('user32', use_last_error=True)
    INPUT_MOUSE = 0
    MOUSEEVENTF_LEFTDOWN = 0x0002
    MOUSEEVENTF_LEFTUP   = 0x0004

    class MOUSEINPUT(ctypes.Structure):
        _fields_ = (
            ('dx', wintypes.LONG),
            ('dy', wintypes.LONG),
            ('mouseData', wintypes.DWORD),
            ('dwFlags', wintypes.DWORD),
            ('time', wintypes.DWORD),
            ('dwExtraInfo', ctypes.POINTER(wintypes.ULONG)),
        )

    class INPUT(ctypes.Structure):
        class _INPUT(ctypes.Union):
            _fields_ = (('mi', MOUSEINPUT),)
        _anonymous_ = ('_input',)
        _fields_ = (('type', wintypes.DWORD), ('_input', _INPUT))

    inputs = (INPUT * 2)()
    inputs[0].type = INPUT_MOUSE
    inputs[0].mi.dwFlags = MOUSEEVENTF_LEFTDOWN
    inputs[1].type = INPUT_MOUSE
    inputs[1].mi.dwFlags = MOUSEEVENTF_LEFTUP

    user32.SendInput(2, ctypes.byref(inputs), ctypes.sizeof(INPUT))

# ================= YOLO =================
model = YOLO(r"runs\detect\aimlab_model\weights\best.pt")
model.to("cuda")

# ================= Screen =================
sct = mss.mss()
monitor = sct.monitors[1]
W, H = monitor["width"], monitor["height"]

# ================= Overlay Window =================
cv2.namedWindow("overlay", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("overlay", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

blank = np.zeros((H, W, 3), dtype=np.uint8)
cv2.imshow("overlay", blank)
cv2.waitKey(1)

hwnd = win32gui.FindWindow(None, "overlay")
assert hwnd != 0, "找不到 overlay 窗口"

style = win32gui.GetWindowLong(hwnd, win32con.GWL_STYLE)
style &= ~(win32con.WS_CAPTION | win32con.WS_THICKFRAME)
win32gui.SetWindowLong(hwnd, win32con.GWL_STYLE, style)

ex_style = win32gui.GetWindowLong(hwnd, win32con.GWL_EXSTYLE)
win32gui.SetWindowLong(
    hwnd,
    win32con.GWL_EXSTYLE,
    ex_style | win32con.WS_EX_TOPMOST | win32con.WS_EX_LAYERED |
    win32con.WS_EX_TRANSPARENT | win32con.WS_EX_NOACTIVATE
)

win32gui.SetLayeredWindowAttributes(
    hwnd, win32api.RGB(0, 0, 0), 0, win32con.LWA_COLORKEY
)

win32gui.SetWindowPos(
    hwnd, win32con.HWND_TOPMOST, 0, 0, W, H,
    win32con.SWP_SHOWWINDOW | win32con.SWP_NOACTIVATE
)

print("Overlay ready. 按 ESC 退出")

# ================= Main Loop =================
while True:
    img = np.array(sct.grab(monitor))
    frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    overlay = np.zeros((H, W, 3), dtype=np.uint8)

    screen_center_x = monitor["left"] + W // 2
    screen_center_y = monitor["top"] + H // 2

    targets = []

    results = model(frame, conf=0.4, verbose=False)
    for r in results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])

            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(overlay, f"{conf:.2f}", (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            tx_global = x1 + monitor["left"] + (x2 - x1) // 2
            ty_global = y1 + monitor["top"] + (y2 - y1) // 2

            targets.append((tx_global, ty_global))

            tx_local = (x1 + x2) // 2
            ty_local = (y1 + y2) // 2
            cv2.circle(overlay, (tx_local, ty_local), 5, (0, 255, 0), -1)

    target = None
    if targets:
        distances = [
            ((tx - screen_center_x) ** 2 + (ty - screen_center_y) ** 2)
            for (tx, ty) in targets
        ]
        min_index = distances.index(min(distances))
        target = targets[min_index]

        tx_local = target[0] - monitor["left"]
        ty_local = target[1] - monitor["top"]
        cv2.circle(overlay, (int(tx_local), int(ty_local)), 7, (0, 0, 255), 2)

    cv2.circle(overlay, (W // 2, H // 2), 5, (255, 0, 0), -1)

    cv2.imshow("overlay", overlay)

    if target:
        current_x, current_y = win32api.GetCursorPos()
        dx = target[0] - current_x
        dy = target[1] - current_y
        move_relative(dx, dy)
        time.sleep(0.0001)
        left_click()

    if cv2.waitKey(1) & 0xFF == 27:
        break

    time.sleep(0.0001)
cv2.destroyAllWindows()