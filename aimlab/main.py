# yolo_sendinput_debug_overlay.py
import time
import cv2
import numpy as np
from ultralytics import YOLO
import mss
import pyautogui

from sendinput_mouse import move_absolute, left_click  # 绝对坐标版本
from screen import grab_screen  # 你自己的截图函数

# =============================== 1️⃣ YOLO ===============================
model = YOLO(r"runs\detect\aimlab_model\weights\best.pt")
model.to("cuda")

# =============================== 2️⃣ 屏幕参数 ===============================
sct = mss.mss()
monitor = sct.monitors[1]
W, H = monitor["width"], monitor["height"]
CX, CY = W // 2, H // 2  # 屏幕中心

print(f"抓屏分辨率: {W}x{H}")

# =============================== 3️⃣ 主循环 ===============================
print("启动，按 Q 退出")

while True:
    # 截屏
    frame = grab_screen()  # 或者 np.array(sct.grab(monitor))
    overlay = frame.copy()

    target = None

    # YOLO 检测
    results = model(frame, conf=0.4, verbose=False)

    for r in results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])

            # 绘制框
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(overlay, f"{conf:.2f}", (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # 取第一个目标
            tx = (x1 + x2) // 2
            ty = (y1 + y2) // 2
            target = (tx, ty)
            break
        break

    if target:
        # SendInput 绝对坐标
        abs_x = int(target[0] * 65535 / (W - 1))
        abs_y = int(target[1] * 65535 / (H - 1))

        # 移动鼠标
        move_absolute(abs_x, abs_y)

        # 自动点击（目标靠近中心）
        if abs(target[0] - CX) < 8 and abs(target[1] - CY) < 8:
            left_click()

        # 红点：YOLO 框中心
        cv2.circle(overlay, target, 5, (0, 0, 255), -1)

        # 调试打印
        print(f"YOLO 框中心: {target} -> 绝对坐标: ({abs_x},{abs_y})")

    # 蓝点：屏幕中心
    cv2.circle(overlay, (CX, CY), 5, (255, 0, 0), -1)

    # 显示
    cv2.imshow("YOLO + SendInput Overlay", overlay)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    time.sleep(0.002)

cv2.destroyAllWindows()
