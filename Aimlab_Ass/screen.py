# screen.py
import mss
import numpy as np
import cv2

def grab_screen():
    with mss.mss() as sct:
        monitor = sct.monitors[1]  # 主屏
        img = np.array(sct.grab(monitor))
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
