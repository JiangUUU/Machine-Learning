# screen.py
import numpy as np
import mss

sct = mss.mss()
monitor = sct.monitors[1]
W, H = monitor["width"], monitor["height"]

def grab_screen():
    """返回屏幕图像 (H, W, 3) BGR"""
    img = np.array(sct.grab(monitor))
    # mss 默认 BGRA -> 转 BGR
    img = img[:, :, :3][:, :, ::-1]
    return img
