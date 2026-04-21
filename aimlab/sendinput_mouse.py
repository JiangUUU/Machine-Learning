# sendinput_mouse.py
import ctypes
from ctypes import wintypes

# Windows API
user32 = ctypes.windll.user32
SendInput = user32.SendInput

# 类型定义
ULONG_PTR = wintypes.LPARAM  # ctypes 没有 ULONG_PTR, 用 LPARAM 替代
DWORD = wintypes.DWORD
LONG = wintypes.LONG

class MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", LONG),
        ("dy", LONG),
        ("mouseData", DWORD),
        ("dwFlags", DWORD),
        ("time", DWORD),
        ("dwExtraInfo", ULONG_PTR),
    ]

class INPUT(ctypes.Structure):
    class _INPUT(ctypes.Union):
        _fields_ = [("mi", MOUSEINPUT)]
    _anonymous_ = ("_input",)
    _fields_ = [
        ("type", DWORD),
        ("_input", _INPUT),
    ]

# 常量
MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_ABSOLUTE = 0x8000
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004

# ----------------------------- 绝对移动 -----------------------------
def move_absolute(x, y):
    """x, y 屏幕绝对坐标，范围 0~65535"""
    inp = INPUT()
    inp.type = 0  # INPUT_MOUSE
    inp.mi.dx = x
    inp.mi.dy = y
    inp.mi.mouseData = 0
    inp.mi.dwFlags = MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE
    inp.mi.time = 0
    inp.mi.dwExtraInfo = 0
    SendInput(1, ctypes.byref(inp), ctypes.sizeof(inp))

# ----------------------------- 左键点击 -----------------------------
def left_click():
    inp = INPUT()
    inp.type = 0
    inp.mi.dx = 0
    inp.mi.dy = 0
    inp.mi.mouseData = 0
    inp.mi.dwFlags = MOUSEEVENTF_LEFTDOWN
    inp.mi.time = 0
    inp.mi.dwExtraInfo = 0
    SendInput(1, ctypes.byref(inp), ctypes.sizeof(inp))

    inp.mi.dwFlags = MOUSEEVENTF_LEFTUP
    SendInput(1, ctypes.byref(inp), ctypes.sizeof(inp))
