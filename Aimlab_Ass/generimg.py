import cv2
import os
import random
import re

video_path = "E:\\即时重放\\Aim Lab\\Aim Lab 2026.01.01 - 09.51.15.01.mp4"
save_dir = "dataset/images"
os.makedirs(save_dir, exist_ok=True)

# 找到当前目录下已有的 ball 图片编号
existing_files = [f for f in os.listdir(save_dir) if re.match(r'ball\d+\.png', f)]
if existing_files:
    max_num = max([int(re.findall(r'\d+', f)[0]) for f in existing_files])
else:
    max_num = 0  # 没有已有图片则从 1 开始

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise ValueError("无法打开视频，请检查路径或格式")

frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print("视频总帧数:", frame_count)

num_samples = min(10, frame_count)
sample_frames = random.sample(range(frame_count), num_samples)
print("将随机抽取的帧数:", len(sample_frames))

count = 0
saved = 0
for i in range(frame_count):
    ret, frame = cap.read()
    if not ret:
        break
    if i in sample_frames:
        max_num += 1
        cv2.imwrite(os.path.join(save_dir, f"ball{max_num}.png"), frame)
        saved += 1

cap.release()
print(f"已保存 {saved} 张随机 PNG 图片到 {save_dir}，命名从 ball{max_num - saved + 1} 开始")


# label_dir = "dataset/labels"  # 标签目录
# txt_files = [f for f in os.listdir(label_dir) if f.endswith(".txt") and f.startswith("ball")]

# for file_name in txt_files:
#     file_path = os.path.join(label_dir, file_name)
#     with open(file_path, "r") as f:
#         lines = f.readlines()

#     new_lines = []
#     for line in lines:
#         parts = line.strip().split()
#         if len(parts) >= 5:
#             parts[0] = "0"  # 类别改为 0
#             new_lines.append(" ".join(parts) + "\n")

#     with open(file_path, "w") as f:
#         f.writelines(new_lines)

# print(f"已修改 {len(txt_files)} 个以 ball 开头的标签文件，类别已改为 0")
