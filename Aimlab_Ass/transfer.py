from ultralytics import YOLO

if __name__ == "__main__":
    # 1. 加载预训练模型
    model = YOLO("yolov8n.pt")  # 也可以换成 yolov8s.pt/m.pt

    # 2. 开始训练（迁移学习）
    model.train(
        data="dataset/dataset.yaml",  # 数据集配置
        epochs=50,                     # 根据数据量调整
        imgsz=640,                     # 输入尺寸
        batch=8,                       # batch大小
        augment=True,
        name="aimlab_model"             # 保存的模型文件夹名
    )


