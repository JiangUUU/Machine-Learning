from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("yolov8n.pt")  

    model.train(
        data="dataset/dataset.yaml",  
        epochs=50,                   
        imgsz=640,                  
        batch=8,                   
        augment=True,
        name="aimlab_model"  
    )


