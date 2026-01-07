import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm

# =========================
# 1️⃣ 数据
# =========================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

train_dataset = datasets.ImageFolder("training_set", transform=transform)
val_dataset = datasets.ImageFolder("test_set", transform=transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

# =========================
# 2️⃣ 模型
# =========================
model = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=2)

# 如果只训练头，可以冻结前面层
for name, param in model.named_parameters():
    if "head" not in name:
        param.requires_grad = False

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

# =========================
# 3️⃣ 损失函数 + 优化器
# =========================
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

# =========================
# 4️⃣ 训练循环（简单版）
# =========================
for epoch in range(5):
    model.train()
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    
    # 验证
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"Epoch {epoch+1}: Acc={correct/total:.4f}")
torch.save(model, "model/vit_cat_dog.pth")

