import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import timm
from config import MODEL_PATH, TESTSET_PATH, LOG_PATH,TRAININGSET_PATH



EPOCHS = 10
device = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# 1️⃣ 数据
# =========================
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
    transforms.ToTensor(),
])


test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

train_dataset = datasets.ImageFolder(
    TRAININGSET_PATH,
    transform=train_transform
)

test_dataset = datasets.ImageFolder(
    TESTSET_PATH,
    transform=test_transform
)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
writer = SummaryWriter(log_dir=LOG_PATH)

# =========================
# 2️⃣ 模型
# =========================
model = timm.create_model("vit_base_patch16_224",
                          pretrained=True, num_classes=2)

# 如果只训练头，可以冻结前面层
for name, param in model.named_parameters():
    if "head" not in name:
        param.requires_grad = False

model = model.to(device)

# =========================
# 3️⃣ 损失函数 + 优化器
# =========================
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

# =========================
# 4️⃣ 训练循环
# =========================
global_step = 0
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for batch_idx,(imgs, labels) in enumerate(train_loader):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        writer.add_scalar('Train/Loss', loss.item(), global_step)
        running_loss+=loss.item()
        global_step+=1
    
    avg_train_loss = running_loss/len(train_loader)
    # 验证
    model.eval()
    correct, total = 0, 0
    val_loss = 0.0
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            
            loss = criterion(outputs,labels)
            val_loss += loss.item()
            
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    val_acc = correct / total
    val_loss /= len(test_loader)
    
    writer.add_scalar('Val/Acc',val_acc,epoch)
    writer.add_scalar('Val/Loss',val_loss,epoch)
    writer.add_scalar('Train/Loss_epoch',avg_train_loss,epoch)
        
    print(
        f"Epoch [{epoch+1}/{EPOCHS}] "
        f"TrainLoss={avg_train_loss:.4f} "
        f"ValLoss={val_loss:.4f} "
        f"ValAcc={val_acc:.4f}"
    )
torch.save(model.state_dict(), MODEL_PATH)
writer.close()