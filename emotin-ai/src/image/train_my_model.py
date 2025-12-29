import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from my_model import SimpleEmotionCNN

# 配置
BATCH_SIZE = 64
EPOCHS = 20
LR = 0.001
NUM_CLASSES = 7
MODEL_PATH = 'my_emotion_model.pth'
FER_LABELS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

# 数据集处理（FER2013格式，train/val/test子文件夹，每类一个子目录）
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

def get_dataloaders(data_dir):
    train_ds = datasets.ImageFolder(root=f'{data_dir}/train', transform=transform)
    val_ds = datasets.ImageFolder(root=f'{data_dir}/val', transform=transform)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    return train_loader, val_loader

def train(data_dir):
    train_loader, val_loader = get_dataloaders(data_dir)
    model = SimpleEmotionCNN(num_classes=NUM_CLASSES)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    best_acc = 0
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        # 验证
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        acc = correct / total
        print(f'Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss:.4f}, Val Acc: {acc:.4f}')
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), MODEL_PATH)
            print('模型已保存！')

if __name__ == '__main__':
    # 假设数据集在 data/fer2013
    train('data/fer2013')
