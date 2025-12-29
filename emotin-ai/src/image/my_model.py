import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np

class SimpleEmotionCNN(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*12*12, 128), nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        return self.fc(self.conv(x))

class MyImageEmotionRecognizer:
    def __init__(self, model_path='my_emotion_model.pth'):
        self.labels = ['愤怒', '厌恶', '恐惧', '开心', '悲伤', '惊讶', '中性']
        self.model = SimpleEmotionCNN(num_classes=len(self.labels))
        self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((48, 48)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

    def predict(self, image_path):
        img = Image.open(image_path)
        img = self.transform(img).unsqueeze(0)
        with torch.no_grad():
            out = self.model(img)
            prob = torch.softmax(out, dim=1).cpu().numpy()[0]
        result = {self.labels[i]: float(prob[i]) for i in range(len(self.labels))}
        print('自定义模型情绪概率分布:', result)
        return self.labels[int(np.argmax(prob))]
