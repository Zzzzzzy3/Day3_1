import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 1. 检查GPU是否可用，并选择使用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. 将模型和数据移动到GPU


train_path = 'F:/SARdata/tendata/train'
test_path = 'F:/SARdata/tendata/test'
batch_size = 16


# 定义卷积神经网络模型
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=5)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=4)
        self.conv2 = nn.Conv2d(96, 256, kernel_size=5, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(256, 384, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(384, 384, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(1024, 1024)  # 输入维度需要根据前面的层计算
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024, 1024)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(1024, 10)

    def forward(self, x):
        x = nn.ReLU()(self.conv1(x))
        x = self.pool1(x)
        x = nn.ReLU()(self.conv2(x))
        x = self.pool2(x)
        x = nn.ReLU()(self.conv3(x))
        x = nn.ReLU()(self.conv4(x))
        x = nn.ReLU()(self.conv5(x))
        x = self.pool3(x)
        x = torch.flatten(x, 1)  # 展平池化层的输出
        # print(x.shape)
        x = nn.ReLU()(self.fc1(x))
        x = self.dropout1(x)
        x = nn.ReLU()(self.fc2(x))
        x = self.dropout2(x)
        x = nn.LogSoftmax(dim=1)(self.fc3(x))
        return x


# 转换数据生成器
transform_train = transforms.Compose([
    transforms.Resize((100, 100)),
    transforms.Grayscale(num_output_channels=3),  # 将灰度图像转换为RGB
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_test = transforms.Compose([
    transforms.Resize((100, 100)),
    transforms.Grayscale(num_output_channels=3),  # 将灰度图像转换为RGB
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 加载数据集
train_dataset = datasets.ImageFolder(root=train_path, transform=transform_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

validation_dataset = datasets.ImageFolder(root=test_path, transform=transform_test)
validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)

# 实例化模型
model = ConvNet()

# 将模型移动到 GPU 上
model.to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=5e-4, momentum=0.9)


# 训练模型
# 训练模型
def train(model, train_loader, criterion, optimizer, epochs, validation_loader=None):
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            # 将数据和标签移动到 GPU 上
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}")

        if validation_loader:
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for images, labels in validation_loader:
                    # 将数据和标签移动到 GPU 上
                    images, labels = images.to(device), labels.to(device)

                    outputs = model(images)
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            val_acc = correct / total
            print(f"Validation Accuracy: {val_acc:.4f}")


# 训练模型
train(model, train_loader, criterion, optimizer, epochs=300, validation_loader=validation_loader)
