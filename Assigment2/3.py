import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import time

# 1. 定义ResNet-32
def conv3x3(in_channels, out_channels, stride):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = conv3x3(out_channels, out_channels, stride=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if in_channels != out_channels:
            self.downsample = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(out_channels))
        else:
            self.downsample = None

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample:
            x = self.downsample(x)
        out += x
        out = torch.relu(out)
        return out

class ResNet32(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet32, self).__init__()
        self.device1 = torch.device("cuda:0")
        self.device2 = torch.device("cuda:1")
        self.conv = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False).to(self.device1)
        self.bn = nn.BatchNorm2d(16).to(self.device1)
        self.layer1 = self._make_layer(16, 16, 5, stride=1).to(self.device1)
        self.layer2 = self._make_layer(16, 32, 5, stride=2).to(self.device2)
        self.layer3 = self._make_layer(32, 64, 5, stride=2).to(self.device2)
        self.linear = nn.Linear(64, num_classes).to(self.device2)

    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = []
        layers.append(BasicBlock(in_channels, out_channels, stride))
        for i in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.to(self.device1)
        out = torch.relu(self.bn(self.conv(x)))
        out = self.layer1(out)
        out = out.to(self.device2)
        out = self.layer2(out)
        out = self.layer3(out)
        out = nn.AvgPool2d(8)(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

# 2. 加载CIFAR-10数据集
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=True, num_workers=2)

# 3. 定义损失和优化器
net = ResNet32(num_classes=10)
criterion = nn.CrossEntropyLoss().to(torch.device("cuda:1"))
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

# 4. 训练模型
for epoch in range(100):
    start_time = time.time()
    net.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for i, (images, labels) in enumerate(trainloader):
        labels = labels.to(torch.device("cuda:1"))
        optimizer.zero_grad()
        outputs = net(images)

        # 计算准确度
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        loss = criterion(outputs, labels)
        running_loss += loss.item()

        loss.backward()
        optimizer.step()

    accuracy = 100. * correct / total  # 计算准确度
    end_time = time.time()
    elapsed_time = end_time - start_time
    avg_loss = running_loss / (i + 1)

    print(f"Epoch [{epoch + 1}/100], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%, Time: {elapsed_time:.2f} seconds")

print('Finished Training')
