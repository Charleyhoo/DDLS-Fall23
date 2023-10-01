import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import time

# 1. 数据预处理和加载
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=512, shuffle=True, num_workers=4)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=512, shuffle=False, num_workers=4)

# 2. 定义ResNet-32模型
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = nn.ReLU()(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = nn.ReLU()(out)
        return out

class ResNet32(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet32, self).__init__()
        self.conv = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(16, 16, 5, stride=1)
        self.layer2 = self._make_layer(16, 32, 5, stride=2)
        self.layer3 = self._make_layer(32, 64, 5, stride=2)
        self.linear = nn.Linear(64, num_classes)

    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(BasicBlock(in_channels, out_channels, stride))
        for _ in range(num_blocks-1):
            layers.append(BasicBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = nn.ReLU()(self.bn(self.conv(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = nn.AvgPool2d(8)(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

# 3. 实例化模型、损失函数和优化器
net = ResNet32()

# 使用数据并行
if torch.cuda.device_count() > 1:
    net = nn.DataParallel(net)

net.to('cuda')
criterion = nn.CrossEntropyLoss().cuda()
optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

# 4. 训练模型
for epoch in range(100):
    start_time = time.time()  # 记录开始时间
    net.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for i, (images, labels) in enumerate(trainloader):
        images, labels = images.to('cuda'), labels.to('cuda')
        optimizer.zero_grad()
        outputs = net(images)

        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        loss = criterion(outputs, labels)
        running_loss += loss.item()

        loss.backward()
        optimizer.step()

    accuracy = 100. * correct / total  # 计算准确度
    end_time = time.time()  # 记录结束时间
    elapsed_time = end_time - start_time  # 计算消耗的时间
    avg_loss = running_loss / (i + 1)

    print(f"Epoch [{epoch + 1}/100], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%, Time: {elapsed_time:.2f} seconds")

print('Finished Training')