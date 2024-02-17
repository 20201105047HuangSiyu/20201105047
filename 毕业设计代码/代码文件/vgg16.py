import torch.nn as nn
import torch.nn.functional as F

# 定义简单的卷积神经网络模型

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        # 输入图像通道数为1（灰度图像），输出通道数为64，卷积核大小为3x3，填充为1
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        # 最大池化层，降低特征图的空间尺寸
        self.pool = nn.MaxPool2d(2, 2)
        # 第二个卷积层，输入通道数为64，输出通道数为128，卷积核大小为3x3，填充为1
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        # 全连接层，输入大小为 128 * 8 * 8，输出大小为256
        self.fc1 = nn.Linear(128 * 8 * 8, 256)
        # 输出层，输出大小为类别数
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        # 第一层卷积和激活
        x = F.relu(self.conv1(x))
        # 最大池化
        x = self.pool(x)
        # 第二层卷积和激活
        x = F.relu(self.conv2(x))
        # 最大池化
        x = self.pool(x)
        # 将特征图展平
        x = x.view(-1, 128 * 8 * 8)
        # 全连接层和激活
        x = F.relu(self.fc1(x))
        # 输出层
        x = self.fc2(x)
        return x
    
class VGG16(nn.Module):
    def __init__(self, num_classes):
        super(VGG16, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),  # 输入通道数为1，输出通道数为64，卷积核大小为3x3，填充为1
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),  # 输入通道数为64，输出通道数为64，卷积核大小为3x3，填充为1
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 进行2x2的最大池化操作，步长为2

            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # 输入通道数为64，输出通道数为128，卷积核大小为3x3，填充为1
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),  # 输入通道数为128，输出通道数为128，卷积核大小为3x3，填充为1
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 进行2x2的最大池化操作，步长为2

            nn.Conv2d(128, 256, kernel_size=3, padding=1),  # 输入通道数为128，输出通道数为256，卷积核大小为3x3，填充为1
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),  # 输入通道数为256，输出通道数为256，卷积核大小为3x3，填充为1
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),  # 输入通道数为256，输出通道数为256，卷积核大小为3x3，填充为1
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 进行2x2的最大池化操作，步长为2

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        # 修改全连接层的输入维度
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)  # 进行特征提取
        x = x.view(x.size(0), -1)  # 将输入的维度转换为(batch_size, -1)
        x = self.classifier(x)  # 进行全连接层的计算
        return x

class SimpleVGG16(nn.Module):
    def __init__(self, num_classes):
        super(SimpleVGG16, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # 修改全连接层的输入维度
        self.classifier = nn.Sequential(
            nn.Linear(256 * 4 * 4, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# 设置类别数量
num_classes = 10  # 替换为你的数据集类别数量

# 创建VGG16模型
vgg16_model = VGG16(num_classes)

# 打印模型结构
print(vgg16_model)
