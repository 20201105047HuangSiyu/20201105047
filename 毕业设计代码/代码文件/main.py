import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from 显示数据集信息 import run
import numpy as np
from vgg16 import VGG16, SimpleVGG16, SimpleCNN
# 加载数据集
train_images, train_labels, test_images, test_labels, train_nums, test_nums = run()

# 数据预处理
train_images = train_images[:, np.newaxis, :, :]
train_labels = train_labels - 1
test_images = test_images[:, np.newaxis, :, :]
test_labels = test_labels - 1

# 超参数设置
num_epochs = 30
learning_rate = 0.001
num_classes = len(set(train_labels))
batch_size = 32
gamma = 0.5

# 创建VGG16模型
# vgg16_model = VGG16(num_classes)
vgg16_model = SimpleVGG16(num_classes)
# vgg16_model = SimpleCNN(num_classes)

# 转换为PyTorch张量并创建数据集
train_dataset = TensorDataset(torch.from_numpy(train_images).float(), torch.tensor(train_labels, dtype=torch.long))
test_dataset = TensorDataset(torch.from_numpy(test_images).float(), torch.tensor(test_labels, dtype=torch.long))

# 创建数据加载器
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(vgg16_model.parameters(), lr=learning_rate)

# 学习率衰减设置
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=gamma)

# 权重初始化
def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

# 在模型创建后应用初始化
vgg16_model.apply(weights_init)

# 训练模型
for epoch in range(num_epochs):
    for inputs, labels in train_dataloader:
        optimizer.zero_grad()
        outputs = vgg16_model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    # 每个epoch结束后可以输出一些信息
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    
    # 学习率衰减
    scheduler.step()

# 保存模型
torch.save(vgg16_model.state_dict(), 'vgg16_model.pth')

# 模型训练完成，可以在测试集上评估模型性能
with torch.no_grad():
    correct = 0
    total = 0
    for inputs, labels in test_dataloader:
        outputs = vgg16_model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f'Test Accuracy: {accuracy:.2%}')
