import torch  # 引入 PyTorch 库
import torch.optim as optim  # 引入 PyTorch 中的优化器模块
import torch.nn as nn  # 引入 PyTorch 中的神经网络模块
from torch.utils.data import DataLoader  # 引入 PyTorch 中的数据加载器模块
from src.model.CNNModel import CNNModel  # 从自定义模型文件中引入 CNNModel 类
from src.data.CNN_Dataset import CNN_Dataset, transform  # 从自定义数据文件中引入 CNN_Dataset 类和 transform 变量

def train(model, trainloader, criterion, optimizer, device, epochs=5):
    """
    训练模型的函数

    参数:
    model: 训练的模型
    trainloader: 训练数据的数据加载器
    criterion: 损失函数
    optimizer: 优化器
    device: 训练使用的设备（CPU 或 GPU）
    epochs (int, optional): 训练的轮数，默认值为 5

    返回:
    model: 训练好的模型
    trainloader: 训练数据的数据加载器
    """
    model.train()  # 将模型设置为训练模式
    for epoch in range(epochs):  # 遍历每一个 epoch
        running_loss = 0.0  # 初始化当前轮次的累计损失
        correct = 0  # 初始化当前轮次的正确预测数
        total = 0  # 初始化当前轮次的总样本数
        for i, data in enumerate(trainloader, 0):  # 遍历每一个 batch
            inputs, labels = data  # 获取输入数据和对应的标签
            inputs, labels = inputs.to(device), labels.to(device)  # 将数据和标签移动到指定的设备上（CPU 或 GPU）
            optimizer.zero_grad()  # 将优化器的梯度缓存清零
            outputs = model(inputs)  # 前向传播计算模型输出
            loss = criterion(outputs, labels)  # 计算损失
            loss.backward()  # 反向传播计算梯度
            optimizer.step()  # 更新模型参数

            running_loss += loss.item()  # 累加损失值
            _, predicted = torch.max(outputs, 1)  # 获取预测的类别
            total += labels.size(0)  # 累加总样本数
            correct += (predicted == labels).sum().item()  # 累加正确预测数

            if i % 100 == 99:  # 每 100 个 batch 打印一次训练状态
                print(f"[{epoch + 1}, {i + 1}] loss: {running_loss / 100:.3f} accuracy: {100 * correct / total:.2f}%")
                running_loss = 0.0  # 重置累计损失

    print("Finished Training")  # 打印训练完成信息
    return model, trainloader  # 返回训练好的模型和数据加载器

def create_dataloader(root_dir, transform):
    """
    创建数据加载器的函数

    参数:
    root_dir (str): 数据集的根目录
    transform: 图像预处理和转换操作

    返回:
    DataLoader: 数据加载器
    """
    dataset = CNN_Dataset(root_dir=root_dir, transform=transform)  # 实例化自定义数据集
    return DataLoader(dataset, batch_size=64, shuffle=True)  # 创建数据加载器，并设置 batch size 和是否打乱数据
