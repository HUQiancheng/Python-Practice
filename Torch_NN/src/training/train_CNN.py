import torch  # 引入 PyTorch 库
import torch.optim as optim  # 引入 PyTorch 中的优化器模块
import torch.nn as nn  # 引入 PyTorch 中的神经网络模块
from torch.utils.data import DataLoader  # 引入 PyTorch 中的数据加载器模块
from src.model.CNNModel import CNNModel  # 从自定义模型文件中引入 CNNModel 类
from src.data.CNN_Dataset import CNN_Dataset, transform  # 从自定义数据文件中引入 CNN_Dataset 类和 transform 变量

def train(model, trainloader, valloader, criterion, optimizer, device, epochs=5, print_interval=100):
    """
    训练模型的函数

    参数:
    model: 训练的模型
    trainloader: 训练数据的数据加载器
    valloader: 验证数据的数据加载器
    criterion: 损失函数
    optimizer: 优化器
    device: 训练使用的设备（CPU 或 GPU）
    epochs (int, optional): 训练的轮数，默认值为 5
    print_interval (int, optional): 每隔多少个 batch 打印一次，默认值为 100

    返回:
    tuple: 训练好的模型，训练过程中的损失和准确率记录
    """
    history = {'train_loss': [], 'train_accuracy': [], 'val_loss': [], 'val_accuracy': []}  # 初始化记录训练和验证损失、准确率的字典

    for epoch in range(epochs):  # 遍历每一个 epoch
        running_loss = 0.0  # 初始化当前轮次的累计损失
        correct = 0  # 初始化当前轮次的正确预测数
        total = 0  # 初始化当前轮次的总样本数

        # 训练阶段
        model.train()
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

            if (i + 1) % print_interval == 0:  # 每 print_interval 个 batch 打印一次
                print(f"Batch [{i + 1}/{len(trainloader)}] "
                      f"Running loss: {running_loss / (i + 1):.3f}, accuracy: {100 * correct / total:.2f}%")

        epoch_loss = running_loss / len(trainloader)  # 计算当前轮次的平均损失
        epoch_accuracy = 100 * correct / total  # 计算当前轮次的准确率
        history['train_loss'].append(epoch_loss)  # 记录当前轮次的训练损失
        history['train_accuracy'].append(epoch_accuracy)  # 记录当前轮次的训练准确率

        # 验证阶段
        model.eval()  # 将模型设置为评估模式
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():  # 不计算梯度
            for i, data in enumerate(valloader, 0):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        epoch_val_loss = val_loss / len(valloader)  # 计算验证损失
        epoch_val_accuracy = 100 * val_correct / val_total  # 计算验证准确率
        history['val_loss'].append(epoch_val_loss)  # 记录当前轮次的验证损失
        history['val_accuracy'].append(epoch_val_accuracy)  # 记录当前轮次的验证准确率

        # 打印当前轮次的训练和验证损失及准确率
        print(f"Epoch [{epoch + 1}/{epochs}] "
              f"Train loss: {epoch_loss:.3f}, accuracy: {epoch_accuracy:.2f}% "
              f"Val loss: {epoch_val_loss:.3f}, accuracy: {epoch_val_accuracy:.2f}%")

    print("Finished Training")  # 打印训练完成信息
    return model, history  # 返回训练好的模型和训练记录

def create_dataloader(root_dir, transform, batch_size=64, train=True, train_ratio=0.8):
    """
    创建数据加载器的函数

    参数:
    root_dir (str): 数据集的根目录
    transform: 图像预处理和转换操作
    batch_size (int, optional): 每个 batch 的样本数量，默认为 64
    train (bool, optional): 指定是训练集还是验证集，默认为 True
    train_ratio (float, optional): 训练集的比例，默认为 0.8

    返回:
    DataLoader: 数据加载器
    """
    dataset = CNN_Dataset(root_dir=root_dir, transform=transform, train=train, train_ratio=train_ratio)  # 实例化自定义数据集
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)  # 创建数据加载器，并设置 batch size 和是否打乱数据
