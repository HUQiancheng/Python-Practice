import matplotlib.pyplot as plt  # 引入 matplotlib 库用于绘图
import numpy as np  # 引入 NumPy 库用于数值计算
import torch  # 引入 PyTorch 库

def imshow(img, ax):
    """
    显示图像的辅助函数

    参数:
    img: 图像张量
    ax: matplotlib 子图对象
    """
    img = img / 2 + 0.5  # 反标准化，将图像数据从 [-1, 1] 还原到 [0, 1]
    npimg = img.numpy()  # 将图像张量转换为 NumPy 数组
    ax.imshow(np.transpose(npimg, (1, 2, 0)), cmap='gray')  # 转置并显示图像

def show_sample(dataset, index):
    """
    显示数据集中某一索引处的图像和标签

    参数:
    dataset: 数据集对象
    index: 图像和标签的索引
    """
    image, label = dataset[index]  # 获取指定索引处的图像和标签
    fig, ax = plt.subplots()  # 创建一个子图
    imshow(image, ax)  # 显示图像
    plt.title(f'Label: {label}')  # 设置标题为标签值
    plt.show()  # 显示子图

def plot_predictions(model, dataloader, device):
    """
    绘制模型的预测结果

    参数:
    model: 训练好的模型
    dataloader: 数据加载器
    device: 设备（CPU 或 GPU）
    """
    model.eval()  # 将模型设置为评估模式
    images, labels, predictions = [], [], []  # 初始化图像、标签和预测列表
    with torch.no_grad():  # 在不计算梯度的上下文中进行预测
        for i, data in enumerate(dataloader):  # 遍历数据加载器中的数据
            inputs, labels_batch = data  # 获取输入数据和对应的标签
            inputs, labels_batch = inputs.to(device), labels_batch.to(device)  # 将数据和标签移动到指定的设备上（CPU 或 GPU）
            outputs = model(inputs)  # 获取模型输出
            _, predicted_batch = torch.max(outputs, 1)  # 获取预测的类别

            images.extend(inputs.cpu())  # 将输入数据添加到图像列表中，并移动到 CPU
            labels.extend(labels_batch.cpu())  # 将标签添加到标签列表中，并移动到 CPU
            predictions.extend(predicted_batch.cpu())  # 将预测结果添加到预测列表中，并移动到 CPU

            if len(images) >= 100:  # 如果图像数量达到 100 张，停止获取更多数据
                break

    fig, axes = plt.subplots(10, 10, figsize=(15, 15))  # 创建 10x10 的子图网格
    for idx, ax in enumerate(axes.flat):  # 遍历每个子图
        if idx < len(images):  # 如果索引在图像列表范围内
            img = images[idx]  # 获取图像
            img = img / 2 + 0.5  # 反标准化，将图像数据从 [-1, 1] 还原到 [0, 1]
            npimg = img.numpy().squeeze()  # 将图像张量转换为 NumPy 数组并去掉多余的维度
            ax.imshow(npimg, cmap='gray')  # 显示图像
            ax.set_title(f"Prediction: {predictions[idx].item()}", fontsize=8)  # 设置标题为预测结果
            ax.axis('off')  # 关闭坐标轴
    plt.tight_layout()  # 调整子图间距
    plt.show()  # 显示绘图结果

def plot_training_history(history):
    """
    绘制训练和验证过程中的损失和准确率曲线

    参数:
    history (dict): 训练过程中记录的损失和准确率
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))  # 创建一个包含两个子图的绘图窗口

    # 绘制训练和验证损失曲线
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Val Loss')
    ax1.set_title('Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()

    # 绘制训练和验证准确率曲线
    ax2.plot(history['train_accuracy'], label='Train Accuracy')
    ax2.plot(history['val_accuracy'], label='Val Accuracy')
    ax2.set_title('Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()

    plt.tight_layout()  # 调整子图间距
    plt.show()  # 显示绘图结果
