from src import *
import os  # 引入 os 库用于文件和目录操作
import torch  # 引入 PyTorch 库
from torch.utils.data import DataLoader  # 引入 DataLoader 用于数据加载

def main():
    """
    主函数，负责训练和评估模型
    """
    root_dir = 'data/dataset'  # 数据集的根目录
    model_path = 'src/runs/cnn_model.pth'  # 保存模型的路径
    
    # 创建训练数据加载器和验证数据加载器
    trainloader = create_dataloader(root_dir, transform, batch_size=64, train=True)
    valloader = create_dataloader(root_dir, transform, batch_size=64, train=False)

    # 确定使用的设备（GPU 或 CPU）
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 检查是否存在已保存的模型
    if os.path.exists(model_path):
        use_saved_model = input("Saved model found. Do you want to use it? (y/n): ").strip().lower()
        if use_saved_model == 'y':
            model = load_model(model_path)  # 加载已保存的模型
            model.to(device)  # 将模型移动到指定设备
            print("Loaded saved model.")
            plot_predictions(model, valloader, device)  # 绘制模型预测结果
            summarize_model(model)  # 总结并打印模型信息
            return
        else:
            print("Training a new model.")  # 训练新模型

    # 创建新的模型并移动到指定设备
    model = CNNModel().to(device)
    summarize_model(model)  # 总结并打印模型信息
    criterion = torch.nn.CrossEntropyLoss()  # 定义损失函数
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)  # 定义优化器

    # 训练模型
    model, history = train(model, trainloader, valloader, criterion, optimizer, device, epochs=5)

    # 保存训练好的模型
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    # 绘制训练历史
    plot_training_history(history)

    # 绘制模型预测结果
    plot_predictions(model, valloader, device)

if __name__ == '__main__':
    main()  # 调用主函数
