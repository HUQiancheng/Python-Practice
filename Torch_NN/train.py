import os  # 引入 os 库用于文件和目录操作
import torch  # 引入 PyTorch 库
from torch.utils.data import DataLoader  # 引入 DataLoader 用于数据加载
from src.training.train_CNN import train, create_dataloader  # 从训练模块引入 train 和 create_dataloader 函数
from src.model.CNNModel import CNNModel  # 从模型模块引入 CNNModel 类
from src.data.CNN_Dataset import transform, CNN_Dataset  # 从数据模块引入 transform 和 CNN_Dataset
from src.util.visualization import plot_predictions, show_sample  # 从可视化模块引入 plot_predictions 和 show_sample 函数
from src.inference.CNN_Inference import load_model  # 从推理模块引入 load_model 函数

def main():
    """
    主函数，负责训练和评估模型
    """
    root_dir = 'data/dataset'  # 数据集的根目录
    model_path = 'src/runs/cnn_model.pth'  # 保存模型的路径
    
    # 创建数据集和数据加载器
    dataset = CNN_Dataset(root_dir, transform=transform)
    trainloader = DataLoader(dataset, batch_size=64, shuffle=True)

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
            plot_predictions(model, trainloader, device)  # 绘制模型预测结果
            return
        else:
            print("Training a new model.")  # 训练新模型

    # 创建新的模型并移动到指定设备
    model = CNNModel().to(device)
    criterion = torch.nn.CrossEntropyLoss()  # 定义损失函数
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)  # 定义优化器

    # 训练模型
    model, trainloader = train(model, trainloader, criterion, optimizer, device, epochs=5)

    # 保存训练好的模型
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    # 绘制模型预测结果
    plot_predictions(model, trainloader, device)

if __name__ == '__main__':
    main()  # 调用主函数
