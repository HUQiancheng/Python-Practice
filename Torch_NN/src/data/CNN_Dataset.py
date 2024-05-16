import os  # 用于处理文件和目录路径
from PIL import Image, ImageOps  # 用于图像处理和颜色反转
import torch  # PyTorch 库的核心
from torch.utils.data import Dataset  # 用于创建自定义数据集
from torchvision import transforms  # 用于图像数据的转换和预处理
from sklearn.model_selection import train_test_split  # 用于数据集划分

# 定义自定义数据集类
class CNN_Dataset(Dataset):
    def __init__(self, root_dir, transform=None, train=True, train_ratio=0.8):
        """
        初始化数据集

        参数:
        root_dir (str): 数据集的根目录
        transform (callable, optional): 应用于图像的转换函数
        train (bool): 指定是训练集还是验证集
        train_ratio (float): 训练集比例，默认为0.8
        """
        self.root_dir = root_dir  # 保存数据集的根目录
        self.transform = transform  # 保存图像转换函数
        self.train = train  # 是否是训练集
        self.image_files = []  # 存储所有图像文件路径
        self.labels = []  # 存储所有图像对应的标签
        self._load_data(root_dir)  # 加载数据

        # 划分训练集和验证集
        train_files, val_files, train_labels, val_labels = train_test_split(
            self.image_files, self.labels, train_size=train_ratio, stratify=self.labels, random_state=42
        )
        if self.train:
            self.image_files, self.labels = train_files, train_labels
        else:
            self.image_files, self.labels = val_files, val_labels

    def _load_data(self, root_dir):
        """
        加载数据，将图像文件路径和标签存储到相应的列表中

        参数:
        root_dir (str): 数据集的根目录
        """
        for label in range(10):  # 遍历0到9的标签目录
            folder = os.path.join(root_dir, str(label))  # 获取每个标签目录的路径
            for root, _, files in os.walk(folder):  # 遍历标签目录中的文件
                for filename in files:  # 遍历每个文件
                    if filename.endswith('.png'):  # 仅处理PNG文件
                        self.image_files.append(os.path.join(root, filename))  # 将图像文件路径添加到列表中
                        self.labels.append(label)  # 将对应的标签添加到列表中

    def __len__(self):
        """
        返回数据集的大小

        返回:
        int: 数据集中图像的数量
        """
        return len(self.image_files)  # 返回图像文件数量

    def __getitem__(self, idx):
        """
        获取指定索引的图像和标签

        参数:
        idx (int): 图像和标签的索引

        返回:
        tuple: 包含图像和标签的元组
        """
        img_name = self.image_files[idx]  # 获取图像文件路径
        image = Image.open(img_name).convert("RGBA")  # 打开图像并转换为RGBA模式

        # 创建一个白色背景
        white_background = Image.new("RGBA", image.size, (255, 255, 255, 255))  # 创建一个白色背景图像
        white_background.paste(image, (0, 0), image)  # 将原始图像粘贴到白色背景上

        # 转换为RGB模式以移除alpha通道
        image = white_background.convert("RGB")  # 将图像转换为RGB模式

        # 反转颜色
        image = ImageOps.invert(image.convert('L'))  # 将图像转换为灰度图像并进行颜色反转

        label = self.labels[idx]  # 获取对应的标签
        if self.transform:
            image = self.transform(image)  # 如果有转换函数，应用于图像
        return image, label  # 返回图像和标签

# 数据预处理和转换
transform = transforms.Compose([
    transforms.ToTensor(),  # 将图像转换为PyTorch张量
    transforms.Normalize((0.5,), (0.5,))  # 标准化图像数据
])
