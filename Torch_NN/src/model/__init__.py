# 从 CNNModel 模块中导入 CNNModel 类
from .CNNModel import CNNModel
# CNNModel 定义在 src/model/CNNModel.py 中，包含卷积神经网络模型的结构定义。

# 模块回顾:
# 1. CNNModel 模块:
# - CNNModel: 卷积神经网络模型类，包含模型的结构定义。定义在 src/model/CNNModel.py 中。
#   - 功能: 该类定义了卷积神经网络的层结构和前向传播过程。包括卷积层、池化层和全连接层等。
#   - 主要步骤:
#     1. 在 __init__ 方法中定义模型的各层，包括两个卷积层、池化层和两个全连接层。
#     2. 在 forward 方法中定义前向传播过程，依次通过卷积层、激活函数、池化层、展平层和全连接层。
#   - 使用: 在主函数中实例化该模型类，并在训练和预测时使用。

# 详细注释解释:
# - __init__ 方法: 初始化模型的各层结构。包括:
#   - 卷积层 (conv1, conv2): 用于提取图像特征。
#   - 池化层 (pool): 用于下采样，减少特征图的尺寸。
#   - 全连接层 (fc1, fc2): 用于将特征图展平并进行分类。

# - forward 方法: 定义模型的前向传播过程。包括:
#   - 依次通过卷积层、激活函数 (ReLU)、池化层。
#   - 将池化后的特征图展平为一维向量。
#   - 通过全连接层进行分类，最终输出预测结果。

# - 该类在训练和预测过程中被实例化:
#   - 在训练过程中，通过定义优化器和损失函数，对模型进行训练和参数更新。
#   - 在预测过程中，将模型设置为评估模式，使用测试数据进行预测。

# 这些详细注释有助于理解模型的定义和使用，便于后续的维护和扩展。
