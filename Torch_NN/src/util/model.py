import torch.nn as nn

def summarize_model(model):
    """
    输出模型的详细信息，包括每一层的名称、参数的数量、参数的形状以及总参数数量

    参数:
    model (nn.Module): 需要总结的模型
    """
    print("Model Summary")
    print("=" * 100)
    print("{:<30} {:<30} {:<30}".format("Layer (type)", "Output Shape", "Param #"))
    print("=" * 100)

    total_params = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            layer_param_count = param.numel()
            total_params += layer_param_count
            print("{:<30} {:<30} {:<30}".format(name, str(param.shape), layer_param_count))

    print("=" * 100)
    print("Total parameters:", total_params)
    print("=" * 100)
