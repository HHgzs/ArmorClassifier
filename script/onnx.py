import torch
import torch.onnx
from networks.resnet import resnet18
import onnx

def convert_pth_to_onnx(pth_path, onnx_path, num_classes=8):
    # 初始化模型
    model = resnet18(num_classes=num_classes)
    
    # 加载模型权重
    model.load_state_dict(torch.load(pth_path, map_location='cpu'))
    model.eval()
    
    # 创建一个示例输入
    dummy_input = torch.randn(1, 3, 224, 224)  # 这里假设输入是224x224的RGB图像
    
    # 打印一些调试信息
    print(f"Model: {model}")
    print(f"Dummy Input: {dummy_input.shape}")

    # 将模型导出为ONNX格式
    try:
        torch.onnx.export(model, dummy_input, onnx_path, 
                          export_params=True, 
                          opset_version=12, 
                          do_constant_folding=True, 
                          input_names=['input'], 
                          output_names=['output'])
        print(f'Model has been converted to ONNX and saved at {onnx_path}')
    except Exception as e:
        print(f"Error occurred while converting to ONNX: {e}")

if __name__ == '__main__':
    pth_path = './models/model_epoch_8_acc_99.03.pth'  # 这里替换为你实际的模型路径
    onnx_path = './models/model.onnx'  # 这里替换为你想要保存ONNX模型的路径
    convert_pth_to_onnx(pth_path, onnx_path)
