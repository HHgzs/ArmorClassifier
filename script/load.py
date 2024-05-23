import os
import numpy as np
from PIL import Image

def load_binary_dataset(data_dir, img_size=(20, 28)):
    """
    读取指定目录下的图像数据集,并将其转换为适合机器学习模型训练的格式。
    
    参数:
    data_dir (str): 包含图像文件夹的数据集目录路径。
    img_size (tuple): 图像大小,默认为(20, 28)。
    test_size (float): 测试集占总数据集的比例,默认为0.2。
    random_state (int): 随机种子,用于保证数据集划分的可重复性,默认为42。
    
    返回:
    (X_train, y_train, X_test, y_test): 训练集图像、训练集标签、测试集图像、测试集标签。
    """
    # 获取所有文件夹名称
    folder_names = [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]

    # 初始化数据和标签列表
    X = []
    y = []

    # 遍历每个文件夹
    for folder_name in folder_names:
        folder_path = os.path.join(data_dir, folder_name)
        
        # 遍历文件夹内的图片
        for filename in os.listdir(folder_path):
            if filename.endswith('.jpg') or filename.endswith('.png'):
                img_path = os.path.join(folder_path, filename)
                
                # 读取图片并调整大小
                image = Image.open(img_path).convert('L')
                image = image.resize(img_size, resample=Image.BICUBIC)
                
                # 将图像转换为numpy数组,并展平为一维数组
                img_data = np.array(image).flatten() / 255.0
                
                # 将图像和标签添加到列表中
                X.append(img_data)
                y.append(folder_name)

    # 将数据转换为numpy数组
    X = np.array(X)
    y = np.array(y)

    return X, y


def load_color_dataset(data_dir, img_size=(20, 28)):
    """
    读取指定目录下的图像数据集,并将其转换为适合机器学习模型训练的格式。
    
    参数:
    data_dir (str): 包含图像文件夹的数据集目录路径。
    img_size (tuple): 图像大小,默认为(20, 28)。
    test_size (float): 测试集占总数据集的比例,默认为0.2。
    random_state (int): 随机种子,用于保证数据集划分的可重复性,默认为42。
    
    返回:
    (X_train, y_train, X_test, y_test): 训练集图像、训练集标签、测试集图像、测试集标签。
    """
    # 获取所有文件夹名称
    folder_names = [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]

    # 初始化数据和标签列表
    X = []
    y = []

    # 遍历每个文件夹
    for folder_name in folder_names:
        folder_path = os.path.join(data_dir, folder_name)
        
        # 遍历文件夹内的图片
        for filename in os.listdir(folder_path):
            if filename.endswith('.jpg') or filename.endswith('.png'):
                img_path = os.path.join(folder_path, filename)
                
                # 读取图片并调整大小
                image = Image.open(img_path).convert('RGB')
                image = image.resize(img_size, resample=Image.BICUBIC)
                
                # 将图像转换为numpy数组,并展平为一维数组
                img_data = np.array(image).flatten() / 255.0
                
                # 将图像和标签添加到列表中
                X.append(img_data)
                y.append(folder_name)

    # 将数据转换为numpy数组
    X = np.array(X)
    y = np.array(y)

    return X, y


if __name__ == '__main__':
    # 测试数据集路径
    data_dir = '/home/nvidia/Code/ArmorClassifier/datasets/binary/train'
    
    # 加载数据集
    X, y = load_binary_dataset(data_dir)
    
    # 打印数据集信息
    print(f'图像数量: {len(X)}')
    print(f'标签数量: {len(y)}')
    print('数据集加载成功!')

