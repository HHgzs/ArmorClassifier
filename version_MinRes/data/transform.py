import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import random

# 定义一个复杂的增强变换
class ComplexTransform:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.RandomApply([
                transforms.RandomAffine(degrees=30, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=10),
            ], p=0.5),
            transforms.RandomApply([
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            ], p=0.5),
            transforms.RandomApply([
                transforms.RandomRotation(degrees=45),
            ], p=0.5),
            transforms.RandomApply([
                transforms.RandomPerspective(distortion_scale=0.5, p=1.0),
            ], p=0.5),
            transforms.RandomApply([
                transforms.RandomResizedCrop(size=(256, 256), scale=(0.8, 1.5)),
            ], p=0.5),
            transforms.ToTensor(),
        ])

    def __call__(self, img):
        return self.transform(img)


# 在竖直方向上随机拉长图像
class StretchTransform:
    def __init__(self, p=0.5, ratio=3):
        self.p = p
        self.ratio = ratio

    def __call__(self, img):
        w, h = img.size  # 获取图像的宽度和高度

        if random.random() < self.p:
            new_h = int(random.uniform(h, self.ratio * h))
            new_img = img.resize((w, new_h), Image.BILINEAR)

            top = (new_h - h) // 2
            bottom = new_h - top
            left = 0
            right = w
            cropped_img = new_img.crop((left, top, right, bottom))
            
            return cropped_img

        return img
 


class CombinedTransform:
    def __init__(self):
        # 初始化 ComplexTransform
        self.complex_transform = ComplexTransform()
        # 初始化 StretchTransform
        self.stretch_transform = StretchTransform()

    def __call__(self, img):
        
        # 首先应用 ComplexTransform
        transformed_img = self.complex_transform(img)
        
        # 然后应用 StretchTransform
        final_img = self.stretch_transform(transformed_img)
        return final_img
    
    
    
# 测试 CombinedTransform 并显示10个随机变换后的图片
def test_transform(show_n=10):
    combined_transform = CombinedTransform()
    img = Image.open('/mnt/d/Project/ArmorClassifier/datasets/TIT_yolo/images/044414.jpg')

    # 显示原始图像
    plt.figure(figsize=(10, 10))
    plt.subplot(2, show_n + 1, 1)
    plt.imshow(img)
    plt.title('Original Image')
    plt.axis('off')

    # 应用变换并显示结果
    for i in range(show_n):
        plt.subplot(2, show_n + 1, i + 2)
        transformed_img = combined_transform(img)
        plt.imshow(transformed_img.permute(1, 2, 0))  # 转换通道顺序以适应matplotlib
        plt.title('Transformed Image {}'.format(i + 1))
        plt.axis('off')

    plt.show()

# 调用测试函数
test_transform(50)