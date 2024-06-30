import os
import cv2
import numpy as np
from tqdm import tqdm


# 将目标检测装甲板数据集转换为纯分类数据集

path = "/home/nvidia/Documents/TIT_yolo/labels"
save_path = "/home/nvidia/Documents/TIT_armor"
class_map = [0, 1, 2, 3, 4, 5, 6, 7, 7, 0, 1, 2, 3, 4, 5, 6, 7, 7, 0, 1, 2, 3, 4, 5, 6, 7, 7, 0, 1, 2, 3, 4, 5, 6, 7, 7]

# image size
imgsz_width = 224
imgsz_height = 224

count = 100000
 
 
# 读取每一个label文件，保证读取文件顺序从小到大
for root, dirs, files in os.walk(path):
    dirs.sort()
    files.sort()
    for file in tqdm(files, desc="Processing files"):
        if file.endswith(".txt"):
            
            # 每一行的label定义为: 类别 x1 y1 x2 y2 x3 y3 x4 y4
            with open(os.path.join(root, file), 'r') as f:
                lines = f.readlines()
                for line in lines:
                    line = line.strip().split(" ")
                    line = [float(i) for i in line]
                    label_class = int(line[0])
                    armor_class = class_map[label_class]
 
                    # 读取对应的image文件
                    img_path = os.path.join(root.replace('labels', 'images'), file.replace('txt', 'jpg'))
                    
                    # 如果文件空，直接跳过
                    if not os.path.exists(img_path):
                        continue
                    img = cv2.imread(img_path)
                    
                    # 读取坐标
                    x1, y1, x2, y2, x3, y3, x4, y4 = line[1:]
                    
                    # 将归一化的坐标转换为真实坐标
                    x1, y1, x2, y2, x3, y3, x4, y4 = x1 * img.shape[1], y1 * img.shape[0], x2 * img.shape[1], y2 * img.shape[0], x3 * img.shape[1], y3 * img.shape[0], x4 * img.shape[1], y4 * img.shape[0]
                    
                    # 求出外接矩形
                    x_min = min(x1, x2, x3, x4)
                    x_max = max(x1, x2, x3, x4)
                    y_min = min(y1, y2, y3, y4)
                    y_max = max(y1, y2, y3, y4)
                    
                    # y方向扩大 1/3，x方向扩大 1/3 倍
                    x_min = max(0, x_min - (x_max - x_min) / 6)
                    x_max = min(img.shape[1], x_max + (x_max - x_min) / 6)
                    y_min = max(0, y_min - (y_max - y_min) / 6)
                    y_max = min(img.shape[0], y_max + (y_max - y_min) / 6)
 
                    # 将图片根据对应的label存入对应的文件夹
                    save_dir = os.path.join(save_path, str(armor_class))
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    img_save_path = os.path.join(save_dir, str(count) + ".jpg")
                    count += 1
                    
                    
                    # 将图片resize为指定大小
                    y_len = y_max - y_min
                    x_len = x_max - x_min
                    max_len = max(y_len, x_len)
                    center_y = (y_min + y_max) / 2
                    center_x = (x_min + x_max) / 2
                    
                    left = max(min(center_x - max_len / 2, img.shape[1] - 1), 0)
                    right = max(min(center_x + max_len / 2, img.shape[1] - 1), 0)
                    top = max(min(center_y - max_len / 2, img.shape[0] - 1), 0)
                    bottom = max(min(center_y + max_len / 2, img.shape[0] - 1), 0)
                    
                    box = img[int(top):int(bottom), int(left):int(right)]
                    box = cv2.resize(box, (imgsz_width, imgsz_height))
                    cv2.imwrite(img_save_path, box)
 
      
print("all done!")
 