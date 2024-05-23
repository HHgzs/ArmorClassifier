import os
import cv2
import numpy as np


# 将目标检测装甲板数据集转换为纯分类数据集

path = "/openbayes/input/input1/armor_finnal/labels"
save_path = "/openbayes/input/input0/dark/"
 

# image size
imgsz_width = 224
imgsz_height = 224

count = 100000
 
 
# 读取每一个label文件，保证读取文件顺序从小到大
for root, dirs, files in os.walk(path):
    dirs.sort()
    files.sort()
    for file in files:
        if file.endswith(".txt"):
            
            # 每一行的label定义为: 类别 x1 y1 x2 y2 x3 y3 x4 y4
            with open(os.path.join(root, file), 'r') as f:
                lines = f.readlines()
                for line in lines:
                    line = line.strip().split(" ")
                    line = [float(i) for i in line]
                    label_class = int(line[0])
 
                    # 读取对应的image文件
                    img_path = os.path.join(root.replace('labels', 'images_dark_8'), file.replace('txt', 'jpg'))
                    
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
                    
                    # y方向扩大两倍，x方向扩大1.5倍
                    x_min = max(0, x_min - (x_max - x_min) / 2)
                    x_max = min(img.shape[1], x_max + (x_max - x_min) / 2)
                    y_min = max(0, y_min - (y_max - y_min))
                    y_max = min(img.shape[0], y_max + (y_max - y_min))
 
                    # 将图片根据对应的label存入对应的文件夹
                    save_dir = os.path.join(save_path, str(label_class))
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    img_save_path = os.path.join(save_dir, str(count) + ".jpg")
                    count += 1
                    
                    
                    # 将图片resize为指定大小
                    box = img[int(y_min):int(y_max), int(x_min):int(x_max)]
                    box = cv2.resize(box, (imgsz_width, imgsz_height))
                    cv2.imwrite(img_save_path, box)
                    print(img_save_path)
 
      
print("all done!")
 