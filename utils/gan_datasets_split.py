import os
import shutil
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# 设置参数
src_dir = '/mnt/d/Project/ArmorClassifier/datasets/TIT_armor'
dst_dir = '/mnt/d/Project/ArmorClassifier/datasets/gan_armor'
train_name = 'trainA'
test_name = 'valA'
file_suffix = '.jpg'
test_size = 0.3

# 创建目标文件夹
if not os.path.exists(dst_dir):
    os.makedirs(dst_dir)
    os.makedirs(os.path.join(dst_dir, train_name))
    os.makedirs(os.path.join(dst_dir, test_name))

# 递归查找文件
file_paths = []
for root, dirs, files in os.walk(src_dir):
    for file in files:
        if file.endswith(file_suffix):
            file_paths.append(os.path.join(root, file))

# 进行 train test split
X_train, X_test = train_test_split(file_paths, test_size=test_size, random_state=42)

# 拷贝文件到目标文件夹
for src_path in tqdm(X_train):
    dst_path = os.path.join(dst_dir, train_name, os.path.basename(src_path))
    shutil.copy2(src_path, dst_path)

for src_path in tqdm(X_test):
    dst_path = os.path.join(dst_dir, test_name, os.path.basename(src_path))
    shutil.copy2(src_path, dst_path)

print('Files have been successfully copied to the destination directory.')