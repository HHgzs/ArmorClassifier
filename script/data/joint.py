import os
from PIL import Image
import random
from tqdm import tqdm


def create_image(image_folder, save_path, output_size, num_rows, num_cols):
    # 初始化一个列表来存储图片
    images = []

    # 遍历图片文件夹,随机选择图片
    image_files = os.listdir(image_folder)
    selected_images = random.sample(image_files, num_rows * num_cols)

    for image_file in tqdm(selected_images, desc='Loading images'):
        image_path = os.path.join(image_folder, image_file)
        image = Image.open(image_path)
        
        # 将图片调整为正方形
        image = image.resize(output_size, resample=Image.BICUBIC)
        images.append(image)

    # 创建一个大图片
    big_image = Image.new('RGB', (num_cols * output_size[0], num_rows * output_size[1]))

    # 将图片拼接到大图片上
    for i in tqdm(range(num_rows), desc='Stitching images'):
        for j in range(num_cols):
            index = i * num_cols + j
            big_image.paste(images[index], (j * output_size[0], i * output_size[1]))

    # 保存大图片
    big_image.save(save_path)
    print('Output image saved: ' + save_path)
    

if __name__ == '__main__':

    # 指定输出图像的大小
    output_size = (20, 28)
    num_rows = 10
    num_cols = 14
    
    
    for i in range(0, 8):
        
        # output_size = (50, 50)
        # num_rows = 10
        # num_cols = 10
        # image_folder = './datasets/TIT_armor/' + str(i)
        # save_path = './doc/color' + str(i) + '.jpg'
        
        
        # output_size = (20, 28)
        # num_rows = 15
        # num_cols = 21
        # image_folder = './datasets/binary_armor/train/' + str(i)
        # save_path = './doc/binary' + str(i) + '.jpg'
        
        output_size = (128, 72)
        num_rows = 9
        num_cols = 16
        image_folder = './datasets/binary_armor/train/' + str(i)
        save_path = './doc/binary' + str(i) + '.jpg'
        
        
        create_image(image_folder, save_path, output_size, num_rows, num_cols)
        
        
