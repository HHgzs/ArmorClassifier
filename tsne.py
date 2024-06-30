import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from networks.resnet import resnet18
from data.dataset import ArmorDataset
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm

def extract_features(model, data_loader, device):
    model.eval()
    features = []
    labels_list = []
    
    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc="Extracting Features"):
            images = images.to(device)
            outputs = model(images)
            features.append(outputs.cpu().numpy())
            labels_list.append(labels.cpu().numpy())

    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    return features, labels

def plot_tsne(features, labels, epoch, save_path):
    os.makedirs(save_path, exist_ok=True)
    tsne = TSNE(n_components=2, random_state=0)
    features_2d = tsne.fit_transform(features)
    
    plt.figure(figsize=(10, 10))
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=labels, cmap='tab10')
    plt.legend(*scatter.legend_elements(), title="Classes")
    plt.title(f't-SNE Visualization - Epoch {epoch+1}')
    plt.savefig(os.path.join(save_path, f'tsne_epoch_{epoch+1}.png'))
    plt.close()

def main():
    model_path = './models/model_epoch_8_acc_99.03.pth'
    save_path = './doc/tsne'
    batch_size = 200
    epoch = 9
    
    # 数据转换
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 创建数据集和数据加载器
    test_dataset = ArmorDataset(txt_file='./datasets/TIT_armor/test.txt', transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 定义模型
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = resnet18(num_classes=8)
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)

    # 提取特征并可视化 t-SNE
    features, labels = extract_features(model, test_loader, device)
    plot_tsne(features, labels, epoch, save_path)
    
if __name__ == '__main__':
    main()
