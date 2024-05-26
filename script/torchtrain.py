import torch
import torch.optim as optim
import torch.nn as nn
import wandb
import os
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from data.dataset import ArmorDataset
from networks.resnet import resnet18
from tqdm import tqdm

def train(model, train_loader, criterion, optimizer, epoch, device):
    model.train()
    train_loss = 0.0
    correct = 0
    total = 0
    
    for i, (images, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
            
    train_acc = 100. * correct / total
    train_loss /= len(train_loader)
            
    wandb.log({
        "train_loss": train_loss,
        "train_accuracy": train_acc
    })
    
    print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            

def test(model, test_loader, criterion, epoch, device):
    model.eval()
    correct = 0
    total = 0
    test_loss = 0.0
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_acc = 100. * correct / total
    test_loss /= len(test_loader)

    wandb.log({
        "test_loss": test_loss,
        "test_accuracy": test_acc
    })

    print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')
    return test_acc, test_loss


def save_model(model, save_path, epoch, best_acc):
    """保存模型"""
    os.makedirs(save_path, exist_ok=True)
    model_path = os.path.join(save_path, f'model_epoch_{epoch+1}_acc_{best_acc:.2f}.pth')
    torch.save(model.state_dict(), model_path)
    print(f'Model saved to {model_path}')

def main():
    lr = 0.001
    num_epochs = 10
    batch_size = 200
    save_path = './models'
    
    
    # 初始化 wandb
    wandb.init(project="armor-resnet18-v1")
    wandb.config.update({
        "num_epochs": num_epochs,
        "learning_rate": lr,
        "batch_size": batch_size
    })
    
    # 数据转换
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 创建数据集和数据加载器
    train_dataset = ArmorDataset(txt_file='./datasets/TIT_armor/train.txt', transform=transform)
    test_dataset = ArmorDataset(txt_file='./datasets/TIT_armor/test.txt', transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 定义模型
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = resnet18(num_classes=8)
    model = model.to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_acc = 0.0
    for epoch in range(num_epochs):
        train(model, train_loader, criterion, optimizer, epoch, device)
        accuracy, test_loss = test(model, test_loader, criterion, epoch, device)
        
        # 如果当前模型的准确率高于之前最好的模型,则保存当前模型
        if accuracy > best_acc:
            best_acc = accuracy
            save_model(model, save_path, epoch, best_acc)
            
    wandb.finish()

if __name__ == '__main__':
    main()
