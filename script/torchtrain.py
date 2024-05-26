import torch
import torch.optim as optim
import torch.nn as nn
import tqdm as tqdm
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from data.dataset import ArmorDataset
from models.resnet import resnet18

def train(model, train_loader, criterion, optimizer, epoch, device):
    model.train()
    running_loss = 0.0
    for i, (images, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if (i + 1) % 100 == 0:
            print(f'Step [{i + 1}/{len(train_loader)}], Loss: {running_loss / 100:.4f}')
            running_loss = 0.0
            
import torch
from tqdm import tqdm

def test(model, test_loader, criterion, device):
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

    accuracy = 100 * correct / total
    test_loss = test_loss / len(test_loader)
    print(f'Accuracy of the model on the test images: {accuracy}%')
    print(f'Test Loss: {test_loss:.4f}')
    return accuracy, test_loss



def main():
    # 数据转换
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 创建数据集和数据加载器
    train_dataset = ArmorDataset(txt_file='train.txt', transform=transform)
    test_dataset = ArmorDataset(txt_file='test.txt', transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # 定义模型
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = resnet18(num_classes=8)
    model = model.to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 主训练和测试循环
    num_epochs = 10
    for epoch in range(num_epochs):
        train(model, train_loader, criterion, optimizer, epoch, device)
        test(model, test_loader, criterion, device)

if __name__ == '__main__':
    main()
