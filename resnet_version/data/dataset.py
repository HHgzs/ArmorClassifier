from PIL import Image
from torch.utils.data import Dataset

class ArmorDataset(Dataset):
    def __init__(self, txt_file, transform=None):
        self.transform = transform
        self.samples = []

        with open(txt_file, 'r') as f:
            for line in f:
                image_path, label = line.strip().split()
                self.samples.append((image_path, int(label)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, label = self.samples[idx]
        image = Image.open(image_path)

        if self.transform:
            image = self.transform(image)

        return image, label
