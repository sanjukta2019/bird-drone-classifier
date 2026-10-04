import os
import torch
from torch.utils.data import Dataset
from PIL import Image

class BirdDroneDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        # Filter out images with missing or empty label files
        self.images = [
            img for img in sorted(os.listdir(image_dir))
            if os.path.exists(os.path.join(label_dir, img.replace('.jpg', '.txt')))
            and os.path.getsize(os.path.join(label_dir, img.replace('.jpg', '.txt'))) > 0
        ]

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        label_path = os.path.join(self.label_dir, img_name.replace('.jpg', '.txt'))

        image = Image.open(img_path).convert("RGB")
        with open(label_path) as f:
            first_line = f.readline().strip()
            if first_line == '':
                raise ValueError(f"Empty label file for image: {img_name}")
            else:
                label = int(first_line.split()[0])

        if self.transform:
            image = self.transform(image)
        return image, label
