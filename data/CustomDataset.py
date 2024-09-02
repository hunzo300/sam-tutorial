import os

from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class CustomDataset(Dataset):
    def __init__(self, root_dir, mode, transform=None):
        self.image_dir = os.path.join(root_dir, "images")
        self.mask_dir = os.path.join(root_dir, "masks")
        self.images = []
        self.masks = []
        self.transform = transform

        for img_name in os.listdir(self.image_dir):
            img_path = os.path.join(self.image_dir, img_name)
            self.images.append(img_path)
            
            mask_path = os.path.join(self.mask_dir, mask_name)
            self.masks.append(mask_path)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert("RGB")
        
        mask_path = self.masks[idx]
        mask = Image.open(mask_path).convert("L")

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask
