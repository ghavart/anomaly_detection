import os
import torch
from PIL import Image

from torchvision import transforms


class ImageDataset(torch.utils.data.Dataset):

    def __init__(self, image_dir, out_size=128, transform=None): 
        super(ImageDataset, self).__init__()
        self.image_dir = image_dir
        self.out_size = out_size
        self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.jpg')])
        self.transform = transform

    def __len__(self):
        return len(self.image_files) 

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)

        img = Image.open(img_path)
        if self.transform:
            img = self.transform(img)

        return img 
    

data_transforms = transforms.Compose([
    transforms.RandomCrop(128),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229,0.224,0.225])
])