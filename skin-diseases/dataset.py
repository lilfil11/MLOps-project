import torch
from torch.utils.data import Dataset
from torchvision.io import read_image


class MedicalImageDataset(Dataset):
    def __init__(self, img_dir, data='train', transform=None):
        super().__init__()
        self.img_dir = img_dir,
        self.data = data
        self.labels = torch.load(f'{img_dir}/{data}/labels.pt')
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = read_image(f'{self.img_dir[0]}/{self.data}/images/({idx})-img.jpg') / 255
        if self.transform:
            image = self.transform(image)

        return (image, self.labels[idx])
