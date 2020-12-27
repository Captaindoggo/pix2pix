from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, datasets
from PIL import Image

import os
import torch


class mod_dataset(Dataset):

    def __init__(self, file_names, root_dir, norm=True, transform=None):
        self.file_names = file_names
        self.root_dir = root_dir
        self.transform = transform
        if norm:
            self.norm = True
            self.normalizer = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.file_names[idx])
        image = Image.open(img_name)

        to_tensor = transforms.ToTensor()
        image = to_tensor(image)

        w = image.shape[2]

        B = image[:, :, :w // 2]
        A = image[:, :, w // 2:]
        if self.norm:
            A = self.normalizer(A)
            B = self.normalizer(B)

        if self.transform:               # transforms are applied to concatenated (on color channels dimension) image AB in order to achieve
            AB = self.transform(AB)      # same random transformations for both A and B images
            A = AB[:3, :]
            B = AB[3:, :]

        sample = [A, B]

        return sample