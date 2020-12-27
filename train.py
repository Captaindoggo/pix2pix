import torch
from os import listdir
from os.path import isfile, join
import subprocess

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, datasets
from model import Pix2Pix
from dataset import mod_dataset
import random
import numpy as np

def set_seed(n):
    torch.backends.cudnn.deterministic = True
    torch.manual_seed = n
    random.seed(n)
    np.random.seed(n)

if __name__ == '__main__':
    set_seed(42)
    DATA = 'facades'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    lr = 3e-4

    model = Pix2Pix(device, lmbda=100).to(device)

    batch_size = 32

    # Load dataset

    if DATA == 'facades':
        bashCommand = 'wget http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/facades.tar.gz'
    elif DATA == 'maps':
        bashCommand = 'wget http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/maps.tar.gz'

    process1 = subprocess.Popen(bashCommand.split())

    process1.wait()

    if DATA == 'facades':
        bashCommand = 'tar -xzf facades.tar.gz'
    elif DATA == 'maps':
        bashCommand = 'tar -xzf maps.tar.gz'

    process2 = subprocess.Popen(bashCommand.split())

    process2.wait()

    # Creat dataloaders

    train_path = DATA + '/train'
    data_train = [f for f in listdir(train_path) if isfile(join(train_path, f))]

    train_dataset = mod_dataset(data_train, train_path, transform=transforms.Compose(
        [transforms.Resize((286, 286)), transforms.RandomCrop((256, 256)), transforms.RandomHorizontalFlip()]))

    val_path = DATA + '/val'
    data_val = [f for f in listdir(val_path) if isfile(join(val_path, f))]

    val_dataset = mod_dataset(data_val, val_path)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=3, shuffle=True)

    if DATA == 'facades':
        epochs = 800
    else:
        epochs = 100

    for epoch in range(epochs):
        running_loss_d = 0.0
        running_loss_g = 0.0
        val_loss = 0.0

        ctr = 0
        val_ctr = 0

        model.train()
        for batch in train_loader:
            A, B = batch
            A = A.to(device)
            B = B.to(device)
            gloss, dloss = model.step(A, B)
            running_loss_g += gloss
            running_loss_d += dloss
            ctr += 1
        # print('epoch', epoch+1, ':', 'Generator loss:', running_loss_g/ctr, 'Discriminator loss:', running_loss_d/ctr)