from multiprocessing import freeze_support

import torch
from torch import nn
import torchvision
from torch.autograd import Variable
from torch.utils.data import DataLoader, Sampler
from torchvision import datasets
from torchvision.transforms import transforms
from torch.optim import Adam

import matplotlib.pyplot as plt
import numpy as np
import PIL
from pathlib import Path

num_classes = 5
batch_size = 100
num_of_workers = 5

DATA_PATH_TRAIN = Path('C:/Users/Aeryes/PycharmProjects/simplecnn/images/train')
DATA_PATH_TEST = Path('C:/Users/Aeryes/PycharmProjects/simplecnn/images/test')

trans = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.Resize(32),
    transforms.CenterCrop(32),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))
    ])

train_dataset = datasets.ImageFolder(root=DATA_PATH_TRAIN, transform=trans)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_of_workers)

def imshow(img):
    img = img / 2 + 0.5 # unnormalize
    #npimg = img.numpy()
    plt.imshow(np.transpose(img[0].numpy(), (1, 2, 0)))
    plt.show()

def main():
    # get some random training images
    dataiter = iter(train_loader)
    images, labels = dataiter.next()

    # show images
    imshow(images)

if __name__ == "__main__":
    main()