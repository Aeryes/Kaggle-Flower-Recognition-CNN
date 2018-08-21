import torch
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from PIL import Image
from cnn_main import CNNet
from pathlib import Path

model = CNNet(5)
checkpoint = torch.load(Path('C:/Users/Aeryes/PycharmProjects/simplecnn/src/19.model'))
model.load_state_dict(checkpoint)
trans = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.Resize(32),
    transforms.CenterCrop(32),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))
    ])

model.eval()

image = Image.open(Path('C:/Users/Aeryes/PycharmProjects/simplecnn/images/pretrain_classify/tulip_classify.jpg'))
input_loader = DataLoader(dataset=image, batch_size=1, shuffle=False, num_workers=5)
input_loader = trans(image)
input_loader = input_loader.view(1, 3, 32,32)

output = model(input_loader)

prediction = output.data.numpy().argmax()
print(prediction)

if (prediction == 0):
    print ('daisy')
if (prediction == 1):
    print ('dandelion')
if (prediction == 2):
    print ('rose')
if (prediction == 3):
    print ('sunflower')
if (prediction == 4):
    print ('tulip')