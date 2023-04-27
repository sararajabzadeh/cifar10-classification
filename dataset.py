import torch
from torchvision import transforms
import torchvision.datasets as datasets

trainTransform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

testTransform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


trainData = datasets.CIFAR10(root='./data', train=True, download=True, transform=trainTransform)
testData = datasets.CIFAR10(root='./data', train=False, download=True, transform=testTransform)

trainD, validD = torch.utils.data.random_split(trainData, [40000, 10000])
