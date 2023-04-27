from train_test import train, test, inference
import argparse
import torch
import torchvision.models as models
from model import Resnet18
from torch.utils.data import DataLoader
from dataset import trainD, validD, testData

parser = argparse.ArgumentParser(description='Arguments for running')
parser.add_argument("--epochs", type=int, default=25)
parser.add_argument('--mode', type=str, default='train', choices=['train', 'test', 'inference'], help='Choose to train or test or inference')

args = parser.parse_args()
print(args)

config = {
    'epochs': 25,
    'batch_size': 32,
    'lr': 0.03
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

config['epochs'] = args.epochs

# Load the pre-trained ResNet18 model
resnet18 = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model = Resnet18(3, 10)
model.conv1.weight.data = resnet18.conv1.weight.data
model.to(device)

if args.mode == 'train':
    trainLoader = DataLoader(trainD, batch_size=config['batch_size'], shuffle=True)
    validLoader = DataLoader(validD, batch_size=config['batch_size'], shuffle=False)

    train(model, config, trainLoader, validLoader, device)

elif args.mode == 'test':
    testLoader = DataLoader(testData, batch_size=config['batch_size'], shuffle=False)

    model.load_state_dict(torch.load('best_model.pt'))
    test(model, testLoader, device)

elif args.mode == 'inference':
    img_path = input("Please indicate your image path:\n")
    model.load_state_dict(torch.load('best_model.pt'))
    inference(model, img_path, device)
