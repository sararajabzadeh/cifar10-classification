import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from torch.optim import lr_scheduler
from torchvision import transforms
from PIL import Image
import torchvision
# from torch.utils.tensorboard import SummaryWriter
# writer = SummaryWriter()

classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


def get_lr(optimizer):
    for p in optimizer.param_groups:
        return p["lr"]


def train(model, config, trainLoader, validLoader, device):

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=config['lr'], momentum=0.9)
    m1 = int(0.4 * config['epochs'])
    m2 = int(0.8 * config['epochs'])
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[m1, m2], gamma=0.1)

    train_loss = []
    validation_loss = []
    best_loss = np.inf

    for epoch in range(config['epochs']):

        print(f"Start the training of epoch {epoch + 1}")

        training_loss = 0.0
        valid_loss = 0.0

        sum_labels = torch.tensor([]).to(device)
        sum_preds = torch.tensor([]).to(device)
        val_labels = torch.tensor([]).to(device)
        val_preds = torch.tensor([]).to(device)

        model.train()
        for batch, data in enumerate(tqdm(trainLoader)):
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            # writer.add_scalar("Loss/train", loss, epoch)
            loss.backward()
            optimizer.step()
            preds = outputs.argmax(1)
            sum_labels = torch.cat([sum_labels, labels], dim=0)
            sum_preds = torch.cat([sum_preds, preds], dim=0)

            training_loss += loss.item()

            # writer.add_scalar("learning rate", get_lr(optimizer), epoch)
        scheduler.step()

        corr = sum_preds.eq(sum_labels).sum().item()
        # writer.add_scalar("Accuracy/train", corr/40000, epoch)
        train_loss.append(training_loss / len(trainLoader))
        print()
        print(f"training {epoch + 1} accuracy = {corr / 40000} loss = {training_loss / len(trainLoader)}")
        print()
        print(f"Start the validation of epoch {epoch + 1}")

        model.eval()
        with torch.no_grad():
            for batch, data in enumerate(tqdm(validLoader)):
                images, labels = data
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)
                # writer.add_scalar("Loss/valid", loss, epoch)
                preds = outputs.argmax(1)
                val_labels = torch.cat([val_labels, labels], dim=0)
                val_preds = torch.cat([val_preds, preds], dim=0)

                valid_loss += loss.item()

        corr = val_preds.eq(val_labels).sum().item()
        # writer.add_scalar("Accuracy/valid", corr/10000, epoch)
        validation_loss.append(valid_loss / len(validLoader))
        print(f"validation {epoch + 1} accuracy = {corr / 10000} loss = {valid_loss / len(validLoader)}")

        if best_loss > valid_loss / len(validLoader):
            best_loss = valid_loss / len(validLoader)
            torch.save(model.state_dict(), 'best_model.pt')
            print(f"best loss is {best_loss}")

        print("-----------------------------------------")
        print()

    # writer.flush()


def test(model, testLoader, device):

    model.eval()
    total = 0
    corr = 0
    with torch.no_grad():
        for _, data in enumerate(tqdm(testLoader)):
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(1)
            total += labels.size(0)
            corr += preds.eq(labels).sum().item()

    print(f"The accuracy is {corr / total}")


def inference(model, img_path, device):

    with torch.no_grad():
        image = Image.open(img_path)
        torchvision.transforms.Resize(32 * 32)(image)
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        img_tensor = transform(image).float()
        model.eval()
        img_tensor = img_tensor.to(device)
        img_tensor = img_tensor.reshape(1, 3, 32, 32)
        output = model(img_tensor)
        m = nn.Softmax(dim=1)
        out = m(output)
        prob = out / torch.sum(out, dim=-1).unsqueeze(-1)
        prob = torch.mul(prob, 100)
        prob = prob[0].tolist()
    for i in range(len(classes)):
        print(f"The {classes[i]} probability is {round(prob[i], 2)}%.")
    print("------------------------------------")
    print(f"The winner is the {classes[prob.index(max(prob))]} class!")
