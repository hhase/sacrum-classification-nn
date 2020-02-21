import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import PIL.Image as Image

import matplotlib.pyplot as plt
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import WeightedRandomSampler
from polyaxon_client.tracking import Experiment, get_data_paths, get_outputs_path

# TF compat export: https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html

# PARAMETERS

IMG_HEIGHT  = 272
IMG_WIDTH   = 258
train_on_gpu = torch.cuda.is_available()
print("Cuda is available ->", train_on_gpu)

cluster = True if os.getenv('POLYAXON_RUN_OUTPUTS_PATH') else False

if cluster:
    data_paths = get_data_paths()
    train_data_path = data_paths['data1'] + "/HHase_Robotic_RL/Sacrum_Classification/testing/"
    val_data_path = data_paths['data1'] + "/HHase_Robotic_RL/Sacrum_Classification/validation/"
    output_path = get_outputs_path()
    BATCH_SIZE = 128
else:
    train_data_path = "./Data/training/"
    val_data_path = "./Data/validation/"
    output_path = "./logs/"
    BATCH_SIZE = 16

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        img = img.resize((IMG_WIDTH, IMG_HEIGHT), Image.ANTIALIAS)
        return img.convert('RGB')


# LOAD TRAINING DATA

transform = transforms.Compose([
    transforms.RandomResizedCrop((IMG_WIDTH, IMG_HEIGHT), scale=(0.08, 1.0), ratio=(0.75, 1.3333333333333333), interpolation=2),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(5),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor()
    ])

train_data = datasets.DatasetFolder(train_data_path, loader=pil_loader, extensions='.png', transform=transform)

classes, class_sample_counts = np.unique(np.asarray(train_data.targets), return_counts=True)

class_weights = 1./torch.tensor(class_sample_counts, dtype=torch.float)
samples_weights = class_weights[train_data.targets]

sampler = torch.utils.data.sampler.WeightedRandomSampler(samples_weights, num_samples=len(train_data), replacement=True)

train_data_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, sampler=sampler, pin_memory=train_on_gpu, num_workers=4)

# LOAD VALIDATION DATA

val_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor()
    ])

validation_data = datasets.DatasetFolder(val_data_path, loader=pil_loader, extensions='.png', transform=val_transform)

classes, class_sample_counts = np.unique(np.asarray(validation_data.targets), return_counts=True)
class_weights = 1./torch.tensor(class_sample_counts, dtype=torch.float)
samples_weights = class_weights[validation_data.targets]
validation_sampler = torch.utils.data.sampler.WeightedRandomSampler(samples_weights, num_samples=len(train_data), replacement=True)

validation_data_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, sampler=sampler, pin_memory=train_on_gpu)#, num_workers=4)

# INITIALIZE MODEL

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(int(IMG_HEIGHT/16) * int(IMG_WIDTH/16) * 256, 1000)
        self.fc2 = nn.Linear(1000, 300)
        self.drop_out2 = nn.Dropout()
        self.fc3 = nn.Linear(300, 1)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
        out = self.fc1(out)
        out = self.drop_out2(out)
        out = self.fc2(out)
        out = self.fc3(out)
        return out

model = ConvNet()
print(model)

if train_on_gpu:
    model.cuda()

criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.003, momentum= 0.9)

n_epochs = 50  # you may increase this number to train a final model
valid_loss_min = np.Inf  # track change in validation loss

for epoch in range(1, n_epochs + 1):

    # keep track of training and validation loss
    train_loss = 0.0
    valid_loss = 0.0

    ###################
    # train the model #
    ###################
    model.train()
    counter = 0
    for data, target in train_data_loader:
        # move tensors to GPU if CUDA is available
        if train_on_gpu:
            data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)
        target = target.unsqueeze(1).type_as(output)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * data.size(0)

    ######################
    # validate the model #
    ######################
    model.eval()
    for data, target in validation_data_loader:
        if train_on_gpu:
            data, target = data.cuda(), target.cuda()
        output = model(data)
        target = target.unsqueeze(1).type_as(output)
        loss = criterion(output, target)
        valid_loss += loss.item() * data.size(0)

    # calculate average losses
    train_loss = train_loss / len(train_data_loader.dataset)
    valid_loss = valid_loss / len(validation_data_loader.dataset)

    # print training/validation statistics
    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
        epoch, train_loss, valid_loss))

    # save model if validation loss has decreased
    if valid_loss <= valid_loss_min:
        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
            valid_loss_min,
            valid_loss))
        torch.save(model.state_dict(), output_path + 'sacrum_stop_model.pt')
        valid_loss_min = valid_loss
