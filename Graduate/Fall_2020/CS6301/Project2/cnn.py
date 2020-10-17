################################################################################
#
# LOGISTICS
#
#    Daniel Crawford
#    dsc160130
#
# DESCRIPTION
#
#    Image classification in PyTorch for ImageNet reduced to 100 classes and
#    down sampled such that the short side is 64 pixels and the long side is
#    >= 64 pixels
#
#    This script achieved a best accuracy of ??.??% on epoch ?? with a learning
#    rate at that point of ?.????? and time required for each epoch of ~ ??? s
#
# INSTRUCTIONS
#
#    1. Go to Google Colaboratory: https://colab.research.google.com/notebooks/welcome.ipynb
#    2. File - New Python 3 notebook
#    3. Cut and paste this file into the cell (feel free to divide into multiple cells)
#    4. Runtime - Run all
#
# NOTES
#
#    0. For a mapping of category names to directory names see:
#       https://gist.github.com/aaronpolhamus/964a4411c0906315deb9f4a3723aac57
#
#    1. The original 2012 ImageNet images are down sampled such that their short
#       side is 64 pixels (the other side is >= 64 pixels) and only 100 of the
#       original 1000 classes are kept.
#
#    2. Build and train a RegNetX image classifier modified as follows:
#
#       - Set stride = 1 (instead of stride = 2) in the stem
#       - Replace the first stride = 2 down sampling building block in the
#         original network by a stride = 1 normal building block
#       - The fully connected layer in the decoder outputs 100 classes instead
#         of 1000 classes
#
#       The original RegNetX takes in 3x224x224 input images and generates Nx7x7
#       feature maps before the decoder, this modified RegNetX will take in
#       3x56x56 input images and generate Nx7x7 feature maps before the decoder.
#       For reference, an implementation of this network took ~ 112 s per epoch
#       for training, validation and checkpoint saving on Sep 27, 2020 using a
#       free GPU runtime in Google Colab.
#
################################################################################

################################################################################
#
# IMPORT
#
################################################################################

# torch
import torch
import torch.nn       as     nn
import torch.optim    as     optim
from torch.autograd import Function

# torch utils
import torchvision
import torchvision.transforms as transforms

# additional libraries
import os
import urllib.request
import zipfile
import time
import math
import numpy             as np
import matplotlib.pyplot as plt
from tqdm import tqdm
################################################################################
#
# PARAMETERS
#
################################################################################

# data
DATA_DIR_1 = 'data'
DATA_DIR_2 = 'data/imagenet64'
DATA_DIR_TRAIN = 'data/imagenet64/train'
DATA_DIR_TEST = 'data/imagenet64/val'
DATA_FILE_TRAIN_1 = 'Train1.zip'
DATA_FILE_TRAIN_2 = 'Train2.zip'
DATA_FILE_TRAIN_3 = 'Train3.zip'
DATA_FILE_TRAIN_4 = 'Train4.zip'
DATA_FILE_TRAIN_5 = 'Train5.zip'
DATA_FILE_TEST_1 = 'Val1.zip'
DATA_URL_TRAIN_1 = 'https://github.com/arthurredfern/UT-Dallas-CS-6301-CNNs/raw/master/Data/Train1.zip'
DATA_URL_TRAIN_2 = 'https://github.com/arthurredfern/UT-Dallas-CS-6301-CNNs/raw/master/Data/Train2.zip'
DATA_URL_TRAIN_3 = 'https://github.com/arthurredfern/UT-Dallas-CS-6301-CNNs/raw/master/Data/Train3.zip'
DATA_URL_TRAIN_4 = 'https://github.com/arthurredfern/UT-Dallas-CS-6301-CNNs/raw/master/Data/Train4.zip'
DATA_URL_TRAIN_5 = 'https://github.com/arthurredfern/UT-Dallas-CS-6301-CNNs/raw/master/Data/Train5.zip'
DATA_URL_TEST_1 = 'https://github.com/arthurredfern/UT-Dallas-CS-6301-CNNs/raw/master/Data/Val1.zip'
DATA_BATCH_SIZE = 256#512
DATA_NUM_WORKERS = 4
DATA_NUM_CHANNELS = 3
DATA_NUM_CLASSES = 100
DATA_RESIZE = 64
DATA_CROP = 56
DATA_MEAN = (0.485, 0.456, 0.406)
DATA_STD_DEV = (0.229, 0.224, 0.225)

# model
HEAD_OUTPUT_CHANNELS = 32
STAGE_0_CHANNELS = 24
STAGE_1_CHANNELS = 56
STAGE_2_CHANNELS = 152
STAGE_3_CHANNELS = 368
STAGE_0_BLOCKS = 1
STAGE_1_BLOCKS = 1
STAGE_2_BLOCKS = 4
STAGE_3_BLOCKS = 7

# training
LEARNING_RATE = 0.001
NUM_EPOCHS = 20


# file
# add file parameters here

################################################################################
#
# DATA
#
################################################################################

# create a local directory structure for data storage
if (os.path.exists(DATA_DIR_1) == False):
    os.mkdir(DATA_DIR_1)
if (os.path.exists(DATA_DIR_2) == False):
    os.mkdir(DATA_DIR_2)
if (os.path.exists(DATA_DIR_TRAIN) == False):
    os.mkdir(DATA_DIR_TRAIN)
if (os.path.exists(DATA_DIR_TEST) == False):
    os.mkdir(DATA_DIR_TEST)

# download data
if (os.path.exists(DATA_FILE_TRAIN_1) == False):
    urllib.request.urlretrieve(DATA_URL_TRAIN_1, DATA_FILE_TRAIN_1)
if (os.path.exists(DATA_FILE_TRAIN_2) == False):
    urllib.request.urlretrieve(DATA_URL_TRAIN_2, DATA_FILE_TRAIN_2)
if (os.path.exists(DATA_FILE_TRAIN_3) == False):
    urllib.request.urlretrieve(DATA_URL_TRAIN_3, DATA_FILE_TRAIN_3)
if (os.path.exists(DATA_FILE_TRAIN_4) == False):
    urllib.request.urlretrieve(DATA_URL_TRAIN_4, DATA_FILE_TRAIN_4)
if (os.path.exists(DATA_FILE_TRAIN_5) == False):
    urllib.request.urlretrieve(DATA_URL_TRAIN_5, DATA_FILE_TRAIN_5)
if (os.path.exists(DATA_FILE_TEST_1) == False):
    urllib.request.urlretrieve(DATA_URL_TEST_1, DATA_FILE_TEST_1)

# extract data
with zipfile.ZipFile(DATA_FILE_TRAIN_1, 'r') as zip_ref:
    zip_ref.extractall(DATA_DIR_TRAIN)
with zipfile.ZipFile(DATA_FILE_TRAIN_2, 'r') as zip_ref:
    zip_ref.extractall(DATA_DIR_TRAIN)
with zipfile.ZipFile(DATA_FILE_TRAIN_3, 'r') as zip_ref:
    zip_ref.extractall(DATA_DIR_TRAIN)
with zipfile.ZipFile(DATA_FILE_TRAIN_4, 'r') as zip_ref:
    zip_ref.extractall(DATA_DIR_TRAIN)
with zipfile.ZipFile(DATA_FILE_TRAIN_5, 'r') as zip_ref:
    zip_ref.extractall(DATA_DIR_TRAIN)
with zipfile.ZipFile(DATA_FILE_TEST_1, 'r') as zip_ref:
    zip_ref.extractall(DATA_DIR_TEST)

# transforms
transform_train = transforms.Compose(
    [transforms.RandomResizedCrop(DATA_CROP), transforms.RandomHorizontalFlip(p=0.5), transforms.ToTensor(),
     transforms.Normalize(DATA_MEAN, DATA_STD_DEV)])
transform_test = transforms.Compose(
    [transforms.Resize(DATA_RESIZE), transforms.CenterCrop(DATA_CROP), transforms.ToTensor(),
     transforms.Normalize(DATA_MEAN, DATA_STD_DEV)])

# data sets
dataset_train = torchvision.datasets.ImageFolder(DATA_DIR_TRAIN, transform=transform_train)
dataset_test = torchvision.datasets.ImageFolder(DATA_DIR_TEST, transform=transform_test)

# data loader
dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=DATA_BATCH_SIZE, shuffle=True,
                                               num_workers=DATA_NUM_WORKERS, pin_memory=True, drop_last=True)
dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=DATA_BATCH_SIZE, shuffle=False,
                                              num_workers=DATA_NUM_WORKERS, pin_memory=True, drop_last=True)


################################################################################
#
# NETWORK BUILDING BLOCK
#
################################################################################

# X block
class XBlock(nn.Module):

    # initialization
    def __init__(self, Ni, No, Fr, Fc, Sr, Sc, G):
        # parent initialization
        super(XBlock, self).__init__()

        self.layers = nn.ModuleList([
            nn.Conv2d(Ni, No, (1, 1), stride=1, padding=0, bias=False),
            nn.BatchNorm2d(No),
            nn.ReLU(),
            nn.Conv2d(No, No, (Fr, Fc), stride=(Sr, Sc), padding=1, bias=False, groups=G),
            nn.BatchNorm2d(No),
            nn.ReLU(),
            nn.Conv2d(No, No, (1, 1), stride=1, padding=0, bias=False),
            nn.BatchNorm2d(No),
            nn.ReLU()
        ])

        self.conv = nn.ModuleList([
            nn.Conv2d(Ni, No, (1, 1), stride=(Sr, Sc), padding=0, bias=False),
            nn.BatchNorm2d(No),
            nn.ReLU()
        ])
        # operations needed to create a parameterized XBlock

    # forward path
    def forward(self, x):
        # add your code here
        # tie together the operations to create a parameterized XBlock
        y = x
        for layer in self.layers:
            y = layer(y)

        for layer in self.conv:
            x = layer(x)
        # return
        return y + x


################################################################################
#
# NETWORK
#
################################################################################

# define
class Model(nn.Module):

    # initialization
    def __init__(self,
                 data_num_channels,
                 head_output_channels,
                 stage_0_channels,
                 stage_0_blocks,
                 stage_1_channels,
                 stage_1_blocks,
                 stage_2_channels,
                 stage_2_blocks,
                 stage_3_channels,
                 stage_3_blocks,
                 data_num_classes):
        # parent initialization
        super(Model, self).__init__()

        G = 8
        # stem
        self.stem = nn.ModuleList([
            nn.Conv2d(data_num_channels, head_output_channels, (3, 3), stride=(1, 1), padding=1, bias=False),
            nn.BatchNorm2d(head_output_channels),
            nn.ReLU()
        ])

        self.enc0 = nn.ModuleList()
        self.enc0.append(XBlock(head_output_channels, stage_0_channels, 3, 3, 1, 1, G))
        for i in range(stage_0_blocks - 1):
            self.enc0.append(XBlock(stage_0_channels, stage_0_channels, 3, 3, 1, 1, G))

        self.enc1 = nn.ModuleList()
        self.enc1.append(XBlock(stage_0_channels, stage_1_channels, 3, 3, 2, 2, G))
        for i in range(stage_1_blocks - 1):
            self.enc1.append(XBlock(stage_1_channels, stage_1_channels, 3, 3, 1, 1, G))

        self.enc2 = nn.ModuleList()
        self.enc2.append(XBlock(stage_1_channels, stage_2_channels, 3, 3, 2, 2, G))
        for i in range(stage_2_blocks - 1):
            self.enc2.append(XBlock(stage_2_channels, stage_2_channels, 3, 3, 1, 1, G))

        self.enc3 = nn.ModuleList()
        self.enc3.append(XBlock(stage_2_channels, stage_3_channels, 3, 3, 2, 2, G))
        for i in range(stage_3_blocks - 1):
            self.enc3.append(XBlock(stage_3_channels, stage_3_channels, 3, 3, 1, 1, G))

        self.head = nn.ModuleList([
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(stage_3_channels, data_num_classes)
        ])

        self.output = nn.Softmax(dim=1)



    # forward path
    def forward(self, x):
        # add your code here
        # tie together the operations to create a modified RegNetX-200MF
        y = x
        for layer in self.stem:
            y = layer(y)

        for layer in self.enc0:
            y = layer(y)

        for layer in self.enc1:
            y = layer(y)

        for layer in self.enc2:
            y = layer(y)

        for layer in self.enc3:
            y = layer(y)

        for layer in self.head:
            y = layer(y)

        # return 100 classes
        return self.output(y)

def weights_init(module):
    print(module)
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.Linear):
        nn.init.xavier_normal_(module.weight.data)
        nn.init.xavier_normal_(module.weight.data)

# create
model = Model(DATA_NUM_CHANNELS,
              HEAD_OUTPUT_CHANNELS,
              STAGE_0_CHANNELS,
              STAGE_0_BLOCKS,
              STAGE_1_CHANNELS,
              STAGE_1_BLOCKS,
              STAGE_2_CHANNELS,
              STAGE_2_BLOCKS,
              STAGE_3_CHANNELS,
              STAGE_3_BLOCKS,
              DATA_NUM_CLASSES)

model.apply(weights_init)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# enable data parallelization for multi GPU systems
if (torch.cuda.device_count() > 1):
    model = nn.DataParallel(model)
else:
    model = model.to(device)

print('Using {0:d} GPU(s)'.format(torch.cuda.device_count()), flush=True)

################################################################################
#
# ERROR AND OPTIMIZER
#
################################################################################

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

################################################################################
#
# TRAINING
#
################################################################################

time.sleep(0.2)
for epoch in range(NUM_EPOCHS):
    train_loss = 0
    n = 0
    tq = tqdm(dataloader_train)

    tq.set_description('Average Train Loss: {0: 5.6f}'.format(float('0')))
    model.train()
    for i, data in enumerate(tq):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        optimizer.zero_grad()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * len(inputs)
        n += len(inputs)

        tq.set_description('Average Train Loss: {0:5.6f}'.format(train_loss / n))

    model.eval()
    with torch.no_grad():
        test_len = 0
        test_correct = 0
        for data in dataloader_test:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            _, predicted = torch.max(outputs.data, 1)
            test_len += len(labels)
            test_correct += (predicted == labels).sum().item()
    print('Test Accuracy: {0:6.2f}%'.format(test_correct / test_len * 100))
    time.sleep(0.2)




# add your code here
# perform network training, validation and checkpoint saving
# see previous examples in the Code directory
