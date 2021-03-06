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
DATA_BATCH_SIZE = 32
DATA_NUM_WORKERS = 4
DATA_NUM_CHANNELS = 3
DATA_NUM_CLASSES = 100
DATA_RESIZE = 64
DATA_CROP = 56
DATA_MEAN = (0.485, 0.456, 0.406)
DATA_STD_DEV = (0.229, 0.224, 0.225)

# model
STAGE_0_CHANNELS = 24
STAGE_1_CHANNELS = 56
STAGE_2_CHANNELS = 152
STAGE_3_CHANNELS = 368
STAGE_0_BLOCKS = 1
STAGE_1_BLOCKS = 1
STAGE_2_BLOCKS = 4
STAGE_3_BLOCKS = 7

# training
WEIGHT_DECAY = 5e-5
OPTIMIZER_NAME = 'Adam'
CURRENT_EPOCH = 0

# training (linear warm up with cosine decay learning rate)
if OPTIMIZER_NAME == 'SGD':                                                                                                                                                                             
  TRAINING_LR_MAX          = 0.1
  TRAINING_LR_INIT_SCALE   = 0.1
  TRAINING_LR_FINAL_SCALE  = 0.01
else:
  TRAINING_LR_MAX          = 0.001
  TRAINING_LR_INIT_SCALE   = 0.01
  TRAINING_LR_FINAL_SCALE  = 0.01
TRAINING_LR_INIT_EPOCHS  = 5
TRAINING_LR_FINAL_EPOCHS = 100
TRAINING_NUM_EPOCHS      = TRAINING_LR_INIT_EPOCHS + TRAINING_LR_FINAL_EPOCHS
TRAINING_LR_INIT         = TRAINING_LR_MAX*TRAINING_LR_INIT_SCALE
TRAINING_LR_FINAL        = TRAINING_LR_MAX*TRAINING_LR_FINAL_SCALE


# file
MODEL_PATH = 'model.pt'
FILE_LOAD = False
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

# download data & extract
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


with zipfile.ZipFile(DATA_FILE_TRAIN_1, 'r') as zip_ref:
  zip_ref.extractall(DATA_DIR_TRAIN)
with zipfile.ZipFile(DATA_FILE_TRAIN_2, 'r') as zip_ref:
  zip_ref.extractall(DATA_DIR_TRAIN) 
with zipfile.ZipFile(DATA_FILE_TRAIN_3, 'r') as zip_ref:
  zip_ref.extractall(DATA_DIR_TRAIN)
with zipfile.ZipFile(DATA_FILE_TRAIN_4, 'r') as zip_ref:
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
    def __init__(self, Ni, No, Fr, Fc, Sr, Sc, G, DS=False):
        # parent initialization
        super(XBlock, self).__init__()

        self.layers = nn.ModuleList([
            nn.Conv2d(Ni, No, (1, 1), stride=1, padding=0, bias=False),
            nn.BatchNorm2d(No, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.Conv2d(No, No, (Fr, Fc), stride=(Sr, Sc), padding=1, bias=False, groups=G),
            nn.BatchNorm2d(No, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.Conv2d(No, No, (1, 1), stride=1, padding=0, bias=False),
            nn.BatchNorm2d(No, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU()
        ])

        if DS:
            self.conv = nn.ModuleList([
                nn.Conv2d(Ni, No, (1, 1), stride=(Sr, Sc), padding=0, bias=False)
            ])
        else:
            self.conv = nn.ModuleList()
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
            nn.Conv2d(data_num_channels, stage_0_channels, (3, 3), stride=(1, 1), padding=1, bias=False)
        ])

        self.enc0 = nn.ModuleList()
        self.enc0.append(XBlock(stage_0_channels, stage_0_channels, 3, 3, 1, 1, G, False))
        for i in range(stage_0_blocks - 1):
            self.enc0.append(XBlock(stage_0_channels, stage_0_channels, 3, 3, 1, 1, G))

        self.enc1 = nn.ModuleList()
        self.enc1.append(XBlock(stage_0_channels, stage_1_channels, 3, 3, 2, 2, G, True))
        for i in range(stage_1_blocks - 1):
            self.enc1.append(XBlock(stage_1_channels, stage_1_channels, 3, 3, 1, 1, G))

        self.enc2 = nn.ModuleList()
        self.enc2.append(XBlock(stage_1_channels, stage_2_channels, 3, 3, 2, 2, G, True))
        for i in range(stage_2_blocks - 1):
            self.enc2.append(XBlock(stage_2_channels, stage_2_channels, 3, 3, 1, 1, G))

        self.enc3 = nn.ModuleList()
        self.enc3.append(XBlock(stage_2_channels, stage_3_channels, 3, 3, 2, 2, G, True))
        for i in range(stage_3_blocks - 1):
            self.enc3.append(XBlock(stage_3_channels, stage_3_channels, 3, 3, 1, 1, G))

    
        self.headpool = nn.AdaptiveAvgPool2d(output_size=1)
        self.headfc = nn.Linear(stage_3_channels, data_num_classes, bias=True)

    # forward path
    def forward(self, x):
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

        y = self.headpool(y)
        y = y.view(y.size(0), -1)
        y = self.headfc(y)

        # return 100 classes
        return y

# create
model = Model(DATA_NUM_CHANNELS,
              STAGE_0_CHANNELS,
              STAGE_0_BLOCKS,
              STAGE_1_CHANNELS,
              STAGE_1_BLOCKS,
              STAGE_2_CHANNELS,
              STAGE_2_BLOCKS,
              STAGE_3_CHANNELS,
              STAGE_3_BLOCKS,
              DATA_NUM_CLASSES)


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
if OPTIMIZER_NAME == 'Adam':
  optimizer = optim.Adam(model.parameters(), lr=TRAINING_LR_MAX, betas=(0.9, 0.999), eps=1e-08, weight_decay=WEIGHT_DECAY, amsgrad=False)
else:
  optimizer = optim.SGD(model.parameters(), lr=TRAINING_LR_MAX,weight_decay=WEIGHT_DECAY, momentum=0.9, nesterov=True)

################################################################################
#
# TRAINING
#
################################################################################

def lr_schedule(epoch):
    # linear warmup followed by cosine decay                                                                                                                                                                                            
    if epoch < TRAINING_LR_INIT_EPOCHS:
        lr = (TRAINING_LR_MAX - TRAINING_LR_INIT)*(float(epoch)/TRAINING_LR_INIT_EPOCHS) + TRAINING_LR_INIT
    else:
        lr = (TRAINING_LR_MAX - TRAINING_LR_FINAL)*max(0.0, math.cos(((float(epoch) - TRAINING_LR_INIT_EPOCHS)/(TRAINING_LR_FINAL_EPOCHS - 1.0))*(math.pi/2.0))) + TRAINING_LR_FINAL

    return lr


time.sleep(0.2)
train_losses = []
test_losses = []
train_accuracy = []
test_accuracy = []

if FILE_LOAD:
  checkpoint = torch.load(MODEL_PATH)
  model.load_state_dict(checkpoint['model_state_dict'])
  optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
  CURRENT_EPOCH = checkpoint['epoch']

current_best_model = {
    'epoch': CURRENT_EPOCH,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': float('inf')
}



for epoch in range(CURRENT_EPOCH, TRAINING_NUM_EPOCHS):
    train_loss = 0
    n = 0
    tq = tqdm(dataloader_train)

    for g in optimizer.param_groups:
      g['lr'] = lr_schedule(epoch)

    it_str = 'Epoch {0: d}, lr: {1: 5.6f}, Average Train Loss: {2: 5.6f}'
    tq.set_description(it_str.format(epoch + 1, optimizer.param_groups[0]['lr'], float(0)))
    model.train()
    for i, data in enumerate(tq):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * len(inputs)
        n += len(inputs)
        tq.set_description(it_str.format(epoch + 1, optimizer.param_groups[0]['lr'], train_loss / n))

    model.eval()

    test_len = 0
    test_correct = 0
    test_loss = 0
    train_len = 0
    train_correct = 0
    train_loss = 0
    with torch.no_grad():
        for data in dataloader_test:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            _, predicted = torch.max(outputs.data, 1)
            test_len += len(labels)
            test_correct += (predicted == labels).sum().item()
            loss = criterion(outputs, labels)
            test_loss += loss.item() * len(labels)
        for data in dataloader_train:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            _, predicted = torch.max(outputs.data, 1)
            train_len += len(labels)
            train_correct += (predicted == labels).sum().item()
            loss = criterion(outputs, labels)
            train_loss += loss.item() * len(labels)

    train_losses.append(train_loss / train_len)
    test_losses.append(test_loss / test_len)
    train_accuracy.append(train_correct / train_len * 100)
    test_accuracy.append(test_correct / test_len * 100)

    if test_loss < current_best_model['loss']:
          current_best_model = {
              'epoch': epoch + 1,
              'model_state_dict': model.state_dict(),
              'optimizer_state_dict': optimizer.state_dict(),
              'loss': test_loss
          }


    print('Train/Test Accuracy: {0:5.2f}%/{1:5.2f}%'.format(train_correct / train_len * 100, test_correct / test_len * 100))
    print('Train/Test Loss: {0:4.2f}/{1:4.2f}'.format(train_loss / train_len, test_loss / test_len))
    time.sleep(0.2)

torch.save(current_best_model, MODEL_PATH)

epoch_list = list(range(TRAINING_NUM_EPOCHS))
plt.plot(epoch_list, train_losses, label='Avg Train Loss')
plt.plot(epoch_list, test_losses, label='Avg Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Average Loss (Cross Entropy)')
plt.legend()
plt.show()

plt.plot(epoch_list, train_accuracy, label='Train Accuracy')
plt.plot(epoch_list, test_accuracy, label='Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
