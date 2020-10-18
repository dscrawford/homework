# Made by Daniel Crawford
# Student Net ID: dsc160130
# Course: CS6364 - Artificial Intelligence

import torch
import torch.nn as nn
import cv2
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from numpy import moveaxis



class Q1Model(nn.Module):
    def __init__(self, data_num_channels):
        super(Q1Model, self).__init__()
        self.conv1 = nn.Conv2d(data_num_channels, data_num_channels, kernel_size=(3, 3), padding=0, stride=1,
                               bias=False)
        self.conv1.weight.data[:, :, :, ] = torch.Tensor([1, 0, -1])
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=(3, 3), stride=1, padding=1)

        self.conv2 = nn.Conv2d(data_num_channels, data_num_channels, kernel_size=(3, 3), padding=0, stride=1,
                               bias=False)
        self.conv2.weight.data[:, :, ] = torch.Tensor([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
        print(self.conv2.weight.data)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=(3, 3), stride=1, padding=1)

    def forward(self, x):
        x = x.reshape(1, 1, *x.shape)
        altered_images = []
        y = x

        y = self.conv1(y)
        altered_images.append(y[0][0])
        y = self.relu1(y)
        altered_images.append(y[0][0])
        y = self.maxpool1(y)
        altered_images.append(y[0][0])

        y = x
        y = self.conv2(y)
        altered_images.append(y[0][0])
        y = self.relu2(y)
        altered_images.append(y[0][0])
        y = self.maxpool2(y)
        altered_images.append(y[0][0])

        return altered_images

def q1():
    img = cv2.imread('Q1.jpg', cv2.IMREAD_GRAYSCALE)
    img = torch.Tensor(img)
    model = Q1Model(1)
    images = model(img)

    for image in images:
        image = image.detach().numpy()
        imshow(image, cmap='gray')
        plt.show()
