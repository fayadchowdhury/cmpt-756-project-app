import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

import torch.optim as optim

import os
import time

from PIL import Image, ImageOps

import random


class twoConvBlock(nn.Module):
  def __init__(self, input_channels, output_channels):
    super(twoConvBlock, self).__init__()
    #todo
    #initialize the block
    self.conv_layer1 = nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=1)
    self.conv_layer2 = nn.Conv2d(output_channels, output_channels, kernel_size=3, stride =1)
    self.batch_norm_layer = nn.BatchNorm2d(output_channels)

  def forward(self,image):
    #todo
    #implement the forward path
    image = self.conv_layer1(image)
    image = F.relu(image)
    image = self.conv_layer2(image)
    image = self.batch_norm_layer(image)
    image = F.relu(image)
    return image

class downStep(nn.Module):
  def __init__(self):
    super(downStep, self).__init__()
    #todo
    #initialize the down path
    self.max_pool_layer = nn.MaxPool2d(kernel_size=2, stride = 2)

  def forward(self, image):
    #todo
    #implement the forward path
    image = self.max_pool_layer(image)
    return image


class upStep(nn.Module):
  def __init__(self, input_channels, output_channels):
    super(upStep, self).__init__()
    #todo
    #initialize the up path
    self.up_sampling_layer = nn.ConvTranspose2d(input_channels, output_channels, kernel_size = 2, stride=2)
    self.conv = twoConvBlock(input_channels, output_channels)

  def forward(self,up, conv):
    #todo
    #implement the forward path
    upsampled = self.up_sampling_layer(up)
    crop_h = (conv.size()[2]-upsampled.size()[2])
    crop_w = (conv.size()[3]-upsampled.size()[3])
    # print(upsampled.shape)
    pad_upsampled = F.pad(upsampled, [crop_w // 2, crop_w - crop_w // 2,
                        crop_h // 2, crop_h - crop_h // 2])

    # padding = [crop_w // 2, crop_w - crop_w // 2, crop_h // 2, crop_h - crop_h // 2]

    # print(padding)

    # pad_upsampled = F.pad(upsampled, padding)

    #crop_conv = conv[:, :, crop_h_fix:conv.shape[2]-crop_h, crop_w_fix:conv.shape[3]-crop_w]
    # print(conv.shape)
    # print(pad_upsampled.shape)
    return torch.cat([conv, pad_upsampled],dim=1)

class UNet(nn.Module):
  def __init__(self):
    super(UNet, self).__init__()
    #todo
    #initialize the complete model
    self.conv1 = twoConvBlock(1,64)
    self.conv2 = twoConvBlock(64, 128)
    self.conv3 = twoConvBlock(128, 256)
    self.conv4 = twoConvBlock(256, 512)
    self.conv5 = twoConvBlock(512, 1024)
    self.conv6 = twoConvBlock(1024, 512)
    self.conv7 = twoConvBlock(512, 256)
    self.conv8 = twoConvBlock(256, 128)
    self.conv9 = twoConvBlock(128, 64)
    self.conv10 = nn.Conv2d(in_channels=64, out_channels=2, kernel_size=1, stride=1)

    self.down_step = downStep()

    self.upstep1 = upStep(1024, 512)
    self.upstep2 = upStep(512, 256)
    self.upstep3 = upStep(256, 128)
    self.upstep4 = upStep(128, 64)


  def forward(self, input):
    #todo
    #implement the forward path
    conv1_ = self.conv1(input)


    downsampled = self.down_step(conv1_)
    conv2_ = self.conv2(downsampled)
    downsampled = self.down_step(conv2_)
    conv3_ = self.conv3(downsampled)
    downsampled = self.down_step(conv3_)
    conv4_ = self.conv4(downsampled)
    downsampled = self.down_step(conv4_)
    conv5_ = self.conv5(downsampled)
    concat = self.upstep1(conv5_, conv4_)
    conv6_ = self.conv6(concat)
    concat = self.upstep2(conv6_, conv3_)
    conv7_ = self.conv7(concat)
    concat = self.upstep3(conv7_, conv2_)
    conv8_ = self.conv8(concat)
    concat = self.upstep4(conv8_, conv1_)
    conv9_ = self.conv9(concat)
    out = self.conv10(conv9_)

    return out

def transform(image):
  transforms1 = transforms.Compose([transforms.ToTensor()])
  transforms2 = transforms.Compose([transforms.PILToTensor()])
  img = transforms1(image)
  mean, std = img.mean([1, 2]), img.std([1, 2])
  normTrans = transforms.Compose([transforms.Normalize(mean, std)])
  img = normTrans(img)
  return img