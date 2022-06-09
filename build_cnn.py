#!/usr/bin/env python3

import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
import torchvision
from torch.utils.data import Dataset, DataLoader, TensorDataset, ConcatDataset, random_split

print("")
print('\t CNN functions:')
print('\t minmax')
def minmax(x, offset=0.001):
	xBool = x!=0
	minimum = np.min(x[np.nonzero(x)])
	maximum = np.max(x[np.nonzero(x)])
	xNorm = xBool*((x-minimum)/maximum)
	return xNorm+offset, minimum, maximum, offset

print('\t stdmean')
def stdmean(x, offset=0.001):
	xBool = x!=0
	std = np.std(x[np.nonzero(x)])
	mean = np.mean(x[np.nonzero(x)])
	xNorm = xBool*(x-mean)/std
	return xNorm+offset, std, mean, offset

print('\t minmaxstd')
def minmaxstd(x, offset=0.001):
	xBool = x!=0
	std = np.std(x[np.nonzero(x)])
	x = xBool*(x/std)
	minimum = np.min(x[np.nonzero(x)])
	maximum = np.max(x[np.nonzero(x)])
	xNorm = xBool*((x-minimum)/maximum)
	return xNorm+offset, std, minimum, maximum, offset

print('\t Residual2D (class)')

# 2D block
class Residual2D(nn.Module):
	def __init__(self, in_channels, out_channels):
		super(Residual2D, self).__init__()
		self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(2,2), padding=(1,1), stride=(1,1))
		self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(4,4), padding=(1,1), stride=(1,1))
		self.bn1 = nn.BatchNorm2d(out_channels)
		self.bn2 = nn.BatchNorm2d(out_channels)
		# if the channel size changes, need to change the input size to match the output size
		if in_channels != out_channels:
			self.convx = nn.Conv2d(in_channels, out_channels, kernel_size=(1,1), stride=(1,1))
		else:
			self.convx = None
		
	def forward(self, x):
		y1 = F.relu(self.bn1(self.conv1(x)))
		y2 = self.bn2(self.conv2(y1))
		if self.convx:
			x = self.convx(x)
		return F.relu(y2+x)

print('\t Residual3D (class)')
class Residual3D(nn.Module):
	def __init__(self, in_channels, out_channels):
		super(Residual3D, self).__init__()
		self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=(2,2,2), padding=(1,1,1), stride=(1,1,1))
		self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=(4,4,4), padding=(1,1,1), stride=(1,1,1))
		self.bn1 = nn.BatchNorm3d(out_channels)
		self.bn2 = nn.BatchNorm3d(out_channels)
		# if the channel size changes, need to change the input size to match the output size
		if in_channels != out_channels:
			self.convx = nn.Conv3d(in_channels, out_channels, kernel_size=(1,1,1), stride=(1,1,1))
		else:
			self.convx = None
		
	def forward(self, x):
		y1 = F.relu(self.bn1(self.conv1(x)))
		y2 = self.bn2(self.conv2(y1))
		if self.convx:
			x = self.convx(x)
		return F.relu(y2+x)
	
print('\t resnet_block')
# producing a chain of blocks, the same input and ouput size    
def resnet_block(in_channels, out_channels, num_residuals, dimension='2D'):
	blk = []
	for i in range(num_residuals):
		if dimension == '2D':
			if i==0:
				blk.append(Residual2D(in_channels, out_channels))
			else:
				blk.append(Residual2D(out_channels, out_channels))
		if dimension == '3D':
			if i==0:
				blk.append(Residual3D(in_channels, out_channels))
			else:
				blk.append(Residual3D(out_channels, out_channels))
	return nn.Sequential(*blk)

print("\t ReConX (class)")
class ReConX(nn.Module):
	def __init__(self):
		super(ReCon, self).__init__()
		#self.block1 = resnet_block(in_channels=4, out_channels=4, num_residuals=1, dimension='2D')
		#self.block2 = resnet_block(in_channels=4, out_channels=2, num_residuals=1, dimension='2D')
		#self.block3 = resnet_block(in_channels=2, out_channels=1, num_residuals=1, dimension='2D')
		#self.mainArray = nn.Sequential(self.block1, self.block2, self.block3, nn.Flatten())

		self.block1 = resnet_block(in_channels=4, out_channels=4, num_residuals=1, dimension='2D')
		self.block2 = resnet_block(in_channels=4, out_channels=2, num_residuals=1, dimension='2D')
		self.block3 = resnet_block(in_channels=2, out_channels=1, num_residuals=1, dimension='2D')
		self.mainArray = nn.Sequential(self.block1, self.block2, self.block3, nn.Flatten())

		self.fc1 = nn.Sequential(
			#nn.Linear(300, 30), # 6480 = 10*10*60+300+8*50
			nn.Linear(357, 35),
			nn.ReLU())
		self.fc2 = nn.Sequential(
			nn.Linear(35, 16),
			nn.ReLU())
		self.fc3 = nn.Linear(16, 1) # output (xmax)
		
		# Error
		self.fc4 = nn.Sequential(
			#nn.Linear(300+4, 30),
			nn.Linear(357+1, 35),
			nn.ReLU())
		self.fc5 = nn.Sequential(
			nn.Linear(35, 16),
			nn.ReLU())
		self.fc6 = nn.Linear(16, 1) # output (xmax)
		
	def forward(self, x0):
		# export value
		y0 = self.mainArray(x0)
		pred = self.fc3(self.fc2(self.fc1(y0)))
			
		# export uncertainty 
		xErr = torch.cat((y0.clone().detach(), pred.clone().detach()),dim=-1) # gradient stop 
		predErr = self.fc6(self.fc5(self.fc4(xErr)))
		
		return pred, predErr

print("\t ReCon (class)")
class ReCon(nn.Module):
	def __init__(self):
		super(ReCon, self).__init__()

		self.block1 = resnet_block(in_channels=4, out_channels=4, num_residuals=1, dimension='2D')
		self.block2 = resnet_block(in_channels=4, out_channels=2, num_residuals=1, dimension='2D')
		self.block3 = resnet_block(in_channels=2, out_channels=1, num_residuals=1, dimension='2D')
		self.mainArray = nn.Sequential(
			self.block1, 
			self.block2, 
			self.block3, 
			nn.Flatten()
			)

		self.linear_relu_stack = nn.Sequential(
			nn.Linear(357, 35),
			nn.ReLU(),
			nn.Linear(35, 16),
			nn.ReLU(),
			nn.Linear(16, 4) # output (xmax)
			) 
		
		# Error
		self.linear_relu_stack_error = nn.Sequential(
			nn.Linear(357+4, 35),
			nn.ReLU(),
			nn.Linear(35, 16),
			nn.ReLU(),
			nn.Linear(16, 4) # output (error on xmax)
			)
		
	def forward(self, x0):
		# export value
		y0   = self.mainArray(x0)
		pred = self.linear_relu_stack(y0)
			
		# export uncertainty 
		xErr    = torch.cat((y0.clone().detach(), pred.clone().detach()),dim=-1) # gradient stop 
		predErr = self.linear_relu_stack_error(xErr)
		
		return pred, predErr

# # Loss and Optimizer
print("\t loss\n")
def loss(pred, predErr, label):
	return nn.MSELoss(reduction='mean')(pred, label) + nn.MSELoss(reduction='mean')(predErr, torch.abs(pred-label))






