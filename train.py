import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from .model import FlowPred, FlowInterp,  BackWarp

## Initialize models
flowPred = FlowPred()
flowInterp = FlowInterp()
backWarp = BackWarp(352, 352)
## Add to GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
flowPred.to(device)
flowInterp.to(device)
## Model.train()
flowPred.train()
flowInterp.train()

## Optimizer
params = list(flowInterp.parameters()) + list(flowPred.parameters())
optimizer = optim.Adam(params, lr=0.0001)
# decrease learing rate by a factor of 10 in every 200 epochs
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[200, 400], gamma=0.1)

## Loss Metrics
L1Loss = nn.L1Loss()
MSELoss = nn.MSELoss()

## Features for calculating perceptual loss
vgg16 = torchvision.models.vgg16(pretrained=True)
conv_4_3 = nn.Sequential(*list(vgg16.children())[0][:22])
conv_4_3.to(device)
for param in conv_4_3.parameters():
		param.requires_grad = False
    
## Training Loops
# Assume that trainDataLoader contains trainDataList, 
# where the first element is the Image at time 0 and the last element is the Image at time 1
# The other elements are itermediate images at time t, where t is between 0 and 1
# Total of 500 epochs per authors
for epoch in range(500):
  for trainIndex, trainDataList in enumerate(trainDataLoader):
    I0 = trainDataList[0].to(device)
    I1 = trainDataList[-1].to(device)
    F01, F10 = flowPred(torch.stack((I0, I1), dim=1))
    lrList, lpList, lwList = [], [], []
    for i in range(1,8):
      t = i/8
      It = trainDataList[t]
      ##### Image Construction ####
      Ft1Hat = (1-t)*(1-t)*F01 - t*(1-t)*F10
      Ft0Hat = -(1-t)*t*F01 + t*t*F10
      gI1Ft1Hat = backWarp(I1, Ft1Hat)
      gI0Ft0Hat = backWarp(I0, Ft0Hat)
      modelInput = torch.cat((I1, gI1Ft1Hat, Ft1Hat, I0, gI0Ft0Hat, Ft0Hat), dim =1)
      deltaFt1, deltaFt0, Vt0 = flowInterp(modelInput)
      Vt1 = 1-Vt0
      Ft0 = Ft0Hat+deltaFt0
      Ft1 = Ft1Hat+deltaFt1
      #Normalization
      Z=(1-t)*Vt0+t*Vt1
      ItHat=(1/Z)*((1-t)*Vt0*backWarp(I0,Ft0)+t*Vt1*backWarp(I1,Ft1))
      ##### End of Image Construction ####

      #### ith iteration Loss Calculation ####
      #Reconstruction Loss
      lri = L1Loss(ItHat, It)
      lrList.append(lri)
      #Perceptual Loss
      lpi = MSELoss(conv_4_3(ItHat), conv_4_3(It))
      lpList.append(lpi)
      #Warping Loss
      lwi = L1Loss(ItHat, gI1Ft1Hat)+L1Loss(ItHat, gI0Ft0Hat)
      lwList.append(lwi)
      #### End of ith iteration Loss Calculation ####
    
    #### Loss Calculation for each pair of I0 and I1 ####
    lr = sum(lrList)/len(lrList)
    lp = sum(lpList)/len(lpList)
    ## Warping Loss
    gI1F01 = backWarp(I1, F01)
    gI0F10 = backWarp(I0, Ft10)
    lw = L1Loss(I0, gI1F01)+L1Loss(I1, gI0F10) + sum(lwList)/len(lwList)
    # Smoothness Loss
    ls01 = torch.mean(torch.abs(F10[:, :, :, :-1] - F10[:, :, :, 1:])) + torch.mean(torch.abs(F10[:, :, :-1, :] - F10[:, :, 1:, :]))
    ls10 = torch.mean(torch.abs(F01[:, :, :, :-1] - F01[:, :, :, 1:])) + torch.mean(torch.abs(F01[:, :, :-1, :] - F01[:, :, 1:, :]))
    ls = ls01+ls10
    # Total Loss
    loss = 0.8*lr+0.005*lp+0.4*lw+ls

    #BackProp
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
