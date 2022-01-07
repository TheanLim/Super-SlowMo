import torch
import torch.nn as nn
import numpy as np

def encode(in_channels, out_channels, kernel_size=3):
  return nn.Sequential(
      # Set padding based on kernel_size (with stride=1) to maintain the same shape
      nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=int((kernel_size - 1) / 2)),
      nn.LeakyReLU(inplace=True, negative_slope=0.1),
      nn.Conv2d(out_channels, out_channels, kernel_size,stride=1, padding=int((kernel_size - 1) / 2)),
      nn.LeakyReLU(inplace=True, negative_slope=0.1),
      nn.AvgPool2d(kernel_size=2,stride=2))

def decode(in_channels, out_channels, kernel_size=3, scale_factor=2):
  return nn.Sequential(
    nn.Upsample(scale_factor, mode="bilinear", align_corners=False),
    #in_channels is doubled because of upsample
    #Remove the last layer = Average pool layer
    *encode(in_channels*scale_factor, out_channels, kernel_size)[:-1] 
  )

class FlowPred(nn.Module):
  def __init__(self, in_channels=6):
    super(FlowPred, self).__init__()
    self.down1 = encode(in_channels, 32, 7)
    self.down2 = encode(32, 64, 5)
    self.down3 = encode(64, 128, 3)
    self.down4 = encode(128, 256, 3)
    self.down5 = encode(256, 512, 3)
    self.bottleneck = encode(512, 512, 3)[:-1] # Remove the Average Pooling
    self.up5 = decode(512, 512, 3)
    self.up4 = decode(512, 256, 3)
    self.up3 = decode(256, 128, 3)
    self.up2 = decode(128, 64, 3)
    self.up1 = decode(64, 32, 3)

    self.refine = nn.Sequential(
        nn.Conv2d(2*32, 32, kernel_size=3, padding=1), # 2*32 because we will concat up1 and down1 output
        nn.LeakyReLU(inplace=True, negative_slope=0.1))
    self.flowForw = nn.Conv2d(32, 2, kernel_size=1)
    self.flowBack = nn.Conv2d(32, 2, kernel_size=1)

  def forward(self, x):
    down1 = self.down1(x)
    down2 = self.down2(down1)
    down3 = self.down3(down2)
    down4 = self.down4(down3)
    down5 = self.down5(down4)
    bottleneck = self.bottleneck(down5)
    up5 = self.up5(bottleneck)
    up4 = self.up4(torch.concat((down5, up5), dim=1))
    up3 = self.up3(torch.concat((down4, up4), dim=1))
    up2 = self.up2(torch.concat((down3, up3), dim=1))
    up1 = self.up1(torch.concat((down2, up2), dim=1))

    refine = self.refine(torch.concat((down1, up1), dim=1))
    flowForw = self.flowForw(refine)
    flowBack = self.flowBack(refine)
    return flowForw, flowBack

class FlowInterp(nn.Module):
  def __init__(self, in_channels=16):
    super(FlowInterp, self).__init__()
    self.down1 = encode(in_channels, 32, 7)
    self.down2 = encode(32, 64, 5)
    self.down3 = encode(64, 128, 3)
    self.down4 = encode(128, 256, 3)
    self.down5 = encode(256, 512, 3)
    self.bottleneck = encode(512, 512, 3)[:-1] # Remove the Average Pooling
    self.up5 = decode(512, 512, 3)
    self.up4 = decode(512, 256, 3)
    self.up3 = decode(256, 128, 3)
    self.up2 = decode(128, 64, 3)
    self.up1 = decode(64, 32, 3)

    self.refine = nn.Sequential(
        nn.Conv2d(2*32, 32, kernel_size=3, padding=1), # 2*32 because we will concat up1 and down1 output
        nn.LeakyReLU(inplace=True, negative_slope=0.1))
    self.flowForw = nn.Conv2d(32, 2, kernel_size=1) 3 # One channel for vertical,and one for horizontal
    self.flowBack = nn.Conv2d(32, 2, kernel_size=1)
    self.vizMap = nn.Conv2d(32, 1, kernel_size=1)

  def forward(self, x):
    down1 = self.down1(x)
    down2 = self.down2(down1)
    down3 = self.down3(down2)
    down4 = self.down4(down3)
    down5 = self.down5(down4)
    bottleneck = self.bottleneck(down5)
    up5 = self.up5(bottleneck)
    up4 = self.up4(torch.concat((down5, up5), dim=1))
    up3 = self.up3(torch.concat((down4, up4), dim=1))
    up2 = self.up2(torch.concat((down3, up3), dim=1))
    up1 = self.up1(torch.concat((down2, up2), dim=1))

    refine = self.refine(torch.concat((down1, up1), dim=1))
    flowForw = self.flowForw(refine)
    flowBack = self.flowBack(refine)
    vizMap = torch.sigmoid(self.vizMap(refine)) # Limit the range to [0,1]
    return flowForw, flowBack, vizMap
    
class BackWarp(nn.Module):
  def __init__(self, width=352, height=352):
    super(BackWarp, self).__init__()
    self.width, self.height = width, height
    gridX, gridY = np.meshgrid(np.arange(width), np.arange(height))
    # Matching the shape of a 'flow'
    # Shape (1,)+(height,width) = (1,height,width)
    gridX, gridY = gridX.reshape((1,)+ gridX.shape), gridY.reshape((1,)+ gridY.shape)

  def forward(self, image, flow):
    # The 'image' has to be 4D
    # The 'flow' has the shape of (batchSize, c=2, height, width)
    # Extract horizontal and vertical flows.
    # These are relative displacement
    u = flow[:, 0, :, :]
    v = flow[:, 1, :, :]
    # Calculate absolute displacement along height and width axis
    # Expand Grids to match flow's shape
    x = self.gridX.expand_as(u) + u
    y = self.gridY.expand_as(v) + v
    # Indices to [-1,1]
    x = 2 * x / (self.width - 1) - 1
    y = 2 * y / (self.height - 1) - 1
    # stacking X and Y to get a shape of (batchSize, height, width, 2)
    grid = torch.stack((x,y), dim=3)
    return torch.nn.functional.grid_sample(image, grid, mode='bilinear', padding_mode='zeros', align_corners=None)
