import torch
import torch.nn as nn

class DropGrad(nn.Module):
  def __init__(self, p=0.1, mode='Gaussian', schedule=True):
    self.p = p
    self.p_cur = p
    self.mode = mode

  def config_p(self, epoch, stop_epoch):
    self.p_cur = epoch / (stop_epoch - 1) * self.p

  def foward(self, g):
