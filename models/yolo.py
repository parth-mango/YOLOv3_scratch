import torch.nn as nn


try:
  import thop # Pypi thop - for FLOPs computations
except:
  thop= None

  class Detect(nn.Module):
    stride= None #strides computed during build
    onnx_dynamic= False #ONNX export parameter

    def __init__(self, nc= 80, anchors= (), ch= (), inplace= True): #Detection Layer
      super().__init__()
      self.nc= nc # number of classes
      self.no= nc + 5 # number of outputs per anchor i.e. x, y, h, w, object_confidence_score
      self.nl= len(anchors) #Number of detection layers
      self.na= len(anchors[0]) // 2 # number of anchors - floor division to avoid decimal values ?
      self.grid= [torch.zeros(1)] * self.nl #init_grid
      self.anchor_grid= [torch.zeros(1)] * self.nl # init anchor grid 
      self.register_buffer('anchors', torch.tensor(anchors).float().view(self.nl, -1, 2))