import torch.nn as nn
from pathlib import Path


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
    self.na= len(anchors[0]) // 2 # number of anchors - 3 anchors have been taken for yolov3
    self.grid= [torch.zeros(1)] * self.nl #init_grid
    self.anchor_grid= [torch.zeros(1)] * self.nl # init anchor grid 
    self.register_buffer('anchors', torch.tensor(anchors).float().view(self.nl, -1, 2)) #shape(nl, na, 2)
    # If you have parameters in your model, which should be saved and restored in the state_dict , but not trained by the optimizer, you should register them as buffers.
    self.m= nn.ModuleList(nn.conv2d(x, self.no * self.na, 1) for x in ch) # output for each channel i.e. number of anchor * output of each anchor
    

class Model(nn.Module):
  def __init__(self, cfg= 'yolov3.yaml', ch=3, nc= None, anchors= None): #model, input channels and number od classes
    super().__init__()
    if isinstance(cfg, dict):
      self.yaml= cfg # model dict
    else: # is *.yaml
      import yaml
      self.yaml_file= Path(cfg).name
      with open(cfg, encoding= 'ascii', errors= 'ignore') as f:
        self.yaml= yaml.safe_load(f) # model dict
        print(self.yaml)     