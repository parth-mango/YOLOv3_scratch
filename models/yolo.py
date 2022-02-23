import torch.nn as nn
from pathlib import Path
from copy import deepcopy
from utils.general import LOGGER
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

    #Define model
    ch1= self.yaml['ch']= self.yaml.get('ch', ch)       
    if nc and nc != self.yaml['nc']:
      LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc= {nc}")
      self.yaml['nc'] = nc #overriding yaml values
    if anchors:
      LOGGER.info(f"Overriding model.yaml nc= {self.yaml['nc']} with nc= {nc}")
      self.yaml['anchors']= round(anchors)
    # print(type(self.yaml))
    self.model, self.save= parse_model(deepcopy(self.yaml), ch= [ch]) #model , savelist


def parse_model(d, ch): #model_dict, input_channels(3)
  LOGGER.info(f"\n{'':>3}{'from':>18}{'n':>3}{'params':>10}  {'module':<40}{'arguments':<30}")
  anchors, nc, gd, gw = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple']
  na= (len(anchors[0]) // 2) if instance(anchors, list) else anchors  #number of anchors
  no= na * (nc + 5) # number of outputs = anchors * (classes + 5)

  layers, save, c2= [], [], [] # layers, savelist, ch out
  for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']): # from(number of pre-layer for connection) 
