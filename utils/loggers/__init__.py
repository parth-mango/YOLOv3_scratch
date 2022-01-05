import os
import warnings
from utils.general import colorstr, emojis
import torch
from torch.utils.tensorboard import SummaryWriter
from utils.loggers.wandb.wandb_utils import WandbLogger
LOGGERS= ('csv', 'tb', 'wandb')
RANK= int(os.getenv('RANK', -1))




try:
  import wandb
  try:
    wandb_login_success= wandb.login(timeout= 30)
  except wandb.errors.UsageError:
    wandb_login_success= False
  if not wandb_login_success:
    wandb= None
except (ImportError, AssertionError):
  wandb= None  



class Loggers(): #Loggers class
  def __init__(self, save_dir= None, weights= None, opt= None, hyp= None, logger= None, include= LOGGERS):
    self.save_dir= save_dir
    self.weights= weights
    self.opt= opt
    self.hyp= hyp
    self.logger= logger #For printing results to console
    self.include= include
    self.keys= ['train/box_loss', 'train/obj_loss', 'train/cls_loss', #train_loss
                'metrics/precision', 'metrics/recall', 'metrics/mAP_0.5', 'metrics/mAP_0.5:0.95', #metrics
                'val/box_loss', 'val/obj_loss', 'val/cls_loss', # val_loss 
                'x/lr0', 'x/lr1', 'x/lr2'] #params
    for k in LOGGERS:
      setattr(self, k, None) # The setattr() function sets the value of the attribute of an object. Here the value is None
    self.csv =True
    
    #Message
    if not wandb:
      
      prefix= colorstr('Weights and Biases: ')
      
      s= f"{prefix}run 'pip install wandb'  to automatically track and visualize yoloV3 runs (RECOMMENDED) "
      print(emojis(s))

    #Tensorboard   
    s= self.save_dir
    if 'tb' in self.include and not self.opt.evolve:
      prefix= colorstr('Tensorboard: ')
      self.logger.info(f"{prefix} Start with 'tensorboard --logdir {s.parent}, view at http://localhost:6006/")
      self.tb= SummaryWriter(str(s))

    #W&B
    if wandb and 'wandb' in self.include:
      wandb_artifact_resume= isinstance(self.opt.resume, str) and self.opt.resume.startswith('wandb-artifact://')
      run_id= torch.load(self.weights).get('wandb_id') if self.opt.resume and not wandb_artifact_resume else None
      # print("run_id: ",run_id)
      self.opt.hyp= self.hyp
      self.wandb = WandbLogger(self.opt, run_id)
      # print(dir(self.wandb))
    else:
      self.wandb= None