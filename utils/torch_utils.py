import torch 
import torch.distributed as dist
from contextlib import contextmanager
from pathlib import Path
import datetime
import subprocess
import os
from utils.general import LOGGER
import platform 

@contextmanager
def torch_distributed_zero_first(local_rank: int):
  """
  Decorator to make all process in distributed training wait for each local_master to do something.

  """
  if local_rank not in [-1, 0]:
    dist.barrier(device_ids= [local_rank])

  yield
  if local_rank == 0:
    dist.barrier(device_ids= [0])

def date_modified(path= __file__):
  #return human-readable file modification date
  
  t= datetime.datetime.fromtimestamp(Path(path).stat().st_mtime)
  return f'{t.year}-{t.month}-{t.day}'

def git_describe(path= Path(__file__).parent):  #path must be a directory
  # return human-readable git description  
  
  s= f'git -C {path} describe --tags --long --always'
  try:
    return subprocess.check_output(s, shell= True, stderr= subprocess.STDOUT).decode()[:-1]
  except subprocess.CalledProcessError as e:
    return ''  
def select_device(device= '', batch_size= None, newline= True):
  #device = 'cpu' or  '0' or '0, 1, 2, 3'
  s= f'YOLOv3 { git_describe() or date_modified()} torch {torch.__version__}'
  device= str(device).strip().lower().replace('cuda', '') #  to string, "cuda:0" to '0
  cpu= device== 'cpu'
  if cpu:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # force torch.cuda.is_available() =False
  elif device:
    os.environ['CUDA_VISIBLE_DEVICES'] = device #set environment variable
    assert torch.cuda.is_available(), f'CUDA unavailable, invalid device {device} requested' 

  cuda = not cpu and torch.cuda.is_available()
  if cuda:
    devices= device.split(',') if device else '0' #range(torch.cuda.device_count()) #i.e. 0, 1, 6, 7
    n= len(devices)
    if n> 1 and batch_size: #check batch_size is divisible by device count
      assert batch_size % n == 0, f' batch size {batch_size} not multiple of GPU count {n}'
    space= ' ' * (len(s)+1)
    for i, d in eumerate(devices):
      p = torch.cuda.get_device_properties(i)
      s += f"{'' if i == 0 else space}CUDA:{d} ({p.name}, {p.total_memory / 1024 ** 2:.0f}MiB)\n"
  else:
    s+= 'CPU\n'
  
  if not newline:
    s= s.rstrip()

  LOGGER.info(s.encode().decode()('ascii', 'ignore') if platform.system() == 'Windows' else s)
  return torch.device('cuda:0' if cuda else 'cpu')
     


