#utils func
from math import e
import platform
import logging
import os
import urllib3
from pathlib import Path
import glob
import torch
import yaml
from zipfile import ZipFile 

FILE= Path(__file__).resolve()
ROOT= FILE.parents[1] 


def colorstr(*input):
  *args, string= input if len(input)>1 else ('blue', 'bold', input[0])
  colors= {
      'black': '\033[30m',
      'red': '\033[31m',
      'green': '\033[32m',
      'yellow': '\033[33m',
      'blue': '\033[34m',
      'magenta': '\033[35m',
      'cyan': '\033[36m',
      'white': '\033[37m',
      'bright_black': '\033[90m',  # bright colors
      'bright_red': '\033[91m',
      'bright_green': '\033[92m',
      'bright_yellow': '\033[93m',
      'bright_blue': '\033[94m',
      'bright_magenta': '\033[95m',
      'bright_cyan': '\033[96m',
      'bright_white': '\033[97m',
      'end': '\033[0m',  # misc
      'bold': '\033[1m',
      'underline': '\033[4m'
  }
  return ''.join(colors[x] for x in args) + f'{string}' + colors['end']

def emojis(str= ''):
  #Return platform dependent emoji-safe version of string
  return str.encode().decode('ascii', 'ignore') if platform.system() == 'Windows' else str


def set_logging(name= None, verbose= True):
  #sets level and returns logger
  rank = int(os.getenv('RANK', -1))
  logging.basicConfig(format= "%(message)s", level= logging.INFO if (verbose and rank in (-1, 0)) else logging.WARNING)
  return logging.getLogger(name)

LOGGER= set_logging(__name__)

def methods(instance):
  # get class/instance methods
  return [f for f in dir(instance) if callable(getattr(instance, f)) and not f.startswith("__")]

def check_suffix(file= 'yolov3.pt', suffix=('.pt',), msg= ''):
  #Check files for acceptable suffix
  if file and suffix:
    if isinstance(suffix, str):
      suffix= [suffix]
      for f in file if isinstance(file, (list, tuple)) else [file]:
        s= Path(f).suffix.lower()
        if len(s):
          assert s in suffix, f"(msg){f} acceptable suffix is {suffix}"

def check_file(file, suffix=''):
  # Search/downoad file and return path
  check_suffix(file, suffix) 
  file= str(file)
  if Path(file).is_file() or file == '':
    return file
  elif file.startswith(('http:/', 'https:/')):
    url = str(Path(file)).replace(':/', '://') #Pathlib turns :/ -> ://
    file= Path(urllib3.parse.unquote(file).split('?')[0]).name # '%2F' to '/', split https://url.com/file.txt?auth 
    if Path(file).is_file():
      print(f'Found {url} locally at {file}')
    else:
      print(f'Downloading {url} to {file}...')
      torch.hub.download_url_to_file(url, file)
      assert Path(file).exists() and Path(file).stat().st_size > 0, f'File download failed: {url}'
      return file
  else:
    files= []
    for d in 'data', 'models', 'utils' :
      files.extend(glob.glob(str(ROOT / d / '**' / file), recursive= True))
    assert len(files), f'File not found: {file}'
    assert len(files) == 1, f"Multiple files match '{file}', specify exact path: {files}"
    return files[0]

def check_dataset(data, autodownload= True):
  # Download and/or unzip dataset if not found locally
  # Usage: https://github.com/ultralytics/yolov5/releases/download/v1.0/coco128_with_yaml.zip
  
  
  extract_dir= ''
  #Download data - optional
  if isinstance(data, (str, Path)) and str(data).endswith('.zip'): ## i.e. gs://bucket/dir/coco128.zip
    download(data, dir= '../datasets', unzip= True, delete= False, curl= False, threads =1)
  
  #Read yaml

  if isinstance(data, (str, Path)):
    with open(data, errors= 'ignore') as f:
      data= yaml.safe_load(f)

  #Parse yaml
  
  path= extract_dir or Path(data.get('path') or '')
  for k in 'train', 'val', 'test':
    if data.get(k):
      data[k]= str(path / data[k]) if isinstance(data[k], str) else [str(path / x) for x in data[k]]
      
      
  assert 'nc' in data, "Dataset 'nc' key missing" # Number of classes info
  if 'names' not in data:
    data['names']= [f'class{i}' for i in range(data['nc'])]
  # print(data, "Data genral 119")
  train, val, test, s= (data.get(x) for x in ('train', 'val', 'test', 'download'))

  if val:
    val= [Path(x).resolve() for x in (val if isinstance(val, list) else [val])] #Val Path
    if not all(x.exists() for x in val):
      print('\nWARNING: Dataset not found, nonexistent paths: %s' % [strx(x) for x in val if not x.exists()])
      if s and autodownload:
        root= path.parent if 'path' in data else '..'
        if s.startswith('http') and s.endswith('.zip'): #url
          f= Path(s).name #filename
          print(f'Downloading {s} to {f}..')
          torch.hub.download_url_to_file(s, f)
          Path(root).mkdir(parents= True, exist_ok= True)
          ZipFile(f).extractall(path=root) #unzip
          Path(f).unlink() #remove zip
          r= None
        elif s.startswith('bash'): #bashscript
          print(f'Running {s} ...')
          r= os.system(s)
        else:
          r= exec(s, {'yaml': data})
        print(f"Dataset autodownload {f'success, saved to {root}' if r in (0, None) else 'failure'}\n")  
      else:
        raise Exception('Dataset not found.')    
  
  return data

   


def download(url, dir= '', unzip= True, delete= True, curl= False, threads= -1):
  # Multi-threaded file download and unzip function, used in data.yaml for autodownload
  def download_one(url, dir):
    #download one file
    f= dir / Path(url).name #filename
    print("name 99 general.py",f)
    if Path(url).is_file(): # exists in current path
      Path(url).rename(f)   # move to dir
    elif not f.exists():
      print(f"Downloading {url} to {f} ...")
      if curl:
        os.system(f"curl -L '{url}' -o '{f}' --retry 9 -C -")
      else:
        print("General 107 ")  



def init_seeds(seed= 0):
  #Initialize random number generator seeds
  #cudnn seed 0 settings are slower and more reducible, else faster and less reducible

  import torch.backends.cudnn as cudnn
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  cudnn.benchmark, cudnn.deterministic= (False, True) if seed == 0 else (True, False)