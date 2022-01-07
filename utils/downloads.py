from pathlib import Path
import urllib 
import requests
import subprocess
import torch

def safe_download(file, url, url2= None, min_bytes= 1E0, error_msg= ''):
  #Attempts to download file from url or url2, checks and removes incomplete download < min_bytes
  file= Path(file)
  assert_msg= f"Downloaded file '{file}' does not exist or size < min_bytes- {min_bytes}"
  print(file)
  try:
    print(f'Downloading {url} to {file}...')
    torch.hub.download_url_to_file(url, str(file))
    assert file.exists() and file.stat().st_size > min_bytes, assert_msg #Check
  except:
    file.unlink(missing_ok= True)
    print(f'ERROR: {e}\nRe-attempting {url2 or url} to {file}...')  
    os.system(f"curl -L '{url2 or url}' -o '{file}' --retry 3 -C -") # curl download, retry and resume on fail.

  finally:
    if not file.exists() or file.stat().st_size < min_bytes:
      file.unlink(missing_ok= True)
      print(f"ERROR: {assert_msg}\n{error_msg}")

    print('')  

def attempt_download(file, repo= 'ultralytics/yolov3'):
  #Attempt file download if does not exist
  file =Path(str(file).strip().replace("'", '')) #strip removes spaces in the begining as well as in the end
  if not file.exists():
    name= Path(urllib.parse.unquote(str(file))).name 
    if str(file).startswith(('http:/', 'https:/')): 
      url= str(file).replace(':/', '://') #pathlib turns :/ to ://
      name= name.split('?')[0] 
      safe_download(file= name, url= url, min_bytes= 1E5)
      return name

    #Github assets
    file.parent.mkdir(parents= True, exist_ok= True)
    try:
      response= requests.get(f'https://api.github.com/repos/{repo}/releases/latest').json() #Github api
      assets= [x['name'] for x in response['assets']] # release assets i.e. ['yolov3.pt'... ]
      tag= response['tag_name']
    except:
      assets= ['yolov3.pt', 'yolov3-spp-pt', 'yolov3-tiny-pt'] 
      try:
        tag= subprocess.check_output('git_tag', shell=True, stderr= subprocess.STDOUT).decode().split()[-1]
      except:
        tag= 'v9.5.0' #Current release
    if name in assets:
      safe_download(file, 
                    url = f'https://github.com/{repo}/releases/download/{tag}/{name}',
                    min_bytes= 1E5,
                    error_msg= f'{file} missing, try downloading from https://github.com{repo}/releases/')
  
  return str(file)
