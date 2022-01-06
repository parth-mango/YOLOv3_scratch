from pathlib import Path
import urllib 
import requests


def attempt_download(file, repo= 'ultralytics/yolov3'):
  #Attempt file download if does not exist
  file =Path(str(file).strip().replace("'", '')) #strip removes spaces in the begining as well as in the end
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
    # assets= [x['name'] for x in response['assets']] # release assets i.e. ['yolov3.pt'... ]
  except:
    assets= ['yolov3.pt', 'yolov3-spp-pt', 'yolov3-tiny-pt']  