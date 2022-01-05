import logging
import os
import sys
from pathlib import Path
from utils.general import check_dataset, check_file
import yaml
try:
  import wandb

  assert hasattr(wandb, '__version__')
except (ImportError, AssertionError):
  wandb= None
  

WANDB_ARTIFACT_PREFIX= 'wandb-artifact://'

def remove_prefix(from_string, prefix= WANDB_ARTIFACT_PREFIX):
  # print("out5 : ", from_string[len(prefix):] )
  return from_string[len(prefix):]

def get_run_info(run_path):
  run_path= Path(remove_prefix(run_path, WANDB_ARTIFACT_PREFIX))
  print("run path: ", run_path)
  run_id= run_path.stem
  project= run_path.parent.stem
  entity= run_path.parent.parent.stem
  model_artifact_name= 'run_' + run_id + '_model'
  return entity, project, run_id, model_artifact_name

def check_wandb_dataset(data_file):
  is_trainset_wandb_artifact= False
  is_valset_wandb_artifact= False
  # print(data_file, "Data file 68 wandb_utils")
  data_file= str(data_file)
  if check_file(data_file) and data_file.endswith('.yaml'):
    with open(data_file, errors= 'ignore') as f:
      data_dict= yaml.safe_load(f)
    is_trainset_wandb_artifact= (isinstance(data_dict['train'], str)and
                                  data_dict['train'].startswith(WANDB_ARTIFACT_PREFIX))  
    is_valset_wandb_artifact= (isinstance(data_dict['val'], str)and
                                  data_dict['val'].startswith(WANDB_ARTIFACT_PREFIX))

    if is_trainset_wandb_artifact or is_valset_wandb_artifact:
      return data_dict
    else:
      return check_dataset(data_file)  


class WandbLogger():
  """
  Logs training runs, datasets, models, and predictions to weights and
  biases.

  The information that's sent to wandb.ai includes hyperparameters, system 
  configuration and metrics, model metrics, and basic data metrics and analyses.

  """
  def __init__(self, opt, run_id= None, job_type= 'Training'):

    """
    - Initialize WandbLogger instance
    - Upload dataset if opt.upload_dataset is True
    - Set up training process if job_type is Training

    arguments:

    - opt(namespace) -- commandline arguments for this run
    - run_id(str) -- RUN ID of W&B run to be resumed
    - job_type(str) -- To set the job_type for this run 

    """

    #Pre-training routine
    self.job_type= job_type
    self.wandb, self.wandb_run= wandb, None if not wandb else wandb.run
    self.train_artifact, self.val_artifact= None, None
    self.train_artifact_path, self.val_artifact_path= None, None
    self.result_artifact= None
    self.val_table, self.result_table= None, None
    self.bbox_media_panel_images= []
    self.val_table_path_map = None
    self.max_imgs_to_log= 16
    self.wandb_artifact_data_dict= None
    self.data_dict= None

    if isinstance(opt.resume, str): #Checks resume from artifact
      if opt.resume.startswith(WANDB_ARTIFACT_PREFIX):
        entity, project, run_id, model_artifact_name= get_run_info(opt.resume)
        # print( entity, project, run_id, model_artifact_name)
        model_artifact_name= WANDB_ARTIFACT_PREFIX + get_run_info(opt.resume)
        assert wandb, 'install wandb to resume wandb runs'
        self.wandb_run= wandb.init(id= run_id, 
                                   project= project,
                                   entity= entity,
                                   resume= 'allow',
                                   allow_val_change= True)
        opt.resume= model_artifact_name
    elif self.wandb:
      self.wandb_run= wandb.init(config= opt, 
                                 resume= 'allow',
                                 project= 'YOLOv3' if opt.project== 'runs/train' else Path(opt.project).stem,
                                 entity= opt.entity,
                                 name= opt.name if opt.name != 'exp' else None,
                                 job_type= job_type,
                                 id= run_id,
                                 allow_val_change= True) if not wandb.run else wandb.run
    if self.wandb_run:
      # print("Run executed 88")  
      if self.job_type == 'Training':
        if opt.upload_dataset:
          if not opt.resume:
            self.wandb_artifact_data_dict= self.check_and_upload_dataset(opt)
        if opt.resume:
          #resume from artifact
          if isinstance(opt.resume, str) and opt.resume.startswith(WANDB_ARTIFACT_PREFIX):
            self.data_dict= dict(self.wandb_run.config.data_dict)
          else:
            self.data_dict = check_wandb_dataset(opt.data)
        else: 
          # print("In else loop 153 wandb_utils")
          self.data_dict= check_wandb_dataset(opt.data) 
          self.wandb_artifact_data_dict= self.wandb_artifact_data_dict or self.data_dict   
          
          #Write data_dict to config. Useful for resuming artifacts. Do this only when not resuming/
          self.wandb_run.config.update({'data_dict': self.wandb_artifact_data_dict}, allow_val_change= True)
        self.setup_training(opt)

  def check_and_upload_dataset(self, opt):
    """
    Check if the dataset format is compatible and upload it as W&B artifact

    arguments:
    opt(namespace) -- Commandline arguments for current run

    returns:
    Updated dataset info dictionary where local dataset paths are 
    replaced by WAND_ARTIFACT_PREFIX

    """
    assert wandb, 'install wandb to upload dataset'
    config_path= self.log_dataset_artifact(opt.data, 
                                          opt.single_cls,
                                          'YOLOv3' if opt.project == 'runs/train' else Path(opt.project).stem)



  def setup_training(self, opt):

    """
    Setup the necessary processes for training yolo models :
    - Attempt to download model checkpoint and dataset artifacts if opt.resume startswith WANDB_ARTIFACT_PREFIX
    - Update data_dict to contain info of previous run if resumed and the paths of dataset artifact if downloaded
    - Setup log_dict, initialize bbox_interval
    
    arguments:
    opt(namespace) -- commandline arguments for this run

    """
    self.log_dict, self.current_epoch= {}, 0
    self.bbox_interval= opt.bbox_interval
    
    print("Here in the loop 162 wandbutils")
    
    if isinstance(opt.resume, str):
      modeldir, _ =self.download_model_artifact(opt)
      print(modeldir, "Modeldir 121 wandbutils")

  
  def download_model_artifact(self, opt):

    """
    Download the model checkpoint artifact if the resume path starts with WANDB_ARTIFACT_PREFIX

    arguments:
    opt(namespace) -- Commandline arguments for this run
    
    """
    if opt.resume.startswith(WANDB_ARTIFACT_PREFIX):
      model_artifact= wandb.use_artifact(remove_prefix(opt.resume, WANDB_ARTIFACT_PREFIX)+ ":latest")
      assert model_artifact is not None, "Error: W&B model artifact doesn't exist."
      modeldir= model_artifact.download()
      epochs_trained= model_artifact.metadata.get('epochs_trained')
      total_epochs= model_artifact.metadata.get('total_epochs')
      is_finished= total_epochs is None
      assert not is_finished, 'training is finished, can only resume incomplete runs.'
      return modeldir, model_artifact
    return None, None

  def log_dataset_artifact(self, data_file, single_cls, project, overwrite_config= False):
    """
    Logs the dataset as W&B artifact and return the new data file with W&B links
    
    arguments:

    data_file(str) -- the .yaml file with information about the dataset like - path, classes, etc.
    single_cls(boolean) -- train multi class data as single class
    project (str) -- Project name, used to construct the artifact path
    overwrite_config (boolean) -- overwrites the data.yaml file if set to true otherwise creates a new file

    returns:
    
    The new .yaml file with artifact links. It can be used to start training directly from artifacts.
    
    """
    print("In log_dataset_artifact 45 wandb_utils.py")
    self.data_dict= check_dataset(data_file)













