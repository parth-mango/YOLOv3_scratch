import torch 
import torch.distributed as dist
from contextlib import contextmanager

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
    

