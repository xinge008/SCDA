import os
import torch
import torch.distributed as dist
from torch.nn import Module
import torch.multiprocessing as mp
import logging
logger = logging.getLogger('global')

def average_gradients(model):
    """ average gradients """
    for param in model.parameters():
        if param.requires_grad and not (param.grad is None):
            dist.all_reduce(param.grad.data)

def broadcast_params(model):
    """ broadcast model parameters """
    # for models in model:
    for p in model.state_dict().values():
        dist.broadcast(p, 0)

def dist_init(port, backend = 'nccl'):
    method = mp.get_start_method(allow_none=True)
    if method is None:
        mp.set_start_method('spawn')
    logger.info('multiprocessing start method:{}'.format(method))
    proc_id = int(os.environ['SLURM_PROCID'])
    ntasks = int(os.environ['SLURM_NTASKS'])
    node_list = os.environ['SLURM_NODELIST']
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(proc_id%num_gpus)

    if '[' in node_list:
        beg = node_list.find('[')
        pos1 = node_list.find('-', beg)
        if pos1 < 0:
            pos1 = 1000
        pos2 = node_list.find(',', beg)
        if pos2 < 0:
            pos2 = 1000
        node_list = node_list[:min(pos1,pos2)].replace('[', '')
    addr = node_list[8:].replace('-', '.')
    os.environ['MASTER_PORT'] = port
    os.environ['MASTER_ADDR'] = addr
    os.environ['WORLD_SIZE'] = str(ntasks)
    os.environ['RANK'] = str(proc_id)
    if backend == 'nccl':
        dist.init_process_group(backend='nccl')
    else:
        dist.init_process_group(backend='gloo', rank=proc_id, world_size=ntasks)

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    return rank, world_size

