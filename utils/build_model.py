import os

import torch

from model.vast import (
    VAST
)

from .logger import (
    LOGGER
)


def build_model(args):
    assert args.run_cfg.checkpoint

    model = VAST(args.model_cfg)
    checkpoint = torch.load(args.run_cfg.checkpoint)
    model.load_state_dict(checkpoint)
    model.to(gpu)

    return model, None, 0

def load_from_pretrained_dir(args):


    try: ### huggingface trainer
        checkpoint_dir = args.run_cfg.pretrain_dir
        checkpoint_ls = [ i for i in os.listdir(checkpoint_dir) if i.startswith('checkpoint')]
        checkpoint_ls = [int(i.split('-')[1]) for i in checkpoint_ls]
        checkpoint_ls.sort()    
        step = checkpoint_ls[-1]
        
        try:
            checkpoint_name = f'checkpoint-{step}/pytorch_model.bin'
            ckpt_file = os.path.join(checkpoint_dir, checkpoint_name)
            checkpoint = torch.load(ckpt_file, map_location = 'cpu')
        except:
            checkpoint_name1 = f'checkpoint-{step}/pytorch_model-00001-of-00002.bin'
            ckpt_file1 = torch.load(os.path.join(checkpoint_dir, checkpoint_name1), map_location = 'cpu')
            checkpoint_name2 = f'checkpoint-{step}/pytorch_model-00002-of-00002.bin'
            ckpt_file2 = torch.load(os.path.join(checkpoint_dir, checkpoint_name2), map_location = 'cpu')
            ckpt_file1.update(ckpt_file2)
            checkpoint = ckpt_file1
        # checkpoint = {k.replace('module.',''):v for k,v in checkpoint.items()}
        LOGGER.info(f'load_from_pretrained: {ckpt_file}')

    except:
        checkpoint_dir = os.path.join(args.run_cfg.pretrain_dir,'ckpt')
        checkpoint_ls = [ i for i in os.listdir(checkpoint_dir) if i.startswith('model_step')]
        checkpoint_ls = [int(i.split('_')[2].split('.')[0]) for i in checkpoint_ls]
        checkpoint_ls.sort()    
        step = checkpoint_ls[-1]
            
        checkpoint_name = 'model_step_'+str(step)+'.pt'
        ckpt_file = os.path.join(checkpoint_dir, checkpoint_name)
        checkpoint = torch.load(ckpt_file, map_location = 'cpu')
        # checkpoint = {k.replace('module.',''):v for k,v in checkpoint.items()}
        LOGGER.info(f'load_from_pretrained: {ckpt_file}')


    return checkpoint


def load_from_resume(run_cfg):
    ckpt_dir = os.path.join(run_cfg.output_dir,'ckpt')
    previous_optimizer_state = [i  for i in os.listdir(ckpt_dir) if i.startswith('optimizer')]
    steps = [i.split('.pt')[0].split('_')[-1] for i in  previous_optimizer_state] 
    steps = [ int(i) for i in steps]
    steps.sort()
    previous_step = steps[-1]
    previous_optimizer_state = f'optimizer_step_{previous_step}.pt'
    previous_model_state = f'model_step_{previous_step}.pt'
    previous_step = int(previous_model_state.split('.')[0].split('_')[-1])
    previous_optimizer_state = os.path.join(ckpt_dir, previous_optimizer_state)
    previous_model_state = os.path.join(ckpt_dir, previous_model_state)
    
    assert os.path.exists(previous_optimizer_state) and os.path.exists(previous_model_state)
    LOGGER.info("choose previous model: {}".format(previous_model_state))
    LOGGER.info("choose previous optimizer: {}".format(previous_optimizer_state))
    previous_model_state = torch.load(previous_model_state,map_location='cpu')
    previous_optimizer_state = torch.load(previous_optimizer_state,map_location='cpu')
    return previous_model_state, previous_optimizer_state, previous_step






