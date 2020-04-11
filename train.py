import numpy as np
import random
import torch
import os

from methods import backbone
from methods.backbone import model_dict
from data.datamgr import SetDataManager
from methods.maml import MAML
from options import parse_args, get_resume_file

def train(base_loader, val_loader, model, max_acc, total_it, start_epoch, stop_epoch, params):

  for epoch in range(start_epoch,stop_epoch):

    # update lr
    model.update_lr()

    # train
    model.train()
    total_it = model.train_loop(epoch, stop_epoch, base_loader, total_it)
    model.eval()

    # validate
    acc = model.test_loop(val_loader, epoch=epoch)
    if acc > max_acc :
      print("best model! save...")
      max_acc = acc
      outfile = os.path.join(params.checkpoint_dir, 'best_model.tar')
      torch.save({'epoch':epoch, 'total_it':total_it, 'max_acc':max_acc, 'state':model.state_dict(), 'optimizer':model.optimizer.state_dict()}, outfile)
    else:
      print("QQ! best accuracy {:f}".format(max_acc))

    # save model
    if (epoch % params.save_freq==0) or (epoch==stop_epoch - 1):
      outfile = os.path.join(params.checkpoint_dir, '{:d}.tar'.format(epoch))
      torch.save({'epoch':epoch, 'total_it':total_it, 'max_acc':max_acc, 'state':model.state_dict(), 'optimizer':model.optimizer.state_dict()}, outfile)

  return model

if __name__=='__main__':

  params = parse_args('train')
  print('--- training: {} ---'.format(params.name))
  print(params)

  # fix seed
  if params.seed != 41608:
    torch.manual_seed(params.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(params.seed)
    random.seed(params.seed)

  # output and tensorboard directory
  params.tf_dir = '%s/log/%s'%(params.save_dir, params.name)
  params.checkpoint_dir = '%s/checkpoints/%s'%(params.save_dir, params.name)
  if not os.path.isdir(params.checkpoint_dir):
    os.makedirs(params.checkpoint_dir)

  # prepare dataset
  print('--- prepare dataloader {} ---'.format(params.dataset))#{} and model {} ---'.format(params.dataset, params.method))
  if 'Conv' in params.model:
    image_size = 84
  else:
    image_size = 224
  base_file  = os.path.join(params.data_dir, params.dataset, 'base.json')
  val_file   = os.path.join(params.data_dir, params.dataset, 'val.json')
  n_query = max(1, int(16* params.test_n_way/params.train_n_way)) #if test_n_way is smaller than train_n_way, reduce n_query to keep batch size small
  train_few_shot_params    = dict(n_way = params.train_n_way, n_support = params.n_shot)
  base_datamgr            = SetDataManager(image_size, n_query = n_query,  **train_few_shot_params)
  base_loader             = base_datamgr.get_data_loader( base_file , aug = params.train_aug )
  test_few_shot_params     = dict(n_way = params.test_n_way, n_support = params.n_shot)
  val_datamgr             = SetDataManager(image_size, n_query = n_query, **test_few_shot_params)
  val_loader              = val_datamgr.get_data_loader( val_file, aug = False)
  train_few_shot_params['dropout_method'] = params.dropout_method
  train_few_shot_params['dropout_p'] = params.dropout_p
  train_few_shot_params['dropout_schedule'] = params.dropout_schedule

  # prepare model
  print('--- prepare model {} (backbone {}) ---'.format(params.method, params.model))
  if params.method in ['maml' , 'maml_approx']:
    backbone.ConvBlock.maml = True
    backbone.SimpleBlock.maml = True
    backbone.BottleneckBlock.maml = True
    backbone.ResNet.maml = True
    model           = MAML(model_dict[params.model], approx=(params.method=='maml_approx'), tf_path=params.tf_dir, **train_few_shot_params)
  else:
    raise ValueError('Unknown method')
  model = model.cuda()

  max_acc = 0
  total_it = 0
  start_epoch = params.start_epoch
  stop_epoch = params.stop_epoch

  # resume/warmup or not
  if params.resume != '':
    resume_file = get_resume_file('%s/checkpoints/%s'%(params.save_dir, params.resume))
    if resume_file is not None:
      print('  load resume file: {}'.format(params.resume))
      tmp = torch.load(resume_file)
      max_acc = tmp['max_acc']
      total_it = tmp['total_it']
      start_epoch = tmp['epoch']+1
      model.load_state_dict(tmp['state'])
      model.optimizer.load_state_dict(tmp['optimizer'])
    else:
      raise ValueError('No resume file')
  elif params.warmup != 'gg3b0':
    warmup_resume_file = get_resume_file('%s/checkpoints/%s'%(params.save_dir, params.warmup))
    if warmup_resume_file is not None:
      print('  load pretrained file: {}'.format(params.warmup))
      tmp = torch.load(warmup_resume_file)
      state = tmp['state']
      state_keys = list(state.keys())
      for i, key in enumerate(state_keys):
        if "feature." in key:
          newkey = key.replace("feature.","")  # an architecture model has attribute 'feature', load architecture feature to backbone by casting name from 'feature.trunk.xx' to 'trunk.xx'
          state[newkey] = state.pop(key)
        else:
          state.pop(key)
      model.feature.load_state_dict(state)
    else:
      raise ValueError('No warm_up file')
  model.set_scheduler(start_epoch - 1, stop_epoch)

  # start training
  print('--- starting ---')
  model = train(base_loader, val_loader,  model, max_acc, total_it, start_epoch, stop_epoch, params)
