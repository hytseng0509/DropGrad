import torch
import numpy as np
import torch.optim
import torch.utils.data.sampler
import os

from methods import backbone
from methods.backbone import model_dict
from data.datamgr import SetDataManager
from methods.maml import MAML
from options import parse_args, get_best_file , get_assigned_file

if __name__ == '__main__':
  params = parse_args('test')
  print('test: {}'.format(params.name))
  acc_all = []
  iter_num = 1000

  # create model
  few_shot_params = dict(n_way = params.test_n_way , n_support = params.n_shot)
  if params.method in ['maml' , 'maml_approx']:
    backbone.ConvBlock.maml = True
    backbone.SimpleBlock.maml = True
    backbone.BottleneckBlock.maml = True
    backbone.ResNet.maml = True
    model = MAML(  model_dict[params.model], approx = (params.method == 'maml_approx') , **few_shot_params )
  else:
    raise ValueError('Unknown method')
  model = model.cuda()

  # load model
  checkpoint_dir = '%s/checkpoints/%s' %(params.save_dir, params.name)
  if params.save_iter != -1:
    modelfile   = get_assigned_file(checkpoint_dir,params.save_iter)
  else:
    modelfile   = get_best_file(checkpoint_dir)
  if modelfile is not None:
    tmp = torch.load(modelfile)
    model.load_state_dict(tmp['state'])

  # load data
  split = params.split
  if 'Conv' in params.model:
    image_size = 84
  else:
    image_size = 224
  datamgr = SetDataManager(image_size, n_eposide = iter_num, n_query = 15 , **few_shot_params)
  loadfile = os.path.join(params.data_dir, params.dataset, split + '.json')#configs.data_dir[params.dataset] + split + '.json'
  novel_loader     = datamgr.get_data_loader( loadfile, aug = False)

  # testing
  model.eval()
  acc_mean, acc_std = model.test_loop( novel_loader, return_std = True)

  # result
  print('%s-%s-%s %syway%sshot_test' %(params.dataset, params.model, params.method, params.test_n_way, params.n_shot))
  print('%d Test Acc = %4.2%% +- %4.2f%%' % (iter_num, acc_mean, 1.96*acc_std/np.sqrt(iter_num)))
