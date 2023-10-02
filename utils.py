import os,logging,torch
import pickle
from sklearn.metrics import roc_auc_score, accuracy_score,mean_squared_error
import numpy as np

# from numba import njit
# from scipy.stats import rankdata
#
# @njit
# def _auc(actual, pred_ranks):
#     actual = np.asarray(actual)
#     pred_ranks = np.asarray(pred_ranks)
#     n_pos = np.sum(actual)
#     n_neg = len(actual) - n_pos
#     return (np.sum(pred_ranks[actual==1]) - n_pos*(n_pos+1)/2) / (n_pos*n_neg)
#
# def auc(actual, predicted):
#     pred_ranks = rankdata(predicted)
#     return _auc(actual, pred_ranks)

###
class NoamOpt:
    "Optim wrapper that implements rate."

    def __init__(self, model_size, factor, warmup, optimizer,step=0):
        self.optimizer = optimizer
        self._step = step
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
    def zero_grad(self):
        self.optimizer.zero_grad()

    def rate(self, step=None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
               (self.model_size ** (-0.5) *
                min(step ** (-0.5), step * self.warmup ** (-1.5)))

# def get_std_opt(model):
#     return NoamOpt(model.src_embed[0].d_model, 2, 4000,
#                    torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

from Operations import Get_auc
def get_metrics(output_dict_list):
    predictions = [torch.squeeze(output_dict['predictions'].detach()) for output_dict in output_dict_list]
    predictions = torch.cat(predictions).cpu()
    labels = [output_dict['labels'].detach() for output_dict in output_dict_list]
    labels = torch.cat(labels).cpu()
    # print("type of predictions ", type(predictions))  # <class 'torch.Tensor'>
    # print("size of predictions ", predictions.size())  # torch.Size([18192, 99])
    # print("predictions:\n", predictions)
    predictions_list = [float('%.4f' % i) for i in list(torch.squeeze(predictions))]
    predictions_round = [int(round(i)) for i in predictions_list]
    labels_list = [int(i) for i in list(torch.squeeze(labels))]

    accuracy = accuracy_score(labels_list, predictions_round)
    auc = roc_auc_score(labels_list, predictions_list)
    rmse = np.sqrt(mean_squared_error(labels_list, predictions_list))
    # auc = Get_auc(labels_list, predictions_list)

    return_dict = {'acc': float('%.6f' % accuracy), 'auc': float('%.6f' % auc),'rmse':float('%.6f' % rmse)}
    return return_dict


def get_optimal_value(effect):
    epoch = effect['val_auc'].index(max(effect['val_auc']))
    optimal_effect = {'epoch': epoch+1,
                      'train_acc': effect['train_acc'][epoch],
                      'train_auc': effect['train_auc'][epoch],
                      'val_acc': effect['val_acc'][epoch],
                      'val_auc': effect['val_auc'][epoch],
                      'test_acc': effect['test_acc'][epoch],
                      'test_auc': effect['test_auc'][epoch]}

    return optimal_effect

def initialize_effect():
    effect = {'train_loss': list(),
              'train_acc': list(),
              'train_auc': list(),
              'val_loss': list(),
              'val_acc': list(),
              'val_auc': list(),
              'val_rmse':list(),
              'test_loss': list(),
              'test_acc': list(),
              'test_auc': list(),
              'test_rmse':list(),
              }
    return effect

def get_Average_optimal_effect(k_fold_optimal_effect):

    train_acc = [ i['train_acc'] for i in k_fold_optimal_effect]
    train_auc = [ i['train_auc'] for i in k_fold_optimal_effect]
    val_acc = [ i['val_acc'] for i in k_fold_optimal_effect]
    val_auc = [ i['val_auc'] for i in k_fold_optimal_effect]
    test_acc = [ i['test_acc'] for i in k_fold_optimal_effect]
    test_auc = [ i['test_auc'] for i in k_fold_optimal_effect]

    Average_optimal_effect={'train_acc': sum(train_acc)/len(train_acc),
                            'train_auc': sum(train_auc)/len(train_auc),
                            'val_acc': sum(val_acc)/len(val_acc),
                            'val_auc': sum(val_auc)/len(val_auc),
                            'test_acc': sum(test_acc)/len(test_acc),
                            'test_auc': sum(test_auc)/len(test_auc),}

    return Average_optimal_effect


import json

def get_dataset_information(config):
    config.style = 'student'
    f = open('{}/{}/info.json'.format(config.data_dir,config.dataset),'r')
    info_data = json.load(f)
    config.data_info = info_data
    return config




#---------------------------------------------

def setup_device(n_gpu_use):
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print("Warning: There\'s no GPU available on this machine, training will be performed on CPU.")
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print("Warning: The number of GPU\'s configured to use is {}, but only {} are available on this machine.".format(n_gpu_use, n_gpu))
        n_gpu_use = n_gpu
    device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
    list_ids = list(range(n_gpu_use))
    return device, list_ids



# TODO
def count_parameters_in_MB(model):
    return sum([m.numel() for m in model.parameters()])/1e6

# DIR and log handle
def process_config(config):
    print(' *************************************** ')
    print(' The experiment name is {} '.format(config.exp_name))
    print(' *************************************** ')

    if not os.path.exists(config.exp_name):
        print('-----------------making experiment dir: \"{}\" -----------------'.format(config.exp_name))
        os.makedirs(config.exp_name)

    message = ''
    message += '           ----------------- Config ---------------\n'
    for k, v in sorted(vars(config).items()):
        comment = ''
        message += '{:>35}: {:<30}{}\n'.format(str(k), str(v), comment)
    message += '           ----------------- End -------------------'
    print(message)

def save_log(config):
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(filename='{}/log.log'.format(config.exp_name), level=logging.INFO,
                        format=log_format, datefmt='%m/%d %I:%M:%S %p')

    message = '\n           '
    message += '----------------- Config ---------------\n'
    for k, v in sorted(vars(config).items()):
        comment = ''
        message += '{:>35}: {:<30}{}\n'.format(str(k), str(v), comment)
    message += '            ----------------- End -------------------'
    logging.info(message)


#----------saving results and model----------
def save_effect(effect, config, k_fold_num):
    save_path = config.exp_name + 'cross_' + str(config.cross_validation) + \
                '_Effect_fold_' + str(k_fold_num)
    np.save(save_path, effect)

def save_model(model, config, k_fold_num, epoch):
    Model_Save_Path = config.exp_name + 'cross_' + \
                      str(config.cross_validation) + '_fold_' + str(k_fold_num) +\
                      '_epoch_' + str(epoch)+'.pth'
    f = open(Model_Save_Path, 'wb')
    torch.save(model.state_dict(), f)
    f.close()




