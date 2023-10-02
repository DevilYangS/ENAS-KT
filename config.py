import argparse,os,logging

from datetime import datetime

from utils import process_config,save_log,get_dataset_information
from process_data.datareader import Datareader, Datareader_fold, Devide_Fold_and_Save

def get_common_search_config():
    parser = argparse.ArgumentParser("Searching for Transformer via Evolutionary Algorithm")
    parser.add_argument("--seed",type=int, default=1001, help="random seed")
    # dataset
    parser.add_argument("--data-dir", type=str, default='./data', help='data folder')
    parser.add_argument("--dataset", type=str, default='Ednet', help="dataset for evaluation",choices=['Ednet',
                                                                                                   'Assistment','Riid'])
    # base model setting
    parser.add_argument("--evalmodel",type=str,default='single',choices=['single'])
    parser.add_argument("--pre-train-path",type=str,default='./experiment/remote/Riid/60/cross_True_fold_t_epoch_best.pth', help='the path of pretrained state_dict')
    parser.add_argument("--dropout", type=float, default=0.0, help='dropout rate for FFN and self-attention')

    # validation method
    parser.add_argument("--cross-validation",default=True, action='store_true', help='flag of using cross_validation or not')
    parser.add_argument("--k-fold",type=int,default=5,help="the number of k-fold validation, available under cross-validation is True")

    # hyperparameters for model/ task
    parser.add_argument("--max_length",type=int,default=100,help="the maximal length of input sequence")
    parser.add_argument("--embed_size",type=int,default=128,help="the dimension of embedding size")

    #hyperparameters for training process
    parser.add_argument("--batch-size", type=int, default=128, help="batch size")

    # GPU settings
    parser.add_argument("--num-workers", type=int, default=0, help="number of workers")
    parser.add_argument("--n-gpu", type=int, default=1, help="number of gpus to use")

    # Saving DIR setiing
    parser.add_argument('--exp-name', type=str, default="./experiment/EvolutionarySearch", help="experiment name")
    config = parser.parse_args()
    config.exp_name = config.exp_name +'/'+ config.dataset+ 'Search'+('_' + datetime.now().strftime('%y-%m-%d_%H-%M-%S'))+'/'


    config = deal_with_DataPath_and_GeneratePkl(config)
    config = get_dataset_information(config)
    config = eval("get_{}_training_config".format(config.dataset))(config)
    process_config(config)
    save_log(config)

    return config



def get_common_train_config():
    parser = argparse.ArgumentParser("Augments Settings for Training Model")
    parser.add_argument("--seed",type=int, default=1, help="random seed")
    # dataset
    parser.add_argument("--data-dir", type=str, default='./data', help='data folder')
    parser.add_argument("--dataset", type=str, default='Ednet', help="dataset for evaluation",choices=['Ednet',  'Riid'])

    parser.add_argument('--no-augmentation',type=bool,default=False,help='Using augmentation or not: False is using')
    parser.add_argument("--num-classes", type=int, default=2, help="number of classes in dataset")

    # base model setting
    parser.add_argument("--model", type=str, default="Transformer_super_V3", help='model setting to use',
                        choices=['Transformer_super_V3,SAINT_p,Transformer_super_V1'])

    parser.add_argument("--evalmodel",type=str,default='single',choices=['single','weight-sharing'])
    parser.add_argument('--NAS',type = list, default=None)
    parser.add_argument("--pre-train-path",type=str,default=None, help='the path of pretrained state_dict')
    # validation method
    parser.add_argument("--cross-validation",default=True, action='store_true', help='flag of using cross_validation or not')
    parser.add_argument("--k-fold",type=int,default=5,help="the number of k-fold validation, available under cross-validation is True")
    parser.add_argument("--early-stop",action='store_true',default=True,help='flag of using early-stoping or not')
    parser.add_argument("---early-stop-num", type=int, default=5,
                        help="the number of early_stop_num, available under early-stoping  is True")
    # hyperparameters for model/ task
    parser.add_argument("--max_length",type=int,default=100,help="the maximal length of input sequence")
    parser.add_argument("--embed_size",type=int,default=128,help="the dimension of embedding size") # 128
    #hyperparameters for training process
    parser.add_argument("--batch-size", type=int, default=128, help="batch size")
    parser.add_argument("--epochs", type=int, default=30, help="number of training/fine-tunning epochs")
    parser.add_argument("--warmup-steps", type=int, default=4000, help='learning rate warm up steps')
    parser.add_argument("--lr", type=float, default=0.0005, help="learning rate")
    parser.add_argument("--wd", type=float, default=5e-06, help='weight decay')
    parser.add_argument("--dropout", type=float, default=0.0, help='dropout rate for FFN and self-attention')
    parser.add_argument("--use-adamw",action='store_true',default=False,help='flag of using early-stoping or not')
    # GPU settings
    parser.add_argument("--num-workers", type=int, default=0, help="number of workers")
    parser.add_argument("--n-gpu", type=int, default=1, help="number of gpus to use")
    # Saving DIR setiing
    parser.add_argument('--exp-name', type=str, default="./experiment", help="experiment name")
    config = parser.parse_args()

    if 'Transformer_super_V' in config.model :
        config.exp_name = config.exp_name +"/{}/super_model".format(config.model) if config.evalmodel == 'weight-sharing' else config.exp_name +"/{}/single_model".format(config.model)
    else:
        config.exp_name = "./new_experiment/{}".format(config.model)
    config.exp_name = config.exp_name +'/'+ config.dataset+ '_Training'+('_' + datetime.now().strftime('%y-%m-%d_%H-%M-%S'))+'/'


    if config.evalmodel == 'single' and 'Transformer_super_V' in config.model:


        # solution on EdNet
        config.NAS =  [[1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0], [0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0],
                       [0, 0, 2, 2, 2, 2, 0, 0, 1, 1, 0, 1], [1, 0, 1, 2, 1, 4, 2, 2, 3, 0, 2, 1],
                       [[0, 0, 0, 1],[0, 0, 1, 0],[0, 0, 0, 1], [0, 1, 0, 0]]                      ]

        config.pre_train_path = './Super_pth/EdNet/cross_True_fold_t_epoch_best.pth'



    config = deal_with_DataPath_and_GeneratePkl(config)
    config = get_dataset_information(config)
    config = eval("get_{}_training_config".format(config.dataset))(config)

    process_config(config)
    save_log(config)
    # exit(1)
    return config


def deal_with_DataPath_and_GeneratePkl(config):
    config.data_path = []
    if config.cross_validation:
        for i in range(config.k_fold):
            config.data_path.append(
                '{}/{}/data_information_{}_{}.train'.format(config.data_dir, config.dataset, str(config.max_length),str(i)))
    else:
        config.data_path.append( '{}/{}/data_information_{}.train'.format(config.data_dir,config.dataset,config.max_length))

    if not os.path.exists(config.data_path[0]): # if not os.path.exists(config.data_path[-1]):
        Devide_Fold_and_Save(config)

    return config


def get_Ednet_training_config(config):
    config.no_augmentation = True

    if config.evalmodel == 'single':
        # single model config
        config.epochs = 30
        config.lr = 0.0005  # useless for NoamOpt, its lr is related to {factor(default to 1), embedding_size}
        config.wd = 5e-06
        config.warmup_steps = 4000


    elif config.evalmodel == 'weight-sharing':
        # super model config
        config.early_stop = False
        config.epochs = 60*2
        config.lr = 0.005  # useless for NoamOpt, its lr is related to {factor(default to 1), embedding_size}
        config.wd = 0.0
        config.warmup_steps = 8000*2        #


    if config.pre_train_path is not None: # only for my approach

        config.warmup_steps = 4000
        config.wd = 5e-6
        config.use_adamw = True

        pass

        # config.dropout = 0.1

    return config





if __name__ == '__main__':
    get_common_train_config()