import os,random,torch,logging,time

import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"]='0'
import torch.backends.cudnn as cudnn

from config import get_common_train_config

from process_data.data_loader import *

from model.Transformer_super_V3 import Transformer_super_V3

from utils import *

def main():
    # get config
    config = get_common_train_config()
    # device
    config.device, config.device_ids = setup_device(config.n_gpu)

    # fix random seed
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)

    cudnn.enabled = True
    cudnn.benchmark = True
    cudnn.deterministic = True

    #----------------train--validation by once or k-fold-------------

    k_fold_optimal_effect = []

    for fold_i, fold_i_path in enumerate(config.data_path):

        if not config.cross_validation:
            print('******* No cross validation *******')
            logging.info('******* No cross validation *******')
        else:
            print('******* No.{}-fold cross validation *******'.format(fold_i + 1))
            logging.info('******* No.{}-fold cross validation *******'.format(fold_i + 1))


        optimal_effect = train_student(config, fold_i_path)
        k_fold_optimal_effect.append(optimal_effect)
        Average_optimal_effect = get_Average_optimal_effect(k_fold_optimal_effect)


    print_info = "******* Average optimal effect after {}-fold cross validation *******\n".format(config.k_fold)
    print(print_info)
    logging.info(print_info)
    for item in Average_optimal_effect.items():
        print(item)
        logging.info(item)

def train_student(config, fold_path):
    assert config.style == 'student'
    #  dataset and dataloader
    train_data = CTLSTMDataset(config = config,mode='train',fold_path=fold_path)
    test_data = CTLSTMDataset(config = config,mode='test',fold_path=fold_path)
    # train_data = test_data
    train_dataloader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True,
                                  drop_last=False, num_workers=config.num_workers, collate_fn=train_data.pad_batch_fn)
    test_dataloader = DataLoader(test_data, batch_size=config.batch_size*4, shuffle=False,
                                 drop_last=False, num_workers=config.num_workers, collate_fn=test_data.pad_batch_fn)
    del train_data, test_data
    gc.collect()
    #  define model
    #  Loss is intergated into model
    model = eval('{0}'.format(config.model))(config)
    if config.pre_train_path is not None:
        state_dict = torch.load(config.pre_train_path)
        model.load_state_dict(state_dict)
    if len(config.device_ids)>1:
       model = torch.nn.DataParallel(model, device_ids=config.device_ids)
    else:
        model = model.to(config.device)

    #  define optimizer and lr_schedule

    if config.pre_train_path is not None:
        optimzer = NoamOpt(config.embed_size, 1, config.warmup_steps, torch.optim.Adam(model.parameters(),
                                                                                       lr=config.lr,weight_decay=config.wd),step = config.warmup_steps)
    else:
        optimzer = NoamOpt(config.embed_size, 1, config.warmup_steps, torch.optim.Adam(model.parameters(),
                                                                                       lr=config.lr, weight_decay=config.wd))
    if config.use_adamw:
        optimzer = torch.optim.AdamW(model.parameters(),lr = 3e-4,weight_decay = config.wd)

    #  define training loop
    best_AUC = 0.0
    effect = initialize_effect()
    input_settings = [model,optimzer,config]

    for epoch in range(config.epochs):
        print_info = 'Epoch {}/{} starts training.'.format(epoch,config.epochs)
        print(print_info)
        logging.info(print_info)
    #---------------------------------------------------Training process ---------------------------------------------------
        if config.evalmodel =='weight-sharing':
            traning_epoch_avg_loss, epoch_acc, epoch_auc = train_super(input_settings,train_dataloader)
        else:
            traning_epoch_avg_loss, epoch_acc, epoch_auc = train(input_settings,train_dataloader)


        effect['train_loss'].append(traning_epoch_avg_loss)
        effect['train_acc'].append(epoch_acc)
        effect['train_auc'].append(epoch_auc)

    # ---------------------------------------------------Testing process ---------------------------------------------------
        test_epoch_avg_loss, test_epoch_acc, test_epoch_auc,test_rmse  = validation(input_settings,test_dataloader,test=True)
        effect['test_loss'].append(test_epoch_avg_loss)
        effect['test_acc'].append(test_epoch_acc)
        effect['test_auc'].append(test_epoch_auc)
        effect['test_rmse'].append(test_rmse)

        effect['val_loss'].append(test_epoch_avg_loss)
        effect['val_acc'].append(test_epoch_acc)
        effect['val_auc'].append(test_epoch_auc)
        effect['val_rmse'].append(test_rmse)

        #------------------=---------------------------saving model and results------------------=---------------------------
        save_effect(effect, config, k_fold_num = fold_path[-5])
        if effect['val_auc'][-1]>best_AUC:
            best_AUC = effect['val_auc'][-1]
            save_model(model,config,k_fold_num = fold_path[-5],epoch = 'best')
        save_model(model,config,k_fold_num = fold_path[-5],epoch = 'last')
    # ------------------=---------------------------early stopping ?------------------=---------------------------
        if config.early_stop == True and epoch > config.early_stop_num:
            val_auc_temp = effect['val_auc'][-config.early_stop_num:]
            max_val_auc_temp = max(val_auc_temp)
            if max_val_auc_temp == val_auc_temp[0]:
                print_info = "epoch = {} early stop!".format(epoch)
                print(print_info)
                logging.info(print_info)
                break

    # ------------------=---------------------------print final results------------------=---------------------------
    optimal_effect = get_optimal_value(effect)
    for item in optimal_effect.items():
        print(item)
        logging.info(item)
    return optimal_effect




def train_super(settings, train_dataloader):
    random_n = 3 # 3

    model, optimzer, config = settings
    model.train()

    start_time = time.time()
    epoch_train_loss = []
    output_dict_list = []

    total = len(train_dataloader)
    for idx,item in enumerate(train_dataloader):
        optimzer.zero_grad()

        # minmal
        output_dict = model.forward(item,NAScoding='minimal')
        loss = model.loss(output_dict)
        loss.backward()
        # maximal
        output_dict = model.forward(item,NAScoding='maximal')
        loss = model.loss(output_dict)
        loss.backward()
        # random sample n times
        for i in range(random_n):
            output_dict = model.forward(item)
            loss = model.loss(output_dict)
            loss.backward()

        optimzer.step()
        print('\r               [Training {0:>2d}/{1:>2d}, Loss: {3:.5f}, used_time {2:.2f}min({4:.2f} s)]'.format(idx + 1, total,
                                                                                                                   (time.time() - start_time) / 60,loss,(time.time() - start_time)), end='')
        output_dict_list.append(output_dict)
        epoch_train_loss.append(loss.item())

    traning_epoch_avg_loss = np.mean(epoch_train_loss)
    metrics = epoch_acc = epoch_auc = 0.8
    metrics = get_metrics(output_dict_list)
    epoch_acc = metrics['acc']
    epoch_auc = metrics['auc']
    cost_time = time.time() - start_time

    print_info = "              for train loss: {0:.5f}, training time: {2:.3f}s, metrics: {1}".format(traning_epoch_avg_loss,metrics,cost_time)
    print(print_info)
    logging.info(print_info)

    return traning_epoch_avg_loss,epoch_acc,epoch_auc


def train(settings, train_dataloader):
    model, optimzer, config = settings
    model.train()

    start_time = time.time()
    epoch_train_loss = []
    output_dict_list = []

    total = len(train_dataloader)
    for idx,item in enumerate(train_dataloader):
        optimzer.zero_grad()
        output_dict = model.forward(item)
        loss = model.loss(output_dict)
        epoch_train_loss.append(loss.item())
        # if loss<np.mean(epoch_train_loss):
        #     continue
        loss.backward()
        optimzer.step()
        print('\r               [Training {0:>2d}/{1:>2d}, Loss: {3:.5f}, used_time {2:.2f}min({4:.2f} s)]'.format(idx + 1, total,
                                                                           (time.time() - start_time) / 60,loss,(time.time() - start_time)), end='')
        output_dict_list.append(output_dict)


    traning_epoch_avg_loss = np.mean(epoch_train_loss)
    metrics = epoch_acc = epoch_auc = 0.8
    metrics = get_metrics(output_dict_list)
    epoch_acc = metrics['acc']
    epoch_auc = metrics['auc']
    cost_time = time.time() - start_time

    print_info = "              for train loss: {0:.5f}, training time: {2:.3f}s, metrics: {1}".format(traning_epoch_avg_loss,metrics,cost_time)
    print(print_info)
    logging.info(print_info)

    return traning_epoch_avg_loss,epoch_acc,epoch_auc


def validation(settings,val_dataloader,test =False):
    model, optimzer, config = settings
    model.eval()
    if test:
        test_str = 'Test'
    else:
        test_str = 'Validation'

    start_time = time.time()
    output_dict_list = []
    epoch_val_loss = []
    with torch.no_grad():
        total = len(val_dataloader)
        for idx,item in enumerate(val_dataloader):
            output_dict = model.forward(item)
            loss = model.loss(output_dict)
            print('\r              [{3} {0:>2d}/{1:>2d}, Loss: {4:.5f}, used_time {2:.2f}min({5:.2f} s)]'.format(idx + 1, total,(time.time() - start_time) / 60,test_str,loss, time.time() - start_time), end='')
            output_dict_list.append(output_dict)
            epoch_val_loss.append(loss.item())

    val_epoch_avg_loss = np.mean(epoch_val_loss)
    metrics = get_metrics(output_dict_list)
    epoch_acc = metrics['acc']
    epoch_auc = metrics['auc']
    epoch_rmse = metrics['rmse']
    cost_time = time.time() - start_time

    print_info = "              for {3} loss: {0:.5f}, {3} time: {2:.3f}s, metrics: {1}".format(val_epoch_avg_loss,metrics,cost_time,test_str)
    print(print_info)
    logging.info(print_info)
    return val_epoch_avg_loss,epoch_acc,epoch_auc,epoch_rmse


if __name__ == '__main__':

    #1. train the super-Transformer (evalmode 'weight-sharing')
    #2. Run EvoTransformer to get the best architecture
    #3. re-run this script to fine-tune the best architecture based on the trained super-Transformer (evalmode 'single')
    main()

