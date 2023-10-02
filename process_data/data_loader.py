import logging
import pickle
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import numpy as np

from tqdm import tqdm

import gc


class CTLSTMDataset(Dataset):

    def __init__(self,config,mode='train',fold_path = None):

        self.dataset = config.dataset
        self.mode = mode
        assert mode == 'train' or mode == 'test' or mode == 'val'
        assert config.style == 'student' or config.style == 'time'


        if config.style == 'student':
            self.file_path =fold_path
        else:
            print("没有配置 style='time' !")
            self.file_path = fold_path


        #load dataset from stored pkl file
        pkl_file = open(self.file_path[:-5]+mode, 'rb')
        data_use = pickle.load(pkl_file)
        pkl_file.close()
        # columns
        pkl_file = open(self.file_path[:self.file_path.index(self.dataset)]+self.dataset+'/attribute.columns', 'rb')
        columns = pickle.load(pkl_file)
        pkl_file.close()
        self.len = len(data_use)
        self.columns = columns
        self.config =config

        self.min_seq = 2


        self.no_augmentation = True
        # self.no_augmentation = False

        # TODO  设置属性
        for i,item in enumerate(self.columns):
            if i>0:
                setattr(self,item,[])


        for Item_set in tqdm(enumerate(data_use), total=len(data_use), desc="Loading Dataset"):

            data_temp = Item_set[1] # columns:
            content_len = len(data_temp[1]) # skill_id for calculating length

            for idx in range(np.ceil(content_len/config.max_length).astype(int)):
                if idx>0 and self.no_augmentation:
                    break

                if len(data_temp[1][idx * config.max_length:(idx + 1)*config.max_length])<self.min_seq: # skill_id for example
                    break

                for i, item in enumerate(self.columns):
                    if i>0:
                        assert item==self.columns[i]
                        df = data_temp[i][idx * config.max_length:(idx + 1)*config.max_length]
                        getattr(self,item).append(df)



        self.len = len( getattr(self,self.columns[1]))

        print(self.columns)
        logging.info(self.columns)
        print('The length of {} dataset is {}'.format(config.dataset,self.len))
        logging.info('The length of {} dataset is {}'.format(config.dataset,self.len))
        del data_use
        gc.collect()

    def __len__(self):
        return self.len
    def __getitem__(self, index):


        batch_dict = {}
        for i, item in enumerate(self.columns):
            if i>0:
                batch_dict[item] = getattr(self,item)[index]
                # batch_dict[item] = getattr(self,item)[index][round(start*length):round(end*length)]
        return batch_dict








    def pad_batch_fn(self,many_batch_dict):
        # TODO:按照 x['event_seq'] 逆序排列 batch_data 当中的元组数据



        sorted_batch = sorted(many_batch_dict, key=lambda x: len(x[self.columns[1]]), reverse=True)

        seqs_length = [len(x[self.columns[1]]) for x in sorted_batch]

        data_list = []
        data_tensor=[]
        for i, item in enumerate(self.columns):
            if i > 0:
                if 'difficulty' in item:
                    data_list.append([torch.FloatTensor(seq[item]) for seq in sorted_batch])
                elif 'tags_set_seq' in item:
                    data_list.append([seq[item] for seq in sorted_batch])
                else:
                    data_list.append([torch.LongTensor(seq[item]) for seq in sorted_batch])

                if 'correct' in item:
                    data_tensor.append(torch.full((len(sorted_batch), self.config.max_length), 2, dtype=int).long())
                elif 'difficulty' in item:
                    data_tensor.append(torch.zeros(len(sorted_batch), self.config.max_length).float())
                elif 'tags_set_seq' in item:
                    data_tensor.append( torch.zeros([len(sorted_batch), self.config.max_length,7+self.config.data_info['tags_set_num'][-1] + 1]).long() )
                else:
                    data_tensor.append(torch.zeros(len(sorted_batch), self.config.max_length).long())



        for i, item in enumerate(self.columns):
            if i > 0:
                K_index = i-1
                for idx,data in enumerate(data_list[K_index]):
                    assert seqs_length[idx] == len(data)

                    if item=='tags_set_seq':


                        data_tensor[K_index][idx][:seqs_length[idx],:7] = torch.LongTensor(data)


                    else:
                        data_tensor[K_index][idx,:seqs_length[idx]] = data



        return_dict = {}
        for i,item in enumerate(self.columns):
            if i>0:
                k_index = i-1
                return_dict[item+'_tensor'] = data_tensor[k_index]


        return return_dict



def pad_batch_fn_EdNet(many_batch_dict):
    # TODO:按照 x['event_seq'] 逆序排列 batch_data 当中的元组数据

    n_sequence = len(many_batch_dict)  # 统计有多少条数据
    sorted_batch = sorted(many_batch_dict, key=lambda x: len(x['problem_seq']), reverse=True)
    problem_seqs = [torch.LongTensor(seq['problem_seq']) for seq in sorted_batch]
    skill_seqs = [torch.LongTensor(seq['skill_seq']) for seq in sorted_batch]
    time_lag_seqs = [torch.FloatTensor(seq['time_lag_seq']) for seq in sorted_batch]
    correct_seqs = [torch.LongTensor(seq['correct_seq']) for seq in sorted_batch]
    timestamp_seqs = [torch.FloatTensor(seq['timestamp_seq']) for seq in sorted_batch]

    elapsetime_seqs = [torch.FloatTensor(seq['elapsed_time'])*1e-5 for seq in sorted_batch]
    bundle_seqs = [torch.LongTensor(seq['bundle_seq']) for seq in sorted_batch]
    tags_seqs = [torch.LongTensor(seq['tags_seq']) for seq in sorted_batch]

    seqs_length = torch.LongTensor(list(map(len, skill_seqs)))

    problem_seqs_tensor = torch.zeros(len(sorted_batch), seqs_length.max()).long()
    skill_seqs_tensor = torch.zeros(len(sorted_batch), seqs_length.max()).long()
    time_lag_seqs_tensor = torch.zeros(len(sorted_batch), seqs_length.max()).float()
    correct_seqs_tensor = torch.full((len(sorted_batch), seqs_length.max()), 2,dtype=int).long()#-1
    timestamp_seqs_tensor = torch.zeros(len(sorted_batch), seqs_length.max()).float()

    elapsetime_seqs_tensor = torch.zeros(len(sorted_batch), seqs_length.max()).float()
    bundle_seqs_tensor = torch.zeros(len(sorted_batch), seqs_length.max()).long()
    tags_seqs_tensor = torch.zeros(len(sorted_batch), seqs_length.max()).long()



    for idx, (problem_seq, skill_seq, time_lag_seq, correct_seq, timestamp_seq, seq_len,elapsetime_seq,bundle_seq,tags_seq) \
            in enumerate(zip(problem_seqs, skill_seqs, time_lag_seqs,
                             correct_seqs, timestamp_seqs, seqs_length,elapsetime_seqs,bundle_seqs,tags_seqs)):

        problem_seqs_tensor[idx, :seq_len] = torch.LongTensor(problem_seq)
        skill_seqs_tensor[idx, :seq_len] = torch.LongTensor(skill_seq)
        time_lag_seqs_tensor[idx, :seq_len] = torch.FloatTensor(time_lag_seq)
        correct_seqs_tensor[idx, :seq_len] = torch.LongTensor(correct_seq)
        timestamp_seqs_tensor[idx, :seq_len] = torch.FloatTensor(timestamp_seq)

        elapsetime_seqs_tensor[idx, :seq_len] = torch.FloatTensor(elapsetime_seq)
        bundle_seqs_tensor[idx, :seq_len] = torch.LongTensor(bundle_seq)
        tags_seqs_tensor[idx, :seq_len] = torch.LongTensor(tags_seq)




    return_dict = {'problem_seq_tensor': problem_seqs_tensor,
                   'skill_seq_tensor': skill_seqs_tensor,
                   'time_lag_seq_tensor': time_lag_seqs_tensor,
                   'correct_seq_tensor': correct_seqs_tensor,
                   'timestamp_seq_tensor': timestamp_seqs_tensor,
                   'seqs_length': seqs_length,
                   'elapsed_time_seq_tensor': elapsetime_seqs_tensor,
                   'bundle_seq_tensor':bundle_seqs_tensor,
                   'tags_seq_tensor':tags_seqs_tensor
                   }
    return return_dict






