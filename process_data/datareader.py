
# TODO 此 Datareader.py 文件用于将 data 文件中的 interactions.csv 文件加载
# TODO 要求：接收 max_length 用以控制最大长度，按照给定的比例创建 dataframe 格式的 train\val\test 集合

import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
import copy
import json

# TODO 此类用于将 interactions.csv 中的记录转为对象格式并保存到.pkl文件当中。我们有两种风格 style='student' 和 style='time',
# TODO 分别对应使用新用户时的划分（此时我们直接在.pkl文件当中划分出训练、验证和测试集）以及已知上一时刻预测下一时刻的划分
# TODO 注
'''当 style='time' 时，由于在得到 batch 时需要对 batch 中的序列按照序列长短由大到小排序进入 Model_list。
但本 Datareader 函数并不知道 batch_size 大小, 因此我们不在此处划分 style='time' 时的训练、验证和测试集，所有的序列数据都会
被存储在 data.data_df['train'] 当中，data.data_df['val'] 和 data.data_df['test'] 里是空的。
划分操作计划放在 myDataloader.py 中进行'''

class Datareader(object):

    def __init__(self, train_per=0.7, val_per=0.1, test_per=0.2, max_length=100, path='data/Junyi/interactions.csv', sep='\t'):
        assert train_per + val_per + test_per == 1

        self.train_per = train_per
        self.val_per = val_per
        self.test_per = test_per
        self.max_length = int(max_length)
        self.path = path
        self.sep = sep

        self.data_df = {
            'train': pd.DataFrame(), 'val': pd.DataFrame(), 'test': pd.DataFrame()
        }

        # self.inter_df = pd.read_csv(self.path, sep=self.sep)
        self.inter_df = pd.read_csv(self.path, sep=self.sep)
        user_wise_dict = dict()
        cnt, n_inters = 0, 0

        for user, user_df in self.inter_df.groupby('user_id'):

            df = user_df[:self.max_length]  # consider the first 50 interactions 只考虑前 50(self.max_step) 次互动, 如果不足 50 会得到所有互动
            user_wise_dict[cnt] = {
                'user_id': cnt+1,
                'skill_seq': df['skill_id'].values.tolist(),
                'problem_seq': df['problem_id'].values.tolist(),
                'timestamp': df['timestamp'].values.tolist(),
                'correct_seq': df['correct'].values.tolist()
            }
            user_wise_dict[cnt]['time_lag'] = \
                [0.] + list(map(lambda x: float('%.2f' % (x[0] - x[1])), zip(user_wise_dict[cnt]['timestamp'][1:], user_wise_dict[cnt]['timestamp'][:-1])))
            cnt += 1
            n_inters += len(df)
        # print('cnt:', cnt)
        self.user_seq_df = pd.DataFrame.from_dict(user_wise_dict, orient='index')  # 将 dict 转成 dataframe 格式
        # print("self.user_seq_df.head(2):\n", self.user_seq_df.head(2))
        # print("self.user_seq_df.columns:\n", self.user_seq_df.columns)  # Index(['user_id', 'skill_seq', 'problem_seq', 'time_seq', 'correct_seq'], dtype='object')
        self.n_users = int(max(self.inter_df['user_id'].values))
        self.n_skills = int(max(self.inter_df['skill_id']))
        self.n_problems = int(max(self.inter_df['problem_id']))

        self.sum_length = [len(value.iloc[0]) for index, value in self.user_seq_df[['skill_seq']].iterrows()]

    def show_columns(self):
        # print("self.user_seq_df.columns:\n", self.user_seq_df.columns)
        return self.user_seq_df.columns

    def devide_data(self, style='student'):
        assert style == 'time' or style == 'student'
        self.style = style

        if style == 'time':
            print("style='time' 并未配置！")
            self.data_df['train'] = self.user_seq_df
            self.train_length = [int(self.test_per * i) for i in self.sum_length]
            self.val_length = [int(self.val_per * i) for i in self.sum_length]
            self.test_length = [self.sum_length[i] - self.train_length[i] for i in range(len(self.sum_length))]

        elif style=='student':
            n_examples = len(self.user_seq_df)
            train_number = int(n_examples * self.train_per)
            val_number = int(n_examples * self.val_per)
            test_number = n_examples - train_number - val_number
            # print(train_number, val_number, test_number)

            self.data_df['train'] = self.user_seq_df.iloc[: train_number]
            self.data_df['val'] = self.user_seq_df.iloc[train_number: train_number + val_number]
            self.data_df['val'].index = range(len(self.data_df['val']))
            self.data_df['test'] = self.user_seq_df.iloc[train_number + val_number:]
            self.data_df['test'].index = range(len(self.data_df['test']))
    # def save_data(self):
    #     np.save('data/devide_data.npy', self.data_df)

# 为交叉验证设置的类
class Datareader_fold(object):

    def __init__(self, cross_validation=True,
                 k_fold=5,
                 max_length=100,
                 path='data/Junyi/interactions.csv',
                 sep='\t'):
        assert cross_validation == True  # 此类为 k 折交叉验证
        assert k_fold >= 1

        self.max_length = int(max_length)
        self.path = path
        self.sep = sep
        self.k_fold = k_fold
        if "Ednet" in self.path or 'Riid' in self.path:
            self.unit_time_lag = 1e-5
        elif "Junyi" in self.path:
            self.unit_time_lag = 1e-7
        else:
            self.unit_time_lag = 1

        self.data_df = {
            'train': pd.DataFrame(), 'test': pd.DataFrame()
        }


        if "Ednet" in self.path :
            self.inter_df = pd.read_csv(self.path)

            tags_list =np.array([])
            length = []
            sotred_str = ''
            for idx,itemt in enumerate(tqdm(self.inter_df['tags'].drop_duplicates().values)):
                sotred_str = sotred_str+itemt+';'
                length.append(len(itemt.split(';')))


            sotred_str = sotred_str[:-1]
            temp_tag = []
            for x in sotred_str.split(';'):
                temp_tag.extend([int(x)])
            tags_list = np.unique(np.hstack((tags_list, np.array(temp_tag))))

            tags_set_num = tags_list.shape[0]
            tags_set_num = tags_list.max()
            tags_set_Maxi_len = max(length)


        elif 'Riid' in self.path:


            self.inter_df = pd.read_csv(self.path)


        else:
            self.inter_df = pd.read_csv(self.path, sep=self.sep)

        self.no_augmentation = True# True



        user_wise_dict = dict()
        cnt, n_inters = 0, 0
        for user, user_df in tqdm(self.inter_df.groupby('user_id')):
            user_df.sort_values(by='timestamp', inplace=True)
            df = user_df

            if "Ednet" in self.path:

                tags_vector = []
                for jdxt,itemt in enumerate(df['tags'].values.tolist()):
                    temp_tag = np.zeros([tags_set_Maxi_len,])
                    for idx,x in enumerate(itemt.split(';')):
                        temp_tag[idx] = int(x)+2
                    tags_vector.append(temp_tag.astype(np.int32))
                tags_vector = np.array(tags_vector).astype(np.int32)


                user_wise_dict[cnt] = {
                    'user_id': cnt + 1,
                    'skill_seq': (df['skill_id'].values).astype(np.int32),
                    'problem_seq': (df['problem_id'].values).astype(np.int32),
                    'timestamp_seq': df['timestamp'].values,
                    'correct_seq': (df['correct'].values).astype(np.int16),

                    'elapsed_time_seq': np.array([0] + df['elapsed_time'].values.tolist()[:-1]),
                    'bundle_seq': (df['bundle_id'].values).astype(np.int32),
                    'tags_seq': (df['tags_id'].values).astype(np.int32),
                    'tags_set_seq': tags_vector,
                }

            elif 'Riid' in self.path:

                tags_vector = []
                for jdxt,itemt in enumerate(df['tags'].values.tolist()):
                    temp_tag = np.zeros([7,])
                    for idx,x in enumerate(itemt.split(' ')):
                        temp_tag[idx] = int(x)+2
                    tags_vector.append(temp_tag.astype(np.int32))
                tags_vector = np.array(tags_vector).astype(np.int32)

                user_wise_dict[cnt] = {
                    'user_id': cnt + 1,
                    'skill_seq': (df['skill_id'].values-1).astype(np.int32),
                    'problem_seq': (df['problem_id'].values).astype(np.int32),
                    'correct_seq': (df['correct'].values-1).astype(np.int16),

                    'elapsed_time_seq':  (df['elapsed_time'].values*1e5).astype(int),   # int is more efficient than float (memory)
                    'timestamp_seq': df['timestamp'].values,

                    'tags_seq': (df['tags_id'].values).astype(np.int32),
                    'tags_set_seq': tags_vector,

                    'explanation_seq': (df['prior_question_had_explanation'].values).astype(np.int16),
                    'difficulty_seq': (df['difficulty'].values).astype(np.float32)
                }


            else:
                user_wise_dict[cnt] = {
                    'user_id': cnt + 1,
                    'skill_seq': (df['skill_id'].values).astype(np.int32),
                    'problem_seq': (df['problem_id'].values).astype(np.int32),
                    'timestamp_seq': df['timestamp'].values.tolist(),
                    'correct_seq': (df['correct'].values).astype(np.int18),

                }



            user_wise_dict[cnt]['time_lag'] = np.array(
                [0] + list(map(lambda x: int(x[0] - x[1]),
                               zip(user_wise_dict[cnt]['timestamp_seq'][1:],
                                   user_wise_dict[cnt]['timestamp_seq'][:-1])))
            )

            del user_wise_dict[cnt]['timestamp_seq']
            cnt += 1
            n_inters += len(df)

        self.user_seq_df = pd.DataFrame.from_dict(user_wise_dict, orient='index')  # 将dict 转成 dataframe 格式
        self.get_information()


        del self.inter_df
        self.inter_df = []




    def get_information(self):
        self.info_dict = {}

        self.n_users = int(max(self.user_seq_df['user_id'].values))
        self.n_examples = len(self.user_seq_df)

        self.n_skills = int(max(self.inter_df['skill_id']))
        self.n_problems = int(max(self.inter_df['problem_id']))

        self.info_dict = {'user_num': (False, self.n_users), # attribute: (flag of being used for model or not , number of items)
                          'n_examples': (False, self.n_examples),
                          'timestamp': (False, 1),

                          'time_lag_num': (True, 1),
                          'elapsed_time_num': (True, 1),
                          'problem_num':(True,self.n_problems),
                          }


        if "Ednet" in self.path:
            self.n_bundles = int(max(self.inter_df['bundle_id']))
            self.n_tags = int(max(self.inter_df['tags_id']))

            self.info_dict['tags_num'] = (True,self.n_tags)
            self.info_dict['bundle_num'] = (True,self.n_bundles)
            self.info_dict['tags_set_num'] = (True,302) # num 189, max 300

        elif 'Riid' in self.path:
            self.n_skills -= 1  # TODO 在preprocessing 多加了 1，这里需要减去
            self.n_tags = int(max(self.inter_df['tags_id']))

            self.info_dict['tags_num'] = (True,self.n_tags)
            self.info_dict['tags_set_num'] = (True,189)

            self.info_dict['lag_s_num'] = (True,300)
            self.info_dict['lag_m_num'] = (True,1440)
            self.info_dict['lag_d_num'] = (True,365)

            self.info_dict['elapsed_num'] = (True,300)

            self.info_dict['explanation_num'] = (True,2)
            self.info_dict['difficulty_num'] = (True,1)


        self.info_dict['skill_num'] = (True,self.n_skills)

    def save_info(self,dir):
        info_json = json.dumps(self.info_dict,sort_keys=False,indent=4,separators=(',',':'))
        print(info_json)
        f = open('{}/info.json'.format(dir),'w')
        f.write(info_json)

        pickle_file = open(dir + '/attribute.columns', 'wb')
        pickle.dump(self.user_seq_df.columns,pickle_file)
        pickle_file.close()



    def get_info_dict(self):
        return self.info_dict

    def show_columns(self):
        return self.user_seq_df.columns

    def get_fold_position(self):
        # fold_begin = list()
        self.sequence = np.random.permutation(self.n_examples)

        fold_size = int(self.n_examples/self.k_fold)
        fold_begin = [i * fold_size for i in range(self.k_fold)]
        fold_end = [(i+1) * fold_size for i in range(self.k_fold)]
        fold_end[-1] = self.n_examples
        # fold_infor = {'fold_begin': fold_begin,
        #               'fold_end': fold_end}
        return fold_begin, fold_end

    def devide_data(self, fold_begin_num, fold_end_num, style='student'):
        assert style == 'time' or style == 'student'
        self.style = style
        if style == 'time':
            print("'style='time' 无法做k折交叉验证，未配置！")
        elif style == 'student':

            self.data_df['test'] = self.user_seq_df.iloc[self.sequence[fold_begin_num: fold_end_num]]
            self.data_df['test'].index = range(len(self.data_df['test']))

            residual_df = pd.concat([self.user_seq_df.iloc[self.sequence[0: fold_begin_num]], self.user_seq_df.iloc[self.sequence[fold_end_num:self.n_examples]]])
            # dev_size = int(0.1 * len(residual_df))  # 验证集是剩下 4 折的 10%
            # dev_indices = np.random.choice(residual_df.index, dev_size, replace=False)
            # # random 验证集在剩下 k-1 折里面 随机 取了 dev_size, replace=False 代表抽样之后不放回
            # self.data_df['val'] = self.user_seq_df.iloc[dev_indices]
            # self.data_df['val'].index = range(len(self.data_df['val']))
            # self.data_df['train'] = residual_df.drop(dev_indices)
            self.data_df['train'] = residual_df
            self.data_df['train'].index = range(len(self.data_df['train']))



def Devide_Fold_and_Save(settings):
    dataset = settings.dataset
    max_length = settings.max_length
    path = settings.data_dir + '/' + dataset + '/' + 'interactions.csv'
    cross_validation = settings.cross_validation

    if dataset=='Riid':
        path = settings.data_dir + '/' + dataset + '/' + 'interactions.csv'

    if cross_validation == False:
        print("prepare data for no cross validation...")
        data = Datareader(path=path, max_length=max_length)
        data.devide_data(style='student')
        save_path = 'data/' + dataset + '/data_information_' + str(data.max_length) + '.pkl'
        pickle_file = open(save_path, 'wb')
        pickle.dump(data, pickle_file)
        pickle_file.close()

    elif cross_validation == True:
        assert settings.k_fold >= 1
        print("prepare data for {}-fold cross validation...".format(settings.k_fold))
        Data = Datareader_fold(path=path, max_length=max_length)
        Data.save_info( settings.data_dir + '/' + dataset)

        for count in tqdm(range(settings.k_fold)):
            data = copy.copy(Data)
            # data = Datareader_fold(path=path, max_length=max_length)

            fold_begin, fold_end = data.get_fold_position()
            data.devide_data(fold_begin_num=fold_begin[count], fold_end_num=fold_end[count])
            save_path =  settings.data_dir+'/' + dataset + '/data_information_' + str(data.max_length) + '_' + str(count) + '.pkl'

            pickle_file = open(save_path[:-4]+'.train', 'wb')
            pickle.dump(data.data_df['train'].values,pickle_file)
            pickle_file.close()

            pickle_file = open(save_path[:-4]+'.test', 'wb')
            pickle.dump(data.data_df['test'].values,pickle_file)
            pickle_file.close()




