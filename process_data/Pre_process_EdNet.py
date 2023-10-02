

import pandas as pd
import numpy as np
import time
import os
from tqdm import tqdm
import time



def get_problem_information():

    problem_data = pd.read_csv('../EdNet-Contents/contents/questions.csv')
    # print(problem_data.head(3))
    # print("problem_data.cloumns:\n", problem_data.columns)
    #  Index(['question_id', 'bundle_id', 'explanation_id', 'correct_answer', 'part', 'tags', 'deployed_at'], dtype='object')
    return problem_data


# TODO 处理并获得问题对应的问题编码, 编码从 1 开始
def get_problem_index():
    problem_data = get_problem_information()
    print("problem columns:\n", problem_data.columns)
    question_id = problem_data[['question_id']].drop_duplicates()
    print("type of question_id", type(question_id))
    print("len of question_id", len(question_id))
    question_id['question_index'] = question_id.index + 1
    question_id.to_csv('question_id_index.csv')
    # print(question_id.head(7))

def bundle_id_index():
    problem_data = get_problem_information()
    bundle_id = problem_data[['bundle_id']].drop_duplicates()
    print("type of bundle_id", type(bundle_id))
    print("len of bundle_id", len(bundle_id))
    bundle_id['bundle_index'] = np.arange(len(bundle_id))+1  # np.arange(len(bundle_id))+1 bundle_id.index + 1
    bundle_id.to_csv('bundle_id_index.csv')

def tags_id_index():
    problem_data = get_problem_information()
    tags = problem_data[['tags']].drop_duplicates()
    print("type of tags", type(tags))
    print("len of tags", len(tags))
    tags['tags_id'] = np.arange(len(tags)) + 1  # tags.index+1
    tags.to_csv('tags_index.csv')

#-------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------

# TODO  把每个学生的 csv 里面的放进一个大文件
def get_user_information():

    filepath = '../EdNet-KT1/KT1'
    sub_files = list()
    for root, dirs, files in os.walk(filepath):
        sub_files = files
    name = [i.split('.')[0] for i in sub_files]
    name_index = [int(i.split('u')[1]) for i in name]
    name_index.sort()
    sub_files = ['u'+str(i)+'.csv' for i in name_index]
    user_file_paths = [filepath + '/' + i for i in sub_files]
    user_informations = pd.DataFrame()
    user_count = 0
    csv_count = 0

    for index, user_file_path in enumerate(tqdm(user_file_paths)):
        user_information = pd.read_csv(user_file_path)
        user_information_select = user_information[['elapsed_time','timestamp', 'question_id', 'user_answer']]
        user_information_select = user_information_select.dropna(axis=0, how='any')  # TODO 删除有缺失值的行
        if len(user_information_select) == 0:
            continue
        user_information_select.loc[:, 'user_id'] = [user_count for _ in range(len(user_information_select))]
        user_count += 1
        if len(user_informations) == 0:
            user_informations = user_information_select
        else:
            user_informations = pd.concat([user_informations, user_information_select], ignore_index=True)
        
        if user_count % 3000 == 0:
            path = 'user_informations/user_informations_' + str(csv_count) + '.csv'
            user_informations.to_csv(path)
            user_informations = pd.DataFrame()
            csv_count += 1

    path = 'user_informations/user_informations_' + str(csv_count) + '.csv'
    user_informations.to_csv(path)
    csv_count += 1


# TODO  将上面大文件中的内容再合并到一个 总的 csv 文件当中
def hebing_user_information():

    filepath = 'user_informations'
    sub_files = list()
    for root, dirs, files in os.walk(filepath):
        sub_files = files
    user_file_paths = [filepath + '/' + i for i in sub_files]
    user_informations = pd.DataFrame()

    for index, user_file_path in enumerate(tqdm(user_file_paths)):

        user_information = pd.read_csv(user_file_path)
        user_information_select = user_information[['user_id', 'elapsed_time','timestamp', 'question_id', 'user_answer']]
        user_information_select = user_information_select.dropna(axis=0, how='any')  # TODO 删除有缺失值的行
        if len(user_information_select) == 0:
            continue
        if len(user_informations) == 0:
            user_informations = user_information_select
        else:
            user_informations = pd.concat([user_informations, user_information_select], ignore_index=True)

    user_informations.to_csv("concat_user_informations.csv")

# TODO 获得 question.csv 中的信息

def process_problem():
    get_problem_index()
    bundle_id_index()
    tags_id_index()




    problem_data = get_problem_information()
    print(len(problem_data['tags'].drop_duplicates()))

    tags = pd.read_csv('tags_index.csv')
    bundle_id = pd.read_csv('bundle_id_index.csv')
    question_index = pd.read_csv('question_id_index.csv')

    interactions = pd.merge(right=problem_data, left=question_index, on='question_id')
    interactions.to_csv('problem/1.csv')

    interactions= pd.merge(right=interactions, left=tags, on='tags')
    interactions.to_csv('problem/2.csv')
    print(len(tags['tags']))

    interactions= pd.merge(right=interactions, left=bundle_id, on='bundle_id')
    interactions.to_csv('problem/3.csv')
    print(len(tags['tags']))

    interactions = interactions[['question_id','question_index','part','correct_answer','tags_id','tags','bundle_index']]
    interactions.to_csv('problem/new_problem.csv')





# TODO 这一步中，我们将 concat_user_informations.csv 中的内容和 question.csv 中内容依据 键 'question_id' 联结起来
def user_question_concat():
    problem_data = pd.read_csv('problem/new_problem.csv')
    print(len(problem_data['tags'].drop_duplicates()))
    print(len(problem_data['question_id'].drop_duplicates()))

    user_data = pd.read_csv('concat_user_informations.csv')
    print('user_data.columns :', user_data.columns)
    print("problem_data.columns: ", problem_data.columns)
    user_question = pd.merge(right=user_data, left=problem_data, on='question_id')
    user_question = user_question[['user_id','elapsed_time', 'timestamp', 'user_answer',
                                   'question_id','question_index','part','correct_answer','tags_id','tags','bundle_index']]
    # [['question_id','question_index','part','correct_answer','tags_id','tags','bundle_index']]

    print(len(user_question['tags'].drop_duplicates()))
    print(len(user_question['question_id'].drop_duplicates()))

    if len(user_question['tags'].drop_duplicates()) <= len(problem_data['tags'].drop_duplicates()):
        user_question.to_csv('user_question_concat.csv')
    user_question.head().to_csv('user_question_concat_head.csv')

    # 1792
    # 13169
    # 1785
    # 12283

# # TODO 这一步获取得到的 user_answer 和 correct_answer 得到的 correct
# def append_correct():
#     data = pd.read_csv('user_question_concat.csv')
    data = user_question
    data['correct'] = (data['user_answer'] == data['correct_answer'])
    print(len(data['tags'].drop_duplicates()))
    print(len(data['question_id'].drop_duplicates()))
    data.to_csv('append_correct.csv')


# # TODO 挑选出合适的列
# def select_append_correct():
#     data = pd.read_csv('append_correct.csv')


    data_select = data[['user_id', 'elapsed_time','timestamp',
                        'question_id', 'question_index', 'part', 'correct', 'tags_id', 'tags', 'bundle_index'  ]]
    print(len(data_select['tags'].drop_duplicates()))
    print(len(data_select['question_id'].drop_duplicates()))

    data_select.to_csv('interaction_all.csv')








# TODO 将记录数低于 threshold 的学生去掉
def select_student(threshold):

    save_path = 'interaction_' + str(threshold) + '.csv'
    data = pd.read_csv('interaction_all.csv')
    print("len of data ", len(data))

    user_id = data[['user_id']].drop_duplicates()
    print("len of user_id", len(user_id))

    need_to_drop = pd.DataFrame()
    for index, value in tqdm(data.groupby('user_id')):
        # print('index ', index)
        if len(value) <= threshold:
        # if len(value) <= threshold or (value['tags']==-1).sum()>0:
            if len(need_to_drop) == 0:
                need_to_drop = value
            else:
                need_to_drop = pd.concat([need_to_drop, value])

    need_to_drop.to_csv('need_to_drop.csv')

    print("len of data ", len(data))
    print('len of need_to_drop ', len(need_to_drop))
    print("正在删去行...")
    data.drop(data.index[need_to_drop.index], inplace=True)
    data.index = [i for i in range(len(data))]
    data.to_csv(save_path)

def select_student_based_on_dropcsv(threshold=10):

    save_path = 'interaction_' + str(threshold) + '.csv'
    need_to_drop = pd.read_csv('need_to_drop.csv')
    data = pd.read_csv('interaction_all.csv')
    print("len of data ", len(data))
    print(len(data[['tags']].drop_duplicates()))
    print(len(data['question_id'].drop_duplicates()))

    print(len(data[['user_id']].drop_duplicates()) - len(need_to_drop['user_id'].drop_duplicates()))

    data = data[['user_id', 'elapsed_time','timestamp',
                        'question_id', 'question_index', 'part', 'correct', 'tags_id', 'tags', 'bundle_index'  ]]
    need_to_drop = need_to_drop[['user_id','elapsed_time','timestamp','question_id','part','correct']]

    interactions = pd.concat([data,need_to_drop],sort=False)
    print(len(data)+len(need_to_drop))
    print(len(interactions))
    interactions.drop_duplicates(subset=['user_id','elapsed_time','timestamp','question_id','part','correct'],keep=False,inplace=True)
    print(len(interactions))

    # data.drop(data.index[need_to_drop.index], inplace=True)

    data.index = [i for i in range(len(data))]
    print('after drop')
    print("len of data ", len(data))
    print(len(data[['tags']].drop_duplicates()))
    print(len(data['question_id'].drop_duplicates()))

    user_id = data[['user_id']].drop_duplicates()
    print("type of user_id", type(user_id))
    print("len of user_id", len(user_id))
    # data.to_csv(save_path)












# TODO 获得可用于代码使用的、未进行学生随机取出但进行过学生答题数过滤的学生
def get_interactions():
    interactions = pd.read_csv('interaction_10.csv')
    print("已获得数据集，正在处理...")
    interactions = interactions[['user_id', 'elapsed_time', 'timestamp',
      'question_index', 'part', 'correct', 'bundle_index','tags_id', 'tags' ]]
    new_columns = ['user_id', 'elapsed_time','timestamp', 'problem_id', 'skill_id', 'correct','bundle_id','tags_id', 'tags']
    interactions.columns = new_columns
    print("interactions.columns:\n", interactions.columns)
    interactions.to_csv('Ednet/interactions_norandom.csv')
    # interactions.head().to_csv('Ednet/interactions_head.csv')




def randperm_interactions():
    load_path = 'Ednet/interactions_norandom.csv'
    user_data = pd.read_csv(load_path)


    user_id = user_data[['user_id']].drop_duplicates()
    print("type of user_id", type(user_id))
    print("len of user_id", len(user_id))
    randperm_user_id = np.random.permutation(len(user_id))
    user_id['randperm_user_id'] = randperm_user_id

    interactions = pd.merge(right=user_data, left=user_id, on='user_id')
    # interactions.to_csv('Ednet/interactions_tags_no_random.csv')
    # interactions.head().to_csv('Ednet/interactions_tags(no_drop-1)_no_random_head.csv')

    print("拼接完毕，处理中...")
    interactions = interactions[
        ['randperm_user_id', 'elapsed_time','timestamp', 'problem_id', 'skill_id', 'correct','bundle_id','tags_id', 'tags']]
    new_columns = ['user_id', 'elapsed_time','timestamp', 'problem_id', 'skill_id', 'correct','bundle_id','tags_id', 'tags']
    interactions.columns = new_columns

    interactions.to_csv('Ednet/interactions.csv')
    # interactions.head().to_csv('Ednet/interactions_head.csv')






def get_interactions_complete():
    interactions = pd.read_csv('interaction_all.csv')
    print("已获得数据集，正在处理...")
    interactions = interactions[['user_id', 'elapsed_time', 'timestamp',
                                 'question_index', 'part', 'correct', 'bundle_index','tags_id', 'tags' ]]
    new_columns = ['user_id', 'elapsed_time','timestamp', 'problem_id', 'skill_id', 'correct','bundle_id','tags_id', 'tags']
    interactions.columns = new_columns
    print("interactions.columns:\n", interactions.columns)
    interactions.to_csv('complete_Ednet/interactions_norandom.csv')
    # interactions.head().to_csv('Ednet/interactions_head.csv')

def randperm_interactions_complete():
    load_path = 'complete_Ednet/interactions_norandom.csv'
    user_data = pd.read_csv(load_path)


    user_id = user_data[['user_id']].drop_duplicates()
    print("type of user_id", type(user_id))
    print("len of user_id", len(user_id))
    randperm_user_id = np.random.permutation(len(user_id))
    user_id['randperm_user_id'] = randperm_user_id

    interactions = pd.merge(right=user_data, left=user_id, on='user_id')
    # interactions.to_csv('Ednet/interactions_tags_no_random.csv')
    # interactions.head().to_csv('Ednet/interactions_tags(no_drop-1)_no_random_head.csv')

    print("拼接完毕，处理中...")
    interactions = interactions[
        ['randperm_user_id', 'elapsed_time','timestamp', 'problem_id', 'skill_id', 'correct','bundle_id','tags_id', 'tags']]
    new_columns = ['user_id', 'elapsed_time','timestamp', 'problem_id', 'skill_id', 'correct','bundle_id','tags_id', 'tags']
    interactions.columns = new_columns

    interactions.to_csv('complete_Ednet/interactions.csv')

    
    










    






# TODO 这数据太多了，用的时候截一点儿吧
if __name__== "__main__":
    get_user_information()   # 把每个学生的 csv 里面的放进一个大文件
    hebing_user_information()  # 将上面大文件中的内容再合并到一个 总的 csv 文件当中
    get_problem_information()  # 获得 question.csv 中的信息
    user_question_concat()  # 将 concat_user_informations.csv 中的内容和 question.csv 中内容依据 键 'question_id' 联结起来
    # append_correct()  # 这一步获取得到的 user_answer 和 correct_answer 得到的 correct
    # select_append_correct()

    select_student(threshold=10)
    # select_student_based_on_dropcsv()




    get_interactions()

    randperm_interactions()


#--------------------------threshold >10
    # get_user_information()   # 把每个学生的 csv 里面的放进一个大文件
    # hebing_user_information()
    #
    # process_problem()
    # user_question_concat()

    # select_student_based_on_dropcsv()
    # select_student(threshold=10)
    #
    # get_interactions()

    # randperm_interactions()
    
#------------------------complete Ednet

    # get_user_information()   # 把每个学生的 csv 里面的放进一个大文件
    # hebing_user_information()
    #
    # process_problem()
    # user_question_concat()
    get_interactions_complete()
    randperm_interactions_complete()

