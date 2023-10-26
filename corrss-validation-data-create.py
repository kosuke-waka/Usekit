#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd
import  matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold


import sys
sys.path.append('./mymodelkit/')
from mymodelkit import TextDataAugmentation, jumanTextAlignment



from OneCrossInitialParameter1 import InitialParameter
parameter = InitialParameter()
AUGMENT_TRUE = False #データ拡張を行うのか
crross_num = parameter.training_args.num_train_epochs


ReadFile = './original/sokushinラベル確定_result.xlsx'

all_data_pd = pd.read_excel(ReadFile)
all_data_pd = all_data_pd[all_data_pd['判定結果'] != 'その他']
all_data_pd.reset_index(drop=True,inplace=True)
all_data_pd




#データのラベル構成確認
from collections import defaultdict
result_dict = defaultdict(int)

for label in parameter.label_list:
    result_dict[label] = 0
    
for idx, row in all_data_pd.iterrows():
    label_ = row[1]
    #if '|' in label_:
    #    result_dict[label_] += 1 #ラベルのタプルを数えるため
    #label_ = label_.split('|')
    #for la in label_:
    #    result_dict[la] += 1
    result_dict[label_] += 1
    
result_dict



for label in parameter.label_list:
    all_data_pd[label] = 0

for idx, row in all_data_pd.iterrows():
    labels = row[1].split('|')
    for label in labels:
        all_data_pd.loc[idx, label] = 1
all_data_pd.drop('判定結果', axis=1, inplace=True)
all_data_pd.iloc[50:100, :]


metrics_dict = defaultdict(list)
all_train = pd.DataFrame()
all_test = pd.DataFrame()
for label, trafile,tesfile in zip(parameter.label_list[0:], parameter.train_list[0:], parameter.test_list[0:]): 
    print(label)
    #対象のデータ取得
    correct_pd = all_data_pd[ (all_data_pd[label] == 1)]
    no_correct_pd = all_data_pd[ (all_data_pd[label] == 0)]
    
    #correct_pdとno_correct_pdの行数の最小値を取得する（アンダーサンプリングのために）
    min_value = min(len(correct_pd), len(no_correct_pd))
    
    #アンダーサンプリング
    correct_pd = correct_pd.sample(min_value)
    no_correct_pd = no_correct_pd.sample(min_value)
    
    correct_pd['implimention_num'] = -1
    no_correct_pd['implimention_num'] = -1
    
    #交差検証データ作成
    kf = KFold(n_splits=crross_num, shuffle=True, random_state=1)
    idx = 0
    for train_list, test_list in kf.split(correct_pd):
        #交差検証用の番号を割り当てる
        correct_pd.iloc[test_list, list(correct_pd.columns).index("implimention_num")] = idx
        no_correct_pd.iloc[test_list, list(correct_pd.columns).index("implimention_num")] = idx
        idx += 1
    #学習データとテストデータの結合
    tmp_train = pd.concat([correct_pd, no_correct_pd], axis=0, ignore_index=True)
    tmp_test = pd.concat([correct_pd, no_correct_pd], axis=0, ignore_index=True)
    
    print('train_len:', len(tmp_train))
    print('test_len:', len(tmp_test))

    #学習データ全てをテキスト拡張の対象とする
    if AUGMENT_TRUE:
        textaugmentation = TextDataAugmentation.TextAugmentation(parameter)
        train_aug = textaugmentation.run(tmp_train)
    else:
        train_aug = tmp_train

    #学習データとテストデータの分かち書きを行う
    jumantextalignment = jumanTextAlignment.jumanTextAlignment(parameter)
    print(label)
    print("学習データの分かち書き開始")
    juman_augmented_pd = jumantextalignment.run(train_aug)
    print("テストデータの分かち書き開始")
    test_juman_augmented_pd = jumantextalignment.run(tmp_test)
    
    metrics_dict['label'].append(label)
    metrics_dict['correct_pd'].append(len(correct_pd))
    metrics_dict['no_correct_pd'].append(len(no_correct_pd))
    metrics_dict['tmp_train'].append(len(tmp_train))
    metrics_dict['tmp_test'].append(len(tmp_test))
    metrics_dict['train_correct'].append(len(tmp_train[(tmp_train[label]==1) & (tmp_train["implimention_num"]!=0)]))
    metrics_dict['train_no_correct'].append(len(tmp_train[(tmp_train[label]==0) & (tmp_train["implimention_num"]!=0)]))
    metrics_dict['test_correct'].append(len(tmp_test[(tmp_test[label]==1) & (tmp_test["implimention_num"]==0)]))
    metrics_dict['test_no_correct'].append(len(tmp_test[(tmp_test[label]==0)& (tmp_test["implimention_num"]==0)]))
    metrics_dict['train_correct_augmentation'].append(len(train_aug[(train_aug[label]==1) & (train_aug["implimention_num"]!=0)]))
    metrics_dict['train_no_correct_augmentation'].append(len(train_aug[(train_aug[label]==0) & (train_aug["implimention_num"]!=0)]))
    

    juman_augmented_pd.to_excel(os.path.join(parameter.train_data_dir, trafile), index=None)
    test_juman_augmented_pd.to_excel(os.path.join(parameter.test_data_dir, tesfile), index=None)
    
    #全データ結合
    all_train = pd.concat([all_train, tmp_train], axis=0, ignore_index=True)
    all_test = pd.concat([all_test, tmp_test], axis=0, ignore_index=True)




pd_ = pd.DataFrame.from_dict(metrics_dict)




pd_.to_excel(os.path.join(parameter.save_dir, parameter.metrics_data_result), index=None)


train_aug[(train_aug[label]==1) & (tmp_test["implimention_num"]==0)].index


train_aug[(train_aug[label]==1) & (tmp_test["implimention_num"]==0)]


all_train.to_excel(f'./data/tmp_traindata{parameter.metrics_data_result}', index=None)
all_test.to_excel(f'./data/tmp_testdata{parameter.metrics_data_result}', index=None)


# # データ結合
from OneCrossInitialParameter import InitialParameter
parameter = InitialParameter()
all_train = pd.read_excel(f'./data/tmp_traindata{parameter.metrics_data_result}')
all_test = pd.read_excel(f'./data/tmp_testdata{parameter.metrics_data_result}')


#################################################################################################################
from MultiCrossInitialParameter1 import InitialParameter

parameter = InitialParameter()
parameter
#################################################################################################################

# In[25]:


#データ拡張する前の学習データとテストデータを取得
duplicated_train = all_train.copy()
duplicated_train['implimention_num'] = -1
duplicated_test = all_test.copy()
duplicated_test['implimention_num'] = -1

#データの重複削除
duplicated_train = duplicated_train[[not target for target in duplicated_train.duplicated()]]
duplicated_train = duplicated_train.reset_index(drop=True)
duplicated_test = duplicated_test[[not target for target in duplicated_test.duplicated()]]
duplicated_test = duplicated_test.reset_index(drop=True)





#データのラベル構成確認
from collections import defaultdict
result_dict = defaultdict(int)

for label in parameter.label_list:
    result_dict[label] = 0
    
for idx, row in duplicated_train.iterrows():
    for label_ in  parameter.label_list:
        if row[label_] == 1:
            result_dict[label_] += 1
    

metrics_dict = defaultdict(list)
for trafile,tesfile in zip(parameter.train_list[0:], parameter.test_list[0:]): 
    
    #交差検証データ作成
    kf = KFold(n_splits=crross_num, shuffle=True, random_state=1)
    idx = 0
    for train_list, test_list in kf.split(duplicated_train):
        #交差検証用の番号を割り当てる
        duplicated_train.iloc[test_list, list(duplicated_train.columns).index("implimention_num")] = idx
        duplicated_test.iloc[test_list, list(duplicated_test.columns).index("implimention_num")] = idx
        idx += 1
    
    print('train_len:', len(duplicated_train))
    print('test_len:', len(duplicated_test))

    #学習データ全てをテキスト拡張の対象とする
    if AUGMENT_TRUE:
        textaugmentation = TextDataAugmentation.TextAugmentation(parameter)
        #train_aug = textaugmentation.run(duplicated_train)
        kwarg = {"alpha_sr":0.05, "alpha_ri":0.05, "alpha_rs":0.05, "p_rd":0.05, "num_aug":8}
        train_aug = textaugmentation.run(duplicated_train, kwarg)
    else:
        train_aug = duplicated_train #データ拡張を行わない場合に有効にする

    #学習データとテストデータの分かち書きを行う
    jumantextalignment = jumanTextAlignment.jumanTextAlignment(parameter)

    print("学習データの分かち書き開始")
    juman_augmented_pd = jumantextalignment.run(train_aug)
    print("テストデータの分かち書き開始")
    test_juman_augmented_pd = jumantextalignment.run(duplicated_test)
    
    metrics_dict['duplicated_train'].append(len(duplicated_train))
    metrics_dict['duplicated_test'].append(len(duplicated_test))
    metrics_dict['juman_augmented_pd'].append(len(juman_augmented_pd))
    metrics_dict['test_juman_augmented_pd'].append(len(test_juman_augmented_pd))

    juman_augmented_pd.to_excel(os.path.join(parameter.train_data_dir, trafile), index=None)
    test_juman_augmented_pd.to_excel(os.path.join(parameter.test_data_dir, tesfile), index=None)


pd_ = pd.DataFrame.from_dict(metrics_dict)


pd_.to_excel(os.path.join(parameter.save_dir, parameter.metrics_data_result), index=None)


"""# # メモ

# In[ ]:


metrics_dict = defaultdict(list)
train = pd.DataFrame()
test = pd.DataFrame()
for label, trafile,tesfile in zip(parameter.label_list[0:], parameter.train_list[0:], parameter.test_list[0:]): 
    print(label)
    #対象のデータ取得
    correct_pd = all_data_pd[ (all_data_pd[label] == 1)]
    no_correct_pd = all_data_pd[ (all_data_pd[label] == 0)]
    
    #correct_pdとno_correct_pdの行数の最小値を取得する（アンダーサンプリングのために）
    min_value = min(len(correct_pd), len(no_correct_pd))
    
    #アンダーサンプリング
    correct_pd = correct_pd.sample(min_value)
    no_correct_pd = no_correct_pd.sample(min_value)
    
    correct_pd['implimention_num'] = -1
    no_correct_pd['implimention_num'] = -1
    
    #交差検証データ作成
    tmp_train = pd.DataFrame()
    tmp_test = pd.DataFrame()
    kf = KFold(n_splits=5, shuffle=True, random_state=1)
    idx = 0
    for train, test in kf.split(correct_pd):
        #correct_pd(対象ラベルデータ)を学習データとテストデータに分割する
        tmp_train_correct = correct_pd.iloc[train, :]
        tmp_train_correct['implimention_num'] = idx #交差検証用の番号
        
        tmp_test_correct = correct_pd.iloc[test, :]
        tmp_test_correct['implimention_num'] = idx
        
        #no_correct_pd(非対象ラベルデータ)を学習データとテストデータに分割する
        tmp_train_no_correct = no_correct_pd.iloc[train, :]
        tmp_train_no_correct['implimention_num'] = idx
        
        tmp_test_no_correct = no_correct_pd.iloc[test, :]
        tmp_test_no_correct['implimention_num'] = idx
        
        #学習データ，テストデータの結合
        associate_train = pd.concat([tmp_train_correct, tmp_train_no_correct], axis=0)
        associate_test = pd.concat([tmp_test_correct, tmp_test_no_correct], axis=0)
        
        tmp_train = pd.concat([tmp_train, associate_train], axis=0)
        tmp_test = pd.concat([tmp_test, associate_test], axis=0)
        idx += 1
        

    print('train_len:', len(tmp_train))
    print('test_len:', len(tmp_test))
    textaugmentation = TextDataAugmentation.TextAugmentation(parameter) 

    #学習データ全てをテキスト拡張の対象とする
    train_aug = textaugmentation.run(tmp_train)
    
    metrics_dict['label'].append(label)
    metrics_dict['correct_pd'].append(len(correct_pd))
    metrics_dict['no_correct_pd'].append(len(no_correct_pd))
    metrics_dict['tmp_train_correct'].append(len(tmp_train_correct))
    metrics_dict['tmp_test_correct'].append(len(tmp_test_correct))
    metrics_dict['tmp_train_no_correct'].append(len(tmp_train_no_correct))
    metrics_dict['tmp_test_no_correct'].append(len(tmp_test_no_correct))
    metrics_dict['train_correct_augmentation'].append(len(train_aug[train_aug[label]==1]))
    metrics_dict['train_no_correct_augmentation'].append(len(train_aug[train_aug[label]==0]))
    
    #学習データとテストデータの分かち書きを行う
    jumantextalignment = jumanTextAlignment.jumanTextAlignment(parameter)
    print(label)
    print("学習データの分かち書き開始")
    juman_augmented_pd = jumantextalignment.run(train_aug)
    print("テストデータの分かち書き開始")
    test_juman_augmented_pd = jumantextalignment.run(test)
    
    juman_augmented_pd.to_excel(os.path.join(parameter.train_data_dir, trafile), index=None)
    test_juman_augmented_pd.to_excel(os.path.join(parameter.test_data_dir, tesfile), index=None)
    
    #for文内の全データを結合
    train = pd.concat([train, juman_augmented_pd], axis=0)
    test = pd.concat([test, test_juman_augmented_pd], axis=0)


# In[16]:


r = './data/train/all_undersampling_train.xlsx'
a = pd.read_excel(r)
a


# In[21]:


g = [idx  for idx, aa in enumerate(a.duplicated()) if aa==True]


# In[22]:


a.iloc[g,:]


# In[ ]:




"""