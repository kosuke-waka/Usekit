#!/usr/bin/env python
# coding: utf-8

# In[41]:


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib
import numpy as np

from collections import defaultdict

from sklearn.metrics import accuracy_score,f1_score,recall_score,precision_score
from sklearn.metrics import confusion_matrix,roc_curve, roc_auc_score, precision_recall_curve
from sklearn.model_selection import KFold

import torch

from transformers import AutoTokenizer, AutoModel
from transformers import AutoModelForSequenceClassification
from transformers import DefaultDataCollator
from transformers import Trainer, TrainingArguments

from datasets import load_dataset
from datasets import Dataset
from datasets import DatasetDict
import evaluate

from MultiCrossInitialParameter1 import InitialParameter
parameter = InitialParameter()
parameter.training_args.num_train_epochs = 5
parameter





def processing(example):
    tokenized = tokenizer(example[parameter.text_column_name], max_length=parameter.max_length, truncation=True, padding="max_length")
    judge_len = len(example[parameter.text_column_name])
    label_list = list(example.keys())[1:-1] #ツイート列と最後の列を除外して、他の列名のリストを取得
    #print(label_list)
    
    labels = []
    #print(len(example[parameter.label_list[1]]))
    for idx in range(judge_len):
        dummy_label = []
        for label_ in label_list:
            dummy_label.append(float(example[label_][idx])) 
        labels.append(dummy_label)
    tokenized['labels'] = labels

    return tokenized


#Best_thresholds = list(prepre['test_best_thresholds'])

def compute_metrics(pred):
    labels = pred.label_ids
    probability = pred.predictions
    preds = np.where(pred.predictions > 0, 1, 0)
    result_dict = defaultdict(list)
    
    f1 = f1_score(labels, preds,average="macro")
    acc = accuracy_score(labels, preds)
    precision = precision_score(labels, preds,average="macro")
    recall = recall_score(labels, preds, average="macro")
    auc = roc_auc_score(labels ,probability, average="macro")
    #結果ラベル全体を考慮した性能を返す
    return {"accuracy":acc,"precision":precision,'recall': recall, "f1":f1,
            'auc':auc}
def result_metrics(pred):
    labels = pred.label_ids
    probability = pred.predictions
    preds = np.where(pred.predictions > 0, 1, 0)
    result_dict = defaultdict(list)
    
    for idx, label_ in enumerate(parameter.label_list):
        fpr, tpr, thresholds = roc_curve(labels[:, idx], probability[:, idx])
        
        f1 = f1_score(labels[:, idx], preds[:, idx])
        acc = accuracy_score(labels[:, idx], preds[:, idx])
        precision = precision_score(labels[:, idx], preds[:, idx])
        recall = recall_score(labels[:, idx], preds[:, idx])
        auc = roc_auc_score(labels[:, idx],probability[:, idx])
        result = {'label':label_, "accuracy":acc, "precision":precision, 'recall': recall, "f1":f1,'auc':auc}
        for key, value in result.items():
            result_dict[key].append(value)
    return result_dict 

def metrics_only_dict(met_dict):
    m_d_l = defaultdict(list)
    for k, v in met_dict.items():
        m_d_l[k].append(v)
    return m_d_l

def metrics_list_dict(met_list_dict):
    m_d_l = defaultdict(list)
    for lis in met_list_dict:
        for k, v in lis.items():
            m_d_l[k].append(v)
    return m_d_l

data_collator = DefaultDataCollator()
train_metrics = []
val_metrics = []
all_result_pd = pd.DataFrame()

crross_num = parameter.training_args.num_train_epochs
now = 0
for trafile,tesfile, savefile, use_tokenize in zip(parameter.train_list, parameter.test_list, parameter.save_list, parameter.use_model_list):
    
    print("学習ファイルの読み込み： ", os.path.join(parameter.train_data_dir, trafile))
    train_pd = pd.read_excel(os.path.join(parameter.train_data_dir, trafile))    
    print("テストファイルの読み込み： ", os.path.join(parameter.train_data_dir, trafile))
    test_pd = pd.read_excel(os.path.join(parameter.test_data_dir, tesfile))
    #Dataset化
    train_data = Dataset.from_pandas(train_pd)
    test_data = Dataset.from_pandas(test_pd)
    
    dataset = DatasetDict({
        "train": train_data,
        "test": test_data,
    })
    
    tokenizer = AutoTokenizer.from_pretrained(use_tokenize)
    model = AutoModelForSequenceClassification.from_pretrained(use_tokenize,num_labels=len(parameter.label_list),problem_type = "multi_label_classification").to(parameter.device)
    
    dataset_tokenized = dataset.map(processing,batched=True)
    dataset_tokenized = dataset_tokenized.remove_columns(list(train_pd.columns))
    dataset_tokenized.set_format("torch")
    
    dataset_train = np.array([{k:v.to(parameter.device) for k, v in data.items()} for data in dataset_tokenized['train']])
    dataset_test = np.array([{k:v.to(parameter.device) for k, v in data.items()} for data in dataset_tokenized['test']])

    tmp_train_metrics = []
    tmp_val_metrics = []
    parameter.training_args.num_train_epochs = 1
    for idx in range(crross_num):
        tmp_train_list = list(train_pd[train_pd['implimention_num'] != idx].index)
        tmp_test_list = list(test_pd[test_pd['implimention_num'] == idx].index)
        
        #全体の学習データとテストデータから交差検証用のデータを取得する
        tmp_train_dataset = dataset_train[tmp_train_list]
        tmp_test_dataset = dataset_test[tmp_test_list]
        
        #学習開始
        trainer = Trainer(
            model=model,
            data_collator = data_collator,
            args=parameter.training_args,
            train_dataset=tmp_train_dataset,
            eval_dataset=tmp_test_dataset,
            compute_metrics=compute_metrics,
        )
        trainer.train()
        
        tmp_train_dict = trainer.predict(tmp_train_dataset).metrics
        tmp_train_dict['model'] = now
        tmp_train_metrics.append(tmp_train_dict)
        train_metrics.append(tmp_train_dict) #全体の学習データ評価を保存
        
        tmp_val_dict = trainer.predict(tmp_test_dataset).metrics
        tmp_val_dict['model'] = now
        tmp_val_metrics.append(tmp_val_dict)
        val_metrics.append(tmp_val_dict) #全体の検証データの評価を保存

        #各ラベル毎の評価を保存
        result_pd = trainer.predict(tmp_test_dataset)
        result_pd = result_metrics(result_pd)
        result_pd['model'] = now
        result_pd = pd.DataFrame.from_dict(result_pd)
        all_result_pd = pd.concat([all_result_pd, result_pd])        
    print("学習終了")
    trainer.save_model(os.path.join(parameter.save_dir, f"{savefile}"))

    now += 1
    
savefile = os.path.join(parameter.save_dir, parameter.metrics_model_result)
print(savefile)
train_result_pd = pd.DataFrame.from_dict(metrics_list_dict(train_metrics)).set_index('model')
val_result_pd = pd.DataFrame.from_dict(metrics_list_dict(val_metrics)).set_index('model')
with pd.ExcelWriter(savefile) as writer:
    train_result_pd.to_excel(writer, sheet_name='train')
    val_result_pd.to_excel(writer, sheet_name='val')
    all_result_pd.to_excel(writer, sheet_name='val_result', index=None)


# # 単体モデル分類一括実行


from OneCrossInitialParameter import InitialParameter

parameter = InitialParameter()
parameter.training_args.num_train_epochs = 5
parameter
crross_num=5


# In[61]:


def processing_one(example):
    tokenized = tokenizer(example[parameter.text_column_name], max_length=parameter.max_length, truncation=True, padding="max_length")
    judge_len = len(example[parameter.text_column_name])
    label_list = list(example.keys())[1:-1] #ツイート列を除外して、他の列名のリストを取得
    
    labels = []
    #print(len(example[parameter.label_list[1]]))
    for idx in range(judge_len):
        dummy_label = []
        for label_ in label_list:
            dummy_label.append(int(example[label_][idx]))
        labels.append(dummy_label)
    tokenized['labels'] = labels
    #print(tokenized['labels'][1:10])
    return tokenized

from sklearn.metrics import accuracy_score,f1_score,recall_score,precision_score
from sklearn.metrics import confusion_matrix,roc_curve, roc_auc_score

def softmax(x):
    f = np.exp(x)/np.sum(np.exp(x), axis = 1, keepdims = True)
    return f

#ネガティブの再現率
def specificity_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).flatten()
    return tn / (tn + fp)

#陰性適中率
#ネガティブの適合率
def negativePredictive(y_true,y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).flatten()
    return tn / (tn + fn)    

def compute_metrics(pred):
    labels = pred.label_ids
    probability = pred.predictions[:,1]
    preds = pred.predictions.argmax(-1)
    f1 = f1_score(labels,preds)
    acc = accuracy_score(labels,preds)
    precision = precision_score(labels,preds)
    recall = recall_score(labels,preds)
    
    specificity = specificity_score(labels,preds)
    negative_pre = negativePredictive(labels,preds)
    f1_pre = 2 * negative_pre * specificity / (negative_pre + specificity)
    
    
    auc = roc_auc_score(labels,probability)
    return {"accuracy":acc,"precision":precision,'recall': recall, "f1":f1,
            'negative_precision ':negative_pre,'negative_recall':specificity,'negative_f1':f1_pre,
            'auc':auc}

def metrics_only_dict(met_dict):
    m_d_l = defaultdict(list)
    for k, v in met_dict.items():
        m_d_l[k].append(v)
    return m_d_l

def metrics_list_dict(met_list_dict):
    m_d_l = defaultdict(list)
    for lis in met_list_dict:
        for k, v in lis.items():
            m_d_l[k].append(v)
    return m_d_l

train_metrics = []
val_metrics = []
data_collator = DefaultDataCollator()

for trafile, tesfile, savfile, use_tokenize, label in zip(parameter.train_list[0:], parameter.test_list[0:], parameter.save_list[0:], parameter.use_model_list[0:], parameter.label_list[0:]):
    print(label)
    train_pd = pd.read_excel(os.path.join(parameter.train_data_dir, trafile))
    train_pd = train_pd.loc[:, [parameter.text_column_name, label, "implimention_num"]] #特定の列のみ取得
    test_pd = pd.read_excel(os.path.join(parameter.test_data_dir, tesfile))
    test_pd = test_pd.loc[:, [parameter.text_column_name, label, "implimention_num"]]    #特定の列のみ取得
    #Dataset化
    train_dataset = Dataset.from_pandas(train_pd)
    test_dataset = Dataset.from_pandas(test_pd)
    print(list(train_pd.columns))
    dataset = DatasetDict({
        "train": train_dataset,
        "test": test_dataset,
    })
    
    print("学習データ：", os.path.join(parameter.train_data_dir, trafile))
    print("テストデータ：", os.path.join(parameter.test_data_dir, tesfile))
    
    tokenizer = AutoTokenizer.from_pretrained(use_tokenize)
    model = AutoModelForSequenceClassification.from_pretrained(use_tokenize,num_labels=2).to(parameter.device)
    
    try:
        dataset_tokenized = dataset.map(processing_one, batched=True)
    except:
        print("エラー発生")
        print(label)
        print("学習データ：", os.path.join(parameter.train_data_dir, trafile))
        print("テストデータ：", os.path.join(parameter.test_data_dir, tesfile))
        break
    dataset_tokenized = dataset_tokenized.remove_columns(list(train_pd.columns))
    dataset_tokenized.set_format("torch")
    
    dataset_train = np.array([{k:v.to(parameter.device) for k, v in data.items()} for data in dataset_tokenized['train']])
    dataset_test = np.array([{k:v.to(parameter.device) for k, v in data.items()} for data in dataset_tokenized['test']])
    
    parameter.training_args.num_train_epochs = 1
    for idx in range(crross_num):
        tmp_train_list = train_pd[train_pd['implimention_num'] != idx].index
        tmp_test_list = test_pd[test_pd['implimention_num'] == idx].index
        
        #全体の学習データとテストデータから交差検証用のデータを取得する
        tmp_train = dataset_train[tmp_train_list]
        tmp_test = dataset_test[tmp_test_list]
        
        #学習開始
        print(f"学習中:{label}")
        trainer = Trainer(
            model=model,
            data_collator = data_collator,
            args=parameter.training_args,
            train_dataset=tmp_train,
            eval_dataset=tmp_test,
            compute_metrics=compute_metrics,
        )
        trainer.train()
        
        tmp_train_dict = trainer.predict(tmp_train).metrics
        tmp_train_dict['model'] = label
        train_metrics.append(tmp_train_dict) #全体の学習データ評価を保存
        
        tmp_val_dict = trainer.predict(tmp_test).metrics
        tmp_val_dict['model'] = label
        val_metrics.append(tmp_val_dict) #全体の検証データの評価を保存
    print("学習終了")
    trainer.save_model(os.path.join(parameter.save_dir, f"{savfile}"))



savefile = os.path.join(parameter.save_dir, parameter.metrics_model_result)

train_result_pd = pd.DataFrame.from_dict(metrics_list_dict(train_metrics)).set_index('model')
val_result_pd = pd.DataFrame.from_dict(metrics_list_dict(val_metrics)).set_index('model')
with pd.ExcelWriter(savefile) as writer:
    train_result_pd.to_excel(writer, sheet_name='train')
    val_result_pd.to_excel(writer, sheet_name='val')




