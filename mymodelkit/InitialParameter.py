import os
import torch
from transformers import Trainer, TrainingArguments 
from dataclasses import dataclass, field

@dataclass
class InitialParameter:
    """
    implemnt_lenは学習データを何個使うのかを設定している。
    #学習データのディレクトリを指定して、そのディレクトリにあるどのファイルを読み込むのかを指定する
    train_listの順番は重要、理由はtrain_listは学習をに利用するファイルが格納されている、
    一方test_listはテストに利用するファイルが格納されており、それぞれの要素の順番ごとにモデルの学習データとそのテストとして利用するから
    save_dirも同様の理由で、学習済みモデルやその他の結果を保存するディレクトリを指定する
    implment_lenの数は train_list, test_listとsave_listと同じでなくてはならない
    """
    implemnt_len: int = 1
    train_data_dir: str = './data'
    train_list: list[str] = field(default_factory=lambda: ['dummy_multi_train.xlsx'])

    test_data_dir: str = './data'
    test_list: list[str] = field(default_factory=lambda: ['dummy_multi_test.xlsx'])
    
    save_dir: str = os.path.join(os.getcwd(), 'result')
    save_list: list[str] = field(default=list)
        
    use_model_list: list[str] = field(default_factory=lambda: ['nlp-waseda/roberta-base-japanese'])
        
    text_column_name: str = 'ツイート'
    label_column_name: str = '判定結果'
    label_list: list[str] = field(default_factory=lambda: ['推奨', '行動抑制', '励まし', '願望'])
    
    #ラベルを日本語から英語へ
    JaToEn_label: list[str] = field(default_factory=lambda: {'推奨':'recommend', '行動抑制':'suppression', '励まし': 'encouragement', '願望': 'desire'})
    #列を日本語から英語へ
    JaToEn_column : list[str] = field(default_factory=lambda: {'ツイート': 'tweet', '判定結果': 'label'})
        
    device = "cuda:0" if torch.cuda.is_available() else 'cpu'

    max_length: int = 256
    training_args: TrainingArguments = TrainingArguments(
        output_dir=save_dir,
        num_train_epochs=5,
        evaluation_strategy="epoch",
        save_strategy='epoch',
        learning_rate=5e-6,
        dataloader_pin_memory=False,
        weight_decay=0.1,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        metric_for_best_model='f1',
        save_total_limit = 1,
        load_best_model_at_end=True,
    )
    
    
    
   
    
    
    
   