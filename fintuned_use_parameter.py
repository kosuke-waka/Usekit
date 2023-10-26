import os
import torch
from transformers import Trainer, TrainingArguments 
from dataclasses import dataclass, field

@dataclass
class fintuned_use_parameter:
    """

    """
    use_data_dir: str = './data/test'
    use_list: list[str] = field(default_factory=lambda: ['sokushin_test.xlsx'])
    
    save_dir: str = os.path.join(os.getcwd(), 'result')
    save_list: list[str] = field(default=lambda: ['shoshin'])
        
    use_tokenizer: list[str] = field(default_factory=lambda: ['nlp-waseda/roberta-base-japanese'])
    use_model_dir_list: list[str] = field(default_factory=lambda: ['./result/sokushin'])
        
    text_column_name: str = 'ツイート'
    label_column_name: list[str] = field(default_factory=lambda: ['促進軸'])
    label_list: list[str] = field(default_factory=lambda: [['推奨', '行動抑制', '励まし', '願望']])
    
    #ラベルを日本語から英語へ
    JaToEn_label: dict[str] = field(default_factory=lambda: {'推奨':'recommend', '行動抑制':'suppression', '励まし': 'encouragement', '願望': 'desire'})
    #列を日本語から英語へ
    JaToEn_column : dict[str] = field(default_factory=lambda: {'ツイート': 'tweet', '判定結果': 'label'})
        
    device = "cuda:0" if torch.cuda.is_available() else 'cpu'

    max_length: int = 256
