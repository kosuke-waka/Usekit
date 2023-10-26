import os
import re

from abc import ABCMeta, abstractmethod
from abc import ABC
import pandas as pd
from InitialParameter import InitialParameter

from daaja.methods.eda.easy_data_augmentor import EasyDataAugmentor
from baseTextAlignment import baseTextAlignment

class TextAugmentation(baseTextAlignment):
    def alignment(self, data: pd.DataFrame, kwargs=None) -> pd.DataFrame:
        if kwargs is None:
            augmentor = EasyDataAugmentor(alpha_sr=0.1, alpha_ri=0.1, alpha_rs=0.1, p_rd=0.1, num_aug=4)
        else:
            augmentor = EasyDataAugmentor(**kwargs)
        all_pd = pd.DataFrame()
        # dataを一行一行読み込んでいく
        for index, row in data.iterrows():
            tmp_row = row.copy()
            texts = augmentor.augments(tmp_row[self.text_column_name]) #行ごとのテキストを拡張したリストを取得
            for text in texts:
                tmp_row[self.text_column_name] = text #テキストだけを変えて、他の値はそのまま使う
                #tmp_row.to_frame() でtmp_rowをseries型からDataFrame型へと変換する　ー＞concatをうまくいくようにするため
                all_pd = pd.concat([all_pd, tmp_row.to_frame().T], axis=0)
        all_pd = all_pd.reset_index(drop=True)
        return all_pd
        
    def run(self, data: pd.DataFrame, kwargs: dict=None) -> pd.DataFrame:
        return self.alignment(data, kwargs)
    
