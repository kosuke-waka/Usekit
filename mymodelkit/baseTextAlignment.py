import os
import re

from abc import ABCMeta, abstractmethod
from abc import ABC
import pandas as pd
from InitialParameter import InitialParameter


from abc import ABC, abstractmethod
import unicodedata
import pandas as pd

class baseTextAlignment(ABC):
    
    def __init__(self, parameters: InitialParameter):
        self.text_column_name = parameters.text_column_name
    
    def save(self, data: pd.DataFrame, save_dir: str, save_file: str) -> str:
        save_location = os.path.join(save_dir, self.__class__.__name__ + '_' + save_file)
        data.to_excel(save_location, index=None)
        return save_location
    
    
    def text_clean(self, text:str):
        replaced_text = text
        replaced_text = unicodedata.normalize("NFKC",text)
        replaced_text = re.sub(r"RT @([a-zA-Z_0-9])+[:]","", replaced_text)
        replaced_text = re.sub(r'RT ','', replaced_text)
        replaced_text = re.sub(r'[【】]', '', replaced_text)       # 【】の除去
        replaced_text = re.sub('https?.+ ','',replaced_text)
        replaced_text = re.sub("https?://[\w/:%#\$&\?\(\)~\.=\+\-]+",'',replaced_text)
        replaced_text = re.sub(r'@.+ ','',replaced_text)
        replaced_text = re.sub(r'[a-zA-Z]','',replaced_text)
        replaced_text = re.sub('[0-9０-９]','',replaced_text)
        replaced_text = re.sub(r"\s", "", replaced_text)
        replaced_text = re.sub(r'[●▼■★▽]','',replaced_text)
        replaced_text = re.sub(r'[…]','',replaced_text)
        replaced_text = re.sub(r'[ωﾟ]','',replaced_text)
        replaced_text = re.sub('[!"#$%&\'\\\\()*+,-./:;<=>?@[\\]^_`{|}~「」〔〕“”〈〉『』【】＆＊・（）＄＃＠。、？！｀＋￥％]','',replaced_text)
        replaced_text = re.sub(r'[,]','',replaced_text)

        return replaced_text 
    
    @abstractmethod
    def alignment(self, data: pd.DataFrame, kawrgs=None) -> pd.DataFrame:  #データに対する処理の方法指定
        pass

    @abstractmethod
    def run(self, data: pd.DataFrame, kwargs=None) -> pd.DataFrame: #　alignmentの実行方法を指定
        pass