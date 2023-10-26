import os
import re

from abc import ABCMeta, abstractmethod
from abc import ABC
import pandas as pd
from InitialParameter import InitialParameter
from baseTextAlignment import baseTextAlignment

from pyknp import Juman

class jumanTextAlignment(baseTextAlignment):
    def alignment(self, data: pd.DataFrame) -> pd.DataFrame:
        jumanpp = Juman()
        def wakati_jumanpp(text):
            text = self.text_clean(text)
              #テキストを解析
            analysis = jumanpp.analysis(text)
            result = []
            for m in analysis.mrph_list():
                result.append(m.midasi)
            result = ' '.join(result)
            return result
        all_pd = data.copy()
        all_pd[self.text_column_name] = all_pd[self.text_column_name].apply(lambda x: wakati_jumanpp(x))
        return all_pd
            
    
    def run(self, data: pd.DataFrame, kwargs=None) -> pd.DataFrame:
        return self.alignment(data)