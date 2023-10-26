import pandas as pd
from dataclasses import dataclass, field

@dataclass
class Dataset:
    train_pd: pd.DataFrame
    test_pd: pd.DataFrame
        
    train_encoding: list[dict]
    
    test_encoding: list[dict]