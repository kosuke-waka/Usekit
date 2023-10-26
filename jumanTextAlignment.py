

class MultiTextDataLoader(baseTextDataLoader):
    """
        ラベル列をもとにダミー列を作成する。その後、元のDataFrameのラベル列を削除しダミー列を結合する
    """
    def load(self , labels: list=None) -> Dataset:
        if labels == None:
            labels = self.parameters.label_list
            
        train_pd, test_pd = self._load()
        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        #ラベル毎のダミー列を作成
        for label in labels:
            train_pd[label] = train_pd[self.label_column_name].apply(lambda x:1 if label in x else 0)
            test_pd[label] = test_pd[self.label_column_name].apply(lambda x:1 if label in x else 0)
        #もともとあったラベル列を削除
        train_pd = train_pd.drop(self.label_column_name, axis=1)
        test_pd = test_pd.drop(self.label_column_name, axis=1)
        train_encoding = []
        test_encoding = []
        #訓練用データをトークナイズする
        for index, row in train_pd.iterrows():
            encoding = tokenizer(row[self.text_column_name],max_length=self.parameters.max_length, padding='max_length',truncation=True)
            #テキストデータ以外の値（ダミー列）のリストを作成してpytorchのテンソルにする
            encoding['labels'] = torch.tensor([value for key, value in row.to_dict().items() if key != self.text_column_name]) 
            encoding = {k:torch.tensor(v).to(self.parameters.device) for k,v in encoding.items()}
            train_encoding.append(encoding)
        #テストデータをトークナイズする
        for index, row in test_pd.iterrows():
            encoding = tokenizer(row[self.text_column_name],max_length=self.parameters.max_length, padding='max_length',truncation=True)
            encoding['labels'] = torch.tensor([value for key, value in row.to_dict().items() if key != self.text_column_name])
            encoding = {k:torch.tensor(v).to(self.parameters.device) for k,v in encoding.items()}
            test_encoding.append(encoding)
        return Dataset(train_pd, test_pd, train_encoding, test_encoding)