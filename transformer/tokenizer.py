import string

import numpy as np
import pandas as pd

class CharTokenizer:
    from tqdm import tqdm
    def __init__(self, verbose=0, max_len=1200):
        self.verbose = verbose
        self.max_len = max_len
        self.k2v = None
        self.v2k = None
        self.mask = None
        self.item = None

    def fit(self,x):
        item = []
        mask = []
        self.total_items = x.shape[0]
        assert type(x) in [pd.core.series.Series, list, np.ndarray], f"x should be in [pd.Series, list, ndarray] but got, {type(x)}"
        self.unique_chars = CharTokenizer.get_unique(x)
        self.k2v = {c:i+1 for i,c in enumerate(self.unique_chars)}
        self.v2k = {i+1:c for i,c in enumerate(self.unique_chars)}
        for c in self.tqdm(x, total=len(x)):
            item.append(self.tokenize(c))
            mask.append(self.masked(c))
        self.item = np.array(item, dtype=np.int32)
        self.mask = np.array(mask, dtype=np.bool_)
        if self.verbose:
            print(f"total items, {self.item.shape[0]}")
    
    def tokenize(self, sen):
        tokens = [self.k2v[c] for c in list(sen)]
        tokens = tokens[:self.max_len]
        padded = [0 for _ in range(self.max_len - len(tokens))]
        tokens.extend(padded)
        return tokens
    
    def masked(self, sen):
        sen = sen[:self.max_len]
        n = len(list(sen))
        return [1 for _ in range(n)] + [0 for _ in range(self.max_len - n)]

    @staticmethod
    def get_unique(dataset):
        chars = set()
        for s in dataset:
            chars.update(set(list(s)))
        chars = list(chars)
        chars.sort()
        return chars

    def decode(self, tokens: np.ndarray, mask: np.ndarray=None):
        if mask is None:
            mask = np.where(tokens != 0 , True, False)
        return "".join([self.v2k[t] for t in tokens[mask]])


if __name__ == '__main__':
	tokenizer = CharTokenizer(max_len=120)
	ds = pd.read_csv("D:\Research Lab\datasets\language-dataset\hi-en-text/hindi_english_parallel.csv")
	ds['wc'] = ds['english'].map(lambda x: len(str(x).split(" ")))
	from utils import isalphanum

	ds['isalphanum'] = ds['english'].map(lambda x: isalphanum(str(x)))
	dataset = ds[(ds['wc']>10) & (ds['isalphanum'])].reset_index(drop=True)

	tokenizer.fit(dataset['english'])
	print(tokenizer.item[0])
	print(tokenizer.decode(tokenizer.item[0]))
