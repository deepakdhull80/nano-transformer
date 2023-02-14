import string
from abc import ABC, abstractmethod
from typing import *

import numpy as np
from tqdm import tqdm
from nltk.tokenize import word_tokenize

class Tokenizer(ABC):
    @abstractmethod
    def fit(self,x):
        pass

    @abstractmethod
    def encode(self, sen):
        pass

    @abstractmethod
    def decode(self, tokens):
        pass

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


class WordTokenizer(Tokenizer):
    def __init__(self,
                 max_length,
                 filter = lambda x: x not in string.punctuation,
                 **kwargs
    ):
        """WordTokenizer

        Args:
            max_length (int): maximum tokens in a sentence.
            filter (lambda, optional): lambda|function which filter the tokens. Defaults to lambda x: x not in string.punctuation.
        """
        self.filter = filter
        self.max_length = max_length
        self.tokens = set()
        self.SOT = "<SOT>"
        self.EOT = "<EOT>"

    @staticmethod
    def tokenize(sentence: str) -> List:
        return word_tokenize(sentence)

    def preprocess(self, sentence:str) -> List[str]:
        sentence = sentence.lower().strip()
        tokens = WordTokenizer.tokenize(sentence)
        return set(filter(self.filter, tokens))
    
    def fit(self, docs: List[str], enable_docs_persist=False):
        
        print(f"Tokenizing the docs. No of docs {len(docs)}.")
        for sentence in tqdm(docs, total=len(docs)):
            token = self.preprocess(sentence)
            self.tokens = self.tokens.union(token)
        
        self.tokens = list(self.tokens)
        print(f"The number of tokens are {len(self.tokens)}")
        self.tokens.insert(0,self.SOT)
        self.tokens.insert(0,self.EOT)

        self.k2w = {
            i+1:w for i, w in enumerate(self.tokens)
        }

        self.w2k = {
            v:k for k,v in self.k2w.items()
        }
        if enable_docs_persist:
            self.vec, self.mask = self.encode(docs)



    def encode(self, docs:List[str]) -> Tuple[np.ndarray, np.ndarray]:
        sen_li = []
        mask_li = []
        for sentence in docs:
            _x = [self.w2k[w] for w in self.preprocess(sentence)]
            _x = _x[:self.max_length]
            sen_len = len(_x)
            if self.max_length > sen_len:
                padding = self.max_length - sen_len
                _x = _x + [-1 for _ in range(padding)]
            
            _mask = [1 for _ in range(sen_len)] + [0 for _ in range(self.max_length - sen_len)]
            sen_li.append(_x)
            mask_li.append(_mask)
        
        return np.array(sen_li), np.array(mask_li)

    def decode(self, tokens:Union[List[List[int]], np.ndarray]) -> List[str]:
        
        def _decode(_tokens: List[int]):
            return " ".join([self.k2w[token] for token in _tokens if token != 0])
        
        res_li = []
        for _tokens in tokens:
            res_li.append(_decode(_tokens))
        return res_li



if __name__ == '__main__':
    import pandas as pd
    ### CharTokenizer

	# tokenizer = CharTokenizer(max_len=120)
	# ds = pd.read_csv("D:\Research Lab\datasets\language-dataset\hi-en-text/hindi_english_parallel.csv")
	# ds['wc'] = ds['english'].map(lambda x: len(str(x).split(" ")))
	# from utils import isalphanum

	# ds['isalphanum'] = ds['english'].map(lambda x: isalphanum(str(x)))
	# dataset = ds[(ds['wc']>10) & (ds['isalphanum'])].reset_index(drop=True)

	# tokenizer.fit(dataset['english'])
	# print(tokenizer.item[0])
	# print(tokenizer.decode(tokenizer.item[0]))

    ### WordTokenizer
    df = pd.read_csv("data/wiki-data/cnn_dailymail/validation.csv")
    
    tokenizer = WordTokenizer(12)
    tokenizer.fit(df['highlights'].to_list()[:20])
    
    vec, mask = tokenizer.encode(df['highlights'].to_list()[:10])
    
    print(vec.shape, mask.shape)
    print(vec[0])
    print(mask[0])
    print(tokenizer.decode(vec[:2]))