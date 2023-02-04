import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import SelfAttention

class Block(nn.Module):
    def __init__(self, max_len:int, emb_dim:int, masked:bool):
        super().__init__()
        self.max_len = max_len
        self.emb_dim = emb_dim
        self.masked = masked
        self.attn = SelfAttention(max_len, emb_dim)
    
    def forward(self,x:torch.Tensor):
        x = self.attn(x)
        return x

class TextModel(nn.Module):
    def __init__(self, token_size, max_len, emb_dim, n_block):
        super().__init__()
        self.token_size = token_size
        self.max_len = max_len
        self.emb_dim = emb_dim
        self.n_block = n_block
        self.embedding = nn.Embedding(token_size,emb_dim)
        self.positional = nn.Embedding(token_size,emb_dim)
        self.blk = nn.Sequential(*[Block(max_len, emb_dim, masked=True) for _ in range(n_block)])
        self.final_mlp = nn.Linear(emb_dim, token_size)
        self._loss = torch.nn.CrossEntropyLoss()

    def loss(self, x, y):
        return self._loss(x, y)
    
    def inference(self, x):
        pred = None
        x = x[:,0].view(-1,1)
        
        logits,_ = self.forward(x)
        

    def forward(self,x,y=None):
        x = self.embedding(x) + self.positional(x)
        for blk in self.blk:
            x = blk(x)
        x = self.final_mlp(x)
        loss = None
        if y is not None:
            loss = self.loss( x.view(-1,x.shape[-1]), y.view(-1) )
        return x, loss