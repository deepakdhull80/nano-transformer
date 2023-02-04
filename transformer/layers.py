import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, max_len, emb_dim):
        
        super().__init__()
        self.max_len = max_len
        self.emb_dim = emb_dim
        self.qkv = nn.Linear(emb_dim, 3*emb_dim)
        self.register_buffer("mask",torch.tril(torch.ones(max_len, max_len)))
    
    def forward(self, x, mask=False):
        B,T,D = x.shape
        q, k, v = self.qkv(x).split(self.emb_dim, dim=2)
        qk = (q @ torch.transpose(k,1,2)) * self.emb_dim ** -0.5
        if mask:
            qk = qk.masked_fill(self.mask!=1,float("-inf"))
        att = F.softmax(qk,dim=-1)
        return att @ v