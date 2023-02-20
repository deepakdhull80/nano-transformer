import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, emb_dim):
        
        super().__init__()
        self.emb_dim = emb_dim
        self.Q = nn.Linear(emb_dim, emb_dim)
        self.K = nn.Linear(emb_dim, emb_dim)
        self.V = nn.Linear(emb_dim, emb_dim)
    
    def forward(self, q,k,v, mask=None):
        '''
        q - query (B,h,t`, d)
        k - key (B,h,t`, d)
        v - value (B,h,t`, d)
        2x1
        1x2  -> 2x2
        return attentionVector (B,h,t`, d)
        '''
        q = self.Q(q)
        k = self.K(k)
        v = self.V(v)
        
        qk = (q @ torch.transpose(k,-2,-1)) * self.emb_dim ** -0.5
        if mask != None:
            qk = qk.masked_fill(mask==1, -1e-9)
        att = F.softmax(qk, dim=-1)
        return att @ v

class MultiHeadAttention(nn.Module):
    def __init__(self, heads, emb_dim, masked:bool=False) -> None:
        super().__init__()
        self.heads = heads
        self.emb_dim = emb_dim
        self.attention = Attention(
            self.emb_dim
        )
        self.linear = nn.Linear(emb_dim,emb_dim)
        self.masked = masked
    
    def forward(self, q, k, v):
        '''
        q,k,v - (B,T,d)
        
        return attentionVec (B, T,d)
        '''
        B, T, D = q.shape
        #*** Ques - why do we need to permute it, we can split it into (B,h,-1,D)
        #Ans: If we directly do (B,h,-1,D) it will split vector in sequential way, but we want to split head1:{a[i+head*j]}
        q = q.view(B, -1, self.heads, D).permute(0,2,1,3)
        k = k.view(B, -1, self.heads, D).permute(0,2,1,3)
        v = v.view(B, -1, self.heads, D).permute(0,2,1,3)
        if self.masked:
            mask = torch.triu(torch.ones(q.shape[-1],k.shape[-1]),1)
        att = self.attention(q,k,v, mask)
        att = att.permute(0,2,1,3).reshape(B, T, D)
        return self.linear(att)

if __name__ == "__main__":
    q = torch.randn(2,8,16)
    k = torch.randn(2,16,16)
    
    res = MultiHeadAttention(2,16,masked=True)(q,k,k)
    print(res.shape)