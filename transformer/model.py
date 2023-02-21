import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import MultiHeadAttention

class EncoderBlock(nn.Module):
    def __init__(self, heads, max_len:int, emb_dim:int):
        super().__init__()
        self.max_len = max_len
        self.emb_dim = emb_dim
        self.heads = heads
        self.attn = MultiHeadAttention(heads, emb_dim)
        self.norm = torch.nn.LayerNorm(emb_dim)
        self.linear = nn.Linear(emb_dim, emb_dim)
        
    def forward(self,x:torch.Tensor):
        x = self.norm(x + self.attn(x, x, x))
        x = self.norm(x + self.linear(x))
        return x

class DecoderBlock(nn.Module):
    def __init__(self, heads, max_len:int, emb_dim:int) -> None:
        super().__init__()
        self.max_len = max_len
        self.emb_dim = emb_dim
        self.heads = heads
        self.mask = True
        self.masked_attn = MultiHeadAttention(heads, emb_dim, True)
        self.attn = MultiHeadAttention(heads, emb_dim)
        self.norm = torch.nn.LayerNorm(emb_dim)
        self.linear = nn.Linear(emb_dim, emb_dim)
    
    def forward(self, encoder, x):
        x = self.norm(x + self.masked_attn(x, x, x))
        
        x = self.norm(x + self.attn(x, encoder, encoder))
        x = self.norm(x + self.linear(x))
        return x


class NanoTransformer(nn.Module):
    def __init__(self, token_size, max_len, heads, emb_dim, n_block, loss_fn = torch.nn.CrossEntropyLoss(),tokenizer=None):
        super().__init__()
        self.tokenizer = tokenizer
        self.token_size = token_size
        self.max_len = max_len
        self.emb_dim = emb_dim
        self.n_block = n_block
        self.embedding = nn.Embedding(token_size, emb_dim)
        self.positional = nn.Embedding(token_size, emb_dim)
        self.blk = nn.Sequential(*[EncoderBlock(max_len, heads, emb_dim) for _ in range(n_block)])
        self.final_mlp = nn.Linear(emb_dim, token_size)
        self._loss = loss_fn

    def loss(self, x, y):
        return self._loss(x, y)

    @torch.no_grad()
    def inference(self, x):
        #TODO: need to implement.
        logits,_ = self.forward(x)
        return logits

    def forward(self,x,y=None):
        _x = x.clone()
        try:
            x = self.embedding(x) + self.positional(x)
        except Exception as e:
            print(x.max(), x.min())
            raise Exception(e)
        for blk in self.blk:
            x = blk(x)
        x = self.final_mlp(x)
        loss = None
        if y is not None:
            c = _x != self.tokenizer.w2k[self.tokenizer.PADD]
            # print(y[c])
            loss = self.loss( x[c], y[c] )

        x = torch.argmax(x,dim=-1)
        return x, loss


if __name__ == "__main__":

	token_size = int(1e6)
	model = NanoTransformer(
		token_size=token_size,
		max_len=200,
		emb_dim=128,
		n_block=2,
		heads=4
	)

	inp = torch.randint(0,token_size, size=(2,200))
	out = model.inference(inp)
	print(out)