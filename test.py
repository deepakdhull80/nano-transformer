import pandas as pd
import torch


from transformer.tokenizer import CharTokenizer
from transformer.data import get_dataloader
from transformer.utils import isalphanum
from transformer.model import TextModel

MAX_LEN = 23000
EMBEDDING_SIZE = 128
BATCH_SIZE = 32
LR = 1e-3
EPOCH = 100

tokenizer = CharTokenizer(max_len=120)
ds = pd.read_csv("D:\Research Lab\datasets\language-dataset\hi-en-text/hindi_english_parallel.csv")
ds['wc'] = ds['english'].map(lambda x: len(str(x).split(" ")))

ds['isalphanum'] = ds['english'].map(lambda x: isalphanum(str(x)))
dataset = ds[(ds['wc']>10) & (ds['isalphanum'])].reset_index(drop=True)

tokenizer.fit(dataset['english'])
print(tokenizer.item[0])
print(tokenizer.decode(tokenizer.item[0]))


train_dl , val_dl = get_dataloader(tokenizer, train_split=0.8, batch_size=BATCH_SIZE)

model = TextModel(
	len(tokenizer.k2v.keys()),MAX_LEN, EMBEDDING_SIZE, 3
)
device = torch.device("cuda:0")
model.to(device)

optimizer = torch.optim.Adagrad(model.parameters(), lr=LR)

for epoch in range(EPOCH):
	print(f"Training epoch {epoch}")
	for batch in train_dl:
		optimizer.zero_grad()
		X,mask = batch[0].to(device), batch[1].to(device)
		logit, loss = model(X,X)

		loss.backward()
		optimizer.step()
		with torch.no_grad():
			print("Loss:",loss.detach().numpy())

	print(f"VALIDATION LOSS")
	for batch in val_dl:
		with torch.no_grad():
			X,mask = batch[0].to(device), batch[1].to(device)
			logit, loss = model(X,X)
			print("Loss:",loss)
	print("-"*30)