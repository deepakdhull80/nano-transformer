import argparse

from tqdm import tqdm
import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter

from transformer.tokenizer import CharTokenizer
from transformer.data import get_dataloader
from transformer.utils import isalphanum
from transformer.model import TextModel

argparser = argparse.ArgumentParser()
argparser.add_argument("--device", default="cpu")
argparser.add_argument("--batch_size", default=16)
argparser.add_argument("--max_len", default=1200)
argparser.add_argument("--embedding_size",default=128)
argparser.add_argument("--lr",default=1e-3)
argparser.add_argument("--epoch",default=5)
argparser.add_argument("--train_split",default=0.8)
argparser.add_argument("--n_block",default=3)

def run():
	for epoch in range(EPOCH):
		print(f"Starting epoch {epoch}")
		train_loss = 0
		train_iter = tqdm(train_dl, total=len(train_dl))
		model.train()
		for i,batch in enumerate(train_iter):
			
			optimizer.zero_grad()
			X = batch[0].to(device)
			logit, loss = model(X,X)

			loss.backward()
			optimizer.step()
			with torch.no_grad():
				loss = loss.detach().cpu().item()
				train_loss+=loss
				train_iter.set_description(f"Loss: {loss: .2f},Mean Loss {train_loss/(i+1): .2f}(train)")
		writer.add_scalar('training loss',
					train_loss/len(train_iter),
					epoch+1)

		val_iter = tqdm(val_dl, total=len(val_dl))
		model.eval()
		val_loss = 0
		for i, batch in enumerate(val_iter):
			with torch.no_grad():
				X = batch[0].to(device)
				logit, loss = model(X,X)
				loss = loss.detach().cpu().item()
				val_loss+=loss
				val_iter.set_description(f"Loss: {loss: .2f},Mean Loss {val_loss/(i+1): .2f}(val)")
		
		writer.add_scalar('val loss',
					val_loss/len(val_iter),
					epoch+1)
		with torch.no_grad():
			mask = [1]
			mask.extend([0 for _ in range(MAX_LEN-1)])
			text = model.inference(tokenizer.item[-2:], [mask for _ in range(2)])
		
		writer.add_text('text-predict', "")

if __name__ == '__main__':
	parser = argparser.parse_args()
	writer = SummaryWriter('runs/')

	MAX_LEN = int(parser.max_len)
	EMBEDDING_SIZE = int(parser.embedding_size)
	BATCH_SIZE = int(parser.batch_size)
	LR = float(parser.lr)
	EPOCH = int(parser.epoch)
	TRAIN_SPLIT = float(parser.train_split)
	N_BLOCK = int(parser.n_block)

	device = torch.device(parser.device)

	datapath = "data/hindi_english_parallel.csv"
	ds = pd.read_csv(datapath)
	ds['wc'] = ds['english'].map(lambda x: len(str(x).split()))

	ds['isalphanum'] = ds['english'].map(lambda x: isalphanum(str(x)))
	dataset = ds[(ds['wc']>10) & (ds['isalphanum'])].reset_index(drop=True)
	dataset = dataset.iloc[:1000]
	tokenizer = CharTokenizer(max_len=MAX_LEN)
	tokenizer.fit(dataset['english'])

	train_dl , val_dl = get_dataloader(tokenizer, train_split=TRAIN_SPLIT, batch_size=BATCH_SIZE)

	model = TextModel(
		len(tokenizer.k2v.keys())+1, MAX_LEN, EMBEDDING_SIZE, N_BLOCK
	)

	model = model.to(device)

	optimizer = torch.optim.Adagrad(model.parameters(), lr=LR)

	run()

	writer.close()