import os
import logging

from tqdm import tqdm
import pandas as pd
import torch
import yaml

from transformer.tokenizer import WordTokenizer
from transformer.data import get_dataloader
from transformer.model import NanoTransformer

indx = len(os.listdir("logs/")) + 1
log_filename = f"logs/output_{indx}.log"
os.makedirs(os.path.dirname(log_filename), exist_ok=True)
logging.basicConfig(filename=log_filename,
                    filemode='w',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.INFO)

logger = logging.getLogger('TORCH_TRAINER')

def run():
	g_val_loss = 10
	model_chkpt_path = f"checkpoints/cpk_{indx}_{MODEL_NAME}.pth"
	for epoch in range(EPOCH):
		print(f"Starting epoch {epoch}")
		logger.info(f"EPOCH start: {epoch}")
		train_loss = 0
		train_iter = tqdm(train_dl, total=len(train_dl))
		model.train()
		for i,batch in enumerate(train_iter):
			
			optimizer.zero_grad()
			X, Y = batch[0].to(device), batch[1].to(device)
			_, loss = model(X,Y)
			loss.backward()
			optimizer.step()
			with torch.no_grad():
				loss = loss.detach().cpu().item()
				train_loss+=loss
				train_iter.set_description(f"Loss: {loss: .2f},Mean Loss {train_loss/(i+1): .2f}(train)")
		logger.info(f"TRAIN - Mean Loss {train_loss/(len(train_dl)): .2f}")

		val_iter = tqdm(val_dl, total=len(val_dl))
		model.eval()
		val_loss = 0
		for i, batch in enumerate(val_iter):
			with torch.no_grad():
				X, Y = batch[0].to(device), batch[1].to(device)
				logit, loss = model(X,Y)
				loss = loss.detach().cpu().item()
				val_loss+=loss
				val_iter.set_description(f"Loss: {loss: .2f},Mean Loss {val_loss/(i+1): .2f}(val)")
		
		logger.info(f"VAL - Mean Loss {val_loss/(len(val_dl)): .2f}")
		t = val_loss/(len(val_dl))
		if g_val_loss>t:
			g_val_loss = t
			torch.save({
				'iter': epoch,
				'cp': model.state_dict(),
				'loss': g_val_loss
			},model_chkpt_path)
			print(f"model checkpoint saved -> {model_chkpt_path}")
			logger.info(f"model checkpoint saved -> {model_chkpt_path}")
		# with torch.no_grad():
		# 	mask = [1]
		# 	mask.extend([0 for _ in range(MAX_LEN-1)])
		# 	text = model.inference(tokenizer.item[-2:], [mask for _ in range(2)])
		

if __name__ == '__main__':
	with open("config.yaml", "r") as stream:
		config = yaml.safe_load(stream)
	MAX_LEN = config['MAX_LEN']
	EMBEDDING_SIZE = config['EMBEDDING_SIZE']
	BATCH_SIZE = config['BATCH_SIZE']
	LR = config['LR']
	EPOCH = config['EPOCH']
	TRAIN_SPLIT = config['TRAIN_SPLIT']
	N_BLOCK = config['N_BLOCK']
	HEADS = config['transformer_heads']
	MASK_FRACTION = config['sentence_mask_fraction']
	MODEL_NAME = config['model_name']

	device = torch.device(config['device'])

	tokenizer = WordTokenizer(
		max_length=MAX_LEN
	)
	df = pd.read_csv(config['data_file_path'])
	data = df['article'].to_list()[:1000]
	
	tokenizer.fit(
		docs=data
	)
	tokens = tokenizer.encode(data)

	# dataloader
	train_dl, val_dl = get_dataloader(
		tokens, tokenizer, batch_size=BATCH_SIZE,shuffle=True, pin_memory=True,train_split=TRAIN_SPLIT,mask_fraction=MASK_FRACTION
	)

	# model
	model = NanoTransformer(
		token_size=tokenizer.n_tokens+1,
		max_len=MAX_LEN,
		heads=HEADS,
		emb_dim=EMBEDDING_SIZE,
		n_block=N_BLOCK,
		tokenizer=tokenizer
	)
	model = model.to(device)
	# optimizer
	optimizer = torch.optim.Adam(model.parameters(), lr=LR)
	run()