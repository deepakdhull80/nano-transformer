from typing import *

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class MaskedTextDataset(Dataset):
	def __init__(self, text_data, masks, tokenizer, mask_fraction=0.1):
		self.text_data = text_data
		self.masks = masks
		self.tokenizer = tokenizer
		self.mask_fraction = mask_fraction

	def __getitem__(self, indx):
		tokens, mask = torch.tensor(self.text_data[indx], dtype=torch.long), torch.tensor(self.masks[indx], dtype=torch.bool)
		masked_idx = torch.randint(0,self.tokenizer.max_length,(int(self.tokenizer.max_length*self.mask_fraction),))
		masked_token = torch.clone(tokens)
		masked_token[masked_idx] = self.tokenizer.w2k[self.tokenizer.MASK_TOKEN]
		return masked_token, tokens, mask

	def __len__(self):
		return self.text_data.shape[0]

def get_dataloader(text: np.ndarray, mask: np.ndarray, tokenizer, **kwargs):
	assert text.shape[0] > 0, "tokenizer doesn't have data and mask."

	batch_size = kwargs.get("batch_size", 8)
	shuffle = kwargs.get("shuffle", False)
	pin_memory = kwargs.get("pin_memory", False)
	train_split = kwargs.get("train_split")
	if not train_split:
	# split is not required
		raise NotImplementedError
	total_train_size = int(text.shape[0] * train_split)
	train_ds = MaskedTextDataset(text[:total_train_size], mask[:total_train_size], tokenizer, kwargs.get("mask_fraction",0.1))
	val_ds = MaskedTextDataset(text[total_train_size:], mask[total_train_size:], tokenizer, kwargs.get("mask_fraction",0.1))

	train_dl = DataLoader(
	train_ds, batch_size=batch_size, shuffle=shuffle,pin_memory=pin_memory
	)
	val_dl = DataLoader(
	val_ds, batch_size=batch_size, shuffle=False,pin_memory=pin_memory
	)
	return train_dl, val_dl


if __name__ == "__main__":
	import pandas as pd
	from tokenizer import WordTokenizer
	df = pd.read_csv("data/wiki-data/cnn_dailymail/validation.csv")

	tokenizer = WordTokenizer(12)
	tokenizer.fit(df['highlights'].to_list()[:100])

	vec, mask = tokenizer.encode(df['highlights'].to_list()[:50])

	train_dl, val_dl = get_dataloader(vec, mask, tokenizer, train_split=0.8, batch_size=2)
	for b in train_dl:
		print(b)
		break