import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from .tokenizer import CharTokenizer

class TextDataset(Dataset):
    def __init__(self, text_data, masks):
        self.text_data = text_data
        self.masks = masks
        
    def __getitem__(self, indx):
        return torch.tensor(self.text_data[indx], dtype=torch.long), torch.tensor(self.masks[indx], dtype=torch.bool)
    def __len__(self):
        return self.text_data.shape[0]

def get_dataloader(tokenizer:CharTokenizer, **kwargs):
	assert tokenizer.item.shape[0] > 0, "tokenizer doesn't have data and mask."
	batch_size = kwargs.get("batch_size", 8)
	shuffle = kwargs.get("shuffle", False)
	pin_memory = kwargs.get("pin_memory", False)
	train_split = kwargs.get("train_split")
	if not train_split:
		# split is not required
		raise NotImplementedError
	total_train_size = int(tokenizer.total_items * train_split)
	train_ds = TextDataset(tokenizer.item[:total_train_size], tokenizer.mask[:total_train_size])
	val_ds = TextDataset(tokenizer.item[total_train_size:], tokenizer.mask[total_train_size:])

	train_dl = DataLoader(
		train_ds, batch_size=batch_size, shuffle=shuffle,pin_memory=pin_memory
	)
	val_dl = DataLoader(
		val_ds, batch_size=batch_size, shuffle=False,pin_memory=pin_memory
	)
	return train_dl, val_dl
