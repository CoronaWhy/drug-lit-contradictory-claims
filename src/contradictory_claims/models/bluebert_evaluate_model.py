"""Functions for evaluating model and creating various reports for data viz/analysis."""

# -*- coding: utf-8 -*-

import datetime
import os

from .bluebert_train_model import ContraDataset
import pandas as pd
from torch.utils.data import DataLoader
from transformers import BertTokenizer


def bluebert_make_predictions(df: pd.DataFrame,
							  bluebert_pretrained_path: str,
							  model,
							  device,
							  model_name: str,
							  max_len: int = 512,
							  method: str = "multiclass"):
    """
    Make predictions using trained model and data to predict on.

    :param df: Pandas DataFrame containing data to predict on
	:param bluebert_pretrained_path: path to pretrained bluebert model
    :param model: end-to-end trained Transformer model
	:param device: CPU vs GPU definition for torch
    :param model_name: name of model to be loaded by Transformer to get proper tokenizer
    :param max_len: max length of string to be encoded
    :param method: "multiclass" or "binary"--describes setting for prediction outputs
    :return: Pandas DataFrame augmented with predictions made using trained model
    """
    # First insert the CLS and SEP tokens
    inputs = []
    for i in range(len(df)):
        # NOTE: this expects columns named "text1" and "text2" for the two claims
        inputs.append(str('[CLS]' + df.loc[i, 'text1'] + '[SEP]' + df.loc[i, 'text2']))

	# Next prepare the dataloader
	tokenizer = BertTokenizer.from_pretrained(bluebert_pretrained_path)
	labels = np_utils.to_categorical(df.label, dtype='int')
	eval_dataset = ContraDataset(inputs, labels, tokenizer, max_len=512)
    eval_dataloader = DataLoader(eval_dataset, batch_size=1)
	
	# Put the model in evaluation mode--the dropout layers behave differently during evaluation.
	model.eval()

    # Then make predictions
    for batch in dataloader:
		claim = batch[0].to(device)
		mask = batch[1].to(device)

		with torch.no_grad():                    
		  pred_labels = model(claim, mask)
		
		pred_labels = pred_labels.detach().cpu().numpy()
		
		if method == "multiclass":
			# Get index of largest softmax prediction
			pred_flat = np.argmax(pred_labels, axis=1).flatten()
			df['predicted_class'] = pred_flat
		elif method == "binary":
			# TODO: Add binary class model architecture code
			pass
		else:
			raise ValueError(f"{method} not a valid method type. Must be \"multiclass\" or \"binary\"")

    return predictions
