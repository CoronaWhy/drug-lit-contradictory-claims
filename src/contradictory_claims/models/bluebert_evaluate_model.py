"""Functions for evaluating model and creating various reports for data viz/analysis."""

# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import torch
from keras.utils import np_utils
from torch.utils.data import DataLoader
from transformers import BertTokenizer

from .bluebert_train_model import ContraDataset


def bluebert_make_predictions(df: pd.DataFrame,
                              bluebert_pretrained_path: str,
                              model,
                              device,
                              model_name: str,
                              max_len: int = 512,
                              multi_class: bool = True):
    """
    Make predictions using trained model and data to predict on.

    :param df: Pandas DataFrame containing data to predict on
    :param bluebert_pretrained_path: path to pretrained bluebert model
    :param model: end-to-end trained Transformer model
    :param device: CPU vs GPU definition for torch
    :param model_name: name of model to be loaded by Transformer to get proper tokenizer
    :param max_len: max length of string to be encoded
    :param multi_class: "multiclass" or "binary"--describes setting for prediction outputs
    :return: Pandas DataFrame augmented with predictions made using trained model
    """
    # First insert the CLS and SEP tokens
    inputs = []
    if multi_class:
        for i in range(len(df)):
            # NOTE: this expects columns named "text1" and "text2" for the two claims
            inputs.append(str('[CLS]' + df.loc[i, 'text1'] + '[SEP]' + df.loc[i, 'text2']))
    else:
        # Add the category info (CON, ENT, NEU) as auxillary text at the end
        for i in range(len(df)):
            inputs.append(str('[CLS]' + df.loc[i, 'text1'] + '[SEP]' + df.loc[i, 'text2']
                              + '[SEP]' + 'CON'))  # noqa: 
        for i in range(len(df)):
            inputs.append(str('[CLS]' + df.loc[i, 'text1'] + '[SEP]' + df.loc[i, 'text2']
                              + '[SEP]' + 'ENT'))  # noqa: W503
        for i in range(len(df)):
            inputs.append(str('[CLS]' + df.loc[i, 'text1'] + '[SEP]' + df.loc[i, 'text2']
                              + '[SEP]' + 'NEU'))  # noqa: W503

    # Map labels to numerical (categorical) values
    labels = [2 if label == 'contradiction' else 1 if label == 'entailment' else 0 for
              label in df.annotation]

    # Next prepare the dataloader
    tokenizer = BertTokenizer.from_pretrained(bluebert_pretrained_path)
    labels = np_utils.to_categorical(labels, dtype='int')
    eval_dataset = ContraDataset(inputs, labels, tokenizer, max_len=512)
    eval_dataloader = DataLoader(eval_dataset, batch_size=1)

    # Put the model in evaluation mode--the dropout layers behave differently during evaluation.
    model.eval()

    # Then make predictions
    for i, batch in enumerate(eval_dataloader):
        claim = batch[0].to(device)
        mask = batch[1].to(device)

        with torch.no_grad():
            pred_labels = model(claim, mask)

        pred_labels = pred_labels.detach().cpu().numpy()

        if multi_class:
            df.loc[i, 'predicted_con'] = pred_labels[0][2]
            df.loc[i, 'predicted_ent'] = pred_labels[0][1]
            df.loc[i, 'predicted_neu'] = pred_labels[0][0]
            # Get index of largest softmax prediction
            pred_flat = np.argmax(pred_labels, axis=1).flatten()
            df.loc[i, 'predicted_class'] = int(pred_flat)
        else:
            # TODO: Add binary class model architecture code
            pass

    # Map labels back to class names
    df.predicted_class.replace(to_replace={2: 'contradiction',
                               1: 'entailment',
                               0: 'neutral'}, inplace=True)

    return df
