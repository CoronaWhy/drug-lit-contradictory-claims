"""General module to help train SBERT for NLI tasks."""


import shutil
import os
import wget

import numpy as np
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import BertModel, BertTokenizer
from sentence_transformers import models
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from .dataloader import collate_fn, multi_acc
from .dataloader import ClassifierDataset
from ..data.make_dataset import remove_tokens_get_sentence_sbert


class SBERT_Predictor(SentenceTransformer):
    def __init__(self,
                 word_embedding_model,
                 pooling_model,
                 num_classes: int = 3,
                 device: str = None):
        super().__init__()
        self.embedding_model = SentenceTransformer(modules=[word_embedding_model, pooling_model], device=device)
        self.linear = nn.Linear(6912, num_classes)
        self.sigmoid = nn.Sigmoid()
        # self.softmax = nn.Softmax(num_classes=3)
        if device is None:
            self._target_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self._target_device = torch.device(device)
        self.to(self._target_device)

    def forward(self, sentence1, sentence2):
        sentence1_embedding = torch.tensor(self.embedding_model.encode(sentence1, is_pretokenized=True),
                                           device=self._target_device).reshape(-1, 2304)
        sentence2_embedding = torch.tensor(self.embedding_model.encode(sentence2, is_pretokenized=True),
                                           device=self._target_device).reshape(-1, 2304)
        net_vector = torch.cat((sentence1_embedding, sentence2_embedding,
                                torch.abs(sentence1_embedding - sentence2_embedding)), 1)
        linear = self.linear(net_vector)
        h_out = self.sigmoid(linear)
        return h_out


def trainer(model: SBERT_Predictor,
            train_dataloader: ClassifierDataset,
            val_dataloader: ClassifierDataset,
            class_weights: torch.tensor,
            epochs: int,
            learning_rate: float = 1e-5,
            ):
    """Train the SBERT model using a training data loader and a validation dataloader.

    :param model: SBERTPredicor model
    :type model: SBERT_Predictor
    :param train_dataloader: train dataloader
    :type train_dataloader: ClassifierDataset
    :param val_dataloader: validatoin Dataloader
    :type val_dataloader: ClassifierDataset
    :param class_weights: class weiights tensor
    :type class_weights: torch.tensor
    :param epochs: numer of epochs
    :type epochs: int
    :param learning_rate: learning rate
    :type learning_rate: float
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model.to(device)
    accuracy_stats = {"train": [],
                      "val": []
                      }
    loss_stats = {"train": [],
                  "val": []
                  }

    print("------TRAINING STARTS----------")  # noqa: T001

    for e in range(epochs):
        train_epoch_loss = 0
        train_epoch_acc = 0
        model.train()
        for sentence1, sentence2, label in tqdm(train_dataloader):
            label = label.to(device)
            optimizer.zero_grad()
            y_train_pred = model(sentence1, sentence2)

            train_loss = criterion(y_train_pred, label)
            train_acc = multi_acc(y_train_pred, label)

            train_loss.backward()
            optimizer.step()

            train_epoch_loss += train_loss.item()
            train_epoch_acc += train_acc.item()

        # VALIDATION
        with torch.no_grad():

            val_epoch_loss = 0
            val_epoch_acc = 0

            model.eval()
            for sentence1, sentence2, label in val_dataloader:
                label = label.to(device)
                y_val_pred = model(sentence1, sentence2)

                val_loss = criterion(y_val_pred, label)
                val_acc = multi_acc(y_val_pred, label)

                val_epoch_loss += val_loss.item()
                val_epoch_acc += val_acc.item()

        loss_stats['train'].append(train_epoch_loss / len(train_dataloader))
        loss_stats['val'].append(val_epoch_loss / len(val_dataloader))
        accuracy_stats['train'].append(train_epoch_acc / len(train_dataloader))
        accuracy_stats['val'].append(val_epoch_acc / len(val_dataloader))
        print(f"Epoch {e+0:03}: | Train Loss: {train_epoch_loss/len(train_dataloader):.5f} \
            | Val Loss: {val_epoch_loss / len(val_dataloader):.5f} \
            | Train Acc: {train_epoch_acc/len(train_dataloader):.3f} \
            | Val Acc: {val_epoch_acc/len(val_dataloader):.3f}")

    print("---------TRAINING ENDED------------")  # noqa: T001


def train_sbert_model(model_name,
                      mancon_corpus=False,
                      med_nli=False,
                      multi_nli=False,
                      multi_nli_train_x: np.ndarray = None,
                      multi_nli_train_y: np.ndarray = None,
                      multi_nli_test_x: np.ndarray = None,
                      multi_nli_test_y: np.ndarray = None,
                      batch_size: int = 2,
                      num_epochs: int = 1,
                      ):
    if models == "deepset/covid_bert_base":
        covid_bert_path = "covid_bert_path"
        model_save_path = covid_bert_path
        wget.download("https://cdn.huggingface.co/deepset/covid_bert_base/vocab.txt",
                      out=model_save_path)  # download the vocab file

    else:
        model_name = "dmis-lab/biobert-v1.1"
        model_save_path = "biobert_path"
        wget.download("https://cdn.huggingface.co/dmis-lab/biobert-v1.1/vocab.txt",
                      out=model_save_path)  # download the vocab file

    os.makedirs(model_save_path, exist_ok=True)
    bert_model = BertModel.from_pretrained(model_name)
    bert_model.save_pretrained(model_save_path)
    covid_ert_tokenizer = BertTokenizer.from_pretrained(model_name)
    del bert_model

    word_embedding_model = models.Transformer(model_save_path)
    shutil.rmtree(model_save_path)
    pooling_model = models.Pooling(768,
                                   pooling_mode_mean_tokens=True,
                                   pooling_mode_cls_token=True,
                                   pooling_mode_max_tokens=True)
    # generating biobert sentence embeddings (mean pooling of sentence embedding vectors)
    sbert_model = SBERT_Predictor(word_embedding_model, pooling_model)
    if multi_nli:
        if multi_nli_train_x is not None:

            df_mancon_train = remove_tokens_get_sentence_sbert(multi_nli_train_x, multi_nli_train_y)
            df_mancon_val = remove_tokens_get_sentence_sbert(multi_nli_test_x, multi_nli_test_y)

            mancon_train_dataset = ClassifierDataset(df_mancon_train, tokenizer=covid_ert_tokenizer)
            mancon_val_dataset = ClassifierDataset(df_mancon_val, tokenizer=covid_ert_tokenizer)

            class_weights = mancon_train_dataset.class_weights

            train_loader = DataLoader(dataset=mancon_train_dataset,
                                      batch_size=batch_size, collate_fn=collate_fn)
            val_loader = DataLoader(dataset=mancon_val_dataset, batch_size=1, collate_fn=collate_fn)

            trainer(model=sbert_model, train_dataloader=train_loader, val_dataloader=val_loader,
                    class_weights=class_weights, batch_size=batch_size, epochs=num_epochs)

    return sbert_model
