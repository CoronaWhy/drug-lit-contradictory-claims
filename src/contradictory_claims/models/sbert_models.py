"""General module to help train SBERT for NLI tasks."""


import datetime
import math
import os
import pickle
import shutil

import numpy as np
import torch
import torch.optim as optim
import wget
from sentence_transformers import SentenceTransformer, SentencesDataset
from sentence_transformers import losses, models
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from .dataloader import ClassifierDataset, NLIDataReader, collate_fn, multi_acc
from .dataloader import format_create
from ..data.make_dataset import remove_tokens_get_sentence_sbert


class SBERTPredictor(SentenceTransformer):
    """SBERT Prediction class."""

    def __init__(self,
                 word_embedding_model,
                 pooling_model,
                 num_classes: int = 3,
                 logistic_model=True,
                 device: str = None):
        """Initialize the class.

        :param word_embedding_model: the rod embedding model
        :param pooling_model: the pooling model
        :param num_classes: number of classes in output, defaults to 3
        :param logistic_model: should a logistic regression model be used for classification
        :param device: device type (cuda/cpu)
        :type device: str, optional
        """
        super().__init__()
        self.embedding_model = SentenceTransformer(
            modules=[word_embedding_model, pooling_model], device=device)
        #         self.linear = nn.Linear(6912, num_classes)
        #         self.linear = nn.Linear(4608, num_classes)
        self.linear = nn.Linear(2304, num_classes)  # using mean of tokens only
        # self.sigmoid = nn.Sigmoid()
        # self.softmax = nn.Softmax(dim=1)
        if device is None:
            self._target_device = torch.device(
                "cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self._target_device = torch.device(device)
        self.to(self._target_device)
        self.logistic_model = logistic_model  # should logistic model be trained or not
        self.logisticregression = LogisticRegression(
            warm_start=False, max_iter=500, class_weight="balanced")

    def forward(self, sentence1, sentence2):
        """Forward function.

        :param sentence1: batch of sentence1
        :param sentence2: batch of sentence2
        :return: sigmoid output
        :rtype: torch.Tensor
        """
        sentence1_embedding = torch.tensor(self.embedding_model.encode(
            sentence1, is_pretokenized=True), device=self._target_device).reshape(-1, 768)
        sentence2_embedding = torch.tensor(self.embedding_model.encode(
            sentence2, is_pretokenized=True), device=self._target_device).reshape(-1, 768)
        net_vector = torch.cat(
            (sentence1_embedding,
             sentence2_embedding,
             torch.abs(sentence1_embedding - sentence2_embedding)),
            1)
        # net_vector = torch.cat((sentence1_embedding, sentence2_embedding), 1)
        linear = self.linear(net_vector)
        # h_out = self.sigmoid(linear)
        # h_out = self.softmax(linear)
        h_out = linear
        return h_out

    def vector(self, sentence1, sentence2):
        """Create word tensors to be given as input to Logistic regression.

        :param sentence1: batch of sentence1
        :param sentence2: batch of sentence2
        :return: sigmoid output
        :rtype: torch.Tensor
        """
        sentence1_embedding = self.embedding_model.encode(
            sentence1, is_pretokenized=False)
        sentence2_embedding = self.embedding_model.encode(
            sentence2, is_pretokenized=False)
        net_vector = np.concatenate(
            (sentence1_embedding,
             sentence2_embedding,
             np.abs(sentence1_embedding - sentence2_embedding)),
            1)

        return net_vector

    def predict(self, sentence1, sentence2):
        """Predict class based on input sentences.

        :param sentence1: list of input sentence1
        :type sentence1: list(str)
        :param sentence2: list of input sentence2
        :type sentence2: list(str)
        """
        if self.logistic_model is True:
            net_vector = self.vector(sentence1, sentence2)
            predictions = self.logisticregression.predict(net_vector)
            return predictions
        else:
            net_vector = self.vector(sentence1, sentence2)
            predictions = self.linear(
                torch.tensor(
                    net_vector,
                    device=self._target_device))
            predictions = torch.log_softmax(predictions, dim=1)
            predictions = torch.argmax(predictions, dim=1)
            return predictions.cpu().numpy()


def freeze_layer(layer):
    """Freeze's the mentioned layer.

    :param layer: torch model layer
    """
    for param in layer.parameters():
        param.requires_grad = False


def unfreeze_layer(layer):
    """Unfreeze's the mentioned layer.

    :param layer: torch model layer
    """
    for param in layer.parameters():
        param.requires_grad = True


def trainer(model: SBERTPredictor,
            tokenizer,
            df_train,
            df_val,
            epochs: int = 1,
            learning_rate: float = 1e-5,
            batch_size: int = 16,
            embedding_epochs: int = None,
            enable_class_weights: bool = True,
            ):
    """Train the SBERT model using a training data loader and a validation dataloader.

    :param model: SBERTPredicor model
    :type model: SBERT_Predictor
    :param tokenizer: tokenizer used in SBERT model
    :param df_train: train dataframe
    :type train_dataloader: pd.DataFrame()
    :param df_val: validation dataframe
    :type df_val: pd.DataFrame()
    :param epochs: numer of epochs
    :type epochs: int
    :param learning_rate: learning rate
    :type learning_rate: float
    :param batch_size: batch size to be used for training
    :type batch_size: int
    """
    if embedding_epochs is None:
        embedding_epochs = epochs
    nli_reader = NLIDataReader(df_train.append(df_val))
    train_num_labels = nli_reader.get_num_labels()

    train_data = SentencesDataset(
        nli_reader.get_examples(),
        model=model.embedding_model)
    train_data.label_type = torch.long
    # some bug in sentence_transformer library causes it to be identified as
    # float by default
    train_dataloader_embed = DataLoader(
        train_data, shuffle=True, batch_size=batch_size)
    train_loss_embed = losses.SoftmaxLoss(
        model=model.embedding_model,
        sentence_embedding_dimension=model.embedding_model.get_sentence_embedding_dimension(),
        num_labels=train_num_labels)

    val_nli_reader = NLIDataReader(df_val)
    dev_data = SentencesDataset(
        val_nli_reader.get_examples(),
        model=model.embedding_model)
    dev_data.label_type = torch.long
    evaluator = EmbeddingSimilarityEvaluator(
        sentences1=df_val["sentence1"].values,
        sentences2=df_val["sentence2"].values,
        scores=df_val["label"].values / 2.,
        batch_size=batch_size)
    warmup_steps = math.ceil(
        len(train_dataloader_embed) * epochs / batch_size * 0.1)
    # 10% of train data for warm-up

    # now to train the final layer
    train_dataset = ClassifierDataset(df_train, tokenizer=tokenizer)
    val_dataset = ClassifierDataset(df_val, tokenizer=tokenizer)
    if enable_class_weights is False:
        class_weights = None
    else:
        class_weights = train_dataset.class_weights()

    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=batch_size,
                                  collate_fn=collate_fn,
                                  shuffle=True)
    val_dataloader = DataLoader(dataset=val_dataset,
                                batch_size=1,
                                collate_fn=collate_fn)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model.to(device)

    print("------TRAINING STARTS----------")  # noqa: T001
    # train embedding layer
    unfreeze_layer(model.embedding_model)
    model.embedding_model.fit(
        train_objectives=[
            (train_dataloader_embed,
             train_loss_embed)],
        evaluator=evaluator,
        epochs=1,
        evaluation_steps=1000,
        warmup_steps=warmup_steps,
    )  # train the Transformer layer
    freeze_layer(model.embedding_model)
    x, y = format_create(df=df_train.append(df_val), model=model)
    x_test, y_test = format_create(df=df_val, model=model)
    if model.logistic_model is True:
        model.logisticregression.fit(x, y)
        print(classification_report(y_test, model.logisticregression.predict(x_test)))  # noqa: T001
    else:
        accuracy_stats = {"train": [],
                          "val": [],
                          }
        loss_stats = {"train": [],
                      "val": [],
                      }

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

            loss_stats['train'].append(
                train_epoch_loss / len(train_dataloader))
            loss_stats['val'].append(val_epoch_loss / len(val_dataloader))
            accuracy_stats['train'].append(
                train_epoch_acc / len(train_dataloader))
            accuracy_stats['val'].append(val_epoch_acc / len(val_dataloader))
            print(f"Epoch {e+0:03}: | Train Loss: {train_epoch_loss/len(train_dataloader):.5f} \
                | Val Loss: {val_epoch_loss / len(val_dataloader):.5f} \
                | Train Acc: {train_epoch_acc/len(train_dataloader):.3f} \
                | Val Acc: {val_epoch_acc/len(val_dataloader):.3f}")  # noqa: T001

    print("---------TRAINING ENDED------------")  # noqa: T001


def build_sbert_model(model_name: str, logistic_model: bool = True):
    """Build SBERT model, based on model name provided.

    :param model_name: model to be used, currently supported: covidbert or biobert
    :type model_name: str
    :param logistic_model: use logistic regression as classifier
    :type logistic_model: bool
    :return: SBERT model and corresponding tokenizer
    """
    if model_name == "covidbert":
        model_name = "deepset/covid_bert_base"
        covid_bert_path = "covid_bert_path"
        model_save_path = covid_bert_path
        os.makedirs(model_save_path, exist_ok=True)
        wget.download(
            "https://cdn.huggingface.co/deepset/covid_bert_base/vocab.txt",
            out=f"{model_save_path}/")  # download the vocab file

    else:
        model_name = "allenai/biomed_roberta_base"
        model_save_path = "biobert_path"
        os.makedirs(model_save_path, exist_ok=True)
        wget.download(
            "https://cdn.huggingface.co/allenai/biomed_roberta_base/merges.txt",
            out=f"{model_save_path}/")
        wget.download(
            "https://cdn.huggingface.co/allenai/biomed_roberta_base/vocab.json",
            out=f"{model_save_path}/")  # download the vocab file

    bert_model = AutoModel.from_pretrained(model_name)
    bert_model.save_pretrained(model_save_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    del bert_model

    word_embedding_model = models.Transformer(model_save_path)
    shutil.rmtree(model_save_path)
    pooling_model = models.Pooling(768,
                                   pooling_mode_mean_tokens=True,
                                   pooling_mode_cls_token=False,
                                   pooling_mode_max_tokens=False)
    # generating biobert sentence embeddings (mean pooling of sentence
    # embedding vectors)
    sbert_model = SBERTPredictor(
        word_embedding_model,
        pooling_model,
        logistic_model=logistic_model)
    return sbert_model, tokenizer


def train_sbert_model(sbert_model,
                      tokenizer,
                      use_man_con=False,
                      use_med_nli=False,
                      use_multi_nli=False,
                      use_cord=False,
                      multi_nli_train_x: np.ndarray = None,
                      multi_nli_train_y: np.ndarray = None,
                      multi_nli_test_x: np.ndarray = None,
                      multi_nli_test_y: np.ndarray = None,
                      med_nli_train_x: np.ndarray = None,
                      med_nli_train_y: np.ndarray = None,
                      med_nli_test_x: np.ndarray = None,
                      med_nli_test_y: np.ndarray = None,
                      man_con_train_y: np.ndarray = None,
                      man_con_train_x: np.ndarray = None,
                      man_con_test_x: np.ndarray = None,
                      man_con_test_y: np.ndarray = None,
                      cord_train_x: np.ndarray = None,
                      cord_train_y: np.ndarray = None,
                      cord_test_x: np.ndarray = None,
                      cord_test_y: np.ndarray = None,
                      batch_size: int = 2,
                      num_epochs: int = 1,
                      learning_rate: float = 1e-7,
                      embedding_epochs: int = None,
                      enable_class_weights: bool = True,
                      ):
    """Train SBERT on any NLI dataset.

    :param model_name: model to be used, currently supported: covidbert or biobert
    :param tokenizer: the tokenizer corresponding to the model being used"
    :param use_man_con: [description], defaults to False
    :type use_man_con: bool, optional
    :param use_med_nli: [description], defaults to False
    :type use_med_nli: bool, optional
    :param use_multi_nli: [description], defaults to False
    :type use_multi_nli: bool, optional
    :param multi_nli_train_x: [description], defaults to None
    :type multi_nli_train_x: np.ndarray, optional
    :param multi_nli_train_y: [description], defaults to None
    :type multi_nli_train_y: np.ndarray, optional
    :param multi_nli_test_x: [description], defaults to None
    :type multi_nli_test_x: np.ndarray, optional
    :param multi_nli_test_y: [description], defaults to None
    :type multi_nli_test_y: np.ndarray, optional
    :param batch_size: [description], defaults to 2
    :type batch_size: int, optional
    :param num_epochs: [description], defaults to 1
    :type num_epochs: int, optional
    :param learning_rate: defaults to 1e-7
    :type learning_rate: float
    """
    if use_multi_nli:
        if multi_nli_train_x is not None:

            df_multi_train = remove_tokens_get_sentence_sbert(
                multi_nli_train_x, multi_nli_train_y)
            df_multi_val = remove_tokens_get_sentence_sbert(
                multi_nli_test_x, multi_nli_test_y)

            trainer(
                model=sbert_model,
                tokenizer=tokenizer,
                df_train=df_multi_train,
                df_val=df_multi_val,
                epochs=num_epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                embedding_epochs=embedding_epochs,
                enable_class_weights=enable_class_weights)

    if use_med_nli:
        if med_nli_train_x is not None:

            df_mednli_train = remove_tokens_get_sentence_sbert(
                med_nli_train_x, med_nli_train_y)
            df_mednli_val = remove_tokens_get_sentence_sbert(
                med_nli_test_x, med_nli_test_y)

            trainer(
                model=sbert_model,
                tokenizer=tokenizer,
                df_train=df_mednli_train,
                df_val=df_mednli_val,
                epochs=num_epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                embedding_epochs=embedding_epochs,
                enable_class_weights=enable_class_weights)

    if use_man_con:
        if man_con_train_x is not None:

            df_mancon_train = remove_tokens_get_sentence_sbert(
                man_con_train_x, man_con_train_y)
            df_mancon_val = remove_tokens_get_sentence_sbert(
                man_con_test_x, man_con_test_y)

            trainer(
                model=sbert_model,
                tokenizer=tokenizer,
                df_train=df_mancon_train,
                df_val=df_mancon_val,
                epochs=num_epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                embedding_epochs=embedding_epochs,
                enable_class_weights=enable_class_weights)

    if use_cord:
        if cord_train_x is not None:

            df_cord_train = remove_tokens_get_sentence_sbert(
                cord_train_x, cord_train_y)
            df_cord_val = remove_tokens_get_sentence_sbert(
                cord_test_x, cord_test_y)

            trainer(
                model=sbert_model,
                tokenizer=tokenizer,
                df_train=df_cord_train,
                df_val=df_cord_val,
                epochs=num_epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                embedding_epochs=embedding_epochs,
                enable_class_weights=enable_class_weights)

    # return sbert_model


def save_sbert_model(model: SBERTPredictor,
                     timed_dir_name: bool = True,
                     transformer_dir: str = 'output/sbert_model'):
    """Save SBERT trained model.

    :param model: end-to-end SBERT model
    :type model: SBERTPredictor
    :param timed_dir_name: should directory name have time stamp, defaults to True
    :param transformer_dir: directory name, defaults to 'output/sbert_model'
    :type transformer_dir: str, optional
    """
    if timed_dir_name:
        now = datetime.datetime.now()
        transformer_dir = os.path.join(
            transformer_dir, f"{now.month}-{now.day}-{now.year}")

    if not os.path.exists(transformer_dir):
        os.makedirs(transformer_dir)

    pickle.dump(
        model,
        open(
            os.path.join(
                transformer_dir,
                'sigmoid.pickle'),
            "wb"))


def load_sbert_model(transformer_dir: str = 'output/sbert_model',
                     file_name: str = 'sigmoid.pickle'):
    """Load the pickle file containing the model weights.

    :param transformer_dir: folder directory, defaults to 'output/sbert_model'
    :param file_name: file name, defaults to 'sigmoid.pickle'
    :return: SBERT model stored at given location
    :rtype: SBERTPredictor
    """
    sbert_model = pickle.load(
        open(
            os.path.join(
                transformer_dir,
                file_name),
            "rb"))
    return sbert_model
