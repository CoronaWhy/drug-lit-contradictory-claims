"""Functions to build, save, load and train the contradictory claims Transformer model based on biomed_roberta_base."""

# -*- coding: utf-8 -*-

import datetime
import os
import pickle
import shutil
from collections import Counter
from random import randrange

import numpy as np
import tensorflow as tf
import torch
import transformers
import wandb
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import PolynomialDecay
from transformers import AutoModel, AutoModelWithLMHead, AutoTokenizer, TFAutoModel
from wandb.keras import WandbCallback


def regular_encode(texts: list, tokenizer: transformers.AutoTokenizer, maxlen: int = 512,
                   multi_class: bool = True):
    """
    Encode sentences for input to Transformer models.

    :param texts: list of strings to be encoded
    :param tokenizer: tokenizer for encoding
    :param maxlen: max number of characters of input string being encoded
    :param multi_class: if True, the default truncation is applied. If False, implies auxillary input and
        custom truncation is applied.
    :return: numpy array of encoded strings
    """
    # TODO: Intersphinx link to transformers.AutoTokenizer is failing. What's wrong with my docs/source/conf.py?
    if not multi_class:
        # If len > maxlen, truncate text upto maxlen-8 characters and append the 8-character auxillary input
        texts = [text[:maxlen - 8] + text[-8:] if len(text) > maxlen else text for text in texts]

    enc_di = tokenizer.batch_encode_plus(texts,
                                         return_attention_mask=False,
                                         return_token_type_ids=False,
                                         padding='max_length',
                                         # sep_token='[SEP]',
                                         max_length=maxlen,
                                         truncation=True)  # Is this what we want?

    return np.array(enc_di['input_ids'])


def build_model(transformer, max_len: int = 512, multi_class: bool = True, init_learning_rate: float = 1e-6, lr_decay: bool = False):  # noqa: D205
    """
    Build an end-to-end Transformer model. Requires a transformer of type TFAutoBert.
    https://www.kaggle.com/xhlulu/jigsaw-tpu-distilbert-with-huggingface-and-keras

    :param transformer: Transformer model
    :param max_len: maximum length of encoded sequence
    :param multi_class: if True, final layer is multiclass so softmax is used. If False, final layer
        is sigmoid and binary crossentropy is evaluated.
    :param init_learning_rate: initial learning rate
    :param lr_decay: if True, use a learning rate decay schedule. If False, use a constant learning rate.
    :return: Constructed Transformer model
    """
    input_word_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
    sequence_output = transformer(input_word_ids)[0]
    cls_token = sequence_output[:, 0, :]
    if multi_class:
        out = Dense(3, activation='softmax', name='softmax')(cls_token)
    else:
        out = Dense(1, activation='sigmoid', name='sigmoid')(cls_token)

    model = Model(inputs=input_word_ids, outputs=out)

    if lr_decay:
        # There are various options, starting with a linear decay for now
        # TODO: Tune for the best decay schedule
        lr = PolynomialDecay(initial_learning_rate=2e-5,
                             decay_steps=10000,
                             end_learning_rate=1e-6,
                             power=1)
    else:
        lr = init_learning_rate

    if multi_class:
        # NOTE: adding in gradient clipping
        model.compile(Adam(learning_rate=lr, clipvalue=1.0), loss='categorical_crossentropy',
                      metrics=[tf.keras.metrics.Recall(), tf.keras.metrics.Precision(),
                               tf.keras.metrics.CategoricalAccuracy()])
    else:
        model.compile(Adam(learning_rate=lr, clipvalue=1.0), loss='binary_crossentropy',
                      metrics=[tf.keras.metrics.Recall(), tf.keras.metrics.Precision(), 'accuracy'])

    return model


def save_model(model, timed_dir_name: bool = True, transformer_dir: str = 'output/transformer'):
    """
    Save a Keras model that uses a Transformer layer.

    :param model: end-to-end Transformer model
    :param timed_dir_name: if True, save model to a directory where date is recorded
    :param transformer_dir: directory to save model
    :return: None
    """
    if timed_dir_name:
        now = datetime.datetime.now()
        transformer_dir = os.path.join(transformer_dir, f"{now.month}-{now.day}-{now.year}")

    if not os.path.exists(transformer_dir):
        os.makedirs(transformer_dir)

    transformer = model.layers[1]
    transformer.save_pretrained(transformer_dir)
    sigmoid = model.get_layer(index=3).get_weights()
    pickle.dump(sigmoid, open(os.path.join(transformer_dir, 'sigmoid.pickle'), 'wb'))


def load_model(pickle_path: str, transformer_dir: str = 'transformer', max_len: int = 512, multi_class: bool = True):
    """
    Load a Keras model that uses a Transformer layer.

    :param pickle_path: path to pickle file containing learned weights from the last layer
    :param transformer_dir: directory of saved model
    :param max_len: maximum length of encoded sequence
    :param multi_class: if True, final layer is multiclass so softmax is used. If False, final layer
        is sigmoid and binary crossentropy is evaluated.
    :return: loaded model
    """  # is this function overriding the Tensorflow.keras.models function?
    transformer = TFAutoModel.from_pretrained(transformer_dir)
    model = build_model(transformer, max_len=max_len, multi_class=multi_class)
    sigmoid = pickle.load(open(pickle_path, 'rb'))
    if multi_class:
        model.get_layer('softmax').set_weights(sigmoid)
    else:
        model.get_layer('sigmoid').set_weights(sigmoid)

    return model


# def get_class_weights(target_list: list):
#    """Return the class weights to tackle skewness in data while training.
#     :param target_list: list indicating class membership for calculating imbalance
#     :return: list of weights
#     """
#     print(target_list)
#     count_dict = Counter(target_list)
#     class_count = [count_dict[i] for i in range(3)]
#     class_weights = len(target_list) / \
#         torch.tensor(class_count, dtype=torch.float)
#     class_weights = class_weights / len(class_weights)
#     return class_weights.tolist()

def get_class_weights(target_list: np.array, use_class_weights: bool = True):
    """Return the class weights to tackle skewness in data while training.
    :param target_list: numpy array list indicating binary-encoded class membership for calculating imbalance
    :param use_class_weights: if True, calculate class weights based on num. in each class. If False, all weights equals
    :return: list of weights
    """
    n, n_classes = target_list.shape
    if use_class_weights:
        weights = float(n) / target_list.sum(axis=0)
        weights /= n_classes  # 3 typically
    else:
        weights = np.ones(n_classes)
    print(weights)
    weights_dict = dict(zip([0, 1, 2], weights))
    return weights_dict


def train_model(multi_nli_train_x: np.ndarray,
                multi_nli_train_y: np.ndarray,
                multi_nli_test_x: np.ndarray,
                multi_nli_test_y: np.ndarray,
                med_nli_train_x: np.ndarray,
                med_nli_train_y: np.ndarray,
                med_nli_test_x: np.ndarray,
                med_nli_test_y: np.ndarray,
                man_con_train_x: np.ndarray,
                man_con_train_y: np.ndarray,
                man_con_test_x: np.ndarray,
                man_con_test_y: np.ndarray,
                cord_train_x: np.ndarray,
                cord_train_y: np.ndarray,
                cord_test_x: np.ndarray,
                cord_test_y: np.ndarray,
                drug_names: list,  # not using currently...
                virus_names: list,  # not using currently...
                model_name: str,
                out_dir: str,
                multi_class: bool = True,
                continue_fine_tuning: bool = False,
                model_continue_sigmoid_path: str = None,
                model_continue_transformer_path: str = None,
                use_multi_nli: bool = True,
                use_med_nli: bool = True,
                use_man_con: bool = True,
                use_cord: bool = True,
                combined_data_for_training: bool = False,
                epochs: int = 3,
                max_len: int = 512,
                batch_size: int = 32,
                learning_rate: float = 1e-6,
                lr_decay: bool = False,
                class_weights: bool = True):  # currently just always using this...
    """
    Train the Transformer model.

    :param multi_nli_train_x: MultiNLI training sentence pairs
    :param multi_nli_train_y: MultiNLI training labels
    :param multi_nli_test_x: MultiNLI test sentence pairs
    :param multi_nli_test_y: MultiNLI test labels
    :param med_nli_train_x: MedNLI training sentence pairs
    :param med_nli_train_y: MedNLI training labels
    :param med_nli_test_x: MedNLI test sentence pairs
    :param med_nli_test_y: MedNLI test labels
    :param man_con_train_x: ManConCorpus training sentence pairs
    :param man_con_train_y: ManConCorpus training labels
    :param man_con_test_x: ManConCorpus test sentence pairs
    :param man_con_test_y: ManConCorpus test labels
    :param cord_train_x: CORD-19 training sentence pairs
    :param cord_train_y: CORD-19 training labels
    :param cord_test_x: CORD-19 test sentence pairs
    :param cord_test_y: CORD-19 test labels
    :param drug_names: drug lexicon list
    :param virus_names: virus lexicon list
    :param model_name: model name to load from the pre-trained Transformers package. Expecting either
        "deepset/covid_bert_base" or "allenai/biomed_roberta_base"
    :param out_dir: name of directory to output results
    :param multi_class: if True, final layer is multiclass so softmax is used. If False, final layer
        is sigmoid and binary crossentropy is evaluated.
    :param continue_fine_tuning: if True, continue fine tuning from a saved model
    :param model_continue_sigmoid_path: if continue_fine_tuning is True, this is the path to the pickle file with
        saved model weights
    :param model_continue_transformer_path: if continue_fine_tuning is True, this is the directory for the Transformer
        model being loaded
    :param use_multi_nli: if True, use MultiNLI in fine-tuning
    :param use_med_nli: if True, use MedNLI in fine-tuning
    :param use_man_con: if True, use ManConCorpus in fine-tuning
    :param use_cord: if True, use CORD-19 in fine-tuning
    :param combined_data_for_training: if True, use all datasets combined in fine-tuning
    :param epochs: number of epochs for training
    :param max_len: length of encoded inputs
    :param batch_size: batch size
    :param learning_rate: learning rate
    :param lr_decay: if True, use a learning rate decay schedule. If False, use a constant learning rate.
    :param class_weights: if True, use class weights when fine tuning
    :return: fine-tuned Transformer model
    """
    if model_name != 'deepset/covid_bert_base':
        model_name = 'allenai/biomed_roberta_base'

    # First load the real tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if use_multi_nli or combined_data_for_training:
        multi_nli_train_x_str = [str(sen) for sen in multi_nli_train_x]
        multi_nli_train_x = regular_encode(multi_nli_train_x_str, tokenizer, maxlen=max_len, multi_class=multi_class)
        print("Done with multi_nli_train_x_str")  # noqa: T001

        multi_nli_test_x_str = [str(sen) for sen in multi_nli_test_x]
        multi_nli_test_x = regular_encode(multi_nli_test_x_str, tokenizer, maxlen=max_len, multi_class=multi_class)
        print("Done with multi_nli_test_x_str")  # noqa: T001

    if use_med_nli or combined_data_for_training:
        med_nli_train_x_str = [str(sen) for sen in med_nli_train_x]
        med_nli_train_x = regular_encode(med_nli_train_x_str, tokenizer, maxlen=max_len, multi_class=multi_class)
        print("Done with med_nli_train_x_str")  # noqa: T001

        med_nli_test_x_str = [str(sen) for sen in med_nli_test_x]
        med_nli_test_x = regular_encode(med_nli_test_x_str, tokenizer, maxlen=max_len, multi_class=multi_class)
        print("Done with med_nli_test_x_str")  # noqa: T001

    if use_man_con or combined_data_for_training:
        man_con_train_x_str = [str(sen) for sen in man_con_train_x]
        man_con_train_x = regular_encode(man_con_train_x_str, tokenizer, maxlen=max_len, multi_class=multi_class)
        print("Done with man_con_train_x_str")  # noqa: T001

        man_con_test_x_str = [str(sen) for sen in man_con_test_x]
        man_con_test_x = regular_encode(man_con_test_x_str, tokenizer, maxlen=max_len, multi_class=multi_class)
        print("Done with man_con_test_x_str")  # noqa: T001

    if use_cord or combined_data_for_training:
        cord_train_x_str = [str(sen) for sen in cord_train_x]
        cord_train_x = regular_encode(cord_train_x_str, tokenizer, maxlen=max_len, multi_class=multi_class)
        print("Done with cord_train_x_str")  # noqa: T001

        cord_test_x_str = [str(sen) for sen in cord_test_x]
        cord_test_x = regular_encode(cord_test_x_str, tokenizer, maxlen=max_len, multi_class=multi_class)
        print("Done with cord_test_x_str")  # noqa: T001

    es = EarlyStopping(monitor='val_accuracy',
                       min_delta=0.001,
                       patience=3,
                       verbose=1,
                       mode='max',
                       restore_best_weights=True)

    ###strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())
    now = datetime.datetime.now()
    if continue_fine_tuning:
        ###with strategy.scope():
        model = load_model(model_continue_sigmoid_path, model_continue_transformer_path, multi_class=multi_class)
        #batch_size = 2 * strategy.num_replicas_in_sync

    else:
        if model_name == 'deepset/covid_bert_base':
            model = AutoModelWithLMHead.from_pretrained("deepset/covid_bert_base")
            model.resize_token_embeddings(len(tokenizer))
            ri = randrange(1000)
            tmp_dir = f"covid_bert_base-{now.day}_{now.month}_{now.year}-{now.hour}:{now.minute}:{now.second}_{ri}"
            if os.path.exists(tmp_dir):
                raise Exception("Directory conflict when saving model temporarily!")
            os.makedirs(tmp_dir)
            model.save_pretrained(tmp_dir)
            ###with strategy.scope():
            model = TFAutoModel.from_pretrained(tmp_dir, from_pt=True)  #indented
            model = build_model(model, init_learning_rate=learning_rate)  #indented
            shutil.rmtree(tmp_dir)
        else:
            now = datetime.datetime.now()
            model = AutoModel.from_pretrained("allenai/biomed_roberta_base")
            model.resize_token_embeddings(len(tokenizer))
            ri = randrange(1000)
            tmp_dir = f"biomed_roberta_base-{now.day}_{now.month}_{now.year}-{now.hour}:{now.minute}:{now.second}_{ri}"
            if os.path.exists(tmp_dir):
                raise Exception("Directory conflict when saving model temporarily!")
            os.makedirs(tmp_dir)
            model.save_pretrained(tmp_dir)
            ###with strategy.scope():  #next 2 lines were indented before
            model = TFAutoModel.from_pretrained(tmp_dir, from_pt=True)
            model = build_model(model, multi_class=multi_class, init_learning_rate=learning_rate, lr_decay=lr_decay)
            shutil.rmtree(tmp_dir)
        #batch_size = 2 * strategy.num_replicas_in_sync

    print(model.summary())  # noqa: T001

    print("Okay now it's training time.......\n\n\n")  # noqa: T001
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    if tf.test.gpu_device_name():
        print(f'Default GPU Device:{tf.test.gpu_device_name()}')  # noqa: T001
    else:
        print("Please install GPU version of TF")  # noqa: T001

    # Initialize WandB for tracking the training progress
    wandb_dir = f"{out_dir}/wandb_artifacts"
    if not os.path.exists(wandb_dir):
        os.makedirs(wandb_dir)
    wandb.init(dir=f"{out_dir}/wandb_artifacts")

    train_hist_list = []

    if combined_data_for_training:
        # Combine everything

        combined_train_x = np.concatenate((multi_nli_train_x, med_nli_train_x, man_con_train_x, cord_train_x), axis=0)
        combined_train_y = np.concatenate((multi_nli_train_y, med_nli_train_y, man_con_train_y, cord_train_y), axis=0)
        train_idx_shuff = np.random.permutation(len(combined_train_x))
        combined_train_x = combined_train_x[train_idx_shuff]
        combined_train_y = combined_train_y[train_idx_shuff]

        combined_test_x = np.concatenate((multi_nli_test_x, med_nli_test_x, man_con_test_x, cord_test_x), axis=0)
        combined_test_y = np.concatenate((multi_nli_test_y, med_nli_test_y, man_con_test_y, cord_test_y), axis=0)
        test_idx_shuff = np.random.permutation(len(combined_test_x))
        combined_test_x = combined_test_x[test_idx_shuff]
        combined_test_y = combined_test_y[test_idx_shuff]

        train_history = model.fit(combined_train_x,
                                  combined_train_y,
                                  batch_size=batch_size,
                                  validation_data=(combined_test_x, combined_test_y),
                                  callbacks=[es, WandbCallback()],
                                  epochs=epochs) #,
                                  # class_weight=get_class_weights(combined_train_y, use_class_weights=class_weights))
        train_hist_list.append(train_history)

        print("passed the multiNLI train. Now the history:")  # noqa: T001
        print(train_history)  # noqa: T001

    else:
        # Fine tune on MultiNLI
        if use_multi_nli:
            train_history = model.fit(multi_nli_train_x,
                                      multi_nli_train_y,
                                      batch_size=batch_size,
                                      validation_data=(multi_nli_test_x, multi_nli_test_y),
                                      callbacks=[es, WandbCallback()],
                                      epochs=epochs) #,
                                      # class_weight=get_class_weights(multi_nli_train_y, use_class_weights=class_weights))
            train_hist_list.append(train_history)

            print("passed the multiNLI train. Now the history:")  # noqa: T001
            print(train_history)  # noqa: T001
            print("NOW TO EVALUATE:")  #noqa: T001
            print(model.evaluate())

        # Fine tune on MedNLI
        if use_med_nli:
            train_history = model.fit(med_nli_train_x,
                                      med_nli_train_y,
                                      batch_size=batch_size,
                                      validation_data=(med_nli_test_x, med_nli_test_y),
                                      callbacks=[es, WandbCallback()],
                                      epochs=epochs) #,
                                      # class_weight=get_class_weights(med_nli_train_y, use_class_weights=class_weights))
            train_hist_list.append(train_history)
            print("NOW TO EVALUATE:")  #noqa: T001
            print(model.evaluate())

        # Fine tune on ManConCorpus
        if use_man_con:
            train_history = model.fit(man_con_train_x,
                                      man_con_train_y,
                                      batch_size=batch_size,
                                      validation_data=(man_con_test_x, man_con_test_y),
                                      callbacks=[es, WandbCallback()],
                                      epochs=epochs) #,
                                      # class_weight=get_class_weights(man_con_train_y, use_class_weights=class_weights))
            train_hist_list.append(train_history)
            print("NOW TO EVALUATE:")  #noqa: T001
            print(model.evaluate())

        # Fine tune on CORD-19
        if use_cord:
            train_history = model.fit(cord_train_x,
                                      cord_train_y,
                                      batch_size=batch_size,
                                      validation_data=(cord_test_x, cord_test_y),
                                      callbacks=[es, WandbCallback()],
                                      epochs=epochs) #,
                                      # class_weight=get_class_weights(cord_train_y, use_class_weights=class_weights))
            train_hist_list.append(train_history)

    return model, train_hist_list
