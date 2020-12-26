"""Function for loading various datasets used in training contradictory-claims BERT model."""

# -*- coding: utf-8 -*-

import json

import numpy as np
import pandas as pd
from contradictory_claims.models.evaluate_model import read_data_from_excel
from keras.utils import np_utils
from sklearn.model_selection import train_test_split


def load_multi_nli(train_path: str, test_path: str, multi_class: bool = True):
    """
    Load MultiNLI data for training.

    :param train_path: path to MultiNLI training data
    :param test_path: path to MultiNLI test data
    :param multi_class: if True, data is prepared for multiclass classification. If False, implies auxillary input
        and data is prepared for binary classification.
    :return: MultiNLI sentence pairs and labels for training and test sets, respectively
    """
    multinli_train_data = pd.read_csv(train_path, sep='\t', error_bad_lines=False)
    multinli_test_data = pd.read_csv(test_path, sep='\t', error_bad_lines=False)

    # Map labels to numerical (categorical) values
    multinli_train_data['gold_label'] = [2 if label == 'contradiction' else 1 if label == 'entailment' else 0 for
                                         label in multinli_train_data.gold_label]
    multinli_test_data['gold_label'] = [2 if label == 'contradiction' else 1 if label == 'entailment' else 0 for
                                        label in multinli_test_data.gold_label]

    # Insert the CLS and SEP tokens
    if multi_class:
        x_train = '[CLS]' + multinli_train_data.sentence1 + '[SEP]' + multinli_train_data.sentence2
        x_train = x_train.to_numpy()
        x_test = '[CLS]' + multinli_test_data.sentence1 + '[SEP]' + multinli_test_data.sentence2
        x_test = x_test.to_numpy()

        # Reformat to one-hot categorical variable (3 columns)
        y_train = np_utils.to_categorical(multinli_train_data.gold_label, dtype='int')
        y_test = np_utils.to_categorical(multinli_test_data.gold_label, dtype='int')
    else:
        # Add the category info (CON, ENT, NEU) as auxillary text at the end
        x_train_1 = '[CLS]' + multinli_train_data.sentence1 + '[SEP]' + multinli_train_data.sentence2 +\
                    '[SEP]' + 'CON'
        x_train_2 = '[CLS]' + multinli_train_data.sentence1 + '[SEP]' + multinli_train_data.sentence2 +\
                    '[SEP]' + 'ENT'
        x_train_3 = '[CLS]' + multinli_train_data.sentence1 + '[SEP]' + multinli_train_data.sentence2 +\
                    '[SEP]' + 'NEU'
        x_train = x_train_1.append(x_train_2)
        x_train = x_train.append(x_train_3)
        x_train = x_train.to_numpy()
        x_test_1 = '[CLS]' + multinli_test_data.sentence1 + '[SEP]' + multinli_test_data.sentence2 +\
                   '[SEP]' + 'CON'
        x_test_2 = '[CLS]' + multinli_test_data.sentence1 + '[SEP]' + multinli_test_data.sentence2 +\
                   '[SEP]' + 'ENT'
        x_test_3 = '[CLS]' + multinli_test_data.sentence1 + '[SEP]' + multinli_test_data.sentence2 +\
                   '[SEP]' + 'NEU'
        x_test = x_test_1.append(x_test_2)
        x_test = x_test.append(x_test_3)
        x_test = x_test.to_numpy()

        # Reformat to binary variable
        y_train_1 = [1 if label == 2 else 0 for label in multinli_train_data.gold_label]
        y_train_2 = [1 if label == 1 else 0 for label in multinli_train_data.gold_label]
        y_train_3 = [1 if label == 0 else 0 for label in multinli_train_data.gold_label]
        y_train = y_train_1 + y_train_2 + y_train_3
        y_train = np.array(y_train)
        y_test_1 = [1 if label == 2 else 0 for label in multinli_test_data.gold_label]
        y_test_2 = [1 if label == 1 else 0 for label in multinli_test_data.gold_label]
        y_test_3 = [1 if label == 0 else 0 for label in multinli_test_data.gold_label]
        y_test = y_test_1 + y_test_2 + y_test_3
        y_test = np.array(y_test)

    return x_train, y_train, x_test, y_test


def load_med_nli(train_path: str, dev_path: str, test_path: str, num_training_pairs_per_class: int = None,
                 multi_class: bool = True):
    """
    Load MedNLI data for training.

    :param train_path: path to MedNLI training data
    :param dev_path: path to MedNLI dev data
    :param test_path: path to MedNLI test data
    :param num_training_pairs_per_class: number of pairs of sentences to retrieve per class.
        If None, all sentence pairs are retrieved
    :param multi_class: if True, data is prepared for multiclass classification. If False, implies auxillary input
        and data is prepared for binary classification.
    :return: MedNLI sentence pairs and labels for training and test sets, respectively
    """
    # Question: how do you have long docstrings that violate character limit but not get flake8 on my case
    # about splitting into 2 lines? TODO: Figure this out
    mednli_data1 = pd.DataFrame()
    mednli_data2 = pd.DataFrame()
    mednli_test_data = pd.DataFrame()

    with open(train_path, 'r', encoding='utf-8') as f:
        for line in f:
            mednli_data1 = mednli_data1.append(json.loads(line.rstrip('\n|\r')), ignore_index=True)
    with open(dev_path, 'r', encoding='utf-8') as f:
        for line in f:
            mednli_data2 = mednli_data2.append(json.loads(line.rstrip('\n|\r')), ignore_index=True)

    # Join together training and dev sets
    mednli_data = mednli_data1.append(mednli_data2, ignore_index=True).reset_index(drop=True)

    with open(test_path, 'r', encoding='utf-8') as f:
        for line in f:
            mednli_test_data = mednli_test_data.append(json.loads(line.rstrip('\n|\r')), ignore_index=True)

    # Map labels to numerical (categorical) values
    mednli_data['gold_label'] = [2 if label == 'contradiction' else 1 if label == 'entailment' else 0 for
                                 label in mednli_data.gold_label]
    mednli_test_data['gold_label'] = [2 if label == 'contradiction' else 1 if label == 'entailment' else 0 for
                                      label in mednli_test_data.gold_label]

    # Number of training pairs per class to use. If None, use all training pairs
    if num_training_pairs_per_class is not None:
        print(f'Using only a subset of MedNLI for training: {num_training_pairs_per_class} training pairs per class')  # noqa: T001,E501
        temp = mednli_data[mednli_data.gold_label == 2].head(num_training_pairs_per_class).append(
            mednli_data[mednli_data.gold_label == 1].head(num_training_pairs_per_class)).reset_index(drop=True)
        mednli_data = temp.append(mednli_data[mednli_data.gold_label == 0].head(num_training_pairs_per_class))\
            .reset_index(drop=True)

    # Insert the CLS and SEP tokens
    if multi_class:
        x_train = '[CLS]' + mednli_data.sentence1 + '[SEP]' + mednli_data.sentence2
        x_train = x_train.to_numpy()
        x_test = '[CLS]' + mednli_test_data.sentence1 + '[SEP]' + mednli_test_data.sentence2
        x_test = x_test.to_numpy()

        # Reformat to one-hot categorical variable (3 columns)
        y_train = np_utils.to_categorical(mednli_data.gold_label)
        y_test = np_utils.to_categorical(mednli_test_data.gold_label)
    else:
        # Add the category info (CON, ENT, NEU) as auxillary text at the end
        x_train_1 = '[CLS]' + mednli_data.sentence1 + '[SEP]' + mednli_data.sentence2 +\
                    '[SEP]' + 'CON'
        x_train_2 = '[CLS]' + mednli_data.sentence1 + '[SEP]' + mednli_data.sentence2 +\
                    '[SEP]' + 'ENT'
        x_train_3 = '[CLS]' + mednli_data.sentence1 + '[SEP]' + mednli_data.sentence2 +\
                    '[SEP]' + 'NEU'
        x_train = x_train_1.append(x_train_2)
        x_train = x_train.append(x_train_3)
        x_train = x_train.to_numpy()
        x_test_1 = '[CLS]' + mednli_test_data.sentence1 + '[SEP]' + mednli_test_data.sentence2 +\
                   '[SEP]' + 'CON'
        x_test_2 = '[CLS]' + mednli_test_data.sentence1 + '[SEP]' + mednli_test_data.sentence2 +\
                   '[SEP]' + 'ENT'
        x_test_3 = '[CLS]' + mednli_test_data.sentence1 + '[SEP]' + mednli_test_data.sentence2 +\
                   '[SEP]' + 'NEU'
        x_test = x_test_1.append(x_test_2)
        x_test = x_test.append(x_test_3)
        x_test = x_test.to_numpy()

        # Reformat to binary variable
        y_train_1 = [1 if label == 2 else 0 for label in mednli_data.gold_label]
        y_train_2 = [1 if label == 1 else 0 for label in mednli_data.gold_label]
        y_train_3 = [1 if label == 0 else 0 for label in mednli_data.gold_label]
        y_train = y_train_1 + y_train_2 + y_train_3
        y_train = np.array(y_train)
        y_test_1 = [1 if label == 2 else 0 for label in mednli_test_data.gold_label]
        y_test_2 = [1 if label == 1 else 0 for label in mednli_test_data.gold_label]
        y_test_3 = [1 if label == 0 else 0 for label in mednli_test_data.gold_label]
        y_test = y_test_1 + y_test_2 + y_test_3
        y_test = np.array(y_test)

    return x_train, y_train, x_test, y_test


def load_mancon_corpus_from_sent_pairs(mancon_sent_pair_path: str,
                                       multi_class: bool = True):  # noqa: D205,D400
    """
    Load ManConCorpus data. NOTE: this data must be preprocessed as sentence pairs. This format is a TSV with four
        columns: label, guid, text_a (sentence 1), and text_b (sentence 2).

    :param mancon_sent_pair_path: path to ManCon sentence pair file
    :param multi_class: if True, data is prepared for multiclass classification. If False, implies auxillary input
        and data is prepared for binary classification.
    :return: ManConCorpus sentence pairs and labels for training and test sets, respectively
    """
    mancon_data = pd.read_csv(mancon_sent_pair_path, sep='\t')
    mancon_data['label'] = [2 if label == 'contradiction' else 1 if label == 'entailment' else 0 for
                            label in mancon_data.label]
    print(f"Number of contradiction pairs: {len(mancon_data[mancon_data.label == 2])}")  # noqa: T001
    print(f"Number of entailment pairs: {len(mancon_data[mancon_data.label == 1])}")  # noqa: T001
    print(f"Number of neutral pairs: {len(mancon_data[mancon_data.label == 0])}")  # noqa: T001

    # Insert the CLS and SEP tokens
    x_train, x_test, y_train_tmp, y_test_tmp = train_test_split(
        '[CLS]' + mancon_data.text_a + '[SEP]' + mancon_data.text_b, mancon_data['label'], test_size=0.2,
        stratify=mancon_data['label'])
    if multi_class:
        x_train = x_train.to_numpy()  # TODO: need to double check this is sufficient for not having TF complain
        x_test = x_test.to_numpy()

        # Reformat to one-hot categorical variable (3 columns)
        y_train = np_utils.to_categorical(y_train_tmp)
        y_test = np_utils.to_categorical(y_test_tmp)
    else:
        # Add the category info (CON, ENT, NEU) as auxillary text at the end
        x_train_1 = x_train + '[SEP]' + 'CON'
        x_train_2 = x_train + '[SEP]' + 'ENT'
        x_train_3 = x_train + '[SEP]' + 'NEU'
        x_train = x_train_1.append(x_train_2)
        x_train = x_train.append(x_train_3)
        x_train = x_train.to_numpy()
        x_test_1 = x_test + '[SEP]' + 'CON'
        x_test_2 = x_test + '[SEP]' + 'ENT'
        x_test_3 = x_test + '[SEP]' + 'NEU'
        x_test = x_test_1.append(x_test_2)
        x_test = x_test.append(x_test_3)
        x_test = x_test.to_numpy()

        # Reformat to binary variable
        y_train_1 = [1 if label == 2 else 0 for label in y_train_tmp]
        y_train_2 = [1 if label == 1 else 0 for label in y_train_tmp]
        y_train_3 = [1 if label == 0 else 0 for label in y_train_tmp]
        y_train = y_train_1 + y_train_2 + y_train_3
        y_train = np.array(y_train)
        y_test_1 = [1 if label == 2 else 0 for label in y_test_tmp]
        y_test_2 = [1 if label == 1 else 0 for label in y_test_tmp]
        y_test_3 = [1 if label == 0 else 0 for label in y_test_tmp]
        y_test = y_test_1 + y_test_2 + y_test_3
        y_test = np.array(y_test)

    return x_train, y_train, x_test, y_test


def load_drug_virus_lexicons(drug_lex_path: str, virus_lex_path: str):
    """
    Load drug and virus lexicons.

    :param drug_lex_path: path to drug lexicon
    :param virus_lex_path: path to COVID-19 lexicon
    :return: lists representing drug lexicon and virus lexicon
    """
    drug_names = pd.read_csv(drug_lex_path, header=None)
    drug_names = list(drug_names[0])
    drug_names = [drug.lower() for drug in drug_names]
    print(f"{len(drug_names)} unique drugs found in training & testing corpus:")  # noqa: T001

    virus_names = pd.read_csv(virus_lex_path, header=None)
    virus_names = list(virus_names[0])

    return drug_names, virus_names


def remove_tokens_get_sentence_sbert(x: np.ndarray, y: np.ndarray):
    """Convert Data recieved as a single format by preprocessing multi_nli, med_nli or mancon.

    :param x: array containing output from one of the above functions
    :param y: array containing labels from one of the above functions

    :return: dataframe containing sentences and labels as different columnn
    """
    x_df = pd.DataFrame(x, columns=["sentences"])
    x_df["sentences"] = x_df["sentences"].astype(str)
    x_df["sentences"] = x_df["sentences"].apply(lambda x: x.replace("[CLS]", ""))
    x_df["sentence1"] = x_df["sentences"].apply(lambda x: x.split("[SEP]")[0])
    x_df["sentence2"] = x_df["sentences"].apply(lambda x: x.split("[SEP]")[-1])
    x_df.drop(["sentences"], axis=1, inplace=True)
    y_transformed = np.argmax(y, axis=1)
    # {"contradiction": 2, "entailment": 1, "neutral": 0},
    # for sbert need to change this to entail:2, contra:0, neut:1
    y_df = pd.DataFrame(y_transformed, columns=["label"])
    convert_dict = {0: 1, 1: 2, 2: 0}
    y_df["label"] = y_df["label"].apply(lambda x: convert_dict[x]).astype(int)
    df = pd.concat([x_df, y_df], axis=1)
    return df


def load_cord_pairs(data_path: str, active_sheet: str, multi_class: bool = True):
    """
    Load CORD-19 annotated claim pairs for training.

    :param data_path: path to CORD training data
    :param active_sheet: name of active sheet with data
    :param multi_class: if True, data is prepared for multiclass classification. If False, implies auxillary input
        and data is prepared for binary classification.
    :return: CORD-19 sentence pairs and labels for training and test sets, respectively
    """
    cord_data = read_data_from_excel(data_path, active_sheet)

    cord_data['label'] = [2 if label == 'contradiction' else 1 if label == 'entailment' else 0 for
                          label in cord_data.annotation]
    print(f"Number of contradiction pairs: {len(cord_data[cord_data.label == 2])}")  # noqa: T001
    print(f"Number of entailment pairs: {len(cord_data[cord_data.label == 1])}")  # noqa: T001
    print(f"Number of neutral pairs: {len(cord_data[cord_data.label == 0])}")  # noqa: T001

    # Insert the CLS and SEP tokens
    x_train, x_test, y_train_tmp, y_test_tmp = train_test_split(
        '[CLS]' + cord_data.text1 + '[SEP]' + cord_data.text2, cord_data['label'], test_size=0.2,
        stratify=cord_data['label'])
    if multi_class:
        x_train = x_train.to_numpy()  # TODO: need to double check this is sufficient for not having TF complain
        x_test = x_test.to_numpy()

        # Reformat to one-hot categorical variable (3 columns)
        y_train = np_utils.to_categorical(y_train_tmp)
        y_test = np_utils.to_categorical(y_test_tmp)
    else:
        # Add the category info (CON, ENT, NEU) as auxillary text at the end
        x_train_1 = x_train + '[SEP]' + 'CON'
        x_train_2 = x_train + '[SEP]' + 'ENT'
        x_train_3 = x_train + '[SEP]' + 'NEU'
        x_train = x_train_1.append(x_train_2)
        x_train = x_train.append(x_train_3)
        x_train = x_train.to_numpy()
        x_test_1 = x_test + '[SEP]' + 'CON'
        x_test_2 = x_test + '[SEP]' + 'ENT'
        x_test_3 = x_test + '[SEP]' + 'NEU'
        x_test = x_test_1.append(x_test_2)
        x_test = x_test.append(x_test_3)
        x_test = x_test.to_numpy()

        # Reformat to binary variable
        y_train_1 = [1 if label == 2 else 0 for label in y_train_tmp]
        y_train_2 = [1 if label == 1 else 0 for label in y_train_tmp]
        y_train_3 = [1 if label == 0 else 0 for label in y_train_tmp]
        y_train = y_train_1 + y_train_2 + y_train_3
        y_train = np.array(y_train)
        y_test_1 = [1 if label == 2 else 0 for label in y_test_tmp]
        y_test_2 = [1 if label == 1 else 0 for label in y_test_tmp]
        y_test_3 = [1 if label == 0 else 0 for label in y_test_tmp]
        y_test = y_test_1 + y_test_2 + y_test_3
        y_test = np.array(y_test)

    return x_train, y_train, x_test, y_test
