"""Function for loading various datasets used in training contradictory-claims BERT model."""

# -*- coding: utf-8 -*-

import json
import os
import xml.etree.ElementTree as ET  # TODO: Fix error # noqa: S405, N817
from itertools import combinations
from typing import List

import numpy as np
import pandas as pd
from contradictory_claims.models.evaluate_model import read_data_from_excel
from keras.utils import np_utils
from sklearn.model_selection import train_test_split


def replace_drug_with_spl_token(text: List[str], drug_names: List[str] = None):
    """
    Replace drug name with a special token.

    :param text: list of input text containing drug name
    :param drug_names: list of drug names to replace
    :return: Text with all drug occurrances replaced by special token
    """
    text_out = text
    for drug in drug_names:
        text_out = [str(t).replace(drug, "$drug$") for t in text_out]

    return text_out


def load_multi_nli(train_path: str, test_path: str, drug_names: List[str] = None, multi_class: bool = True,
                   repl_drug_with_spl_tkn: bool = False, downsample: float = 0.1):
    """
    Load MultiNLI data for training.

    :param train_path: path to MultiNLI training data
    :param test_path: path to MultiNLI test data
    :param multi_class: if True, data is prepared for multiclass classification. If False, implies auxillary input
        and data is prepared for binary classification.
    :param drug_names: list of drug names to replace
    :param repl_drug_with_spl_tkn: if True, replace drug names with a special token
    :param downsample: fraction to downsample MultiNLI to facilitate training
    :return: MultiNLI sentence pairs and labels for training and test sets, respectively
    """
    # If not drop NaNs in sentences now, could lead to problems when encoding
    multinli_train_data = pd.read_csv(train_path, sep='\t', error_bad_lines=False).dropna(subset=["sentence1", "sentence2"])
    multinli_test_data = pd.read_csv(test_path, sep='\t', error_bad_lines=False).dropna(subset=["sentence1", "sentence2"])

    # Map labels to numerical (categorical) values
    multinli_train_data['gold_label'] = [2 if label == 'contradiction' else 1 if label == 'entailment' else 0 for
                                         label in multinli_train_data.gold_label]
    multinli_test_data['gold_label'] = [2 if label == 'contradiction' else 1 if label == 'entailment' else 0 for
                                        label in multinli_test_data.gold_label]

    # Insert the CLS and SEP tokens
    if multi_class:
        x_train = '[CLS]' + multinli_train_data.sentence1 + '[SEP]' + multinli_train_data.sentence2
        x_test = '[CLS]' + multinli_test_data.sentence1 + '[SEP]' + multinli_test_data.sentence2

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
        x_test_1 = '[CLS]' + multinli_test_data.sentence1 + '[SEP]' + multinli_test_data.sentence2 +\
                   '[SEP]' + 'CON'
        x_test_2 = '[CLS]' + multinli_test_data.sentence1 + '[SEP]' + multinli_test_data.sentence2 +\
                   '[SEP]' + 'ENT'
        x_test_3 = '[CLS]' + multinli_test_data.sentence1 + '[SEP]' + multinli_test_data.sentence2 +\
                   '[SEP]' + 'NEU'
        x_test = x_test_1.append(x_test_2)
        x_test = x_test.append(x_test_3)

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

    # Replace drug name occurence with special token
    if repl_drug_with_spl_tkn:
        x_train = replace_drug_with_spl_token(x_train, drug_names)
        x_test = replace_drug_with_spl_token(x_test, drug_names)

    x_train = np.array(x_train)
    x_test = np.array(x_test)

    return x_train, y_train, x_test, y_test


def load_med_nli(train_path: str, dev_path: str, test_path: str, drug_names: List[str] = None,
                 num_training_pairs_per_class: int = None,
                 multi_class: bool = True, repl_drug_with_spl_tkn: bool = False):
    """
    Load MedNLI data for training.

    :param train_path: path to MedNLI training data
    :param dev_path: path to MedNLI dev data
    :param test_path: path to MedNLI test data
    :param num_training_pairs_per_class: number of pairs of sentences to retrieve per class.
        If None, all sentence pairs are retrieved
    :param multi_class: if True, data is prepared for multiclass classification. If False, implies auxillary input
        and data is prepared for binary classification.
    :param drug_names: list of drug names to replace
    :param repl_drug_with_spl_tkn: if True, replace drug names with a special token
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

    print("TRAINING:")  # noqa: T001
    print(f"Number of contradiction pairs: {len(mednli_data[mednli_data.label == 2])}")  # noqa: T001
    print(f"Number of entailment pairs: {len(mednli_data[mednli_data.label == 1])}")  # noqa: T001
    print(f"Number of neutral pairs: {len(mednli_data[mednli_data.label == 0])}")  # noqa: T001

    print("\nTEST")  # noqa: T001
    print(f"Number of contradiction pairs: {len(mednli_data[mednli_data.label == 2])}")  # noqa: T001
    print(f"Number of entailment pairs: {len(mednli_data[mednli_data.label == 1])}")  # noqa: T001
    print(f"Number of neutral pairs: {len(mednli_data[mednli_data.label == 0])}")  # noqa: T001

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
        x_test = '[CLS]' + mednli_test_data.sentence1 + '[SEP]' + mednli_test_data.sentence2

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
        x_test_1 = '[CLS]' + mednli_test_data.sentence1 + '[SEP]' + mednli_test_data.sentence2 +\
                   '[SEP]' + 'CON'
        x_test_2 = '[CLS]' + mednli_test_data.sentence1 + '[SEP]' + mednli_test_data.sentence2 +\
                   '[SEP]' + 'ENT'
        x_test_3 = '[CLS]' + mednli_test_data.sentence1 + '[SEP]' + mednli_test_data.sentence2 +\
                   '[SEP]' + 'NEU'
        x_test = x_test_1.append(x_test_2)
        x_test = x_test.append(x_test_3)

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

    # Replace drug name occurence with special token
    if repl_drug_with_spl_tkn:
        x_train = replace_drug_with_spl_token(x_train, drug_names)
        x_test = replace_drug_with_spl_token(x_test, drug_names)

    x_train = np.array(x_train)
    x_test = np.array(x_test)

    return x_train, y_train, x_test, y_test


def create_mancon_sent_pairs_from_xml(xml_path: str, save_path: str, eval_data_save_path: str):
    """
    Create sentence pairs dataset from the original xml ManCon Corpus.

    :param xml_path: path to xml corpus
    :param save_path: path to save the sentence pairs dataset
    :param eval_data_save_path: path to save the evaluation version of manconcorpus (for benchmarking)
    """
    xtree = ET.parse(xml_path)  # TODO: Fix error # noqa: S314
    xroot = xtree.getroot()

    manconcorpus_data = pd.DataFrame(columns=['claim', 'assertion', 'question'])

    for node in xroot:
        for claim in node.findall('CLAIM'):
            manconcorpus_data = manconcorpus_data.append({'claim': claim.text,
                                                          'assertion': claim.attrib.get('ASSERTION'),
                                                          'question': claim.attrib.get('QUESTION')},
                                                         ignore_index=True)

    # Going to output a version of this to use as evaluation data for benchmarking...
    mcc_eval_data = manconcorpus_data.rename(columns={"question": "text1", "claim": "text2", "assertion": "annotation"})
    mcc_eval_data["annotation"] = mcc_eval_data["annotation"].str.replace('YS', 'entailment')
    mcc_eval_data["annotation"] = mcc_eval_data["annotation"].str.replace('NO', 'contradiction')
    mcc_eval_data.to_csv(eval_data_save_path, sep='\t', index=False)
    # print(len(manconcorpus_data))

    questions = list(set(manconcorpus_data.question))
    con = pd.DataFrame(columns=['text_a', 'text_b', 'label'])
    ent = pd.DataFrame(columns=['text_a', 'text_b', 'label'])
    neu = pd.DataFrame(columns=['text_a', 'text_b', 'label'])

    for q in questions:
        claim_yes = pd.DataFrame(manconcorpus_data.loc[(manconcorpus_data.question == q)
                                                       & (manconcorpus_data.assertion == 'YS'), 'claim'])  # noqa: W503
        claim_no = pd.DataFrame(manconcorpus_data.loc[(manconcorpus_data.question == q)
                                                      & (manconcorpus_data.assertion == 'NO'), 'claim'])  # noqa: W503
        temp = claim_yes.assign(key=1).merge(claim_no.assign(key=1), on='key').drop('key', 1)
        temp1 = temp.rename(columns={'claim_x': 'text_a', 'claim_y': 'text_b'})
        con = con.append(temp1)
        con['label'] = 'contradiction'
        con.drop_duplicates(inplace=True)

        for i, j in list(combinations(claim_yes.index, 2)):
            ent = ent.append({'text_a': claim_yes.claim[i],
                              'text_b': claim_yes.claim[j],
                              'label': 'entailment'},
                             ignore_index=True)

        for i, j in list(combinations(claim_no.index, 2)):
            ent = ent.append({'text_a': claim_no.claim[i],
                              'text_b': claim_no.claim[j],
                              'label': 'entailment'},
                             ignore_index=True)

        claim1 = pd.DataFrame(manconcorpus_data.loc[(manconcorpus_data.question == q), 'claim'])
        claim2 = pd.DataFrame(manconcorpus_data.loc[(manconcorpus_data.question != q), 'claim'])
        temp = claim1.assign(key=1).merge(claim2.assign(key=1), on='key').drop('key', 1)
        temp1 = temp.rename(columns={'claim_x': 'text_a', 'claim_y': 'text_b'})
        neu = neu.append(temp1)
        neu['label'] = 'neutral'
        neu.drop_duplicates(inplace=True)

    transfer_data = pd.concat([con, ent, neu]).reset_index(drop=True)
    transfer_data['guid'] = transfer_data.index
    # print(len(con))
    # print(len(ent))
    # print(len(neu))
    # print(len(transfer_data))

    # Ensuring the file doesn't already exist to avoid conflicts from multiple jobs creating this file.
    if not os.path.exists(save_path):
        transfer_data.to_csv(save_path, sep='\t', index=False)

    # Might as well also return this data, although it gets saved to a TSV too.
    return mcc_eval_data


def load_mancon_corpus_from_sent_pairs(mancon_sent_pair_path: str,
                                       drug_names: List[str] = None,
                                       multi_class: bool = True,  # noqa: D205,D400
                                       repl_drug_with_spl_tkn: bool = False,
                                       downsample_neutrals: bool = True):
    """
    Load ManConCorpus data.

    NOTE: This data must be preprocessed as sentence pairs. This format is a TSV with four
    columns: label, guid, text_a (sentence 1), and text_b (sentence 2).

    :param mancon_sent_pair_path: path to ManCon sentence pair file
    :param multi_class: if True, data is prepared for multiclass classification. If False, implies auxiliary input
        and data is prepared for binary classification.
    :param drug_names: list of drug names to replace
    :param repl_drug_with_spl_tkn: if True, replace drug names with a special token
    :param downsample_neutrals: if True, downsample all classes to the minimum class
    :return: ManConCorpus sentence pairs and labels for training and test sets, respectively
    """
    mancon_data = pd.read_csv(mancon_sent_pair_path, sep='\t')
    mancon_data['label'] = [2 if label == 'contradiction' else 1 if label == 'entailment' else 0 for
                            label in mancon_data.label]
    if downsample_neutrals:
        g = mancon_data.groupby('label', group_keys=False)
        mancon_data = pd.DataFrame(g.apply(lambda x: x.sample(g.size().min()))).reset_index(drop=True)

    print(f"Number of contradiction pairs: {len(mancon_data[mancon_data.label == 2])}")  # noqa: T001
    print(f"Number of entailment pairs: {len(mancon_data[mancon_data.label == 1])}")  # noqa: T001
    print(f"Number of neutral pairs: {len(mancon_data[mancon_data.label == 0])}")  # noqa: T001

    # Insert the CLS and SEP tokens
    x_train, x_test, y_train_tmp, y_test_tmp = train_test_split(
        '[CLS]' + mancon_data.text_a + '[SEP]' + mancon_data.text_b, mancon_data['label'], test_size=0.2,
        stratify=mancon_data['label'])
    if multi_class:
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
        x_test_1 = x_test + '[SEP]' + 'CON'
        x_test_2 = x_test + '[SEP]' + 'ENT'
        x_test_3 = x_test + '[SEP]' + 'NEU'
        x_test = x_test_1.append(x_test_2)
        x_test = x_test.append(x_test_3)

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

    # Replace drug name occurence with special token
    if repl_drug_with_spl_tkn:
        x_train = replace_drug_with_spl_token(x_train, drug_names)
        x_test = replace_drug_with_spl_token(x_test, drug_names)

    x_train = np.array(x_train)
    x_test = np.array(x_test)

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


def load_cord_pairs(data_path: str, active_sheet: str, drug_names: List[str] = None, multi_class: bool = True,
                    repl_drug_with_spl_tkn: bool = False):
    """
    Load CORD-19 annotated claim pairs for training.

    :param data_path: path to CORD training data
    :param active_sheet: name of active sheet with data
    :param multi_class: if True, data is prepared for multiclass classification. If False, implies auxillary input
        and data is prepared for binary classification.
    :param drug_names: list of drug names to replace
    :param repl_drug_with_spl_tkn: if True, replace drug names with a special token
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


def load_cord_pairs_v2(data_path: str, train_sheet: str, dev_sheet: str, multi_class: bool = True,
                       repl_drug_with_spl_tkn: bool = False, drug_names: List[str] = None):
    """
    Load CORD-19 annotated claim pairs for training.

    :param data_path: path to CORD training data
    :param train_sheet: name of the Excel sheet containing the train set
    :param dev_sheet: name of the Excel sheet containing the dev set
    :param multi_class: if True, data is prepared for multiclass classification. If False, implies auxillary input
        and data is prepared for binary classification.
    :param repl_drug_with_spl_tkn: if True, replace drug names with a special token
    :param drug_names: list of drug names to replace
    :return: CORD-19 sentence pairs and labels for training and test sets, respectively
    """
    cord_data_train = read_data_from_excel(data_path, train_sheet)
    cord_data_dev = read_data_from_excel(data_path, dev_sheet)

    cord_data_train['label'] = [2 if label == 'contradiction' else 1 if label == 'entailment' else 0 for
                                label in cord_data_train.annotation]
    print(f"Number of contradiction pairs: {len(cord_data_train[cord_data_train.label == 2])}")  # noqa: T001
    print(f"Number of entailment pairs: {len(cord_data_train[cord_data_train.label == 1])}")  # noqa: T001
    print(f"Number of neutral pairs: {len(cord_data_train[cord_data_train.label == 0])}")  # noqa: T001

    cord_data_dev['label'] = [2 if label == 'contradiction' else 1 if label == 'entailment' else 0 for
                              label in cord_data_dev.annotation]
    print(f"Number of contradiction pairs: {len(cord_data_dev[cord_data_dev.label == 2])}")  # noqa: T001
    print(f"Number of entailment pairs: {len(cord_data_dev[cord_data_dev.label == 1])}")  # noqa: T001
    print(f"Number of neutral pairs: {len(cord_data_dev[cord_data_dev.label == 0])}")  # noqa: T001

    # Insert the CLS and SEP tokens
    # x_train, x_test, y_train_tmp, y_test_tmp = train_test_split(
    #    '[CLS]' + cord_data.text1 + '[SEP]' + cord_data.text2, cord_data['label'], test_size=0.2,
    #    stratify=cord_data['label'])

    x_train = '[CLS]' + cord_data_train.text1 + '[SEP]' + cord_data_train.text2
    y_train_tmp = cord_data_train.label

    x_test = '[CLS]' + cord_data_dev.text1 + '[SEP]' + cord_data_dev.text2
    y_test_tmp = cord_data_dev.label

    if multi_class:
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
        x_test_1 = x_test + '[SEP]' + 'CON'
        x_test_2 = x_test + '[SEP]' + 'ENT'
        x_test_3 = x_test + '[SEP]' + 'NEU'
        x_test = x_test_1.append(x_test_2)
        x_test = x_test.append(x_test_3)

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

    # Replace drug name occurrence with special token
    if repl_drug_with_spl_tkn:
        x_train = replace_drug_with_spl_token(x_train, drug_names)
        x_test = replace_drug_with_spl_token(x_test, drug_names)

    x_train = np.array(x_train)
    x_test = np.array(x_test)

    return x_train, y_train, x_test, y_test
