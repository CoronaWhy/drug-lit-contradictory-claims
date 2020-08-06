"""Functions for evaluating model and creating various reports for data viz/analysis."""

# -*- coding: utf-8 -*-

import datetime
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from contradictory_claims.models.train_model import regular_encode
from sklearn.metrics import accuracy_score, auc, f1_score, precision_score, recall_score, roc_curve
from sklearn.preprocessing import label_binarize
from transformers import AutoTokenizer


def read_data_from_excel(data_path: str, active_sheet: str, drop_na: bool = True):
    """
    Read data from an Excel sheet.

    :param data_path: path of data to import
    :param active_sheet: name of active sheet in Excel containing data
    :param drop_na: if True, drop rows containing NAs from resulting DataFrame
    :return: Pandas DataFrame containing data
    """
    df = pd.read_excel(data_path, sheet_name=active_sheet)
    if drop_na:
        df = df.dropna().reset_index(drop=True)

    return df


def make_predictions(df: pd.DataFrame, model, model_name: str, max_len: int = 512, method: str = "multiclass"):
    """
    Make predictions using trained model and data to predict on.

    :param df: Pandas DataFrame containing data to predict on
    :param model: end-to-end trained Transformer model
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

    # Then make predictions
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    encoded_inputs = regular_encode(inputs, tokenizer, maxlen=max_len)
    predictions = model.predict(encoded_inputs)

    if method == "multiclass":
        df['predicted_con'] = predictions[:, 0]
        df['predicted_ent'] = predictions[:, 1]
        df['predicted_neu'] = predictions[:, 2]
        # Calculate predicted class as the max predicted label
        df['predicted_class'] = df[['predicted_con', 'predicted_ent', 'predicted_neu']].idxmax(axis=1)
        df.predicted_class.replace(to_replace={'predicted_con': 'contradiction',
                                               'predicted_ent': 'entailment',
                                               'predicted_neu': 'neutral'}, inplace=True)
    elif method == "binary":
        df.predicted_con = predictions[:, 0]
    else:
        raise ValueError(f"{method} not a valid method type. Must be \"multiclass\" or \"binary\"")

    return df


def print_pair(claim1: str, claim2: str, score: float, round_num: int = 3):
    """
    Print the claims pair in a nicely formatted way.

    :param claim1: claim 1 string
    :param claim2: claim 2 string
    :param score: score associated with the pair
    :param round_num: number of places to round to
    :return: string representation of pairs nicely formatteed

    """
    out = "\nClaim 1\n"
    out += claim1 + "\n"
    out += "\nClaim 2\n"
    out += claim2 + "\n"
    out += f"\nScore: {score:.{round_num}f}\n"
    out += "------------"

    return out


def create_report(df: pd.DataFrame,
                  model_id: str,
                  out_report_file: str,
                  out_plot_dir: str,
                  method: str = "multiclass",
                  include_top_scoring: bool = True,
                  k: int = 10,
                  plot_rocs: bool = True,
                  round_num: int = 3):
    """
    Generate a report with the results of the predictions and various evaluation metrics.

    :param df: Pandas DataFrame containing data with predictions made using trained model
    :param model_id: string identifier for the model being used for predictions
    :param out_report_file: path to write report to
    :param out_plot_dir: path (directory) to write figures to
    :param method: "multiclass" or "binary"--describes setting for prediction outputs
    :param include_top_scoring: if True, include examples of top-scoring contradictory claims
    :param k: number of top-scoring contradictory claims to include in the report
    :param plot_rocs: if True, plot ROCs of performance
    :param round_num: number of decimal places to round to
    """
    with open(out_report_file, 'w') as out_file:

        # Count number of annotations of each class
        # NOTE: This expects column name to be "annotation" for true labels
        n_con = len(df.loc[df.annotation == 'contradiction', :])
        n_ent = len(df.loc[df.annotation == 'entailment', :])
        n_neu = len(df.loc[df.annotation == 'neutral', :])

        out_file.write("Annotated data distribution:\n")
        out_file.write("============================\n\n")

        out_file.write(f"Number of annotated contradictions: {n_con}\n")
        out_file.write(f"Number of annotated entailments: {n_ent}\n")
        out_file.write(f"Number of annotated neutrals: {n_neu}\n\n\n")

        # Calculate accuracy/PR metrics
        out_file.write("Accuracy/Precision/Recall/F1 Metrics:\n")
        out_file.write("=====================================\n\n")

        if method == "multiclass":
            accuracy = accuracy_score(df.annotation, df.predicted_class)
            precision = precision_score(df.annotation, df.predicted_class, average=None)
            recall = recall_score(df.annotation, df.predicted_class, average=None)
            f1 = f1_score(df.annotation, df.predicted_class, average=None)

            out_file.write(f"Accuracy: {accuracy:.{round_num}f}\n")
            out_file.write(f"Precision: {np.round(precision, round_num)}\n")
            out_file.write(f"Recall: {np.round(recall, round_num)}\n")
            out_file.write(f"F1-score: {np.round(f1, round_num)}\n\n\n")

        elif method == "binary":
            pass  # NOTE: I don't think we should be doing binary classification

        else:
            raise ValueError(f"{method} not a valid method type. Must be \"multiclass\" or \"binary\"")

        # Code for including the top predicted examples of each class in the report
        if include_top_scoring:
            out_file.write(f"The top {k} most contradictory pairs are as follows:\n")
            out_file.write("===================================================\n")
            df_top_con = df.sort_values(by='predicted_con', ascending=False).head(k)
            for i in range(k):
                pair_i = df_top_con.iloc[i]
                formatted_pair = print_pair(pair_i.text1, pair_i.text2, pair_i.predicted_con)
                out_file.write(formatted_pair)
            out_file.write("\n\n")

            out_file.write(f"The top {k} most entailing pairs are as follows:\n")
            out_file.write("=================================================\n")
            df_top_ent = df.sort_values(by='predicted_ent', ascending=False).head(k)
            for i in range(k):
                pair_i = df_top_ent.iloc[i]
                formatted_pair = print_pair(pair_i.text1, pair_i.text2, pair_i.predicted_ent)
                out_file.write(formatted_pair)
            out_file.write("\n\n")

            out_file.write(f"The top {k} most neutral pairs are as follows:\n")
            out_file.write("===============================================\n")
            df_top_neu = df.sort_values(by='predicted_neu', ascending=False).head(k)
            for i in range(k):
                pair_i = df_top_neu.iloc[i]
                formatted_pair = print_pair(pair_i.text1, pair_i.text2, pair_i.predicted_neu)
                out_file.write(formatted_pair)
            out_file.write("\n\n")

    if plot_rocs:
        n_classes = 3
        binarized_annotations = label_binarize(df.annotation, classes=["contradiction", "entailment", "neutral"])
        predicted_annotations = np.array(df[['predicted_con', 'predicted_ent', 'predicted_neu']])

        # Compute ROC curve and ROC area for each class
        fpr = {}
        tpr = {}
        roc_auc = {}
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(binarized_annotations[:, i], predicted_annotations[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(binarized_annotations.ravel(), predicted_annotations.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        # Plot ROC curve
        plt.figure()
        plt.plot(fpr["micro"], tpr["micro"],
                 label=f'micro-average ROC curve (area = {roc_auc["micro"]:0.{round_num}f})')
        for i in range(n_classes):
            plt.plot(fpr[i], tpr[i], label=f'ROC curve of class {i} (area = {roc_auc[i]:0.{round_num}f})')

        now = datetime.datetime.now()
        plot_path = os.path.join(out_plot_dir, f"{model_id}_{now.month}-{now.day}-{now.year}.png")
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Some extension of Receiver operating characteristic to multi-class')
        plt.legend(loc="lower right")
        plt.savefig(plot_path)
