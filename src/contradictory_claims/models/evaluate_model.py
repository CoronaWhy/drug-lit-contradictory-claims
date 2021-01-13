"""Functions for evaluating model and creating various reports for data viz/analysis."""

# -*- coding: utf-8 -*-

import datetime
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from contradictory_claims.models.dataloader import ClassifierDataset
from contradictory_claims.models.train_model import regular_encode
from sklearn.metrics import accuracy_score, auc, confusion_matrix, f1_score, precision_score, recall_score, roc_curve
from sklearn.preprocessing import label_binarize
from transformers import AutoTokenizer


def read_data_from_excel(
        data_path: str,
        active_sheet: str,
        drop_na: bool = True):
    """
    Read data from an Excel sheet.

    :param data_path: path of data to import
    :param active_sheet: name of active sheet in Excel containing data
    :param drop_na: if True, drop rows containing NAs from resulting DataFrame
    :return: Pandas DataFrame containing data
    """
    df = pd.read_excel(data_path, sheet_name=active_sheet)
    # NOTE: first column contains junk, so dropping it. If we change data
    # schema need to change this
    df = df.drop(df.columns[0], axis=1)
    print(f"Length of DF: {len(df)}")  # noqa: T001
    if drop_na:
        df = df.dropna().reset_index(drop=True)
        print(f"Dropped NAs. Resulting length of DF: {len(df)}")  # noqa: T001

    return df


def make_predictions(df: pd.DataFrame, model, model_name: str, max_len: int = 512, multi_class: bool = True):
    """
    Make predictions using trained model and data to predict on.

    :param df: Pandas DataFrame containing data to predict on
    :param model: end-to-end trained Transformer model
    :param model_name: name of model to be loaded by Transformer to get proper tokenizer
    :param max_len: max length of string to be encoded
    :param multi_class: "multiclass" or "binary"--describes setting for prediction outputs
    :return: Pandas DataFrame augmented with predictions made using trained model
    """
    # First insert the CLS and SEP tokens
    inputs = []

    # NOTE: this expects columns named "text1" and "text2" for the two claims
    if multi_class:
        for i in range(len(df)):
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

    # Then make predictions
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    encoded_inputs = regular_encode(inputs, tokenizer, maxlen=max_len, multi_class=multi_class)
    predictions = model.predict(encoded_inputs)

    if multi_class:
        df['predicted_con'] = predictions[:, 2]
        df['predicted_ent'] = predictions[:, 1]
        df['predicted_neu'] = predictions[:, 0]

    else:
        # Note: For the binary method using auxillary input, after retrieving the prediction probability
        # for each class, we structure the prediction output dataframe in the same format
        # as the multiclass method.
        df['predicted_con'] = predictions[0:len(df)]
        df['predicted_ent'] = predictions[len(df):(2 * len(df))]
        df['predicted_neu'] = predictions[(2 * len(df)):]

    # Calculate predicted class as the max predicted label
    df['predicted_class'] = df[['predicted_con', 'predicted_ent', 'predicted_neu']].idxmax(axis=1)
    df.predicted_class.replace(to_replace={'predicted_con': 'contradiction',
                                           'predicted_ent': 'entailment',
                                           'predicted_neu': 'neutral'}, inplace=True)

    return df


def make_sbert_predictions(
        df: pd.DataFrame,
        model,
        model_name: str,
        max_len: int = 512):
    """Make predictions using SBERt trained model.

    :param df: Pandas DataFrame containing data to predict on
    :param model: end-to-end trained Transformer model
    :param model_name: name of model to be loaded by Transformer to get proper tokenizer
    :param max_len: max length of string to be encoded
    :param method: "multiclass" or "binary"--describes setting for prediction outputs
    :return: Pandas DataFrame augmented with predictions made using trained model
    """
    # if model_name == "covidbert":
    #     model_name = "deepset/covid_bert_base"
    # else:
    #     model_name = "allenai/biomed_roberta_base"
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    # with torch.no_grad():
    #     predictions = model(tokenizer.batch_encode_plus(df['text1'].values.tolist(),
    #                                                     max_length=max_len,
    #                                                     pad_to_max_length=True,
    #                                                     truncation=True)["input_ids"],
    #                         tokenizer.batch_encode_plus(df['text2'].values.tolist(),
    #                                                     max_length=max_len,
    #                                                     pad_to_max_length=True,
    #                                                     truncation=True)["input_ids"])
    #     predictions = torch.log_softmax(predictions, dim=1)
    labels = ClassifierDataset.get_labels()
    df_temp = df.rename(
        columns={
            "text1": "sentence1",
            "text2": "sentence2",
            "annotation": "label"})
    df_temp.label = df_temp.label.map(labels)
    # eval_vector, eval_label = format_create(df=df_temp, model=model)
    # dictionary_mapping = ClassifierDataset.get_mappings()
    # predictions = predictions.cpu().numpy()
    # predictions = model.logisticregression.predict(eval_vector)
    predictions = model.predict(df.text1.values, df.text2.values)
    df['predicted_con'] = np.where(
        predictions == labels['contradiction'], 1, 0)
    # df['predicted_con'] = predictions[:, labels['contradiction']]
    df['predicted_ent'] = np.where(predictions == labels['entailment'], 1, 0)
    # df['predicted_ent'] = predictions[:, labels['entailment']]
    df['predicted_neu'] = np.where(predictions == labels['neutral'], 1, 0)
    # df['predicted_neu'] = predictions[:, labels['neutral']]
    # Calculate predicted class as the max predicted label
    df['predicted_class'] = df[['predicted_con',
                                'predicted_ent', 'predicted_neu']].idxmax(axis=1)
    df.predicted_class.replace(
        to_replace={
            'predicted_con': 'contradiction',
            'predicted_ent': 'entailment',
            'predicted_neu': 'neutral'},
        inplace=True)
    # df.loc[:, 'prediction'] = predictions.argmax(axis=1).to("cpu").numpy()

    # df.loc[:, 'predicted_class'] = df['prediction'].apply(lambda x: dictionary_mapping[x])
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


def print_pair_2(
        claim1: str,
        claim2: str,
        true_label: str,
        predicted_label: str,
        score: float,
        round_num: int = 3):
    """
    Print the claims pair in a nicely formatted way when the model disagrees with annotation.

    :param claim1: claim 1 string
    :param claim2: claim 2 string
    :param true_label: annotation for claim pair
    :param predicted_label: prediction about claim pair
    :param score: score associated with the pair
    :param round_num: number of places to round to
    :return: string representation of pairs nicely formatteed
    """
    out = "\nClaim 1\n"
    out += claim1 + "\n"
    out += "\nClaim 2\n"
    out += claim2 + "\n"
    out += f"\nAnnotated label: {true_label}\t Predicted label: {predicted_label} (Prob = {score:.{round_num}f})\n"
    out += "----------------------------------------------------------------------------\n"

    return out


def custom_plot_confusion_matrix(cm,
                                 target_names,
                                 out_plot_dir: str,
                                 model_id: str,
                                 time: datetime.datetime,
                                 title='Confusion matrix',
                                 cmap=None,
                                 normalize=False):
    """
    Make a nice plot from a confusion matrix.

    :param cm: confusion matrix from sklearn.metrics.confusion_matrix
    :param target_names: class names, for example: ['high', 'medium', 'low']
    :param out_plot_dir: directory to save plot
    :param model_id: string identifier for the model used, for file naming purposes
    :param time: time function was called, for file naming purposes
    :param title: the text to display at the top of the matrix
    :param cmap: the gradient of the values displayed from matplotlib.pyplot.cm
    :param normalize: if False, plot the raw numbers, else plot the proportions

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, f"{cm[i,j]:0.4f}",
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, f"{cm[i, j]:,}",
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel(
        f"Predicted label\naccuracy={accuracy:0.4f}; misclass={misclass:0.4f}")
    # plt.show()
    plot_path = os.path.join(
        out_plot_dir,
        f"ConfMat_{model_id}_{time.month}-{time.day}-{time.year}.png")
    plt.savefig(plot_path)


def create_report(df: pd.DataFrame,
                  model_id: str,
                  out_report_dir: str,
                  out_plot_dir: str,
                  method: str = "multiclass",
                  include_top_scoring: bool = True,
                  k: int = 10,
                  plot_rocs: bool = True,
                  plot_confusion_matrix: bool = True,
                  write_disagreements: bool = True,
                  round_num: int = 3):
    """
    Generate a report with the results of the predictions and various evaluation metrics.

    :param df: Pandas DataFrame containing data with predictions made using trained model
    :param model_id: string identifier for the model being used for predictions
    :param out_report_dir: path to write report to
    :param out_plot_dir: path (directory) to write figures to
    :param method: "multiclass" or "binary"--describes setting for prediction outputs
    :param include_top_scoring: if True, include examples of top-scoring contradictory claims
    :param k: number of top-scoring contradictory claims to include in the report
    :param plot_rocs: if True, plot ROCs of performance
    :param plot_confusion_matrix: if True, plot confusion matrix of performance
    :param write_disagreements: if True, write out the cases where the model disagrees with annotations
    :param round_num: number of decimal places to round to
    """
    with open(os.path.join(out_report_dir, "summary_report.txt"), 'w') as out_file:

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

        now = datetime.datetime.now()

        if method == "multiclass":
            accuracy = accuracy_score(df.annotation, df.predicted_class)
            precision = precision_score(
                df.annotation, df.predicted_class, average=None)
            recall = recall_score(
                df.annotation,
                df.predicted_class,
                average=None)
            f1 = f1_score(df.annotation, df.predicted_class, average=None)

            out_file.write(f"Accuracy: {accuracy:.{round_num}f}\n")
            out_file.write(f"Precision: {np.round(precision, round_num)}\n")
            out_file.write(f"Recall: {np.round(recall, round_num)}\n")
            out_file.write(f"F1-score: {np.round(f1, round_num)}\n\n\n")

        elif method == "binary":
            # Note: For the binary method using auxillary input, after retrieving the prediction probability
            # for each class, we structure the prediction output dataframe in the same format
            # as the multiclass method.
            pass

        else:
            raise ValueError(
                f"{method} not a valid method type. Must be \"multiclass\" or \"binary\"")

        # Code for including the top predicted examples of each class in the
        # report
        if include_top_scoring:
            out_file.write(
                f"The top {k} most contradictory pairs are as follows:\n")
            out_file.write(
                "===================================================\n")
            df_top_con = df.sort_values(
                by='predicted_con', ascending=False).head(k)
            for i in range(k):
                pair_i = df_top_con.iloc[i]
                formatted_pair = print_pair(
                    pair_i.text1, pair_i.text2, pair_i.predicted_con)
                out_file.write(formatted_pair)
            out_file.write("\n\n")

            out_file.write(
                f"The top {k} most entailing pairs are as follows:\n")
            out_file.write(
                "=================================================\n")
            df_top_ent = df.sort_values(
                by='predicted_ent', ascending=False).head(k)
            for i in range(k):
                pair_i = df_top_ent.iloc[i]
                formatted_pair = print_pair(
                    pair_i.text1, pair_i.text2, pair_i.predicted_ent)
                out_file.write(formatted_pair)
            out_file.write("\n\n")

            out_file.write(f"The top {k} most neutral pairs are as follows:\n")
            out_file.write("===============================================\n")
            df_top_neu = df.sort_values(
                by='predicted_neu', ascending=False).head(k)
            for i in range(k):
                pair_i = df_top_neu.iloc[i]
                formatted_pair = print_pair(
                    pair_i.text1, pair_i.text2, pair_i.predicted_neu)
                out_file.write(formatted_pair)
            out_file.write("\n\n")

    if plot_rocs:
        n_classes = 3
        # Need to double check if this is the right mapping of 0 1 2 to
        # con/ent/neu
        binarized_annotations = label_binarize(
            df.annotation, classes=[
                "contradiction", "entailment", "neutral"])
        predicted_annotations = np.array(
            df[['predicted_con', 'predicted_ent', 'predicted_neu']])

        # Compute ROC curve and ROC area for each class
        fpr = {}
        tpr = {}
        roc_auc = {}
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(
                binarized_annotations[:, i], predicted_annotations[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(
            binarized_annotations.ravel(), predicted_annotations.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        # Plot ROC curve
        plt.figure()
        plt.plot(
            fpr["micro"],
            tpr["micro"],
            label=f'micro-average ROC curve (area = {roc_auc["micro"]:0.{round_num}f})')
        for i in range(n_classes):
            plt.plot(
                fpr[i],
                tpr[i],
                label=f'ROC curve of class {i} (area = {roc_auc[i]:0.{round_num}f})')

        plot_path = os.path.join(
            out_plot_dir,
            f"ROC_{model_id}_{now.month}-{now.day}-{now.year}.png")
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(
            'Some extension of Receiver operating characteristic to multi-class')
        plt.legend(loc="lower right")
        plt.savefig(plot_path)

    if plot_confusion_matrix:
        conf_mat = confusion_matrix(df.annotation, df.predicted_class)
        # Need to check the order of the labels is right
        custom_plot_confusion_matrix(
            conf_mat, [
                "contradiction", "entailment", "neutral"], out_plot_dir, model_id, now)

    if write_disagreements:
        disagreements = df[df.annotation != df.predicted_class]
        disagreements["predicted_class_prob"] = disagreements.loc[:, ["predicted_con", "predicted_ent", "predicted_neu"]].max(axis=1)  # noqa: E501

        with open(os.path.join(out_report_dir, "disagreements.txt"), 'w') as dis_file:

            # disagreements[['text1', 'text2', 'annotation', 'predicted_class', 'predicted_class_prob']]
            dis_file.write(
                "PAIRS OF CLAIMS WHOSE PREDICTIONS DISAGREE WITH OUR ANNOTATIONS\n")
            dis_file.write(
                "===============================================================\n")
            dis_file.write(f"\n{len(disagreements)} disagreements total\n")

            for i in range(len(disagreements)):
                pair_i = disagreements.iloc[i]
                out = print_pair_2(
                    pair_i.text1,
                    pair_i.text2,
                    pair_i.annotation,
                    pair_i.predicted_class,
                    pair_i.predicted_class_prob)
                dis_file.write(out)
            dis_file.write("\n")
