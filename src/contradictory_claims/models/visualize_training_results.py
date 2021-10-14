"""Functions for visualizing the results of the BERT training."""

# -*- coding: utf-8 -*-

import os
import csv
import re
from pathlib import Path, PurePath
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sherlock_dir = "/share/PI/rbaltman/dnsosa/projects/drug-lit-contradictory-claims/output/"
# sherlock_dir = "/Users/dnsosa/Desktop/AltmanLab/ContradictoryClaims/drug-lit-contradictory-claims/output/sample_output"


def summarize_and_generate_figure(root_dir: str = sherlock_dir,
                                  lr: float = 0.0001,
                                  bs: int = 8,
                                  eps: int = 4,
                                  rep: int = 1):
    """
    Generate figures based on the training results.

    :param root_dir: directory where all the results are coming from
    :param lr: learning rate
    :param bs: batch size
    :param eps: epochs
    :param rep: replicate (meaningless)
    """

    root_dir = Path(root_dir)

    model_short = "seq_jobs_100121"

    all_res_file = PurePath.joinpath(root_dir, f"all_results_{model_short}.csv")
    mancon_res_file = PurePath.joinpath(root_dir, f"all_mancon_results_{model_short}.csv")

    models = ["biobert", "bluebert"]

    inputs_list = ["multi",
                   "multi_med",
                   "multi_med_man",
                   "multi_med_man_roam",
                   "med",
                   "med_man",
                   "med_man_roam",
                   "man",
                   "man_roam",
                   "roam",
                   "combined"]

    for compiled_results_output_file in [all_res_file, mancon_res_file]:

        all_lines_to_write = []

        with open(compiled_results_output_file, 'w') as out_file:

            for model in models:
                for inputs in inputs_list:

                    model_dir = f"{model_short}_{inputs}_eps{eps}_bs{bs}_rep{rep}_lr{lr}_{model}"
                    res_dir = PurePath.joinpath(root_dir, model_dir)

                    if not Path.is_dir(res_dir):
                        print(f"WARNING: {res_dir} directory not found.")
                        continue

                    line_to_write = [res_dir, model, inputs]

                    hyperparameters = re.findall(r"[-+]?\d*\.\d+|\d+", ' '.join(str(res_dir).split('_')[-5::]))
                    line_to_write.extend(hyperparameters)

                    if compiled_results_output_file == all_res_file:
                        reports = list(PurePath.joinpath(res_dir, "reports/").rglob("summary_report.txt"))
                    else:  # ManCon
                        reports = list(PurePath.joinpath(res_dir, "mancon_reports/").rglob("summary_report.txt"))

                    if len(reports) == 0:
                        continue

                    with open(reports[0], 'r') as res_file:
                        all_model_results = []
                        for i, line in enumerate(res_file):

                            # Accuracy is captured on line 11
                            if i == 11:
                                model_result_str = line.split(' ')[1].split('\n')[0]
                                all_model_results.extend([float(model_result_str)])

                            # Precision, recall F1 captured on the other lines
                            elif 12 <= i <= 14:
                                model_result_strs = ' '.join(line.split('[')[1].split(']')[0].split()).split()
                                model_results = [float(model_result_str) for model_result_str in model_result_strs]
                                all_model_results.extend(model_results)
                    line_to_write.extend(all_model_results)
                    all_lines_to_write.append(line_to_write)

            write = csv.writer(out_file)
            write.writerows(all_lines_to_write)

    col_names = ["Full File Path", "Model", "Input List", "Epochs", "Batch Size", "Learning Rate",
                 "Rep",
                 "Accuracy",
                 "Prec - Con", "Prec - Ent", "Prec - Neu",
                 "Rec - Con", "Rec - Ent", "Rec - Neu",
                 "F1 - Con", "F1 - Ent", "F1 - Neu"]

    all_res_df = pd.read_csv(all_res_file, names=col_names)
    mancon_res_df = pd.read_csv(mancon_res_file, names=col_names)

    width = 0.2
    ticklabels = ['MultiNLI', 'MultiNLI - MedNLI', 'MultiNLI - MedNLI - ManCon', 'MultiNLI - MedNLI - ManCon - Roam',
                  'MedNLI', 'MedNLI - ManCon', 'MedNLI - ManCon - Roam', 'ManCon', 'ManCon - Roam', 'Roam', 'Combined']
    x = np.arange(len(ticklabels))

    plt.rc('xtick', labelsize=22)
    plt.rc('ytick', labelsize=22)
    plt.rc('axes', labelsize=30)
    plt.rc('legend', fontsize=20)

    for res_df in [all_res_df, mancon_res_df]:
        for model in ["BioBERT", "BlueBERT"]:
            print(f"Model: {model}")

            res_model_df = res_df[res_df["Model"] == model.lower()]

            fig, axs = plt.subplots(2, 2, figsize=(25, 25))
            fig.suptitle(model, fontsize=40, weight='bold')

            # plot data in grouped manner of bar type
            axs[0, 0].bar(x, res_model_df["Accuracy"], width, color='black')
            axs[0, 0].set_xticks(x)
            axs[0, 0].set_xticklabels(ticklabels, rotation=45, ha='right', fontdict=None, minor=False)
            axs[0, 0].set_ylim([0, 1])
            axs[0, 0].set_title("Accuracy", size=30)

            axs[0, 1].bar(x - 0.2, res_model_df["Prec - Con"] + .005, width, color='red')
            axs[0, 1].bar(x, res_model_df["Prec - Neu"] + .005, width, color='gray')
            axs[0, 1].bar(x + 0.2, res_model_df["Prec - Ent"] + .005, width, color='blue')
            axs[0, 1].set_xticks(x)
            axs[0, 1].set_xticklabels(ticklabels, rotation=45, ha='right', minor=False, fontdict=None)
            axs[0, 1].legend(["Round 1", "Round 2", "Round 3"])
            axs[0, 1].legend(["Contradict", "Neutral", "Entail"], loc='upper left')
            axs[0, 1].set_ylim([0, 1])
            axs[0, 1].set_title("Precision", size=30)

            axs[1, 0].bar(x - 0.2, res_model_df["Rec - Con"] + .005, width, color='red')
            axs[1, 0].bar(x, res_model_df["Rec - Neu"] + .005, width, color='gray')
            axs[1, 0].bar(x + 0.2, res_model_df["Rec - Ent"] + .005, width, color='blue')
            axs[1, 0].set_xticks(x)
            axs[1, 0].set_xticklabels(ticklabels, rotation=45, ha='right', fontdict=None, minor=False)
            axs[1, 0].legend(["Contradict", "Neutral", "Entail"], loc='upper left')
            axs[1, 0].set_ylim([0, 1])
            axs[1, 0].set_title("Recall", size=30)

            axs[1, 1].bar(x - 0.2, res_model_df["F1 - Con"] + .005, width, color='red')
            axs[1, 1].bar(x, res_model_df["F1 - Neu"] + .005, width, color='gray')
            axs[1, 1].bar(x + 0.2, res_model_df["F1 - Ent"] + .005, width, color='blue')
            axs[1, 1].set_xticks(x)
            axs[1, 1].set_xticklabels(ticklabels, rotation=45, ha='right', fontdict=None, minor=False)
            axs[1, 1].legend(["Contradict", "Neutral", "Entail"], loc='upper left')
            axs[1, 1].set_ylim([0, 1])
            axs[1, 1].set_title("F1", size=30)

            fig.text(0.5, -.1, 'Training Regime', ha='center', size=30)

            plt.subplots_adjust(left=0.1,
                                bottom=0.1,
                                right=0.9,
                                top=0.9,
                                wspace=0.3,
                                hspace=0.55)

            fig_out_file = PurePath.joinpath(root_dir, f"{model_short}_{model}_aggregate_results_fig.png")

            plt.savefig(fig_out_file,
                        pad_inches=1,
                        facecolor="w",
                        transparent=False)
