"""Command line interface for contradictory-claims."""

# -*- coding: utf-8 -*-

import os
import shutil

import click

from .data.make_dataset import \
    load_drug_virus_lexicons, load_mancon_corpus_from_sent_pairs, load_med_nli, load_multi_nli
from .models.train_model import save_model, train_model


@click.command()
def main():
    """Run main function."""
    # File paths
    root_dir = os.path.abspath(os.path.join(__file__, "../../.."))

    # Data loads. NOTE: currently, it is expected that all data is found in an input/ directory with the proper
    # directory structure and file names as follows.
    # MultiNLI paths
    multinli_train_path = os.path.join(root_dir, 'input/multinli/multinli_1.0_train.txt')
    multinli_test_path = os.path.join(root_dir, 'input/multinli-dev/multinli_1.0_dev_matched.txt')

    # MedNLI paths
    mednli_train_path = os.path.join(root_dir, 'input/mednli/mli_train_v1.jsonl')
    mednli_dev_path = os.path.join(root_dir, 'input/mednli/mli_dev_v1.jsonl')
    mednli_test_path = os.path.join(root_dir, 'input/mednli/mli_test_v1.jsonl')

    # ManConCorpus processed path
    mancon_sent_pairs = os.path.join(root_dir, 'input/manconcorpus-sent-pairs/manconcorpus_sent_pairs_200516.tsv')

    # Other input paths
    drug_lex_path = os.path.join(root_dir, 'input/drugnames/DrugNames.txt')
    virus_lex_path = os.path.join(root_dir, 'input/virus-words/virus_words.txt')

    # Load data
    multi_nli_train_x, multi_nli_train_y, multi_nli_test_x, multi_nli_test_y = \
        load_multi_nli(multinli_train_path, multinli_test_path)
    med_nli_train_x, med_nli_train_y, med_nli_test_x, med_nli_test_y = \
        load_med_nli(mednli_train_path, mednli_dev_path, mednli_test_path)
    man_con_train_x, man_con_train_y, man_con_test_x, man_con_test_y = \
        load_mancon_corpus_from_sent_pairs(mancon_sent_pairs)
    drug_names, virus_names = load_drug_virus_lexicons(drug_lex_path, virus_lex_path)

    # Train model
    trained_model, _ = train_model(multi_nli_train_x, multi_nli_train_y, multi_nli_test_x, multi_nli_test_y,
                                   med_nli_train_x, med_nli_train_y, med_nli_test_x, med_nli_test_y,
                                   man_con_train_x, man_con_train_y, man_con_test_x, man_con_test_y,
                                   drug_names, virus_names,
                                   model_name="allenai/biomed_roberta_base")

    # Save model
    out_dir = 'output/working/'
    save_model(trained_model)
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    shutil.make_archive('biobert_output', 'zip', root_dir=out_dir)  # ok currently this seems to do nothing


if __name__ == '__main__':
    main()
