"""Command line interface for contradictory-claims."""

# -*- coding: utf-8 -*-

import os
import shutil

import click

from .data.make_dataset import \
    load_drug_virus_lexicons, load_mancon_corpus_from_sent_pairs, load_med_nli, load_multi_nli
from .data.preprocess_cord import filter_metadata_for_covid19
from .models.evaluate_model import create_report, make_predictions, read_data_from_excel
from .models.train_model import load_model, save_model, train_model


@click.command()
@click.option('--train/--no-train', 'train', default=False)
@click.option('--report/--no-report', 'report', default=False)
def main(train, report):
    """Run main function."""
    # Model parameters
    model_name = "allenai/biomed_roberta_base"
    model_id = "biomed_roberta"

    # File paths
    root_dir = os.path.abspath(os.path.join(__file__, "../../.."))
    trained_model_out_dir = 'output/transformer/biomed_roberta/24-7-2020_16-23'  # Just temporary!

    # CORD-19 metadata path
    metadata_path = os.path.join(root_dir, 'input/cord19/metadata.csv')

    # CORD-19 publication cut off date
    pub_date_cutoff = '2019-10-01'

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

    if train:
        # Load and preprocess CORD-19 data. TODO: Use this with rest of pipeline
        covid19_df = filter_metadata_for_covid19(metadata_path, virus_lex_path, pub_date_cutoff)  # noqa: F841

        # Load BERT train and test data
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
                                       model_name=model_name)

        # Save model
        out_dir = 'output/working/'
        save_model(trained_model)
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        shutil.make_archive('biobert_output', 'zip', root_dir=out_dir)  # ok currently this seems to do nothing

    else:
        transformer_dir = os.path.join(root_dir, trained_model_out_dir)
        pickle_file = os.path.join(transformer_dir, 'sigmoid.pickle')
        trained_model = load_model(pickle_file, transformer_dir)

    if report:
        eval_data_dir = os.path.join(root_dir, "input")
        eval_data_path = os.path.join(eval_data_dir, "drug_individual_claims_similarity_annotated.xlsx")
        active_sheet = "drug_individual_claims_similari"
        eval_data = read_data_from_excel(eval_data_path, active_sheet=active_sheet)

        # Make predictions using trained model
        eval_data = make_predictions(df=eval_data, model=trained_model, model_name=model_name)

        # Now create the report
        out_report_file = os.path.join(trained_model_out_dir, "results_report.txt")
        create_report(eval_data, model_id=model_id, out_report_file=out_report_file, out_plot_dir=trained_model_out_dir)


if __name__ == '__main__':
    main()
