"""Command line interface for contradictory-claims."""

# -*- coding: utf-8 -*-

import os
import shutil

import click

from .data.make_dataset import \
    load_drug_virus_lexicons, load_mancon_corpus_from_sent_pairs, load_med_nli, load_multi_nli
# from .data.preprocess_cord import clean_text, construct_regex_match_pattern, extract_json_to_dataframe,\
#     extract_regex_pattern, filter_metadata_for_covid19
from .models.evaluate_model import create_report, make_predictions, read_data_from_excel
from .models.train_model import load_model, save_model, train_model


@click.command()
@click.option('--train/--no-train', 'train', default=False)
@click.option('--report/--no-report', 'report', default=False)
@click.option('--cord-version', 'cord_version', default='2020-08-10')
def main(train, report, cord_version):
    """Run main function."""
    # Model parameters
    model_name = "allenai/biomed_roberta_base"
    model_id = "biomed_roberta"

    # File paths
    root_dir = os.path.abspath(os.path.join(__file__, "../../.."))
    trained_model_out_dir = 'output/transformer/biomed_roberta/24-7-2020_16-23'  # Just temporary!

    # CORD-19 metadata path
    # NOTE: I'd like to discuss how we want to establish naming conventions around CORD-19 input directory
    # metadata_path = os.path.join(root_dir, 'input/cord19/metadata.csv')
    # metadata_path = os.path.join(root_dir, 'input/2020-08-10/metadata.csv')
    # metadata_path = os.path.join(root_dir, 'input', cord_version, 'metadata.csv')

    # CORD-19 json files zip folder path
    # json_text_file_dir = os.path.join(root_dir, 'input/cord19/json.zip')
    # json_text_file_dir = os.path.join(root_dir, 'input/2020-08-10/document_parses.tar.gz')
    # json_text_file_dir = os.path.join(root_dir, 'input', cord_version, 'document_parses.tar.gz')

    # Path for temporary file storage during CORD-19 processing
    # json_temp_path = os.path.join(root_dir, 'input', cord_version, 'extracted/')

    # CORD-19 publication cut off date
    # pub_date_cutoff = '2019-10-01'

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

    # Load and preprocess CORD-19 data
    # Extract names of files containing convid-19 synonymns in abstract/title
    # and published after a suitable cut-off date
    # covid19_metadata = filter_metadata_for_covid19(metadata_path, virus_lex_path, pub_date_cutoff)
    # pdf_filenames = list(covid19_metadata.pdf_json_files)
    # pmc_filenames = list(covid19_metadata.pmc_json_files)

    # Extract full text for the files identified in previous step
    # NOTE: This seems to take a really long time, so I'm ommittng for now
    # covid19_df = extract_json_to_dataframe(covid19_metadata, json_text_file_dir, json_temp_path,
    #                                        pdf_filenames, pmc_filenames)

    # Construct regex match pattern for putative conclusion section headers
    # search_terms = ['conclusion',
    #                 'discussion',
    #                 'interpretation',
    #                 'added value of this study',
    #                 'research in context',
    #                 'concluding',
    #                 'closing remarks',
    #                 'summary of findings',
    #                 'outcome']
    # search_pattern = construct_regex_match_pattern(search_terms)

    # Extract section headers for title\abstract\conclusion sections
    # unique_sections = set(covid19_df.section.tolist())
    # section_list = extract_regex_pattern(unique_sections, search_pattern)
    # section_list = [i.lower() for i in section_list]
    # section_list.append('abstract')
    # section_list.append('title')

    # Extract title\abstract\conclusion sections from publication text
    # covid19_filt_section_df = covid19_df.loc[covid19_df.section.str.lower().isin(section_list)]

    # Clean the text to keep only meaningful sentences
    # covid19_clean_df = clean_text(covid19_filt_section_df)  # noqa: F841

    if train:
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
        # eval_data_path = os.path.join(eval_data_dir, "drug_individual_claims_similarity_annotated.xlsx")
        # active_sheet = "drug_individual_claims_similari"
        eval_data_path = os.path.join(eval_data_dir, "Pilot_Contra_Claims_Annotations_06.30.xlsx")
        active_sheet = "All_phase2"
        eval_data = read_data_from_excel(eval_data_path, active_sheet=active_sheet)

        # Make predictions using trained model
        eval_data = make_predictions(df=eval_data, model=trained_model, model_name=model_name)

        # Now create the report
        out_report_dir = os.path.join(trained_model_out_dir)
        create_report(eval_data, model_id=model_id, out_report_dir=out_report_dir, out_plot_dir=trained_model_out_dir)


if __name__ == '__main__':
    main()
