"""Command line interface for contradictory-claims."""

# -*- coding: utf-8 -*-

import datetime
import json
import os
import shutil
from random import randrange

import click
import pandas as pd

# CHANGE ALL TO RELATIVE IMPORTS!
from .models.bluebert_evaluate_model import bluebert_make_predictions
from .data.make_dataset import \
    load_cord_pairs, load_cord_pairs_v2, load_drug_virus_lexicons, load_mancon_corpus_from_sent_pairs,\
    load_med_nli, load_multi_nli
from .data.preprocess_cord import clean_text, extract_json_to_dataframe,\
    extract_section_from_text, filter_metadata_for_covid19,\
    filter_section_with_drugs, merge_section_text
from .data.process_claims import add_cord_metadata, initialize_nlp, pair_similar_claims,\
    split_papers_on_claim_presence, tokenize_section_text
from .models.bluebert_train_model import bluebert_create_train_model,\
    bluebert_load_model, bluebert_save_model
from .models.evaluate_model import create_report, make_predictions, make_sbert_predictions, read_data_from_excel
from .models.sbert_models import build_sbert_model, load_sbert_model, save_sbert_model, train_sbert_model
from .models.train_model import load_model, save_model, train_model


@click.command()
@click.option('--train/--no-train', 'train', default=False)
@click.option('--biobert/--no-biobert', 'biobert', default=True)
@click.option('--bluebert/--no-bluebert', 'bluebert', default=False)
@click.option('--bluebert_model_path', 'bluebert_model_path', default='ttumyche/bluebert')
@click.option('--sbert/--no-sbert', 'sbert', default=False)
@click.option('--sbert-logistic-regression/--no-sbert-logistic-regression', 'logistic_model', default=True)
@click.option('--multinli/--no-multinli', 'use_multinli', default=True)
@click.option('--mednli/--no-mednli', 'use_mednli', default=False)
@click.option('--mancon/--no-mancon', 'use_mancon', default=False)
@click.option('--roamdev/--no-roamdev', 'use_roamdev', default=False)
@click.option('--extract-claims/--no-extract-claims', 'extract_claims', default=False)
@click.option('--report/--no-report', 'report', default=True)
@click.option('--multi_class/--binary_class', 'multi_class', default=True)
@click.option('--cord-version', 'cord_version', default='2020-08-10')
@click.option('--learning_rate', 'learning_rate', default=1e-6)
@click.option('--batch_size', 'batch_size', default=2)
@click.option('--epochs', 'epochs', default=3)
@click.option('--class_weights', 'class_weights', default=False)
@click.option('--aux_input', 'aux_input', default=False)
def main(train, biobert, bluebert, bluebert_model_path, sbert, logistic_model, use_multinli, use_mednli,
         use_mancon, use_roamdev, extract_claims, report, multi_class, cord_version, learning_rate,
         batch_size, epochs, class_weights, aux_input):

    """Run main function."""
    # Model parameters
    # Find path of bluebert cloned repo containing pretrained model
    # if bluebert_train:
    #     for root, dirs, _files in os.walk("."):
    #         for name in dirs:
    #             if name == 'bluebert_model_init':
    #                 bluebert_repo_path = os.path.abspath(os.path.join(root, name))
    #                 break

    config = {
        "train" : train,
        "biobert" : biobert,
        "bluebert" : bluebert,
        "bluebert_model_path" : bluebert_model_path,
        "sbert" : sbert,
        "sbert_logistic_regression" : logistic_model,
        "use_multinli" : use_multinli,
        "use_mednli" : use_mednli,
        "use_mancon" : use_mancon,
        "use_roamdev" : use_roamdev,
        "extract_claims" : extract_claims,
        "report" : report,
        "multi_class" : multi_class,
        "cord_version" : cord_version,
        "learning_rate" : learning_rate,
        "batch_size" : batch_size,
        "epochs" : epochs,
        "class_weights" : class_weights,
        "aux_input" : aux_input
    }

    # File paths
    root_dir = os.path.abspath(os.path.join(__file__, "../../.."))
    now = datetime.datetime.now()
    ri = str(randrange(1000))
    uid = f"{now}_RI{ri}"

    # CORD-19 metadata path
    metadata_path = os.path.join(root_dir, 'input', cord_version, 'metadata.csv')
    json_text_file_dir = os.path.join(root_dir, 'input', cord_version, 'document_parses.tar.gz')

    # Path for temporary file storage during CORD-19 processing
    json_temp_path = os.path.join(root_dir, 'input', cord_version, 'extracted/')

    # CORD-19 publication cut off date
    pub_date_cutoff = '2019-10-01'

    # Claims data path
    # claims_data_path = os.path.join(root_dir, 'input/cord19/claims/claims_data.csv')
    claims_data_path = os.path.join(root_dir, "input/cord19/claims/drug_individual_claims_filtered_200620.csv")

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

    # CORD-19 annotated training data path
    ## cord19_training_data_path = \
    ##    os.path.join(root_dir, 'input/cord-training/Coronawhy-Contra-Claims-Scaling-v2-annotated-2020-10-21.xlsx')
    cord19_training_data_path = \
        os.path.join(root_dir, 'input/cord-training/Roam_annotations_trainvaltest_split_V2.xlsx')

    # Other input paths
    drug_lex_path = os.path.join(root_dir, 'input/drugnames/DrugNames.txt')
    virus_lex_path = os.path.join(root_dir, 'input/virus-words/virus_words.txt')
    conc_search_terms_path = os.path.join(root_dir, 'input/conclusion-search-terms/Conclusion_Search_Terms.txt')

    if extract_claims:
        # Load and preprocess CORD-19 data
        # Extract names of files containing convid-19 synonymns in abstract/title
        # and published after a suitable cut-off date
        covid19_metadata = filter_metadata_for_covid19(metadata_path, virus_lex_path, pub_date_cutoff)
        pdf_filenames = list(covid19_metadata.pdf_json_files)
        pmc_filenames = list(covid19_metadata.pmc_json_files)

        # Extract full text for the files identified in previous step
        covid19_df = extract_json_to_dataframe(covid19_metadata, json_text_file_dir, json_temp_path,
                                               pdf_filenames, pmc_filenames)

        # Extract title\abstract\conclusion sections from publication text
        covid19_filt_section_df = extract_section_from_text(conc_search_terms_path, covid19_df)

        # Clean the text to keep only meaningful sentences
        # and merge sentences belonging to each section of each paper into contiguous text passages
        covid19_clean_df = clean_text(covid19_filt_section_df)

        # Merge all sentences belonging to each section of each paper into contiguous text passages
        covid19_merged_df = merge_section_text(covid19_clean_df)

        # Filter to sections where section text contains drug terms
        covid19_drugs_section_df = filter_section_with_drugs(covid19_merged_df,  # noqa: F841
                                                             drug_lex_path)

        # TODO: Replace with claim extraction code
        claims_df = covid19_drugs_section_df

        # Separate papers with at least 1 claim from those with no claims
        claims_data, no_claims_data = split_papers_on_claim_presence(claims_df)

        # For papers with no claims, tokenize section text to sentences
        # and append to claims
        # This is because when no claims are identified, we want to consider all sentences
        # rather than ignoring the paper altogether
        no_claims_data = tokenize_section_text(no_claims_data)
        claims_data = claims_data.append(no_claims_data).reset_index(drop=True)
    else:
        claims_data = pd.read_csv(claims_data_path)

    ## NOTE: Commenting out 12/3/20 because don't need for training
    # Initialize scispacy nlp object and add virus terms to the vocabulary
    ## nlp = initialize_nlp(virus_lex_path)

    # Pair similar claims
    ## claims_paired_df = pair_similar_claims(claims_data, nlp)

    # Add paper publish time and title info
    ## claims_paired_df = add_cord_metadata(claims_paired_df, metadata_path)

    # For evaluaitng the models
    eval_data_path = \
        os.path.join(root_dir, 'input/cord-training/Roam_annotations_trainvaltest_split_V2.xlsx')
    eval_data = read_data_from_excel(eval_data_path, active_sheet="Val")

    if train:
        # Load BERT train and test data
        multi_nli_train_x, multi_nli_train_y, multi_nli_test_x, multi_nli_test_y = \
            load_multi_nli(multinli_train_path, multinli_test_path, multi_class=multi_class)
        med_nli_train_x, med_nli_train_y, med_nli_test_x, med_nli_test_y = \
            load_med_nli(mednli_train_path, mednli_dev_path, mednli_test_path, multi_class=multi_class)
        man_con_train_x, man_con_train_y, man_con_test_x, man_con_test_y = \
            load_mancon_corpus_from_sent_pairs(mancon_sent_pairs, multi_class=multi_class)
        cord_train_x, cord_train_y, cord_test_x, cord_test_y = \
            load_cord_pairs_v2(cord19_training_data_path, 'Train', 'Val', multi_class=multi_class)
        drug_names, virus_names = load_drug_virus_lexicons(drug_lex_path, virus_lex_path)

        if biobert:
            # Train model
            trained_model, train_history = train_model(multi_nli_train_x, multi_nli_train_y,
                                                       multi_nli_test_x, multi_nli_test_y,
                                                       med_nli_train_x, med_nli_train_y,
                                                       med_nli_test_x, med_nli_test_y,
                                                       man_con_train_x, man_con_train_y,
                                                       man_con_test_x, man_con_test_y,
                                                       cord_train_x, cord_train_y,
                                                       cord_test_x, cord_test_y,
                                                       drug_names, virus_names,
                                                       model_name="allenai/biomed_roberta_base",
                                                       use_multi_nli=use_multinli,
                                                       use_med_nli=use_mednli,
                                                       use_man_con=use_mancon,
                                                       use_cord=use_roamdev,
                                                       epochs=epochs,
                                                       batch_size=batch_size,  # class_weights, aux_input,
                                                       learning_rate=learning_rate,
                                                       multi_class=multi_class)

            # Save model
            # out_dir = 'output/working/biomed_roberta_base/'
            out_dir = output_dir
            biobert_out_dir = f"output/trained_biobert/biobert_{uid}"
            if not os.path.exists(biobert_out_dir):
                os.makedirs(biobert_out_dir)
            save_model(trained_model, timed_dir_name=False, transformer_dir=biobert_out_dir)
            shutil.make_archive('biobert_output', 'zip', root_dir=out_dir)  # ok currently this seems to do nothing

            # Save model train history
            out_train_hist_dir = os.path.join(biobert_out_dir, 'train_history.txt')
            with open(out_train_hist_dir, 'w') as f:
                for item in train_history:
                    f.write(str(item.history) + "\n")

            # Save config
            myJSON = json.dumps(config)
            with open(os.path.join(biobert_out_dir, "config.json"), "w") as jsonfile:
                jsonfile.write(myJSON)
                print("Written config file for this run.")

            if report:
                # Make predictions using trained model
                eval_data = make_predictions(df=eval_data, model=trained_model,
                                             model_name="allenai/biomed_roberta_base",
                                             multi_class=multi_class)

                # Now create the report
                out_report_dir = os.path.join(biobert_out_dir, 'reports')
                if not os.path.exists(out_report_dir):
                    os.makedirs(out_report_dir)
                create_report(eval_data, model_id="biomed_roberta", out_report_dir=out_report_dir,
                              out_plot_dir=out_report_dir)

        if bluebert:
            # Train model
            bluebert_trained_model, bluebert_train_hist, \
            device = bluebert_create_train_model(multi_nli_train_x, multi_nli_train_y,
                                                 multi_nli_test_x, multi_nli_test_y,
                                                 med_nli_train_x, med_nli_train_y,
                                                 med_nli_test_x, med_nli_test_y,
                                                 man_con_train_x, man_con_train_y,
                                                 man_con_test_x, man_con_test_y,
                                                 cord_train_x, cord_train_y,
                                                 cord_test_x, cord_test_y,
                                                 bluebert_model_path,
                                                 use_multinli, use_mednli,
                                                 use_mancon, use_roamdev,
                                                 ##learning_rate, #FIGURE THIS OUT
                                                 batch_size=batch_size,
                                                 epochs=epochs,  # class_weights, aux_input,
                                                 multi_class=multi_class)
            # Save model
            bluebert_out_dir = f"output/trained_bluebert/bluebert_{uid}"
            if not os.path.exists(bluebert_out_dir):
                os.makedirs(bluebert_out_dir)
            bluebert_save_model(bluebert_trained_model, bluebert_save_path=bluebert_out_dir)

            # Save model train history
            out_train_hist_dir = os.path.join(bluebert_out_dir, 'train_history.txt')
            with open(out_train_hist_dir, 'w') as f:
                for item in bluebert_train_hist:
                    f.write(str(item) + "\n")

            # Save config
            myJSON = json.dumps(config)
            with open(os.path.join(bluebert_out_dir, "config.json"), "w") as jsonfile:
                jsonfile.write(myJSON)
                print("Written config file for this run.")

            if report:
                # Make predictions using trained model
                eval_data = bluebert_make_predictions(df=eval_data, bluebert_pretrained_path=bluebert_model_path,
                                                      model=bluebert_trained_model, device=device,
                                                      model_name='bluebert',
                                                      multi_class=multi_class)

                # Now create the report
                out_report_dir = os.path.join(bluebert_out_dir, 'reports')
                if not os.path.exists(out_report_dir):
                    os.makedirs(out_report_dir)
                create_report(eval_data, model_id="bluebert", out_report_dir=out_report_dir,
                              out_plot_dir=out_report_dir)

        if sbert:
            # note uses biobert model as base, so model_name is correct
            sbert_model, tokenizer = build_sbert_model("allenai/biomed_roberta_base", logistic_model=logistic_model)
            sbert_model = train_sbert_model(sbert_model,
                                            tokenizer=tokenizer,
                                            use_man_con=use_mancon,
                                            use_med_nli=use_mednli,
                                            use_multi_nli=use_multinli,
                                            use_cord=use_roamdev,
                                            multi_nli_train_x=multi_nli_train_x,
                                            multi_nli_train_y=multi_nli_train_y,
                                            multi_nli_test_x=multi_nli_test_x,
                                            multi_nli_test_y=multi_nli_test_y,
                                            med_nli_train_x=med_nli_train_x,
                                            med_nli_train_y=med_nli_train_y,
                                            med_nli_test_x=med_nli_test_x,
                                            med_nli_test_y=med_nli_test_y,
                                            man_con_train_y=man_con_train_y,
                                            man_con_train_x=man_con_train_x,
                                            man_con_test_x=man_con_test_x,
                                            man_con_test_y=man_con_test_y,
                                            cord_train_x=cord_train_x,
                                            cord_train_y=cord_train_y,
                                            cord_test_x=cord_test_x,
                                            cord_test_y=cord_test_y,
                                            batch_size=batch_size,
                                            num_epochs=epochs)
                                            # MISSING LEARNING RATE

            # Save model
            sbert_out_dir = f"output/trained_sbert/sbert_{uid}"
            if not os.path.exists(sbert_out_dir):
                os.makedirs(sbert_out_dir)
            save_sbert_model(model=sbert_model, transformer_dir=sbert_out_dir)

            # Save config
            myJSON = json.dumps(config)
            with open(os.path.join(sbert_out_dir, "config.json"), "w") as jsonfile:
                jsonfile.write(myJSON)
                print("Written config file for this run.")

            if report:
                eval_data = make_sbert_predictions(df=eval_data, model=sbert_model,
                                                   model_name="allenai/biomed_roberta_base")

                # Now create the report
                out_report_dir = os.path.join(sbert_out_dir, 'reports')
                if not os.path.exists(out_report_dir):
                    os.makedirs(out_report_dir)

                create_report(eval_data, model_id="biobert_sbert", out_report_dir=out_report_dir,
                              out_plot_dir=out_report_dir)
    else:  # if not train
        # NOTE: Need to revisit all of this
        if biobert:
            pass
            # transformer_dir = os.path.join(root_dir, trained_model_out_dir)
            # pickle_file = os.path.join(transformer_dir, 'sigmoid.pickle')
            # trained_model = load_model(pickle_file, transformer_dir, multi_class=multi_class)
        if bluebert:
            pass
            # bluebert_trained_model, device = bluebert_load_model(bluebert_model_path)
        if sbert:
            pass
            # sbert_dir = os.path.join(root_dir, sbert_trained_model_out_dir)
            # sbert_model = load_sbert_model(sbert_dir, 'sigmoid.pickle')


if __name__ == '__main__':
    main()
    #main(train=True,
    #     output_dir="output/trained_bluebert/bluebert_med_man_roam_rep5_epochs1_bs2_lr0.000001",
    #     roberta=False,
    #     bluebert=True,
    #     bluebert_model_path="ttumyche/bluebert",
    #     use_multinli=False,
    #     use_mednli=False,
    #     use_mancon=False,
    #     use_roamdev=True,
    #     extract_claims=False,
    #     report=False,
    #     bluebert_report=True,
    #     multi_class=True,
    #     learning_rate=0.00001,
    #     batch_size=2,
    #     epochs=1,
    #     cord_version='2020-08-10',
    #     class_weights=False,
    #     aux_input=False)

## python -m contradictory_claims --train --no-roberta --bluebert --output_folder output/trained_bluebert/bluebert_med_man_roam_rep5_epochs1_bs2_lr0.000001 --learning_rate 0.000001 --epochs 1 --batch_size 2 --no-multinli --roamdev