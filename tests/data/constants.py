"""File paths used in testing."""

# -*- coding: utf-8 -*-

import os

ROOT_DIR = os.path.abspath(os.path.join(__file__, "../../.."))

# CORD-19 paths
metadata_path = os.path.join(ROOT_DIR, 'input/cord19/test_metadata.csv')
sample_metadata_path = os.path.join(ROOT_DIR, 'tests/data/resources/cord19/test_metadata.csv')

# CORD-19 json files zip folder path
json_text_file_dir = os.path.join(ROOT_DIR, 'input/cord19/test_json.zip')
sample_json_text_file_dir = os.path.join(ROOT_DIR, 'tests/data/resources/cord19/test_json.zip')

# Path for temporary file storage during CORD-19 processing
json_temp_path = os.path.join(ROOT_DIR, 'input/cord19/extracted/')
sample_json_temp_path = os.path.join(ROOT_DIR, 'tests/data/resources/cord19/extracted/')

# CORD-19 pdf files to extracted
pdf_filenames = ['document_parses/pdf_json/000a0fc8bbef80410199e690191dc3076a290117.json',
                 'document_parses/pdf_json/000affa746a03f1fe4e3b3ef1a62fdfa9b9ac52a.json']

# CORD-19 pmc files to extracted
pmc_filenames = ['document_parses/pmc_json/PMC1054884.xml.json',
                 'document_parses/pmc_json/PMC1065028.xml.json']

# CORD-19 publication cut off date
pub_date_cutoff = '2019-10-01'

# MultiNLI paths
multinli_train_path = os.path.join(ROOT_DIR, 'input/multinli/multinli_1.0_train.txt')
multinli_test_path = os.path.join(ROOT_DIR, 'input/multinli-dev/multinli_1.0_dev_matched.txt')
sample_multinli_train_path = os.path.join(ROOT_DIR, 'tests/data/resources/sample_multinli_train.txt')
sample_multinli_test_path = os.path.join(ROOT_DIR, 'tests/data/resources/sample_multinli_dev.txt')

# MedNLI paths
mednli_train_path = os.path.join(ROOT_DIR, 'input/mednli/mli_train_v1.jsonl')
mednli_dev_path = os.path.join(ROOT_DIR, 'input/mednli/mli_dev_v1.jsonl')
mednli_test_path = os.path.join(ROOT_DIR, 'input/mednli/mli_test_v1.jsonl')

# ManConCorpus processed path
mancon_sent_pairs = os.path.join(ROOT_DIR, 'input/manconcorpus-sent-pairs/manconcorpus_sent_pairs_200516.tsv')
sample_mancon_sent_pairs = os.path.join(ROOT_DIR, 'tests/data/resources/sample_mancon.txt')

# Other input paths
drug_lex_path = os.path.join(ROOT_DIR, 'input/drugnames/DrugNames.txt')
virus_lex_path = os.path.join(ROOT_DIR, 'input/virus-words/virus_words.txt')
sample_drug_lex_path = os.path.join(ROOT_DIR, 'tests/data/resources/DrugNames.txt')
sample_virus_lex_path = os.path.join(ROOT_DIR, 'tests/data/resources/virus_words.txt')
