"""File paths used in testing."""

# -*- coding: utf-8 -*-

import os

ROOT_DIR = os.path.abspath(os.path.join(__file__, "../../.."))

# CORD-19 paths
metadata_path = os.path.join(ROOT_DIR, 'tests/data/resources/cord19/metadata.csv')

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
