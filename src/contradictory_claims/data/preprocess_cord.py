"""Function for preprocessing cord-19 dataset."""

# -*- coding: utf-8 -*-

import pandas as pd


def filter_metadata_for_covid19(metadata_path: str, virus_lex_path: str, pub_date_cutoff: str):
    """
    Filter metadata to publications containing a COVID-19 synonym in title or abstract and published after cut-off date.

    :param metadata_path: path to CORD-19 metadata.csv file
    :param virus_lex_path: path to COVID-19 lexicon
    :param pub_date_cutoff: cut-off for publication date in the format 'yyyy-mm-dd'
    :return: Metadata for filtered publications
    """
    metadata_df = pd.read_csv(metadata_path)

    # Concatenate title and abstract text into a single, lower-cased column
    metadata_df = metadata_df.fillna('')
    metadata_df.loc[:, 'title_abstract'] = metadata_df.loc[:, 'title'].str.lower() + ' '\
        + metadata_df.loc[:, 'abstract'].str.lower()
    metadata_df.loc[:, 'title_abstract'] = metadata_df.loc[:, 'title_abstract'].fillna('')

    # Load file with COVID-19 lexicon (1 per line) and generate a search pattern
    with open(virus_lex_path) as f:
        covid_19_terms = f.read().splitlines()
        covid_19_term_pattern = '|'.join([i.lower() for i in covid_19_terms])

    covid19_df = metadata_df.loc[metadata_df.title_abstract.str.contains(covid_19_term_pattern)]
    covid19_df = covid19_df.loc[metadata_df['publish_time'] > pub_date_cutoff]

    return covid19_df
