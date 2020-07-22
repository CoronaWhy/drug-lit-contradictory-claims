"""Function for preprocessing cord-19 dataset."""

# -*- coding: utf-8 -*-

import json
import re
from datetime import datetime
from typing import List
from zipfile import ZipFile

import pandas as pd
from pandas.io.json import json_normalize


def filter_metadata_for_covid19(metadata_path: str, virus_lex_path: str, pub_date_cutoff: str):
    """
    Filter metadata to publications containing a COVID-19 synonym in title or abstract and published after cut-off date.

    :param metadata_path: path to CORD-19 metadata.csv file
    :param virus_lex_path: path to COVID-19 lexicon
    :param pub_date_cutoff: cut-off for publication date in the format 'yyyy-mm-dd'
    :return: Dataframe of metadata for filtered publications
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

    covid19_df = metadata_df.loc[metadata_df.title_abstract.str.contains(covid_19_term_pattern)]\
                            .copy().reset_index(drop=True)

    covid19_df['publish_time'] = pd.to_datetime(covid19_df['publish_time'], format='%d-%m-%Y')
    covid19_df = covid19_df.loc[covid19_df['publish_time'] > datetime.strptime(pub_date_cutoff, "%Y-%m-%d")]\
                           .copy().reset_index(drop=True)

    return covid19_df


def extract_json_to_dataframe(covid19_metadata: pd.DataFrame,
                              json_text_file_dir: str,
                              json_temp_path: str,
                              pdf_filenames: List[str],
                              pmc_filenames: List[str]):
    """
    Extract publications text from json files for a specified set of filenames and store in a dataframe.

    :param covid19_metadata: pandas dataframe, output of filter_metadata_for_covid19()
    :param json_text_file_dir: path to zip directory containing json files
    :param json_temp_path: path for temporary file storage
    :param pdf_filenames: list of pdf file names to extract
    :param pmc_filenames: list of pmc file names to extract
    :return: Dataframe of publication texts for the specified filenames
    """
    covid19_dict = {}

    # Replace characters with their readable format
    replace_dict = {'â€œ': '“',
                    'â€': '”',
                    'â€™': '’',
                    'â€˜': '‘',
                    'â€”': '–',
                    'â€“': '—',
                    'â€¢': '-',
                    'â€¦': '…'}

    # TODO: Parallelize the code below
    with ZipFile(json_text_file_dir, 'r') as zipobj:
        list_of_filenames = zipobj.namelist()
        # print('Number of files to iterate over:',len(list_of_filenames))
        k = 0
        # iter_num = 0
        for filename in list_of_filenames:
            # iter_num = iter_num + 1
            # Check filename ends with json and file exists in filtered list of cord papers
            if (filename in pdf_filenames) or (filename in pmc_filenames):
                zipobj.extract(filename, json_temp_path)
                json_op = []
                with open(json_temp_path + filename, 'r', encoding='utf8') as f:
                    temp = []
                    temp.append("".join([line.replace('\n', '').replace('\r', '').replace('\t', '') for line in f]))
                    for jsonobj in temp:
                        json_dict = json.loads(jsonobj, encoding='utf8')
                        json_op.append(json_dict)
                        for i in range(len(json_op)):
                            temp_json = json_normalize(json_op[i])
                            check_file_name = ((filename == covid19_metadata.pdf_json_files)
                                               | (filename == covid19_metadata.pmc_json_files))  # noqa: W503
                            cord_uid = list(covid19_metadata.loc[check_file_name, 'cord_uid'])[0]
                            try:
                                text = temp_json['abstract'][0][0]['text']
                                section = temp_json['abstract'][0][0]['section']
                                for key, v in replace_dict.items():
                                    text = text.replace(key, v)
                                    section = section.replace(key, v)
                                covid19_dict[k] = {'cord_uid': cord_uid,
                                                   'sentence': text,
                                                   'section': section}
                                k = k + 1
                            except KeyError:
                                pass

                            for temp_dict in temp_json['body_text'][0]:
                                text = temp_dict['text']
                                section = temp_dict['section']
                                for key, v in replace_dict.items():
                                    text = text.replace(key, v)
                                    section = section.replace(key, v)
                                covid19_dict[k] = {'cord_uid': cord_uid,
                                                   'sentence': text,
                                                   'section': section}
                                k = k + 1

#        if iter_num%100==0:
#            print('Number of files read:', iter_num)

    return pd.DataFrame.from_dict(covid19_dict, orient='index')


def construct_regex_match_pattern(search_terms: List[str], search_type: bool = 0):
    """
    Construct regex search pattern for the specified terms.

    :param terms_dict: list of search terms
    :param search_type: 1 = exact pattern, 0 = fuzzy pattern
    :return: Regex search pattern
    """
    if search_type == 1:
        exact_pattern = '|'.join(search_terms)

        return exact_pattern

    else:
        # TODO: fix flake8 error code FS001
        fuzzy_terms = ['.*%s.*' % i for i in search_terms]  # noqa: FS001
        fuzzy_pattern = '|'.join(fuzzy_terms)

        return fuzzy_pattern


def extract_regex_pattern(section_list: List[str], pattern: str):
    """
    Extract list of section names that match the specified regex pattern.

    :param section_list: list of section names to search in
    :param pattern: regex pattern to search for
    :return: List of extracted section names
    """
    r = re.compile(pattern, re.IGNORECASE)
    extracted_list = list(filter(r.match, section_list))
    # remaining_list = list(set(section_list) - set(extracted_list))

    return extracted_list


def clean_text(input_data):
    """
    Filter text to keep only sentences containing at least 3 meaningful words.

    :param input_data: pandas dataframe with publication text
    :return: clean dataframe
    """
    # List of words to ignore
    rep = {"text": "", "cite_spans": "", "ref_spans": "", "section": "", "abstract": "",
           "biorxiv preprint": "", "medrxiv preprint": "", "doi:": ""}
    rep = {re.escape(k): v for k, v in rep.items()}
    pattern = re.compile("|".join(rep.keys()))
    sentences_temp = [pattern.sub(lambda m: rep[re.escape(m.group(0))], s) for s in input_data.sentence.str.lower()]
    pattern = re.compile(".*[A-Za-z].*")
    sentences_to_keep = [(bool(re.search(pattern, s))) & (len(s.split(' ')) > 2) for s in sentences_temp]
    input_processed = input_data.loc[sentences_to_keep, :]

    return input_processed
