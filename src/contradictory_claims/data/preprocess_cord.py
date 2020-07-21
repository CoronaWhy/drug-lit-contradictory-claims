"""Function for preprocessing cord-19 dataset."""

# -*- coding: utf-8 -*-

import json
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

    replace_dict = {'â€œ': '“',
                    'â€': '”',
                    'â€™': '’',
                    'â€˜': '‘',
                    'â€”': '–',
                    'â€“': '—',
                    'â€¢': '-',
                    'â€¦': '…'}

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

                            for j, temp_dict in enumerate(temp_json['body_text'][0]):
                                text = temp_dict[j]['text']
                                section = temp_dict[j]['section']
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
