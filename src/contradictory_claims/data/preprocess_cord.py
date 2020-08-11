"""Functions for preprocessing cord-19 dataset."""

# -*- coding: utf-8 -*-

import json
import re
from datetime import datetime
from typing import List
from zipfile import ZipFile

import pandas as pd
from pandas.io.json import json_normalize


def construct_regex_match_pattern(search_terms_file_path: str, search_type: str = 'fuzzy'):
    """
    Construct regex search pattern for the specified terms.

    :param terms_dict: file path for list of search terms
    :param search_type: "exact" vs "flank_white_space" vs "fuzzy" pattern
    :return: Regex search pattern
    """
    with open(search_terms_file_path) as f:
        search_terms = f.read().splitlines()

    if search_type == 'exact':
        exact_pattern = '|'.join([i.lower() for i in search_terms])

        return exact_pattern

    elif search_type == 'flank_white_space':
        exact_pattern = '\W' + '\W|\W'.join([i.lower() for i in search_terms]) + '\W'  # noqa: W605

        return exact_pattern

    else:
        # TODO: fix flake8 error code FS001
        fuzzy_terms = ['.*%s.*' % i.lower() for i in search_terms]  # noqa: FS001
        fuzzy_pattern = '|'.join(fuzzy_terms)

        return fuzzy_pattern


def filter_metadata_for_covid19(metadata_path: str, virus_lex_path: str, pub_date_cutoff: str = None):
    """
    Filter metadata to publications containing a COVID-19 synonym in title or abstract and published after cut-off date.

    :param metadata_path: path to CORD-19 metadata.csv file
    :param virus_lex_path: path to COVID-19 lexicon
    :param pub_date_cutoff: cut-off for publication date in the format 'yyyy-mm-dd'
    :return: Dataframe of metadata for filtered publications
    """
    if pub_date_cutoff is not None:
        pub_date_cutoff = datetime.strptime(pub_date_cutoff, "%Y-%m-%d")

    metadata_df = pd.read_csv(metadata_path)

    # Concatenate title and abstract text into a single, lower-cased column
    metadata_df = metadata_df.fillna('')
    metadata_df.loc[:, 'title_abstract'] = metadata_df.loc[:, 'title'].str.lower() + ' '\
        + metadata_df.loc[:, 'abstract'].str.lower()
    metadata_df.loc[:, 'title_abstract'] = metadata_df.loc[:, 'title_abstract'].fillna('')

    # Load file with COVID-19 lexicon (1 per line) and generate a search pattern
    with open(virus_lex_path) as f:
        covid_19_terms = f.read().splitlines()
        covid_19_term_pattern = construct_regex_match_pattern(covid_19_terms, 'exact')

    covid19_df = metadata_df.loc[metadata_df.title_abstract.str.contains(covid_19_term_pattern)]\
                            .copy().reset_index(drop=True)

    if pub_date_cutoff is not None:
        # Format & convert publish_time column to datetime type of uniform format
        covid19_df['publish_time'] = pd.to_datetime(covid19_df['publish_time'])
        covid19_df['publish_time'] = covid19_df['publish_time'].dt.strftime('%Y-%m-%d')
        covid19_df['publish_time'] = pd.to_datetime(covid19_df['publish_time'])
        covid19_df = covid19_df.loc[covid19_df['publish_time'] > pub_date_cutoff]\
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
    # Empty dictonary to store the extracted section text
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
                with open(json_temp_path + filename, 'r', encoding='utf8') as f:
                    # Read each line in the file separately, remove tabs, spaces and newlines
                    # and concatenate all lines together for further parsing
                    json_str = "".join([" ".join(line.split()) for line in f])
                    # Parse the json string into the json dictionary format
                    json_dict = json.loads(json_str, encoding='utf8')
                    # Convert the json dictionary object to a pandas dataframe
                    paper_df = json_normalize(json_dict)
                    # In the covid19 metadata dataframe,
                    # filter to the row representing the current json file being processed
                    # and extract the cord_uid
                    check_file_name = ((filename == covid19_metadata.pdf_json_files)
                                       | (filename == covid19_metadata.pmc_json_files))  # noqa: W503
                    cord_uid = list(covid19_metadata.loc[check_file_name, 'cord_uid'])[0]
                    # If an abstract section exists, extract the text
                    try:
                        text = paper_df['abstract'][0][0]['text']
                        section = paper_df['abstract'][0][0]['section']
                        # Replace characters with their readable format
                        for key, v in replace_dict.items():
                            text = text.replace(key, v)
                            section = section.replace(key, v)
                        covid19_dict[k] = {'cord_uid': cord_uid,
                                           'sentence': text,
                                           'section': section}
                        k = k + 1
                    # If an abstract section does not exist, skip
                    except KeyError:
                        pass

                    for temp_dict in paper_df['body_text'][0]:
                        text = temp_dict['text']
                        section = temp_dict['section']
                        # Replace characters with their readable format
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


def extract_section_from_text(conc_search_terms_path: str, covid19_df: pd.Dataframe):
    """
    Extract title, abstract, and conclusion sections from publication text.

    :param conc_search_terms_path: file path for search terms for putative conclusion section headers
    :param covid19_df: pandas dataframe of publication text
    :return: dataframe of title, abstract, and conclusion section text
    """
    # Construct regex match pattern for putative conclusion section headers
    search_pattern = construct_regex_match_pattern(conc_search_terms_path)

    # Extract section headers for title\abstract\conclusion sections
    unique_sections = set(covid19_df.section.tolist())
    section_list = extract_regex_pattern(unique_sections, search_pattern)
    section_list = [i.lower() for i in section_list]
    section_list.append('abstract')
    section_list.append('title')

    # Extract title\abstract\conclusion sections from publication text
    covid19_filt_section_df = covid19_df.loc[covid19_df.section.str.lower().isin(section_list)]

    return covid19_filt_section_df


def clean_text(input_data: pd.DataFrame):
    """
    Filter text to keep only sentences containing at least 3 meaningful words.

    :param input_data: pandas dataframe with publication text
    :return: Clean dataframe
    """
    # List of words-to-ignore
    rep = {"text": "", "cite_spans": "", "ref_spans": "", "section": "", "abstract": "",
           "biorxiv preprint": "", "medrxiv preprint": "", "doi:": ""}
    # Escape non alphanumeric characters and construct regex pattern
    rep = {re.escape(k): v for k, v in rep.items()}
    pattern = re.compile("|".join(rep.keys()))
    # Lower case the input text
    sentences_temp = input_data.sentence.str.lower()
    # Find and replace all occurences of words-to-ignore with empty string
    sentences_temp = [pattern.sub(lambda m: rep[re.escape(m.group(0))], s) for s in sentences_temp]

    # Construct regex pattern for alphabets
    pattern = re.compile(".*[A-Za-z].*")
    # Check if text contains alphabets and at least 3 words
    sentences_to_keep = [(bool(re.search(pattern, s))) & (len(s.split(' ')) > 2) for s in sentences_temp]
    input_processed = input_data.loc[sentences_to_keep, :]

    return input_processed


def merge_section_text(input_data: pd.DataFrame):
    """
    Merge all sentences belonging to each section of each paper into contiguous text passages.

    :param input_data: pandas dataframe with publication text
    :return: Dataframe with merged section text
    """
    # Merge all sentences belonging to each section of each paper into contiguous text passages
    merged_df = input_data.groupby(['cord_uid', 'section'], as_index=False).agg({'sentence': ' '.join})
    # Convert text to lower case
    merged_df.loc[:, 'sentence'] = merged_df.loc[:, 'sentence'].str.lower()
    # Rename column
    merged_df = merged_df.rename(columns={'sentence': 'text'})

    return merged_df


def filter_section_with_drugs(input_data: pd.DataFrame, drug_lex_path: str):
    """
    Filter to sections where section text contains drug terms.

    :param input_data: pandas dataframe with publication text
    :param drug_lex_path: file path for list of drug terms to search for
    :return: Dataframe with sections containing drug terms
    """
    # Construct regex match pattern for drug terms
    # Flank drug terms with white space for accurate match
    drug_terms_pattern = construct_regex_match_pattern(drug_lex_path, 'flank_white_space')

    # Replace drug name short forms with full forms
    drug_replace_dict = {'hcq': 'hydroxychloroquine', ' cq ': 'chloroquine', ' azt ': 'azithromycin',
                         ' azi ': 'azithromycin', ' az ': 'azithromycin'}
    for key, value in drug_replace_dict.items():
        input_data['text'] = [t.lower().replace(key, value) for t in input_data.text]

    # Filter to sections where section text contains drug terms
    contain_drug_mask = input_data['text'].str.contains(drug_terms_pattern, case=False)
    drugs_section_df = input_data[contain_drug_mask]

    # Add a new column for storing the drug term matches
    drugs_section_df['drug_terms_used'] = ''

    # Flank drug terms with white space for accurate match
    with open(drug_lex_path) as f:
        drug_terms = f.read().splitlines()
    drug_terms = [' ' + d + ' ' for d in drug_terms]

    # Populate the drug term matches column
    for index, row in drugs_section_df.iterrows():
        drugs_used = [drug for drug in drug_terms if drug in row.text]
        drugs_section_df.at[index, 'drug_terms_used'] = ','.join(drugs_used)
