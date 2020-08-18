"""Tests for preprocessing CORD-19 data."""

# -*- coding: utf-8 -*-

import os
import unittest
# from datetime import datetime

import pandas as pd
from contradictory_claims.data.preprocess_cord import clean_text, construct_regex_match_pattern,\
    extract_json_to_dataframe, extract_regex_pattern, extract_section_from_text, filter_metadata_for_covid19,\
    filter_section_with_drugs, merge_section_text

from .constants import pdf_filenames, pmc_filenames, pub_date_cutoff,\
    sample_conclusion_search_terms_path, sample_covid19_df_path, sample_drug_lex_path, sample_json_temp_path,\
    sample_json_text_file_dir, sample_metadata_path, sample_virus_lex_path


class TestPreprocessCord(unittest.TestCase):
    """Tests for preprocessing CORD-19."""

    def test_find_files(self):
        """Test that input files are found properly."""
        self.assertTrue(os.path.isfile(sample_metadata_path),
                        f"Metadata.csv not found at {sample_metadata_path}")

    def test_pub_date_cutoff(self):
        """Test that incorrect publish date cut-off format throws error."""
        # self.assertIsInstance(datetime.strptime(pub_date_cutoff, "%Y-%m-%d"), datetime)
        with self.assertRaises(ValueError):
            filter_metadata_for_covid19(sample_metadata_path, sample_virus_lex_path, '20191001')

    def filter_metadata_for_covid19(self):
        """Test that CORD-19 metadata is filtered properly."""
        covid_metadata = filter_metadata_for_covid19(sample_metadata_path, sample_virus_lex_path, pub_date_cutoff)
        self.assertEqual(len(covid_metadata), 4)

    def test_extract_json_to_dataframe(self):
        """Test that CORD-19 json files are loaded properly."""
        covid_metadata = filter_metadata_for_covid19(sample_metadata_path, sample_virus_lex_path, pub_date_cutoff)
        covid19_df = extract_json_to_dataframe(covid_metadata, sample_json_text_file_dir, sample_json_temp_path,
                                               pdf_filenames, pmc_filenames)
        self.assertEqual(len(covid19_df.columns), 3)
        self.assertTrue(len(covid19_df) >= 1)

    def test_construct_regex_match_pattern(self):
        """Test that regex pattern is constrcuted properly."""
        regex_pattern = construct_regex_match_pattern(sample_conclusion_search_terms_path)
        self.assertEqual(regex_pattern, '.*conclusion.*|.*discussion.*')

    def test_extract_regex_pattern(self):
        """Test that regex pattern is extracted properly."""
        unique_sections = ['IV -Discussion',
                           'Additional considerations for large-scale manufacturing and dissemination '
                           '::: Discussion',
                           'Discussion of case reports',
                           'Conclusion']
        section_list = extract_regex_pattern(unique_sections, '.*conclusion.*|.*discussion.*')
        self.assertEqual(len(section_list), 4)

    def test_extract_section_from_text(self):
        """Test that title, abstract, and conclusion sections are extracted properly from publication text."""
        covid19_df = pd.read_csv(sample_covid19_df_path)
        covid19_filt_section_df = extract_section_from_text(sample_conclusion_search_terms_path, covid19_df)
        self.assertEqual(len(covid19_filt_section_df), 8)

    def test_clean_text(self):
        """Test that text is cleaned properly."""
        covid19_df = pd.read_csv(sample_covid19_df_path)
        clean_df = clean_text(covid19_df)
        self.assertEqual(len(clean_df), 7)

    def test_merge_section_text(self):
        """Test that section text is merged properly."""
        covid19_df = pd.read_csv(sample_covid19_df_path)
        merged_df = merge_section_text(covid19_df)
        self.assertEqual(len(merged_df), 6)
        self.assertEqual((merged_df.columns == ['cord_uid', 'text', 'section']).all())

    def test_filter_section_with_drugs(self):
        """Test that section filtering for drugs is performed properly."""
        covid19_df = pd.read_csv(sample_covid19_df_path)
        covid19_df.rename(columns={'sentence': 'text'}, inplace=True)
        drugs_section_df = filter_section_with_drugs(covid19_df, sample_drug_lex_path)
        self.assertEqual(len(drugs_section_df), 3)
        self.assertTrue((drugs_section_df.columns == ['cord_uid', 'text', 'section', 'drug_terms_used']).all())
