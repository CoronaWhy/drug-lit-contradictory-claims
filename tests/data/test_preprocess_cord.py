"""Tests for making datasets for contradictory-claims."""

# -*- coding: utf-8 -*-

import os
import unittest
from datetime import datetime

import pandas as pd
from contradictory_claims.data.preprocess_cord import clean_text, construct_regex_match_pattern,\
    extract_json_to_dataframe, extract_regex_pattern, filter_metadata_for_covid19

from .constants import pdf_filenames, pmc_filenames, pub_date_cutoff,\
    sample_covid19_df_path, sample_json_temp_path, sample_json_text_file_dir, sample_metadata_path,\
    sample_virus_lex_path


class TestPreprocessCord(unittest.TestCase):
    """Tests for preprocessing CORD-19."""

    def test_find_files(self):
        """Test that input files are found properly."""
        self.assertTrue(os.path.isfile(sample_metadata_path),
                        "Metadata.csv not found at {}".format(sample_metadata_path))

    def test_pub_date_cutoff(self):
        """Test that publish date cut-off is in the correct format."""
        self.assertIsInstance(datetime.strptime(pub_date_cutoff, "%Y-%m-%d"), datetime)
#        try:
#            datetime.strptime(pub_date_cutoff, "%Y-%m-%d")
#        except ValueError:
#            raise ValueError("Incorrect date format, should be YYYY-MM-DD")

    def test_extract_json_to_dataframe(self):
        """Test that json files are loaded properly."""
        covid_metadata = filter_metadata_for_covid19(sample_metadata_path, sample_virus_lex_path, pub_date_cutoff)
        covid19_df = extract_json_to_dataframe(covid_metadata, sample_json_text_file_dir, sample_json_temp_path,
                                               pdf_filenames, pmc_filenames)
        self.assertEqual(len(covid19_df.columns), 3)
        self.assertTrue(len(covid19_df) >= 1)

    def test_construct_regex_match_pattern(self):
        """Test that regex pattern is constrcuted properly."""
        regex_pattern = construct_regex_match_pattern(['conclusion', 'discussion'])
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

    def test_clean_text(self):
        """Test that text is cleaned properly."""
        covid19_df = pd.read_csv(sample_covid19_df_path)
        clean_df = clean_text(covid19_df)
        self.assertEqual(len(clean_df), 6)
