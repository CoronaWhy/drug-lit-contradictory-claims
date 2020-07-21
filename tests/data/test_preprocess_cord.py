"""Tests for making datasets for contradictory-claims."""

# -*- coding: utf-8 -*-

import os
import unittest
from datetime import datetime

from contradictory_claims.data.preprocess_cord import extract_json_to_dataframe,\
    filter_metadata_for_covid19

from .constants import pdf_filenames, pmc_filenames, pub_date_cutoff,\
    sample_json_temp_path, sample_json_text_file_dir, sample_metadata_path,\
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
