"""Tests for making datasets for contradictory-claims."""

# -*- coding: utf-8 -*-

import os
import unittest
from datetime import datetime

# from contradictory_claims.data.preprocess_cord import filter_metadata_for_covid19

from .constants import metadata_path, pub_date_cutoff


class TestPreprocessCord(unittest.TestCase):
    """Tests for preprocessing CORD-19."""

    def test_find_files(self):
        """Test that input files are found properly."""
        self.assertTrue(os.path.isfile(metadata_path),
                        "Metadata.csv not found at {}".format(metadata_path))

    def test_pub_date_cutoff(self):
        """Test that publish date cut-off is in the correct format."""
#        self.assertIsInstance(datetime.strptime('2019-10-01', "%Y-%m-%d"),datetime)
        try:
#        with self.assertRaises(ValueError):
            datetime.strptime(pub_date_cutoff, "%Y-%m-%d")
        except ValueError:
            raise ValueError("Incorrect date format, should be YYYY-MM-DD")
