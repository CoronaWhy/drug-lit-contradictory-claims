"""Tests for processing CORD-19 claims."""

# -*- coding: utf-8 -*-

import unittest
# Disable sorting of test methods so they run in the same order as defined below,
# since we want a sequential data flow between the tests

import pandas as pd
from contradictory_claims.data.process_claims import add_cord_metadata, initialize_nlp,\
    pair_similar_claims, split_papers_on_claim_presence, tokenize_section_text

from .constants import sample_metadata_path, sample_raw_claims_df_path, sample_virus_lex_path


# unittest.TestLoader.sortTestMethodsUsing = None


class TestProcessClaims(unittest.TestCase):
    """Tests for processing CORD-19 claims."""

    def setUp(self) -> None:
        """Set up class variables to pass between test functions."""
        self.__class__.claims_df = pd.read_csv(sample_raw_claims_df_path)
        self.__class__.claims_data = pd.DataFrame()
        self.__class__.no_claims_data = pd.DataFrame()
        self.__class__.claims_paired_df = pd.DataFrame()

    def test_1_split_papers_on_claim_presence(self):
        """Test that papers are split correctly based on claim presence."""
        self.__class__.claims_data, self.__class__.no_claims_data\
            = split_papers_on_claim_presence(self.__class__.claims_df)
        self.assertEqual(len(self.__class__.claims_data), 12)
        self.assertEqual(len(self.__class__.no_claims_data), 5)

    def test_2_tokenize_section_text(self):
        """Test that section text is tokenized properly."""
        tok_no_claims_data = tokenize_section_text(self.__class__.no_claims_data)
        self.assertEqual(len(tok_no_claims_data), 15)

    def test_3_initialize_nlp(self):
        """Test that scispacy nlp object is initialized properly."""
        nlp = initialize_nlp(sample_virus_lex_path, "en_core_sci_sm")
        self.assertEqual(type(nlp), 'spacy.lang.en.English')

    def test_4_pair_similar_claims(self):
        """Test that CORD-19 claims are paired properly."""
        nlp = initialize_nlp(sample_virus_lex_path, "en_core_sci_sm")
        self.claims_paired_df = pair_similar_claims(self.claims_data, nlp)
        self.assertTrue(len(self.claims_paired_df) >= 1)
        self.assertEqual(len(self.claims_paired_df.columns), 7)

    def test_5_add_cord_metadata(self):
        """Test that input CORD metadata is added properly."""
        claims_paired_meta_df = add_cord_metadata(self.claims_paired_df, sample_metadata_path)
        self.assertEqual(len(claims_paired_meta_df.columns), 11)
