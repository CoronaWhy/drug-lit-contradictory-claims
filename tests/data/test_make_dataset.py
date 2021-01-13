"""Tests for making datasets for contradictory-claims."""

# -*- coding: utf-8 -*-

import os
import unittest

from contradictory_claims.data.make_dataset import create_mancon_sent_pairs_from_xml, \
    load_drug_virus_lexicons, load_mancon_corpus_from_sent_pairs, load_med_nli, load_multi_nli

from .constants import drug_lex_path, mancon_sent_pairs, mancon_xml_path, \
    mednli_dev_path, mednli_test_path, mednli_train_path, \
    multinli_test_path, multinli_train_path, sample_drug_lex_path, sample_mancon_sent_pairs, \
    sample_mancon_xml_path, sample_multinli_test_path, sample_multinli_train_path, \
    sample_virus_lex_path, virus_lex_path


class TestMakeDataset(unittest.TestCase):
    """Tests for making datasets for contradictory-claims."""

    @unittest.skip("This test can be used to check that datasets are found at the correct locations locally")
    def test_find_files(self):
        """Test that input files are found properly."""
        self.assertTrue(os.path.isfile(multinli_train_path),
                        f"MultiNLI training data not found at {multinli_train_path}")
        self.assertTrue(os.path.isfile(multinli_test_path),
                        f"MultiNLI test data not found at {multinli_test_path}")

        self.assertTrue(os.path.isfile(mednli_train_path),
                        f"MedNLI training data not found at {mednli_train_path}")
        self.assertTrue(os.path.isfile(mednli_dev_path),
                        f"MedNLI dev set data not found at {mednli_dev_path}")
        self.assertTrue(os.path.isfile(mednli_test_path),
                        f"MedNLI test data not found at {mednli_test_path}")

        self.assertTrue(os.path.isfile(mancon_sent_pairs),
                        f"ManConCorpus sentence pairs training data not found at {mancon_sent_pairs}")

        self.assertTrue(os.path.isfile(drug_lex_path),
                        f"Drug lexicon not found at {drug_lex_path}")
        self.assertTrue(os.path.isfile(virus_lex_path),
                        f"Virus lexicon not found at {virus_lex_path}")

    @unittest.skip("This test can be used locally to check that MultiNLI loads properly")
    def test_load_multi_nli(self):
        """Test that MultiNLI is loaded as expected."""
        x_train, y_train, x_test, y_test = load_multi_nli(multinli_train_path, multinli_test_path)

        self.assertEqual(len(x_train), 391165)
        self.assertEqual(y_train.shape, (391165, 3))
        self.assertEqual(len(x_test), 9897)
        self.assertEqual(y_test.shape, (9897, 3))

    def test_load_multi_nli_sample(self):
        """Test that MultiNLI SAMPLE DATA are loaded as expected."""
        x_train, y_train, x_test, y_test = load_multi_nli(sample_multinli_train_path, sample_multinli_test_path)

        self.assertEqual(len(x_train), 49)
        self.assertEqual(y_train.shape, (49, 3))
        self.assertEqual(len(x_test), 49)
        self.assertEqual(y_test.shape, (49, 3))

    @unittest.skip("This test can be used locally to check that MedNLI loads properly")
    def test_load_med_nli(self):
        """Test that MedNLI is loaded as expected."""
        x_train, y_train, x_test, y_test = load_med_nli(mednli_train_path, mednli_dev_path, mednli_test_path)

        self.assertEqual(len(x_train), 12627)
        self.assertEqual(y_train.shape, (12627, 3))
        self.assertEqual(len(x_test), 1422)
        self.assertEqual(y_test.shape, (1422, 3))

    @unittest.skip("This test can be used locally to check that ManConCorpus loads properly")
    def test_load_mancon_corpus_from_sent_pairs(self):
        """Test that ManConCorpus is loaded as expected."""
        create_mancon_sent_pairs_from_xml(mancon_xml_path, mancon_sent_pairs)
        x_train, y_train, x_test, y_test = load_mancon_corpus_from_sent_pairs(mancon_sent_pairs)

        # self.assertEqual(len(x_train), 14328)
        # self.assertEqual(y_train.shape, (14328, 3))
        # self.assertEqual(len(x_test), 3583)
        # self.assertEqual(y_test.shape, (3583, 3))
        self.assertEqual(len(x_train), 50453)
        self.assertEqual(y_train.shape, (50453, 3))
        self.assertEqual(len(x_test), 12614)
        self.assertEqual(y_test.shape, (12614, 3))

    def test_load_mancon_corpus_from_sent_pairs_sample(self):
        """Test that ManConCorpus is loaded as expected."""
        create_mancon_sent_pairs_from_xml(sample_mancon_xml_path, sample_mancon_sent_pairs)
        x_train, y_train, x_test, y_test = load_mancon_corpus_from_sent_pairs(sample_mancon_sent_pairs)

        # self.assertEqual(len(x_train), 39)
        # self.assertEqual(y_train.shape, (39, 3))
        # self.assertEqual(len(x_test), 10)
        # self.assertEqual(y_test.shape, (10, 3))
        self.assertEqual(len(x_train), 50453)
        self.assertEqual(y_train.shape, (50453, 3))
        self.assertEqual(len(x_test), 12614)
        self.assertEqual(y_test.shape, (12614, 3))

    def test_load_drug_virus_lexicons(self):
        """Test that the virus and drug lexicons are loaded properly."""
        drug_names, virus_names = load_drug_virus_lexicons(sample_drug_lex_path, sample_virus_lex_path)

        drugs = ["hydroxychloroquine", "remdesivir", "ritonavir", "chloroquine", "lopinavir"]
        virus_syns = ["COVID-19", "SARS-CoV-2", "Coronavirus Disease 2019"]
        self.assertTrue(set(drugs).issubset(set(drug_names)))
        self.assertTrue(set(virus_syns).issubset(set(virus_names)))
