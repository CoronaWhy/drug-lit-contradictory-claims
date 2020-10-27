""" Test for training SBERT model."""

import os
import unittest

from contradictory_claims.models.sbert_models import build_sbert_model, load_sbert_model, save_sbert_model
from contradictory_claims.models.sbert_models import freeze_layer, unfreeze_layer


class TestSbertModel(unittest.TestCase):
    """Test SBERT Training modules."""

    def setUp(self):
        """Loads model to extract claims."""
        model_name = "covidbert"
        self.sbert_model, self.tokenizer = build_sbert_model(model_name)
        self.out_dir = 'tests/sbert_models/test_output'
        self.assertIsNotNone(self.sbert_model)
        self.assertIsNotNone(self.tokenizer)

    def test_freeze_layer(self):
        """Test if freeze layer function is working properly."""
        freeze_layer(self.sbert_model.linear)
        for param in self.sbert_model.linear.parameters():
            self.assertFalse(param.requires_grad)

    def test_unfreeze_layer(self):
        """Test if unfreeze layer function is working properly."""
        unfreeze_layer(self.sbert_model.linear)
        for param in self.sbert_model.linear.parameters():
            self.assertTrue(param.requires_grad)

    def test_save_load_sbert_model(self):
        """Test Loading and Saving of model."""
        os.makedirs(self.out_dir, exist_ok=True)
        save_sbert_model(self.sbert_model, self.out_dir)
        self.assertTrue(os.path.exists(self.out_dir))
        sbert_model = load_sbert_model(self.out_dir)
        self.assertIsNotNone(sbert_model)
