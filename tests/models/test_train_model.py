"""Tests for training the model for contradictory-claims."""

# -*- coding: utf-8 -*-

import os
import shutil
import unittest

import numpy as np
import tensorflow as tf
from contradictory_claims.models.train_model import build_model, regular_encode
from transformers import AutoModel, AutoTokenizer, TFAutoModel


class TestTrainModel(unittest.TestCase):
    """Test for training the model for contradictory-claims."""

    def setUp(self) -> None:
        """Set up for the tests--load tokenizer."""
        self.test_tokenizer = AutoTokenizer.from_pretrained("allenai/biomed_roberta_base")

    def test_regular_encode(self):
        """Test that encoding is done properly."""
        test_input = ["this is a test", "so is this"]
        len_encoding = 20
        encoded_input = regular_encode(test_input, self.test_tokenizer, len_encoding)
        expected_encoded_input = np.array([[0, 9226, 16, 10, 1296, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                           [0, 2527, 16, 42, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])

        self.assertTrue((encoded_input == expected_encoded_input).all())

    def test_build_model(self):
        """Test that full model is built properly."""
        strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())
        model = AutoModel.from_pretrained("allenai/biomed_roberta_base")
        model.resize_token_embeddings(len(self.test_tokenizer))
        os.makedirs("biomed_roberta_base")
        model.save_pretrained("biomed_roberta_base")
        with strategy.scope():
            model = TFAutoModel.from_pretrained("biomed_roberta_base", from_pt=True)
            model = build_model(model)
        shutil.rmtree("biomed_roberta_base")

        self.assertEqual(str(type(model)), "<class 'tensorflow.python.keras.engine.training.Model'>")

    def test_save_model(self):
        """Test that the model can be saved."""
        # TODO: Implement something
        pass

    def test_load_model(self):
        """Test that the model can be loaded."""
        # TODO: Implement something
        pass

    def test_train_model(self):
        """Test that the model can be trained."""
        # What's a good way to test this?
        # TODO: Implement something
        pass
