"""Tests for training the model for contradictory-claims."""

# -*- coding: utf-8 -*-

import copy
import os
import shutil
import unittest

import contradictory_claims.models.bluebert_train_model as blue
from contradictory_claims.models.train_model import build_model, load_model, regular_encode, save_model
import numpy as np
import tensorflow as tf
import torch
from torch.utils.data import DataLoader, RandomSampler
from transformers import AutoModel, AutoTokenizer, TFAutoModel


class TestTrainModel(unittest.TestCase):
    """Test for training the model for contradictory-claims."""

    def setUp(self) -> None:
        """Set up for the tests--load tokenizer."""
        self.test_tokenizer = AutoTokenizer.from_pretrained("allenai/biomed_roberta_base")
        self.model = AutoModel.from_pretrained("allenai/biomed_roberta_base")
        self.model.resize_token_embeddings(len(self.test_tokenizer))
        self.out_dir = 'tests/models/test_output'

    def test_regular_encode(self):
        """Test that encoding is done properly."""
        test_input = ["this is a test", "so is this"]
        len_encoding = 20
        encoded_input = regular_encode(test_input, self.test_tokenizer, len_encoding)
        expected_encoded_input = np.array([[0, 9226, 16, 10, 1296, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                           [0, 2527, 16, 42, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])

        self.assertTrue((encoded_input == expected_encoded_input).all())

    def test_build_save_load_model(self):
        """Test that full model is built properly."""
        strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())
        os.makedirs("biomed_roberta_base")
        self.model.save_pretrained("biomed_roberta_base")
        with strategy.scope():
            model = TFAutoModel.from_pretrained("biomed_roberta_base", from_pt=True)
            model = build_model(model)
        shutil.rmtree("biomed_roberta_base")

        # Note: this changed recently and I don't know why... Maybe different TF version?
        # self.assertEqual(str(type(model)), "<class 'tensorflow.python.keras.engine.training.Model'>")
        self.assertEqual(str(type(model)), "<class 'tensorflow.python.keras.engine.functional.Functional'>")

        save_model(model, timed_dir_name=False, transformer_dir=self.out_dir)

        self.assertTrue(os.path.isfile(os.path.join(self.out_dir, 'sigmoid.pickle')))
        self.assertTrue(os.path.isfile(os.path.join(self.out_dir, 'config.json')))
        self.assertTrue(os.path.isfile(os.path.join(self.out_dir, 'tf_model.h5')))

        pickle_path = os.path.join(self.out_dir, 'sigmoid.pickle')
        model = load_model(pickle_path=pickle_path, transformer_dir=self.out_dir)

        # Same comment here applies
        # self.assertEqual(str(type(model)), "<class 'tensorflow.python.keras.engine.training.Model'>")
        self.assertEqual(str(type(model)), "<class 'tensorflow.python.keras.engine.functional.Functional'>")

    @unittest.skip("Yeah I don't know how to reasonably test this sorry")
    def test_train_model(self):
        """Test that the model can be trained."""
        # What's a good way to test this?
        # TODO: Implement something
        pass

    def tearDown(self):
        """Clean-up after all tests have run."""
        if os.path.isdir(self.out_dir):
            shutil.rmtree(self.out_dir)

class TestBlueBertTrain(tf.test.TestCase):
    """Test for training BlueBert model for contradictory-claims."""
    def setUp(self) -> None:
        """Get local version of BlueBert model and sample inputs and lables"""
        super(TestBlueBertTrain, self).setUp()
        self.model, self.tokenizer, self.device = blue.bluebert_create_model("bionlp/bluebert_pubmed_uncased_L-12_H-768_A-12")
        self.test_inputs = ["This is not a test", "This is a test", "This might be a test", "This is definitely not a test"]
        self.test_labels = np.asarray([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0]])
        self.max_len = 10
        self.test_dataset = blue.ContraDataset(self.test_inputs, self.test_labels, self.tokenizer, max_len=self.max_len)

    def test_encode(self):
        """Check first element is returned properly by dataset"""
        claim, attn_mask, label = self.test_dataset.__getitem__(0)
        expected_claim = self.tokenizer(self.test_inputs[0],
                                        return_token_type_ids=False,
                                        padding='max_length',
                                        max_length=self.max_len,
                                        truncation=True)['input_ids']
        expected_attn_mask = tf.Variable([1., 1., 1., 1., 1., 1., 1., 0., 0., 0.])
        expected_label = tf.Variable([1, 0, 0])
        self.assertAllEqual(claim, expected_claim)
        self.assertAllEqual(attn_mask, expected_attn_mask)
        self.assertAllEqual(label, expected_label)

    def test_train(self):
        """Sanity check to make sure that after each training iteration, weights are changing and we eventually overfit"""
        test_sampler = RandomSampler(self.test_dataset)
        test_dataloader = DataLoader(self.test_dataset, sampler=test_sampler, batch_size=1)

        before = copy.deepcopy(self.model).parameters()
        blue.bluebert_train_model(self.model, test_dataloader, device=torch.device("cpu"))
        after = self.model.parameters()

        # Note: using this method in place of self.assertNotAllEqual because while most layers are different, 
        # some layers are unchanged after training.
        has_difference = False
        for b, a in zip(before, after):
            self.assertNotAllEqual(b.detach(), a.detach())
            if not torch.equal(b.detach(), a.detach()):
                has_difference = True
                break
        self.assertTrue(has_difference)
