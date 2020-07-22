"""Testing claim extraction functions"""

import pandas
import numpy
from contradictory_claims.claims.extract_claims import load_model, load_archive
import os
import shutil
import unittest
from .contants import MODEL_PATH, WEIGHT_PATH


class TestExtractClaims(unittest.TestCase):
    """Test for loading the model and returning claims"""

    def test_load_model(self) -> None:
        """Loads the model to extract claims"""
        self.model=load_model(MODEL_PATH = MODEL_PATH, WEIGHT_PATH= WEIGHT_PATH)

    def test_extract_claims(self):
        """Check if it indeed returns claims"""
        # TODO implement something
        pass
    




