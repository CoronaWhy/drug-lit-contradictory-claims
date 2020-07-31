"""Testing claim extraction functions."""


import unittest

from contradictory_claims.claims.extract_claims import load_model

from .constants import MODEL_PATH, WEIGHT_PATH


class TestExtractClaims(unittest.TestCase):
    """Test for loading the model and returning claims."""

    def test_load_model(self) -> None:
        """Loads the model to extract claims."""
        self.model = load_model(model_path=MODEL_PATH, weight_path=WEIGHT_PATH)

    def test_extract_claims(self):
        """Check if it indeed returns claims."""
        # TODO implement something
        pass
