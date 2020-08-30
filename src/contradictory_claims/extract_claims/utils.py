"""This file contains the utilities needed to load the claim extraction module."""

import json
import os

MODEL_PATH = r"https://storage.googleapis.com/contradictory_claims_model_weights/model_crf.tar.gz"
WEIGHT_PATH = r"https://storage.googleapis.com/contradictory_claims_model_weights/model_crf_tf.th"


def read_json(file_path):
    """Read list from JSON path."""
    if not os.path.exists(file_path):
        return []
    else:
        with open(file_path, 'r') as fp:
            ls = [json.loads(line) for line in fp]
        return ls
