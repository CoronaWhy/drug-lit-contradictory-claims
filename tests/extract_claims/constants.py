"""File paths used in testing."""


import os

ROOT_DIR = os.path.abspath(os.path.join(__file__, "../../.."))


# model weights stored in local path
# MODEL_PATH = os.path.join(ROOT_DIR, "input", "model_weights", "claim_extraction" , "model_crf.tar.gz")

# WEIGHT_PATH = os.path.join(ROOT_DIR, "input", "model_weights", "claim_extraction" , "model_crf_tf.th")
MODEL_PATH = r"https://storage.googleapis.com/contradictory_claims_model_weights/model_crf.tar.gz"
WEIGHT_PATH = r"https://storage.googleapis.com/contradictory_claims_model_weights/model_crf_tf.th"
