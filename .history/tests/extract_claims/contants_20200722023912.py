"""File paths used in testing"""


import os

ROOT_DIR = os.path.abspath(os.path.join(__file__, "../../.."))


# model weights stored in local path
# MODEL_PATH = os.path.join(ROOT_DIR, "input", "model_weights", "claim_extraction" , "model_crf.tar.gz")

# WEIGHT_PATH = os.path.join(ROOT_DIR, "input", "model_weights", "claim_extraction" , "model_crf_tf.th")
MODEL_PATH = r"https://drive.google.com/drive/folders/1w2E7njiMAiFaOnns9_HdJOB1cjK4fl6N/model_crf.tar.gz"
WEIGHT_PATH = r"https://drive.google.com/drive/folders/1w2E7njiMAiFaOnns9_HdJOB1cjK4fl6N/model_crf_tf.th"