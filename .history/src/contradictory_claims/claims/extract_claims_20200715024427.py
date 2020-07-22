"""Function to extract Claims"""

import pandas as pd
import os

from tqdm import tqdm
import torch

from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor

class ClaimCrfPredictor(Predictor):
    """"
    Predictor wrapper for the AcademicPaperClassifier
    """
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
#         print(json_dict)
        sentences = json_dict['sentences']
        instance = self._dataset_reader.text_to_instance(sents=sentences)
        return instance


def extract_claims(file_path: str):
    """
    Extract Claims from given columns in a dataset to extract the claim
    
    :param: file_path: path to input file, which contains sentences, file is expected to contain columns cord_uid, sentence 

    :return: labels, if a sentence is a claim or not
    """

    archive = load_archive("./model_crf.tar.gz") ## available at github
    predictor = Predictor.from_archive(archive, 'discourse_crf_predictor')
    archive_ = load_archive("./model_crf.tar.gz")
    discourse_predictor = Predictor.from_archive(archive_, 'discourse_crf_predictor')


