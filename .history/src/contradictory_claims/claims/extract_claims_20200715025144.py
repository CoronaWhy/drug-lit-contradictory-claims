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

def load_model(WEIGHT_PATH="https://s3-us-west-2.amazonaws.com/pubmed-rct/model_crf_tf.th": str):
    """This loads the weight from the path specified
    
    :param WEIGHT_PATH: location of model path, can be downloaded offline or link can be given, default link is specified
    
    :return: the model using the WEIGHT_PATH specified
    """


    model = predictor._model
    for param in list(model.parameters()):
        param.requires_grad = False ## not train weights
    EMBEDDING_DIM = 300
    num_classes, constraints, include_start_end_transitions = 2, None, False
    model.crf = ConditionalRandomField(num_classes, constraints, 
                                    include_start_end_transitions=include_start_end_transitions)
    model.label_projection_layer = TimeDistributed(Linear(2 * EMBEDDING_DIM, num_classes))
    model.load_state_dict(torch.load(cached_path(WEIGHT_PATH), map_location='cpu'))
    reader = CrfPubmedRCTReader()
    claim_predictor = ClaimCrfPredictor(model, dataset_reader=reader)
    return claim_predictor




def extract_claims(file_path: str, WEIGHT_PATH="https://s3-us-west-2.amazonaws.com/pubmed-rct/model_crf_tf.th": str):
    """
    Extract Claims from given columns in a dataset to extract the claim
    
    :param file_path: path to input file, which contains sentences, file is expected to contain columns cord_uid, sentence 
    :param WEIGHT_PATH: location to path where the model weights are kept, default path is specified of the place where model repo is kept
    
    :return: labels, if a sentence is a claim or not
    """

    archive = load_archive("./model_crf.tar.gz") ## available at github
    predictor = Predictor.from_archive(archive, 'discourse_crf_predictor')
    archive_ = load_archive("./model_crf.tar.gz")
    discourse_predictor = Predictor.from_archive(archive_, 'discourse_crf_predictor')
    WEIGHT_PATH
    claim_predictor=


