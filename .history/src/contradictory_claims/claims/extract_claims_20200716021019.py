"""Function to extract Claims"""

import pandas as pd
import os

from tqdm import tqdm
import torch

from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor
from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.data.dataset_readers.dataset_reader import DatasetReader

from allennlp.common import Params
from allennlp.common.checks import check_dimensions_match
from allennlp.data import Vocabulary
from allennlp.modules import Seq2VecEncoder, TimeDistributed, TextFieldEmbedder, ConditionalRandomField, FeedForward
from allennlp.modules.conditional_random_field import allowed_transitions
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn import util
from allennlp.training.metrics import SpanBasedF1Measure, CategoricalAccuracy
from typing import Dict, Optional, List, Any
from overrides import overrides

from .utils import read_json
from discourse.dataset_readers import ClaimAnnotationReaderJSON, CrfPubmedRCTReader
from discourse.predictors import DiscourseClassifierPredictor


class DiscourseClassifierPredictor(Predictor):
    """"
    Predictor wrapper for the DiscourseClassifier
    """
    # @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        sentence = json_dict['sentence']
        instance = self._dataset_reader.text_to_instance(sent=sentence)
        return instance

class ClaimCrfPredictor(Predictor):
    """"
    Predictor wrapper for the AcademicPaperClassifier
    """
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
#         print(json_dict)
        sentences = json_dict['sentences']
        instance = self._dataset_reader.text_to_instance(sents=sentences)
        return instance

def load_model(MODEL_PATH = "https://s3-us-west-2.amazonaws.com/pubmed-rct/model_crf.tar.gz": str, WEIGHT_PATH  = "https://s3-us-west-2.amazonaws.com/pubmed-rct/model_crf_tf.th": str):
    """This loads the weight from the path specified
    
    :param MODEL_PATH: location of model path, can be downloaded offline or link can be given
    :param WEIGHT_PATH: location of model weight path, can be downloaded offline or link can be given, default link is specified
    
    :return: the model using the WEIGHT_PATH specified
    """

    archive = load_archive(MODEL_PATH) 
    predictor = Predictor.from_archive(archive, 'discourse_crf_predictor')
    # archive_ = load_archive("./model_crf.tar.gz")
    # discourse_predictor = Predictor.from_archive(archive_, 'discourse_crf_predictor')
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




def extract_claims(file_path: str, MODEL_PATH = "https://s3-us-west-2.amazonaws.com/pubmed-rct/model_crf.tar.gz": str, WEIGHT_PATH="https://s3-us-west-2.amazonaws.com/pubmed-rct/model_crf_tf.th": str, col_name = "sentence": str):
    """
    Extract Claims from given columns in a dataset to extract the claim
    
    :param file_path: path to input file, which contains sentences, file is expected to contain columns cord_uid, sentence, csv file
    :param WEIGHT_PATH: location to path where the model weights are kept, default path is specified of the place where model repo is kept
    :param col_name: name of column on which claim is to be identified
    :return: labels, if a sentence is a claim or not
    """

    claim_predictor=load_model(MODEL_PATH, WEIGHT_PATH)

    df=pd.read_csv(file_path)
    assert col_name in df.columns, f"No column named {col_name}"


import discourse
    

@DatasetReader.register("crf_pubmed_rct")
class CrfPubmedRCTReader(DatasetReader):
    """
    Reads a file from Pubmed RCT text file.
    Parameters
    ----------
    tokenizer : ``Tokenizer``, optional (default=``WordTokenizer()``)
        We use this ``Tokenizer`` for both the premise and the hypothesis.  See :class:`Tokenizer`.
    token_indexers : ``Dict[str, TokenIndexer]``, optional (default=``{"tokens": SingleIdTokenIndexer()}``)
        We similarly use this for both the premise and the hypothesis.  See :class:`TokenIndexer`.
    """

    def __init__(self,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}

    @overrides
    def _read(self, file_path):
        file_path = cached_path(file_path)
        with open(file_path, 'r') as file:
            sents = []
            labels = []
            for line in file:
                if not line.startswith('#') and line.strip() != '' and '\t' in line:
                    label, sent = line.split('\t')
                    sents.append(sent.strip())
                    labels.append(label)
                elif len(sents) > 0 and len(labels) > 0:
                    yield self.text_to_instance(sents, labels)
                    sents = []
                    labels = []
                else:
                    continue

    @overrides
    def text_to_instance(self,
                         sents: List[str],
                         labels: List[str] = None) -> Instance:
        fields: Dict[str, Field] = {}
        tokenized_sents = [self._tokenizer.tokenize(sent) for sent in sents]
        sentence_sequence = ListField([TextField(tk, self._token_indexers) for tk in tokenized_sents])
        fields['sentences'] = sentence_sequence
        if labels is not None:
            fields['labels'] = SequenceLabelField(labels, sentence_sequence)
        return Instance(fields)








