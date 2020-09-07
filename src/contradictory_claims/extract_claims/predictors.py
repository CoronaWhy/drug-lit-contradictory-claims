"""This file contains the predictoer required for predicting, whether it is a claim or not."""


import en_core_web_sm
from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.predictors import Predictor

nlp = en_core_web_sm.load()


class ClaimCrfPredictor(Predictor):
    """Predictor wrapper for the AcademicPaperClassifier."""

    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        """Json to instance override."""
        # print(json_dict)
        sentences = json_dict['sentences']
        instance = self._dataset_reader.text_to_instance(sents=sentences)
        return instance
