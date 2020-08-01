"""This file contains the predictoer required for predicting, whether it is a claim or not."""


import en_core_web_sm
from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.predictors import Predictor
from overrides import overrides

nlp = en_core_web_sm.load()


@Predictor.register('discourse_predictor')
class DiscourseClassifierPredictor(Predictor):
    """Predictor wrapper for the DiscourseClassifier."""

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        """Json to instance override."""
        sentence = json_dict['sentence']
        instance = self._dataset_reader.text_to_instance(sent=sentence)
        return instance


@Predictor.register('discourse_crf_predictor')
class DiscourseCRFClassifierPredictor(Predictor):
    """Predictor wrapper for the DiscourseClassifier."""

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        """Json to instance override."""
        abstract = json_dict["abstract"]
        abstract = nlp(abstract)
        sentences = [sent.text.strip() for sent in abstract.sents]
        instance = self._dataset_reader.text_to_instance(sents=sentences)
        return instance


class ClaimCrfPredictor(Predictor):
    """Predictor wrapper for the AcademicPaperClassifier."""

    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        """Json to instance override."""
        # print(json_dict)
        sentences = json_dict['sentences']
        instance = self._dataset_reader.text_to_instance(sents=sentences)
        return instance
