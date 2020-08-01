"""This file contains the utilities needed to load the claim extraction module."""

import json
import os
from typing import Dict, List, Optional

# import spacy
import en_core_web_sm
import torch
from allennlp.common.file_utils import cached_path
from allennlp.common.util import JsonDict
from allennlp.data import Instance, Vocabulary
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, ListField, SequenceLabelField, TextField
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Tokenizer, WordTokenizer
from allennlp.models.model import Model
from allennlp.modules import ConditionalRandomField, Seq2VecEncoder, TimeDistributed
from allennlp.modules.text_field_embedders import TextFieldEmbedder  # noqa:F401
from allennlp.nn import InitializerApplicator, RegularizerApplicator  # noqa:F401
from allennlp.nn import util
from allennlp.predictors import Predictor
from allennlp.training.metrics import CategoricalAccuracy
from overrides import overrides
from torch.nn.modules.linear import Linear


MODEL_PATH = r"https://storage.googleapis.com/contradictory_claims_model_weights/model_crf.tar.gz"
WEIGHT_PATH = r"https://storage.googleapis.com/contradictory_claims_model_weights/model_crf_tf.th"
# nlp = spacy.load('en_core_web_sm')
nlp = en_core_web_sm.load()


def read_json(file_path):
    """Read list from JSON path."""
    if not os.path.exists(file_path):
        return []
    else:
        with open(file_path, 'r') as fp:
            ls = [json.loads(line) for line in fp]
        return ls


@Model.register("discourse_crf_classifier")
class DiscourseCrfClassifier(Model):
    """Discourse CRF model class."""

    global InitializerApplicator

    def __init__(self,
                 vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,  # noqaF821
                 sentence_encoder: Seq2VecEncoder,
                 initializer: InitializerApplicator = InitializerApplicator(), # noqaB008
                 dropout: Optional[float] = None,
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        """
        Initialize the class.

        :param vocab: The vocabulary file
        :param text_field_embedder: TextFieldEmbedder class object
        :param: sentence_encoder: Seq2VecEncoder class object
        :param initializer: InitializerApplicator object
        :param dropout: float value of dropout
        :param regularizer: RegularizerApplication
        """
        super(DiscourseCrfClassifier, self).__init__(vocab, regularizer)

        self.text_field_embedder = text_field_embedder
        self.num_classes = self.vocab.get_vocab_size("labels")
        self.sentence_encoder = sentence_encoder
        if dropout:
            self.dropout = torch.nn.Dropout(dropout)
        else:
            self.dropout = None
        self.metrics = {
            "accuracy": CategoricalAccuracy(),
            "accuracy3": CategoricalAccuracy(top_k=3),
        }
        self.loss = torch.nn.CrossEntropyLoss()
        self.label_projection_layer = TimeDistributed(Linear(self.sentence_encoder.get_output_dim(),
                                                             self.num_classes))

        constraints = None  # allowed_transitions(label_encoding, labels)
        self.crf = ConditionalRandomField(
            self.num_classes, constraints,
            include_start_end_transitions=False,
        )
        initializer(self)

    @overrides
    def forward(self,
                sentences: Dict[str, torch.LongTensor],
                labels: torch.LongTensor = None) -> Dict[str, torch.Tensor]:
        """Forward pass of the model."""
        # print(sentences['tokens'].size())
        # print(labels.size())

        embedded_sentences = self.text_field_embedder(sentences)
        token_masks = util.get_text_field_mask(sentences, 1)
        sentence_masks = util.get_text_field_mask(sentences)

        # get sentence embedding
        encoded_sentences = []
        n_sents = embedded_sentences.size()[1]  # size: (n_batch, n_sents, n_tokens, n_embedding)
        for i in range(n_sents):
            encoded_sentences.append(self.sentence_encoder(embedded_sentences[:, i, :, :], token_masks[:, i, :]))
        encoded_sentences = torch.stack(encoded_sentences, 1)

        # dropout layer
        if self.dropout:
            encoded_sentences = self.dropout(encoded_sentences)

        # print(encoded_sentences.size()) # size: (n_batch, n_sents, n_embedding)

        # CRF prediction
        logits = self.label_projection_layer(encoded_sentences)  # size: (n_batch, n_sents, n_classes)
        best_paths = self.crf.viterbi_tags(logits, sentence_masks)
        predicted_labels = [x for x, y in best_paths]

        output_dict = {
            "logits": logits,
            "mask": sentence_masks,
            "labels": predicted_labels,
        }

        # referring to https://github.com/allenai/allennlp/blob/master/allennlp/models/crf_tagger.py#L229-L239
        if labels is not None:
            log_likelihood = self.crf(logits, labels, sentence_masks)
            output_dict["loss"] = -log_likelihood

            class_probabilities = logits * 0.
            for i, instance_labels in enumerate(predicted_labels):
                for j, label_id in enumerate(instance_labels):
                    class_probabilities[i, j, label_id] = 1

            for metric in self.metrics.values():
                metric(class_probabilities, labels, sentence_masks.float())

        return output_dict

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Covert tag ids to actual tags."""
        output_dict["labels"] = [[self.vocab.get_token_from_index(label, namespace='labels')
                                  for label in instance_labels]
                                 for instance_labels in output_dict["labels"]
                                 ]
        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        """Return the metrics as dictionary."""
        return {metric_name: metric.get_metric(reset) for metric_name, metric in self.metrics.items()}


@DatasetReader.register("claim_annotation_json")
class ClaimAnnotationReaderJSON(DatasetReader):
    """
    A ``DatasetReader`` knows how to turn a file containing a dataset into a collection of ``Instance`` s.

    Reading annotation from dataset.
    Reading annotation dataset in the following JSON format:
    {
        "paper_id": ...,
        "user_id": ...,
        "sentences": [..., ..., ...],
        "labels": [..., ..., ...]
    }
    """

    def __init__(self,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 lazy: bool = False) -> None:
        """Initialize the datareader with tokenizer and token Indexers."""
        super().__init__(lazy)
        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}

    @overrides
    def _read(self, file_path):
        """Read the json file and returns in fixed format the sentences and labels."""
        file_path = cached_path(file_path)
        with open(file_path, 'r') as file:
            for line in file:
                example = json.loads(line)
                sents = example['sentences']
                labels = example['labels']
                yield self.text_to_instance(sents, labels)

    @overrides
    def text_to_instance(self,
                         sents: List[str],
                         labels: List[str] = None) -> Instance:
        """
        Doing whatever tokenization or processing is necessary to go from textual input to an ``Instance``.

        Returns Instance objects when given sentence and objects as inputs
        :params sents: list of sentences
        :params: labels: list of labels corresponding to the sentences
        :return: Instance object
        """
        fields: Dict[str, Field] = {}
        tokenized_sents = [self._tokenizer.tokenize(sent) for sent in sents]
        sentence_sequence = ListField([TextField(tk, self._token_indexers) for tk in tokenized_sents])
        fields['sentences'] = sentence_sequence

        if labels is not None:
            fields['labels'] = SequenceLabelField(labels, sentence_sequence)
        return Instance(fields)


@DatasetReader.register("crf_pubmed_rct")
class CrfPubmedRCTReader(DatasetReader):
    """
    A ``DatasetReader`` knows how to turn a file containing a dataset into a collection of ``Instance`` s.

    Reads a file from Pubmed RCT text file.
    Parameters
    ----------
    tokenizer : ``Tokenizer``, optional
        We use this ``Tokenizer`` for both the premise and the hypothesis.  See :class:`Tokenizer`.
    token_indexers : ``Dict[str, TokenIndexer]``, optional (default=``{"tokens": SingleIdTokenIndexer()}``)
        We similarly use this for both the premise and the hypothesis.  See :class:`TokenIndexer`.
    """

    def __init__(self,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 lazy: bool = False) -> None:
        """Initialize the datareader with tokenizer and token Indexers."""
        super().__init__(lazy)
        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}

    @overrides
    def _read(self, file_path):
        """Read method overwritten."""
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
        """
        Doing whatever tokenization or processing is necessary to go from textual input to an ``Instance``.

        Returns Instance objects when given sentence and objects as inputs
        :params sents: list of sentences
        :params: labels: list of labels corresponding to the sentences
        :return: Instance object
        """
        fields: Dict[str, Field] = {}
        tokenized_sents = [self._tokenizer.tokenize(sent) for sent in sents]
        sentence_sequence = ListField([TextField(tk, self._token_indexers) for tk in tokenized_sents])
        fields['sentences'] = sentence_sequence
        if labels is not None:
            fields['labels'] = SequenceLabelField(labels, sentence_sequence)
        return Instance(fields)


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
