"""This file contains the dataset reqder required for the model."""

import json
from typing import Dict, List

from allennlp.common.file_utils import cached_path
from allennlp.data import Instance
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Tokenizer, WordTokenizer
from allennlp.data.fields import Field, ListField, SequenceLabelField, TextField
from overrides import overrides


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
