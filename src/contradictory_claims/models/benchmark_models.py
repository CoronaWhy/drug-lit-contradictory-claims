"""Functions for benchmarking against when finding contradictory claims."""

# -*- coding: utf-8 -*-

import os
import ssl

import gensim.downloader as api
import logging
import pandas as pd

from fse import SplitIndexedList
from fse.models import uSIF
from scipy.spatial import distance
from ..data.prepare_claims_for_roam import polarity_v_score, polarity_tb_score


INPUT_CLAIMS_FILE = "/Users/dnsosa/Downloads/processed_claims_150820.csv"

if not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None):
    ssl._create_default_https_context = ssl._create_unverified_context

logging.basicConfig(format='%(asctime)s : %(threadName)s : %(levelname)s : %(message)s', level=logging.INFO)
glove = api.load("glove-wiki-gigaword-300")

all_claims_df = pd.read_csv(INPUT_CLAIMS_FILE)
all_claims = all_claims_df.claims.values
s = SplitIndexedList(all_claims)

# Train the uSIF model
model = uSIF(glove, workers=2, lang_freq="en")
model.train(s)


def uSIF_similarity(text1: str, text2: str, model) -> float:
    """
    Calculate sentence-level similarity using uSIF.
    :param text1: sentence 1
    :param text2: sentence 2
    :param model: trained model for inferring new sentence vectors
    :return similarity: uSIF similarity
    """
    input1 = (text1.split(), 0)
    input2 = (text2.split(), 0)
    vec1 = model.infer(input1)
    vec2 = model.infer(input2)
    similarity = distance.cosine(vec1, vec2)
    return similarity


def w2v_similarity(text1: str, text2: str) -> float:
    """
    Calculate sentence-level similarity using uSIF.
    :param text1: sentence 1
    :param text2: sentence 2
    :return similarity: uSIF similarity
    """
    pass


def polarity_similarity_classifier(text1: str,
                        text2: str,
                        min_polarity: float = 0.0,
                        vader: bool = True,
                        use_similarity: bool = True,
                        sim_threshold: float = 0.5,
                        sim_method: str = "uSIF") -> int:
    """
    Classify as entail, contradict, neutral based on polarity and similarity. If polarity is opposite = contradiction,
    if polarity is same = entailment, if at least one isn't polar, then neutral.
    :param text1: claim 1
    :param text2: claim 2
    :param min_polarity: minimum absolute polarity to call positive or negative
    :param vader: if True, use Vader for polarity detection; else use TextBlob
    :param use_similarity: if True, claims must pass similarity threshold as well based on ___ embeddings
    :param sim_threshold: minimum threshold (cosine distance) for calling pair sufficiently similar
    :param sim_method: type of similarity method to use, can be {uSIF or W2V}
    :return: -1 = contradiction, 0 = neutral, 1 = entail [CHECK THIS]
    """
    if min_polarity < 0:
        assert f"min_polarity needs to be positive. Instead entered {min_polarity}"

    if vader:
        pol1 = polarity_v_score(text1)
        pol2 = polarity_v_score(text2)
    else:
        pol1 = polarity_tb_score(text1)
        pol2 = polarity_tb_score(text2)

    if use_similarity:
        if sim_method == "uSIF":
            similarity = uSIF_similarity(text1, text2)
        elif sim_method == "W2V":
            similarity = w2v_similarity(text1, text2)
        else:
            assert f"sim_method must be a valid option. Instead entered {sim_method}"

        if similarity < sim_threshold:
            return 0

    if pol1 > min_polarity and pol2 < -min_polarity:
        return -1
    elif (pol1 > min_polarity and pol2 > min_polarity) or (pol1 < -min_polarity and pol2 < -min_polarity):
        return 1
    else:
        return 0



