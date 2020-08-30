"""Function to extract Claims."""


try:
    import discourse  # noqa:F401
except Exception:
    import pip
    pip._internal.main(["install", "git+https://github.com/titipata/detecting-scientific-claim.git"])
    import discourse  # noqa:F401

import nltk
import numpy as np
import pandas as pd
import torch
from allennlp.common.file_utils import cached_path
from allennlp.models.archival import load_archive
from allennlp.modules import ConditionalRandomField, TimeDistributed
from allennlp.predictors import Predictor
from discourse.dataset_readers import CrfPubmedRCTReader
from discourse.models import DiscourseCrfClassifier  # noqa:F401
from nltk import sent_tokenize
from torch.nn import Linear

from .predictors import ClaimCrfPredictor
from .utils import MODEL_PATH, WEIGHT_PATH

nltk.download('punkt')


def load_claim_extraction_model(model_path: str = MODEL_PATH, weight_path: str = WEIGHT_PATH):
    """
    Load the Conditional Random field model using allennlp used by titipat in the repo.

    see: http://github.com/titipata/detecting-scientific-claim
    :param model_path: location of model, can be downloaded offline or link can be given
    :param weight_path: location of model weight, can be downloaded offline or link can be given
    :return: the model using the WEIGHT_PATH specified
    """
    archive = load_archive(model_path)
    predictor = Predictor.from_archive(archive, 'discourse_crf_predictor')
    # NOTE(alpha_darklord): We are creating a CRF model based on how allennlp is creating it
    # , for reference go to: http://github.com/titipata/detecting-scientific-claim
    model = predictor._model
    for param in list(model.parameters()):
        param.requires_grad = False  # not to train weights
    embedding_dim = 300
    num_classes, constraints, include_start_end_transitions = 2, None, False
    model.crf = ConditionalRandomField(num_classes, constraints,
                                       include_start_end_transitions=include_start_end_transitions)
    model.label_projection_layer = TimeDistributed(Linear(2 * embedding_dim, num_classes))
    model.load_state_dict(torch.load(cached_path(weight_path), map_location='cpu'))
    return model


def extract_claims(data: pd.DataFrame(),
                   model_path: str = MODEL_PATH,
                   weight_path: str = WEIGHT_PATH,
                   col_name: str = "sentence"):
    """
    Extract Claims from given columns in a dataset to extract the claim.

    :param file_path: path to input file, which contains sentences
    :param model_path: location of model, can be downloaded offline or link can be given
    :param weight_path: location of model weight, can be downloaded offline or link can be given
    :param col_name: name of column on which claim is to be identified, should not be "sentences
    :return: labels, if a sentence is a claim or not
    """
    model = load_claim_extraction_model(model_path, weight_path)
    # print("MODEL LOADED!!!")  # noqa: T001
    reader = CrfPubmedRCTReader()
    claim_predictor = ClaimCrfPredictor(model, dataset_reader=reader)

    df = data
    if col_name not in df.columns:
        return None
    df_sentence = df.copy()

    # NOTE(alpha_darklord): The function returns a list of labels, whether a particular
    # sentence is a claim or not (0 or 1), best_paths is used to get this label,
    # later we extract sentences which have 1 label and transfer them into a list contained in column "claims"
    df_sentence["sentences"] = df_sentence[col_name]
    df_sentence["sentences"] = df_sentence.sentences.apply(sent_tokenize)
    df_sentence['pred'] = df_sentence.sentences.apply(lambda x: claim_predictor.predict_json({'sentences': x}))
    df_sentence['best_paths'] = df_sentence.pred.apply(
        lambda x:
        model.crf.viterbi_tags(torch.FloatTensor(x['logits']).unsqueeze(0),
                               torch.LongTensor(x['mask']).unsqueeze(0)))
    df_sentence['p_claims'] = df_sentence['best_paths'].apply(lambda x: 100 * np.array(x[0][0]))
    df_sentence['claims'] = df_sentence.apply(lambda x: np.extract(x['p_claims'], x['sentences']), axis=1)
    df_claims = df_sentence[~ (df_sentence.claims.str.len() == 0)]
    del df_sentence
    # NOTE(alpha_darklord): This converts a list present inside a column to different rows
    # containing individual items
    df_updated = df_claims[[col_name, "claims"]].explode("claims")
    df_updated["claim_flag"] = 1
    df_merged = df.merge(df_updated, on=[col_name], how="left")
    df_merged["claim_flag"].fillna(0, inplace=True)
    return df_merged
