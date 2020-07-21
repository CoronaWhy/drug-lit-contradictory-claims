"""Function to extract Claims"""

import pandas as pd
import os

from tqdm import tqdm
import torch
from torch.nn import ModuleList, Linear
import torch.nn.functional as F
from nltk import word_tokenize, sent_tokenize

from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor
from allennlp.common.file_utils import cached_path
from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.modules import Seq2VecEncoder, TimeDistributed, TextFieldEmbedder, ConditionalRandomField, FeedForward

from .utils import read_json
from .utils import ClaimAnnotationReaderJSON, CrfPubmedRCTReader
from .utils import DiscourseClassifierPredictor
from .utils import ClaimCrfPredictor



def load_model(MODEL_PATH = "https://s3-us-west-2.amazonaws.com/pubmed-rct/model_crf.tar.gz", WEIGHT_PATH  = "https://s3-us-west-2.amazonaws.com/pubmed-rct/model_crf_tf.th"):
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
    return model





def extract_claims(file_path: str, MODEL_PATH = "https://s3-us-west-2.amazonaws.com/pubmed-rct/model_crf.tar.gz", WEIGHT_PATH="https://s3-us-west-2.amazonaws.com/pubmed-rct/model_crf_tf.th", col_name = "sentence"):
    """
    Extract Claims from given columns in a dataset to extract the claim
    
    :param file_path: path to input file, which contains sentences, file is expected to contain columns cord_uid, sentence, csv file
    :param WEIGHT_PATH: location to path where the model weights are kept, default path is specified of the place where model repo is kept
    :param col_name: name of column on which claim is to be identified, should not be "sentences
    :return: labels, if a sentence is a claim or not
    """

    model=load_model(MODEL_PATH, WEIGHT_PATH)
    reader = CrfPubmedRCTReader()
    claim_predictor = ClaimCrfPredictor(model, dataset_reader=reader)

    df=pd.read_csv(file_path)
    assert col_name in df.columns, f"No column named {col_name}"
    df_sentence=df[[col_name]]
    df_sentence["sentences"]=df_sentence[col_name]
    df_sentence["sentences"] = df_sentence.sentences.apply(sent_tokenize)
    df_sentence['pred'] = df_sentence.sentences.apply(lambda x:claim_predictor.predict_json({'sentences': x}))
    df_sentence['best_paths'] = df_sentence.pred.apply(lambda x: model.crf.viterbi_tags(torch.FloatTensor(x['logits']).unsqueeze(0), 
                                    torch.LongTensor(x['mask']).unsqueeze(0)))
    df_sentence['p_claims']=df_sentence['best_paths'].apply(lambda x:100*np.array(x[0][0]))
    df_sentence['claims']=df_sentence.apply(lambda x: np.extract(x['p_claims'],x['sentences']),axis=1)
    df_claims=df_sentence[~(df_sentence.claims.str.len()==0)]
    del df_claims
    df_updated=df_claims[["claims"]].explode("claims")
















