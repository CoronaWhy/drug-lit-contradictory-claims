"""Functions for preparing cord-19 claim pairs for Roam annotation."""

# -*- coding: utf-8 -*-

import os
import re
import ssl

import gensim.downloader as api
import itertools
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import logging
import pandas as pd
from fse import SplitIndexedList
# from fse.models.average import FAST_VERSION, MAX_WORDS_IN_BATCH
from fse.models import uSIF
from textblob import TextBlob


INPUT_CLAIMS_FILE = "/Users/dnsosa/Downloads/processed_claims_150820.csv"
NON_CLAIMS_FILE = "/Users/dnsosa/Downloads/RoamPairs/non_claims_file.txt"
OUTPUT_CLAIMS_FILE = "/Users/dnsosa/Downloads/RoamPairs/pairs_by_topic_09_18_20_abs_conc_no_nonclaims.xlsx"

def polarity_tb_score(text: str) -> float:
    """
    Calculate polarity of a sentence using TextBlob.
    :param text: input sentence
    :return: polarity value of sentence. Ranges from -1 (negative) to 1 (positive).
    """
    return TextBlob(text).sentiment.polarity


def polarity_v_score(text: str) -> float:
    """
    Calculate polarity of a sentence using Vader.
    :param text: input sentence
    :return: polarity value of sentence. Ranges from -1 (negative) to 1 (positive).
    """
    nltk.download('vader_lexicon')
    vader = SentimentIntensityAnalyzer()
    return vader.polarity_scores(text)['compound']


def generate_claims_for_annotators(input_claims_file: str,
                                   non_claim_file: str,
                                   output_claims_file: str,
                                   min_polarity: float = 0.0,
                                   k: int = 8,
                                   only_we: bool = False):
    """
    Generate claims for Roam annotators.
    :param input_claims_file: input file containing all claims
    :param non_claim_file: file containing some claims manually annotated as false positives
    :param output_claims_file: location of file to be output
    :param min_polarity: absolute threshold above which claim will be called as positive or negative
    :param k: number of positive or negative claims to be retrieved for each (drug, topic) condition
    :param only_we: Boolean--if True, only include claims that contain "we, this study, these results, etc."
    """

    # Load GLOVE, which is necessary for uSIF embeddings
    if not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None):
        ssl._create_default_https_context = ssl._create_unverified_context

    logging.basicConfig(format='%(asctime)s : %(threadName)s : %(levelname)s : %(message)s', level=logging.INFO)
    glove = api.load("glove-wiki-gigaword-300")

    # print(MAX_WORDS_IN_BATCH)
    # print(FAST_VERSION)  # uh oh...

    all_claims_df = pd.read_csv(input_claims_file)
    all_claims_df = all_claims_df.drop(all_claims_df.columns[0:2], axis=1)
    all_claims_df["polarity_vader"] = all_claims_df.apply(lambda row: polarity_v_score(row['claims']), axis = 1)

    print(f"Found {len(all_claims_df)} total claims")
    n_conc = len(all_claims_df[all_claims_df.section == "Conclusion"])

    all_claims = all_claims_df.claims.values
    all_claims_abs = all_claims_df[all_claims_df.section == "Abstract"].claims.values
    all_claims_abs_conc = all_claims_df[(all_claims_df.section == "Abstract") | (all_claims_df.section == "Conclusion")].claims.values
    we_claims = [claim for claim in all_claims if re.match("we |our |this study|this result|these data|these results", claim)]
    hcq_claims = [claim for claim in all_claims if "hydroxychloroquine" in claim]
    hcq_we_claims = [claim for claim in we_claims if "hydroxychloroquine" in claim]
    print(f"{len(all_claims_abs)} claims in 'Abstract' section")
    print(f"{n_conc} claims in 'Conclusion' section")
    print(f"{len(all_claims_abs_conc)} claims in 'Conclusion' or 'Abstract' section")
    print(f"{len(we_claims)} claims contain 'we, our, this study, this result, these data, these results'")
    print(f"{len(hcq_claims)} claims contain the word 'hydroxychloroquine")
    print(f"{len(hcq_we_claims)} of the 'we' claims contain the word 'hydroxychloroquine")

    # Read in and create list of non-claims
    with open(non_claim_file) as f:
        non_claim_list = [line.rstrip() for line in f]

    print(f"{len(non_claim_list)} non-claims loaded")

    topic_list = ["mortality", "effective treatment", "toxicity"]
    drug_list = ["hydroxychloroquine", " chloroquine", "tocilizumab", "remdesivir", "vitamin d", "lopinavir", "dexamethasone"]

    # Remove non-claims from the claims list
    claims_list = list(set(all_claims_abs_conc).difference(set(non_claim_list)))
    full_out_df = None

    for drug in drug_list:

        drug_claims = [claim for claim in claims_list if drug in claim]
        drug_we_claims = [claim for claim in we_claims if drug in claim]

        if only_we:
            input_claims = drug_we_claims
        else:
            input_claims = drug_claims

        s = SplitIndexedList(input_claims)
        print(len(s))
        # NOTE, MANY repeats

        # Train the uSIF model
        model = uSIF(glove, workers=2, lang_freq="en")
        model.train(s)

        for topic in topic_list:
            # Retrieve the top claims relevant to topic
            res_list = model.sv.similar_by_sentence(topic.split(), model=model, indexable=s.items, topn=50)
            results = list(zip(*res_list))
            d = {'claim': results[0], 'similarity_to_topic': results[2], 'drug': [drug] * len(results[0]),
                 'topic': [topic] * len(results[0])}
            res_df = pd.DataFrame(data=d).drop_duplicates()
            res_df["polarity_vader"] = res_df.apply(lambda row: polarity_v_score(row['claim']), axis=1)

            # print(f"DRUG: {drug}, TOPIC: {topic}")
            # Retrieve the "negative" and "positive" claims relevant to topic
            res_df2_neg = res_df[res_df.polarity_vader < -min_polarity].iloc[:k]
            # print(res_df2_neg.claim.values)
            res_df2_pos = res_df[res_df.polarity_vader > min_polarity].iloc[:k]
            # print(res_df2_pos.claim.values)

            drug_topic_df = pd.concat([res_df2_neg, res_df2_pos])
            combos = list(zip(*itertools.combinations(drug_topic_df.claim.values, 2)))

            # in case no claims are retrieved?
            if len(combos) == 0:
                continue

            d = {'text1': combos[0],
                 'text2': combos[1],
                 'drug': [drug] * len(combos[0]),
                 'topic': [topic] * len(combos[0])}
            df_combos = pd.DataFrame(data=d)

            if full_out_df is None:
                full_out_df = df_combos
            else:
                full_out_df = pd.concat([full_out_df, df_combos])

    # Join the claims with all the other metadata about claim
    roam_df_p1 = all_claims_df.merge(full_out_df, left_on="claims", right_on="text1")
    roam_df_p2 = roam_df_p1.merge(all_claims_df, left_on="text2", right_on="claims")
    roam_final_df = roam_df_p2[
        ["cord_uid_x", "cord_uid_y", "text1", "text2", "polarity_vader_x", "polarity_vader_y", "drug",
         "topic"]].sort_values(["drug", "topic"])  # NEED TO CHECK IF SOMETHING HAPPENED?
    roam_final_df = roam_final_df.rename(
        columns={"cord_uid_x": "paper1_cord_uid", "cord_uid_y": "paper2_cord_uid"}).drop_duplicates()

    print(f"Resulting DF has {len(roam_final_df)} rows")
    roam_final_df = roam_final_df.drop_duplicates()
    print(f"After dropping duplicates: {len(roam_final_df)} rows")
    roam_final_df = roam_final_df.groupby(["paper1_cord_uid", "paper2_cord_uid", "drug", "topic"]).sample(n=1)
    print(f"After sampling 1 row per cord/drug/topic group to remove duplicates: {len(roam_final_df)} rows")
    roam_final_df = roam_final_df[roam_final_df.text1 != roam_final_df.text2]
    print(f"After dropping rows where text 1 == text 2: {len(roam_final_df)} rows")
    roam_final_df = roam_final_df.groupby(["text1", "text2"]).sample(n=1)
    print(f"After sampling 1 claim pair per text1/text2 group: {len(roam_final_df)} rows")

    # Sample 1000 and send to Excel
    # Note: sample(frac=1) just shuffles everything
    roam_final_df.sample(n=1000).sample(frac=1)[["paper1_cord_uid", "paper2_cord_uid", "text1", "text2"]].to_excel(
        output_claims_file)
