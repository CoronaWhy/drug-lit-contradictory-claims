"""Functions for benchmarking against when finding contradictory claims."""

# -*- coding: utf-8 -*-


from ..data.prepare_claims_for_roam import polarity_tb_score, polarity_v_score


def classify_on_opposed_polarity(text1: str, text2: str, min_polarity: float = 0.0, vader: bool = True) -> int:
    """
    Classify as entail, contradict, neutral based only on polarity. If polarity is opposite = contradiction, if
    polarity is same = entailment, if at least one isn't polar, then neutral.

    :param text1: claim 1
    :param text2: claim 2
    :param min_polarity: minimum absolute polarity to call positive or negative
    :param vader: if True, use Vader for polarity detection; else use TextBlob
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

    if pol1 > min_polarity and pol2 < -min_polarity:
        return -1
    elif (pol1 > min_polarity and pol2 > min_polarity) or (pol1 < -min_polarity and pol2 < -min_polarity):
        return 1
    else:
        return 0
