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
NON_CLAIMS_FILE = "..."
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

    non_claim_list = ['objectives: in this study, we investigate the effect of hydroxychloroquine on the prevention of novel coronavirus disease in cancer patients being treated.', 'exclusion criteria include previous infection with sars cov2 (positive sars-cov-2 pcr or igg serology), pregnancy or lactation, any contraindication to hydroxychloroquine or evidence of unstable or clinically significant systemic disease.', 'following the demonstration of the efficacy of hydroxychloroquine against severe acute respiratory syndrome coronavirus 2 in vitro, many trials started to evaluate its efficacy in clinical settings.', 'our hypothesis is that oral administration of hydroxychloroquine to healthcare professionals can reduce the incidence and prevalence of infection as well as its severity in this group.','thus, we aimed to investigate the electrocardiographic and mortality effects ofchloroquineand hydroxychloroquine in a primary care population.', 'to determine the relative impact of the use of chloroquine and hydroxychloroquine on outcomes important to patients with covid 19.', 'the current use of chloroquine and/or hydroxychloroquine, a drug currently used to treat autoimmune rheumatic diseases, in treating severe acute respiratory syndrome caused by coronavirus 2 (sars-cov-2) or covid-19-infected patients with pneumonia is a matter of intense consideration.', 'future studies should focus on identification of those at highest arrhythmic risk with hydroxychloroquine therapy to more optimally guide arrhythmia surveillance and prevention in patients with covid-19.', 'controlled, double-blind, randomized trial to assess the efficacy and safety of hydroxychloroquine chemoprophylaxis in sars cov2 infection in healthcare personnel in the hospital setting: a structured summary of a study protocol for a randomised controlled trial abstract background: sars-cov-2 infection presents a high transmission in the group of health professionals in spain (12-15% infected).', 'for this reason, our suggestions include: 1. launch remote control systems and faraway learning (telemedicine, social media account) 2. limit the use of gliptin drugs 3. blood glucose must be controlled 4. limit the use of acei drugs 5. reduce unnecessary hospital admissions 6. attention to nutrition 7. regard to the guidelines of the country\'s health-care system in preventing of infection after infection: 1. monitoring the symptoms and rapid referral 2. monitoring the blood glucose 3. monitoring for aki complication 4. monitoring for ards 5. use of hydroxychloroquine 6. reduction of adverse drug reactions 7. attention to nutrition (hydration, protein, and etc.)', 'further clinical studies are needed to verify the combination of hydroxychloroquine and rifn-ω will be effective and safe treatment for cats with fip.', 'covirl-001 -a multicentre, prospective, randomised trial comparing standard of care (soc) alone, soc plus hydroxychloroquine monotherapy or soc plus a combination of hydroxychloroquine and azithromycin in the treatment of non-critical, sars-cov-2 pcrpositive population not requiring immediate resuscitation or ventilation but who have evidence of clinical decline.', 'the first clinical trials on zinc supplementation alone and in combination with other drugs such as chloroquine have been registered (124, 160–162).', 'to determine the relative impact of the use of chloroquine and hydroxychloroquine on outcomes important to patients with covid 19.', 'the current use of chloroquine and/or hydroxychloroquine, a drug currently used to treat autoimmune rheumatic diseases, in treating severe acute respiratory syndrome caused by coronavirus 2 (sars-cov-2) or covid-19-infected patients with pneumonia is a matter of intense consideration.','the worldwide ongoing trials, including those involving the care of patients in our institute [90], will verify whether the hopes raised by chloroquine in the treatment of covid-19 can be confirmed.', 'public interest in hydroxychloroquine and chloroquine was plotted against the cumulative number of active clinical trials evaluating antimalarials as potential covid-19 therapies over time.', 'no established treatment is currently available; however, several therapies, including remdesivir, hydroxychloroquine and chloroquine, and interleukin (il)-6 inhibitors, are being used off-label and evaluated in ongoing clinical trials.', 'well-designed clinical trials (randomized and controlled) with valuable and less as possible subjective eps are urgently needed to clearly establish safety and effectiveness of quinine derivatives like chloroquine as antiviral treatments.', 'there are additional proposed drugs against covid-19, such as drugs approved by fda for the treatment of other pathologies, including ribavirin, penciclovir, nitazoxanide, nafamostat, chloroquine and two well-known drugs having broad spectrum activity i.e.', 'the worldwide ongoing trials, including those involving the care of patients in our institute [90], will verify whether the hopes raised by chloroquine in the treatment of covid-19 can be confirmed.', 'the current use of chloroquine and/or hydroxychloroquine, a drug currently used to treat autoimmune rheumatic diseases, in treating severe acute respiratory syndrome caused by coronavirus 2 (sars-cov-2) or covid-19-infected patients with pneumonia is a matter of intense consideration.', 'the first clinical trials on zinc supplementation alone and in combination with other drugs such as chloroquine have been registered (124, 160–162).', 'the current use of chloroquine and/or hydroxychloroquine, a drug currently used to treat autoimmune rheumatic diseases, in treating severe acute respiratory syndrome caused by coronavirus 2 (sars-cov-2) or covid-19-infected patients with pneumonia is a matter of intense consideration.', 'well-designed clinical trials (randomized and controlled) with valuable and less as possible subjective eps are urgently needed to clearly establish safety and effectiveness of quinine derivatives like chloroquine as antiviral treatments.', 'to determine the relative impact of the use of chloroquine and hydroxychloroquine on outcomes important to patients with covid 19.', 'the review elaborates the mechanism of action, safety (side effects, adverse effects, toxicity) and worldwide clinical trials for chloroquine and hydroxychloroquine to benefit the clinicians, medicinal chemist, pharmacologist actively involved in the management of covid-19 infection.', 'the chloroquine was suggested a potential drug against sars-cov-2 infection due to its in vitro antiviral effects, it is imperative high-quality data from worldwide clinical trials are necessitated for an approved therapy.', 'this case report describes a rare case of colonic perforation in a critically ill, morbidly obese patient with covid-19 pneumonia on empiric tocilizumab therapy. ', 'to date, data about the use of tocilizumab in the treatment of acute lung injury in patients', 'second, there are numerous clinical trials that are aimed to determine if curbing the proinflammatory state produced during covid-19 with drugs such as il-6 inhibitors (e.g., sarilumab or tocilizumab) or anti-gm-csf compounds (e.g., lenzilumab and gimsilumab) will lead to better clinical outcomes by preventing or reversing ards and multi-organ failure.', 'to determine if tocilizumab treatment in patients hospitalized with laboratory confirmed sars-cov-2 infection and subsequent covid-19 disease provides short-term survival benefit.', 'to date, data about the use of tocilizumab in the treatment of acute lung injury in patients', 'roads less traveled might also be considered over time as an alternative to the current therapies being tested in covid-19 clinical trials, such as the combination of il-6 blocking agents (tocilizumab sarilumab) or antiviral therapies (ribavirin, ritonavir-lopinavir, remdesivir, niclosamide), with kinase inhibitors (imatinib, osimertinib, gilteritinib, abemaciclib, afatinib, sunitinib, sorafenib, erlotinib), or the direct combination of kinase inhibitors with each other that target relevant virus-associated proteins and proteins associated with pulmonary health (sunitinib and erlotinib, or afatinib and nintedanib).', 'second, there are numerous clinical trials that are aimed to determine if curbing the proinflammatory state produced during covid-19 with drugs such as il-6 inhibitors (e.g., sarilumab or tocilizumab) or anti-gm-csf compounds (e.g., lenzilumab and gimsilumab) will lead to better clinical outcomes by preventing or reversing ards and multi-organ failure.', 'on acute hypertriglyceridemia secondary to tocilizumab in patients with severe coronavirus disease .', 'our data allow us to conclude that by the end of june we will have results of almost 20 trials involving 40000 patients for hydroxychloroquine and 5 trials with 4500 patients for remdesivir; however, low statistical power is expected from the 9 clinical trials testing the efficacy of favipiravir or the 5 testing tocilizumab, since they will recruit less than 1000 patients each one.', 'the clinical use of tocilizumab can be referred to drug instruction from the us fda for the treatment of crs or \"diagnosis and treatment plan of novel coronavirus pneumonia (seventh trial edition)\" in china.', 'also, potential drugs listed in table 1, such as remdesivir, atazanavir, saquinavir, and formoterol, and tocilizumab can be introduced as treatments for covid-19 if they prove to be effective in animal and clinical studies.', 'further studies are needed to better understand the safety and efficacy of tocilizumab plus standard of care in hospitalized patients with severe covid-19 infection.', 'please cite this article as: hassoun a, thottacherry ed, muklewicz j, aziz q-ul-ain, edwards j, utilizing tocilizumab for the treatment of cytokine release syndrome in effective treatments are under study particularly those modifying a dysregulated host immune response known as cytokine release syndrome (crs).', 'the efficacy of some promising antivirals, convalescent plasma transfusion, and tocilizumab needs to be investigated by ongoing clinical trials.', 'on acute hypertriglyceridemia secondary to tocilizumab in patients with severe coronavirus disease .', 'roads less traveled might also be considered over time as an alternative to the current therapies being tested in covid-19 clinical trials, such as the combination of il-6 blocking agents (tocilizumab sarilumab) or antiviral therapies (ribavirin, ritonavir-lopinavir, remdesivir, niclosamide), with kinase inhibitors (imatinib, osimertinib, gilteritinib, abemaciclib, afatinib, sunitinib, sorafenib, erlotinib), or the direct combination of kinase inhibitors with each other that target relevant virus-associated proteins and proteins associated with pulmonary health (sunitinib and erlotinib, or afatinib and nintedanib).', 'second, there are numerous clinical trials that are aimed to determine if curbing the proinflammatory state produced during covid-19 with drugs such as il-6 inhibitors (e.g., sarilumab or tocilizumab) or anti-gm-csf compounds (e.g., lenzilumab and gimsilumab) will lead to better clinical outcomes by preventing or reversing ards and multi-organ failure.', 'to date, data about the use of tocilizumab in the treatment of acute lung injury in patients', 'our data allow us to conclude that by the end of june we will have results of almost 20 trials involving 40000 patients for hydroxychloroquine and 5 trials with 4500 patients for remdesivir; however, low statistical power is expected from the 9 clinical trials testing the efficacy of favipiravir or the 5 testing tocilizumab, since they will recruit less than 1000 patients each one.', 'randomized phase iii trials are currently evaluating the efficacy of anti-il-6-directed agents, including tocilizumab and sarilumab, as well as the jak/stat inhibitor ruxolitinib, and will provide definitive data regarding the use of these agents in patients with covid-19.', 'research is needed to define diagnostic criteria for cytokine storm associated with covid-19, establish the clinical efficacy of tocilizumab in placebo-controlled trials, and further describe which covid-19 populations may derive clinical benefit from tocilizumab.',  'as exploratory studies have suggested that interleukin-6 (il-6) levels are elevated in cases of complicated covid-19 and that the anti-il-6 biologic tocilizumab may be beneficial, we undertook a systematic review and meta-analysis to assess the evidence in this field.', 'the efficacy of some promising antivirals, convalescent plasma transfusion, and tocilizumab needs to be investigated by ongoing clinical trials.', 'to determine if tocilizumab treatment in patients hospitalized with laboratory confirmed sars-cov-2 infection and subsequent covid-19 disease provides short-term survival benefit.', 'our data allow us to conclude that by the end of june we will have results of almost 20 trials involving 40000 patients for hydroxychloroquine and 5 trials with 4500 patients for remdesivir; however, low statistical power is expected from the 9 clinical trials testing the efficacy of favipiravir or the 5 testing tocilizumab, since they will recruit less than 1000 patients each one.', 'roads less traveled might also be considered over time as an alternative to the current therapies being tested in covid-19 clinical trials, such as the combination of il-6 blocking agents (tocilizumab sarilumab) or antiviral therapies (ribavirin, ritonavir-lopinavir, remdesivir, niclosamide), with kinase inhibitors (imatinib, osimertinib, gilteritinib, abemaciclib, afatinib, sunitinib, sorafenib, erlotinib), or the direct combination of kinase inhibitors with each other that target relevant virus-associated proteins and proteins associated with pulmonary health (sunitinib and erlotinib, or afatinib and nintedanib).', 'also, potential drugs listed in table 1, such as remdesivir, atazanavir, saquinavir, and formoterol, and tocilizumab can be introduced as treatments for covid-19 if they prove to be effective in animal and clinical studies.', 'during 2020, a huge amount of clinical trials are expected to be completed: 41 trials (60,366 participants) using hydroxychloroquine, 20 trials (1,588 participants) using convalescent\'s plasma, 18 trials (6,830 participants) using chloroquine, 12 trials (9,938 participants using lopinavir/ritonavir, 11 trials (1,250 participants) using favipiravir, 10 trials ( 2,175 participants) using tocilizumab and 6 trials (13,540 participants) using remdesivir.', 'in addition, remdesivir and arbidol, which are potential drug treatments for covid-19, require further research on their safety during pregnancy.', 'they conclude that further studies are needed to (1) better explore possible associations between vitamin d deficiency and covid-19 morbidity and lethality, and (2) assess if compensating such deficiency could avoid or mitigate the worst manifestations of covid-19.', 'n/a fax: +90 322 458 88 54 inhibition of the raas, weight loss, vitamin d supplementation, management of osa as well as prevention of sarcopenia/frailty.', 'this raises the question of whether an inadequate vitamin d supply has an influence on the progression and severity of covid-19 disease.', 'to assess the overall effect of vitamin d supplementation on risk of acute respiratory infection (ari), and to identify factors modifying this effect.', 'finally, do vitamin d levels and bmi correlate independently or as a composite with inflammation and infection in critically ill children, and if so would, supplementation early in disease change outcome, or even more important would targeted vitamin d supplementation for this group reduce the incidence of inflammatory disease in general.', 'moreover, the vitamin d deficiency in elderly men may be worthy of further study regarding the epidemiological aspects of this different susceptibility and lethality between sexes.', 'our arguments respond to an article, published in italy, that describes the high prevalence of hypovitaminosis d in older italian women and raises the possible preventive and therapeutic role of optimal vitamin d levels.', 'nevertheless, while awaiting more robust data, clinicians should treat patients with vitamin d deficiency irrespective of whether or not it has a link with respiratory infections.', 'it should be advisable to perform dedicated studies about vitamin d levels in covid-19 patients with different degrees of disease severity.', 'however, prospective clinical studies are required to address this speculation and overcome the obstacles in our current understanding of vitamin d role as an adjuvant therapy in patients with covid-19.', 'they conclude that further studies are needed to (1) better explore possible associations between vitamin d deficiency and covid-19 morbidity and lethality, and (2) assess if compensating such deficiency could avoid or mitigate the worst manifestations of covid-19.', 'this raises the question of whether an inadequate vitamin d supply has an influence on the progression and severity of covid-19 disease.', 'n/a fax: +90 322 458 88 54 inhibition of the raas, weight loss, vitamin d supplementation, management of osa as well as prevention of sarcopenia/frailty.', 'yet a direct demonstration that vitamin d deficiency is associated with covid-19 fatalities has remained elusive.', 'there is some data that vitamin d may have protective effect, so authors decided to analyze european country-wide data to determine if vitamin d levels are associated with covid-19 population death rate.', 'our arguments respond to an article, published in italy, that describes the high prevalence of hypovitaminosis d in older italian women and raises the possible preventive and therapeutic role of optimal vitamin d levels.', 'a trial investigating the combined use of zinc, vitamin c and vitamin d would seem to be a rational option given their depleted levels in patients with sepsis and ards and their potential as inhibitors of nf-κb.', 'it should be advisable to perform dedicated studies about vitamin d levels in covid-19 patients with different degrees of disease severity.', 'n/a fax: +90 322 458 88 54 inhibition of the raas, weight loss, vitamin d supplementation, management of osa as well as prevention of sarcopenia/frailty.', 'they conclude that further studies are needed to (1) better explore possible associations between vitamin d deficiency and covid-19 morbidity and lethality, and (2) assess if compensating such deficiency could avoid or mitigate the worst manifestations of covid-19.', 'finally, do vitamin d levels and bmi correlate independently or as a composite with inflammation and infection in critically ill children, and if so would, supplementation early in disease change outcome, or even more important would targeted vitamin d supplementation for this group reduce the incidence of inflammatory disease in general.', 'medrxiv preprint 2 key points • question: does vitamin d deficiency predispose to severity of sars-cov-2 infection?', 'moreover, the vitamin d deficiency in elderly men may be worthy of further study regarding the epidemiological aspects of this different susceptibility and lethality between sexes.', 'it concludes with some recommendations regarding supplementation of vitamin d in patients with covid-19.', 'objectives: vitamin d deficiency (vdd) has been proposed to play a role in coronavirus disease 2019 pathophysiology.', 'lopinavir and ritonavir are both hiv protease inhibitors that suppress the cleavage of a polyprotein into multiple functional proteins.', 'the most dangerous pddis were interaction of lopinavir/ritonavir or atazanavir with clopidogrel, prasugrel, and new oral anticoagulants (noacs).', '3. to study the energetic binding affinity of covid-19 mpro with each inhibitor of ritonavir, lopinavir, azithromycin, hydroxychloroquine, n3, ribavirin and new inhibitors based on free energy calculations.', 'during 2020, a huge amount of clinical trials are expected to be completed: 41 trials (60,366 participants) using hydroxychloroquine, 20 trials (1,588 participants) using convalescent\'s plasma, 18 trials (6,830 participants) using chloroquine, 12 trials (9,938 participants using lopinavir/ritonavir, 11 trials (1,250 participants) using favipiravir, 10 trials ( 2,175 participants) using tocilizumab and 6 trials (13,540 participants) using remdesivir.', 'we designed a trial to evaluate the effectiveness of early intravenous dexamethasone administration on the number of days alive and free of mechanical ventilation within 28 days after randomization in adult patients with moderate or severe ards due to confirmed or probable covid-19.', 'decreasing the dose of dexamethasone to 20 mg and giving bortezomib subcutaneously once a week is recommended.', 'our purpose was to minimize dexamethasone exposure during antiemetic prophylaxis for systemic therapy for solid tumors during the covid-19 pandemic, while maintaining control of nausea and emesis.', 'decreasing the dose of dexamethasone to 20 mg and giving bortezomib subcutaneously once a week is recommended.', 'is the local principal investigator of the currently conducted covacta-trial (a 43 study to evaluate the safety and efficacy of tocilizumab in patients with severe covid-19 44']
    ###non_claim_list = READ IN NON_CLAIM_FILE

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
    # Note: sample(frac=1) just shuffles
    roam_final_df.sample(n=1000).sample(frac=1)[["paper1_cord_uid", "paper2_cord_uid", "text1", "text2"]].to_excel(
        output_claims_file)
