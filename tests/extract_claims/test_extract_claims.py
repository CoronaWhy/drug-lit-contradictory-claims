"""Testing claim extraction functions."""


import unittest

import pandas as pd
from contradictory_claims import extract_claims
# from extract_claims import extract_claims, load_claim_extraction_model

from .constants import MODEL_PATH, WEIGHT_PATH


class TestExtractClaims(unittest.TestCase):
    """Test for loading the model and returning claims."""

    def test_load_claim_extraction_model(self) -> None:
        """Loads the model to extract claims."""
        self.model = extract_claims.load_claim_extraction_model(model_path=MODEL_PATH, weight_path=WEIGHT_PATH)
        self.assertIsNotNone(self.model)

    def test_extract_claims(self):
        """Check if it indeed returns claims."""
        df_test = pd.DataFrame({
            "text": [("the predominant pathological features of covid-19 infection largely mimic "
                      "those previously reported for sars-cov-1 infection. they include dry cough, persistent "
                      "fever, progressive dyspnea, and in some cases acute exacerbation of lung function with "
                      "bilateral pneumonia (30). major lung lesions include several pathological signs, such "
                      "as diffuse alveolar damage, inflammatory exudation in the alveoli and interstitial "
                      "tissue, hyperplasia of fibrous tissue, and eventually lung fibrosis (43) (44) (45) . it "
                      "has been shown by fluorescence in situ hybridization technique that sars-cov-1 rna "
                      "locates to the alveolar pneumocytes and alveolar space (46, 47) . considering all these "
                      "facts, it is not surprising that most histopathological analyses have been focused on "
                      "distal parts of the respiratory airways, while the regions other than the alveolus have "
                      "been less systematically studied. the copyright holder for this preprint (which was not "
                      "peer-reviewed) is the . https://doi.org/10.1101/2020.04.13.038752 doi: biorxiv preprint "
                      "intermediate cells between goblet, ciliated, and club cells. if sars-coronaviruses "
                      "predominantly attack these cells, locating along the airway segments including the "
                      "trachea, bronchi, and bronchioles until the last segment that is the respiratory "
                      "bronchioles, it would be obvious that physiological protective mechanisms are severely "
                      "affected. defective mucosal protection and inefficient removal of pathogens due to "
                      "viral infection may contribute to onset of severe bilateral pneumonia that is common "
                      "for sars-diseases (51) . this pathogenic mechanism is supported by previous findings, "
                      "showing that early disease is manifested as a bronchiolar disease with respiratory "
                      "epithelial cell necrosis, loss of cilia, squamous cell metaplasia, and intrabronchiolar "
                      "fibrin deposits when we initiated the present study, we hypothesized that understanding "
                      "better the transcriptional regulation of the ace2 gene might help to explain the "
                      "peculiar distribution pattern of ace2 in tissues. since upregulation of ace2 would "
                      "reflect an increased number of sars-coronavirus receptors on cell surfaces, it could "
                      "possibly help us to understand the mechanisms why certain patients (males more than "
                      "females, old more than young, smokers more than non-smokers) are more susceptible for "
                      "the most detrimental effects of the covid-19 infection. in our study, the signals for "
                      "ace2 mrna in the lung specimens did not vary much in different age groups nor did they "
                      "show significant differences between males and females, which is in line with the "
                      "previous findings (48) . therefore, different expression levels of lung ace2 may not "
                      "explain the variable outcome of the disease concerning age groups and genders. to "
                      "investigate the transcriptional regulation of ace2 gene we made predictions for the "
                      "binding sites of transcription factors within the proximal promoter region of the "
                      "intestine-specific and lungspecific human ace2 transcript promoters. our findings "
                      "introduced several putative binding sites in the ace2 promoter for known transcription "
                      "factors, which showed high levels of coexpression with ace2 in several tissues "
                      "including the ileum, colon, and kidney. the identified transcription factors . cc-by 4.0 "
                      "international license author/funder. it is made available under a the copyright holder "
                      "for this preprint (which was not peer-reviewed) is the . https://doi.org/10.1101/2020.04."
                      "13.038752 doi: biorxiv preprint could represent potential candidate target molecules "
                      "which regulate ace2 expression. two of our predictions, for hnf1a and hnf1b have been "
                      "previously identified experimentally to drive ace2 expression in pancreatic islet cells "
                      "and insulinoma cells, respectively (36). later work by the same group has shown that our "
                      "prediction of foxa binding sites in the ace2 promoter are also likely correct (54) ."
                      "the results suggest that sars-cov infection may target the cell types that are important "
                      "for the protection of airway mucosa and their damage may lead to deterioration of "
                      "epithelial cell function, finally leading to a more severe lung disease with "
                      "accumulation of alveolar exudate and inflammatory cells and lung edema, the signs of "
                      "pneumonia recently described in the lung specimens of two patients with covid-19 "
                      "infection (60) . gene ontology analysis suggested that ace2 is involved in angiogenesis/"
                      "blood vessel morphogenesis processes in addition to its classical function in "
                      "renin-angiotensin system. the copyright holder for this preprint (which was not "
                      "peer-reviewed) is the . https://doi.org/10.1101/2020.04.13.038752 doi: biorxiv preprint "
                      "figure 2 . immunohistochemical localization of ace2 protein in selected human tissues. "
                      "in the duodenum (a), the protein is most strongly localized to the apical plasma "
                      "membrane of absorptive enterocytes (arrows). the goblet cells (arrowheads) show weaker "
                      "apical staining. intracellular staining is confined to the absorptive enterocytes. in "
                      "the kidney (b), ace2 shows prominent apical staining in the epithelial cells of the "
                      "proximal convoluted tubules (arrows) and bowman´s capsule epithelium (arrowheads)."
                      ), "remdesivir inhibits renal fibrosis in obstructed kidneys"]})
        df_final = extract_claims.extract_claims(df_test, col_name="text")
        self.assertIsNotNone(df_final)
        self.assertTrue("claims" in df_final.columns)
        self.assertGreaterEqual(df_final['claim_flag'].sum(), 1)  # check if any semtemce has a claim_flag found
        self.assertGreaterEqual(df_final.shape[0], df_test.shape[0])
