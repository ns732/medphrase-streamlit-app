# MedPhrase - Adverse Drug Reaction Extraction and Concept Normalisation

MedPhrase is a research prototype that analyses patient-written text to
identify possible adverse drug reaction (ADR) mentions and suggest
standardised SNOMED CT concepts.

The current application combines a fine-tuned **BioBERT**
token-classification model for ADR phrase extraction with
**character-level TF-IDF and cosine similarity** for concept
normalisation.

> **Research prototype only:** MedPhrase is not a diagnostic system or
> clinically validated decision-support tool. Suggestions require human
> review. Similarity scores are ranking signals, not calibrated
> probabilities. Do not enter identifiable patient information.

## How it works

1.  The user enters a patient-style description of a possible side
    effect.
2.  Fine-tuned BioBERT predicts token labels (`O`, `B-ADR`, `I-ADR`) and
    extracts possible ADR spans.
3.  Each extracted phrase is compared with medical aliases associated
    with SNOMED CT concepts. Character-level TF-IDF represents text
    using character n-grams, and cosine similarity ranks candidate
    concepts.
4.  The application displays the extracted phrase and ranked SNOMED CT
    suggestions for human review.

``` text
Patient-written text
        |
        v
Fine-tuned BioBERT (ADR extraction)
        |
        v
Extracted ADR phrase
        |
        v
Character-level TF-IDF
        |
        v
Cosine similarity against medical aliases
        |
        v
Ranked SNOMED CT concept suggestions
        |
        v
Human review
```

## Models and methods

### Used in the current application

-   **BioBERT:** `dmis-lab/biobert-base-cased-v1.2`, fine-tuned for
    token-level ADR extraction.
-   **Character-level TF-IDF:** Represents extracted phrases and medical
    aliases using character n-grams.
-   **Cosine similarity:** Scores textual similarity between the
    extracted phrase and candidate aliases. Candidates are ranked using
    these scores.

### Explored experimentally

SapBERT and combined ranking approaches were evaluated during
experimentation. They are **not part of the current application's
deployed inference pipeline**. The current app should be described as
**BioBERT + character-level TF-IDF**, not BioBERT + SapBERT.

## Dataset

The project uses **CADEC (Corpus of Adverse Drug Event Annotations)**, a
corpus of patient-authored adverse drug event reports annotated with
medical mentions and terminology mappings.

Dataset collection: https://data.csiro.au/collection/csiro:10948

Karimi, S., Metke-Jimenez, A., Kemp, M. and Wang, C. (2015) 'CADEC: A
corpus of adverse drug event annotations', *Journal of Biomedical
Informatics*, 55, pp. 73--81. https://doi.org/10.1016/j.jbi.2015.03.010

Annotations were checked by comparing annotated character spans with the
original patient text. Suspected mismatches were inspected, and
confirmed character-offset corrections were applied to a separate
working copy.

**Dataset availability:** Do not assume raw CADEC data is included in
this GitHub repository. Obtain it from the official CSIRO collection and
follow its access, licence and redistribution conditions. Do not commit
raw patient posts or annotation files unless redistribution is
explicitly permitted.

## Data splitting and evaluation

The normalisation benchmark contains 8,410 examples, split by document
ID:

  Split           Examples
  ------------ -----------
  Training           5,879
  Validation         1,175
  Test               1,356
  **Total**      **8,410**

The document-level split helps reduce leakage between training and
evaluation data.

Metrics: - **Accuracy@1:** Whether the correct concept is ranked
first. - **Recall@5:** Whether the correct concept appears among the
first five suggestions. - **Mean Reciprocal Rank (MRR):** How highly the
correct concept is ranked on average.

### Selected normalisation results

On a held-out subset of 428 unseen phrases whose target concepts were
represented in the candidate vocabulary:

  Method                    Accuracy@1   Recall@5        MRR
  ----------------------- ------------ ---------- ----------
  SapBERT only                  18.69%     35.28%   0.260701
  Character TF-IDF only         48.83%     82.24%   0.636570
  Hybrid rank fusion            48.83%     82.01%   0.638091

In a separate eligible test audit of 1,275 examples, character-level
TF-IDF achieved **82.27% Accuracy@1** and **93.96% Recall@5**. These
results come from different evaluation cohorts and must not be treated
as directly interchangeable.

The normalisation metrics evaluate supplied phrases against known gold
concepts. They are not end-to-end scores for the complete pipeline
because they do not include possible BioBERT extraction errors.

## Application architecture

The prototype uses **Streamlit** for its user interface and Python for
backend inference. The frontend collects text and displays extracted ADR
phrases and ranked concept suggestions. The backend loads the model and
normalisation assets, runs BioBERT, and passes each extracted phrase to
the TF-IDF normaliser.

## Repository structure

Typical project files include:

``` text
medphrase/
├── app.py
├── requirements.txt
├── README.md
├── .streamlit/
│   └── config.toml
└── assets/
    └── # Local model / normalisation assets, if configured
```

Large model weights, access tokens, private normalisation assets and raw
CADEC data should not be committed to a public GitHub repository. Keep
secrets in Streamlit's Secrets settings or an untracked local secrets
file. Never commit Hugging Face tokens or other credentials.

## Run locally

Install the dependencies listed in `requirements.txt` and ensure the
model and normalisation assets expected by `app.py` are available.

``` bash
python -m venv .venv

# Windows
.venv\\Scripts\\activate

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

If the application downloads model or normalisation assets from a
private model repository, configure the repository identifier and access
token using the exact secret names expected by `app.py`. Do not put real
tokens in this README or in source control.

## Deploy with Streamlit Community Cloud

1.  Push the application code, `requirements.txt`, README and required
    non-sensitive configuration to GitHub.
2.  Keep model weights, private assets, raw CADEC files and credentials
    out of the public repository.
3.  Create a Streamlit Community Cloud app connected to the GitHub
    repository.
4.  Set `app.py` as the application entry point.
5.  Add any required model-repository identifiers and access tokens in
    **Settings → Secrets**, using the names expected by the application.
6.  Deploy and test the app, including a cold start to confirm required
    assets load correctly.

Deployment guide:
https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy

## Limitations

-   Patient-written expressions can be informal, misspelled, abbreviated
    or ambiguous.
-   Character-level TF-IDF captures character-pattern overlap; it does
    not provide full semantic understanding.
-   A high similarity score is not a probability that a concept is
    clinically correct.
-   BioBERT extraction errors can affect the normalisation results that
    follow.
-   Reported normalisation metrics do not constitute a complete
    end-to-end clinical evaluation.
-   MedPhrase is a research prototype and must not be used to diagnose,
    treat or make clinical decisions.

## References

-   Karimi, S., Metke-Jimenez, A., Kemp, M. and Wang, C. (2015) 'CADEC:
    A corpus of adverse drug event annotations', *Journal of Biomedical
    Informatics*, 55, pp. 73--81.
    https://doi.org/10.1016/j.jbi.2015.03.010
-   Lee, J. et al. (2020) 'BioBERT: a pre-trained biomedical language
    representation model for biomedical text mining', *Bioinformatics*,
    36(4), pp. 1234--1240. https://doi.org/10.1093/bioinformatics/btz682
-   Liu, F., Shareghi, E., Meng, Z., Basaldella, M. and
    Collier, N. (2021) 'Self-alignment pretraining for biomedical entity
    representations', *Proceedings of NAACL-HLT 2021*, pp. 4228--4238.
    https://doi.org/10.18653/v1/2021.naacl-main.334
