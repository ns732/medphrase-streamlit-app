# MedPhrase — Patient Side-Effect Analyzer

A polished Streamlit research demo for the MAAI 902 NLP Solution Design project, **“Beyond Medical Terminology: Can NLP Understand How Patients Describe Side Effects?”** MedPhrase links BioBERT adverse-drug-reaction (ADR) span extraction to SapBERT ranking over a candidate SNOMED CT concept vocabulary.

> **Research prototype only.** This is not a clinical decision-support system or diagnostic tool. Similarity values are ranking signals, not calibrated confidence scores. Predictions can be wrong and require review.

## Interface

The interface uses a dark navy and teal theme: deep navy backgrounds (`#061A2B`), dark blue panels (`#08283B`), bright teal accents (`#10BFC0`), and soft white text (`#EAF5F7`). The Streamlit theme is configured in `.streamlit/config.toml`, with complementary custom CSS in `app.py`. A custom MedPhrase monogram is supplied as `assets/medphrase_logo.png`; the app also supports placing the same image as `medphrase_logo.png` in the repository root for convenient GitHub uploads.


- Patient-style text input with fictional example prompts
- BioBERT-extracted phrases and character offsets
- SapBERT top-five concept ranking with SNOMED CT IDs
- Three selectable SapBERT input strategies: exact extracted phrase (original baseline), phrase with nearby context, or the sentence containing the phrase. These are input strategies for the same SapBERT model, not different models. The original phrase-only setting remains the default until context strategies are validated.
- CSV download of ranked results, including the normalization input used
- The application does not deliberately write submitted text to a database or application-managed log. Hosting providers may have their own operational logs; use fictional examples and do not enter identifiable patient information.


## SapBERT input-strategy experiment

The interface offers three **input strategies for the same SapBERT model**: extracted phrase only (default), phrase plus up to three neighboring whitespace-delimited tokens on each side within the sentence, and the full sentence containing the phrase. BioBERT extraction is unchanged across strategies. The interface displays the exact text sent to SapBERT and includes it in the CSV export.

In an exploratory validation comparison on 229 examples from the *new phrase + known concept* group, phrase-only ranked best overall (Accuracy@1 0.5371, Recall@5 0.8035, MRR 0.6580); phrase + nearby words scored 0.2751, 0.5284, 0.3963; full sentence scored 0.1659, 0.3755, 0.2667. For the illustrative phrase “waking up,” providing the sentence “I keep waking up during the night.” ranked “Frequent night waking” first, whereas phrase-only ranked “Always sleepy” first. This single-example improvement does not outweigh the lower aggregate validation metrics. Accordingly, phrase-only remains the recommended default, and both context strategies are marked experimental. These validation figures are exploratory and must not replace the original held-out test results.

## Optional voice features

- **Patient speech to text:** The app can record a short statement with Streamlit's microphone widget and transcribe it using the `SpeechRecognition` package's Google Web Speech recognition endpoint. This requires an internet connection and may be rate-limited or unavailable. English variants are offered because the ADR model and benchmark are English-language. The user must check a consent box before transcription is submitted to the external recognition service. The recording is also sent to the Streamlit-hosted app as part of its microphone widget. Use fictional, non-identifying examples only; do not record real patient information.
- **Optional read-aloud:** After an analysis, a doctor can expand the read-aloud section and choose to play either the patient statement or a cautious summary of model suggestions. Playback does not start automatically. The browser/device's built-in `SpeechSynthesis` voice is used; support and voice quality depend on the browser/device.
- Speech transcription is a convenience feature, not a validated medical transcription service. Review the recognized text before running the NLP pipeline. The model is not a clinical decision-support system.

## Project files

```text
app.py                         # Streamlit interface + inference pipeline
requirements.txt               # Python dependencies
.streamlit/config.toml         # Theme settings
assets/medphrase_logo.png      # Custom MedPhrase monogram
assets/                        # Optional local normalization assets (not public in GitHub)
colab_export_assets.py         # Run in the existing Colab notebook to export model assets
```

## 1. Export assets from the existing Colab notebook

In the same Colab runtime that contains `trainer_f1`, `MODEL_NAME`, `candidate_adr_table`, and `concept_embeddings_adr`, open `colab_export_assets.py` from this package, copy its entire contents into a **new cell in your existing project notebook**, and run that cell. Do not run it as a separate Python script: it needs the variables already in your notebook memory. It saves the fine-tuned BioBERT model and fast tokenizer, candidate concept table, and SapBERT concept embeddings to `/content/medphrase_streamlit_export.zip`.

Keep `candidate_adr_table.csv` and `concept_embeddings_adr.npy` together and preserve their row order. The app assumes the embeddings were produced using **SapBERT CLS pooling**, as in the original experiment.

If you already ran the earlier exporter, you do not need to retrain or export again just because the project name changed. You can rename your existing export ZIP to `medphrase_streamlit_export.zip`; its contents are compatible with the updated app.

## 2. Configure the fine-tuned BioBERT model for hosting

The fine-tuned BioBERT weights are large and should not be committed to ordinary GitHub history. Create a model repository on Hugging Face Hub and upload the contents of the exported `biobert/` directory there. Review the CADEC dataset license and institutional requirements before making model artifacts public. A **private model repository** is the safer default if you are unsure whether derived model weights may be redistributed.

For a private Hugging Face model repo, create a read-only Hugging Face access token. In Streamlit Community Cloud, open the deployed app's **Settings → Secrets** and add:

```toml
BIOBERT_REPO_ID = "your-huggingface-username/your-finetuned-biobert-repo"
HF_TOKEN = "hf_your_read_token_here"
```

Do not place the real token in `app.py`, GitHub, screenshots, or this README. If using a public model repository, `HF_TOKEN` can be omitted, though redistribution/licensing still needs to be checked. For local use, the app also supports the model at `assets/biobert/`.

## 3. Upload the normalization assets to the private Hugging Face repository

For deployment, upload these two files from the Colab export to the **root** of the same private Hugging Face model repository (the same place you uploaded the four BioBERT files):

- `candidate_adr_table.csv`
- `concept_embeddings_adr.npy`

The app downloads them from that private repository using the configured read token when local copies are not present. This allows the GitHub app repository to contain code without publicly redistributing the candidate vocabulary or embeddings. Keep `BIOBERT_REPO_ID` pointed to the repository that contains all six files.

The candidate table and embedding matrix must come from the same export and stay in exactly the same row order. Do not upload CADEC raw patient texts or annotations. Review CADEC, SNOMED CT, and institutional terms before redistributing derived assets or model weights; keeping the Hugging Face repository private does not replace that review.

## 4. Test locally (optional)

Use Python 3.11 or 3.12 in a clean environment where PyTorch and Transformers are supported. From this folder:

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

If using local BioBERT files, extract the exported `biobert/` directory to `assets/biobert/`. If using a private Hub repo locally, set `BIOBERT_REPO_ID` and `HF_TOKEN` environment variables or put them in an untracked `.streamlit/secrets.toml` file.

## 5. Deploy to Streamlit Community Cloud

1. Create a GitHub repository for the app code. Make sure the required license permits public sharing of the code and any included terminology assets.
2. Add `app.py`, `requirements.txt`, `.streamlit/config.toml`, `README.md`, and `.gitignore`. Add the small `medphrase_logo.png` file to the repository root (or to `assets/medphrase_logo.png`) so the header can display it. Do not add the model files or normalization assets to public GitHub.
3. Do not commit `assets/biobert/`, raw CADEC data, `candidate_adr_table.csv`, `concept_embeddings_adr.npy`, or access tokens.
4. Go to [Streamlit Community Cloud](https://share.streamlit.io/) and sign in with GitHub.
5. Select **Create app**, choose your repository and branch, and set the main file path to `app.py`.
6. Add `BIOBERT_REPO_ID` and `HF_TOKEN` in the app's **Settings → Secrets**. Set `BIOBERT_REPO_ID` to the Hugging Face repository that contains the four BioBERT files plus both normalization files.
7. Deploy and inspect the app logs if the model download or resource limits cause a problem.

Streamlit Community Cloud provides a fixed `streamlit.app` URL, but apps sleep after inactivity and resource quotas can vary. Loading both transformer models can be memory-intensive on the free tier. Test a cold start and a few requests before using the URL for the assignment demonstration. Streamlit documents its app deployment and resource constraints here: [deployment](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy), [resource limits](https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app).

## Reproducibility notes

- BioBERT extraction uses the three-class label mapping `O`, `B-ADR`, `I-ADR` and overlapping token windows for long text.
- SapBERT concept embeddings use CLS pooling, consistent with the original Colab embedding check.
- When the `assets/` directory has no local normalization files, the app fetches them from the same private Hugging Face repository used for BioBERT.
- The candidate concept vocabulary is restricted to concepts built from the training split in the experiment.
- Original evaluation figures should be reported from the Colab notebook outputs, not recalculated from demo examples.
- The demo uses a general-purpose research pipeline; it has not been clinically validated.

## Experimental context-aware normalization

The app allows a user to select the text supplied to SapBERT while keeping BioBERT extraction unchanged. In order, the strategies are: (1) the exact extracted phrase, (2) the phrase plus up to three whitespace-delimited tokens on either side, and (3) the sentence containing the phrase. The phrase-only option reproduces the original app behavior and is the default. The broader-context options are exploratory and should not be described as improvements unless evaluation on the validation split supports them. Use the supplied `colab_context_normalization_experiment.py` by copying it into a new cell in the existing Colab notebook to compare the strategies on validation examples whose phrases are unseen in training but whose concepts are included in the training-derived candidate vocabulary. Keep the held-out test set untouched during method selection.

These options are not separate models: all use the same pretrained SapBERT encoder and saved candidate concept embeddings; only the text sent to SapBERT changes.
