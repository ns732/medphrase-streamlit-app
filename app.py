from __future__ import annotations

import os
import re
import html
import base64
import io
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import joblib
from scipy import sparse
import streamlit as st
import streamlit.components.v1 as components
import speech_recognition as sr
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer
from huggingface_hub import hf_hub_download

# -----------------------------------------------------------------------------
# Page and visual design
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="MedPhrase | Patient Side-Effect Analyzer",
    page_icon="✳",
    layout="wide",
    initial_sidebar_state="collapsed",
)

ROOT = Path(__file__).parent
ASSETS = ROOT / "assets"
LOGO_PATH = ASSETS / "medphrase_logo.png"
# For simple GitHub web uploads, the same logo may also be placed in the repo root.
if not LOGO_PATH.exists() and (ROOT / "medphrase_logo.png").exists():
    LOGO_PATH = ROOT / "medphrase_logo.png"
BIOBERT_LOCAL_DIR = ASSETS / "biobert"
CANDIDATES_PATH = ASSETS / "candidate_adr_table.csv"
CHAR_VECTORIZER_PATH = ASSETS / "char_tfidf_vectorizer.joblib"
CHAR_ALIAS_MATRIX_PATH = ASSETS / "char_alias_matrix.npz"
CHAR_ALIASES_PATH = ASSETS / "char_aliases.csv"
CHAR_CANDIDATES_PATH = ASSETS / "char_candidate_table.csv"
MAX_BODY_TOKENS = 510
WINDOW_STRIDE = 384

st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');
      :root {
        --ink:#EAF5F7;
        --muted:#A4C0D0;
        --teal:#10BFC0;
        --teal-dark:#078E97;
        --line:#16475D;
        --paper:#061A2B;
        --panel:#08283B;
        --panel-alt:#0D3045;
        --mint:#103A4A;
        --gold:#F3C66B;
      }
      html, body, [class*="css"] { font-family:'DM Sans', sans-serif; }
      .stApp {
        color:var(--ink);
        background:
          radial-gradient(ellipse at 85% 0%, rgba(8,110,135,.22) 0%, rgba(8,110,135,0) 38%),
          linear-gradient(145deg,#041522 0%,#061A2B 52%,#041927 100%);
      }
      [data-testid="stHeader"] { background:rgba(4,21,34,.82); }
      [data-testid="stToolbar"] { right:1rem; }
      .block-container { max-width:1240px; padding-top:1.7rem; padding-bottom:3.5rem; }
      h1,h2,h3,h4,h5,h6, p, li, label { font-family:'DM Sans',sans-serif; color:var(--ink); }
      h1,h2,h3,h4 { font-family:'Manrope',sans-serif !important; letter-spacing:-.035em; }
      [data-testid="stMarkdownContainer"] p { color:#B5CAD8; }
      [data-testid="stCaptionContainer"] { color:#94B2C4; }
      .brandline { display:flex; align-items:center; gap:12px; margin-bottom:1.5rem; }
      .brand-logo {
        width:54px;height:54px;object-fit:contain;display:block;flex:0 0 54px;
        filter:drop-shadow(0 5px 18px rgba(16,191,192,.10));
      }
      .brandmark {
        width:42px;height:42px;border-radius:14px;
        background:linear-gradient(145deg,#18D0CE,#078E97);color:#041A2A;
        display:flex;align-items:center;justify-content:center;font-size:13px;
        letter-spacing:-.02em;font-weight:800;box-shadow:0 7px 24px #10BFC02B;
      }
      .brandname { font-family:'Manrope',sans-serif; font-weight:800; letter-spacing:.01em; font-size:16px; color:#F3FAFC; }
      .branddesc { font-size:10px; font-weight:700; letter-spacing:.15em; text-transform:uppercase; color:#8EB8CB; margin-top:3px; }
      .hero {
        background:linear-gradient(118deg,rgba(5,30,47,.98) 0%,rgba(7,45,66,.98) 58%,rgba(5,85,98,.96) 100%);
        color:white;border:1px solid #10516A;border-radius:23px;padding:34px 36px 32px;
        position:relative;overflow:hidden;box-shadow:0 18px 46px rgba(0,0,0,.20);margin-bottom:26px;
      }
      .hero:after {
        content:'';position:absolute;width:300px;height:300px;border:1px solid #10BFC045;
        border-radius:50%;right:-60px;top:-165px;
        box-shadow:0 0 0 36px #10BFC012,0 0 0 76px #10BFC009;
      }
      .eyebrow { font-size:10px;letter-spacing:.18em;text-transform:uppercase;font-weight:800;color:#41D8D4;margin-bottom:13px; }
      .hero h1 { color:#F5FBFC !important;font-size:clamp(30px,4vw,46px);line-height:1.12;font-weight:800;max-width:760px;margin:0 0 13px 0; }
      .hero p { color:#B8D3E1;font-size:15px;line-height:1.75;max-width:780px;margin:0; }
      .hero-pills { display:flex;flex-wrap:wrap;gap:8px;margin-top:22px;position:relative;z-index:1; }
      .pill { border:1px solid #30BFC04D;border-radius:999px;padding:7px 11px;font-size:11px;color:#DDF9FA;background:#0B89901C;font-weight:600; }
      .section-kicker { font-size:10px;text-transform:uppercase;letter-spacing:.17em;color:#49C9CC;font-weight:800;margin:4px 0 8px; }
      .section-title { font-family:'Manrope',sans-serif;font-size:22px;font-weight:800;letter-spacing:-.035em;color:#EDF8FA;margin:0 0 5px; }
      .section-copy { font-size:13px;color:#9FBACA;line-height:1.7;margin-bottom:16px; }
      div[data-testid="stVerticalBlockBorderWrapper"] {
        border-color:#16475D !important;
        background:linear-gradient(150deg,rgba(8,40,59,.94),rgba(5,30,47,.92));
        border-radius:17px;
      }
      div[data-testid="stTextArea"] textarea {
        background:#071F31;border:1px solid #24516A;border-radius:13px;color:#F0F8FA;
        font-size:14px;line-height:1.7;caret-color:#10BFC0;
      }
      div[data-testid="stTextArea"] textarea::placeholder { color:#7997A9; }
      div[data-testid="stTextArea"] textarea:focus { border-color:#10BFC0;box-shadow:0 0 0 2px #10BFC022; }
      div[data-testid="stTextArea"] label, div[data-testid="stSelectbox"] label,
      div[data-testid="stAudioInput"] label { color:#D9EAF0 !important;font-weight:600; }
      div[data-testid="stAudioInput"] { border:1px solid #16475D;border-radius:12px;padding:10px;background:#071F31; }
      .stButton > button, .stDownloadButton > button {
        border-radius:11px;min-height:43px;font-weight:700;
        background:#0A2A3E;color:#DCECF2;border:1px solid #26536A;
      }
      .stButton > button:hover, .stDownloadButton > button:hover { border-color:#10BFC0;color:#FFFFFF;background:#0C354A; }
      .stButton > button[kind="primary"] {
        background:linear-gradient(100deg,#0B9EA5,#10BFC0);color:#041B2A;
        border:1px solid #10BFC0;box-shadow:0 7px 20px #10BFC018;
      }
      .stButton > button[kind="primary"]:hover { background:#38D3D0;border-color:#38D3D0;color:#041B2A; }
      div[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
        border-radius:10px;background:#071F31;border-color:#24516A;color:#EAF5F7;
      }
      div[data-baseweb="popover"] { background:#08283B; }
      div[role="listbox"] { background:#08283B; }
      div[role="option"] { background:#08283B;color:#EAF5F7; }
      div[role="option"]:hover { background:#104459; }
      .metric-card { padding:16px 17px;border:1px solid #16475D;background:linear-gradient(145deg,#0B2B40,#071F31);border-radius:16px;min-height:93px; }
      .metric-label { font-size:10px;text-transform:uppercase;letter-spacing:.13em;font-weight:800;color:#8EB0C1;margin-bottom:8px; }
      .metric-value { font-family:'Manrope',sans-serif;color:#4FE0DB;font-size:23px;font-weight:800;line-height:1.1; }
      .metric-note { color:#91ADBD;font-size:11px;margin-top:5px; }
      .result-card { background:linear-gradient(145deg,#0A2B40,#071F31);border:1px solid #16475D;border-radius:16px;padding:17px 18px;margin:8px 0 12px; }
      .result-top { display:flex;align-items:flex-start;justify-content:space-between;gap:14px;flex-wrap:wrap; }
      .mention { font-family:'Manrope',sans-serif;font-size:18px;font-weight:800;letter-spacing:-.025em;color:#F0FAFC; }
      .offset { font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:10px;color:#9FB9C8;background:#0A2234;border:1px solid #1C495F;border-radius:7px;padding:4px 7px;white-space:nowrap;display:inline-block;margin-top:7px; }
      .concept-label { font-size:10px;text-transform:uppercase;letter-spacing:.12em;font-weight:800;color:#8EB0C1;margin:16px 0 5px; }
      .concept-name { font-family:'Manrope',sans-serif;font-size:16px;font-weight:800;color:#32D5D1; }
      .concept-id { font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:11px;color:#94B2C3;margin-top:5px; }
      .sim-badge { font-size:11px;font-weight:800;color:#CFFBFA;background:#0B777A55;border:1px solid #10BFC04A;border-radius:999px;padding:6px 9px;white-space:nowrap; }
      .empty-state { background:#08283B;border:1px dashed #246078;border-radius:17px;padding:26px;text-align:center;color:#A2BDCA;font-size:13px;line-height:1.7; }
      .note { background:#0B3546;border-left:3px solid #10BFC0;border-radius:0 10px 10px 0;padding:12px 14px;color:#B8D3DF;font-size:12px;line-height:1.7; }
      .footer { border-top:1px solid #16475D;margin-top:36px;padding-top:15px;color:#799AAC;font-size:10px;letter-spacing:.05em;display:flex;justify-content:space-between;gap:15px;flex-wrap:wrap; }
      .stTabs [data-baseweb="tab-list"] { gap:9px;background:transparent; }
      .stTabs [data-baseweb="tab"] { border-radius:9px;padding:8px 12px;color:#B2C9D6; }
      .stTabs [aria-selected="true"] { color:#35D6D2 !important;border-bottom-color:#10BFC0 !important; }
      [data-testid="stDataFrame"], [data-testid="stTable"] { border:1px solid #16475D;border-radius:12px;overflow:hidden; }
      [data-testid="stExpander"] { border:1px solid #16475D;border-radius:12px;background:#08283B; }
      [data-testid="stExpander"] summary { color:#EAF5F7; }
      [data-testid="stAlert"] { border-radius:12px; }
      hr { border-color:#16475D !important; }
      code { color:#8DE5E1;background:#0A2A3D; }
      #MainMenu, footer { visibility:hidden; }
      @media (max-width: 700px) { .hero {padding:25px 22px;} .block-container {padding-top:1rem;} }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Text, labels and utility functions
# -----------------------------------------------------------------------------
SAMPLES = {
    "Everyday side effects": "After taking the medicine, I felt dizzy and tired. I also had a bad stomach upset and a terrible headache.",
    "Stomach discomfort": "The tablets made me feel sick and gave me stomach pain. I also felt very weak and had trouble sleeping.",
    "Informal wording": "Since I started the medication, my head has been pounding and I feel really wiped out. My tummy is upset too.",
}

ID2LABEL = {0: "O", 1: "B-ADR", 2: "I-ADR"}


def secret_value(key: str, default: str = "") -> str:
    try:
        value = st.secrets.get(key, default)
        return str(value).strip() if value is not None else default
    except Exception:
        return os.getenv(key, default).strip()


def source_for_biobert() -> tuple[str, str | None]:
    """Return a local model path when present; otherwise use HF repo settings."""
    has_local_model = (BIOBERT_LOCAL_DIR / "config.json").exists() and any(
        (BIOBERT_LOCAL_DIR / name).exists()
        for name in ("model.safetensors", "pytorch_model.bin")
    )
    if has_local_model:
        return str(BIOBERT_LOCAL_DIR), None

    repo_id = secret_value("BIOBERT_REPO_ID")
    hf_token = secret_value("HF_TOKEN") or None
    if repo_id:
        return repo_id, hf_token
    raise FileNotFoundError(
        "The fine-tuned BioBERT model is not configured yet. Add the model files "
        "to assets/biobert/ for local use, or set BIOBERT_REPO_ID in Streamlit "
        "Secrets. See README.md for the Colab export and deployment steps."
    )


@st.cache_resource(show_spinner=False)
def load_models() -> tuple[Any, Any, pd.DataFrame, Any, Any, pd.DataFrame, torch.device]:
    """Load BioBERT plus the frozen character-TF-IDF normalizer assets."""
    source, hf_token = source_for_biobert()
    device = torch.device("cpu")
    torch.set_num_threads(max(1, min(2, os.cpu_count() or 1)))

    bio_tokenizer = AutoTokenizer.from_pretrained(
        source, use_fast=True, token=hf_token
    )
    if not bio_tokenizer.is_fast:
        raise RuntimeError("The saved BioBERT tokenizer must be a fast tokenizer.")
    bio_model = AutoModelForTokenClassification.from_pretrained(
        source, token=hf_token, low_cpu_mem_usage=True
    )
    bio_model.to(device).eval()

    # Use the exact fitted vectorizer, alias matrix, and aliases exported from Colab.
    # For deployment, keep these artifacts in the same private HF repo as BioBERT.
    asset_names = {
        "vectorizer": (CHAR_VECTORIZER_PATH, "char_tfidf_vectorizer.joblib"),
        "alias_matrix": (CHAR_ALIAS_MATRIX_PATH, "char_alias_matrix.npz"),
        "aliases": (CHAR_ALIASES_PATH, "char_aliases.csv"),
        "candidates": (CHAR_CANDIDATES_PATH, "char_candidate_table.csv"),
    }
    resolved: dict[str, Path] = {}
    asset_repo_id = secret_value("BIOBERT_REPO_ID")
    for key, (local_path, filename) in asset_names.items():
        if local_path.exists():
            resolved[key] = local_path
            continue
        if not asset_repo_id:
            raise FileNotFoundError(
                f"Missing {filename}. Export the character-TF-IDF assets from Colab and "
                "upload them to the private Hugging Face repository configured by BIOBERT_REPO_ID, "
                "or place them in the app's assets/ folder."
            )
        try:
            resolved[key] = Path(hf_hub_download(
                repo_id=asset_repo_id, filename=filename, token=(secret_value("HF_TOKEN") or hf_token)
            ))
        except Exception as exc:
            raise RuntimeError(
                f"Could not download {filename} from the configured Hugging Face repository. "
                "Check that the file is uploaded and HF_TOKEN has read access."
            ) from exc

    vectorizer = joblib.load(resolved["vectorizer"])
    alias_matrix = sparse.load_npz(resolved["alias_matrix"]).tocsr()
    aliases = pd.read_csv(resolved["aliases"], dtype={"snomed_id": str})
    candidates = pd.read_csv(resolved["candidates"], dtype={"snomed_id": str})

    required_alias_cols = {"mention", "snomed_id"}
    required_candidate_cols = {"snomed_id", "concept_name"}
    if not required_alias_cols.issubset(aliases.columns):
        raise ValueError(f"char_aliases.csv must contain columns {sorted(required_alias_cols)}")
    if not required_candidate_cols.issubset(candidates.columns):
        raise ValueError(f"char_candidate_table.csv must contain columns {sorted(required_candidate_cols)}")
    if alias_matrix.shape[0] != len(aliases):
        raise ValueError("Alias matrix row count does not match char_aliases.csv.")
    if alias_matrix.shape[1] != len(vectorizer.get_feature_names_out()):
        raise ValueError("Alias matrix feature count does not match the saved TF-IDF vectorizer.")

    candidates["snomed_id"] = candidates["snomed_id"].astype(str)
    aliases["snomed_id"] = aliases["snomed_id"].astype(str)
    return bio_tokenizer, bio_model, candidates, vectorizer, alias_matrix, aliases, device


def extract_adr_mentions(text: str, bio_tokenizer: Any, bio_model: Any, device: torch.device) -> list[dict[str, Any]]:
    """Extract BIO-tagged ADR spans from a full text using overlapping windows."""
    if not isinstance(text, str) or not text.strip():
        return []

    encoded = bio_tokenizer(
        text, add_special_tokens=False, truncation=False,
        return_offsets_mapping=True
    )
    token_ids = encoded["input_ids"]
    offsets = encoded["offset_mapping"]
    if not token_ids:
        return []

    cls_id = bio_tokenizer.cls_token_id
    sep_id = bio_tokenizer.sep_token_id
    num_labels = int(bio_model.config.num_labels)
    accumulated = np.zeros((len(token_ids), num_labels), dtype=np.float64)
    counts = np.zeros(len(token_ids), dtype=np.int32)

    for window_start in range(0, len(token_ids), WINDOW_STRIDE):
        window_end = min(window_start + MAX_BODY_TOKENS, len(token_ids))
        body_ids = token_ids[window_start:window_end]
        ids = [cls_id] + body_ids + [sep_id]
        model_inputs = {
            "input_ids": torch.tensor([ids], dtype=torch.long, device=device),
            "attention_mask": torch.ones((1, len(ids)), dtype=torch.long, device=device),
        }
        model_inputs["token_type_ids"] = torch.zeros(
            (1, len(ids)), dtype=torch.long, device=device
        )
        with torch.inference_mode():
            logits = bio_model(**model_inputs).logits[0, 1:1 + len(body_ids)]
        accumulated[window_start:window_end] += logits.cpu().numpy()
        counts[window_start:window_end] += 1

    predicted_ids = (accumulated / np.maximum(counts[:, None], 1)).argmax(axis=1)
    spans: list[tuple[int, int]] = []
    active_start: int | None = None
    active_end: int | None = None

    def close_span() -> None:
        nonlocal active_start, active_end
        if active_start is not None and active_end is not None:
            start, end = active_start, active_end
            while start < end and text[start].isspace():
                start += 1
            while end > start and text[end - 1].isspace():
                end -= 1
            if start < end:
                spans.append((start, end))
        active_start, active_end = None, None

    for i, label_id in enumerate(predicted_ids):
        start, end = offsets[i]
        if end <= start:
            close_span()
            continue
        label = ID2LABEL.get(int(label_id), "O")
        if label == "B-ADR":
            close_span()
            active_start, active_end = start, end
        elif label == "I-ADR":
            if active_start is None:
                active_start = start
            active_end = end
        else:
            close_span()
    close_span()

    return [
        {"mention": text[start:end], "start": start, "end": end}
        for start, end in sorted(set(spans))
    ]


def informal_query_variants(phrase: str) -> list[str]:
    """Return conservative spelling/colloquial variants for phrase normalisation.

    The original phrase is always retained. These rules only generate additional
    search queries; they do not alter the extracted span shown to the user.
    They are heuristics and must be evaluated before making performance claims.
    """
    original = str(phrase).strip()
    if not original:
        return [original]

    variants = [original]

    # Common informal forms with relatively direct clinical-language equivalents.
    replacements = (
        (r"\bthrowing\s+up\b", "vomiting"),
        (r"\bpuking\b", "vomiting"),
        (r"\bpuked\b", "vomited"),
        (r"\btummy\b", "stomach"),
        (r"\blight[- ]?headed\b", "light headed"),
        (r"\brlly\b|\brly\b", "really"),
    )
    expanded = original.lower()
    for pattern, replacement in replacements:
        expanded = re.sub(pattern, replacement, expanded, flags=re.IGNORECASE)
    if expanded != original.lower():
        variants.append(expanded)

    # Reduce exaggerated character repetitions while preserving ordinary doubles
    # (e.g. "dizzzzy" -> "dizzy", not "dizy").
    de_stretched = re.sub(r"(.)\1{2,}", r"\1\1", original.lower())
    if de_stretched not in {v.lower() for v in variants}:
        variants.append(de_stretched)

    # Include a punctuation-light variant for casual typing (e.g. light-headed).
    punctuation_light = re.sub(r"[^\w\s'-]", " ", original.lower())
    punctuation_light = re.sub(r"\s+", " ", punctuation_light).strip()
    if punctuation_light and punctuation_light not in {v.lower() for v in variants}:
        variants.append(punctuation_light)

    # Stable de-duplication, preserving the original as the first query.
    seen = set()
    return [v for v in variants if v and not (v.lower() in seen or seen.add(v.lower()))]


def normalize_phrase(
    phrase: str,
    candidates: pd.DataFrame,
    vectorizer: Any,
    alias_matrix: Any,
    aliases: pd.DataFrame,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """Rank concepts by best alias similarity across original and informal query variants."""
    top_k = min(max(1, int(top_k)), len(candidates))
    query_variants = informal_query_variants(str(phrase))
    query_matrix = vectorizer.transform(query_variants)
    all_scores = (query_matrix @ alias_matrix.T).toarray()
    # Each alias gets its strongest score across query variants; concepts then
    # retain the existing max-over-aliases aggregation below.
    alias_scores = all_scores.max(axis=0) if all_scores.size else np.zeros(len(aliases))

    # Preserve the validated approach: each concept receives its strongest alias match.
    concept_scores: dict[str, float] = {}
    alias_ids = aliases["snomed_id"].astype(str).to_numpy()
    for concept_id, score in zip(alias_ids, alias_scores):
        if score > concept_scores.get(concept_id, -1.0):
            concept_scores[concept_id] = float(score)

    ranked: list[dict[str, Any]] = []
    candidates = candidates.reset_index(drop=True)
    score_rows = []
    for _, concept in candidates.iterrows():
        concept_id = str(concept["snomed_id"])
        score_rows.append((concept_scores.get(concept_id, 0.0), concept_id, str(concept["concept_name"])))
    score_rows.sort(key=lambda row: row[0], reverse=True)

    for score, concept_id, concept_name in score_rows[:top_k]:
        ranked.append({
            "snomed_id": concept_id,
            "snomed_name": concept_name,
            "tfidf_similarity": float(score),
        })
    return ranked


def analyze_text(
    text: str,
    normalization_strategy: str = "Character TF-IDF + informal query variants (experimental)",
) -> tuple[list[dict[str, Any]], int]:
    bio_tokenizer, bio_model, candidates, vectorizer, alias_matrix, aliases, device = load_models()
    mentions = extract_adr_mentions(text, bio_tokenizer, bio_model, device)
    results: list[dict[str, Any]] = []
    for item in mentions:
        normalization_input = item["mention"]  # phrase-only was the selected validation setting
        ranked = normalize_phrase(
            normalization_input, candidates, vectorizer, alias_matrix, aliases, top_k=5
        )
        results.append({
            **item,
            "normalization_input": normalization_input,
            "normalization_strategy": "Character TF-IDF + informal query variants (experimental)",
            "ranked_concepts": ranked,
        })
    return results, len(candidates)


def build_read_aloud_summary(results: list[dict[str, Any]], source_text: str) -> str:
    """Build a cautious, concise summary for optional browser read-aloud."""
    if not results:
        return (
            "MedPhrase did not identify any potential adverse reaction phrases in this text. "
            "This does not mean side effects are absent. The full patient statement should be reviewed."
        )

    parts = [
        "MedPhrase identified the following possible side-effect phrases. "
        "These are model suggestions, not diagnoses."
    ]
    for item in results:
        ranked = item.get("ranked_concepts", [])
        phrase = str(item.get("mention", "")).strip()
        if not phrase:
            continue
        if ranked:
            top = ranked[0]
            parts.append(
                f"The phrase, {phrase}. The top-ranked concept is "
                f"{top.get('snomed_name', 'not available')}, SNOMED CT identifier "
                f"{top.get('snomed_id', 'not available')}."
            )
        else:
            parts.append(f"The phrase, {phrase}. No concept suggestions were returned.")
    parts.append("Please review all suggestions; similarity scores are not probabilities.")
    return " ".join(parts)


def render_read_aloud_player(text: str) -> None:
    """Render a doctor-controlled browser text-to-speech player.

    Speech is generated by the browser/device's built-in SpeechSynthesis API;
    it does not call a separate text-to-speech provider from this app.
    """
    text_json = json.dumps(text, ensure_ascii=False).replace("</", r"<\/")
    html_player = f"""
    <div style="font-family:Arial,sans-serif;background:#08283B;border:1px solid #16475D;
                padding:12px;border-radius:12px;color:#EAF5F7;display:flex;gap:10px;
                align-items:center;flex-wrap:wrap;">
      <button id="play" type="button" style="border:0;border-radius:9px;padding:10px 14px;
          background:#10BFC0;color:#061A2B;font-weight:700;cursor:pointer;">
        ▶ Read aloud
      </button>
      <button id="stop" type="button" style="border:1px solid #26536A;border-radius:9px;
          padding:10px 14px;background:#0A2A3E;color:#EAF5F7;font-weight:600;cursor:pointer;">
        ■ Stop
      </button>
      <span id="status" role="status" style="font-size:12px;color:#A4C0D0;">
        Playback is optional. Use the controls when the doctor chooses.
      </span>
    </div>
    <script>
      const speechText = {text_json};
      const statusEl = document.getElementById('status');
      document.getElementById('play').addEventListener('click', () => {{
        if (!('speechSynthesis' in window) || !('SpeechSynthesisUtterance' in window)) {{
          statusEl.textContent = 'This browser does not support built-in speech playback.';
          return;
        }}
        window.speechSynthesis.cancel();
        const utterance = new SpeechSynthesisUtterance(speechText);
        utterance.lang = 'en-US';
        utterance.rate = 0.94;
        utterance.onstart = () => statusEl.textContent = 'Playing audio…';
        utterance.onend = () => statusEl.textContent = 'Playback finished.';
        utterance.onerror = () => statusEl.textContent = 'Playback could not start. Check your browser audio settings.';
        window.speechSynthesis.speak(utterance);
      }});
      document.getElementById('stop').addEventListener('click', () => {{
        if ('speechSynthesis' in window) window.speechSynthesis.cancel();
        statusEl.textContent = 'Playback stopped.';
      }});
    </script>
    """
    components.html(html_player, height=88, scrolling=False)


def render_metric(label: str, value: str, note: str) -> None:
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div><div class="metric-note">{note}</div></div>',
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------
# Header / hero
# -----------------------------------------------------------------------------
# Use the custom MedPhrase monogram when the logo asset is present. The small
# text-based mark is retained as a graceful fallback for development setups.
if LOGO_PATH.exists():
    logo_data = base64.b64encode(LOGO_PATH.read_bytes()).decode("ascii")
    brand_icon_html = (
        f'<img class="brand-logo" src="data:image/png;base64,{logo_data}" '
        'alt="MedPhrase logo" />'
    )
else:
    brand_icon_html = '<div class="brandmark" aria-label="MedPhrase logo">MP</div>'

st.markdown(
    f'<div class="brandline">{brand_icon_html}<div>'
    '<div class="brandname">MedPhrase</div>'
    '<div class="branddesc">Patient-language NLP</div></div></div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<section class="hero"><div class="eyebrow">MAAI 902 · Research prototype</div>'
    '<h1>Beyond medical terminology.</h1>'
    '<p>Explore how natural language processing can connect the way people describe '
    'side effects with standardized medical concepts. Write a patient-style sentence '
    'and inspect the model suggestions, from phrase extraction to concept ranking.</p>'
    '<div class="hero-pills"><span class="pill">01&nbsp; BioBERT · extraction</span>'
    '<span class="pill">02&nbsp; Character TF-IDF · normalization</span>'
    '<span class="pill">03&nbsp; SNOMED CT · terminology</span></div></section>',
    unsafe_allow_html=True,
)

left, right = st.columns([1.25, 0.75], gap="large")
with left:
    st.markdown('<div class="section-kicker">Workspace / 01</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Analyze patient language</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-copy">Enter a short, fictional description of possible side effects. '
        'The system returns predicted spans and the five closest concepts in its training-derived candidate vocabulary.</div>',
        unsafe_allow_html=True,
    )
    example_choice = st.selectbox(
        "Try a sample",
        ["Choose a sample…", *SAMPLES.keys()],
        help="Samples are fictional and intended only to demonstrate the interface.",
    )
    if st.button("Load selected sample", use_container_width=False):
        if example_choice in SAMPLES:
            st.session_state["patient_text"] = SAMPLES[example_choice]
        else:
            st.info("Choose one of the sample texts first.")

    with st.expander("Speak instead of typing (optional)", expanded=False):
        st.markdown("**Record a short patient-style statement**")
        st.caption(
            "Use a fictional, non-identifying example only. Recording is sent to this hosted app, "
            "and clicking Transcribe sends the audio to Google's speech-recognition service. "
            "Do not record real patient information."
        )
        audio_recording = st.audio_input(
            "Record a short statement in English",
            sample_rate=16000,
            key="patient_audio_recording",
        )
        speech_language = st.selectbox(
            "Speech recognition language",
            options=["English (US)", "English (UK)", "English (Australia)", "English (Canada)"],
            index=0,
            key="speech_recognition_language",
        )
        speech_language_codes = {
            "English (US)": "en-US",
            "English (UK)": "en-GB",
            "English (Australia)": "en-AU",
            "English (Canada)": "en-CA",
        }
        speech_consent = st.checkbox(
            "I understand the recording will be sent to an external speech-recognition service when transcribed.",
            key="speech_transcription_consent",
        )
        transcribe_clicked = st.button(
            "Transcribe recording into text",
            key="transcribe_recording_button",
            disabled=(audio_recording is None or not speech_consent),
            use_container_width=True,
        )
        if transcribe_clicked:
            try:
                recognizer = sr.Recognizer()
                with sr.AudioFile(io.BytesIO(audio_recording.getvalue())) as audio_source:
                    audio_data = recognizer.record(audio_source)
                with st.spinner("Transcribing the recording…"):
                    transcript = recognizer.recognize_google(
                        audio_data,
                        language=speech_language_codes[speech_language],
                    )
                if not transcript.strip():
                    st.warning("No speech was recognized. Try a shorter, clearer recording.")
                else:
                    # Set this before rendering the text-area widget below.
                    st.session_state["patient_text"] = transcript.strip()
                    st.session_state["speech_transcription_message"] = "Speech transcribed. Review the text before analyzing it."
                    st.rerun()
            except sr.UnknownValueError:
                st.warning("The speech service could not understand that recording. Try speaking more clearly or typing the text.")
            except sr.RequestError:
                st.error("The speech-recognition service is unavailable right now. You can type the statement instead.")
            except Exception as exc:
                st.error("Could not transcribe this recording. Try another recording or type the text instead.")
                with st.expander("Technical details", expanded=False):
                    st.code(str(exc))
        if st.session_state.get("speech_transcription_message"):
            st.success(st.session_state.pop("speech_transcription_message"))

    patient_text = st.text_area(
        "Patient-style text",
        key="patient_text",
        height=165,
        max_chars=5000,
        placeholder="For example: The tablets made me feel dizzy and gave me stomach pain…",
        label_visibility="visible",
        help="Use fictional text only. Do not enter identifiable patient information.",
    )
    normalization_strategy = "Character TF-IDF + informal query variants (experimental)"
    st.caption(
        "Concept ranking uses character-level TF-IDF over training aliases, with conservative "
        "informal spelling variants added as extra search queries. Original text is preserved; "
        "the heuristic enhancement still needs held-out evaluation."
    )
    run_col, clear_col = st.columns([1, 0.6], gap="small")
    def clear_text_state() -> None:
        st.session_state["patient_text"] = ""
        st.session_state.pop("analysis_results", None)
        st.session_state.pop("analysis_source_text", None)

    with run_col:
        analyze_clicked = st.button("✦  Analyze text", type="primary", use_container_width=True)
    with clear_col:
        clear_clicked = st.button("Clear text", use_container_width=True, on_click=clear_text_state)
    if clear_clicked:
        st.rerun()
    if analyze_clicked:
        if not patient_text.strip():
            st.warning("Enter some text or load a sample before analysis.")
        else:
            try:
                with st.spinner("Loading the research models and analyzing text… first run may take a few minutes."):
                    output, candidate_count = analyze_text(patient_text.strip(), normalization_strategy)
                st.session_state["analysis_results"] = output
                st.session_state["analysis_candidate_count"] = candidate_count
                st.session_state["analysis_source_text"] = patient_text.strip()
                st.session_state["analysis_strategy"] = normalization_strategy
                st.success("Analysis complete.")
            except Exception as exc:
                st.error("The analysis could not run yet. Check the setup details below.")
                with st.expander("Technical details", expanded=True):
                    st.code(str(exc))
                st.info(
                    "Deployment requires the fine-tuned BioBERT model plus the two normalization asset files. "
                    "See README.md for the export and setup steps."
                )

with right:
    st.markdown('<div class="section-kicker">Pipeline / 02</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">From phrase to concept</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-copy">Three linked stages make each suggestion inspectable.</div>', unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown("**01 — Extract**")
        st.caption("BioBERT predicts adverse-reaction spans and their character offsets.")
        st.markdown("**02 — Represent**")
        st.caption("Character TF-IDF compares the extracted phrase with aliases from training data.")
        st.markdown("**03 — Rank**")
        st.caption("The strongest alias similarity ranks candidate SNOMED CT concepts.")
    st.markdown(
        '<div class="note"><b>Interpretation note.</b> Similarity is a ranking signal, '
        'not a calibrated probability. Suggestions require human review and are not diagnoses.</div>',
        unsafe_allow_html=True,
    )

# -----------------------------------------------------------------------------
# Results
# -----------------------------------------------------------------------------
st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<div class="section-kicker">Analysis / 03</div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">Model suggestions</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-copy">Predictions are shown as returned by the models. '
    'Incorrect spans and imperfect concept matches are useful evidence for evaluation.</div>',
    unsafe_allow_html=True,
)

if "analysis_results" not in st.session_state:
    st.markdown(
        '<div class="empty-state"><div style="font-size:25px;margin-bottom:8px;">⌁</div>'
        '<b>Your results will appear here.</b><br>Run an analysis to view extracted phrases, '
        'character offsets, and ranked terminology suggestions.</div>',
        unsafe_allow_html=True,
    )
else:
    results = st.session_state["analysis_results"]
    source_text = st.session_state.get("analysis_source_text", "")
    candidate_count = st.session_state.get("analysis_candidate_count", 0)
    used_strategy = st.session_state.get("analysis_strategy", "Phrase only (recommended baseline)")
    st.caption(f"Normalizer used: {used_strategy}")
    metrics = st.columns(3, gap="medium")
    with metrics[0]:
        render_metric("Candidate spans", str(len(results)), "Predicted by BioBERT")
    with metrics[1]:
        render_metric("Concept vocabulary", f"{candidate_count:,}", "Candidate concepts")
    with metrics[2]:
        render_metric("Ranked candidates", str(sum(len(x["ranked_concepts"]) for x in results)), "Up to five per phrase")

    st.markdown("#### Extracted phrases & top suggestions")
    if not results:
        st.warning("BioBERT did not identify any ADR spans in this text. This can be a model error; it does not prove that no side effect is present.")
    else:
        for index, result in enumerate(results, start=1):
            ranked = result["ranked_concepts"]
            top = ranked[0] if ranked else None
            if top:
                st.markdown(
                    f'<div class="result-card"><div class="result-top">'
                    f'<div><div class="mention">{index:02d} &nbsp; {html.escape(result["mention"])}</div>'
                    f'<div class="offset">Characters {result["start"]}–{result["end"]}</div></div>'
                    f'<div class="sim-badge">TF-IDF {top["tfidf_similarity"]:.3f}</div></div>'
                    f'<div class="concept-label">Top-ranked concept</div>'
                    f'<div class="concept-name">{html.escape(top["snomed_name"])}</div>'
                    f'<div class="concept-id">SNOMED CT · {html.escape(top["snomed_id"])}</div></div>',
                    unsafe_allow_html=True,
                )
            with st.expander(f"View top-five ranking for “{result['mention']}”"):
                st.caption(f"Phrase ranked by character TF-IDF: {result.get('normalization_input', result['mention'])}")
                rank_df = pd.DataFrame([
                    {
                        "Rank": i,
                        "SNOMED CT ID": concept["snomed_id"],
                        "Concept name": concept["snomed_name"],
                        "TF-IDF similarity": round(concept["tfidf_similarity"], 4),
                    }
                    for i, concept in enumerate(ranked, start=1)
                ])
                st.dataframe(rank_df, hide_index=True, use_container_width=True)

        with st.expander("View input text and predicted spans"):
            span_df = pd.DataFrame([
                {"Phrase": x["mention"], "Start": x["start"], "End": x["end"],
                 "Original text check": source_text[x["start"]:x["end"]]}
                for x in results
            ])
            st.dataframe(span_df, hide_index=True, use_container_width=True)
            export_rows = []
            for result in results:
                for rank, concept in enumerate(result["ranked_concepts"], start=1):
                    export_rows.append({
                        "mention": result["mention"],
                        "start": result["start"],
                        "end": result["end"],
                        "normalization_strategy": result.get("normalization_strategy", "Character TF-IDF (phrase-only)"),
                        "normalization_input": result.get("normalization_input", result["mention"]),
                        "rank": rank,
                        "snomed_id": concept["snomed_id"],
                        "snomed_name": concept["snomed_name"],
                        "tfidf_similarity": concept["tfidf_similarity"],
                    })
            export_df = pd.DataFrame(export_rows)
            if not export_df.empty:
                st.download_button(
                    "Download results as CSV",
                    data=export_df.to_csv(index=False).encode("utf-8"),
                    file_name="medphrase_patient_side_effect_analysis.csv",
                    mime="text/csv",
                )

if "analysis_results" in st.session_state:
    read_aloud_results = st.session_state.get("analysis_results", [])
    read_aloud_source = st.session_state.get("analysis_source_text", "")
    with st.expander("Optional read-aloud · doctor-controlled", expanded=False):
        st.caption(
            "The doctor can choose whether to play audio. Playback uses the browser/device's "
            "built-in speech voice; it does not automatically play and no separate TTS provider is called by this app."
        )
        read_aloud_choice = st.radio(
            "What should be read aloud?",
            options=["Patient statement", "Model findings summary"],
            horizontal=True,
            key="read_aloud_content_choice",
        )
        if read_aloud_choice == "Patient statement":
            read_aloud_text = read_aloud_source
        else:
            read_aloud_text = build_read_aloud_summary(
                read_aloud_results, read_aloud_source
            )
        if read_aloud_text.strip():
            render_read_aloud_player(read_aloud_text)

st.markdown(
    '<div class="footer"><span>MEDPHRASE · MAAI 902 NLP SOLUTION DESIGN</span>'
    '<span>RESEARCH DEMO ONLY · NOT FOR CLINICAL USE</span></div>',
    unsafe_allow_html=True,
)
