from __future__ import annotations

import os
import html
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoModelForTokenClassification, AutoTokenizer
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
BIOBERT_LOCAL_DIR = ASSETS / "biobert"
CANDIDATES_PATH = ASSETS / "candidate_adr_table.csv"
EMBEDDINGS_PATH = ASSETS / "concept_embeddings_adr.npy"
SAPBERT_MODEL_NAME = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
MAX_BODY_TOKENS = 510
WINDOW_STRIDE = 384

st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');
      :root { --ink:#142b33; --muted:#637780; --teal:#087f78; --teal-dark:#08635f; --line:#dce7e7; --paper:#f5f8f7; --mint:#e7f5f1; --gold:#c68b37; }
      html, body, [class*="css"] { font-family:'DM Sans', sans-serif; }
      .stApp { background: linear-gradient(180deg,#f7faf9 0%,#f3f7f6 100%); color:var(--ink); }
      [data-testid="stHeader"] { background:rgba(247,250,249,.92); }
      .block-container { max-width:1180px; padding-top:1.7rem; padding-bottom:3.5rem; }
      h1,h2,h3 { font-family:'Manrope',sans-serif !important; color:var(--ink); letter-spacing:-.035em; }
      .brandline { display:flex; align-items:center; gap:11px; margin-bottom:2.0rem; }
      .brandmark { width:38px;height:38px;border-radius:13px;background:#0b6864;color:white;display:flex;align-items:center;justify-content:center;font-size:13px;letter-spacing:-.02em;font-weight:800;box-shadow:0 6px 18px #0b686420; }
      .brandname { font-family:'Manrope',sans-serif; font-weight:800; letter-spacing:.07em; font-size:14px; color:#17343b; }
      .branddesc { font-size:10px; font-weight:700; letter-spacing:.15em; text-transform:uppercase; color:#769096; margin-top:2px; }
      .hero { background:linear-gradient(120deg,#103d43 0%,#0c625f 66%,#087f78 100%); color:white; border-radius:25px; padding:34px 36px 32px; position:relative; overflow:hidden; box-shadow:0 16px 40px rgba(20,66,69,.12); margin-bottom:22px; }
      .hero:after { content:''; position:absolute; width:290px;height:290px;border:1px solid #ffffff20;border-radius:50%;right:-45px;top:-155px;box-shadow:0 0 0 34px #ffffff08,0 0 0 70px #ffffff06; }
      .eyebrow { font-size:11px; letter-spacing:.17em; text-transform:uppercase; font-weight:700; color:#aee4d8; margin-bottom:12px; }
      .hero h1 { color:#fff !important; font-size:clamp(29px,4vw,44px); line-height:1.12; font-weight:800; max-width:720px; margin:0 0 13px 0; }
      .hero p { color:#d8eeea; font-size:15px; line-height:1.7; max-width:740px; margin:0; }
      .hero-pills { display:flex;flex-wrap:wrap;gap:8px;margin-top:22px;position:relative;z-index:1; }
      .pill { border:1px solid #ffffff35; border-radius:999px; padding:7px 11px; font-size:11px; color:#eefaf7; background:#ffffff0c; font-weight:600; }
      .section-kicker { font-size:10px; text-transform:uppercase; letter-spacing:.17em; color:#628087; font-weight:800; margin:4px 0 8px; }
      .section-title { font-family:'Manrope',sans-serif;font-size:22px;font-weight:800;letter-spacing:-.035em;color:#18373e;margin:0 0 5px; }
      .section-copy { font-size:13px;color:#6b8187;line-height:1.6;margin-bottom:16px; }
      div[data-testid="stVerticalBlockBorderWrapper"] { border-color:#dfe9e7 !important; background:rgba(255,255,255,.83); border-radius:18px; }
      div[data-testid="stTextArea"] textarea { background:#fbfdfc; border:1px solid #d8e5e2; border-radius:13px; color:#19383e; font-size:14px; line-height:1.7; }
      div[data-testid="stTextArea"] textarea:focus { border-color:#168b81; box-shadow:0 0 0 2px #168b8115; }
      .stButton > button { border-radius:11px; min-height:43px; font-weight:700; border:1px solid #cfe0dd; }
      .stButton > button[kind="primary"] { background:#087f78; color:white; border:1px solid #087f78; }
      .stButton > button[kind="primary"]:hover { background:#08635f; border-color:#08635f; }
      div[data-testid="stSelectbox"] div[data-baseweb="select"] > div { border-radius:10px; background:#fff; }
      .metric-card { padding:16px 17px; border:1px solid #dfe9e7; background:white; border-radius:16px; min-height:93px; }
      .metric-label { font-size:10px;text-transform:uppercase;letter-spacing:.13em;font-weight:800;color:#738b90;margin-bottom:8px; }
      .metric-value { font-family:'Manrope',sans-serif;color:#123a40;font-size:23px;font-weight:800;line-height:1.1; }
      .metric-note { color:#7b8f93;font-size:11px;margin-top:5px; }
      .result-card { background:white;border:1px solid #dfe9e7;border-radius:16px;padding:17px 18px;margin:8px 0 12px; }
      .result-top { display:flex;align-items:flex-start;justify-content:space-between;gap:14px;flex-wrap:wrap; }
      .mention { font-family:'Manrope',sans-serif;font-size:18px;font-weight:800;letter-spacing:-.025em;color:#17383e; }
      .offset { font-family:ui-monospace, SFMono-Regular, Menlo, monospace;font-size:10px;color:#7a9194;background:#f3f7f6;border:1px solid #e2ece9;border-radius:7px;padding:4px 7px;white-space:nowrap; }
      .concept-label { font-size:10px;text-transform:uppercase;letter-spacing:.12em;font-weight:800;color:#779095;margin:16px 0 5px; }
      .concept-name { font-family:'Manrope',sans-serif;font-size:16px;font-weight:800;color:#08746c; }
      .concept-id { font-family:ui-monospace, SFMono-Regular, Menlo, monospace;font-size:11px;color:#6d8588;margin-top:5px; }
      .sim-badge { font-size:11px;font-weight:800;color:#14655e;background:#e4f5ee;border-radius:999px;padding:6px 9px;white-space:nowrap; }
      .empty-state { background:white;border:1px dashed #cbdcd8;border-radius:17px;padding:26px;text-align:center;color:#7a9094;font-size:13px;line-height:1.7; }
      .note { background:#edf5f3;border-left:3px solid #11877e;border-radius:0 10px 10px 0;padding:12px 14px;color:#527076;font-size:12px;line-height:1.65; }
      .footer { border-top:1px solid #dce7e4;margin-top:36px;padding-top:15px;color:#819397;font-size:10px;letter-spacing:.05em;display:flex;justify-content:space-between;gap:15px;flex-wrap:wrap; }
      .stTabs [data-baseweb="tab-list"] { gap:9px; }
      .stTabs [data-baseweb="tab"] { border-radius:9px;padding:8px 12px; }
      [data-testid="stDataFrame"] { border:1px solid #e0eae8;border-radius:12px;overflow:hidden; }
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
def load_models() -> tuple[Any, Any, Any, Any, pd.DataFrame, torch.Tensor, torch.device]:
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

    sap_tokenizer = AutoTokenizer.from_pretrained(SAPBERT_MODEL_NAME)
    sap_model = AutoModel.from_pretrained(
        SAPBERT_MODEL_NAME, low_cpu_mem_usage=True
    )
    sap_model.to(device).eval()

    # Prefer local assets for development. For deployment, the same private
    # Hugging Face repository can also hold these assets so they need not be
    # committed to the public GitHub source repository.
    candidates_path = CANDIDATES_PATH
    embeddings_path = EMBEDDINGS_PATH
    if not candidates_path.exists() or not embeddings_path.exists():
        asset_repo_id = secret_value("BIOBERT_REPO_ID")
        if not asset_repo_id:
            raise FileNotFoundError(
                "Normalization assets are missing. For local use, place "
                "candidate_adr_table.csv and concept_embeddings_adr.npy in assets/. "
                "For hosted use, upload both files to the private Hugging Face model "
                "repository and set BIOBERT_REPO_ID in Streamlit Secrets."
            )
        try:
            candidates_path = Path(hf_hub_download(
                repo_id=asset_repo_id,
                filename="candidate_adr_table.csv",
                token=hf_token,
            ))
            embeddings_path = Path(hf_hub_download(
                repo_id=asset_repo_id,
                filename="concept_embeddings_adr.npy",
                token=hf_token,
            ))
        except Exception as exc:
            raise RuntimeError(
                "Could not download the normalization assets from the configured "
                "Hugging Face repository. Confirm that both files are in the repo "
                "root and that HF_TOKEN has read access to that repository."
            ) from exc

    candidates = pd.read_csv(candidates_path, dtype={"snomed_id": str})
    embeddings = np.load(embeddings_path).astype(np.float32)
    if len(candidates) != embeddings.shape[0]:
        raise ValueError(
            "The candidate concept table and embedding matrix have different row counts. "
            "Export both from the same notebook run and preserve their row order."
        )
    if embeddings.ndim != 2 or embeddings.shape[1] != 768:
        raise ValueError(
            f"Expected SapBERT concept embeddings of shape (N, 768), got {embeddings.shape}."
        )

    concept_matrix = torch.tensor(embeddings, dtype=torch.float32, device=device)
    concept_matrix = F.normalize(concept_matrix, p=2, dim=1)
    return bio_tokenizer, bio_model, sap_tokenizer, sap_model, candidates, concept_matrix, device


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


def normalize_phrase(
    phrase: str,
    sap_tokenizer: Any,
    sap_model: Any,
    candidates: pd.DataFrame,
    concept_matrix: torch.Tensor,
    device: torch.device,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """Rank candidates with SapBERT CLS embeddings and cosine similarity."""
    encoded = sap_tokenizer(
        [phrase], padding=True, truncation=True, max_length=64,
        return_tensors="pt"
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.inference_mode():
        output = sap_model(**encoded)
        phrase_vector = F.normalize(output.last_hidden_state[:, 0, :], p=2, dim=1)
        similarities = (phrase_vector @ concept_matrix.T)[0]
        scores, indices = torch.topk(similarities, k=min(top_k, len(candidates)))

    ranked: list[dict[str, Any]] = []
    for score, index in zip(scores.cpu().tolist(), indices.cpu().tolist()):
        concept = candidates.iloc[index]
        ranked.append({
            "snomed_id": str(concept["snomed_id"]),
            "snomed_name": str(concept["concept_name"]),
            "cosine_similarity": float(score),
        })
    return ranked


def analyze_text(text: str) -> tuple[list[dict[str, Any]], int]:
    bio_tokenizer, bio_model, sap_tokenizer, sap_model, candidates, concept_matrix, device = load_models()
    mentions = extract_adr_mentions(text, bio_tokenizer, bio_model, device)
    results: list[dict[str, Any]] = []
    for item in mentions:
        ranked = normalize_phrase(
            item["mention"], sap_tokenizer, sap_model,
            candidates, concept_matrix, device, top_k=5
        )
        results.append({**item, "ranked_concepts": ranked})
    return results, len(candidates)


def render_metric(label: str, value: str, note: str) -> None:
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div><div class="metric-note">{note}</div></div>',
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------
# Header / hero
# -----------------------------------------------------------------------------
st.markdown(
    '<div class="brandline"><div class="brandmark">MP</div><div>'
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
    '<span class="pill">02&nbsp; SapBERT · normalization</span>'
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

    patient_text = st.text_area(
        "Patient-style text",
        key="patient_text",
        height=165,
        max_chars=5000,
        placeholder="For example: The tablets made me feel dizzy and gave me stomach pain…",
        label_visibility="visible",
        help="Use fictional text only. Do not enter identifiable patient information.",
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
                    output, candidate_count = analyze_text(patient_text.strip())
                st.session_state["analysis_results"] = output
                st.session_state["analysis_candidate_count"] = candidate_count
                st.session_state["analysis_source_text"] = patient_text.strip()
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
        st.caption("SapBERT encodes each extracted phrase using CLS pooling.")
        st.markdown("**03 — Rank**")
        st.caption("Cosine similarity orders candidate SNOMED CT concepts from most to less similar.")
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
                    f'<div class="sim-badge">Cosine {top["cosine_similarity"]:.3f}</div></div>'
                    f'<div class="concept-label">Top-ranked concept</div>'
                    f'<div class="concept-name">{html.escape(top["snomed_name"])}</div>'
                    f'<div class="concept-id">SNOMED CT · {html.escape(top["snomed_id"])}</div></div>',
                    unsafe_allow_html=True,
                )
            with st.expander(f"View top-five ranking for “{result['mention']}”"):
                rank_df = pd.DataFrame([
                    {
                        "Rank": i,
                        "SNOMED CT ID": concept["snomed_id"],
                        "Concept name": concept["snomed_name"],
                        "Cosine similarity": round(concept["cosine_similarity"], 4),
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
                        "rank": rank,
                        "snomed_id": concept["snomed_id"],
                        "snomed_name": concept["snomed_name"],
                        "cosine_similarity": concept["cosine_similarity"],
                    })
            export_df = pd.DataFrame(export_rows)
            if not export_df.empty:
                st.download_button(
                    "Download results as CSV",
                    data=export_df.to_csv(index=False).encode("utf-8"),
                    file_name="medphrase_patient_side_effect_analysis.csv",
                    mime="text/csv",
                )

st.markdown(
    '<div class="footer"><span>MEDPHRASE · MAAI 902 NLP SOLUTION DESIGN</span>'
    '<span>RESEARCH DEMO ONLY · NOT FOR CLINICAL USE</span></div>',
    unsafe_allow_html=True,
)
