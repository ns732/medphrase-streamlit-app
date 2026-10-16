"""Run this in the existing Google Colab notebook after training/evaluation.

It exports the fine-tuned BioBERT model + fast tokenizer and the SapBERT
candidate table + CLS embeddings needed by the Streamlit demo.
"""

from pathlib import Path
import zipfile
import numpy as np
import pandas as pd
from transformers import AutoTokenizer

EXPORT_ROOT = Path("/content/medphrase_streamlit_export")
BIO_DIR = EXPORT_ROOT / "biobert"
ASSET_DIR = EXPORT_ROOT / "assets"
BIO_DIR.mkdir(parents=True, exist_ok=True)
ASSET_DIR.mkdir(parents=True, exist_ok=True)

required = ["trainer_f1", "MODEL_NAME", "candidate_adr_table", "concept_embeddings_adr"]
missing = [name for name in required if name not in globals()]
if missing:
    raise NameError(
        "Missing notebook variables: " + ", ".join(missing) +
        ". Run this cell in the same Colab runtime after the project experiments."
    )

# Export the trained BioBERT token-classification checkpoint.
trainer_f1.model.save_pretrained(BIO_DIR, safe_serialization=True)
fast_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
fast_tokenizer.save_pretrained(BIO_DIR)

# Export the candidate vocabulary and its matching SapBERT CLS embeddings.
candidates = candidate_adr_table.copy()
embeddings = np.asarray(concept_embeddings_adr, dtype=np.float32)
if len(candidates) != embeddings.shape[0]:
    raise ValueError(
        f"Candidate/embedding mismatch: {len(candidates)} rows vs {embeddings.shape[0]} vectors."
    )
if embeddings.ndim != 2 or embeddings.shape[1] != 768:
    raise ValueError(f"Expected embeddings shape (N, 768); got {embeddings.shape}.")

candidates.to_csv(ASSET_DIR / "candidate_adr_table.csv", index=False)
np.save(ASSET_DIR / "concept_embeddings_adr.npy", embeddings)
(ASSET_DIR / "README.md").write_text(
    "Exported from the CADEC ADR normalization experiment. Keep candidate row order aligned with the NPY vectors.\n",
    encoding="utf-8",
)

zip_path = Path("/content/medphrase_streamlit_export.zip")
with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
    for path in EXPORT_ROOT.rglob("*"):
        if path.is_file():
            archive.write(path, arcname=path.relative_to(EXPORT_ROOT))

print("Export complete:", zip_path)
print("BioBERT files:", [p.name for p in BIO_DIR.iterdir()])
print("Candidate table:", candidates.shape)
print("Embedding matrix:", embeddings.shape)
print("WARNING: Review CADEC and SNOMED CT licenses before redistributing files or model weights.")

try:
    from google.colab import files
    files.download(str(zip_path))
except ImportError:
    print("Download the ZIP from the runtime file browser:", zip_path)
