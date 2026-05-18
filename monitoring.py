"""Production monitoring — shared health payload for Streamlit and Flask."""

from __future__ import annotations

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "sarcasm_detector.keras"
TOKENIZER_PATH = BASE_DIR / "tokenizer_sarcasm.pkl"

MODEL_METADATA = {
    "model_version": "v1.0",
    "architecture": "Bi-LSTM + GloVe",
    "trained_on": "2026-05-18",
    "accuracy_val": 0.87,
    "max_sequence_length": 40,
    "vocab_size": 20_000,
    "embedding_dim": 100,
    "framework": "tensorflow-cpu",
}


def assets_ready() -> bool:
    return MODEL_PATH.is_file() and TOKENIZER_PATH.is_file()


def get_health(*, model_loaded: bool = False) -> dict:
    """Standard MLOps health payload (Streamlit sidebar / Flask /health)."""
    payload = {
        "status": "ok" if assets_ready() else "degraded",
        **MODEL_METADATA,
        "assets": {
            "model_file": str(MODEL_PATH.name),
            "model_present": MODEL_PATH.is_file(),
            "tokenizer_present": TOKENIZER_PATH.is_file(),
        },
        "runtime": {
            "model_loaded_in_memory": model_loaded,
        },
    }
    return payload
