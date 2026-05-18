---
title: Sarcasm Headline Detector
emoji: 📰
colorFrom: red
colorTo: orange
sdk: streamlit
sdk_version: 1.32.0
app_file: app.py
pinned: false
license: mit
---

# 📰 Sarcasm Headline Detector

Bi-LSTM + GloVe classifier for **news headlines** (sarcastic vs. real news).  
Production Streamlit app with MLOps health monitoring.

## Live demo

| Platform | Action |
|----------|--------|
| **Streamlit Community Cloud** | Deploy from GitHub → [share.streamlit.io](https://share.streamlit.io) |
| **Hugging Face Spaces** | New Space → SDK **Streamlit** → link this repo |

Replace the URLs below after your first deploy:

- Streamlit Cloud: `https://<your-app>.streamlit.app`
- Hugging Face: `https://huggingface.co/spaces/<user>/<space-name>`

## Run locally

```bash
git clone https://github.com/<your-user>/Sarcasm_Detection.git
cd Sarcasm_Detection
pip install -r requirements.txt
streamlit run app.py
```

Required artifacts (must be in the repo for cloud deploy):

```
Sarcasm_Detection/
├── app.py
├── requirements.txt
├── tokenizer_sarcasm.pkl
└── model/
    └── sarcasm_detector.keras   # ~45 MB
```

> Install **only** `tensorflow-cpu`, not the full `tensorflow` package (avoids conflicts and segfaults).

## Deploy on Streamlit Community Cloud (~5 min)

1. Push the project to **GitHub** (include `app.py`, `requirements.txt`, `tokenizer_sarcasm.pkl`, `model/sarcasm_detector.keras`, `monitoring.py`).
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**.
3. Select your repo, branch `main`, **Main file path**: `app.py`.
4. Click **Deploy**. First build may take 5–10 min (TensorFlow CPU install).
5. Copy the public URL (`https://<name>.streamlit.app`).

**Settings checklist**

| Field | Value |
|-------|--------|
| Main file | `app.py` |
| Python version | **3.10** or **3.11** (Advanced settings — required for TensorFlow) |
| Secrets | None required |

**If install fails** (`installer returned a non-zero exit code`):

1. App menu **⋮** → **Settings** → **Advanced** → set **Python 3.10** (not 3.13).
2. Ensure only `requirements.txt` is used (no `pyproject.toml` with old `tensorflow`).
3. **Reboot app** after pushing dependency fixes.

Health monitoring is available in the **sidebar** (JSON payload: version, training date, validation accuracy, asset status).

## Deploy on Hugging Face Spaces

1. [huggingface.co/new-space](https://huggingface.co/new-space) → **Streamlit** SDK.
2. Connect the same GitHub repo (or upload files).
3. Ensure `README.md` contains the YAML frontmatter at the top (already configured in this repo).
4. `app_file` in the frontmatter must be `app.py`.
5. Wait for the build; open the Space URL.

Files over 50 MB total may require [Git LFS](https://git-lfs.github.com/) for `model/sarcasm_detector.keras`.

## MLOps — `/health` endpoint (optional)

Streamlit has no HTTP routes. For a standard REST health check (load balancers, Kubernetes, TP grid “production quality”):

```bash
pip install -r requirements-health.txt
python health_api.py
# GET http://localhost:8080/health
```

Example response:

```json
{
  "status": "ok",
  "model_version": "v1.0",
  "architecture": "Bi-LSTM + GloVe",
  "trained_on": "2026-05-18",
  "accuracy_val": 0.87,
  "assets": {
    "model_file": "sarcasm_detector.keras",
    "model_present": true,
    "tokenizer_present": true
  }
}
```

In the Streamlit UI, the same payload appears under **🩺 System health** in the sidebar.

## Model

| Property | Value |
|----------|--------|
| Architecture | Bi-LSTM (64+32) + GloVe embeddings |
| Vocab size | 20 000 |
| Max sequence length | 40 |
| Validation accuracy | ~0.87 (last training epoch) |
| Training notebook | `notebooks/projet_groupe.ipynb` |

## Project structure

```
├── app.py                  # Streamlit production UI
├── monitoring.py           # Shared health metadata
├── health_api.py           # Optional Flask /health
├── requirements.txt        # Streamlit Cloud dependencies
├── tokenizer_sarcasm.pkl
├── model/
│   └── sarcasm_detector.keras
├── notebooks/
│   └── projet_groupe.ipynb
└── .streamlit/
    └── config.toml
```

## License

MIT
