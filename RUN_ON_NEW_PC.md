# Running the project on another computer

The code is on GitHub and the trained model is on Hugging Face, so a new computer needs nothing
from this laptop.

- Code: https://github.com/Aakash25-11-04/Climate_Stance_Indic
- Model: https://huggingface.co/Aakash25-11-04/Climate_Stance_Indic (private: log in first, or make it public)

## What the computer needs

| | Minimum | Notes |
|---|---|---|
| OS | Windows 10/11, Linux or macOS | commands below are for Windows PowerShell; Linux/macOS differences are noted |
| RAM | 8 GB | 16 GB is more comfortable |
| Disk | about 10 GB free | Python packages + 1.9 GB model + 2.3 GB BGE-M3 encoder |
| GPU | **not required** | CPU works (about 0.2 s per comment); an NVIDIA GPU makes it faster |
| Internet | yes, for the first run | downloads about 4–5 GB once, then everything is cached |

## Step 1. Install Python and Git (once)

- **Python 3.11**: https://www.python.org/downloads/ (tick **"Add Python to PATH"** during setup)
- **Git**: https://git-scm.com/downloads

Check in a new terminal:

```powershell
python --version    # Python 3.11.x
git --version
```

## Step 2. Download the code

```powershell
cd $HOME\Desktop
git clone https://github.com/Aakash25-11-04/Climate_Stance_Indic.git
cd Climate_Stance_Indic
```

## Step 3. Create a virtual environment and install packages

```powershell
python -m venv .venv
.venv\Scripts\activate          # Linux/macOS: source .venv/bin/activate
python -m pip install --upgrade pip
```

Install PyTorch. Pick **one** line:

```powershell
# NVIDIA GPU (CUDA 12.6):
pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu126
# No NVIDIA GPU (CPU only, also for macOS):
pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
```

Then the rest of the project:

```powershell
pip install -r requirements.txt
pip install -e .
```

This takes 5–15 minutes. If PowerShell refuses to run `activate` ("running scripts is disabled"), run
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, then try again.

## Step 4. Log in to Hugging Face (only while the model is private)

```powershell
hf auth login
```

Paste a token from https://huggingface.co/settings/tokens (a **Read** token is enough for downloading).
Skip this step if you made the model public.

## Step 5. Label comments

Put your comments in a CSV file with columns `id,text` (like `test_text_only.csv`), then:

```powershell
$env:PYTHONUTF8 = "1"
python scripts/predict.py --input my_comments.csv --output my_results.csv --hub Aakash25-11-04/Climate_Stance_Indic
```

- The first run downloads the model (about 1.9 GB) and the BGE-M3 encoder (about 2.3 GB). Later runs start in seconds.
- Output columns: `id, label, p_Favour, p_Against, p_None`.
- Optional: add a `language` column (`hi` or `bn`). Without it, the language is guessed from the script and words.
- Force CPU with `--device cpu`.

## Step 6 (optional). The demo web app

```powershell
$env:PYTHONUTF8 = "1"
$env:STANCE_MODEL_REPO = "Aakash25-11-04/Climate_Stance_Indic"
python app.py
```

Open http://127.0.0.1:7860 in a browser, type a Hindi or Bengali comment, and see the stance with the
words that influenced it.

## Use it from your own Python code

```python
from huggingface_hub import snapshot_download
from stance.models.ensemble import EnsembleModel

model = EnsembleModel.from_dir(snapshot_download("Aakash25-11-04/Climate_Stance_Indic"))
probs = model.predict_proba(["जलवायु परिवर्तन एक गंभीर समस्या है"], ["hi"])  # columns: Favour, Against, None
```

## Google Colab (no installation on your computer)

Open the ready-made notebook in Colab and run the cells top to bottom:

https://colab.research.google.com/github/Aakash25-11-04/Climate_Stance_Indic/blob/main/notebooks/use_in_colab.ipynb

It clones this repo, installs the packages, loads the model from Hugging Face, labels example comments or an
uploaded CSV, and can start the web demo with a public link. While the model is private, add your token as a Colab
secret named `HF_TOKEN` (key icon in the left sidebar) or paste it when asked.

## Retraining everything from scratch (optional, needs an NVIDIA GPU)

```powershell
.\run_all.ps1           # full pipeline: about a day on a 4 GB GPU
.\run_all.ps1 -Quick    # 10–20 minute smoke test on tiny subsets
```

## Common problems

| Problem | Fix |
|---|---|
| `python` not found | reinstall Python with "Add Python to PATH", then open a new terminal |
| `401` / "Repository not found" when downloading | the model is private: run `hf auth login`, or make the repo public on Hugging Face |
| Garbled Hindi/Bengali text or `UnicodeEncodeError` | run `$env:PYTHONUTF8 = "1"` first |
| `CUDA out of memory` | add `--device cpu` |
| Very slow first run | it is downloading about 4 GB once; later runs are fast |
