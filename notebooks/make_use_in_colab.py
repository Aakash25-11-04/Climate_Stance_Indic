"""Writes notebooks/use_in_colab.ipynb (run: python notebooks/make_use_in_colab.py)."""
import nbformat as nbf

REPO_GH = "https://github.com/Aakash25-11-04/Climate_Stance_Indic.git"
REPO_HF = "Aakash25-11-04/Climate_Stance_Indic"
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

cells = [
    md(f"""# Indic Climate Stance: use the model in Google Colab

Labels Hindi and Bengali comments (native script, Romanized or code-mixed) as **Favour**, **Against** or **None**
toward *"Climate change and global warming is a serious concern."*

* Code: {REPO_GH.removesuffix('.git')}
* Model: https://huggingface.co/{REPO_HF}

**Run the cells from top to bottom** (Shift+Enter). A GPU is optional: *Runtime → Change runtime type → T4 GPU*
makes it faster, but CPU works too. The first run downloads about 4 GB (the model plus the BGE-M3 encoder)."""),
    md("## 1. Get the code and install packages (about 2–3 minutes)\n"
       "Colab already has PyTorch, so it is not reinstalled. If Colab says *\"restart session\"*, click it and "
       "continue with step 2; the files stay."),
    code(f"""import os
if not os.path.isdir("/content/Climate_Stance_Indic"):
    !git clone -q {REPO_GH} /content/Climate_Stance_Indic
%cd /content/Climate_Stance_Indic
# scikit-learn is pinned to the version the saved models were made with
!pip install -q scikit-learn==1.9.1 transformers==4.57.6 sentence-transformers==5.7.0 huggingface_hub==0.36.2 \\
    sentencepiece safetensors accelerate pyyaml tqdm indic-nlp-library indic-transliteration emoji regex \\
    rapidfuzz datasketch captum lime gradio
!pip install -q -e . --no-deps
print("setup done")"""),
    md("""## 2. Log in to Hugging Face (only while the model is private)

Either add a Colab **secret** named `HF_TOKEN` (key icon in the left sidebar → *Add new secret* → paste a **Read**
token from https://huggingface.co/settings/tokens → switch on *Notebook access*), or paste the token into the
box that appears below. Skip this cell if the model is public."""),
    code("""from huggingface_hub import whoami, notebook_login
try:
    print("Logged in as", whoami()["name"])
except Exception:
    notebook_login()"""),
    md("## 3. Load the model (first time: a few minutes of downloading)"),
    code(f"""import os, sys
os.chdir("/content/Climate_Stance_Indic")
os.environ["PYTHONUTF8"] = "1"
for p in ("scripts", "src"):
    if p not in sys.path:
        sys.path.insert(0, p)

import pandas as pd
from predict import FinalModel, guess_language
from stance.utils import LABELS

REPO = "{REPO_HF}"
model = FinalModel("final", hub=REPO)   # the full final model: average of 4 members

def stance(texts, languages=None):
    \"\"\"Label one comment or a list of comments; returns a table with probabilities.\"\"\"
    if isinstance(texts, str):
        texts = [texts]
    languages = languages or [guess_language(t) for t in texts]
    probs = model.predict_proba(texts, languages)
    out = pd.DataFrame({{"text": texts, "language": languages,
                        "stance": [LABELS[i] for i in probs.argmax(1)]}})
    for i, label in enumerate(LABELS):
        out[f"p_{{label}}"] = probs[:, i].round(3)
    return out

print("model ready:", model.spec.get("candidate", model.spec["type"]))"""),
    md("## 4. Try it"),
    code("""stance([
    "जलवायु परिवर्तन एक गंभीर समस्या है, हमें अभी कदम उठाने चाहिए।",
    "ग्लोबल वार्मिंग सब झूठ है भाई, पैसा कमाने का धंधा है",
    "কাল ঢাকায় হালকা বৃষ্টি হবে বলে আবহাওয়া অফিস জানিয়েছে।",
    "climate change real hai yaar, delhi me april me hi 45 degree ho gaya",
    "global warming shob mithye kotha",
])"""),
    md("Type your own comment here:"),
    code("""stance("यहाँ अपना कमेंट लिखें")"""),
    md("""## 5. Label a whole CSV file

The file needs a `text` column (an `id` column and a `language` column with `hi`/`bn` are optional).
Run the cell, choose the file, and the results download automatically as `<name>_result.csv`."""),
    code("""from google.colab import files
uploaded = files.upload()
for name in uploaded:
    df = pd.read_csv(name, keep_default_na=False)
    if "id" not in df.columns:
        df.insert(0, "id", [f"row_{i}" for i in range(len(df))])
    langs = df["language"].tolist() if "language" in df.columns else None
    res = stance(df["text"].astype(str).tolist(), langs)
    res.insert(0, "id", df["id"].values)
    out = name.rsplit(".", 1)[0] + "_result.csv"
    res.to_csv(out, index=False, encoding="utf-8-sig")   # utf-8-sig: opens correctly in Excel
    print(out, res["stance"].value_counts().to_dict())
    files.download(out)"""),
    md("""## 6. (Optional) Web demo with a public link

Starts the Gradio app and prints a `https://….gradio.live` link you can open on any device (valid about 72 hours
while this notebook runs). Stop it with the ■ button."""),
    code(f"""!GRADIO_SHARE=1 STANCE_MODEL_REPO={REPO_HF} PYTHONUTF8=1 python app.py"""),
]

nb = nbf.v4.new_notebook(cells=cells, metadata={
    "accelerator": "GPU", "colab": {"provenance": []},
    "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}})
nbf.write(nb, "notebooks/use_in_colab.ipynb")
print("wrote notebooks/use_in_colab.ipynb")
