"""Loaders for every public source used to build the dataset.

Each loader returns a DataFrame with the columns
    text, label, language, source, source_type, orig_label, split_hint
where `label` is already mapped to our scheme {Favour, Against, None}.

Label-mapping table (also reproduced in README.md / the datasheet)
------------------------------------------------------------------
| source                       | original label        | ours    | justification |
|------------------------------|-----------------------|---------|---------------|
| tweet_eval / stance_climate  | favor                 | Favour  | SemEval-2016 target "Climate Change is a Real Concern" == our claim |
|                              | against               | Against | same target |
|                              | none                  | None    | same target |
| GWSD (Luo et al. 2020)       | agrees (worker vote)  | Favour  | target "climate change/global warming is a serious concern" == our claim |
|                              | disagrees             | Against | |
|                              | neutral               | None    | |
| ClimateStance (Vaid 2022)    | 1 (favor)             | Favour  | stance toward the reality/urgency of climate change |
|                              | 2 (against / denial)  | Against | |
|                              | 0 (neutral/ambiguous) | None    | |
| Dhoroni (Wasi et al. 2024)   | Neutral               | None    | neutral climate/environment news headline == topic-related but no stance |
|                              | Positive, Focus in {Climate Change, Global Warming} | Favour | article affirms climate change as a concern |
|                              | Positive, other focus | dropped | pollution/plastic news: pro-environment but not about our claim |
|                              | Negative, Problematic | dropped | "Negative" = bad environmental news, NOT denial; would mislabel Against |

Licences / citations
--------------------
* tweet_eval: Barbieri et al. 2020 (TweetEval), stance subset from Mohammad et al. 2016
  (SemEval-2016 Task 6). Distributed for research use on the Hugging Face Hub.
* GWSD: Luo, Card & Jurafsky 2020, Findings of EMNLP. GitHub yiweiluo/GWStance (no explicit
  licence file in the repo; research use, cite the paper).
* ClimateStance: Vaid, Pant & Shrivastava 2022, ACL SRW. Hugging Face eci-io/climate-evaluation
  (dataset card lists research use; cite the paper).
* Dhoroni: Wasi et al. 2024 (arXiv 2410.17225). Hugging Face ciol-research/Dhoroni, MIT licence.
"""
from __future__ import annotations

import io
import zipfile
from collections import Counter
from pathlib import Path

import pandas as pd
import requests

from stance.utils import abs_path

RAW = abs_path("data/raw")
HEADERS = {"User-Agent": "indic-climate-stance/0.1 (research; contact via repo)"}

SOURCE_INFO = {
    "tweet_eval_climate": {
        "citation": "Mohammad et al. 2016 (SemEval-2016 Task 6); Barbieri et al. 2020 (TweetEval)",
        "url": "https://huggingface.co/datasets/cardiffnlp/tweet_eval (config stance_climate)",
        "licence": "research use (see dataset card)",
        "language": "en",
    },
    "gwsd": {
        "citation": "Luo, Card & Jurafsky 2020, Detecting Stance in Media on Global Warming, Findings of EMNLP",
        "url": "https://github.com/yiweiluo/GWStance (GWSD.tsv)",
        "licence": "no licence file; research use, cite the paper",
        "language": "en",
    },
    "climatestance": {
        "citation": "Vaid, Pant & Shrivastava 2022, ACL Student Research Workshop",
        "url": "https://huggingface.co/datasets/eci-io/climate-evaluation (ClimateStance/)",
        "licence": "research use (see dataset card)",
        "language": "en",
    },
    "dhoroni": {
        "citation": "Wasi et al. 2024, Dhoroni, arXiv:2410.17225",
        "url": "https://huggingface.co/datasets/ciol-research/Dhoroni",
        "licence": "MIT",
        "language": "bn",
    },
}


def _download(url: str, dest: Path, binary: bool = False) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    r = requests.get(url, headers=HEADERS, timeout=120, allow_redirects=True)
    r.raise_for_status()
    if binary:
        dest.write_bytes(r.content)
    else:
        dest.write_text(r.text, encoding="utf-8")
    return dest


def _frame(texts, labels, orig, source, language, split_hint) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "text": list(texts),
            "label": list(labels),
            "orig_label": list(orig),
            "source": source,
            "language": language,
            "source_type": "original",
            "split_hint": list(split_hint) if not isinstance(split_hint, str) else split_hint,
        }
    )
    df = df[df["label"].isin(["Favour", "Against", "None"])]
    df["text"] = df["text"].astype(str).str.strip()
    return df[df["text"].str.len() > 0].reset_index(drop=True)


# ----------------------------------------------------------------------- SemEval 2016 climate
def load_tweet_eval_climate() -> pd.DataFrame:
    from datasets import load_dataset

    ds = load_dataset("cardiffnlp/tweet_eval", "stance_climate")
    names = ds["train"].features["label"].names  # ['none', 'against', 'favor']
    mapping = {"none": "None", "against": "Against", "favor": "Favour"}
    frames = []
    for split in ("train", "validation", "test"):
        d = ds[split]
        orig = [names[i] for i in d["label"]]
        frames.append(_frame(d["text"], [mapping[o] for o in orig], orig, "tweet_eval_climate", "en", split))
    return pd.concat(frames, ignore_index=True)


# ----------------------------------------------------------------------- GWSD
GWSD_URL = "https://raw.githubusercontent.com/yiweiluo/GWStance/master/GWSD.tsv"


def load_gwsd(min_agreement: float = 0.5) -> pd.DataFrame:
    """GWSD: 8 worker votes per sentence (agrees/disagrees/neutral). We take the majority
    vote and keep the row only if the majority share >= min_agreement (default: strictly
    more than half of the valid votes)."""
    path = _download(GWSD_URL, RAW / "gwsd" / "GWSD.tsv")
    df = pd.read_csv(path, sep="\t")
    worker_cols = [c for c in df.columns if c.startswith("worker_")]
    mapping = {"agrees": "Favour", "disagrees": "Against", "neutral": "None"}
    texts, labels, origs, agree_share, hints = [], [], [], [], []
    for _, row in df.iterrows():
        votes = [row[c] for c in worker_cols if isinstance(row[c], str) and row[c] in mapping]
        if not votes:
            continue
        top, n = Counter(votes).most_common(1)[0]
        share = n / len(votes)
        if share < min_agreement:
            continue
        texts.append(row["sentence"])
        labels.append(mapping[top])
        origs.append(top)
        agree_share.append(share)
        hints.append("test" if bool(row.get("in_held_out_test", False)) else "train")
    out = _frame(texts, labels, origs, "gwsd", "en", hints)
    out["annotator_agreement"] = agree_share[: len(out)]
    return out


# ----------------------------------------------------------------------- ClimateStance
CS_BASE = "https://huggingface.co/datasets/eci-io/climate-evaluation/resolve/main/ClimateStance/"


def load_climatestance() -> pd.DataFrame:
    """ClimateStance (Vaid et al. 2022): tweets around COP25. Labels 0 = neutral/ambiguous,
    1 = favor (believes/supports action on climate change), 2 = against (denial)."""
    mapping = {0: "None", 1: "Favour", 2: "Against"}
    frames = []
    for split in ("train", "val", "test"):
        path = _download(CS_BASE + f"{split}.csv", RAW / "climatestance" / f"{split}.csv")
        d = pd.read_csv(path)
        d = d.dropna(subset=["text", "label"])
        d["label_int"] = pd.to_numeric(d["label"], errors="coerce")
        d = d.dropna(subset=["label_int"])
        d["label_int"] = d["label_int"].astype(int)
        d = d[d["label_int"].isin(mapping)]
        frames.append(
            _frame(d["text"], d["label_int"].map(mapping), d["label_int"].astype(str),
                   "climatestance", "en", "dev" if split == "val" else split)
        )
    return pd.concat(frames, ignore_index=True)


# ----------------------------------------------------------------------- Dhoroni
DHORONI_CSV = "https://huggingface.co/datasets/ciol-research/Dhoroni/resolve/main/dataset.csv"
DHORONI_ZIP = "https://huggingface.co/datasets/ciol-research/Dhoroni/resolve/main/dataset_files.zip"
CLIMATE_FOCUS = {"Climate Change", "Global Warming"}


def load_dhoroni(include_first_sentence: bool = True, max_chars: int = 220) -> pd.DataFrame:
    """Dhoroni: Bengali news articles annotated at article level. We use the headline (and,
    optionally, the first sentence of the article) as short texts.

    Mapping (see module docstring): Neutral -> None; Positive with a climate focus -> Favour;
    everything else dropped because the "Negative" label means negative environmental news,
    not disagreement with the claim.
    """
    csv_path = _download(DHORONI_CSV, RAW / "dhoroni" / "dataset.csv")
    meta = pd.read_csv(csv_path)
    first_sent: dict[str, str] = {}
    if include_first_sentence:
        zip_path = _download(DHORONI_ZIP, RAW / "dhoroni" / "dataset_files.zip", binary=True)
        with zipfile.ZipFile(zip_path) as z:
            for name in z.namelist():
                if not name.endswith(".txt"):
                    continue
                body = z.read(name).decode("utf-8", "replace").strip()
                sent = body.split("।")[0].strip()  # Bengali danda
                if 15 <= len(sent) <= max_chars:
                    first_sent[Path(name).name] = sent + "।"
    texts, labels, origs = [], [], []
    for _, row in meta.iterrows():
        stance = str(row.get("Stance Detection", "")).strip()
        focus = str(row.get("Focus", "")).strip()
        if stance == "Neutral":
            lab = "None"
        elif stance == "Positive" and focus in CLIMATE_FOCUS:
            lab = "Favour"
        else:
            continue
        title = str(row["Title"]).strip()
        texts.append(title)
        labels.append(lab)
        origs.append(f"{stance}/{focus}")
        fs = first_sent.get(str(row.get("File", "")))
        if fs:
            texts.append(fs)
            labels.append(lab)
            origs.append(f"{stance}/{focus}/first_sentence")
    return _frame(texts, labels, origs, "dhoroni", "bn", "train")


LOADERS = {
    "tweet_eval_climate": load_tweet_eval_climate,
    "gwsd": load_gwsd,
    "climatestance": load_climatestance,
    "dhoroni": load_dhoroni,
}


def load_all_sources(names: list[str] | None = None, log=print) -> pd.DataFrame:
    frames = []
    for name in names or LOADERS:
        try:
            df = LOADERS[name]()
            log(f"[loaders] {name}: {len(df)} rows, labels={df['label'].value_counts().to_dict()}")
            frames.append(df)
        except Exception as e:  # noqa: BLE001 - a failed source must not kill the build
            log(f"[loaders] FAILED {name}: {type(e).__name__}: {e}")
    if not frames:
        raise RuntimeError("no source could be loaded")
    return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    df = load_all_sources()
    print(df.groupby(["source", "label"]).size())
