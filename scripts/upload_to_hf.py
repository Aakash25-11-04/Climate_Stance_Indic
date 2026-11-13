#!/usr/bin/env python
"""Upload the final model + model card to the Hugging Face Hub. RUN THIS YOURSELF after
`huggingface-cli login` (run it yourself; the pipeline never does).

    python scripts/upload_to_hf.py --repo <user>/indic-climate-stance [--model final|efficient] [--private]

What is uploaded
* the transformer weights + tokenizer + stance_meta.json at the repo root, so
  `TransformerStanceModel.load(snapshot_download(repo))` and the Gradio Space work. If the final
  model is an ensemble, its transformer member is uploaded at the root and the full ensemble
  (ensemble.json + sparse/dense pickles) under `ensemble/`.
* MODEL.json (the final spec incl. calibration), README.md (= models/MODEL_CARD.md),
  results_final.json and results_tables.md.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def transformer_dir(spec: dict) -> Path | None:
    t = spec.get("type")
    if t in ("transformer", "quantized"):
        return ROOT / spec["path"]
    if t == "ensemble":
        ens = json.loads((ROOT / spec["path"] / "ensemble.json").read_text(encoding="utf-8"))
        for m in ens["members"]:
            if m["type"] == "transformer":
                return ROOT / m["path"]
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="e.g. your-user/indic-climate-stance")
    ap.add_argument("--model", default="final", choices=["final", "efficient"])
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--dry_run", action="store_true", help="print what would be uploaded, upload nothing")
    args = ap.parse_args()
    spec_path = ROOT / "models" / args.model / "MODEL.json"
    if not spec_path.exists():
        sys.exit(f"{spec_path} not found: run scripts/build_final.py first")
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    tdir = transformer_dir(spec)
    plan: list[tuple[Path, str]] = []
    if tdir is not None:
        for f in tdir.iterdir():
            if f.is_file():
                plan.append((f, f.name))
    if spec.get("type") == "ensemble":
        # Every member is uploaded; the uploaded ensemble.json uses paths relative to the Hub repo root,
        # so `EnsembleModel.from_dir(snapshot_download(repo))` works on any machine.
        ens = json.loads((ROOT / spec["path"] / "ensemble.json").read_text(encoding="utf-8"))
        hub_members = []
        for m in ens["members"]:
            src = ROOT / m["path"]
            if m["type"] == "transformer" and tdir is not None and src.resolve() == tdir.resolve():
                hub_path = "."  # already uploaded at the repo root
            elif m["type"] == "transformer":
                hub_path = f"ensemble/{src.name}"
                plan.extend((f, f"{hub_path}/{f.name}") for f in src.iterdir() if f.is_file())
            else:
                hub_path = f"ensemble/{src.name}"
                plan.append((src, hub_path))
            hub_members.append({**m, "path": hub_path})
        hub_spec = ROOT / "results" / "hub_ensemble.json"
        hub_spec.write_text(json.dumps({**ens, "members": hub_members}, indent=2), encoding="utf-8")
        plan.append((hub_spec, "ensemble/ensemble.json"))
    if spec.get("type") == "hybrid":
        plan.append((ROOT / spec["path"], Path(spec["path"]).name))
    plan.append((spec_path, "MODEL.json"))
    for src, dst in ((ROOT / "models" / "MODEL_CARD.md", "README.md"), (ROOT / "results" / "final.json", "results_final.json"),
                     (ROOT / "results" / "tables.md", "results_tables.md")):
        if src.exists():
            plan.append((src, dst))
    total = sum(p.stat().st_size for p, _ in plan) / 1e6
    for p, d in plan:
        print(f"  {d:45s} <- {p.relative_to(ROOT)}")
    print(f"total {total:.0f} MB -> {args.repo}")
    if args.dry_run:
        return
    from huggingface_hub import HfApi, get_token

    if not get_token():
        sys.exit("No Hugging Face token found. Run `huggingface-cli login` first (MANUAL_TASKS.md, step 5).")
    api = HfApi()
    api.create_repo(args.repo, private=args.private, exist_ok=True, repo_type="model")
    for p, d in plan:
        api.upload_file(repo_id=args.repo, path_or_fileobj=str(p), path_in_repo=d)
    print(f"done: https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
