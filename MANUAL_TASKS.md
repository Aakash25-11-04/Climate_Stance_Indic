# Manual tasks (steps that need your accounts)

These steps need logins to GitHub or Hugging Face, so the pipeline scripts never run them.
Numbered checklist; optional items are marked.

## A. Publish the code on GitHub (required)

1. Create an empty repository on GitHub (e.g. `indic-climate-stance`), **without** a README/licence.
2. In this folder run:
   ```powershell
   git remote add origin https://github.com/<your-user>/indic-climate-stance.git
   git push -u origin main
   ```
3. Add collaborators / contributors in the GitHub repository settings if the challenge needs them.
4. (Optional) enable GitHub Pages or attach `reports/` PDFs to a release.

## B. Publish the model on the Hugging Face Hub (required for the demo Space)

5. Create a free account at https://huggingface.co and run once:
   ```powershell
   .venv\Scripts\activate
   huggingface-cli login
   ```
6. Upload the final model and model card (the script never runs automatically). Check the plan first:
   ```powershell
   python scripts/upload_to_hf.py --repo <your-user>/indic-climate-stance --dry_run
   python scripts/upload_to_hf.py --repo <your-user>/indic-climate-stance
   ```
   Add `--private` for a private repo. If the final model is an ensemble, its transformer member is
   uploaded at the repo root and the full ensemble under `ensemble/`.
7. (Optional) Gradio Space demo:
   ```powershell
   python scripts/prepare_space.py --model_repo <your-user>/indic-climate-stance   # builds space/
   huggingface-cli repo create indic-climate-stance-demo --type space --space_sdk gradio
   cd space
   git init; git remote add origin https://huggingface.co/spaces/<your-user>/indic-climate-stance-demo
   git add .; git commit -m "demo"; git push -u origin main
   ```
   The Space runs the transformer on a free CPU; it does not apply the ensemble or the calibration
   (those are used by `scripts/predict.py --model final` locally). Locally, `python app.py` uses the full final model.

## C. IndicTrans2 instead of NLLB (optional, better translations)

8. Visit https://huggingface.co/ai4bharat/indictrans2-en-indic-dist-200M and
   https://huggingface.co/ai4bharat/indictrans2-indic-en-dist-200M and click "Agree and access".
9. After `huggingface-cli login` (step 5) rebuild the dataset; the translator switches to
   IndicTrans2 automatically (`dataset.translation.backend: auto`):
   ```powershell
   python scripts/build_dataset.py --config configs/dataset.yaml
   ```
   Then re-run `run_all.ps1` to retrain. Expect several hours in total.

## D. Real YouTube comments (optional, strongly recommended for the report)

10. Get a free YouTube Data API v3 key: https://console.cloud.google.com -> create project ->
    "APIs & Services" -> enable "YouTube Data API v3" -> Credentials -> API key.
11. Put it in a `.env` file in the repo root: `YOUTUBE_API_KEY=...` (the file is git-ignored).
12. Collect comments (usernames are hashed, no personal data stored):
    ```powershell
    python scripts/collect_youtube.py --config configs/youtube.yaml
    ```
    Output: `data/raw/youtube/comments_unlabelled.csv` and a labelling template
    `data/gold/test_real_TEMPLATE.csv`.
13. Label ~150 comments following `docs/annotation_guideline.md`, save the file as
    `data/gold/test_real.csv` (columns `id,text,language,label`, optional `annotator_note`).
14. Run `python scripts/evaluate.py --gold data/gold/test_real.csv` to get the model-vs-you
    agreement report (Cohen's kappa, confusion matrix) in `results/real_agreement.json`.

## E. Official challenge data (if/when it is released)

15. Put the official files in `data/official/` (`train.csv`, `test.csv` with columns
    `id,text,label`; `language` optional). Then run `run_all.ps1 -Official`. Official data
    becomes the main train/test set and the built data becomes extra training data + a
    robustness test set (see README "Using official data").

## F. Heavier runs on a free Colab GPU (optional)

16. Open `notebooks/train_on_colab.ipynb` in Google Colab (T4 GPU), upload the repo zip or
    clone your GitHub repo, and run all cells. It trains the same configs with 3 seeds and
    XLM-R; copy the resulting `results/` and `models/` folders back here.
