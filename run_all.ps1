# One-command reproduction of the final system (Windows PowerShell).
#   .\run_all.ps1             # full pipeline: dataset -> baselines -> transformers -> ensemble -> final model
#   .\run_all.ps1 -Quick      # smoke run on tiny subsets (checks that everything runs, ~10 min)
#   .\run_all.ps1 -Official   # use data/official/{train,test}.csv as the main data (see README)
#   -Quick writes to the same paths as a full run (data/processed, models/, results/), so it refuses to
#   run where trained models already exist. Use a fresh clone or `git worktree add`, or pass -Force.
param(
    [switch]$Quick,
    [switch]$Official,
    [switch]$SkipDataset,
    [switch]$Force
)
$ErrorActionPreference = "Stop"
if ($Quick -and -not $Force) {
    $trained = Get-ChildItem (Join-Path $PSScriptRoot "models") -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -notin @("final", "efficient") }
    if ($trained) {
        throw "-Quick would overwrite trained models in models/ ($($trained.Count) found). Run it in a fresh clone or a git worktree, or pass -Force."
    }
}
$env:PYTHONUTF8 = "1"
$py = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) {
    Write-Host "Creating .venv ..."
    python -m venv .venv
    & $py -m pip install --upgrade pip
    & $py -m pip install -r requirements.txt
    & $py -m pip install -e .
}
$extra = @()
if ($Quick) { $extra = @("--set", "quick=true") }

function Run($desc, $cmd) {
    Write-Host ""
    Write-Host "==================== $desc ====================" -ForegroundColor Cyan
    # Windows PowerShell 5.1 turns any stderr line of a native command (e.g. tqdm progress bars) into an
    # error record when output is redirected; with "Stop" that silently killed the script. Steps are judged
    # by their exit code instead.
    $ErrorActionPreference = "Continue"
    Invoke-Expression $cmd
    $ErrorActionPreference = "Stop"
    if ($LASTEXITCODE -ne 0) { throw "step failed: $desc" }
}

if (-not $SkipDataset) {
    if ($Quick) {
        Run "Phase 1: dataset (quick)" "& `"$py`" scripts/build_dataset.py --config configs/dataset.yaml --set dataset.translation.max_source_rows=80 dataset.synthetic.n_per_label_lang=40"
    } else {
        Run "Phase 1: dataset" "& `"$py`" scripts/build_dataset.py --config configs/dataset.yaml"
    }
}
if ($Official) {
    Run "Official data merge" "& `"$py`" scripts/prepare_official.py"
}
Run "Phase 7: challenge suite" "& `"$py`" scripts/build_challenge.py"
if ($Quick) {
    Run "Phase 3: baselines (quick)" "& `"$py`" scripts/run_baselines.py --config configs/baselines.yaml --set data.cv_rows=600 cv.n_splits=3 `"models=[majority,lexicon,tfidf_lr,emb_labse_lr,stack_best]`""
    Run "Phase 4: MuRIL pair (quick)" "& `"$py`" scripts/train.py --config configs/transformer_muril_pair.yaml --set data.max_train_rows=300 model.epochs=1"
} else {
    Run "Phase 3: baselines" "& `"$py`" scripts/run_baselines.py --config configs/baselines.yaml"
    Run "Phase 4: transformers" "& `"$py`" scripts/run_transformers.py --config configs/transformers_all.yaml"
}
Run "Phase 8: ensemble + calibration + final model" "& `"$py`" scripts/build_final.py --config configs/final.yaml"
Run "Phase 10: predictions on the gold set with predict.py" "& `"$py`" scripts/predict.py --input data/gold/test_handcrafted.csv --output results/predictions/final_gold_predict.csv --model final"
Write-Host ""
Write-Host "Done. See results/, PROGRESS.md and README.md." -ForegroundColor Green
