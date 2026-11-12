# Unix equivalent of run_all.ps1. `make all` reproduces the final model end to end.
PY ?= .venv/bin/python
export PYTHONUTF8=1

.PHONY: all venv dataset challenge baselines transformers final predict quick eda clean

all: dataset challenge baselines transformers final predict

venv:
	python -m venv .venv && $(PY) -m pip install --upgrade pip && $(PY) -m pip install -r requirements.txt && $(PY) -m pip install -e .

dataset:
	$(PY) scripts/build_dataset.py --config configs/dataset.yaml

challenge:
	$(PY) scripts/build_challenge.py

baselines:
	$(PY) scripts/run_baselines.py --config configs/baselines.yaml

transformers:
	$(PY) scripts/run_transformers.py --config configs/transformers_all.yaml

final:
	$(PY) scripts/build_final.py --config configs/final.yaml

predict:
	$(PY) scripts/predict.py --input data/gold/test_handcrafted.csv --output results/predictions/final_gold_predict.csv --model final

eda:
	$(PY) scripts/eda.py

quick:
	$(PY) scripts/build_dataset.py --config configs/dataset.yaml --set dataset.translation.max_source_rows=80 dataset.synthetic.n_per_label_lang=40
	$(PY) scripts/build_challenge.py
	$(PY) scripts/run_baselines.py --config configs/baselines.yaml --set data.cv_rows=600 cv.n_splits=3 "models=[majority,lexicon,tfidf_lr,emb_labse_lr,stack_best]"
	$(PY) scripts/train.py --config configs/transformer_muril_pair.yaml --set data.max_train_rows=300 model.epochs=1

clean:
	rm -rf results/oof results/predictions/*.csv models/*_s*
