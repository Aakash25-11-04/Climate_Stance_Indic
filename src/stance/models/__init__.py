"""Model zoo: sparse baselines, dense-embedding baselines, transformers, stacking, ensembles.

Every model exposes the same minimal interface:
    fit(texts: list[str], labels: list[str], languages: list[str] | None = None)
    predict_proba(texts: list[str]) -> np.ndarray [n, 3] in LABELS order
    predict(texts) -> list[str]
"""
