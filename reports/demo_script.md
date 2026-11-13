# Live demo script (about 5 minutes)

**Setup before the session.** Run `python app.py` (opens http://127.0.0.1:7860) and keep a
terminal open in the repo. If the GPU is busy or absent the app runs on CPU (about 1 s per
comment; explanations take 3-10 s). Fallback if the app fails: `python scripts/predict.py --input
reports/demo_inputs.csv --output demo_out.csv --model final` and show the CSV.

| # | Input (paste into the text box) | Expected | What to say |
|---|---|---|---|
| 1 | जलवायु परिवर्तन एक गंभीर समस्या है, हमें अभी कदम उठाने चाहिए। | Favour | Plain Hindi. Point at the probabilities and at the highlighted words "गंभीर समस्या" (Integrated Gradients). |
| 2 | ग्लोबल वार्मिंग सब झूठ है भाई, पैसा कमाने का धंधा है | Against | Denial plus money-scheme framing, the most common Against pattern. |
| 3 | কাল ঢাকায় হালকা বৃষ্টি হবে বলে আবহাওয়া অফিস জানিয়েছে। | None | Bengali weather news: topic-related but no stance. This is the classic trap for keyword models. |
| 4 | climate change real hai yaar, delhi me april me hi 45 degree ho gaya | Favour | Romanized Hindi (Hinglish). MuRIL saw transliterated text in pre-training. |
| 5 | ये कहना गलत है कि जलवायु परिवर्तन एक झूठ है। | Favour (**model says Against, 0.60**) | Known failure, shown on purpose. Negation of denial ("it is wrong to say climate change is a lie"). Switch to LIME: "झूठ" (lie) is the strongest push towards Against, and "गलत" (wrong) does not push towards Favour at all (checked: LIME weight +0.10 towards Against). The model has not learned that "it is wrong to say" reverses the claim. This is our largest error group (7 of 21 Favour→Against errors on gold); see `reports/discussion.md` section 7. |
| 6 | हाँ हाँ, धरती गर्म हो रही है, इसलिए दिसंबर में बर्फ पड़ रही है 🙄 | Against | Sarcasm. The final model gets it right but only just (Against 0.42 vs Favour 0.38). Sarcasm is one of our weakest categories (challenge failure rate 0.40). |
| 7 | 👍👍👍 | None | Emoji-only input does not crash and gives a low-confidence None. |
| 8 | Ask a judge for a sentence | ? | Let them type. If it fails, open `results/tables.md` and say which category it belongs to. |

**Closing (30 s).** Show `results/tables.md`: the baseline to beat, the final model, and the challenge
failure rates per category. Point out that every number comes from a file in `results/` and that
`run_all.ps1` reproduces it.
