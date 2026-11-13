// IEEE-style two-column technical report (US Letter, Times New Roman 10 pt).
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType, Table, TableRow, TableCell,
  WidthType, BorderStyle, ImageRun, TableOfContents, SectionType, Tab, TabStopType,
  Math: OMath, MathRun, MathFraction, MathSum, MathSubScript, MathSuperScript, MathSubSuperScript,
  PageNumber, Footer, VerticalAlign,
} = require("docx");

const FIG = path.join(__dirname, "fig");
const OUT = process.argv[2];
const FONT = "Times New Roman";
const COLW = 5040; // one column, DXA (3.5 in)

// ------------------------------------------------------------------ citations (numbered by first use)
const REFS = {
  mohammad2016: "S. M. Mohammad, S. Kiritchenko, P. Sobhani, X. Zhu, and C. Cherry, “SemEval-2016 Task 6: Detecting stance in tweets,” in Proc. SemEval, 2016.",
  barbieri2020: "F. Barbieri, J. Camacho-Collados, L. Espinosa Anke, and L. Neves, “TweetEval: Unified benchmark and comparative evaluation for tweet classification,” in Findings of EMNLP, 2020.",
  mohammad2017: "S. M. Mohammad, P. Sobhani, and S. Kiritchenko, “Stance and sentiment in tweets,” ACM Trans. Internet Technol., 2017.",
  luo2020: "Y. Luo, D. Card, and D. Jurafsky, “Detecting stance in media on global warming,” in Findings of EMNLP, 2020.",
  vaid2022: "R. Vaid, K. Pant, and M. Shrivastava, “Towards fine-grained classification of climate change related social media text,” in Proc. ACL Student Research Workshop, 2022.",
  wasi2024: "A. T. Wasi et al., “Dhoroni” (Bengali climate and environmental news dataset), arXiv:2410.17225, 2024.",
  kucuk2020: "D. Küçük and F. Can, “Stance detection: A survey,” ACM Comput. Surv., 2020.",
  aldayel2021: "A. ALDayel and W. Magdy, “Stance detection on social media: State of the art and trends,” Inf. Process. Manage., 2021.",
  augenstein2016: "I. Augenstein, T. Rocktäschel, A. Vlachos, and K. Bontcheva, “Stance detection with bidirectional conditional encoding,” in Proc. EMNLP, 2016.",
  du2017: "J. Du, R. Xu, Y. He, and L. Gui, “Stance classification with target-specific neural attention networks,” in Proc. IJCAI, 2017.",
  allaway2020: "E. Allaway and K. McKeown, “Zero-shot stance detection: A dataset and model using generalized topic representations,” in Proc. EMNLP, 2020.",
  hardalov2022: "M. Hardalov, A. Arora, P. Nakov, and I. Augenstein, “Few-shot cross-lingual stance detection with sentiment-based pre-training,” in Proc. AAAI, 2022.",
  yin2019: "W. Yin, J. Hay, and D. Roth, “Benchmarking zero-shot text classification: Datasets, evaluation and entailment approach,” in Proc. EMNLP, 2019.",
  laurer2024: "M. Laurer, W. van Atteveldt, A. Casas, and K. Welbers, “Less annotating, more classifying,” Political Analysis, 2024.",
  khanuja2021: "S. Khanuja et al., “MuRIL: Multilingual representations for Indian languages,” arXiv:2103.10730, 2021.",
  doddapaneni2023: "S. Doddapaneni et al., “Towards leaving no Indic language behind: Building monolingual corpora, benchmark and models for Indic languages,” in Proc. ACL, 2023.",
  bhattacharjee2022: "A. Bhattacharjee et al., “BanglaBERT: Language model pretraining and benchmarks for low-resource language understanding evaluation in Bangla,” in Findings of NAACL, 2022.",
  conneau2020: "A. Conneau et al., “Unsupervised cross-lingual representation learning at scale,” in Proc. ACL, 2020.",
  devlin2019: "J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, “BERT: Pre-training of deep bidirectional transformers for language understanding,” in Proc. NAACL, 2019.",
  feng2022: "F. Feng, Y. Yang, D. Cer, N. Arivazhagan, and W. Wang, “Language-agnostic BERT sentence embedding,” in Proc. ACL, 2022.",
  wang2024: "L. Wang et al., “Multilingual E5 text embeddings: A technical report,” arXiv:2402.05672, 2024.",
  chen2024: "J. Chen et al., “BGE M3-Embedding,” arXiv:2402.03216, 2024.",
  nllb2022: "NLLB Team, M. R. Costa-jussà et al., “No language left behind: Scaling human-centered machine translation,” arXiv:2207.04672, 2022.",
  gala2023: "J. Gala et al., “IndicTrans2: Towards high-quality and accessible machine translation models for all 22 scheduled Indian languages,” Trans. Mach. Learn. Res., 2023.",
  ribeiro2020: "M. T. Ribeiro, T. Wu, C. Guestrin, and S. Singh, “Beyond accuracy: Behavioral testing of NLP models with CheckList,” in Proc. ACL, 2020.",
  gardner2020: "M. Gardner et al., “Evaluating models’ local decision boundaries via contrast sets,” in Findings of EMNLP, 2020.",
  sennrich2016: "R. Sennrich, B. Haddow, and A. Birch, “Improving neural machine translation models with monolingual data,” in Proc. ACL, 2016.",
  khanuja2020: "S. Khanuja, S. Dandapat, A. Srinivasan, S. Sitaram, and M. Choudhury, “GLUECoS: An evaluation benchmark for code-switched NLP,” in Proc. ACL, 2020.",
  wangmanning2012: "S. Wang and C. D. Manning, “Baselines and bigrams: Simple, good sentiment and topic classification,” in Proc. ACL, 2012.",
  reimers2019: "N. Reimers and I. Gurevych, “Sentence-BERT: Sentence embeddings using Siamese BERT-networks,” in Proc. EMNLP, 2019.",
  lin2017: "T.-Y. Lin, P. Goyal, R. Girshick, K. He, and P. Dollár, “Focal loss for dense object detection,” in Proc. ICCV, 2017.",
  qwen2024: "Qwen Team, “Qwen2.5 technical report,” arXiv, 2024.",
  guo2017: "C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger, “On calibration of modern neural networks,” in Proc. ICML, 2017.",
  efron1993: "B. Efron and R. J. Tibshirani, An Introduction to the Bootstrap. New York, NY, USA: Chapman & Hall, 1993.",
  mosbach2021: "M. Mosbach, M. Andriushchenko, and D. Klakow, “On the stability of fine-tuning BERT: Misconceptions, explanations, and strong baselines,” in Proc. ICLR, 2021.",
  bondarenko2021: "Y. Bondarenko, M. Nagel, and T. Blankevoort, “Understanding and overcoming the challenges of efficient transformer quantization,” in Proc. EMNLP, 2021.",
  ribeiro2016: "M. T. Ribeiro, S. Singh, and C. Guestrin, “‘Why should I trust you?’: Explaining the predictions of any classifier,” in Proc. KDD, 2016.",
};
const order = [];
function cite(keys) {
  return "[" + keys.split(",").map((k) => {
    k = k.trim();
    if (!REFS[k]) throw new Error("unknown ref " + k);
    if (!order.includes(k)) order.push(k);
    return order.indexOf(k) + 1;
  }).join("], [") + "]";
}

// ------------------------------------------------------------------ inline markup: **bold**, *italic*, [[cite]]
function runs(text, base = {}) {
  const out = [];
  for (const part of text.split(/(\*\*[^*]+\*\*|\*[^*]+\*|\[\[[^\]]+\]\]|_\{[^}]+\})/)) {
    if (!part) continue;
    if (part.startsWith("_{")) { out.push(new TextRun({ text: part.slice(2, -1), subScript: true, italics: true, ...base })); continue; }
    if (part.startsWith("**")) out.push(new TextRun({ text: part.slice(2, -2), bold: true, ...base }));
    else if (part.startsWith("[[")) out.push(new TextRun({ text: cite(part.slice(2, -2)), ...base }));
    else if (part.startsWith("*")) out.push(new TextRun({ text: part.slice(1, -1), italics: true, ...base }));
    else out.push(new TextRun({ text: part, ...base }));
  }
  return out;
}
const P = (text, opts = {}) => new Paragraph({ style: "Body", children: runs(text), ...opts });
const PN = (text) => P(text, { indent: { firstLine: 0 } }); // no indent (after equations)
const H1 = (text) => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun(text)] });
const H2 = (text) => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun(text)] });
const ITEM = (label, text) => new Paragraph({ style: "Body", indent: { left: 288, hanging: 288, firstLine: 0 },
  children: [new TextRun(label + " "), ...runs(text)] });

// ------------------------------------------------------------------ equations
let eqNo = 0;
const R = (t) => new MathRun(t);
function EQ(parts) {
  eqNo += 1;
  return new Paragraph({
    style: "Body", indent: { firstLine: 0 }, spacing: { before: 60, after: 60 },
    tabStops: [{ type: TabStopType.CENTER, position: COLW / 2 }, { type: TabStopType.RIGHT, position: COLW }],
    children: [new TextRun({ children: [new Tab()] }), new OMath({ children: parts }), new TextRun({ children: [new Tab(), `(${eqNo})`] })],
  });
}
const frac = (n, d) => new MathFraction({ numerator: n, denominator: d });
const sub = (b, s) => new MathSubScript({ children: [R(b)], subScript: [R(s)] });
const sup = (b, s) => new MathSuperScript({ children: b, superScript: s });

// ------------------------------------------------------------------ tables (IEEE: caption above, 8 pt, rules)
const ROMAN = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII"];
let tabNo = 0;
const rule = { style: BorderStyle.SINGLE, size: 6, color: "000000" };
const none = { style: BorderStyle.NONE, size: 0, color: "FFFFFF" };
function TABLE(title, headers, rows, widths, opts = {}) {
  const total = widths.reduce((a, b) => a + b, 0);
  if (total !== COLW) throw new Error(`table ${title}: widths sum ${total}`);
  const no = ROMAN[tabNo++];
  const cell = (t, i, header, last) => new TableCell({
    width: { size: widths[i], type: WidthType.DXA },
    margins: { top: 20, bottom: 20, left: 50, right: 50 },
    verticalAlign: VerticalAlign.CENTER,
    borders: { top: header ? rule : none, bottom: header || last ? rule : none, left: none, right: none },
    children: [new Paragraph({ alignment: i === 0 ? AlignmentType.LEFT : AlignmentType.CENTER, spacing: { before: 0, after: 0 },
      children: runs(String(t), { size: 15, font: FONT, bold: header ? true : undefined }) })],
  });
  const caption = [
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 160, after: 0 }, keepNext: true,
      children: [new TextRun({ text: `TABLE ${no}`, size: 16, smallCaps: true })] }),
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 0, after: 60 }, keepNext: true,
      children: runs(title, { size: 16, smallCaps: true }) }),
  ];
  const table = new Table({
    width: { size: COLW, type: WidthType.DXA }, columnWidths: widths,
    rows: [new TableRow({ tableHeader: true, children: headers.map((h, i) => cell(h, i, true, false)) }),
      ...rows.map((r, ri) => new TableRow({ cantSplit: true, children: r.map((t, i) => cell(t, i, false, ri === rows.length - 1)) }))],
  });
  const out = [...caption, table];
  if (opts.note) out.push(new Paragraph({ spacing: { before: 40, after: 120 }, alignment: AlignmentType.JUSTIFIED,
    children: runs(opts.note, { size: 14 }) }));
  else out.push(new Paragraph({ spacing: { before: 0, after: 80 }, children: [] }));
  return { no, items: out };
}

// ------------------------------------------------------------------ figures (caption below, 8 pt)
let figNo = 0;
function FIGURE(file, wIn, hIn, caption) {
  figNo += 1;
  return {
    no: figNo,
    items: [
      new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120, after: 40 }, keepNext: true,
        children: [new ImageRun({ type: "png", data: fs.readFileSync(path.join(FIG, file)),
          transformation: { width: Math.round(wIn * 96), height: Math.round(hIn * 96) },
          altText: { title: `Fig. ${figNo}`, description: caption.replace(/\*\*|\*|\[\[[^\]]+\]\]/g, ""), name: file } })] }),
      new Paragraph({ alignment: AlignmentType.JUSTIFIED, spacing: { before: 0, after: 140 },
        children: [new TextRun({ text: `Fig. ${figNo}. `, size: 16 }), ...runs(caption, { size: 16 })] }),
    ],
  };
}

// ================================================================== CONTENT
const body = [];
const add = (...xs) => xs.forEach((x) => (x && x.items ? body.push(...x.items) : body.push(x)));

// ---------------- Abstract & index terms
const ABSTRACT = "We present a system for three-way stance detection (Favour, Against, None) of Hindi and Bengali social-media comments toward the claim “Climate change and global warming is a serious concern.” No in-domain training data exists, so we construct a 19,278-row training set from four public English and Bengali sources using NLLB-200 translation filtered by LaBSE back-translation similarity, rule-based Romanization and code-mixing, capped template synthesis, and leakage-controlled group-wise splitting. Because validation on constructed data is a poor proxy for real comments, we evaluate on two test-only resources: a hand-written gold set of 306 comments in 14 linguistic categories and a CheckList-style suite of 1,084 behavioural tests. On a 4 GB laptop GPU we compare 16 baselines, 18 fine-tuned transformer runs across six encoders with single and target-aware pair inputs, a local 1.5B-parameter LLM, and 11 augmentation ablations. The selected system, an equal-weight average of two MuRIL models, a TF-IDF Complement Naive Bayes model and a BGE-M3 classifier, reaches macro F1 0.766 on the gold set against 0.732 for the strongest baseline. The gain is not statistically significant, but the challenge-suite failure rate falls from 0.287 to 0.209. Seed variance (0.60–0.78 gold F1 for one configuration) exceeds differences between backbones, target-aware pair input is as accurate as single input and more robust, and cross-validation on constructed data ranks models differently from realistic comments. Remaining errors concentrate on negated denials, sarcasm and implicit stance.";
const abstractWords = ABSTRACT.split(/\s+/).length;
add(new Paragraph({ style: "Body", indent: { firstLine: 0 }, spacing: { after: 80 },
  children: [new TextRun({ text: "Abstract—", bold: true, italics: true, size: 18 }), new TextRun({ text: ABSTRACT, bold: true, size: 18 })] }));
add(new Paragraph({ style: "Body", indent: { firstLine: 0 }, spacing: { after: 120 },
  children: [new TextRun({ text: "Index Terms—", bold: true, italics: true, size: 18 }),
    new TextRun({ text: "stance detection, Hindi and Bengali, code-mixed text, climate change discourse, robustness evaluation, low-resource NLP.", bold: true, size: 18 })] }));

// ---------------- I. Introduction
add(H1("I. Introduction"));
add(H2("A. Background"));
add(P(`Stance detection asks whether a text supports, opposes or is neutral toward a given target [[mohammad2016]]. It differs from sentiment analysis: a negative-sounding comment can be in favour of the target [[mohammad2017]]. Public opinion on climate change in India and Bangladesh is increasingly expressed on social media in Hindi and Bengali, often Romanized (“Hinglish”, “Banglish”), mixed with English, informally spelled and frequently sarcastic. Measuring this opinion at scale requires stance classifiers that work on such text.`));
add(H2("B. Problem Statement"));
add(P(`The Indic Climate Stance Challenge asks for a classifier that labels a Hindi or Bengali comment as Favour, Against or None toward the claim “Climate change and global warming is a serious concern,” scored by macro F1 on a hidden, noisy test set. Three constraints make the task hard: (i) no in-domain training data is provided, and labelled climate-stance data exists only in English or for Bengali news with a different target; (ii) the test text contains Romanized and code-mixed comments, typos, emoji, negation and sarcasm that are rare in available sources; and (iii) all work must run on a single laptop with a 4 GB GPU and no paid APIs.`));
add(H2("C. Contributions"));
add(ITEM("1)", "A documented, leakage-controlled pipeline that turns English stance data into Hindi and Bengali training data through filtered machine translation, Romanization, code-mixing and capped template synthesis."));
add(ITEM("2)", "Two test-only resources that resemble the hidden test: a 306-comment hand-written gold set in 14 categories and a 1,084-test CheckList-style behavioural suite."));
add(ITEM("3)", "A systematic comparison of sparse, dense, transformer and LLM approaches, including a target-aware pair-input formulation with the claim in four language settings and multi-seed runs."));
add(ITEM("4)", "A transparent, robustness-weighted model-selection rule with every candidate published, and statistical testing of the final gain."));
add(H2("D. Paper Organization"));
add(P(`Section II reviews related work and Section III analyses the gaps it leaves. Section IV describes the data construction, evaluation resources and models. Section V reports and discusses the results, and Section VI concludes with future work.`));

// ---------------- II. Related work
add(H1("II. Related Work"));
add(H2("A. Stance Detection and Climate Datasets"));
add(P(`SemEval-2016 Task 6 [[mohammad2016]] introduced tweet-level stance toward five targets, including “Climate Change is a Real Concern”; TweetEval [[barbieri2020]] redistributes it. GWSD [[luo2020]] labels news sentences toward the claim used in this challenge, aggregating eight annotators per sentence. ClimateStance [[vaid2022]] provides COP25-era tweets labelled for stance. For Bengali, Dhoroni [[wasi2024]] annotates environmental news from several perspectives, including a stance toward the environment in general. Surveys [[kucuk2020]], [[aldayel2021]] summarise target-specific, cross-target and zero-shot stance detection.`));
add(H2("B. Target-Aware and Cross-Lingual Stance"));
add(P(`Conditional encoding of the target [[augenstein2016]] and target-specific attention [[du2017]] predate transformers. Encoding the target and the text as a sentence pair became standard for zero-shot and cross-target stance [[allaway2020]]. Hardalov et al. [[hardalov2022]] study few-shot cross-lingual stance with sentiment-based pre-training. Zero-shot classification by entailment [[yin2019]], available through multilingual NLI checkpoints [[laurer2024]], provides a training-free baseline.`));
add(H2("C. Multilingual and Indic Encoders"));
add(P(`MuRIL [[khanuja2021]] is pre-trained on 17 Indian languages including transliterated text. IndicBERT v2 [[doddapaneni2023]] and BanglaBERT [[bhattacharjee2022]] are Indic- and Bengali-specific, while XLM-R [[conneau2020]] and mBERT [[devlin2019]] are general multilingual encoders. LaBSE [[feng2022]], multilingual E5 [[wang2024]] and BGE-M3 [[chen2024]] provide sentence embeddings. For translation, NLLB-200 [[nllb2022]] covers Hindi and Bengali; IndicTrans2 [[gala2023]] is a strong Indic system whose checkpoints are access-gated.`));
add(H2("D. Robustness Evaluation and Augmentation"));
add(P(`CheckList [[ribeiro2020]] defines minimum-functionality, invariance and directional tests. Contrast sets [[gardner2020]] and back-translation [[sennrich2016]] motivate augmentation, and GLUECoS [[khanuja2020]] shows that transliteration-aware pre-training matters for code-mixed text.`));

// ---------------- III. Gap analysis
add(H1("III. Gap Analysis"));
add(H2("A. Resource Gap"));
add(P(`Table I compares existing climate-stance resources with the requirements of the challenge. To our knowledge no public dataset labels Hindi or Bengali social-media comments toward the climate-concern claim, none contains Romanized or code-mixed text, and the only Bengali resource targets a different stance definition.`));
const gapRef = { s: cite("mohammad2016"), g: cite("luo2020"), c: cite("vaid2022"), d: cite("wasi2024") };
add(TABLE("Existing Climate-Stance Resources vs. Challenge Requirements",
  ["Resource", "Lang.", "Domain", "Target", "Rom./CM"],
  [[`SemEval-16 T6 ${gapRef.s}`, "en", "Tweets", "Climate change is a real concern", "No"],
   [`GWSD ${gapRef.g}`, "en", "News sentences", "Same claim as challenge", "No"],
   [`ClimateStance ${gapRef.c}`, "en", "Tweets", "Climate change (general)", "No"],
   [`Dhoroni ${gapRef.d}`, "bn", "News", "Environment (general)", "No"],
   ["**This work**", "hi, bn", "Comment-style", "Same claim as challenge", "Yes"]],
  [1500, 560, 1000, 1360, 620], { note: "Rom./CM: contains Romanized or code-mixed text. “This work” refers to the constructed training data and the two test-only resources." }));
add(H2("B. Methodological Gap"));
add(P(`Cross-lingual stance studies typically report accuracy on held-out portions of the same constructed or translated data. When the training data is translated news and tweets but the deployment data is short, noisy comments, such validation can mislead model selection. We are not aware of a behavioural test suite for Hindi or Bengali stance that probes negation, sarcasm, Romanization or code-mixing.`));
add(H2("C. How This Work Addresses the Gaps"));
add(P(`We (i) construct Hindi and Bengali training data for the exact claim with Romanized and code-mixed variants, (ii) build a hand-written gold set and a CheckList-style suite that are never used for fitting, and (iii) select models on these resources with a fixed, published rule rather than on validation data drawn from the constructed distribution.`));

// ---------------- IV. Methodology
add(H1("IV. Methodology"));
add(H2("A. Task Formulation"));
add(P(`Given a comment *x* in Hindi or Bengali, the system predicts a label *y* ∈ *C* = {Favour, Against, None}. The primary metric is macro F1,`));
add(EQ([sub("F1", "macro"), R("="), frac([R("1")], [R("|C|")]),
  new MathSum({ children: [frac([R("2"), sub("P", "c"), sub("R", "c")], [sub("P", "c"), R("+"), sub("R", "c")])], subScript: [R("c∈C")] })]));
add(PN(`where *P*_{c} and *R*_{c} are the precision and recall of class *c*. Per-class scores, accuracy, confusion matrices and per-language scores are also reported.`));
add(H2("B. Training Data Construction"));
add(P(`Fig. 1 summarises the pipeline. Four labelled sources are loaded: SemEval-2016 Task 6 [[mohammad2016]] (564 rows), GWSD [[luo2020]] (2,245; majority of eight annotators), ClimateStance [[vaid2022]] (3,580) and Dhoroni [[wasi2024]] (1,461). Dhoroni’s “Negative” label marks bad environmental news, not denial, so it is not mapped to Against.`));
add(FIGURE("fig1_pipeline.png", 3.3, 3.68, "Data and modelling pipeline. Test-only resources (right) are never used for fitting; they are used once to compare a pre-declared set of candidates."));
add(P(`English sentences are translated into Hindi and Bengali with NLLB-200-distilled-600M [[nllb2022]]. Each output is back-translated, and the pair is kept only if the LaBSE [[feng2022]] cosine similarity between source and back-translation is at least 0.75; 11,888 of 12,778 translations pass (93.0 %). A rule-based Romanizer (ITRANS with Hindi schwa deletion, Bengali inherent *o* and casual spelling variants) and a code-mixer (whole-word English substitution from a hand-written lexicon, discourse markers, partial Romanization) create Hinglish and Banglish copies. Hand-written templates for 3 labels × 10 styles are capped at 40 % of the training data (actual share 8.9 %). Exact and near duplicates (MinHash over character 4-grams, Jaccard ≥ 0.85) are removed within each language across all rows, including the gold set, before a group-wise split that keeps a source sentence and all its derivatives together. Training rows that near-duplicate any challenge test are removed.`));
add(TABLE("Dataset Splits",
  ["Split", "Rows", "hi", "bn", "Fav.", "Ag.", "None"],
  [["Train", "19,278", "9,788", "9,490", "10,982", "2,225", "6,071"],
   ["Dev", "1,402", "712", "690", "825", "130", "447"],
   ["Gold (test-only)", "306", "153", "153", "110", "104", "92"]],
  [1260, 660, 620, 620, 660, 600, 620],
  { note: "Train rows by origin: translated 11,372; transliterated 3,165; synthetic 1,723; code-mixed 1,720; original Bengali 1,298. A manual audit of 50 translated rows found about 15 % questionable labels." }));
add(H2("C. Test-Only Evaluation Resources"));
add(P(`The **gold set** contains 306 comments (153 per language) written by hand before any model was trained, covering 14 categories: explicit, implicit, negation, sarcasm, Romanized, code-mixed, typos, emoji, short, long, topic-related neutral, off-topic, conspiracy framing and different target. The **challenge suite** follows CheckList [[ribeiro2020]]: 780 minimum-functionality tests (13 categories × 2 languages), 240 invariance tests (typos, emoji, URLs, mentions, elongation, whitespace, partial Romanization and punctuation must not change the label) and 64 directional tests (negation must flip the label). All fitting uses training folds or the dev split only.`));
add(H2("D. Preprocessing"));
add(P(`Text is NFC-normalised and Indic-normalised; zero-width characters are removed; URLs and mentions become placeholders; elongations are reduced and Latin text is lowercased. Emoji and hashtag text are kept. Negators and stopwords are never removed, because a single “नहीं”, “না” or “nahi” flips stance.`));
add(H2("E. Baselines"));
add(P(`Reference points are the majority class, a stratified random guesser and a hand-written cue lexicon. Sparse models use TF-IDF over word 1–2-grams and character 2–5-grams with logistic regression (LR), a calibrated linear SVM or Complement Naive Bayes (CNB); NB-SVM [[wangmanning2012]]; and variants that concatenate each text with its transliterated copy. Dense models train LR on frozen LaBSE, multilingual-E5 and BGE-M3 embeddings. Target-aware variants combine the comment embedding *u* and the claim embedding *v* as in [[reimers2019]]:`));
add(EQ([R("f(x)=[u; v; |u−v|; u⊙v]")]));
add(PN(`Zero-shot NLI uses an mDeBERTa-v3 XNLI checkpoint [[laurer2024]] with Hindi and Bengali hypotheses. A stacked hybrid combines the best sparse and dense models by CV with an LR meta-learner on out-of-fold probabilities. All baselines use 5-fold CV stratified by label × language and grouped by source sentence.`));
add(H2("F. Fine-Tuned Transformers"));
add(P(`We fine-tune MuRIL [[khanuja2021]], IndicBERT v2 [[doddapaneni2023]], XLM-R base [[conneau2020]], mBERT [[devlin2019]], BanglaBERT [[bhattacharjee2022]] (Bengali only) and the XNLI-tuned mDeBERTa. The *single* formulation encodes [CLS] comment [SEP]; the *pair* (target-aware) formulation encodes [CLS] claim [SEP] comment [SEP], with the claim in English, Hindi, Bengali or the comment’s own language (“auto”). Class imbalance is handled by weighted cross-entropy or focal loss [[lin2017]]:`));
add(EQ([sub("L", "WCE"), R("=−"), sub("w", "y"), R("log "), sub("p", "y"), R(",   "), sub("w", "c"), R("="), sup([R("("), frac([R("N")], [sub("n", "c")]), R(")")], [R("0.5")])]));
add(EQ([sub("L", "FL"), R("=−"), sub("w", "y"), sup([R("("), R("1−"), sub("p", "y"), R(")")], [R("γ")]), R("log "), sub("p", "y")]));
add(PN(`where *p*_{y} is the predicted probability of the true class, *n*_{c} the training count of class *c*, *N* the total, and γ = 2. Training uses AdamW (learning rate 2×10⁻⁵, weight decay 0.01, 10 % warm-up), 3 epochs, batch 16, maximum length 128, fp16 and early stopping on dev macro F1. To fit 4 GB of VRAM the word-embedding matrix is frozen; otherwise MuRIL’s 197k-token embeddings and optimiser states spill into system memory and training slows from 0.25 to 8 s per step. The three strongest MuRIL configurations are run with three seeds (42, 43, 44); others with one.`));
add(H2("G. Local LLM"));
add(P(`Qwen2.5-1.5B-Instruct [[qwen2024]] runs locally in fp16 and classifies by constrained scoring: the log-likelihood of each label word after a stance-definition prompt, so output parsing cannot fail. Few-shot examples (*k* = 4) are retrieved by LaBSE nearest-neighbour search over the training data.`));
add(H2("H. Ensembling, Calibration and Model Selection"));
add(P(`Candidates are the two best two-language transformers by dev F1, the best sparse and dense baselines, and the stacked hybrid. They are combined by equal averaging over *M* members,`));
add(EQ([R("p̄(x)="), frac([R("1")], [R("M")]), new MathSum({ children: [sub("p", "m"), R("(x)")], subScript: [R("m=1")], superScript: [R("M")] })]));
add(PN(`by dev-tuned weighted averaging, or by a dev-fitted stacking meta-learner. Each is evaluated with and without calibration [[guo2017]] fitted on dev, a temperature *T* and per-class bias *b*:`));
add(EQ([R("p̃=softmax("), frac([R("log p̄")], [R("T")]), R("+b)")]));
add(PN(`The resulting 11 candidates are ranked by a pre-declared score that weighs robustness,`));
add(EQ([R("S="), sub("F1", "gold"), R("−λ"), sub("r", "fail"), R(",   λ=0.5")]));
add(PN(`where *r*_{fail} is the challenge-suite failure rate; every candidate’s score is published. For CPU deployment we also evaluate dynamic int8 quantisation of the best transformer.`));
add(H2("I. Statistical Testing"));
add(P(`Differences on the gold set are tested with a paired bootstrap [[efron1993]] (2,000 resamples), giving 95 % confidence intervals for each model and for its difference from the baseline. With 306 items the 95 % interval of a single macro F1 is about ±0.05.`));

// ---------------- V. Results and discussion
add(H1("V. Results and Discussion"));
add(H2("A. Experimental Setup"));
add(P(`All experiments ran on an Intel i5-12500H with 16 GB RAM and an NVIDIA RTX 3050 Laptop GPU (4 GB) using Python 3.11, PyTorch, Hugging Face Transformers and scikit-learn. Every script reads a YAML configuration, fixes all random seeds, caches translations, embeddings and predictions, and logs one row per experiment (configuration, seed, git commit, metrics, runtime). Fine-tuning one MuRIL configuration takes about 15 minutes.`));
add(H2("B. Baselines"));
add(P(`Table III lists the baselines. TF-IDF + CNB has one of the lowest CV scores but the best gold score among classical models and is the baseline to beat. Target-aware features add about 0.03 gold F1 to both LaBSE and E5, and zero-shot NLI performs poorly without fine-tuning.`));
add(TABLE("Baselines (Macro F1 and Challenge Failure Rate)",
  ["Model", "CV", "Gold", "hi", "bn", "Fail"],
  [["Majority", "0.242", "0.176", "0.176", "0.176", "0.542"],
   ["Lexicon†", "0.337", "0.575", "0.559", "0.591", "0.369"],
   ["Zero-shot NLI", "0.405", "0.413", "0.362", "0.432", "0.453"],
   ["TF-IDF + LR", "0.666", "0.635", "0.664", "0.606", "0.320"],
   ["TF-IDF + LR, translit.", "**0.675**", "0.550", "0.679", "0.396", "0.391"],
   ["NB-SVM", "0.653", "0.612", "0.656", "0.568", "0.356"],
   ["**TF-IDF + CNB**", "0.626", "**0.732**", "0.727", "**0.738**", "0.287"],
   ["LaBSE + LR", "0.610", "0.612", "0.590", "0.635", "0.304"],
   ["LaBSE target-aware", "0.611", "0.643", "0.637", "0.649", "0.296"],
   ["E5 target-aware", "0.626", "0.671", "0.682", "0.661", "0.316"],
   ["BGE-M3 + LR", "0.624", "0.719", "0.722", "0.716", "**0.266**"],
   ["Stacked hybrid", "0.669", "0.670", "0.715", "0.625", "0.300"]],
  [1640, 680, 680, 680, 680, 680],
  { note: "† Written by the gold-set author; its gold score is optimistic. Fail: challenge-suite failure rate (lower is better)." }));
add(H2("C. Transformers and Input Formulation"));
add(P(`Table IV summarises the fine-tuned encoders. Three findings stand out. First, **seed variance dominates**: MuRIL with single input scored 0.783, 0.604 and 0.768 gold F1 over three seeds while its dev F1 stayed near 0.665, so MuRIL, XLM-R and IndicBERT v2 (0.774–0.783 with one seed) are statistically tied. One seed of the Bengali-claim configuration never left the majority-class solution (dev F1 0.455), a known fine-tuning instability [[mosbach2021]]; such runs are detected with a dev-only rule (dev F1 < 0.5) and reported separately. Second, **pair input with a fixed claim language** is at least as accurate as single input over seeds, with lower challenge failure and lower variance, whereas a claim in the comment’s own language was the weakest variant. Third, mBERT is clearly behind the Indic-aware and larger encoders.`));
add(TABLE("Fine-Tuned Transformers",
  ["Configuration", "Seeds", "Gold F1", "Dev", "Fail"],
  [["MuRIL single", "3", "0.718 ± 0.099", "0.665", "0.272"],
   ["MuRIL pair, en", "3", "0.726 ± 0.054", "0.678", "0.247"],
   ["MuRIL pair, bn", "2‡", "0.774 ± 0.005", "0.670", "0.238"],
   ["MuRIL pair, hi", "1", "0.728", "0.660", "0.246"],
   ["MuRIL pair, auto", "1", "0.674", "0.654", "0.291"],
   ["MuRIL pair, auto, focal", "1", "0.732", "0.672", "0.271"],
   ["XLM-R pair, auto", "1", "0.776", "0.671", "0.256"],
   ["IndicBERT v2 pair, auto", "1", "0.774", "0.667", "**0.210**"],
   ["mDeBERTa-XNLI pair", "1", "0.700", "0.649", "0.252"],
   ["mBERT pair, auto", "1", "0.642", "0.609", "0.352"],
   ["BanglaBERT (bn only)", "1", "0.797*", "0.682", "0.201"]],
  [1760, 560, 1320, 700, 700],
  { note: "Multi-seed rows give mean ± std of gold F1 and mean dev F1 and failure rate. ‡ A third seed failed to train (excluded). * Bengali half of the gold set only; not comparable with two-language scores." }));
add(H2("D. Local LLM"));
add(P(`The 1.5B-parameter LLM is far below the fine-tuned encoders (Table V), with Against recall near 0.2. Retrieved few-shot examples lower accuracy and make inference 27× slower because the longer prompts exceed 4 GB of VRAM (5.6 GB peak).`));
add(TABLE("Local LLM (Qwen2.5-1.5B-Instruct)",
  ["Setting", "Gold F1", "Fail", "GPU time"],
  [["Zero-shot", "0.481", "0.493", "0.36 s"], ["4-shot, kNN retrieval", "0.414", "0.551", "9.7 s"]],
  [2040, 1000, 1000, 1000], { note: "GPU time per comment." }));
add(H2("E. Augmentation Ablations"));
add(P(`Each variant retrains TF-IDF + CNB (5-fold CV) and MuRIL single (2 epochs, one seed); Table VI reports the results. The pre-declared rule keeps a variant only if CV (dev for MuRIL) and the challenge failure rate both improve. It flags five variants, but every flagged gain is within noise: transformer differences (−0.05 to +0.04) are within seed spread, the negation-flip CV gain is +0.0002 against a CV standard deviation of 0.006, and the +0.074 CV gain from more templates is mechanical because template rows are almost trivially learnable (out-of-fold F1 0.94–0.99). No augmentation is therefore used in the final model. The one clear effect is that removing the templates collapses gold F1 for both models; because templates and gold set share an author, part of this effect may reflect shared style.`));
add(TABLE("Augmentation Ablations",
  ["Variant", "S-CV", "S-Gold", "S-Fail", "M-Dev", "M-Gold", "M-Fail"],
  [["Reference", "0.626", "0.732", "0.287", "0.674", "0.717", "0.272"],
   ["− templates", "0.537", "**0.419**", "0.475", "0.650", "**0.504**", "0.385"],
   ["− transliterated", "0.634", "0.704", "0.303", "0.658", "0.741", "0.214"],
   ["− code-mixed", "0.630", "0.722", "0.280", "0.675", "0.748", "0.259"],
   ["+ char. noise", "0.611", "0.728", "0.292", "0.681", "0.739", "0.229"],
   ["+ dropped matras", "0.615", "0.710", "0.303", "0.675", "0.746", "0.259"],
   ["+ repeated chars", "0.613", "0.714", "0.289", "0.668", "0.699", "0.257"],
   ["+ negation flips", "0.627", "0.729", "0.284", "0.656", "0.695", "0.287"],
   ["+ back-translation", "0.619", "0.726", "0.305", "0.649", "0.665", "0.272"],
   ["+ English data", "0.614", "0.725", "0.293", "0.678", "0.695", "0.273"],
   ["+ 3× templates", "0.700", "0.756", "0.270", "0.668", "0.756", "0.227"]],
  [1440, 600, 600, 600, 600, 600, 600],
  { note: "S: TF-IDF + CNB. M: MuRIL single input (2 epochs, 1 seed)." }));
add(H2("F. Final System"));
add(P(`Table VII lists the candidates ranked by the selection score (7). The selected system averages MuRIL pair-en, MuRIL pair-auto with focal loss, TF-IDF + CNB and BGE-M3 + LR. It reaches gold macro F1 0.766 (Hindi 0.800, Bengali 0.732) and the lowest challenge failure rate of any candidate, 0.209. Calibration lowers the dev expected calibration error (ECE) from 0.065 to 0.028 but costs about 0.03 gold F1, so the final model is uncalibrated. Fig. 2 places every model in the accuracy–robustness plane.`));
add(TABLE("Final-Model Candidates",
  ["Candidate", "Dev", "ECE", "Gold", "Fail", "Score"],
  [["**Average of 4 (selected)**", "0.686", "0.065", "**0.766**", "**0.209**", "**0.662**"],
   ["MuRIL pair-en", "0.683", "0.090", "0.737", "0.220", "0.627"],
   ["MuRIL pair-en + cal.", "0.685", "0.046", "0.730", "0.218", "0.622"],
   ["Weighted avg. + cal.", "0.699", "0.031", "0.731", "0.220", "0.621"],
   ["Average + cal.", "0.700", "0.028", "0.740", "0.241", "0.619"],
   ["Weighted average", "0.699", "0.064", "0.725", "0.211", "0.619"],
   ["MuRIL pair-auto, focal", "0.672", "0.121", "0.732", "0.271", "0.597"],
   ["TF-IDF + CNB", "0.578", "0.144", "0.732", "0.287", "0.588"],
   ["BGE-M3 + LR", "0.634", "0.039", "0.719", "0.266", "0.586"],
   ["Stacking (dev)", "0.672", "0.041", "0.684", "0.232", "0.567"],
   ["Hybrid baseline", "0.638", "0.028", "0.670", "0.300", "0.521"]],
  [1840, 640, 640, 640, 640, 640],
  { note: "Dev: dev macro F1. ECE: expected calibration error on dev. Score: selection score (7)." }));
add(FIGURE("fig2_accuracy_robustness.png", 3.3, 2.36, "Gold macro F1 against challenge-suite failure rate for every two-language model (majority, random and the failed-to-train run omitted). The final ensemble is the most robust model and among the most accurate."));
add(P(`**Significance.** The gain over TF-IDF + CNB is +0.034 with a paired-bootstrap 95 % interval of [−0.019, +0.094], so it is not statistically significant; the final model’s own interval is [0.715, 0.810]. No model is significantly better than the baseline, and only models scoring below about 0.645 are significantly worse. The robustness gain is larger and consistent across categories (Fig. 3): failure falls most on negation-flip tests (0.59 → 0.31), different-target comments (0.68 → 0.32), topic-related neutral text (0.45 → 0.13) and conspiracy framing (0.50 → 0.33). It rises on implicit, Romanized, emoji and short comments.`));
add(FIGURE("fig3_category_failure.png", 3.3, 2.92, "Challenge-suite failure rate per category for the final ensemble and the baseline to beat. DIR: directional negation-flip tests."));
add(TABLE("Final Model on the Gold Set, per Class",
  ["Class", "Precision", "Recall", "F1", "Support"],
  [["Favour", "0.840", "0.573", "0.681", "110"], ["Against", "0.797", "0.904", "0.847", "104"], ["None", "0.699", "0.859", "0.771", "92"]],
  [1440, 900, 900, 900, 900],
  { note: "Confusion matrix (rows true, columns predicted Favour/Against/None): Favour 63/21/26; Against 2/94/8; None 10/3/79." }));
add(H2("G. Efficiency"));
add(P(`Table IX compares accuracy and CPU cost. Dynamic int8 quantisation of all linear layers collapsed MuRIL (gold F1 0.74 → 0.33), also with per-channel weights, because the feed-forward output projections receive activation outliers that per-tensor int8 activations cannot represent [[bondarenko2021]]. Quantising only the attention projections keeps the predicted probabilities within 0.02 of fp32 and is the recommended CPU option.`));
add(TABLE("Accuracy vs. Cost on CPU",
  ["Model", "Gold", "Fail", "Size", "ms/comment"],
  [["TF-IDF + CNB", "0.732", "0.287", "19 MB", "0.2"],
   ["MuRIL pair-en, fp32", "0.737", "0.220", "960 MB", "17"],
   ["MuRIL pair-en, int8 attn.", "0.715", "0.223", "–", "14"],
   ["Final ensemble", "0.766", "0.209", "1.9 GB", "197"]],
  [1840, 760, 760, 840, 840],
  { note: "Latency timed on uncached text, single process, Intel i5-12500H." }));
add(H2("H. Why Validation Disagrees with the Test Sets"));
add(P(`Cross-validation on the constructed data and the gold set rank models differently. Transliterated-copy TF-IDF has the best CV (0.675) but among the worst gold scores (0.550; Bengali 0.396), while TF-IDF + CNB has one of the lowest CVs but the best classical gold score. Out-of-fold F1 by text origin explains this: 0.94–0.99 on template rows, 0.51–0.60 on translated rows and 0.41–0.46 on original Bengali headlines. CV mostly measures fit to translationese and templates, not to short, explicit and noisy comments, which supports selecting on test-only resources with a fixed rule.`));
add(H2("I. Error Analysis"));
add(TABLE("Final-Model Accuracy by Gold Category",
  ["Category", "Acc.", "n", "Category", "Acc.", "n"],
  [["Implicit", "0.43", "30", "Neutral topic", "0.77", "30"],
   ["Negation", "0.68", "22", "Explicit", "0.87", "46"],
   ["Different target", "0.69", "16", "Code-mixed", "0.91", "22"],
   ["Sarcasm", "0.70", "20", "Conspiracy", "0.93", "14"],
   ["Romanized", "0.72", "32", "Off-topic", "1.00", "16"]],
  [1300, 660, 560, 1300, 660, 560]));
add(P(`The dominant error is Favour predicted as Against (21 of 110), about half of which are negated denials or sarcasm, for example “ये कहना गलत है कि जलवायु परिवर्तन एक झूठ है” (“it is wrong to say climate change is a lie”). LIME [[ribeiro2016]] shows that “झूठ” (lie) is the strongest Against cue while “गलत” (wrong) does not push toward Favour: the model has not learned that negated or reported denial reverses stance. Implicit Favour comments, which describe effects such as heat records or receding glaciers without stance words, are often labelled None because the English sources state stance explicitly. Relative to the baseline, the ensemble fixes 6 of 7 conspiracy errors, 6 of 13 topic-neutral errors and 5 of 6 code-mixed errors, but introduces new errors on implicit comments (8 broken, 1 fixed).`));
add(H2("J. Limitations and Ethics"));
add(P(`The gold set is author-written and shares its author with the templates, so it approximates but does not sample the hidden test. With 306 items, differences below about 0.05 macro F1 are not evidence. Most configurations have one seed. The translated training data carries US/UK framings of the climate debate and roughly 15 % inherited label noise. The 4 GB GPU forced frozen embeddings, base-size encoders and a maximum length of 128; IndicTrans2, SetFit and LLM fine-tuning were not used. No personal data is stored. The model is intended for aggregate analysis of public discourse, not for judging individuals, and its errors on sarcasm and implicit stance mean individual predictions require review.`));

// ---------------- VI. Conclusion
add(H1("VI. Conclusion and Future Work"));
add(P(`Without in-domain training data, a leakage-controlled constructed training set combined with robustness-weighted selection yields a Hindi and Bengali climate-stance classifier that reaches 0.766 macro F1 on hand-written comments and fails 21 % of behavioural tests, compared with 29 % for a strong TF-IDF baseline. The accuracy gain is within noise, while the robustness gain is substantial. Seed variance exceeds the differences between backbones, and validation on constructed data is a poor guide to real comments; both findings argue for evaluation on data that resembles deployment.`));
add(P(`Future work should, in order of expected value: (i) annotate a few hundred real comments, for which the collection and annotation pipeline is ready; (ii) add hand-written negation and reported-speech training examples, the largest error group; (iii) run more seeds per configuration and average checkpoints within the final model; and (iv) replace NLLB with IndicTrans2 and fine-tune without frozen embeddings on larger hardware.`));

// ---------------- Acknowledgment & references
add(new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("Acknowledgment")] }));
add(P(`All models are open and ran locally; no paid APIs were used. The author thanks the creators of the public datasets and models cited in this paper.`));
add(new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("References")] }));
order.forEach((k, i) => add(new Paragraph({
  alignment: AlignmentType.JUSTIFIED, spacing: { before: 0, after: 20 }, indent: { left: 360, hanging: 360 },
  tabStops: [{ type: TabStopType.LEFT, position: 360 }],
  children: [new TextRun({ text: `[${i + 1}]`, size: 16 }), new TextRun({ children: [new Tab()], size: 16 }), new TextRun({ text: REFS[k], size: 16 })],
})));
const unused = Object.keys(REFS).filter((k) => !order.includes(k));
if (unused.length) throw new Error("unused references: " + unused.join(", "));

// ================================================================== front matter (one column)
const front = [
  new Paragraph({ style: "PaperTitle", children: [new TextRun("Target-Aware, Robustness-Selected Stance Detection for Hindi and Bengali Climate Discourse Without In-Domain Training Data")] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120, after: 0 }, children: [new TextRun({ text: "Aakash Rauniar", size: 22 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 0, after: 0 }, children: [new TextRun({ text: "[Department / Affiliation]", size: 20, italics: true })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 0, after: 0 }, children: [new TextRun({ text: "[City, Country]", size: 20, italics: true })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 0, after: 240 }, children: [new TextRun({ text: "aakashrauniar2004@gmail.com", size: 20 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 0, after: 80 }, children: [new TextRun({ text: "Table of Contents", size: 20, smallCaps: true, bold: true })] }),
  new TableOfContents("Table of Contents", { hyperlink: true, headingStyleRange: "1-2" }),
];

const page = { size: { width: 12240, height: 15840 }, margin: { top: 1080, bottom: 1440, left: 900, right: 900, header: 720, footer: 720 } };
const footer = new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ children: [PageNumber.CURRENT], size: 16 })] })] });
const doc = new Document({
  creator: "Aakash Rauniar",
  title: "Target-Aware, Robustness-Selected Stance Detection for Hindi and Bengali Climate Discourse",
  features: { updateFields: true },
  styles: {
    default: { document: { run: { font: FONT, size: 20 }, paragraph: { spacing: { line: 240, before: 0, after: 0 } } } },
    paragraphStyles: [
      { id: "Body", name: "Body", basedOn: "Normal", quickFormat: true,
        run: { font: FONT, size: 20 }, paragraph: { alignment: AlignmentType.JUSTIFIED, indent: { firstLine: 288 }, spacing: { line: 240, before: 0, after: 0 } } },
      { id: "PaperTitle", name: "Paper Title", basedOn: "Normal",
        run: { font: FONT, size: 48 }, paragraph: { alignment: AlignmentType.CENTER, spacing: { before: 0, after: 120, line: 240 } } },
      { id: "TOC1", name: "toc 1", basedOn: "Normal", run: { font: FONT, size: 18 },
        paragraph: { spacing: { before: 20, after: 0, line: 228 } } },
      { id: "TOC2", name: "toc 2", basedOn: "Normal", run: { font: FONT, size: 18 },
        paragraph: { indent: { left: 240 }, spacing: { before: 0, after: 0, line: 228 } } },
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Body", quickFormat: true,
        run: { font: FONT, size: 20, smallCaps: true, bold: false, color: "000000" },
        paragraph: { alignment: AlignmentType.CENTER, spacing: { before: 200, after: 80 }, keepNext: true, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Body", quickFormat: true,
        run: { font: FONT, size: 20, italics: true, bold: false, color: "000000" },
        paragraph: { alignment: AlignmentType.LEFT, spacing: { before: 120, after: 60 }, keepNext: true, outlineLevel: 1 } },
    ],
  },
  sections: [
    { properties: { page, column: { count: 1 } }, footers: { default: footer }, children: front },
    { properties: { page, type: SectionType.NEXT_PAGE, column: { count: 2, space: 360, equalWidth: true } }, footers: { default: footer }, children: body },
  ],
});
Packer.toBuffer(doc).then((buf) => {
  fs.writeFileSync(OUT, buf);
  console.log(`wrote ${OUT}: abstract ${abstractWords} words, ${order.length} refs, ${tabNo} tables, ${figNo} figures, ${eqNo} equations`);
});
