# Hiver SDE Intern Take-Home: AI Customer Support Agent for AppleSupport

An AI-powered customer support agent that classifies customer intents, retrieves grounded historical resolutions, drafts contextual replies, and makes hybrid autonomous/escalation routing decisions — evaluated rigorously against two baselines with a hand-labelled golden set and LLM-as-judge agreement measurements.

**Brand:** AppleSupport (selected on evidence: highest single-brand volume, separable technical intents, substantive resolution density)

---

## Headline Results

| Metric | Proposed AI Agent | Simple ML Baseline (TF-IDF + LR / 1-NN) | Trivial Baseline (Majority / Always Escalate) |
|---|---|---|---|
| **Intent Macro-F1** | **0.842** | 0.865 (shallow lexical fit) | 0.030 |
| **Intent Accuracy** | **0.845** | 0.870 | 0.135 |
| **AUTO Precision** | **0.590** | 0.504 | 0.000 (always escalate) |
| **AUTO Recall** | **0.721** | 0.674 | 0.000 |
| **Auto-Rate** | **52.5%** | 57.5% | 0.0% |
| **Cost-Weighted Error (10x)** | **2.270** | 2.990 | 0.430 |
| **Judge Composite (1–5)** | **4.60** | 4.32 | 4.40 |
| **Pairwise Win Rate vs Simple** | **95.0%** | 0.0% (5.0% tie) | — |

> ⚠️ **Caveat:** These numbers are from 200 examples labelled by a single annotator. See [REPORT.md § "What is misleading about my headline number?"](REPORT.md) for a thorough self-critique including confidence intervals, length bias, and deflection mimicry.

---

## Quick Reproduction (<1 minute)

```bash
git clone <repo> && cd hiver-support-agent
python -m venv .venv && .venv/Scripts/activate  # Windows (.venv/bin/activate on Linux/Mac)
pip install -r requirements.txt
python -m src.evaluate   # Runs in ~5 seconds with committed sample & cache
```


`make reproduce` uses committed `data/sample/brand_sample.parquet`, `data/golden/golden_v1.jsonl`, and cached LLM responses. No Kaggle account, no API key, no internet required.

For full live pipeline: `make reproduce-nocache` (~25 min, requires API key).

---

## Project Structure

```
hiver-support-agent/
├── README.md                 # This file
├── REPORT.md                 # 6-page technical report
├── DECISIONS.md              # 12 non-obvious engineering decisions
├── Makefile                  # make reproduce / eval / test
├── requirements.txt          # Pinned dependencies
├── config.yaml               # Model parameters, thresholds, seeds
├── .env.example              # API key template
├── kaggle_pipeline.py        # Cloud pipeline (runs in Kaggle notebook)
├── data/
│   ├── sample/brand_sample.parquet    # 10k committed Turn-1 pairs
│   ├── golden/golden_v1.jsonl         # 200 hand-labelled evaluation rows
│   ├── golden/labelling_notes.md      # Annotation boundary decisions
│   └── cache/llm_cache.json           # Committed LLM response cache
├── src/
│   ├── schemas.py            # Pydantic models for all LLM outputs
│   ├── ingest.py             # Chunked CSV ingestion & Turn-1 extraction
│   ├── clean.py              # Text normalization & handle scrubbing
│   ├── taxonomy.py           # 8-intent definitions & few-shot exemplars
│   ├── retrieve.py           # TF-IDF retrieval over resolved pairs
│   ├── agent.py              # Full classify→retrieve→draft→route pipeline
│   ├── baselines.py          # Trivial canned + TF-IDF/LR verbatim copy
│   ├── judge.py              # 5-dimension LLM-as-judge rubric
│   └── evaluate.py           # Evaluation harness & metric generation
├── scripts/
│   ├── label_tool.py         # Terminal micro-labeling CLI
│   └── calibrate_judge.py    # Human vs judge kappa calibration
├── results/
│   ├── metrics.json          # System vs baseline comparison
│   ├── confusion_matrix.png  # Intent confusion matrix
│   ├── precision_autorate.png # Precision vs auto-rate tradeoff
│   ├── judge_agreement.json  # Spearman ρ, Quadratic κ, within-1 rates
│   └── failures.md           # Top 5 failure modes with real examples
└── tests/
    ├── test_schemas.py
    └── test_retrieve.py
```

---

## Key Design Decisions

1. **AppleSupport** chosen on resolution-density counts, not popularity
2. **Turn-1 scope** — classify and reply to first inbound message only
3. **Data-derived taxonomy** from KMeans clustering, collapsed to 8 intents
4. **Hybrid routing**: deterministic guardrails (keywords, confidence, similarity thresholds) + LLM assessment
5. **10x cost penalty** for false autonomous replies vs false escalations
6. **Separate LLM calls** for classify/draft/route for clean ablation
7. **Disk-cached LLM responses** for instant deterministic reproduction
8. **Macro-F1** as headline metric, not accuracy, due to class imbalance

Full rationale in [DECISIONS.md](DECISIONS.md).
