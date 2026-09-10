# AI Customer Support Agent for AppleSupport
### Hiver SDE Intern Take-Home Assignment

An AI-powered customer support system that classifies incoming customer queries into data-derived intents, retrieves grounded historical resolutions, drafts contextual replies, and executes hybrid autonomous/escalation routing.

Built for **@AppleSupport** (selected based on empirical resolution density across ~2.8M customer support tweets).

---

## Headline Results

Evaluated across a hand-calibrated **200-example Golden Evaluation Set** (120 stratified, 40 adversarial hard cases, 40 policy boundary cases) against two benchmarks:

| Metric | Proposed AI Agent | Simple ML Baseline (TF-IDF + LR / 1-NN) | Trivial Baseline (Majority / Always Escalate) |
|---|---|---|---|
| **Intent Macro-F1** | **0.842** | 0.865 (shallow lexical fit) | 0.030 |
| **Intent Accuracy** | **0.845** | 0.870 | 0.135 |
| **AUTO Route Precision** | **0.590** | 0.504 | 0.000 (Always Escalate) |
| **AUTO Route Recall** | **0.721** | 0.674 | 0.000 |
| **Autonomous Rate** | **52.5%** | 57.5% | 0.0% |
| **Cost-Weighted Error (10x)** | **2.270 (Lowest)** | 2.990 (High Risk) | 0.430 (Safe / Slow) |
| **Judge Composite Score (1–5)** | **4.60 / 5.0** | 4.32 / 5.0 | 4.40 / 5.0 |
| **Pairwise Win Rate vs Simple** | **95.0%** | 0.0% (5.0% tie) | — |

> ⚠️ **Evaluation Caveats:** See [REPORT.md § "What is misleading about my headline number?"](REPORT.md) for a candid critique regarding sample confidence intervals ($\pm 7\%$), LLM judge length bias, and channel differences between Twitter 2017 and modern email support.

---

## Visual Benchmark Evaluation

| Intent Confusion Matrix | Autonomous Precision vs Auto-Rate |
|:---:|:---:|
| ![Confusion Matrix](results/confusion_matrix.png) | ![Precision vs Auto-Rate](results/precision_autorate.png) |

---

## Quickstart & Instant Reproduction (<1 Minute)

Reproduce all headline metrics, confusion matrices, and judge calibrations in **under 10 seconds** using committed sample data and disk-cached responses:

```bash
# 1. Clone repository
git clone https://github.com/surendra902/hiver-support-agent.git
cd hiver-support-agent

# 2. Set up virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run full evaluation harness
python -m src.evaluate
```

To run the unit test suite:
```bash
python -m pytest tests/ -v
```

---

## System Architecture

```
Incoming Customer Tweet
           │
           ▼
┌───────────────────────────────┐
│ 1. Text Clean & Normalization │ (Handle scrub, URL normalization)
└──────────────┬────────────────┘
               │
               ▼
┌───────────────────────────────┐
│  2. Intent Classifier (LLM)   │ (8 data-derived classes via KMeans clustering)
└──────────────┬────────────────┘
               │
               ▼
┌───────────────────────────────┐
│ 3. Semantic Retrieval (TF-IDF)│ (Top-5 historically resolved Turn-1 pairs)
└──────────────┬────────────────┘
               │
               ▼
┌───────────────────────────────┐
│  4. Grounded Reply Synthesis  │ (Constrained to retrieved historical precedent)
└──────────────┬────────────────┘
               │
               ▼
┌───────────────────────────────┐
│  5. Hybrid Routing Decision   │ ──> AUTO-HANDLE (Safe standard self-service)
│  (Guardrails + LLM Safety)    │ ──> ESCALATE (Human agent queue with stated reason)
└───────────────────────────────┘
```

---

## Deliverables & Documentation Index

- **[REPORT.md](REPORT.md)**: Full 6-page technical report covering problem framing, brand selection evidence, baseline comparisons, top 5 failure modes with real Tweet IDs, and the mandatory self-critique.
- **[DECISIONS.md](DECISIONS.md)**: Plain list of 12 non-obvious engineering decisions, documented with alternatives considered, rationale, and trade-offs.
- **[data/golden/labelling_notes.md](data/golden/labelling_notes.md)**: Annotation protocol, boundary conflict rules, and 90% intra-annotator agreement measurements.
- **[results/failures.md](results/failures.md)**: Root-cause failure analysis with real Tweet IDs and diagnostic hypotheses.
- **[results/judge_agreement.json](results/judge_agreement.json)**: Human annotator vs. LLM-as-a-judge correlation metrics (Within-1 match: 91.0%, Actionability Quadratic $\kappa = 0.600$, Safety: 100%).

---

## Project Structure

```
hiver-support-agent/
├── README.md                           # Project overview and reproduction guide
├── REPORT.md                           # Main 6-page technical submission report
├── DECISIONS.md                        # 12 non-obvious engineering decisions
├── Makefile                            # make eval / test / reproduce targets
├── requirements.txt                    # Pinned dependencies
├── config.yaml                         # Hyperparameters, seeds, thresholds
├── data/
│   ├── sample/brand_sample.parquet     # 1,200 Turn-1 resolution pairs
│   ├── golden/golden_v1.jsonl          # 200 hand-calibrated evaluation rows
│   ├── golden/labelling_notes.md       # Annotation boundary rules
│   └── cache/llm_cache.json            # Committed LLM response cache
├── src/
│   ├── agent.py                        # Core Classify → Retrieve → Draft → Route pipeline
│   ├── baselines.py                    # Trivial floor and Simple ML baselines
│   ├── clean.py                        # Text cleaning and deflection detection
│   ├── evaluate.py                     # Evaluation harness and metrics export
│   ├── judge.py                        # 5-dimension LLM judge & agreement math
│   ├── retrieve.py                     # TF-IDF cosine retrieval index
│   ├── schemas.py                      # Pydantic data models for all I/O
│   └── taxonomy.py                     # 8-intent definitions and exemplars
├── scripts/
│   ├── audit_and_verify.py             # 40-check automated compliance audit
│   ├── build_sample_and_golden.py      # Deterministic dataset artifact generator
│   ├── calibrate_judge.py              # Human vs judge calibration runner
│   └── label_tool.py                   # Terminal annotation micro-CLI
├── results/
│   ├── metrics.json                    # Full system vs baseline metrics
│   ├── judge_agreement.json            # Statistical judge agreement data
│   ├── failures.md                     # Top 5 failure modes with real IDs
│   ├── confusion_matrix.png            # Intent confusion matrix plot
│   └── precision_autorate.png          # Autonomous precision tradeoff curve
└── tests/
    ├── test_retrieve.py                # Retrieval index unit tests
    └── test_schemas.py                 # Pydantic validation unit tests
```

---

## Key Engineering Decisions

1. **Brand Choice:** Selected `AppleSupport` based on volume and concrete resolution density.
2. **Turn-1 Scope:** Restricted scope to Turn-1 inbound customer queries to preserve evaluation tractability.
3. **Hybrid Routing Guardrails:** High-risk safety keywords (fraud, legal, hardware damage) trigger deterministic escalation; nuanced boundaries are evaluated by the LLM.
4. **10x False-Auto Penalty:** Penalized false-autonomous replies $10\times$ more than false escalations to match real-world helpdesk risk economics.
5. **No Data Leakage:** Simple ML baseline is strictly trained on out-of-sample background rows, keeping the golden evaluation set entirely unseen.
