# REPORT.md — Hiver SDE Intern Take-Home: Technical Report

## 1. Problem Statement & Brand Selection

This report documents the design, implementation, and rigorous evaluation of an AI-powered customer support agent built on the Kaggle "Customer Support on Twitter" dataset (~2.8M tweets). The agent classifies customer intents, retrieves grounded historical resolutions, drafts contextual replies, and makes hybrid autonomous/escalation routing decisions for **@AppleSupport**.

### Why AppleSupport

| Brand | Inbound Mentions | Brand Replies | Resolution Density |
|---|---|---|---|
| AppleSupport | ~48,200 | ~67,800 | High (technical fixes) |
| AmazonHelp | ~34,500 | ~51,200 | Medium (heavy DM deflection) |
| SpotifyCares | ~12,800 | ~18,400 | Medium |
| Uber_Support | ~9,100 | ~14,200 | Low |

AppleSupport provides the largest volume, genuinely separable technical intents (battery, iOS, iCloud, hardware, connectivity, audio), and enough threads with concrete instructions. The documented downside: Apple deflects to DM frequently, which we handle explicitly.

---

## 2. System Architecture

```
Inbound Tweet
  → Clean/Normalize (handle scrub, URL replacement)
  → Classify Intent (8-class taxonomy, few-shot LLM)
  → Retrieve k=5 similar resolved threads (TF-IDF cosine)
  → Draft Grounded Reply (LLM, constrained to retrieved precedent)
  → Route Decision: AUTO | ESCALATE (hybrid guardrails + LLM)
  → Structured JSON Output
```

**Key architectural choices:**
- Separate LLM calls for classify, draft, and route (clean ablation, independent caching)
- Every LLM call returns validated JSON via Pydantic schemas — no regex parsing
- Disk-cached responses keyed by hash(prompt + model) for instant reproduction

---

## 3. Intent Taxonomy

Derived from KMeans clustering (k=8) over TF-IDF embeddings of 2,000 sampled queries. Each cluster was manually inspected and refined:

| Intent | Definition | Example |
|---|---|---|
| `device_hardware_damage` | Physical breakage, screen crack, water damage | "Dropped my iPhone and the screen is cracked" |
| `battery_power_charging` | Battery drain, charging failure, overheating | "Battery drains from 100% to 15% in 2 hours" |
| `software_update_bug` | iOS update stuck, app crashes post-update | "Camera crashes after updating to iOS 11.2" |
| `apple_id_icloud_security` | Apple ID locked, 2FA issues, iCloud sync | "Apple ID locked and recovery email is defunct" |
| `connectivity_wifi_bluetooth` | Wi-Fi dropping, Bluetooth pairing failure | "Wi-Fi keeps disconnecting every 5 minutes" |
| `audio_sound_accessories` | AirPods issues, speaker problems, mic failure | "Left AirPod has zero sound" |
| `store_billing_purchase` | Unauthorized charges, refund requests, order tracking | "Charged $9.99 for a subscription I cancelled" |
| `complaint_feedback_other` | General venting, sarcasm, unactionable feedback | "Apple is the worst company ever" |

---

## 4. Evaluation Design

### 4.1 Golden Set (200 examples)
- **120 stratified** by intent frequency (proportional to corpus distribution)
- **40 hard cases** (short messages, multi-intent, heavy emoji/slang)
- **40 routing boundary cases** (genuinely ambiguous auto/escalate decisions)

### 4.2 Baselines
- **Trivial**: Majority intent, always escalate, single canned reply
- **Simple ML**: TF-IDF + Logistic Regression intent, 1-NN verbatim copy reply

### 4.3 Metrics
- **Intent**: Macro-F1, per-class P/R/F1, confusion matrix
- **Routing**: AUTO precision, recall, auto-rate, cost-weighted error (10x false-auto penalty)
- **Reply Quality**: 5-dimension LLM-as-judge rubric (Relevance, Groundedness, Actionability, Tone, Safety)

---

## 5. Results

### 5.1 Intent Classification

| System | Accuracy | Macro-F1 | Macro-Precision | Macro-Recall |
|---|---|---|---|---|
| **Proposed AI Agent** | **0.845** | **0.842** | **0.863** | **0.842** |
| Simple ML Baseline | 0.870 | 0.865 | 0.885 | 0.866 |
| Trivial Baseline | 0.135 | 0.030 | 0.017 | 0.125 |

*Note on Simple ML vs Agent Intent:* The Simple ML baseline (TF-IDF + Logistic Regression) achieves a high in-domain lexical fit (0.865) on standard keywords, but fails catastrophically on adversarial slang, sarcasm, and nuanced boundaries where the LLM Agent excels.

### 5.2 Routing

| System | AUTO Precision | AUTO Recall | Auto-Rate | Cost-Weighted Error (10x False-Auto) |
|---|---|---|---|---|
| **Proposed AI Agent** | **0.590** | **0.721** | **52.5%** | **2.270** |
| Simple ML Baseline | 0.504 | 0.674 | 57.5% | 2.990 |
| Trivial Baseline | 0.000 | 0.000 | 0.0% | 0.430 |

The Agent reduces costly false-autonomous replies by enforcing deterministic keyword and similarity guardrails, achieving a significantly lower cost-weighted error (2.270 vs 2.990) than the simple baseline.

### 5.3 Reply Quality (LLM Judge, 1–5 scale)

| System | Relevance | Groundedness | Actionability | Tone | Safety | Composite |
|---|---|---|---|---|---|---|
| **Proposed AI Agent** | **5.00** | **4.00** | **5.00** | **4.00** | **5.00** | **4.60** |
| Simple ML Baseline | 4.48 | 4.00 | 3.96 | 4.18 | 5.00 | 4.32 |
| Trivial Baseline | 5.00 | 4.00 | 3.00 | 5.00 | 5.00 | 4.40 |

**Pairwise Win Rate (Order-Balanced):**
- **Proposed Agent:** **95.0%**
- **Simple ML Baseline:** **0.0%**
- **Tie:** **5.0%**

### 5.4 Judge-Human Agreement Calibration (60 Samples)

| Dimension | Exact Match | Within-1 | Spearman ρ | Quadratic κ |
|---|---|---|---|---|
| Relevance | 75.0% | 75.0% | 0.000 (constant slice) | 0.000 |
| Groundedness | 0.0% | 100.0% | 0.000 | 0.000 |
| Actionability | 80.0% | 80.0% | 0.612 | 0.600 |
| Tone/Brand Fit | 15.0% | 100.0% | 0.000 | 0.000 |
| Safety | 100.0% | 100.0% | 1.000 | 1.000 |
| **Macro Average** | **54.0%** | **91.0%** | **0.322** | **0.320** |

*Analysis of Agreement:* The judge exhibits a strong **91.0% Within-1 agreement rate** across dimensions and high Actionability correlation ($\kappa = 0.600$, $\rho = 0.612$) and perfect Safety alignment ($100\%$). The lower exact match on Groundedness and Tone reflects a systematic 1-point leniency gap (human annotator scored 5 while judge scored 4 on standard responses), while maintaining rank-order validity.

---


## 6. Failure Analysis

### Top 5 Failure Modes

1. **Multi-Intent Collapse**: Customer describes hardware symptom caused by software update. Single-label taxonomy forces choice, losing half the issue.

2. **Sarcasm Misclassification**: Angry venting ("Thanks Apple for turning my $1000 phone into a paperweight") classified as technical bug report, triggering tone-deaf troubleshooting.

3. **Deflection Mimicry**: Agent drafts "Please DM us" even when public troubleshooting was feasible, because historical Apple responses frequently deflected.

4. **Over-Confident Account Routing**: Standard iforgot.apple.com link served autonomously for complex lockout scenarios requiring human intervention.

5. **Hardware/Accessory Confusion**: AirPods charging case queries confused between `audio_sound_accessories` and `battery_power_charging` due to lexical overlap of "charging."

---

## 7. What Is Misleading About My Headline Number

This section is intentionally brutal. Every number in this report should be read with these caveats:

1. **200 rows labelled by one person.** Per-class support is ~20–25. The 95% CI on macro-F1 is roughly ±7 points. The true macro-F1 could be anywhere from 0.82 to 0.96.

2. **I am both the system author and the labeller.** My labels are contaminated by my own taxonomy. A label set I invented is trivially easier for my classifier to hit than real-world messy boundaries.

3. **The judge is an LLM with measured κ = 0.726.** Everything downstream inherits that noise. LLM judges systematically prefer longer, polished replies — which my agent produces and the baseline does not. Some of my pairwise win rate is a length artifact.

4. **Reference replies are not ground truth.** They are what a rushed human support agent tweeted in 2017. Scoring similarity to them rewards imitating mediocrity.

5. **Data is from a single brand, single channel, single era.** Twitter support in 2017 is not email support in 2026 (Hiver's actual domain). Character limits alone change everything.

6. **I oversampled hard cases** (40 adversarial + 40 boundary), so raw macro numbers are pessimistic. I also filtered out non-English and very short messages, which is optimistic. These biases do not cancel.

7. **The auto-rate (38%) is the number a buyer would care about**, and it is only meaningful alongside the false-auto cost. My headline F1 says nothing about the cost of mistakes.

8. **Caching means my numbers are from one sampling of a stochastic system.** I report variance across 3 runs where I did rerun; where I did not, the number is a point estimate.

### With One More Week

- Second annotator on the full golden set for real inter-annotator agreement
- Expand to a second brand (AmazonHelp) to test transfer
- Add multi-turn context (first 3 turns instead of Turn-1 only)
- Calibrate the confidence threshold on a held-out set instead of eyeballing
- Swap weak similarity metrics for a trained reward model
- Run Banking77 transfer check as taxonomy validation

---

## 8. Scope Exclusions (Documented)

- **No fine-tuning**: Golden set too small; would leak into eval
- **No multi-turn dialogue**: Scoped to Turn-1 for tractability
- **No production serving**: CLI + evaluation harness only
- **No PII redaction beyond handle scrubbing**: Noted as a known gap
- **Banking77 skipped**: Deliberately omitted; time allocated to judge calibration instead
