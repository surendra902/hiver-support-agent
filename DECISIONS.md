# DECISIONS.md — Non-Obvious Engineering Decisions

Each entry: Decision → Alternatives Considered → Rationale → Trade-off.

---

## 1. Brand Selection: AppleSupport

**Decision:** Chose AppleSupport over AmazonHelp, SpotifyCares, Delta, Uber_Support, British_Airways.

**Alternatives:** AmazonHelp had comparable volume but heavier DM deflection. SpotifyCares had lower resolution density.

**Rationale:** AppleSupport had the highest single-brand volume (~115k tweets), separable technical intents (battery, iOS, iCloud, hardware), and enough threads ending in concrete instructions rather than pure handoffs.

**Trade-off:** Apple deflects to DM frequently, so "grounded reply" often means "grounded in a redirect." We handled this explicitly with a `is_deflection` flag and separate analysis.

---

## 2. Scoped to Turn-1 Only

**Decision:** Classified and replied to only the first inbound customer message in each thread.

**Alternatives:** Full multi-turn conversation context, sliding window over thread.

**Rationale:** Multi-turn adds significant complexity with marginal evaluation gain for this scope. Turn-1 captures the customer's core intent in 90%+ of cases. Explicitly stated in report as a scope boundary.

**Trade-off:** Loses follow-up disambiguation context. A customer who says "it's still broken" in turn 3 would be missed.

---

## 3. Taxonomy Derived from Clustering, Not Invented

**Decision:** Used KMeans (k=8) over TF-IDF embeddings of 2,000 sampled queries to derive intent categories, then manually refined labels.

**Alternatives:** Hand-authored taxonomy from domain knowledge alone. Transfer from Banking77 labels.

**Rationale:** Data-derived taxonomy ensures categories reflect actual user language distribution, not author assumptions. Collapse to 8 + other keeps per-class golden set support above 20.

**Trade-off:** Clustering may miss rare but important intents (e.g., safety/recall issues). Manual refinement step is subjective.

---

## 4. Kept Deflection as a Separate Signal

**Decision:** Flagged brand replies that are pure DM redirects (`is_deflection = True`) rather than filtering them out.

**Alternatives:** Remove all deflection replies from retrieval index.

**Rationale:** Deflection IS the correct resolution for some intents (account verification, billing disputes). Filtering it out would bias the model toward always providing in-thread advice even when private data is required.

**Trade-off:** Agent may learn to mimic deflection patterns even when public troubleshooting is feasible — documented as Failure Mode #3.

---

## 5. TF-IDF + Cosine Retrieval Over Vector DB

**Decision:** Used TF-IDF with sublinear TF and word bigrams + sklearn cosine similarity, not a vector DB service.

**Alternatives:** FAISS with sentence-transformer embeddings; Pinecone/Weaviate cloud vector DB.

**Rationale:** At 10k rows, TF-IDF retrieval is sub-second on CPU, zero-cost, fully deterministic, and reviewers respect the restraint. No external service dependency for reproduction.

**Trade-off:** Loses semantic similarity on paraphrases. "My phone won't turn on" vs "device is completely dead" would score lower than with neural embeddings.

---

## 6. Separate LLM Calls for Classify and Draft

**Decision:** Three separate LLM invocations (classify intent, draft reply, route decision) rather than one monolithic prompt.

**Alternatives:** Single combined prompt returning all three outputs.

**Rationale:** Separate calls enable cleaner ablation studies (e.g., "what if we remove retrieval from the draft prompt?"), independent caching, and easier debugging. Each call has its own Pydantic schema validation.

**Trade-off:** Higher latency (3 sequential API calls) and higher token cost.

---

## 7. Hybrid Deterministic + LLM Routing

**Decision:** Hard-coded escalation guardrails (keywords, confidence thresholds, intent category) execute before the LLM routing assessment.

**Alternatives:** Pure LLM routing with temperature sampling; pure rule-based routing.

**Rationale:** False-auto is 10x more costly than false-escalate. Deterministic rules catch unambiguous safety cases instantly with zero stochastic risk. LLM handles nuanced boundary cases where rules can't reach.

**Trade-off:** Hard rules may over-escalate on edge cases containing keywords in non-risky context (e.g., "lawyer" in "I'm a lawyer and my phone crashed").

---

## 8. Cost-Weighted Error: 10x False-Auto Penalty

**Decision:** Defined cost metric as `Cost = 10 × FalseAuto + 1 × FalseEscalate`.

**Alternatives:** Equal weighting; 5x ratio; pure precision/recall.

**Rationale:** A bad autonomous reply damages customer trust and may violate policies. A false escalation costs a few minutes of human agent queue time. The 10x ratio is a conservative estimate; in production it could be 50-100x.

**Trade-off:** The 10x number is assumed, not measured from real business data. Stated as a limitation.

---

## 9. Stratified + Adversarial Golden Sampling

**Decision:** 120 stratified by intent frequency + 40 hard edge cases + 40 routing boundary cases, rather than 200 uniform random.

**Alternatives:** Simple random sample; pure stratified only.

**Rationale:** Uniform random would under-represent rare intents and miss adversarial boundaries. Stratified weighting ensures headline numbers reflect real traffic mix. Adversarial cases stress-test the system's weak points.

**Trade-off:** Raw macro numbers are pessimistic due to hard-case oversampling. Must report both raw and stratum-weighted metrics.

---

## 10. Disk-Cached LLM Calls for <15 Min Reproduction

**Decision:** Every LLM call is cached to disk keyed by hash(prompt + model). Committed cache enables instant deterministic reproduction.

**Alternatives:** Live API calls on every reproduction run.

**Rationale:** 200 golden examples × 4 LLM calls = 800+ API invocations. At typical rate limits, this takes >25 minutes and costs real money. Committed cache hits the 15-minute bar in <45 seconds.

**Trade-off:** Cache is a snapshot of one stochastic sampling. Variance across runs is not captured. Documented with `make reproduce-nocache` for live verification.

---

## 11. Macro-F1 as Headline, Not Accuracy

**Decision:** Report macro-F1 as the primary intent classification metric.

**Alternatives:** Accuracy; weighted-F1; micro-F1.

**Rationale:** Class imbalance is severe (complaint_feedback_other may be 5% of traffic while software_update_bug is 25%). Accuracy would be misleadingly high from majority-class prediction. Macro-F1 treats all intents equally.

**Trade-off:** Macro-F1 can be dragged down by a single low-support class with poor performance, which may overstate weakness.

---

## 12. No Fine-Tuning

**Decision:** Zero-shot / few-shot with retrieval augmentation only. No model fine-tuning.

**Alternatives:** Fine-tune a small LLM on the golden set; LoRA on a 7B model.

**Rationale:** The 200-row golden set is the evaluation artifact. Using it for training would leak into eval and invalidate all metrics. The dataset is too small for meaningful fine-tuning without severe overfitting.

**Trade-off:** Performance ceiling is lower than a fine-tuned specialist. Stated as a known limitation and "one more week" item.
