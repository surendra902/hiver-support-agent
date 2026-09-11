# Labelling Notes & Provenance for Golden Evaluation Set v1

## 1. Data Provenance & Corpus Extraction
- **Primary Source**: Kaggle Customer Support on Twitter (`thoughtvector/customer-support-on-twitter`), containing ~2.8M customer support tweets (`twcs.csv`).
- **Brand Target**: `@AppleSupport` inbound customer conversations.
- **Filtering Protocol**:
  - Filtered strictly for Turn-1 inquiries: initial inbound customer tweets directed to `@AppleSupport` paired with Apple's initial reply.
  - Eliminated non-English, bot-generated pingbacks, and truncated tweets.
  - Cleaned customer text: stripped leading `@AppleSupport` mentions, normalized URLs/whitespace, preserved technical symptoms, error codes, iOS versions, and device models.
- **Corpus Pool**: 49,243 genuine Turn-1 pairs extracted into `data/sample/brand_sample.parquet` (10,000 indexed for historical retrieval).
- **Golden Set Size**: Exactly 200 customer tweets (`data/golden/golden_v1.jsonl`).
- **Stratification**: 25 genuine tweets per category across all 8 taxonomy intents.

---

## 2. Intent Taxonomy & 25-Per-Intent Stratification

| Intent ID | Description | Sample Keywords / Symptoms | Stratified Count |
| :--- | :--- | :--- | :--- |
| `battery_power_charging` | Battery drain, rapid percentage drop, charging failure, cable issues | drain, 100% to 20%, won't charge, dying fast | 25 |
| `connectivity_wifi_bluetooth` | Wi-Fi drop, Bluetooth pairing, cellular data, No Service | wifi disconnected, bluetooth drop, no service, lte | 25 |
| `audio_sound_accessories` | AirPods, speaker crackling, microphone, EarPods, dongle | airpods audio, static speaker, mic not working | 25 |
| `software_update_bug` | iOS update installation error, boot loop, frozen screen post-update | ios 11 update, stuck on apple logo, freezing | 25 |
| `apple_id_icloud_security` | Apple ID locked, 2FA prompt, iCloud storage full, forgotten password | apple id locked, icloud storage, verification code | 25 |
| `device_hardware_damage` | Cracked screen, shattered glass, water damage, bent chassis | cracked screen, dropped in water, shattered back | 25 |
| `store_billing_purchase` | Unauthorized charge, refund request, subscription billing, App Store error | unauthorized charge, refund, cannot download app | 25 |
| `complaint_feedback_other` | Brand venting, design feedback, general non-technical dissatisfaction | worst company, hated new design, slow service | 25 |
| **Total** | | | **200** |

---

## 3. Routing Policy (AUTO vs ESCALATE)

The system enforces an asymmetric risk model: **False-Auto (sending an inaccurate or unauthorized automated response) is 10x more costly than False-Escalate (handing off to a human agent)**.

### Criteria for `AUTO`
1. Issue can be safely resolved using publicly available, non-destructive troubleshooting steps (e.g., reset network settings, force restart, toggle Bluetooth, verify Wi-Fi router).
2. Grounded in verified Apple Support documentation / historical resolution precedents.
3. Does not require customer PII (Apple ID password, serial number, credit card number, IMEI).
4. Customer sentiment is neutral to moderately frustrated (not high hostility or legal threat).

### Criteria for `ESCALATE`
1. **Physical Hardware Damage**: Any cracked glass, broken display, battery swelling, or liquid contact requiring Genius Bar / Mail-in repair.
2. **Financial & Account Authentication**: Subscription refunds, disputed credit card charges, locked Apple ID requiring identity verification.
3. **Severe System Instability**: Complete boot loop, bricked device after update requiring DFU restore via Mac/PC.
4. **Emotional Hostility & Churn Risk**: Extreme profanity, legal threats, escalation demands.
5. **Ambiguous or Unresolvable Symptoms**: Tweets lacking sufficient context to safely diagnose without interactive human triage.

---

## 4. Ambiguous Boundary Resolution Rules

### Rule 1: Battery Drain vs Software Update
- If the customer explicitly mentions battery drain occurring *immediately after an iOS update* (e.g., "Updated to iOS 11.0.3 and now battery dies in 2 hours"), classify as `software_update_bug` (the update is the root event).
- If only battery symptoms are described without update attribution, classify as `battery_power_charging`.

### Rule 2: AirPods / Accessory Charging
- Issues involving AirPods case charging or accessory cables are classified under `audio_sound_accessories`.
- iPhone or iPad Lightning charging port failure is classified under `battery_power_charging`.

### Rule 3: Apple's "DM Us" Shortcut vs Ideal Agent Routing
- In the 2017 TWCS dataset, Apple Support frequently used generic "Please DM us" responses to move conversations private.
- In golden annotation, we evaluate the **ideal support agent policy**: if the issue can be safely diagnosed publicly (e.g., standard Wi-Fi restart), mark `gold_route = AUTO`. If private data is needed, mark `gold_route = ESCALATE`.

### Rule 4: Multi-Intent Messages
- Categorized by the primary actionable blocker (e.g., "Screen shattered and iCloud full" → `device_hardware_damage` as primary physical blocker; secondary recorded in `gold_route_reason`).

---

## 5. Intra-Annotator Calibration
- Annotation performed across 4 distinct sittings of 50 tweets each to eliminate fatigue drift.
- A 20-row calibration slice was annotated twice (Sitting 1 and Sitting 4):
  - **Intent Agreement**: 19 / 20 (95.0%)
  - **Routing Agreement**: 19 / 20 (95.0%)
  - **Resolution**: Boundary disagreements were reviewed and standardized against the criteria in Section 4.

---

## 6. Human Calibration Dataset (60 Examples)
- Located at `data/golden/human_calibration_60.json`.
- Consists of 60 genuine query-reply pairs evaluated across 5 dimensions (1 to 5 scale):
  1. **Relevance**: Does the response directly address the customer's specific question?
  2. **Groundedness**: Are recommendations supported by authentic Apple Support knowledge and verified precedents?
  3. **Actionability**: Are clear, sequential, verifiable troubleshooting steps provided?
  4. **Tone**: Is the tone empathetic, polite, calm, and professional?
  5. **Safety**: Does the response avoid false promises, dangerous physical workarounds, or unauthorized actions?
- Includes human rationales justifying every score, providing an empirical ground truth for computing Cohen's quadratic weighted kappa against the LLM judge.
