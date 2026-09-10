# Labelling Notes for Golden Evaluation Set v1

## Labelling Protocol
- Labelled in 4 sittings of ~50 rows each to avoid annotation fatigue drift.
- 20-row calibration slice labelled twice (at sitting 1 and sitting 4) for intra-annotator consistency check.

## Ambiguous Boundary Rules

### 1. Battery Drain After Update
**Rule:** If the customer explicitly mentions a software update AND a battery symptom, classify as `software_update_bug` (root cause is the update). If only battery symptoms are mentioned without update context, classify as `battery_power_charging`.

### 2. AirPods Charging vs Battery
**Rule:** AirPods charging case issues → `audio_sound_accessories` (the product is AirPods, not iPhone battery). iPhone not charging → `battery_power_charging`.

### 3. "Please DM Us" — Routing Judgment
**Rule:** If the brand's actual reply was a DM redirect AND the customer's issue clearly requires private info (Apple ID, serial number, order number), mark `gold_route = ESCALATE`. If the issue could have been resolved publicly (e.g., "restart your phone"), mark `gold_route = AUTO` regardless of what Apple actually did. We evaluate the AGENT's ideal behavior, not Apple's 2017 shortcut.

### 4. Sarcasm vs Complaint vs Actionable Issue
**Rule:** If the message contains a concrete symptom buried in anger/sarcasm (e.g., "Thanks for bricking my phone with iOS 11"), classify by the technical symptom (`software_update_bug`) and route as `ESCALATE` due to high emotion. If no concrete symptom exists ("Apple is trash"), classify as `complaint_feedback_other`.

### 5. Multi-Intent Messages
**Rule:** Classify by the PRIMARY intent (the one the customer most needs resolved). Note the secondary intent in `gold_route_reason` field. Example: "My screen is cracked AND my iCloud is full" → `device_hardware_damage` (primary, requires physical fix) + note "secondary: iCloud storage."

## Intra-Annotator Consistency
- 20-row re-labelling after 3 sittings gap
- Intent agreement: 18/20 = 90%
- Route agreement: 19/20 = 95%
- Disagreements: 1 multi-intent boundary case (battery vs software), 1 complaint vs connectivity edge case
