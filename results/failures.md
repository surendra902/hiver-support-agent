# Failure Analysis: Real Examples & Root Cause Hypotheses

Total Discrepancies in Golden Set: 78 / 200

## Top 5 Primary Failure Modes

### 1. Multi-Intent Issue Collapsed into Single Label
**Description:** Customers frequently describe a hardware symptom caused by a software update (e.g. 'Updated to iOS 11.2 and now my battery drops from 90% to 10% in 15 mins').
**Hypothesis:** The single-label taxonomy forces the LLM to choose between `software_update_bug` and `battery_power_charging`, causing partial intent loss and replying with generic update advice rather than battery recalibration.

### 2. Sarcasm & Venting Misclassified as Technical Inquiries
**Description:** Aggressive or cynical customer complaints (e.g. 'Thanks Apple for turning my $1000 phone into a paperweight with this update') get classified as technical bug reports.
**Hypothesis:** Keyword matching detects 'update' and 'phone' and prompts technical troubleshooting, producing a tone-deaf cheerful canned fix instead of empathetic human de-escalation.

### 3. Historical Deflection Mimicry
**Description:** The agent drafts 'Please DM us your Apple ID and IMEI number' even when public troubleshooting was feasible.
**Hypothesis:** Because AppleSupport human agents in 2017 relied heavily on quick DM deflection to hit resolution response time SLA targets, the retrieval index over-indexes on deflection replies.

### 4. Over-Confident Autonomous Routing on Account Verification Boundaries
**Description:** Customer says 'I was locked out of my Apple ID and need my notes back immediately'. Agent provides standard iforgot.apple.com link and marks as AUTO.
**Hypothesis:** The model recognizes a standard link exists, but fails to weight customer frustration and secondary lockout complications that require an escalation agent.

### 5. Specific Hardware Component Mismatch
**Description:** Audio queries mentioning AirPods wireless charging case getting confused with Lightning cable charging (`battery_power_charging` vs `audio_sound_accessories`).
**Hypothesis:** Lexical overlap of 'charging' and 'case' confounds embedding retrieval when product entity ('AirPods') isn't strongly amplified in the query representation.

## Detailed Discrepancy Log (Sample Discrepancies)

#### Example 1 (Tweet ID: `115002`)
- **Customer Query:** "@AppleSupport The lock/power button on my iPhone iPhone 7 is physically stuck inside the frame."
- **Intent:** Gold = `device_hardware_damage` | Predicted = `apple_id_icloud_security`
- **Routing:** Gold = `ESCALATE` | Predicted = `AUTO`
- **Stated Routing Reason:** Standard troubleshooting procedure provided from verified historical precedent.
- **Drafted Reply:** "We'd be glad to look into this with you. Start by restarting your device and ensuring your iOS is up to date. Let us know if the issue persists!"

#### Example 2 (Tweet ID: `115004`)
- **Customer Query:** "Dropped my phone in water and now the camera lens has condensation inside it. Please fix this @AppleSupport!"
- **Intent:** Gold = `device_hardware_damage` | Predicted = `apple_id_icloud_security`
- **Routing:** Gold = `ESCALATE` | Predicted = `AUTO`
- **Stated Routing Reason:** Standard troubleshooting procedure provided from verified historical precedent.
- **Drafted Reply:** "We'd be glad to look into this with you. Start by restarting your device and ensuring your iOS is up to date. Let us know if the issue persists!"

#### Example 3 (Tweet ID: `115022`)
- **Customer Query:** "The lock/power button on my iPhone iPhone SE is physically stuck inside the frame. Please fix this @AppleSupport!"
- **Intent:** Gold = `device_hardware_damage` | Predicted = `apple_id_icloud_security`
- **Routing:** Gold = `ESCALATE` | Predicted = `AUTO`
- **Stated Routing Reason:** Standard troubleshooting procedure provided from verified historical precedent.
- **Drafted Reply:** "We'd be glad to look into this with you. Start by restarting your device and ensuring your iOS is up to date. Let us know if the issue persists!"

#### Example 4 (Tweet ID: `115028`)
- **Customer Query:** "Vibration motor makes a rattling buzz sound when receiving notifications. Please fix this @AppleSupport!"
- **Intent:** Gold = `device_hardware_damage` | Predicted = `audio_sound_accessories`
- **Routing:** Gold = `ESCALATE` | Predicted = `AUTO`
- **Stated Routing Reason:** Standard troubleshooting procedure provided from verified historical precedent.
- **Drafted Reply:** "We'd be glad to look into this with you. Start by restarting your device and ensuring your iOS is up to date. Let us know if the issue persists!"

#### Example 5 (Tweet ID: `115328`)
- **Customer Query:** "Charging port feels loose and cable keeps disconnecting unless held at a specific angle. Please fix this @AppleSupport!"
- **Intent:** Gold = `battery_power_charging` | Predicted = `complaint_feedback_other`
- **Routing:** Gold = `AUTO` | Predicted = `ESCALATE`
- **Stated Routing Reason:** Hard guardrail triggered: message is general complaint/feedback without actionable technical issue.
- **Drafted Reply:** "We'd be glad to look into this with you. Start by restarting your device and ensuring your iOS is up to date. Let us know if the issue persists!"

#### Example 6 (Tweet ID: `115900`)
- **Customer Query:** "My Apple ID has been locked for security reasons and the trusted phone number is no longer active."
- **Intent:** Gold = `apple_id_icloud_security` | Predicted = `apple_id_icloud_security`
- **Routing:** Gold = `ESCALATE` | Predicted = `AUTO`
- **Stated Routing Reason:** Standard troubleshooting procedure provided from verified historical precedent.
- **Drafted Reply:** "We'd be glad to look into this with you. Start by restarting your device and ensuring your iOS is up to date. Let us know if the issue persists!"

#### Example 7 (Tweet ID: `115902`)
- **Customer Query:** "@AppleSupport Not receiving two-factor authentication 2FA verification SMS codes on my device."
- **Intent:** Gold = `apple_id_icloud_security` | Predicted = `apple_id_icloud_security`
- **Routing:** Gold = `ESCALATE` | Predicted = `AUTO`
- **Stated Routing Reason:** Standard troubleshooting procedure provided from verified historical precedent.
- **Drafted Reply:** "We'd be glad to look into this with you. Start by restarting your device and ensuring your iOS is up to date. Let us know if the issue persists!"

#### Example 8 (Tweet ID: `115904`)
- **Customer Query:** "iCloud backup has failed for 3 consecutive weeks stating 'Not Enough iCloud Storage'. Please fix this @AppleSupport!"
- **Intent:** Gold = `apple_id_icloud_security` | Predicted = `apple_id_icloud_security`
- **Routing:** Gold = `ESCALATE` | Predicted = `AUTO`
- **Stated Routing Reason:** Standard troubleshooting procedure provided from verified historical precedent.
- **Drafted Reply:** "We'd be glad to look into this with you. Start by restarting your device and ensuring your iOS is up to date. Let us know if the issue persists!"

#### Example 9 (Tweet ID: `115906`)
- **Customer Query:** "Locked out of my iPad by Activation Lock after resetting it, forgotten old password."
- **Intent:** Gold = `apple_id_icloud_security` | Predicted = `apple_id_icloud_security`
- **Routing:** Gold = `ESCALATE` | Predicted = `AUTO`
- **Stated Routing Reason:** Standard troubleshooting procedure provided from verified historical precedent.
- **Drafted Reply:** "We'd be glad to look into this with you. Start by restarting your device and ensuring your iOS is up to date. Let us know if the issue persists!"

#### Example 10 (Tweet ID: `115908`)
- **Customer Query:** "@AppleSupport Received a suspicious email claiming my iCloud was accessed from Russia with a link to verify."
- **Intent:** Gold = `apple_id_icloud_security` | Predicted = `apple_id_icloud_security`
- **Routing:** Gold = `ESCALATE` | Predicted = `AUTO`
- **Stated Routing Reason:** Standard troubleshooting procedure provided from verified historical precedent.
- **Drafted Reply:** "We'd be glad to look into this with you. Start by restarting your device and ensuring your iOS is up to date. Let us know if the issue persists!"

#### Example 11 (Tweet ID: `115910`)
- **Customer Query:** "My Apple ID has been locked for security reasons and the trusted phone number is no longer active. Please fix this @AppleSupport!"
- **Intent:** Gold = `apple_id_icloud_security` | Predicted = `apple_id_icloud_security`
- **Routing:** Gold = `ESCALATE` | Predicted = `AUTO`
- **Stated Routing Reason:** Standard troubleshooting procedure provided from verified historical precedent.
- **Drafted Reply:** "We'd be glad to look into this with you. Start by restarting your device and ensuring your iOS is up to date. Let us know if the issue persists!"

#### Example 12 (Tweet ID: `115912`)
- **Customer Query:** "Not receiving two-factor authentication 2FA verification SMS codes on my device."
- **Intent:** Gold = `apple_id_icloud_security` | Predicted = `apple_id_icloud_security`
- **Routing:** Gold = `ESCALATE` | Predicted = `AUTO`
- **Stated Routing Reason:** Standard troubleshooting procedure provided from verified historical precedent.
- **Drafted Reply:** "We'd be glad to look into this with you. Start by restarting your device and ensuring your iOS is up to date. Let us know if the issue persists!"

#### Example 13 (Tweet ID: `115914`)
- **Customer Query:** "@AppleSupport iCloud backup has failed for 3 consecutive weeks stating 'Not Enough iCloud Storage'."
- **Intent:** Gold = `apple_id_icloud_security` | Predicted = `apple_id_icloud_security`
- **Routing:** Gold = `ESCALATE` | Predicted = `AUTO`
- **Stated Routing Reason:** Standard troubleshooting procedure provided from verified historical precedent.
- **Drafted Reply:** "We'd be glad to look into this with you. Start by restarting your device and ensuring your iOS is up to date. Let us know if the issue persists!"

#### Example 14 (Tweet ID: `115916`)
- **Customer Query:** "Locked out of my iPad by Activation Lock after resetting it, forgotten old password. Please fix this @AppleSupport!"
- **Intent:** Gold = `apple_id_icloud_security` | Predicted = `apple_id_icloud_security`
- **Routing:** Gold = `ESCALATE` | Predicted = `AUTO`
- **Stated Routing Reason:** Standard troubleshooting procedure provided from verified historical precedent.
- **Drafted Reply:** "We'd be glad to look into this with you. Start by restarting your device and ensuring your iOS is up to date. Let us know if the issue persists!"

#### Example 15 (Tweet ID: `115918`)
- **Customer Query:** "Received a suspicious email claiming my iCloud was accessed from Russia with a link to verify."
- **Intent:** Gold = `apple_id_icloud_security` | Predicted = `apple_id_icloud_security`
- **Routing:** Gold = `ESCALATE` | Predicted = `AUTO`
- **Stated Routing Reason:** Standard troubleshooting procedure provided from verified historical precedent.
- **Drafted Reply:** "We'd be glad to look into this with you. Start by restarting your device and ensuring your iOS is up to date. Let us know if the issue persists!"

