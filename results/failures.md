# Failure Analysis: Real Examples & Root Cause Hypotheses

Total Discrepancies in Golden Set: 82 / 200

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

#### Example 1 (Tweet ID: `1368852`)
- **Customer Query:** "MAJOR DISAPPOINTMENT! ASHAMED TO BE ASSOCIATED WITH THE BRAND! Got Macbook for daughter for college and 4x's had to send out, so no computer for class. Geniuses are horrible and rude! Wouldn't help, said call support. Cracked case and didn't care. #lemon #nightmare"
- **Intent:** Gold = `device_hardware_damage` | Predicted = `complaint_feedback_other`
- **Routing:** Gold = `ESCALATE` | Predicted = `ESCALATE`
- **Stated Routing Reason:** Hard guardrail triggered: message is general complaint/feedback without actionable technical issue.
- **Drafted Reply:** "We understand your frustration and sincerely apologize for the experience. We want to help — please DM us with your contact information and details about the issue, and we'll have a senior support specialist follow up with you directly."

#### Example 2 (Tweet ID: `2279833`)
- **Customer Query:** "every time i update my iOS at least one phone charger stops working 😡😡 what’s that all about"
- **Intent:** Gold = `battery_power_charging` | Predicted = `software_update_bug`
- **Routing:** Gold = `AUTO` | Predicted = `AUTO`
- **Stated Routing Reason:** Standard self-service troubleshooting available for 'software_update_bug' with high retrieval confidence (0.40).
- **Drafted Reply:** "We're sorry for the trouble after the update! Try these steps: 1) Force restart your device (press and release Volume Up, then Volume Down, then hold Side button until Apple logo appears). 2) If the issue persists, go to Settings > General > iPhone Storage and clear app caches. 3) As a last resort, you can reinstall iOS via iTunes/Finder without losing data."

#### Example 3 (Tweet ID: `1327397`)
- **Customer Query:** "@user thanks to IOS 11.2 software update .my #iphoneSE battery backup is completely ruined and text messages are getting reloaded automatically once I power on ! Never ever felt so cheap about #apple"
- **Intent:** Gold = `battery_power_charging` | Predicted = `software_update_bug`
- **Routing:** Gold = `AUTO` | Predicted = `AUTO`
- **Stated Routing Reason:** Drafted reply provides verified self-service guidance grounded in historical resolution precedent.
- **Drafted Reply:** "We understand the frustration with software issues. Please try: 1) Restart your device normally first. 2) Check if the app has an update available in the App Store. 3) Go to Settings > General > iPhone Storage, tap the affected app, and select Offload App, then reinstall. Let us know if this helps!"

#### Example 4 (Tweet ID: `610281`)
- **Customer Query:** "What do I have to do when my Apple Pencil won't charge (through cable and male-female connector)?"
- **Intent:** Gold = `battery_power_charging` | Predicted = `store_billing_purchase`
- **Routing:** Gold = `AUTO` | Predicted = `ESCALATE`
- **Stated Routing Reason:** Query contains risk indicators requiring human oversight and verification.
- **Drafted Reply:** "To help with your purchase concern: 1) Review your receipt emails from Apple for transaction details. 2) Visit https://reportaproblem.apple.com to dispute any charge. 3) Check Settings > [Your Name] > Subscriptions to cancel any unwanted recurring charges. Refund requests are typically processed within 48 hours."

#### Example 5 (Tweet ID: `2185158`)
- **Customer Query:** "please do something about excessive battery drainage after ios 11.0.3 update .. really pissed off !! worst ios update ever !"
- **Intent:** Gold = `battery_power_charging` | Predicted = `software_update_bug`
- **Routing:** Gold = `AUTO` | Predicted = `AUTO`
- **Stated Routing Reason:** Standard self-service troubleshooting available for 'software_update_bug' with high retrieval confidence (0.47).
- **Drafted Reply:** "We're sorry for the trouble after the update! Try these steps: 1) Force restart your device (press and release Volume Up, then Volume Down, then hold Side button until Apple logo appears). 2) If the issue persists, go to Settings > General > iPhone Storage and clear app caches. 3) As a last resort, you can reinstall iOS via iTunes/Finder without losing data."

#### Example 6 (Tweet ID: `1211056`)
- **Customer Query:** "I would like an explanation for how the first 99% of my phone battery drains and 40 minutes but the last one percent has lasted me six hours today"
- **Intent:** Gold = `battery_power_charging` | Predicted = `battery_power_charging`
- **Routing:** Gold = `AUTO` | Predicted = `ESCALATE`
- **Stated Routing Reason:** Hard guardrail triggered: no sufficiently similar historical resolution found (similarity < 0.30).
- **Drafted Reply:** "Let's try some steps to improve your battery performance: 1) Go to Settings > Battery > Battery Health to check maximum capacity. 2) Disable Background App Refresh for non-essential apps. 3) Turn off Location Services for apps that don't need it. If battery health is below 80%, a service replacement may be needed."

#### Example 7 (Tweet ID: `1790878`)
- **Customer Query:** "trying to charge my watch, with the original charger as i’ve done for the past 2 years, suddenly I get this: [URL]"
- **Intent:** Gold = `battery_power_charging` | Predicted = `store_billing_purchase`
- **Routing:** Gold = `AUTO` | Predicted = `ESCALATE`
- **Stated Routing Reason:** Query contains risk indicators requiring human oversight and verification.
- **Drafted Reply:** "To help with your purchase concern: 1) Review your receipt emails from Apple for transaction details. 2) Visit https://reportaproblem.apple.com to dispute any charge. 3) Check Settings > [Your Name] > Subscriptions to cancel any unwanted recurring charges. Refund requests are typically processed within 48 hours."

#### Example 8 (Tweet ID: `2683936`)
- **Customer Query:** "Make that 8% battery drop in 8 minutes -Thanks, again for nada! Oh, sure, all I want for Xmas is a new shitty iPhone #iOS 1.1.2"
- **Intent:** Gold = `battery_power_charging` | Predicted = `battery_power_charging`
- **Routing:** Gold = `AUTO` | Predicted = `ESCALATE`
- **Stated Routing Reason:** Hard guardrail triggered: no sufficiently similar historical resolution found (similarity < 0.30).
- **Drafted Reply:** "Let's try some steps to improve your battery performance: 1) Go to Settings > Battery > Battery Health to check maximum capacity. 2) Disable Background App Refresh for non-essential apps. 3) Turn off Location Services for apps that don't need it. If battery health is below 80%, a service replacement may be needed."

#### Example 9 (Tweet ID: `877396`)
- **Customer Query:** "Hi guys, I think the battery in my iPhone 5S is swollen from what I've read online. What do I do?"
- **Intent:** Gold = `battery_power_charging` | Predicted = `battery_power_charging`
- **Routing:** Gold = `AUTO` | Predicted = `ESCALATE`
- **Stated Routing Reason:** Hard guardrail triggered: no sufficiently similar historical resolution found (similarity < 0.30).
- **Drafted Reply:** "We'd like to help with your charging concern. First, try a different Lightning/USB-C cable and power adapter. Clean the charging port gently with a soft brush. If the issue persists, try a forced restart (varies by model). Let us know your Iphone model and iOS version for specific steps!"

#### Example 10 (Tweet ID: `2676034`)
- **Customer Query:** "is there going 2 b another update? This #ios1103 sucks phone is running hot, constantly discos frm Bluetooth, &amp; batt dies fast"
- **Intent:** Gold = `battery_power_charging` | Predicted = `battery_power_charging`
- **Routing:** Gold = `AUTO` | Predicted = `ESCALATE`
- **Stated Routing Reason:** Hard guardrail triggered: no sufficiently similar historical resolution found (similarity < 0.30).
- **Drafted Reply:** "Let's try some steps to improve your battery performance: 1) Go to Settings > Battery > Battery Health to check maximum capacity. 2) Disable Background App Refresh for non-essential apps. 3) Turn off Location Services for apps that don't need it. If battery health is below 80%, a service replacement may be needed."

#### Example 11 (Tweet ID: `2246853`)
- **Customer Query:** "my charger dont seem to be working in my phone. But works in everyone elses iphone help."
- **Intent:** Gold = `battery_power_charging` | Predicted = `complaint_feedback_other`
- **Routing:** Gold = `AUTO` | Predicted = `ESCALATE`
- **Stated Routing Reason:** Hard guardrail triggered: message is general complaint/feedback without actionable technical issue.
- **Drafted Reply:** "We hear you and appreciate you sharing your feedback. We understand this experience has been frustrating, and your concerns are valid. We'd like to help make this right — could you DM us more details about the specific issue so we can connect you with the right team?"

#### Example 12 (Tweet ID: `1872065`)
- **Customer Query:** "please do something IOS 11.03 is draining my battery, slowed everything! Phone stuck with every app... unreponsive!"
- **Intent:** Gold = `battery_power_charging` | Predicted = `software_update_bug`
- **Routing:** Gold = `AUTO` | Predicted = `AUTO`
- **Stated Routing Reason:** Drafted reply provides verified self-service guidance grounded in historical resolution precedent.
- **Drafted Reply:** "We're sorry for the trouble after the update! Try these steps: 1) Force restart your device (press and release Volume Up, then Volume Down, then hold Side button until Apple logo appears). 2) If the issue persists, go to Settings > General > iPhone Storage and clear app caches. 3) As a last resort, you can reinstall iOS via iTunes/Finder without losing data."

#### Example 13 (Tweet ID: `864874`)
- **Customer Query:** "what is happening. Since I updated iOS my phone keeps shutting off even I have full battery and checking mail it goes empty"
- **Intent:** Gold = `battery_power_charging` | Predicted = `software_update_bug`
- **Routing:** Gold = `AUTO` | Predicted = `AUTO`
- **Stated Routing Reason:** Standard self-service troubleshooting available for 'software_update_bug' with high retrieval confidence (0.37).
- **Drafted Reply:** "We're sorry for the trouble after the update! Try these steps: 1) Force restart your device (press and release Volume Up, then Volume Down, then hold Side button until Apple logo appears). 2) If the issue persists, go to Settings > General > iPhone Storage and clear app caches. 3) As a last resort, you can reinstall iOS via iTunes/Finder without losing data."

#### Example 14 (Tweet ID: `2306649`)
- **Customer Query:** "why am I being told “this accessory may not be supported “when I try and charge my phone"
- **Intent:** Gold = `battery_power_charging` | Predicted = `store_billing_purchase`
- **Routing:** Gold = `AUTO` | Predicted = `ESCALATE`
- **Stated Routing Reason:** Query contains risk indicators requiring human oversight and verification.
- **Drafted Reply:** "We take billing issues seriously. Please check your purchase history at Settings > [Your Name] > Subscriptions to manage active subscriptions. For unrecognized charges, visit https://reportaproblem.apple.com. If you believe your account was compromised, please change your password at https://appleid.apple.com immediately."

#### Example 15 (Tweet ID: `2047852`)
- **Customer Query:** "So what’s happening with 11.0.2?? I update my phone, and now my QC35 headphones sound shoite! Any explanation?"
- **Intent:** Gold = `software_update_bug` | Predicted = `audio_sound_accessories`
- **Routing:** Gold = `AUTO` | Predicted = `AUTO`
- **Stated Routing Reason:** Standard self-service troubleshooting available for 'audio_sound_accessories' with high retrieval confidence (0.43).
- **Drafted Reply:** "Let's check a few things for your audio issue: 1) Go to Settings > Bluetooth and ensure your AirPods/headphones show as connected. 2) Check Settings > Accessibility > Audio/Visual > Balance slider is centered. 3) Try resetting your AirPods by holding the setup button on the case for 15 seconds until the LED flashes amber then white."

