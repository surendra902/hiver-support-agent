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

#### Example 1 (Tweet ID: `1478308`)
- **Customer Query:** "Hey , i have iPhone keep restarting again and again , the touchpad is freezing stuck on passcode screen Also the Itunes can't recognize the iphone and when i use DFU to restore the Iphone i got error 4013 HELP!! Please : ( ."
- **Intent:** Gold = `device_hardware_damage` | Predicted = `software_update_bug`
- **Routing:** Gold = `ESCALATE` | Predicted = `AUTO`
- **Stated Routing Reason:** Standard self-service troubleshooting available for 'software_update_bug' with high retrieval confidence (1.00).
- **Drafted Reply:** "We understand the frustration with software issues. Please try: 1) Restart your device normally first. 2) Check if the app has an update available in the App Store. 3) Go to Settings > General > iPhone Storage, tap the affected app, and select Offload App, then reinstall. Let us know if this helps!"

#### Example 2 (Tweet ID: `565903`)
- **Customer Query:** "Was in the process of writing an e-mail that was started by forwarding a picture, hit the home button so I could grab a URL to paste in...now I can not find e-mail. Not in mail drafts. Considering length &amp; complexity hopefully you have an idea where it is hiding."
- **Intent:** Gold = `device_hardware_damage` | Predicted = `complaint_feedback_other`
- **Routing:** Gold = `ESCALATE` | Predicted = `ESCALATE`
- **Stated Routing Reason:** Hard guardrail triggered: message is general complaint/feedback without actionable technical issue.
- **Drafted Reply:** "We understand your frustration and sincerely apologize for the experience. We want to help — please DM us with your contact information and details about the issue, and we'll have a senior support specialist follow up with you directly."

#### Example 3 (Tweet ID: `2187874`)
- **Customer Query:** "What to do? Wifit button grayed out and unclickable. 😭 @user [URL]"
- **Intent:** Gold = `device_hardware_damage` | Predicted = `connectivity_wifi_bluetooth`
- **Routing:** Gold = `ESCALATE` | Predicted = `AUTO`
- **Stated Routing Reason:** Standard self-service troubleshooting available for 'connectivity_wifi_bluetooth' with high retrieval confidence (1.00).
- **Drafted Reply:** "We'd like to help restore your connection! Try: Settings > Airplane Mode, toggle ON for 30 seconds, then OFF. If the issue persists, go to Settings > General > Transfer or Reset > Reset Network Settings. For Bluetooth devices, also check they're charged and within range."

#### Example 4 (Tweet ID: `2674181`)
- **Customer Query:** "awesome job on new IOS, love the relentless nag screen to update an app I don’t have &amp; the 5 hrs less batt life is handy too"
- **Intent:** Gold = `device_hardware_damage` | Predicted = `software_update_bug`
- **Routing:** Gold = `ESCALATE` | Predicted = `AUTO`
- **Stated Routing Reason:** Standard self-service troubleshooting available for 'software_update_bug' with high retrieval confidence (1.00).
- **Drafted Reply:** "We're sorry for the trouble after the update! Try these steps: 1) Force restart your device (press and release Volume Up, then Volume Down, then hold Side button until Apple logo appears). 2) If the issue persists, go to Settings > General > iPhone Storage and clear app caches. 3) As a last resort, you can reinstall iOS via iTunes/Finder without losing data."

#### Example 5 (Tweet ID: `374796`)
- **Customer Query:** "@user Think you made a mistake updating Weather app. Bottom right button used to show all locations. Now just switches them. [URL]"
- **Intent:** Gold = `device_hardware_damage` | Predicted = `software_update_bug`
- **Routing:** Gold = `ESCALATE` | Predicted = `AUTO`
- **Stated Routing Reason:** Standard self-service troubleshooting available for 'software_update_bug' with high retrieval confidence (1.00).
- **Drafted Reply:** "We understand the frustration with software issues. Please try: 1) Restart your device normally first. 2) Check if the app has an update available in the App Store. 3) Go to Settings > General > iPhone Storage, tap the affected app, and select Offload App, then reinstall. Let us know if this helps!"

#### Example 6 (Tweet ID: `2285339`)
- **Customer Query:** "so my iPhone crashes touch seems to stop working altho I can access wifi menu etc Ask apple the issue they tell me my phone screen needs replaced - get new phone and same issue happens is this a problem with iOS or the 8?"
- **Intent:** Gold = `device_hardware_damage` | Predicted = `software_update_bug`
- **Routing:** Gold = `ESCALATE` | Predicted = `AUTO`
- **Stated Routing Reason:** Standard self-service troubleshooting available for 'software_update_bug' with high retrieval confidence (1.00).
- **Drafted Reply:** "We're sorry for the trouble after the update! Try these steps: 1) Force restart your device (press and release Volume Up, then Volume Down, then hold Side button until Apple logo appears). 2) If the issue persists, go to Settings > General > iPhone Storage and clear app caches. 3) As a last resort, you can reinstall iOS via iTunes/Finder without losing data."

#### Example 7 (Tweet ID: `1924761`)
- **Customer Query:** "iOS 11.1.1 is killing my battery. it has dropped 3% as type this!! Help!!!!"
- **Intent:** Gold = `device_hardware_damage` | Predicted = `battery_power_charging`
- **Routing:** Gold = `ESCALATE` | Predicted = `AUTO`
- **Stated Routing Reason:** Standard self-service troubleshooting available for 'battery_power_charging' with high retrieval confidence (1.00).
- **Drafted Reply:** "We'd like to help with your charging concern. First, try a different Lightning/USB-C cable and power adapter. Clean the charging port gently with a soft brush. If the issue persists, try a forced restart (varies by model). Let us know your device model and iOS version for specific steps!"

#### Example 8 (Tweet ID: `610281`)
- **Customer Query:** "What do I have to do when my Apple Pencil won't charge (through cable and male-female connector)?"
- **Intent:** Gold = `battery_power_charging` | Predicted = `store_billing_purchase`
- **Routing:** Gold = `AUTO` | Predicted = `ESCALATE`
- **Stated Routing Reason:** Query contains risk indicators requiring human oversight and verification.
- **Drafted Reply:** "To help with your purchase concern: 1) Review your receipt emails from Apple for transaction details. 2) Visit https://reportaproblem.apple.com to dispute any charge. 3) Check Settings > [Your Name] > Subscriptions to cancel any unwanted recurring charges. Refund requests are typically processed within 48 hours."

#### Example 9 (Tweet ID: `1790878`)
- **Customer Query:** "trying to charge my watch, with the original charger as i’ve done for the past 2 years, suddenly I get this: [URL]"
- **Intent:** Gold = `battery_power_charging` | Predicted = `store_billing_purchase`
- **Routing:** Gold = `AUTO` | Predicted = `ESCALATE`
- **Stated Routing Reason:** Query contains risk indicators requiring human oversight and verification.
- **Drafted Reply:** "To help with your purchase concern: 1) Review your receipt emails from Apple for transaction details. 2) Visit https://reportaproblem.apple.com to dispute any charge. 3) Check Settings > [Your Name] > Subscriptions to cancel any unwanted recurring charges. Refund requests are typically processed within 48 hours."

#### Example 10 (Tweet ID: `2683936`)
- **Customer Query:** "Make that 8% battery drop in 8 minutes -Thanks, again for nada! Oh, sure, all I want for Xmas is a new shitty iPhone #iOS 1.1.2"
- **Intent:** Gold = `battery_power_charging` | Predicted = `battery_power_charging`
- **Routing:** Gold = `ESCALATE` | Predicted = `AUTO`
- **Stated Routing Reason:** Standard self-service troubleshooting available for 'battery_power_charging' with high retrieval confidence (1.00).
- **Drafted Reply:** "Let's try some steps to improve your battery performance: 1) Go to Settings > Battery > Battery Health to check maximum capacity. 2) Disable Background App Refresh for non-essential apps. 3) Turn off Location Services for apps that don't need it. If battery health is below 80%, a service replacement may be needed."

#### Example 11 (Tweet ID: `1872065`)
- **Customer Query:** "please do something IOS 11.03 is draining my battery, slowed everything! Phone stuck with every app... unreponsive!"
- **Intent:** Gold = `battery_power_charging` | Predicted = `software_update_bug`
- **Routing:** Gold = `AUTO` | Predicted = `AUTO`
- **Stated Routing Reason:** Standard self-service troubleshooting available for 'software_update_bug' with high retrieval confidence (1.00).
- **Drafted Reply:** "We're sorry for the trouble after the update! Try these steps: 1) Force restart your device (press and release Volume Up, then Volume Down, then hold Side button until Apple logo appears). 2) If the issue persists, go to Settings > General > iPhone Storage and clear app caches. 3) As a last resort, you can reinstall iOS via iTunes/Finder without losing data."

#### Example 12 (Tweet ID: `864874`)
- **Customer Query:** "what is happening. Since I updated iOS my phone keeps shutting off even I have full battery and checking mail it goes empty"
- **Intent:** Gold = `battery_power_charging` | Predicted = `software_update_bug`
- **Routing:** Gold = `AUTO` | Predicted = `AUTO`
- **Stated Routing Reason:** Standard self-service troubleshooting available for 'software_update_bug' with high retrieval confidence (1.00).
- **Drafted Reply:** "We're sorry for the trouble after the update! Try these steps: 1) Force restart your device (press and release Volume Up, then Volume Down, then hold Side button until Apple logo appears). 2) If the issue persists, go to Settings > General > iPhone Storage and clear app caches. 3) As a last resort, you can reinstall iOS via iTunes/Finder without losing data."

#### Example 13 (Tweet ID: `2306649`)
- **Customer Query:** "why am I being told “this accessory may not be supported “when I try and charge my phone"
- **Intent:** Gold = `battery_power_charging` | Predicted = `store_billing_purchase`
- **Routing:** Gold = `AUTO` | Predicted = `ESCALATE`
- **Stated Routing Reason:** Query contains risk indicators requiring human oversight and verification.
- **Drafted Reply:** "We take billing issues seriously. Please check your purchase history at Settings > [Your Name] > Subscriptions to manage active subscriptions. For unrecognized charges, visit https://reportaproblem.apple.com. If you believe your account was compromised, please change your password at https://appleid.apple.com immediately."

#### Example 14 (Tweet ID: `941077`)
- **Customer Query:** "Hey , this latest update is making my phone overheat. Known issue? Only me? What’s up?"
- **Intent:** Gold = `battery_power_charging` | Predicted = `software_update_bug`
- **Routing:** Gold = `AUTO` | Predicted = `AUTO`
- **Stated Routing Reason:** Standard self-service troubleshooting available for 'software_update_bug' with high retrieval confidence (1.00).
- **Drafted Reply:** "We understand the frustration with software issues. Please try: 1) Restart your device normally first. 2) Check if the app has an update available in the App Store. 3) Go to Settings > General > iPhone Storage, tap the affected app, and select Offload App, then reinstall. Let us know if this helps!"

#### Example 15 (Tweet ID: `2727101`)
- **Customer Query:** "I want to throw my phone out the window at times!! What is going on???! I am so frustrated with this update! we miss Steve!!!"
- **Intent:** Gold = `software_update_bug` | Predicted = `complaint_feedback_other`
- **Routing:** Gold = `AUTO` | Predicted = `ESCALATE`
- **Stated Routing Reason:** Hard guardrail triggered: message is general complaint/feedback without actionable technical issue.
- **Drafted Reply:** "Thank you for reaching out. We're sorry to hear about your experience. Your feedback helps us improve. If there's a specific issue we can help resolve, please let us know and we'll do our best to assist you."

