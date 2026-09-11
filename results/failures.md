# Failure Analysis: Real Examples & Root Cause Hypotheses

Total Discrepancies in Golden Set: 72 / 200

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
- **Intent:** Gold = `device_hardware_damage` | Predicted = `battery_power_charging`
- **Routing:** Gold = `ESCALATE` | Predicted = `AUTO`
- **Stated Routing Reason:** Standard self-service troubleshooting available for 'battery_power_charging' with high retrieval confidence (1.00).
- **Drafted Reply:** "Let's try some steps to improve your battery performance: 1) Go to Settings > Battery > Battery Health to check maximum capacity. 2) Disable Background App Refresh for non-essential apps. 3) Turn off Location Services for apps that don't need it. If battery health is below 80%, a service replacement may be needed."

#### Example 2 (Tweet ID: `115012`)
- **Customer Query:** "The lock/power button on my iPhone iPhone SE is physically stuck inside the frame."
- **Intent:** Gold = `device_hardware_damage` | Predicted = `battery_power_charging`
- **Routing:** Gold = `ESCALATE` | Predicted = `AUTO`
- **Stated Routing Reason:** Standard self-service troubleshooting available for 'battery_power_charging' with high retrieval confidence (1.00).
- **Drafted Reply:** "We'd like to help with your charging concern. First, try a different Lightning/USB-C cable and power adapter. Clean the charging port gently with a soft brush. If the issue persists, try a forced restart (varies by model). Let us know your Iphone model and iOS version for specific steps!"

#### Example 3 (Tweet ID: `115022`)
- **Customer Query:** "The lock/power button on my iPhone iPhone SE is physically stuck inside the frame. Please fix this @AppleSupport!"
- **Intent:** Gold = `device_hardware_damage` | Predicted = `battery_power_charging`
- **Routing:** Gold = `ESCALATE` | Predicted = `AUTO`
- **Stated Routing Reason:** Standard self-service troubleshooting available for 'battery_power_charging' with high retrieval confidence (0.99).
- **Drafted Reply:** "Battery issues can often be resolved! Please try: Settings > General > Reset > Reset All Settings (this won't delete data). Also check Settings > Battery to identify which apps are consuming the most power. If your Iphone runs iOS 11.3+, check Battery Health for degradation status."

#### Example 4 (Tweet ID: `116200`)
- **Customer Query:** "Wi-Fi button is greyed out in Settings and cannot be toggled on."
- **Intent:** Gold = `connectivity_wifi_bluetooth` | Predicted = `connectivity_wifi_bluetooth`
- **Routing:** Gold = `ESCALATE` | Predicted = `AUTO`
- **Stated Routing Reason:** Standard self-service troubleshooting available for 'connectivity_wifi_bluetooth' with high retrieval confidence (1.00).
- **Drafted Reply:** "Connectivity problems can often be fixed with these steps: 1) Restart your router/modem if it's a Wi-Fi issue. 2) On your device, go to Settings > Wi-Fi, tap the (i) next to your network, and tap Forget This Network, then rejoin. 3) Ensure your iOS is up to date. If Bluetooth, try unpairing and re-pairing the device."

#### Example 5 (Tweet ID: `116204`)
- **Customer Query:** "Bluetooth cannot discover my car audio system or external speaker. Please fix this @AppleSupport!"
- **Intent:** Gold = `connectivity_wifi_bluetooth` | Predicted = `audio_sound_accessories`
- **Routing:** Gold = `AUTO` | Predicted = `AUTO`
- **Stated Routing Reason:** Standard self-service troubleshooting available for 'audio_sound_accessories' with high retrieval confidence (0.99).
- **Drafted Reply:** "We're here to help with your audio concern! First, check if your device is stuck in headphone mode: plug in and unplug headphones several times. Go to Settings > Sounds & Haptics and check the ringer slider. If the issue is with AirPods, make sure both are charging properly and firmware is up to date."

#### Example 6 (Tweet ID: `116210`)
- **Customer Query:** "Wi-Fi button is greyed out in Settings and cannot be toggled on. Please fix this @AppleSupport!"
- **Intent:** Gold = `connectivity_wifi_bluetooth` | Predicted = `connectivity_wifi_bluetooth`
- **Routing:** Gold = `ESCALATE` | Predicted = `AUTO`
- **Stated Routing Reason:** Standard self-service troubleshooting available for 'connectivity_wifi_bluetooth' with high retrieval confidence (0.99).
- **Drafted Reply:** "We'd like to help restore your connection! Try: Settings > Airplane Mode, toggle ON for 30 seconds, then OFF. If the issue persists, go to Settings > General > Transfer or Reset > Reset Network Settings. For Bluetooth devices, also check they're charged and within range."

#### Example 7 (Tweet ID: `116214`)
- **Customer Query:** "@AppleSupport Bluetooth cannot discover my car audio system or external speaker."
- **Intent:** Gold = `connectivity_wifi_bluetooth` | Predicted = `audio_sound_accessories`
- **Routing:** Gold = `AUTO` | Predicted = `AUTO`
- **Stated Routing Reason:** Standard self-service troubleshooting available for 'audio_sound_accessories' with high retrieval confidence (1.00).
- **Drafted Reply:** "We're here to help with your audio concern! First, check if your device is stuck in headphone mode: plug in and unplug headphones several times. Go to Settings > Sounds & Haptics and check the ringer slider. If the issue is with AirPods, make sure both are charging properly and firmware is up to date."

#### Example 8 (Tweet ID: `116220`)
- **Customer Query:** "@AppleSupport Wi-Fi button is greyed out in Settings and cannot be toggled on."
- **Intent:** Gold = `connectivity_wifi_bluetooth` | Predicted = `connectivity_wifi_bluetooth`
- **Routing:** Gold = `ESCALATE` | Predicted = `AUTO`
- **Stated Routing Reason:** Standard self-service troubleshooting available for 'connectivity_wifi_bluetooth' with high retrieval confidence (1.00).
- **Drafted Reply:** "Connectivity problems can often be fixed with these steps: 1) Restart your router/modem if it's a Wi-Fi issue. 2) On your device, go to Settings > Wi-Fi, tap the (i) next to your network, and tap Forget This Network, then rejoin. 3) Ensure your iOS is up to date. If Bluetooth, try unpairing and re-pairing the device."

#### Example 9 (Tweet ID: `116224`)
- **Customer Query:** "Bluetooth cannot discover my car audio system or external speaker."
- **Intent:** Gold = `connectivity_wifi_bluetooth` | Predicted = `audio_sound_accessories`
- **Routing:** Gold = `AUTO` | Predicted = `AUTO`
- **Stated Routing Reason:** Standard self-service troubleshooting available for 'audio_sound_accessories' with high retrieval confidence (1.00).
- **Drafted Reply:** "We're here to help with your audio concern! First, check if your device is stuck in headphone mode: plug in and unplug headphones several times. Go to Settings > Sounds & Haptics and check the ringer slider. If the issue is with AirPods, make sure both are charging properly and firmware is up to date."

#### Example 10 (Tweet ID: `116500`)
- **Customer Query:** "Left AirPod has zero sound and does not show charging status in the battery widget."
- **Intent:** Gold = `audio_sound_accessories` | Predicted = `battery_power_charging`
- **Routing:** Gold = `AUTO` | Predicted = `AUTO`
- **Stated Routing Reason:** Standard self-service troubleshooting available for 'battery_power_charging' with high retrieval confidence (1.00).
- **Drafted Reply:** "Battery issues can often be resolved! Please try: Settings > General > Reset > Reset All Settings (this won't delete data). Also check Settings > Battery to identify which apps are consuming the most power. If your device runs iOS 11.3+, check Battery Health for degradation status."

#### Example 11 (Tweet ID: `116506`)
- **Customer Query:** "AirPods disconnect randomly during workouts and reconnect after 10 seconds."
- **Intent:** Gold = `audio_sound_accessories` | Predicted = `connectivity_wifi_bluetooth`
- **Routing:** Gold = `AUTO` | Predicted = `AUTO`
- **Stated Routing Reason:** Standard self-service troubleshooting available for 'connectivity_wifi_bluetooth' with high retrieval confidence (1.00).
- **Drafted Reply:** "Connectivity problems can often be fixed with these steps: 1) Restart your router/modem if it's a Wi-Fi issue. 2) On your Airpods, go to Settings > Wi-Fi, tap the (i) next to your network, and tap Forget This Network, then rejoin. 3) Ensure your iOS is up to date. If Bluetooth, try unpairing and re-pairing the device."

#### Example 12 (Tweet ID: `116508`)
- **Customer Query:** "@AppleSupport Sound crackling in right earbud when Active Noise Cancellation or Transparency is on."
- **Intent:** Gold = `audio_sound_accessories` | Predicted = `audio_sound_accessories`
- **Routing:** Gold = `ESCALATE` | Predicted = `AUTO`
- **Stated Routing Reason:** Standard self-service troubleshooting available for 'audio_sound_accessories' with high retrieval confidence (1.00).
- **Drafted Reply:** "For sound-related issues: 1) Clean your speaker grills and Lightning/USB-C port gently with a soft brush. 2) Check Settings > Sounds & Haptics to adjust volume. 3) Try playing audio through a different app to isolate whether it's app-specific. 4) If using AirPods, forget them in Bluetooth settings and re-pair."

#### Example 13 (Tweet ID: `116510`)
- **Customer Query:** "Left AirPod has zero sound and does not show charging status in the battery widget. Please fix this @AppleSupport!"
- **Intent:** Gold = `audio_sound_accessories` | Predicted = `battery_power_charging`
- **Routing:** Gold = `AUTO` | Predicted = `AUTO`
- **Stated Routing Reason:** Standard self-service troubleshooting available for 'battery_power_charging' with high retrieval confidence (0.99).
- **Drafted Reply:** "Let's try some steps to improve your battery performance: 1) Go to Settings > Battery > Battery Health to check maximum capacity. 2) Disable Background App Refresh for non-essential apps. 3) Turn off Location Services for apps that don't need it. If battery health is below 80%, a service replacement may be needed."

#### Example 14 (Tweet ID: `116516`)
- **Customer Query:** "AirPods disconnect randomly during workouts and reconnect after 10 seconds. Please fix this @AppleSupport!"
- **Intent:** Gold = `audio_sound_accessories` | Predicted = `connectivity_wifi_bluetooth`
- **Routing:** Gold = `AUTO` | Predicted = `AUTO`
- **Stated Routing Reason:** Standard self-service troubleshooting available for 'connectivity_wifi_bluetooth' with high retrieval confidence (0.99).
- **Drafted Reply:** "Connectivity problems can often be fixed with these steps: 1) Restart your router/modem if it's a Wi-Fi issue. 2) On your Airpods, go to Settings > Wi-Fi, tap the (i) next to your network, and tap Forget This Network, then rejoin. 3) Ensure your iOS is up to date. If Bluetooth, try unpairing and re-pairing the device."

#### Example 15 (Tweet ID: `116518`)
- **Customer Query:** "Sound crackling in right earbud when Active Noise Cancellation or Transparency is on."
- **Intent:** Gold = `audio_sound_accessories` | Predicted = `audio_sound_accessories`
- **Routing:** Gold = `ESCALATE` | Predicted = `AUTO`
- **Stated Routing Reason:** Standard self-service troubleshooting available for 'audio_sound_accessories' with high retrieval confidence (1.00).
- **Drafted Reply:** "For sound-related issues: 1) Clean your speaker grills and Lightning/USB-C port gently with a soft brush. 2) Check Settings > Sounds & Haptics to adjust volume. 3) Try playing audio through a different app to isolate whether it's app-specific. 4) If using AirPods, forget them in Bluetooth settings and re-pair."

