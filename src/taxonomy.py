from typing import Dict, List, Any
from src.schemas import IntentType

TAXONOMY_DEFINITIONS: Dict[IntentType, Dict[str, Any]] = {
    IntentType.DEVICE_HARDWARE_DAMAGE: {
        "title": "Device Hardware Damage & Physical Repair",
        "definition": (
            "Physical breakage, damaged screen, unresponsive buttons, water exposure, camera lens crack, "
            "vibration motor failure, or requests for Apple Store Genius Bar physical repair and replacement."
        ),
        "positive_examples": [
            "Dropped my iPhone X on concrete and the OLED screen is completely cracked and flickering green.",
            "My lock button is physically stuck inside the frame and won't click anymore."
        ],
        "near_miss_example": "My phone screen turns black when I open Instagram. (Software bug, not physical damage)"
    },
    IntentType.BATTERY_POWER_CHARGING: {
        "title": "Battery Drain, Charging & Power",
        "definition": (
            "Severe battery drain, device shutting down at 30%, iPhone not charging when plugged in, "
            "overheating while charging, Lightning cable/port issues, or battery health degradation."
        ),
        "positive_examples": [
            "My iPhone battery drains from 100% to 15% in 2 hours without any apps running in background.",
            "Plugged my phone in overnight with official charger and it won't charge past 4%."
        ],
        "near_miss_example": "My phone gets warm when playing heavy 3D games for 3 hours. (Expected thermal behavior / performance, not battery fault)"
    },
    IntentType.SOFTWARE_UPDATE_BUG: {
        "title": "Software Update, iOS Bugs & System Crashes",
        "definition": (
            "Issues arising immediately after updating iOS/macOS, update installation freezing/stuck on Apple logo, "
            "app crashing loops, keyboard lag, storage bug ('System Data taking 50GB'), or OS glitch."
        ),
        "positive_examples": [
            "Ever since updating to iOS 11.2, my camera app crashes every time I switch to portrait mode.",
            "iOS update has been stuck on 'Estimating time remaining...' for 6 hours."
        ],
        "near_miss_example": "Spotify crashes when playing offline music. (Third-party app issue, not core OS update bug)"
    },
    IntentType.APPLE_ID_ICLOUD_SECURITY: {
        "title": "Apple ID, iCloud, 2FA & Account Security",
        "definition": (
            "Apple ID locked for security reasons, forgotten passwords, two-factor authentication codes not received, "
            "iCloud backup failures, storage purchase sync issues, or Activation Lock."
        ),
        "positive_examples": [
            "My Apple ID has been locked and the recovery email linked to it is an old defunct account.",
            "Can't sign into iCloud on my new iPad because the verification SMS is going to my broken old phone."
        ],
        "near_miss_example": "I got an email saying I bought a $500 gift card from Apple, is this a scam? (Phishing inquiry, route with care)"
    },
    IntentType.CONNECTIVITY_WIFI_BLUETOOTH: {
        "title": "Connectivity: Wi-Fi, Bluetooth & Cellular",
        "definition": (
            "Wi-Fi toggle greyed out, inability to connect to home network, Bluetooth dropping connection to car/speakers, "
            "or persistent 'No Service' / 'Searching' cellular error."
        ),
        "positive_examples": [
            "My iPhone Wi-Fi keeps disconnecting every 5 minutes and says 'Incorrect Password' even when it's correct.",
            "Bluetooth won't discover my car audio system after resetting network settings."
        ],
        "near_miss_example": "AirPods won't connect to my Mac. (Classify under audio_sound_accessories if specific to AirPods)"
    },
    IntentType.AUDIO_SOUND_ACCESSORIES: {
        "title": "Audio, Speakers, Microphones & AirPods",
        "definition": (
            "AirPods one-side not working, crackling sound, charging case not holding charge, iPhone ear speaker muffled, "
            "microphone not picking up voice during calls, or headphone dongle issues."
        ),
        "positive_examples": [
            "The left AirPod has zero sound and the case doesn't show the amber charging light for it.",
            "People on phone calls can't hear me unless I turn on speakerphone."
        ],
        "near_miss_example": "Siri doesn't respond when I say Hey Siri. (May be Siri voice recognition setting or connectivity)"
    },
    IntentType.STORE_BILLING_PURCHASE: {
        "title": "App Store, Billing, Subscriptions & Order Status",
        "definition": (
            "Unauthorized card charges from iTunes/App Store, subscription cancellation & refund requests, "
            "Apple Store online order tracking, delivery delays, or gift card redemption errors."
        ),
        "positive_examples": [
            "I was charged $9.99 for a subscription I cancelled during the free trial last week. I want a refund.",
            "Where is my Apple order #W123456789? It was supposed to be delivered yesterday."
        ],
        "near_miss_example": "Can't download apps because my Apple ID is disabled in App Store. (Dual intent: ID lock vs store; prioritize security/ID)"
    },
    IntentType.COMPLAINT_FEEDBACK_OTHER: {
        "title": "General Complaints, Sentiment & Unactionable Feedback",
        "definition": (
            "General venting, sarcasm, brand praise/criticism, feature requests, marketing inquiries, or vague "
            "statements lacking any troubleshooting detail ('Apple sucks', 'Why did you remove the headphone jack')."
        ),
        "positive_examples": [
            "Apple is honestly the worst company ever, so sick of these overpriced dongles.",
            "Whoever designed the new music app UI should be fired immediately."
        ],
        "near_miss_example": "Your phones are trash, my screen broke after a tiny drop from 1 foot. (Vent + hardware; if asking for remedy, hardware)"
    }
}


def get_taxonomy_prompt_text() -> str:
    """Formats taxonomy for LLM classification prompt."""
    lines = []
    for intent, data in TAXONOMY_DEFINITIONS.items():
        lines.append(f"### {intent.value}: {data['title']}")
        lines.append(f"Definition: {data['definition']}")
        lines.append("Positive Examples:")
        for ex in data['positive_examples']:
            lines.append(f"  - \"{ex}\"")
        lines.append(f"Near-miss: \"{data['near_miss_example']}\"\n")
    return "\n".join(lines)
