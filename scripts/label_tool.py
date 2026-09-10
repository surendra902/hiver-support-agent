import os
import json
import argparse
from typing import List, Dict, Any
import pandas as pd

from src.schemas import IntentType, RouteAction

INTENT_MAP = {
    "1": IntentType.DEVICE_HARDWARE_DAMAGE.value,
    "2": IntentType.BATTERY_POWER_CHARGING.value,
    "3": IntentType.SOFTWARE_UPDATE_BUG.value,
    "4": IntentType.APPLE_ID_ICLOUD_SECURITY.value,
    "5": IntentType.CONNECTIVITY_WIFI_BLUETOOTH.value,
    "6": IntentType.AUDIO_SOUND_ACCESSORIES.value,
    "7": IntentType.STORE_BILLING_PURCHASE.value,
    "8": IntentType.COMPLAINT_FEEDBACK_OTHER.value,
}


def run_interactive_labeler(sample_parquet: str, output_golden: str, target_count: int = 200):
    """Micro-labeling CLI enabling fast 1-key annotation for golden set creation."""
    print("=" * 70)
    print(" HIVER GOLDEN EVALUATION SET: INTERACTIVE LABELING CLI")
    print("=" * 70)
    print("Hotkeys for Intent:")
    for k, v in INTENT_MAP.items():
        print(f"  [{k}] {v}")
    print("Hotkeys for Routing: [a] AUTO | [e] ESCALATE")
    print("=" * 70)

    # Load existing labels if resuming
    existing = []
    seen_ids = set()
    if os.path.exists(output_golden):
        with open(output_golden, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    existing.append(item)
                    seen_ids.add(item["tweet_id"])
        print(f"Loaded {len(existing)} previously labeled examples.")

    df = pd.read_parquet(sample_parquet) if sample_parquet.endswith(".parquet") else pd.read_csv(sample_parquet)
    candidates = df[~df["customer_tweet_id"].astype(str).isin(seen_ids)].to_dict(orient="records")

    labeled_count = len(existing)
    with open(output_golden, "a", encoding="utf-8") as out_f:
        for idx, row in enumerate(candidates):
            if labeled_count >= target_count:
                print(f"Target count of {target_count} golden examples achieved!")
                break

            print(f"\n[{labeled_count + 1}/{target_count}] Tweet ID: {row['customer_tweet_id']}")
            print(f"Customer Tweet: \"{row['customer_text']}\"")
            print(f"Actual Historical Apple Reply: \"{row['brand_reply']}\"")

            # Intent selection
            intent = None
            while not intent:
                i_input = input("Select Intent [1-8] (or 'q' to quit): ").strip()
                if i_input.lower() == "q":
                    print("Exiting and saving progress.")
                    return
                intent = INTENT_MAP.get(i_input)
                if not intent:
                    print("Invalid choice. Enter 1 through 8.")

            # Route selection
            route = None
            while not route:
                r_input = input("Select Route [a=AUTO / e=ESCALATE]: ").strip().lower()
                if r_input == "a":
                    route = RouteAction.AUTO.value
                elif r_input == "e":
                    route = RouteAction.ESCALATE.value
                else:
                    print("Invalid choice. Enter 'a' or 'e'.")

            reason = input("Route Reason (optional, press Enter for default): ").strip()
            if not reason:
                reason = "Routine troubleshooting guidance" if route == RouteAction.AUTO.value else "Requires account access or hardware inspection"

            entry = {
                "tweet_id": str(row["customer_tweet_id"]),
                "text": str(row["customer_text"]),
                "gold_intent": intent,
                "gold_route": route,
                "gold_route_reason": reason,
                "historical_reference_reply": str(row["brand_reply"]),
                "labeller_confidence": 3,
                "is_adversarial_boundary": False
            }

            out_f.write(json.dumps(entry) + "\n")
            out_f.flush()
            labeled_count += 1
            print(f"Saved: {intent} -> {route}")

    print(f"\nLabeling complete. Saved {labeled_count} examples to {output_golden}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", default="data/sample/brand_sample.parquet")
    parser.add_argument("--output", default="data/golden/golden_v1.jsonl")
    parser.add_argument("--count", type=int, default=200)
    args = parser.parse_args()

    run_interactive_labeler(args.sample, args.output, args.count)
