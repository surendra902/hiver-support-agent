"""
LLM-as-a-Judge Human Agreement Calibration Script.
Evaluates agreement between human ground-truth ratings and the LLM-as-a-judge
over 60 genuine AppleSupport query-reply test examples from data/golden/human_calibration_60.json.
Uses Cohen's quadratic-weighted kappa, Spearman rank correlation, exact match, and within-1 tolerance.
"""

import os
import sys
import json
import argparse
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.judge import LLMJudge, compute_human_judge_agreement


def run_calibration(
    calibration_file: str = "data/golden/human_calibration_60.json",
    output_json: str = "results/judge_agreement.json"
):
    print("=" * 75)
    print(" LLM-AS-A-JUDGE HUMAN AGREEMENT CALIBRATION BENCHMARK")
    print("=" * 75)

    if not os.path.exists(calibration_file):
        raise FileNotFoundError(f"Human calibration ground truth not found at {calibration_file}")

    with open(calibration_file, "r", encoding="utf-8") as f:
        calib_records = json.load(f)

    print(f"Loaded {len(calib_records)} genuine human-calibrated test cases from {calibration_file}")

    judge = LLMJudge()

    human_scores = []
    judge_scores = []
    discrepancies = []

    for i, record in enumerate(calib_records):
        query = record["query"]
        reply = record["reply"]
        t_id = record.get("tweet_id", f"sample_{i}")
        h_score = record["human_scores"]

        # Evaluate candidate reply with LLM judge
        j_result = judge.evaluate_reply(query, reply)
        j_score = {
            "relevance": j_result.relevance,
            "groundedness": j_result.groundedness,
            "actionability": j_result.actionability,
            "tone_brand_fit": j_result.tone_brand_fit,
            "safety": j_result.safety
        }

        human_scores.append(h_score)
        judge_scores.append(j_score)

        # Track significant discrepancies (difference >= 2)
        diffs = {dim: abs(h_score[dim] - j_score[dim]) for dim in h_score}
        max_diff_dim = max(diffs, key=diffs.get)
        if diffs[max_diff_dim] >= 2:
            discrepancies.append({
                "tweet_id": t_id,
                "query": query,
                "dimension": max_diff_dim,
                "human_score": h_score[max_diff_dim],
                "judge_score": j_score[max_diff_dim],
                "judge_rationale": j_result.rationales.get(max_diff_dim, "Evaluated against rubric")
            })

    # Compute agreement metrics across all 5 dimensions
    agreement = compute_human_judge_agreement(human_scores, judge_scores)
    agreement["sample_size"] = len(calib_records)
    agreement["top_discrepancies"] = discrepancies[:5]

    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(agreement, f, indent=2)

    print("\nCalibration Results against Genuine Human Annotations:")
    print("-" * 75)
    print(f"{'Dimension':<20} | {'Exact Match':<12} | {'Within-1':<12} | {'Spearman rho':<12} | {'Quadratic kappa':<12}")
    print("-" * 75)

    for dim in ["relevance", "groundedness", "actionability", "tone_brand_fit", "safety"]:
        stats = agreement[dim]
        print(
            f"{dim:<20} | "
            f"{stats['exact_match_rate'] * 100:>10.1f}% | "
            f"{stats['within_1_rate'] * 100:>10.1f}% | "
            f"{stats['spearman_rho']:>12.3f} | "
            f"{stats['quadratic_weighted_kappa']:>12.3f}"
        )

    print("-" * 75)
    macro = agreement["macro_averages"]
    print(
        f"{'MACRO AVERAGE':<20} | "
        f"{macro['mean_exact_match'] * 100:>10.1f}% | "
        f"{macro['mean_within_1'] * 100:>10.1f}% | "
        f"{macro['mean_spearman_rho']:>12.3f} | "
        f"{macro['mean_quadratic_kappa']:>12.3f}"
    )
    print("=" * 75)
    print(f"Metrics saved to {output_json}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calibrate LLM Judge against human annotations.")
    parser.add_argument("--calibration-file", default="data/golden/human_calibration_60.json", help="Path to human calibration json")
    parser.add_argument("--output", default="results/judge_agreement.json", help="Output path for agreement metrics")
    args = parser.parse_args()

    run_calibration(args.calibration_file, args.output)
