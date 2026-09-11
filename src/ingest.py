import os
import glob
import json
import logging
from typing import Optional, Tuple
import pandas as pd
import numpy as np

from src.clean import clean_tweet_text, is_deflection_reply

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def find_twcs_csv(raw_dir: str = "data/raw") -> Optional[str]:
    """Locate twcs.csv in local raw dir or kagglehub cache."""
    # Check direct local dir
    local_path = os.path.join(raw_dir, "twcs.csv")
    if os.path.exists(local_path):
        return local_path
    
    # Check kagglehub cache
    home = os.path.expanduser("~")
    kaggle_pattern = os.path.join(home, ".cache", "kagglehub", "datasets", "thoughtvector", "customer-support-on-twitter", "**", "twcs.csv")
    matches = glob.glob(kaggle_pattern, recursive=True)
    if matches:
        return matches[0]
    
    # Also check without version subdir
    alt_pattern = os.path.join(home, ".cache", "kagglehub", "datasets", "thoughtvector", "customer-support-on-twitter", "twcs.csv")
    if os.path.exists(alt_pattern):
        return alt_pattern
    
    return None


def run_brand_volume_analysis(csv_path: str, output_path: str = "results/brand_counts.json") -> dict:
    """Analyze top brand frequencies to substantiate brand selection in the report."""
    logger.info("Computing brand volumes across twcs.csv chunks...")
    target_brands = ["AppleSupport", "AmazonHelp", "Uber_Support", "SpotifyCares", "Delta", "British_Airways"]
    brand_stats = {b: {"inbound_mentions": 0, "brand_replies": 0, "total": 0} for b in target_brands}
    
    # Process in chunks to maintain low memory footprint
    chunksize = 200_000
    for chunk in pd.read_csv(csv_path, chunksize=chunksize, usecols=["author_id", "inbound", "text"], dtype=str):
        for brand in target_brands:
            brand_replies = (chunk["author_id"] == brand).sum()
            brand_stats[brand]["brand_replies"] += int(brand_replies)
            
            # Mentions in inbound tweets
            inbounds = chunk[chunk["inbound"].str.lower() == "true"]
            mentions = inbounds["text"].str.contains(f"@{brand}", case=False, na=False).sum()
            brand_stats[brand]["inbound_mentions"] += int(mentions)
            brand_stats[brand]["total"] += int(brand_replies + mentions)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(brand_stats, f, indent=2)
    logger.info(f"Brand volume statistics saved to {output_path}")
    return brand_stats


def extract_apple_turn1_pairs(
    csv_path: str,
    output_parquet: str = "data/sample/brand_sample.parquet",
    sample_size: int = 10000,
    seed: int = 42
) -> pd.DataFrame:
    """
    Stream twcs.csv, extract (customer_turn1, apple_reply) pairs, clean, and sample.
    Uses chunked reading to prevent OOM errors on large CSV.
    """
    logger.info(f"Starting chunked Turn-1 pair extraction for AppleSupport from {csv_path}...")
    chunksize = 150_000
    
    apple_replies = []
    customer_candidates = []

    # Read needed columns
    cols = ["tweet_id", "author_id", "inbound", "created_at", "text", "response_tweet_id", "in_response_to_tweet_id"]
    
    for i, chunk in enumerate(pd.read_csv(csv_path, chunksize=chunksize, usecols=cols, dtype=str)):
        # Normalize booleans
        chunk["inbound_bool"] = chunk["inbound"].str.lower() == "true"
        
        # 1. Collect all AppleSupport outbound replies
        apple_chunk = chunk[(~chunk["inbound_bool"]) & (chunk["author_id"] == "AppleSupport")]
        if not apple_chunk.empty:
            apple_replies.append(apple_chunk[["tweet_id", "created_at", "text", "in_response_to_tweet_id"]])
        
        # 2. Collect potential customer opening messages (inbound with no in_response_to)
        cust_chunk = chunk[chunk["inbound_bool"] & chunk["in_response_to_tweet_id"].isna()]
        # Filter for mentions of AppleSupport
        cust_chunk = cust_chunk[cust_chunk["text"].str.contains(r"@AppleSupport", case=False, na=False)]
        if not cust_chunk.empty:
            customer_candidates.append(cust_chunk[["tweet_id", "created_at", "text"]])
        
        if (i + 1) % 5 == 0:
            logger.info(f"Processed { (i + 1) * chunksize } rows...")

    df_apple = pd.concat(apple_replies, ignore_index=True)
    df_cust = pd.concat(customer_candidates, ignore_index=True)
    logger.info(f"Found {len(df_cust)} customer opening tweets and {len(df_apple)} AppleSupport replies.")

    # Merge customer opening tweet with Apple's response
    merged = pd.merge(
        df_cust,
        df_apple,
        left_on="tweet_id",
        right_on="in_response_to_tweet_id",
        suffixes=("_cust", "_apple")
    )
    logger.info(f"Merged {len(merged)} Turn-1 (question, reply) resolution pairs.")

    # Deduplicate by customer tweet_id (take first brand reply)
    merged = merged.drop_duplicates(subset=["tweet_id_cust"]).copy()

    # Clean texts
    merged["cleaned_customer_text"] = merged["text_cust"].apply(clean_tweet_text)
    merged["cleaned_apple_reply"] = merged["text_apple"].apply(lambda t: clean_tweet_text(t, scrub_handles=False))
    merged["is_deflection"] = merged["text_apple"].apply(is_deflection_reply)

    # Filter out empty or very short noise (< 10 chars)
    valid_mask = (merged["cleaned_customer_text"].str.len() >= 10) & (merged["cleaned_apple_reply"].str.len() >= 10)
    merged = merged[valid_mask].copy()

    # Subsample to deterministic target size
    if len(merged) > sample_size:
        sampled = merged.sample(n=sample_size, random_state=seed).reset_index(drop=True)
    else:
        sampled = merged.reset_index(drop=True)

    # Rename final columns
    final_df = sampled[[
        "tweet_id_cust", "created_at_cust", "cleaned_customer_text",
        "tweet_id_apple", "cleaned_apple_reply", "is_deflection"
    ]].rename(columns={
        "tweet_id_cust": "customer_tweet_id",
        "created_at_cust": "timestamp",
        "cleaned_customer_text": "customer_text",
        "tweet_id_apple": "brand_tweet_id",
        "cleaned_apple_reply": "brand_reply"
    })

    os.makedirs(os.path.dirname(output_parquet), exist_ok=True)
    try:
        final_df.to_parquet(output_parquet, index=False)
        logger.info(f"Saved {len(final_df)} sampled pairs to {output_parquet}")
    except Exception as e:
        csv_fallback = output_parquet.replace(".parquet", ".csv")
        final_df.to_csv(csv_fallback, index=False)
        logger.info(f"Parquet engine fallback: saved {len(final_df)} pairs to {csv_fallback}")

    return final_df


if __name__ == "__main__":
    csv_path = find_twcs_csv()
    if csv_path:
        print(f"Found twcs.csv at: {csv_path}")
        extract_apple_turn1_pairs(csv_path)
    else:
        print("twcs.csv not found yet. Please ensure dataset download completes.")
