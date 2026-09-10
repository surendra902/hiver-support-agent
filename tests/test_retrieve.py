import sys
import os
import pytest
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.retrieve import RetrievalIndex
from src.schemas import HistoricalResolution


@pytest.fixture
def sample_dataframe():
    data = {
        "customer_tweet_id": ["101", "102", "103", "104", "105"],
        "customer_text": [
            "My iPhone battery drains from 100 to 20 in an hour",
            "iOS 11.2 update camera app crashes constantly",
            "Wi-Fi keeps disconnecting every few minutes",
            "Screen cracked after dropping phone on floor",
            "Apple ID locked and cannot receive 2FA verification code"
        ],
        "brand_reply": [
            "We'd be glad to help. Let's check Settings > Battery to see which app is using the most power.",
            "Thanks for reaching out. Try force closing the camera app and restarting your iPhone.",
            "Let's get this resolved. Try resetting network settings in Settings > General > Reset.",
            "We can help setup a repair. Visit an Apple Store Genius Bar or start a mail-in repair.",
            "Please visit iforgot.apple.com to initiate secure account recovery."
        ],
        "is_deflection": [False, False, False, False, True]
    }
    return pd.DataFrame(data)


def test_retrieval_index_build_and_search(sample_dataframe):
    index = RetrievalIndex(top_k=3, min_threshold=0.05)
    index.build_index(sample_dataframe)
    assert index.fitted is True

    # Search for battery issue
    results = index.search("battery dying really fast", k=2)
    assert len(results) > 0
    assert isinstance(results[0], HistoricalResolution)
    assert results[0].tweet_id == "101"
    assert "battery" in results[0].brand_reply.lower() or "battery" in results[0].customer_query.lower()


def test_retrieval_index_save_and_load(sample_dataframe, tmp_path):
    index = RetrievalIndex(top_k=2, min_threshold=0.05)
    index.build_index(sample_dataframe)

    save_file = str(tmp_path / "test_index.pkl")
    index.save(save_file)
    assert os.path.exists(save_file)

    loaded_index = RetrievalIndex.load(save_file)
    assert loaded_index.fitted is True
    results = loaded_index.search("camera crash update", k=1)
    assert len(results) == 1
    assert results[0].tweet_id == "102"


if __name__ == "__main__":
    df = sample_dataframe()
    test_retrieval_index_build_and_search(df)
    print("All retrieval tests passed successfully!")
