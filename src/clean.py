import re
from typing import Tuple

# Common regex patterns in Twitter customer support
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
HANDLE_PATTERN = re.compile(r"@[\w_]+")
MULTIPLE_SPACES = re.compile(r"\s+")
DEFLECTION_PATTERNS = [
    re.compile(r"\b(send|dm|direct message)\b.*?\b(us|details|apple)\b", re.IGNORECASE),
    re.compile(r"\b(reach out in a dm|pm us|in dm)\b", re.IGNORECASE),
    re.compile(r"\b(send us a direct message|dm your)\b", re.IGNORECASE)
]


def clean_tweet_text(text: str, scrub_handles: bool = True) -> str:
    """Normalize raw tweet text: remove noise, normalize URLs and handles."""
    if not isinstance(text, str):
        return ""
    
    # Replace URLs with standard placeholder
    cleaned = URL_PATTERN.sub("[URL]", text)
    
    # Optionally scrub handles or normalize target handle
    if scrub_handles:
        cleaned = re.sub(r"@AppleSupport", "", cleaned, flags=re.IGNORECASE)
        cleaned = HANDLE_PATTERN.sub("@user", cleaned)
    
    # Normalize whitespace
    cleaned = MULTIPLE_SPACES.sub(" ", cleaned).strip()
    return cleaned


def is_deflection_reply(reply_text: str) -> bool:
    """Detect if a brand reply is a pure redirect to DM rather than an in-thread resolution."""
    if not isinstance(reply_text, str):
        return False
    text = reply_text.lower()
    for pattern in DEFLECTION_PATTERNS:
        if pattern.search(text):
            return True
    return False


def extract_key_entities(text: str) -> Tuple[list, list]:
    """Extract numbers, versions (e.g. iOS 11.2), and device models from text."""
    versions = re.findall(r"\b(?:ios|macOS|watchOS|tvOS)\s*\d+(?:\.\d+)*\b", text, re.IGNORECASE)
    numbers = re.findall(r"\b\d+(?:\.\d+)?%?\b", text)
    return versions, numbers
