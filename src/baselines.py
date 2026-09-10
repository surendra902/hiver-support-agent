import logging
from typing import List, Optional
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from src.schemas import (
    IntentType, RouteAction, IntentClassificationResult,
    DraftReplyResult, RouteDecision, AgentOutput
)
from src.retrieve import RetrievalIndex

logger = logging.getLogger(__name__)

# Standard historical canned response from AppleSupport
APPLE_MOST_COMMON_CANNED_REPLY = (
    "Thanks for reaching out! We'd be glad to help with this. "
    "Which device model and version of iOS are you currently running?"
)


class TrivialBaseline:
    """
    Baseline 1: Trivial Floor.
    - Intent: Always predicts the empirical majority intent.
    - Routing: Always ESCALATES.
    - Reply: Emits the single most frequent canned greeting.
    """

    def __init__(self, majority_intent: IntentType = IntentType.SOFTWARE_UPDATE_BUG):
        self.majority_intent = majority_intent

    def predict(self, tweet_id: str, query: str) -> AgentOutput:
        return AgentOutput(
            tweet_id=tweet_id,
            inbound_text=query,
            intent=IntentClassificationResult(
                intent=self.majority_intent,
                confidence=0.50,
                reasoning="Trivial baseline: majority class prediction"
            ),
            retrieved_exemplars=[],
            draft=DraftReplyResult(
                reply=APPLE_MOST_COMMON_CANNED_REPLY,
                grounded_in_ids=[],
                actionable_steps=["Ask for device model", "Ask for iOS version"],
                confidence=0.50,
                hallucination_risk=False
            ),
            routing=RouteDecision(
                action=RouteAction.ESCALATE,
                stated_reason="Trivial baseline policy: always escalate all inbound queries to human agents.",
                confidence=1.0,
                hard_guardrail_triggered="AlwaysEscalatePolicy"
            ),
            latency_ms=1.0
        )


class SimpleMLBaseline:
    """
    Baseline 2: Classical ML & Verbatim Retrieval.
    - Intent: TF-IDF + Logistic Regression trained on silver/dev data.
    - Routing: Heuristic threshold on classifier probability.
    - Reply: 1-NN verbatim copy of the most similar historical reply (no LLM rewriting).
    """

    def __init__(self, retrieval_index: RetrievalIndex):
        self.retrieval_index = retrieval_index
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
        self.classifier = LogisticRegression(max_iter=1000, class_weight="balanced")
        self.trained = False

    def train_classifier(self, texts: List[str], labels: List[str]) -> None:
        """Train TF-IDF + Logistic Regression on seed intent labels."""
        X = self.vectorizer.fit_transform(texts)
        self.classifier.fit(X, labels)
        self.trained = True
        logger.info(f"SimpleML baseline trained on {len(texts)} examples across {len(self.classifier.classes_)} classes.")

    def predict(self, tweet_id: str, query: str) -> AgentOutput:
        # 1. Intent prediction
        if self.trained:
            X = self.vectorizer.transform([query])
            pred_intent_str = self.classifier.predict(X)[0]
            probs = self.classifier.predict_proba(X)[0]
            confidence = float(probs.max())
            try:
                pred_intent = IntentType(pred_intent_str)
            except ValueError:
                pred_intent = IntentType.COMPLAINT_FEEDBACK_OTHER
        else:
            pred_intent = IntentType.BATTERY_POWER_CHARGING
            confidence = 0.50

        # 2. Retrieval: 1-NN historical reply verbatim
        exemplars = self.retrieval_index.search(query, k=1)
        if exemplars:
            verbatim_reply = exemplars[0].brand_reply
            grounded_ids = [exemplars[0].tweet_id]
            sim = exemplars[0].similarity
        else:
            verbatim_reply = APPLE_MOST_COMMON_CANNED_REPLY
            grounded_ids = []
            sim = 0.0

        # 3. Simple heuristic routing
        escalate_keywords = ["refund", "stolen", "charge", "lawyer", "unauthorized", "locked"]
        has_keyword = any(kw in query.lower() for kw in escalate_keywords)

        if has_keyword or confidence < 0.60 or sim < 0.30:
            action = RouteAction.ESCALATE
            reason = "Simple ML baseline: query triggered escalation keyword or low retrieval/intent confidence."
        else:
            action = RouteAction.AUTO
            reason = "Simple ML baseline: query matches historical resolution above confidence threshold."

        return AgentOutput(
            tweet_id=tweet_id,
            inbound_text=query,
            intent=IntentClassificationResult(
                intent=pred_intent,
                confidence=round(confidence, 3),
                reasoning="Logistic regression class probability"
            ),
            retrieved_exemplars=exemplars,
            draft=DraftReplyResult(
                reply=verbatim_reply,
                grounded_in_ids=grounded_ids,
                actionable_steps=["Follow historical resolution steps"],
                confidence=round(sim, 3),
                hallucination_risk=False
            ),
            routing=RouteDecision(
                action=action,
                stated_reason=reason,
                confidence=round(confidence, 3),
                hard_guardrail_triggered="KeywordFilter" if has_keyword else None
            ),
            latency_ms=5.0
        )
