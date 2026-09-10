import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.schemas import (
    IntentType, RouteAction, IntentClassificationResult,
    DraftReplyResult, RouteDecision, AgentOutput, JudgeRubricScores
)


def test_intent_classification_schema():
    """Validate that IntentClassificationResult enforces valid intent enum and confidence range."""
    result = IntentClassificationResult(
        intent=IntentType.BATTERY_POWER_CHARGING,
        confidence=0.92,
        reasoning="Customer describes rapid battery drain"
    )
    assert result.intent == IntentType.BATTERY_POWER_CHARGING
    assert 0.0 <= result.confidence <= 1.0


def test_route_decision_schema():
    """Validate RouteDecision enforces valid action enum."""
    decision = RouteDecision(
        action=RouteAction.ESCALATE,
        stated_reason="Contains refund keyword requiring human oversight",
        confidence=1.0,
        hard_guardrail_triggered="Keyword_refund"
    )
    assert decision.action == RouteAction.ESCALATE
    assert decision.hard_guardrail_triggered == "Keyword_refund"


def test_judge_rubric_composite_score():
    """Validate composite score calculation across 5 dimensions."""
    scores = JudgeRubricScores(
        relevance=5, groundedness=4, actionability=5, tone_brand_fit=4, safety=5
    )
    assert scores.composite_score == (5 + 4 + 5 + 4 + 5) / 5.0
    assert scores.composite_score == 4.6


def test_draft_reply_hallucination_flag():
    """Verify hallucination_risk defaults to False and can be set."""
    draft = DraftReplyResult(
        reply="Please restart your device.",
        confidence=0.9
    )
    assert draft.hallucination_risk is False

    draft_risky = DraftReplyResult(
        reply="Your refund of $499 has been processed.",
        confidence=0.5,
        hallucination_risk=True
    )
    assert draft_risky.hallucination_risk is True


def test_agent_output_full_pipeline():
    """Validate end-to-end AgentOutput schema with all nested fields."""
    output = AgentOutput(
        tweet_id="123456",
        inbound_text="My iPhone battery drains in 2 hours",
        intent=IntentClassificationResult(
            intent=IntentType.BATTERY_POWER_CHARGING,
            confidence=0.95,
            reasoning="Battery drain keywords"
        ),
        draft=DraftReplyResult(
            reply="We'd be happy to help! Try restarting your device.",
            grounded_in_ids=["ref_001"],
            confidence=0.88
        ),
        routing=RouteDecision(
            action=RouteAction.AUTO,
            stated_reason="Standard troubleshooting procedure",
            confidence=0.92
        ),
        latency_ms=245.3
    )
    assert output.tweet_id == "123456"
    assert output.intent.intent == IntentType.BATTERY_POWER_CHARGING
    assert output.routing.action == RouteAction.AUTO


if __name__ == "__main__":
    test_intent_classification_schema()
    test_route_decision_schema()
    test_judge_rubric_composite_score()
    test_draft_reply_hallucination_flag()
    test_agent_output_full_pipeline()
    print("All schema tests passed!")
