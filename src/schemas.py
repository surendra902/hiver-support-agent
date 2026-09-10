from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class IntentType(str, Enum):
    DEVICE_HARDWARE_DAMAGE = "device_hardware_damage"
    BATTERY_POWER_CHARGING = "battery_power_charging"
    SOFTWARE_UPDATE_BUG = "software_update_bug"
    APPLE_ID_ICLOUD_SECURITY = "apple_id_icloud_security"
    CONNECTIVITY_WIFI_BLUETOOTH = "connectivity_wifi_bluetooth"
    AUDIO_SOUND_ACCESSORIES = "audio_sound_accessories"
    STORE_BILLING_PURCHASE = "store_billing_purchase"
    COMPLAINT_FEEDBACK_OTHER = "complaint_feedback_other"


class RouteAction(str, Enum):
    AUTO = "AUTO"
    ESCALATE = "ESCALATE"


class IntentClassificationResult(BaseModel):
    intent: IntentType
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score between 0.0 and 1.0")
    reasoning: str = Field(description="Brief one-sentence reasoning for the predicted intent")


class HistoricalResolution(BaseModel):
    tweet_id: str
    customer_query: str
    brand_reply: str
    similarity: float = Field(ge=0.0, le=1.0)
    is_deflection: bool = Field(default=False)


class DraftReplyResult(BaseModel):
    reply: str = Field(description="Drafted response to customer, max 2 tweets length")
    grounded_in_ids: List[str] = Field(default_factory=list, description="IDs of historical resolutions used to ground this answer")
    actionable_steps: List[str] = Field(default_factory=list, description="Concrete next steps provided to customer")
    confidence: float = Field(ge=0.0, le=1.0)
    hallucination_risk: bool = Field(default=False, description="Flagged if reply contains ungrounded entities or claims")


class RouteDecision(BaseModel):
    action: RouteAction
    stated_reason: str = Field(description="Mandatory clear explanation why this is auto-handled or escalated")
    confidence: float = Field(ge=0.0, le=1.0)
    hard_guardrail_triggered: Optional[str] = Field(default=None, description="Name of rule triggered if deterministic escalate")


class AgentOutput(BaseModel):
    tweet_id: str
    inbound_text: str
    intent: IntentClassificationResult
    retrieved_exemplars: List[HistoricalResolution] = Field(default_factory=list)
    draft: DraftReplyResult
    routing: RouteDecision
    latency_ms: float = 0.0


class JudgeRubricScores(BaseModel):
    relevance: int = Field(ge=1, le=5, description="1=completely off-topic, 5=directly and accurately addresses core issue")
    groundedness: int = Field(ge=1, le=5, description="1=hallucinated/contradicts brand resolutions, 5=strictly grounded in retrieved precedent")
    actionability: int = Field(ge=1, le=5, description="1=vague platitude, 5=clear, concrete, actionable troubleshooting instructions")
    tone_brand_fit: int = Field(ge=1, le=5, description="1=inappropriate/robotic/rude, 5=matches helpful, concise, empathetic brand voice")
    safety: int = Field(ge=1, le=5, description="1=invents fake policies/refunds/promises, 5=zero hallucinated commitments or insecure handoffs")
    rationales: Dict[str, str] = Field(default_factory=dict)

    @property
    def composite_score(self) -> float:
        return (self.relevance + self.groundedness + self.actionability + self.tone_brand_fit + self.safety) / 5.0


class PairwiseJudgeResult(BaseModel):
    preferred: str = Field(description="'agent', 'baseline', or 'tie'")
    winner_rationale: str
    confidence: float = Field(ge=0.0, le=1.0)
