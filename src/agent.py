import os
import json
import time
import hashlib
import logging
from typing import Optional, Dict, Any, List

from src.schemas import (
    IntentType, RouteAction, IntentClassificationResult,
    HistoricalResolution, DraftReplyResult, RouteDecision, AgentOutput
)
from src.clean import clean_tweet_text, extract_key_entities
from src.taxonomy import get_taxonomy_prompt_text
from src.retrieve import RetrievalIndex

logger = logging.getLogger(__name__)

# Hard escalation triggers (deterministic security and risk filters)
HARD_ESCALATE_KEYWORDS = [
    "refund", "unauthorized charge", "fraud", "stolen", "stole", "lawyer",
    "attorney", "lawsuit", "sue", "legal action", "compromised", "hacked",
    "police", "dispute charge", "death", "emergency"
]


class LLMCache:
    """Disk-backed JSON cache keyed by MD5 of prompt + model to enable <15 min instant repro."""

    def __init__(self, cache_file: str = "data/cache/llm_cache.json"):
        self.cache_file = cache_file
        self.cache: Dict[str, Any] = {}
        self.load()

    def _hash_key(self, prompt: str, model: str) -> str:
        return hashlib.md5(f"{model}:{prompt}".encode("utf-8")).hexdigest()

    def get(self, prompt: str, model: str) -> Optional[Dict[str, Any]]:
        key = self._hash_key(prompt, model)
        return self.cache.get(key)

    def set(self, prompt: str, model: str, response: Dict[str, Any]) -> None:
        key = self._hash_key(prompt, model)
        self.cache[key] = response

    def load(self) -> None:
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    self.cache = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
                self.cache = {}

    def save(self) -> None:
        os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
        with open(self.cache_file, "w", encoding="utf-8") as f:
            json.dump(self.cache, f, indent=2)


class SupportAgent:
    """
    Production-grade AI Support Agent for AppleSupport.
    Executes: Clean -> Classify Intent -> Retrieve Precedents -> Draft Grounded Reply -> Route (Auto/Escalate).
    """

    def __init__(
        self,
        retrieval_index: RetrievalIndex,
        cache_path: str = "data/cache/llm_cache.json",
        model_name: str = "claude-3-5-haiku-20241022",
        auto_confidence_threshold: float = 0.65
    ):
        self.retrieval_index = retrieval_index
        self.cache = LLMCache(cache_path)
        self.model_name = model_name
        self.auto_confidence_threshold = auto_confidence_threshold
        self._init_llm_client()

    def _init_llm_client(self) -> None:
        """Initialize Anthropic or OpenAI client if keys are present."""
        self.client_type = None
        self.client = None

        if os.getenv("ANTHROPIC_API_KEY"):
            try:
                import anthropic
                self.client = anthropic.Anthropic()
                self.client_type = "anthropic"
                logger.info("Initialized Anthropic client for SupportAgent.")
            except ImportError:
                pass

        if not self.client and os.getenv("OPENAI_API_KEY"):
            try:
                import openai
                self.client = openai.OpenAI()
                self.client_type = "openai"
                logger.info("Initialized OpenAI client for SupportAgent.")
            except ImportError:
                pass

        if not self.client:
            logger.info("No active LLM API key detected; agent will utilize cache or local deterministic heuristics.")

    def _call_llm_json(self, prompt: str, system_prompt: str) -> Dict[str, Any]:
        """Execute LLM call with disk cache fallback and strict JSON parsing."""
        cached = self.cache.get(prompt, self.model_name)
        if cached is not None:
            return cached

        if self.client_type == "anthropic":
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=1000,
                temperature=0.2,
                system=system_prompt + "\nReturn ONLY valid raw JSON without markdown or commentary.",
                messages=[{"role": "user", "content": prompt}]
            )
            raw_text = response.content[0].text
        elif self.client_type == "openai":
            response = self.client.chat.completions.create(
                model=self.model_name,
                temperature=0.2,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt + "\nReturn ONLY valid JSON."},
                    {"role": "user", "content": prompt}
                ]
            )
            raw_text = response.choices[0].message.content
        else:
            # Fallback deterministic response for offline execution
            raw_text = self._offline_heuristic_response(prompt)

        # Parse JSON
        try:
            # Strip markdown formatting if any
            clean_json = raw_text.strip()
            if clean_json.startswith("```json"):
                clean_json = clean_json[7:]
            if clean_json.endswith("```"):
                clean_json = clean_json[:-3]
            parsed = json.loads(clean_json.strip())
        except Exception as e:
            logger.warning(f"JSON parsing error: {e}, falling back to heuristic parsing.")
            parsed = {"error": str(e), "raw": raw_text}

        self.cache.set(prompt, self.model_name, parsed)
        return parsed

    def _offline_heuristic_response(self, prompt: str) -> str:
        """Deterministic offline fallback if no API key is provided and cache misses."""
        p_lower = prompt.lower()
        if "classify the intent" in p_lower:
            if "battery" in p_lower or "drain" in p_lower or "charge" in p_lower:
                intent = "battery_power_charging"
            elif "ios" in p_lower or "update" in p_lower or "bug" in p_lower:
                intent = "software_update_bug"
            elif "wifi" in p_lower or "bluetooth" in p_lower:
                intent = "connectivity_wifi_bluetooth"
            elif "screen" in p_lower or "crack" in p_lower or "damage" in p_lower:
                intent = "device_hardware_damage"
            elif "id" in p_lower or "icloud" in p_lower or "password" in p_lower:
                intent = "apple_id_icloud_security"
            elif "airpod" in p_lower or "sound" in p_lower or "mic" in p_lower:
                intent = "audio_sound_accessories"
            elif "bill" in p_lower or "charge" in p_lower or "refund" in p_lower:
                intent = "store_billing_purchase"
            else:
                intent = "complaint_feedback_other"
            return json.dumps({
                "intent": intent,
                "confidence": 0.85,
                "reasoning": f"Query contains diagnostic keywords for {intent}"
            })
        elif "draft a customer support response" in p_lower:
            return json.dumps({
                "reply": "We'd be glad to look into this with you. Start by restarting your device and ensuring your iOS is up to date. Let us know if the issue persists!",
                "grounded_in_ids": ["offline_ref_01"],
                "actionable_steps": ["Restart device", "Check for software update"],
                "confidence": 0.80,
                "hallucination_risk": False
            })
        elif "decide whether to auto-handle" in p_lower:
            return json.dumps({
                "action": "AUTO",
                "stated_reason": "Standard troubleshooting procedure provided from verified historical precedent.",
                "confidence": 0.85
            })
        return json.dumps({"status": "ok"})

    def classify_intent(self, text: str) -> IntentClassificationResult:
        """Stage 1: Classify incoming customer message into 8 defined intents."""
        system_prompt = (
            "You are an expert customer support intent classifier for AppleSupport.\n"
            "Classify the incoming customer query into exactly one of the defined 8 intents.\n"
            "Return JSON matching schema: {'intent': str, 'confidence': float, 'reasoning': str}.\n"
            f"Taxonomy:\n{get_taxonomy_prompt_text()}"
        )
        prompt = f"Customer Tweet: \"{text}\"\nClassify the intent:"
        res = self._call_llm_json(prompt, system_prompt)

        try:
            intent_val = IntentType(res.get("intent", "complaint_feedback_other"))
        except ValueError:
            intent_val = IntentType.COMPLAINT_FEEDBACK_OTHER

        return IntentClassificationResult(
            intent=intent_val,
            confidence=float(res.get("confidence", 0.70)),
            reasoning=str(res.get("reasoning", "Classified from query text"))
        )

    def draft_reply(self, query: str, intent: IntentType, exemplars: List[HistoricalResolution]) -> DraftReplyResult:
        """Stage 3: Draft response strictly grounded in retrieved historical resolutions."""
        exemplars_str = "\n".join([
            f"Precedent #{i+1} (Tweet ID {ex.tweet_id}):\nCustomer: {ex.customer_query}\nApple Support: {ex.brand_reply}"
            for i, ex in enumerate(exemplars)
        ])

        system_prompt = (
            "You are an official customer support agent for @AppleSupport on Twitter.\n"
            "Draft a helpful, polite, concise reply (max 2 tweets / 280 characters).\n"
            "CRITICAL RULES:\n"
            "1. You must GROUND your reply in the provided historical resolutions. Do not invent order numbers, fake URLs, or non-existent policies.\n"
            "2. If the issue requires personal data (serial number, Apple ID credentials), instruct them to DM securely.\n"
            "3. Return JSON: {'reply': str, 'grounded_in_ids': list[str], 'actionable_steps': list[str], 'confidence': float, 'hallucination_risk': bool}"
        )
        prompt = (
            f"Incoming Customer Query: \"{query}\"\n"
            f"Classified Intent: {intent.value}\n\n"
            f"Retrieved Historical Apple Precedents:\n{exemplars_str}\n\n"
            "Draft a customer support response:"
        )

        res = self._call_llm_json(prompt, system_prompt)
        grounded_ids = [ex.tweet_id for ex in exemplars[:2]] if exemplars else []

        return DraftReplyResult(
            reply=str(res.get("reply", "Thanks for reaching out! We would be glad to help. Please let us know which iOS version you have installed.")),
            grounded_in_ids=res.get("grounded_in_ids", grounded_ids),
            actionable_steps=res.get("actionable_steps", ["Standard troubleshooting"]),
            confidence=float(res.get("confidence", 0.85)),
            hallucination_risk=bool(res.get("hallucination_risk", False))
        )

    def route_decision(
        self,
        query: str,
        intent_res: IntentClassificationResult,
        draft: DraftReplyResult,
        exemplars: List[HistoricalResolution]
    ) -> RouteDecision:
        """
        Stage 4: Hybrid deterministic + LLM routing.
        Hard guardrails run first; LLM assesses autonomous resolution safety second.
        """
        q_lower = query.lower()

        # Hard Guardrail 1: High-risk keywords
        for kw in HARD_ESCALATE_KEYWORDS:
            if kw in q_lower:
                return RouteDecision(
                    action=RouteAction.ESCALATE,
                    stated_reason=f"Hard guardrail triggered: query contains high-risk keyword '{kw}' requiring human oversight.",
                    confidence=1.0,
                    hard_guardrail_triggered=f"Keyword_{kw}"
                )

        # Hard Guardrail 2: Pure complaint / unactionable feedback
        if intent_res.intent == IntentType.COMPLAINT_FEEDBACK_OTHER:
            return RouteDecision(
                action=RouteAction.ESCALATE,
                stated_reason="Hard guardrail triggered: message is general complaint/feedback without actionable technical issue.",
                confidence=0.95,
                hard_guardrail_triggered="NonActionableComplaint"
            )

        # Hard Guardrail 3: Low classifier confidence
        if intent_res.confidence < 0.60:
            return RouteDecision(
                action=RouteAction.ESCALATE,
                stated_reason="Hard guardrail triggered: intent classification confidence is below safety threshold (<0.60).",
                confidence=round(1.0 - intent_res.confidence, 3),
                hard_guardrail_triggered="LowIntentConfidence"
            )

        # Hard Guardrail 4: Weak retrieval precedent
        max_sim = max([ex.similarity for ex in exemplars], default=0.0)
        if max_sim < 0.30:
            return RouteDecision(
                action=RouteAction.ESCALATE,
                stated_reason="Hard guardrail triggered: no sufficiently similar historical resolution found (similarity < 0.30).",
                confidence=0.90,
                hard_guardrail_triggered="NoHistoricalPrecedent"
            )

        # Hard Guardrail 5: Physical damage requires Apple Store inspection
        if intent_res.intent == IntentType.DEVICE_HARDWARE_DAMAGE:
            return RouteDecision(
                action=RouteAction.ESCALATE,
                stated_reason="Physical hardware damage requires in-person Genius Bar diagnosis or mail-in repair appointment.",
                confidence=0.95,
                hard_guardrail_triggered="HardwareDamagePolicy"
            )

        # Stage 4b: LLM routing evaluation for safe autonomous response
        system_prompt = (
            "You are a Senior Quality & Safety Auditor for automated customer support.\n"
            "Decide whether to AUTO-HANDLE (send automated reply) or ESCALATE to a human agent.\n"
            "CRITICAL PRINCIPLE: False-Auto (sending a wrong auto-reply) is 10x more costly than False-Escalate.\n"
            "Escalate if the issue requires account access, private customer verification, or complex edge cases.\n"
            "Return JSON: {'action': 'AUTO' | 'ESCALATE', 'stated_reason': str, 'confidence': float}"
        )
        prompt = (
            f"Customer Tweet: \"{query}\"\n"
            f"Intent: {intent_res.intent.value} (conf: {intent_res.confidence})\n"
            f"Drafted Reply: \"{draft.reply}\"\n"
            f"Top Precedent Match Similarity: {max_sim}\n\n"
            "Decide whether to auto-handle or escalate:"
        )

        res = self._call_llm_json(prompt, system_prompt)
        action_str = res.get("action", "AUTO").upper()
        action = RouteAction.AUTO if action_str == "AUTO" else RouteAction.ESCALATE
        confidence = float(res.get("confidence", 0.80))

        # Enforce threshold
        if action == RouteAction.AUTO and confidence < self.auto_confidence_threshold:
            action = RouteAction.ESCALATE
            reason = f"Autonomous routing confidence ({confidence:.2f}) below required safety threshold ({self.auto_confidence_threshold:.2f})."
        else:
            reason = str(res.get("stated_reason", "Drafted reply provides safe standard self-service troubleshooting."))

        return RouteDecision(
            action=action,
            stated_reason=reason,
            confidence=round(confidence, 3),
            hard_guardrail_triggered=None
        )

    def process(self, tweet_id: str, raw_text: str) -> AgentOutput:
        """Full end-to-end agent processing pipeline."""
        start_t = time.time()

        # 1. Clean
        cleaned = clean_tweet_text(raw_text)

        # 2. Intent
        intent_res = self.classify_intent(cleaned)

        # 3. Retrieve
        exemplars = self.retrieval_index.search(cleaned, k=5)

        # 4. Draft
        draft_res = self.draft_reply(cleaned, intent_res.intent, exemplars)

        # 5. Route
        route_res = self.route_decision(cleaned, intent_res, draft_res, exemplars)

        latency_ms = round((time.time() - start_t) * 1000, 1)

        # Flush cache changes periodically
        self.cache.save()

        return AgentOutput(
            tweet_id=tweet_id,
            inbound_text=cleaned,
            intent=intent_res,
            retrieved_exemplars=exemplars,
            draft=draft_res,
            routing=route_res,
            latency_ms=latency_ms
        )
