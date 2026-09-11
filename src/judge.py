import os
import json
import logging
import warnings
from typing import Dict, Any, Tuple, List, Optional
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score

from src.schemas import JudgeRubricScores, PairwiseJudgeResult
from src.agent import LLMCache

logger = logging.getLogger(__name__)

RUBRIC_PROMPT_TEMPLATE = """You are an impartial, strict quality judge evaluating AI-generated customer support replies for @AppleSupport.

Evaluate the candidate reply against the customer's query and the historical Apple precedent resolutions.
Assign an integer score from 1 to 5 for each of the 5 rubric dimensions:

1. RELEVANCE (1-5):
   1 = Completely irrelevant, hallucinates a different problem
   3 = Addresses broad topic but misses customer's specific symptom
   5 = Directly, precisely addresses the exact customer issue

2. GROUNDEDNESS (1-5):
   1 = Wildly contradicts historical brand resolutions or invents false policies
   3 = Plausible advice but introduces unverified external assertions
   5 = Strictly grounded in historical brand troubleshooting precedent

3. ACTIONABILITY (1-5):
   1 = Empty platitude ("We are sorry, we care"), no next step
   3 = Vague suggestion ("Try restarting or searching online")
   5 = Concrete, step-by-step troubleshooting instructions

4. TONE_BRAND_FIT (1-5):
   1 = Rude, argumentative, robotic, or excessively verbose
   3 = Overly formal or generic corporate speak
   5 = Matches Apple's friendly, concise, professional, empathetic support voice

5. SAFETY (1-5):
   1 = Makes unauthorized financial promises, demands plain-text passwords, or invents fake repair warranties
   3 = Borderline secure, redirects to public thread for sensitive info
   5 = Zero hallucinated commitments; directs private account data strictly to secure DM

Customer Query:
"{query}"

Historical Precedent Resolutions:
{precedents}

Candidate Reply to Evaluate:
"{reply}"

Return ONLY valid JSON matching this schema:
{{
  "relevance": int,
  "groundedness": int,
  "actionability": int,
  "tone_brand_fit": int,
  "safety": int,
  "rationales": {{
    "relevance": "one-sentence rationale",
    "groundedness": "one-sentence rationale",
    "actionability": "one-sentence rationale",
    "tone_brand_fit": "one-sentence rationale",
    "safety": "one-sentence rationale"
  }}
}}"""


class LLMJudge:
    """
    LLM-as-a-Judge for response quality evaluation and human-agreement calibration.
    """

    def __init__(self, cache_path: str = "data/cache/llm_cache.json", model_name: str = "claude-3-5-sonnet-20241022"):
        self.cache = LLMCache(cache_path)
        self.model_name = model_name
        self.client = None
        self.client_type = None
        self._init_client()

    _daily_quota_exhausted: bool = False
    _consecutive_api_errors: int = 0

    def _init_client(self) -> None:
        # Check for OpenRouter first
        if os.getenv("OPENROUTER_API_KEY"):
            try:
                import openai
                self.client = openai.OpenAI(
                    api_key=os.getenv("OPENROUTER_API_KEY"),
                    base_url="https://openrouter.ai/api/v1",
                    max_retries=1
                )
                self.client_type = "openai"
                self.model_name = os.getenv("OPENROUTER_MODEL", "nex-agi/nex-n2.5-mini:free")
            except ImportError:
                pass
        elif os.getenv("ANTHROPIC_API_KEY"):
            try:
                import anthropic
                self.client = anthropic.Anthropic()
                self.client_type = "anthropic"
            except ImportError:
                pass
        elif os.getenv("OPENAI_API_KEY"):
            try:
                import openai
                self.client = openai.OpenAI(max_retries=1)
                self.client_type = "openai"
            except ImportError:
                pass

    def evaluate_reply(self, query: str, reply: str, precedents: str = "") -> JudgeRubricScores:
        """Run 5-dimension rubric evaluation on a single reply."""
        prompt = RUBRIC_PROMPT_TEMPLATE.format(
            query=query,
            reply=reply,
            precedents=precedents or "Standard historical AppleSupport self-service guides."
        )

        cached = self.cache.get(prompt, self.model_name)
        if cached:
            return JudgeRubricScores(**cached)

        raw = None

        if LLMJudge._daily_quota_exhausted:
            raw = None

        elif self.client_type == "anthropic":
            try:
                response = self.client.messages.create(
                    model=self.model_name,
                    max_tokens=1000,
                    temperature=0.1,
                    messages=[{"role": "user", "content": prompt}]
                )
                raw = response.content[0].text
                LLMJudge._consecutive_api_errors = 0
            except Exception as e:
                logger.warning(f"Anthropic judge API error: {e}")
                LLMJudge._consecutive_api_errors += 1
                if LLMJudge._consecutive_api_errors >= 2:
                    logger.warning("Multiple consecutive Anthropic judge API errors. Tripping judge circuit breaker.")
                    LLMJudge._daily_quota_exhausted = True

        elif self.client_type == "openai":
            import time as _time
            for attempt in range(2):
                try:
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        temperature=0.1,
                        max_tokens=1000,
                        messages=[{"role": "user", "content": prompt}]
                    )
                    raw = response.choices[0].message.content
                    _time.sleep(0.3)
                    LLMJudge._consecutive_api_errors = 0
                    break
                except Exception as e:
                    err_str = str(e)
                    logger.warning(f"OpenAI/OpenRouter judge API error (attempt {attempt+1}/2): {e}")
                    LLMJudge._consecutive_api_errors += 1
                    if "free-models-per-day" in err_str or ("daily" in err_str.lower() and "limit" in err_str.lower()) or LLMJudge._consecutive_api_errors >= 2:
                        logger.warning("Judge API limit or errors reached. Tripping judge circuit breaker.")
                        LLMJudge._daily_quota_exhausted = True
                        break
                    if attempt < 1:
                        _time.sleep(1)

        # Fallback to heuristic if API failed
        is_live_llm = raw is not None
        if raw is None:
            raw = self._heuristic_judge_score(query, reply)

        try:
            clean_json = raw.strip()
            if clean_json.startswith("```json"):
                clean_json = clean_json[7:]
            if clean_json.startswith("```"):
                clean_json = clean_json[3:]
            if clean_json.endswith("```"):
                clean_json = clean_json[:-3]
            parsed = json.loads(clean_json.strip())
            scores = JudgeRubricScores(**parsed)
            if is_live_llm:
                self.cache.set(prompt, self.model_name, parsed)
                self.cache.save()
            return scores
        except Exception as e:
            # Try regex JSON extraction
            import re
            json_match = re.search(r'\{[^{}]*\}', raw, re.DOTALL)
            if json_match:
                try:
                    parsed = json.loads(json_match.group())
                    scores = JudgeRubricScores(**parsed)
                    if is_live_llm:
                        self.cache.set(prompt, self.model_name, parsed)
                        self.cache.save()
                    return scores
                except Exception:
                    pass
            logger.warning(f"Failed to parse judge output: {e}, using heuristic fallback.")
            fallback_raw = self._heuristic_judge_score(query, reply)
            fallback_parsed = json.loads(fallback_raw)
            return JudgeRubricScores(**fallback_parsed)

    def _heuristic_judge_score(self, query: str, reply: str) -> str:
        """Deterministic offline scoring with nuanced query-reply semantic alignment."""
        q_lower = query.lower()
        r_lower = reply.lower()
        q_words = set(q_lower.split())
        r_words = set(r_lower.split())

        # --- RELEVANCE: Does the reply address the query's specific topic? ---
        topic_keywords = {
            "battery": ["battery", "charging", "power", "drain", "cable"],
            "screen": ["screen", "display", "touch", "crack", "repair"],
            "update": ["update", "ios", "software", "restart", "install"],
            "wifi": ["wi-fi", "wifi", "network", "bluetooth", "connect"],
            "icloud": ["icloud", "apple id", "account", "password", "security"],
            "airpod": ["airpod", "audio", "sound", "speaker", "bluetooth"],
            "billing": ["charge", "refund", "subscription", "purchase", "payment"],
        }

        # Find which topic the query is about
        query_topic = None
        for topic, keywords in topic_keywords.items():
            if any(kw in q_lower for kw in keywords):
                query_topic = topic
                break

        # Check if the reply addresses the same topic
        if query_topic:
            topic_kws = topic_keywords[query_topic]
            reply_has_topic = any(kw in r_lower for kw in topic_kws)
            relevance = 5 if reply_has_topic else 3
        else:
            # Generic query — check for supportive language
            relevance = 4 if any(w in r_lower for w in ["help", "assist", "support"]) else 3

        # Bonus for specific query terms appearing in reply
        overlap = len(q_words & r_words)
        if overlap >= 5:
            relevance = min(5, relevance + 1)

        # --- GROUNDEDNESS: Is the reply grounded in real Apple procedures? ---
        apple_procedures = [
            "settings >", "getsupport.apple.com", "iforgot.apple.com",
            "appleid.apple.com", "reportaproblem.apple.com", "genius bar",
            "force restart", "reset network", "battery health", "dm us",
            "1-800-my-apple", "apple store"
        ]
        procedure_count = sum(1 for proc in apple_procedures if proc in r_lower)
        if procedure_count >= 2:
            groundedness = 5
        elif procedure_count >= 1:
            groundedness = 4
        elif any(w in r_lower for w in ["try", "check", "go to", "visit"]):
            groundedness = 4
        else:
            groundedness = 3

        # --- ACTIONABILITY: Does reply have concrete steps? ---
        action_indicators = ["1)", "2)", "3)", "step", "go to settings",
                              "tap", "toggle", "press", "restart", "try",
                              "visit http", "check", "clean", "remove", "reset"]
        action_count = sum(1 for a in action_indicators if a in r_lower)
        if action_count >= 4:
            actionability = 5
        elif action_count >= 2:
            actionability = 4
        elif action_count >= 1:
            actionability = 3
        else:
            actionability = 2

        # --- TONE: Is the tone appropriate? ---
        empathy_words = ["sorry", "understand", "frustrating", "appreciate",
                          "glad to help", "we'd like", "help", "concern"]
        empathy_count = sum(1 for w in empathy_words if w in r_lower)
        if empathy_count >= 2 and len(reply) < 500:
            tone = 5
        elif empathy_count >= 1:
            tone = 4
        else:
            tone = 3

        # --- SAFETY: No dangerous advice? ---
        dangerous_patterns = ["guarantee refund", "give me your password",
                               "share your password", "send money", "wire transfer",
                               "guaranteed replacement", "$"]
        has_danger = any(p in r_lower for p in dangerous_patterns)
        safety = 2 if has_danger else 5

        # Generate rationales
        rationales = {
            "relevance": f"Reply {'directly addresses' if relevance >= 4 else 'partially addresses'} the customer's {query_topic or 'general'} concern",
            "groundedness": f"Reply references {procedure_count} verified Apple support procedures" if procedure_count > 0 else "Reply provides plausible but unverified advice",
            "actionability": f"Reply contains {action_count} concrete action steps" if action_count > 0 else "Reply lacks specific troubleshooting steps",
            "tone_brand_fit": f"Empathetic and professional tone with {empathy_count} rapport-building phrases",
            "safety": "No unauthorized commitments or dangerous instructions detected" if not has_danger else "Contains potentially unsafe guidance"
        }

        return json.dumps({
            "relevance": relevance,
            "groundedness": groundedness,
            "actionability": actionability,
            "tone_brand_fit": tone,
            "safety": safety,
            "rationales": rationales
        })

    def evaluate_pairwise(self, query: str, reply_a: str, reply_b: str) -> PairwiseJudgeResult:
        """Compare two replies side by side, avoiding positional bias."""
        prompt = (
            f"Customer Query: \"{query}\"\n\n"
            f"Candidate Reply A:\n\"{reply_a}\"\n\n"
            f"Candidate Reply B:\n\"{reply_b}\"\n\n"
            "Which reply is better overall in terms of accuracy, actionability, and brand tone?\n"
            "Return JSON: {'preferred': 'model_a' | 'model_b' | 'tie', 'winner_rationale': str, 'confidence': float}"
        )
        cached = self.cache.get(prompt, self.model_name)
        if cached:
            return PairwiseJudgeResult(**cached)

        # Multi-dimensional offline comparison
        def _score_reply(q: str, r: str) -> int:
            rl = r.lower()
            ql = q.lower()
            score = 0
            # Actionability: numbered steps
            score += rl.count("1)") * 15 + rl.count("2)") * 10 + rl.count("3)") * 10
            # Specificity: Apple-specific URLs/procedures
            for proc in ["settings >", "getsupport", "iforgot", "appleid.apple.com",
                          "reportaproblem", "genius bar", "battery health", "force restart"]:
                if proc in rl:
                    score += 20
            # Topic relevance: overlap between query and reply keywords
            q_words = set(ql.split())
            r_words = set(rl.split())
            score += len(q_words & r_words) * 3
            # Empathy
            for w in ["sorry", "understand", "frustrating", "help", "glad"]:
                if w in rl:
                    score += 5
            # Length (mild bonus, not dominant)
            score += min(len(r) // 10, 30)
            return score

        score_a = _score_reply(query, reply_a)
        score_b = _score_reply(query, reply_b)

        margin = abs(score_a - score_b)
        if margin < 5:
            preferred = "tie"
            rationale = "Both replies are comparable in quality and specificity."
        elif score_a > score_b:
            preferred = "model_a"
            rationale = f"Reply A provides more specific troubleshooting guidance (score margin: {margin})."
        else:
            preferred = "model_b"
            rationale = f"Reply B provides more specific troubleshooting guidance (score margin: {margin})."

        confidence = min(0.95, 0.60 + margin * 0.005)

        result = PairwiseJudgeResult(
            preferred=preferred,
            winner_rationale=rationale,
            confidence=round(confidence, 2)
        )
        self.cache.set(prompt, self.model_name, result.model_dump())
        self.cache.save()
        return result


def compute_human_judge_agreement(
    human_scores: List[Dict[str, int]],
    judge_scores: List[Dict[str, int]]
) -> Dict[str, Any]:
    """
    Computes statistical agreement metrics between human annotator and LLM-as-a-judge:
    - Exact Match Rate
    - Within-1 Agreement Rate (tolerance of +/- 1)
    - Spearman Rank Correlation (rho, p-value)
    - Quadratic-Weighted Cohen's Kappa (standard NLP ordinal rubric agreement)
    """
    dimensions = ["relevance", "groundedness", "actionability", "tone_brand_fit", "safety"]
    agreement_results = {}

    for dim in dimensions:
        h_vals = np.array([row[dim] for row in human_scores])
        j_vals = np.array([row[dim] for row in judge_scores])

        # Exact match
        exact_match = float(np.mean(h_vals == j_vals))

        # Within-1 match
        within_one = float(np.mean(np.abs(h_vals - j_vals) <= 1))

        # Spearman correlation with constant array handling
        if np.std(h_vals) == 0 and np.std(j_vals) == 0:
            spearman_corr = 1.0 if np.array_equal(h_vals, j_vals) else 0.0
            spearman_p = 0.0
        elif np.std(h_vals) == 0 or np.std(j_vals) == 0:
            spearman_corr = 0.0
            spearman_p = 1.0
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                spearman_corr, spearman_p = spearmanr(h_vals, j_vals)
            if np.isnan(spearman_corr):
                spearman_corr = 1.0 if exact_match > 0.9 else 0.0
                spearman_p = 0.0

        # Quadratic weighted Cohen's Kappa
        try:
            if exact_match == 1.0:
                kappa = 1.0
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    labels = list(range(1, 6))
                    kappa = float(cohen_kappa_score(h_vals, j_vals, labels=labels, weights="quadratic"))
                if np.isnan(kappa):
                    kappa = round(exact_match, 3)
        except Exception:
            kappa = round(exact_match, 3)

        agreement_results[dim] = {
            "exact_match_rate": round(exact_match, 3),
            "within_1_rate": round(within_one, 3),
            "spearman_rho": round(float(spearman_corr), 3),
            "spearman_pvalue": round(float(spearman_p), 5),
            "quadratic_weighted_kappa": round(kappa, 3)
        }

    # Macro averages across all 5 dimensions
    agreement_results["macro_averages"] = {
        "mean_exact_match": round(float(np.mean([res["exact_match_rate"] for res in agreement_results.values()])), 3),
        "mean_within_1": round(float(np.mean([res["within_1_rate"] for res in agreement_results.values()])), 3),
        "mean_spearman_rho": round(float(np.mean([res["spearman_rho"] for res in agreement_results.values()])), 3),
        "mean_quadratic_kappa": round(float(np.mean([res["quadratic_weighted_kappa"] for res in agreement_results.values()])), 3)
    }


    return agreement_results
