from __future__ import annotations
import json
import os
import re
from dataclasses import dataclass
from time import perf_counter
from typing import Any
from .prompts import ACTOR_SYSTEM, EVALUATOR_SYSTEM, REFLECTOR_SYSTEM
from .schemas import QAExample, JudgeResult, ReflectionEntry
from .utils import normalize_answer

FIRST_ATTEMPT_WRONG = {"hp2": "London", "hp4": "Atlantic Ocean", "hp6": "Red Sea", "hp8": "Andes"}
FAILURE_MODE_BY_QID = {"hp2": "incomplete_multi_hop", "hp4": "wrong_final_answer", "hp6": "entity_drift", "hp8": "entity_drift"}
_OPENAI_CLIENT = None


@dataclass
class RuntimeOutput:
    text: str
    total_tokens: int
    latency_ms: int


def _get_openai_client() -> Any:
    global _OPENAI_CLIENT
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. Configure it in your environment or .env before running with mode='openai'.")
    if _OPENAI_CLIENT is None:
        from openai import OpenAI

        _OPENAI_CLIENT = OpenAI(api_key=api_key)
    return _OPENAI_CLIENT


def _chat_completion(system_prompt: str, user_prompt: str, model: str) -> RuntimeOutput:
    client = _get_openai_client()
    start = perf_counter()
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    latency_ms = int((perf_counter() - start) * 1000)
    text = (response.choices[0].message.content or "").strip()
    total_tokens = int(getattr(response.usage, "total_tokens", 0) or 0)
    return RuntimeOutput(text=text, total_tokens=total_tokens, latency_ms=latency_ms)


def _context_block(example: QAExample) -> str:
    return "\n\n".join(f"[{idx}] {chunk.title}\n{chunk.text}" for idx, chunk in enumerate(example.context, start=1))


def _extract_json_payload(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE | re.DOTALL).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))

def actor_answer(
    example: QAExample,
    attempt_id: int,
    agent_type: str,
    reflection_memory: list[str],
    mode: str = "mock",
    model: str = "gpt-4o-mini",
) -> tuple[str, int, int]:
    if mode == "openai":
        memory_block = "\n".join(f"- {item}" for item in reflection_memory[-5:]) if reflection_memory else "- none"
        user_prompt = f"""Question:
{example.question}

Context:
{_context_block(example)}

Agent type: {agent_type}
Attempt: {attempt_id}
Reflection memory:
{memory_block}

Return only the final short answer text.
"""
        runtime = _chat_completion(ACTOR_SYSTEM, user_prompt, model=model)
        answer = runtime.text.splitlines()[0].strip() if runtime.text else "Unknown"
        return answer, runtime.total_tokens, runtime.latency_ms

    if mode != "mock":
        raise ValueError(f"Unsupported mode: {mode}")

    if example.qid not in FIRST_ATTEMPT_WRONG:
        return example.gold_answer, 0, 0
    if agent_type == "react":
        return FIRST_ATTEMPT_WRONG[example.qid], 0, 0
    if attempt_id == 1 and not reflection_memory:
        return FIRST_ATTEMPT_WRONG[example.qid], 0, 0
    return example.gold_answer, 0, 0

def evaluator(example: QAExample, answer: str, mode: str = "mock", model: str = "gpt-4o-mini") -> tuple[JudgeResult, int, int]:
    if mode == "openai":
        user_prompt = f"""Question:
{example.question}

Gold answer:
{example.gold_answer}

Predicted answer:
{answer}

Return strict JSON only.
"""
        runtime = _chat_completion(EVALUATOR_SYSTEM, user_prompt, model=model)
        try:
            payload = _extract_json_payload(runtime.text)
            judge = JudgeResult.model_validate(payload)
        except Exception:
            fallback_score = 1 if normalize_answer(example.gold_answer) == normalize_answer(answer) else 0
            judge = JudgeResult(
                score=fallback_score,
                reason="Evaluator output could not be parsed; fallback used normalized exact match.",
                missing_evidence=[] if fallback_score else ["Evaluator JSON parse failed."],
                spurious_claims=[] if fallback_score else [answer],
            )
        return judge, runtime.total_tokens, runtime.latency_ms

    if mode != "mock":
        raise ValueError(f"Unsupported mode: {mode}")

    if normalize_answer(example.gold_answer) == normalize_answer(answer):
        return JudgeResult(score=1, reason="Final answer matches the gold answer after normalization."), 0, 0
    if normalize_answer(answer) == "london":
        return (
            JudgeResult(
                score=0,
                reason="The answer stopped at the birthplace city and never completed the second hop to the river.",
                missing_evidence=["Need to identify the river that flows through London."],
                spurious_claims=[],
            ),
            0,
            0,
        )
    return (
        JudgeResult(
            score=0,
            reason="The final answer selected the wrong second-hop entity.",
            missing_evidence=["Need to ground the answer in the second paragraph."],
            spurious_claims=[answer],
        ),
        0,
        0,
    )

def reflector(
    example: QAExample,
    attempt_id: int,
    judge: JudgeResult,
    answer: str = "",
    mode: str = "mock",
    model: str = "gpt-4o-mini",
) -> tuple[ReflectionEntry, int, int]:
    if mode == "openai":
        user_prompt = f"""Attempt ID: {attempt_id}
Question: {example.question}
Predicted answer: {answer}
Judge score: {judge.score}
Judge reason: {judge.reason}
Missing evidence: {judge.missing_evidence}
Spurious claims: {judge.spurious_claims}

Context:
{_context_block(example)}

Return strict JSON only.
"""
        runtime = _chat_completion(REFLECTOR_SYSTEM, user_prompt, model=model)
        try:
            payload = _extract_json_payload(runtime.text)
            payload["attempt_id"] = attempt_id
            reflection = ReflectionEntry.model_validate(payload)
        except Exception:
            reflection = ReflectionEntry(
                attempt_id=attempt_id,
                failure_reason=judge.reason,
                lesson="Fix the missing second-hop reasoning before answering.",
                next_strategy="Extract candidate entities from context and verify the final entity with explicit evidence.",
            )
        return reflection, runtime.total_tokens, runtime.latency_ms

    if mode != "mock":
        raise ValueError(f"Unsupported mode: {mode}")

    strategy = "Do the second hop explicitly: birthplace city -> river through that city." if example.qid == "hp2" else "Verify the final entity against the second paragraph before answering."
    return (
        ReflectionEntry(
            attempt_id=attempt_id,
            failure_reason=judge.reason,
            lesson="A partial first-hop answer is not enough; the final answer must complete all hops.",
            next_strategy=strategy,
        ),
        0,
        0,
    )
