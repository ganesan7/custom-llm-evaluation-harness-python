import re
import json
import asyncio
import time
from datetime import datetime, timezone
from pathlib import Path

import litellm
from pydantic import BaseModel, ValidationError


BASE_DIR = Path(__file__).resolve().parent
DATASET = BASE_DIR / "test_cases.json"
RAW_LOG = BASE_DIR / "raw_outputs.jsonl"
RESULTS_LOG = BASE_DIR / "results.jsonl"

MODEL = "ollama/llama3.2:1b"
API_BASE = "http://localhost:11434"
MAX_CONCURRENT = 2


class JudgeScore(BaseModel):
    passed: bool
    reason: str


def append_jsonl(filepath: Path, record: dict) -> None:
    """Append one complete JSON record to a log."""
    with filepath.open("a", encoding="utf-8") as file:
        file.write(
            json.dumps(record, ensure_ascii=False) + "\n"
        )
        file.flush()


async def run_case(
    case: dict, semaphore: asyncio.Semaphore
) -> dict:
    """Run inference and save the raw response before judging."""
    start = time.perf_counter()

    async with semaphore:
        try:
            response = await litellm.acompletion(
                model=MODEL,
                api_base=API_BASE,
                messages=[
                    {"role": "user", "content": case["input"]}
                ],
                temperature=0,
                timeout=120,
                num_retries=2,
            )

            raw = response.choices[0].message.content or ""

            result = {
                **case,
                "raw_output": raw,
                "latency_seconds": round(
                    time.perf_counter() - start, 3
                ),
                "timestamp": datetime.now(
                    timezone.utc
                ).isoformat(),
            }

            # Save immediately, so inference can be re-scored later.
            append_jsonl(RAW_LOG, result)
            return result

        except Exception as exc:
            result = {
                **case,
                "raw_output": "",
                "error": str(exc),
                "latency_seconds": round(
                    time.perf_counter() - start, 3
                ),
                "timestamp": datetime.now(
                    timezone.utc
                ).isoformat(),
            }
            append_jsonl(RAW_LOG, result)
            return result


def deterministic_judge(result: dict) -> dict:
    """Basic case-insensitive keyword matching."""
    output = result.get("raw_output", "").casefold()
    concept = result["expected_concept"].casefold()

    passed = bool(concept) and concept in output

    return {
        **result,
        "passed": passed,
        "reason": (
            "Expected phrase found."
            if passed
            else "Expected phrase not found."
        ),
        "judge": "deterministic",
    }


async def llm_judge(result: dict) -> dict:
    """Ask the model to evaluate meaning against the criterion."""
    prompt = f"""
Evaluate the response against the criterion.

User question:
{result['input']}

Model response:
{result['raw_output']}

Required criterion:
{result['expected_concept']}

Return ONLY valid JSON with:
- passed: true or false
- reason: a brief explanation

Pass only if the response actually satisfies the criterion.
Do not pass merely because it repeats words from the criterion.
"""

    try:
        response = await litellm.acompletion(
            model=MODEL,
            api_base=API_BASE,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            timeout=120,
            num_retries=2,
        )

        content = response.choices[0].message.content or ""
        score = JudgeScore.model_validate_json(content.strip())

        return {
            **result,
            "passed": score.passed,
            "reason": score.reason,
            "judge": "llm",
        }

    except (ValidationError, Exception) as exc:
        return {
            **result,
            "passed": False,
            "reason": f"Judge error: {exc}",
            "judge": "llm",
            "judge_error": True,
        }


async def evaluate_case(
    result: dict, judge_semaphore: asyncio.Semaphore
) -> dict:
    if result.get("error"):
        scored = {
            **result,
            "passed": False,
            "reason": result["error"],
            "judge": "not_run",
        }
    elif result.get("judge_type") == "deterministic":
        scored = deterministic_judge(result)
    else:
        async with judge_semaphore:
            scored = await llm_judge(result)

    append_jsonl(RESULTS_LOG, scored)

    status = "PASS" if scored["passed"] else "FAIL"
    print(f"{status} | {scored['id']} | {scored['reason']}")
    return scored


async def main():
    cases = json.loads(DATASET.read_text(encoding="utf-8"))

    if not cases:
        print("Dataset is empty.")
        return

    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    judge_semaphore = asyncio.Semaphore(1)

    # Inference results are saved before scoring begins.
    raw_results = await asyncio.gather(
        *(run_case(case, semaphore) for case in cases)
    )

    scored_results = await asyncio.gather(
        *(
            evaluate_case(result, judge_semaphore)
            for result in raw_results
        )
    )

    passed = sum(r["passed"] for r in scored_results)
    total = len(scored_results)
    avg_latency = sum(
        r["latency_seconds"] for r in scored_results
    ) / total

    summary = {
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "pass_rate_percent": round(100 * passed / total, 2),
        "average_inference_seconds": round(avg_latency, 3),
    }

    print("\nEvaluation Summary")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
