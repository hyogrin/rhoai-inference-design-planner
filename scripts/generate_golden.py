#!/usr/bin/env python3
"""Generate a golden dataset by re-running design suggestion prompts through a golden model.

Reads   data/traces/design_suggestions.jsonl   (production traces)
Writes  data/traces/golden_suggestions.jsonl   (golden references)

By default uses the OpenAI-compatible endpoint already configured in .env
(OPENAI_BASE_URL / OPENAI_API_KEY) with a different model name.
Optionally pass --backend anthropic to use the Anthropic SDK (requires
ANTHROPIC_API_KEY and `pip install anthropic`).

Usage:
    uv run python scripts/generate_golden.py [--limit N] [--dry-run]
    uv run python scripts/generate_golden.py --backend anthropic --model claude-opus-4-6-20250826
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

TRACES_DIR = PROJECT_ROOT / "data" / "traces"
INPUT_JSONL = TRACES_DIR / "design_suggestions.jsonl"
OUTPUT_JSONL = TRACES_DIR / "golden_suggestions.jsonl"


def load_traces(path: Path, limit: int | None = None) -> list[dict]:
    if not path.exists():
        print(f"ERROR: {path} not found. Run the planner first to generate traces.")
        sys.exit(1)

    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
            if limit and len(records) >= limit:
                break

    print(f"Loaded {len(records)} trace(s) from {path}")
    return records


def load_existing_golden(path: Path) -> set[str]:
    """Return set of trace_ids already present in the golden file."""
    done: set[str] = set()
    if not path.exists():
        return done
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("trace_id"):
                done.add(rec["trace_id"])
    return done


# ---------------------------------------------------------------------------
# Backend: OpenAI-compatible (default)
# ---------------------------------------------------------------------------

def call_openai_compat(
    *,
    base_url: str,
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    verify_ssl: bool,
) -> tuple[str, float]:
    t0 = time.monotonic()
    with httpx.Client(timeout=120.0, verify=verify_ssl) as client:
        resp = client.post(
            f"{base_url}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "max_tokens": 2000,
                "temperature": 0.3,
            },
        )
        resp.raise_for_status()
        data = resp.json()
    latency_ms = (time.monotonic() - t0) * 1000
    text = data["choices"][0]["message"]["content"].strip()
    return text, latency_ms


# ---------------------------------------------------------------------------
# Backend: Anthropic (optional — requires `pip install anthropic`)
# ---------------------------------------------------------------------------

def call_anthropic(
    *,
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
) -> tuple[str, float]:
    try:
        import anthropic
    except ImportError:
        print("ERROR: anthropic package not installed. Run: uv pip install anthropic")
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)
    t0 = time.monotonic()
    message = client.messages.create(
        model=model,
        max_tokens=2000,
        temperature=0.3,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )
    latency_ms = (time.monotonic() - t0) * 1000
    text = message.content[0].text if message.content else ""
    return text, latency_ms


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate golden dataset with a golden model"
    )
    parser.add_argument("--limit", type=int, default=None, help="Max traces to process")
    parser.add_argument("--dry-run", action="store_true", help="Preview without API calls")
    parser.add_argument("--model", type=str, default=None, help="Override golden model name")
    parser.add_argument(
        "--backend",
        choices=["openai", "anthropic"],
        default="openai",
        help="API backend (default: openai-compatible)",
    )
    args = parser.parse_args()

    from backend.config import get_settings

    settings = get_settings()
    golden_model = args.model or settings.golden_model_name

    if args.backend == "anthropic":
        api_key = settings.anthropic_api_key
        if not api_key and not args.dry_run:
            print("ERROR: ANTHROPIC_API_KEY not set. Add it to .env or export it.")
            sys.exit(1)
    else:
        api_key = settings.openai_api_key
        if not api_key and not args.dry_run:
            print("ERROR: OPENAI_API_KEY not set. Add it to .env or export it.")
            sys.exit(1)

    traces = load_traces(INPUT_JSONL, limit=args.limit)
    existing = load_existing_golden(OUTPUT_JSONL)

    pending = [
        t for t in traces
        if t.get("trace_id") not in existing or not t.get("trace_id")
    ]
    if not pending:
        print("All traces already have golden responses. Nothing to do.")
        return

    print(
        f"{len(pending)} trace(s) need golden generation "
        f"(skipping {len(traces) - len(pending)} existing)"
    )
    print(f"Backend: {args.backend} | Model: {golden_model}")

    if args.dry_run:
        for i, trace in enumerate(pending):
            sid = trace.get("session_id", "?")
            repo = trace.get("input", {}).get("model_repo_id", "?")
            plen = len(trace.get("user_prompt", ""))
            print(f"\n--- Trace {i + 1} (session={sid}) ---")
            print(f"  model_repo_id: {repo}")
            print(f"  user_prompt length: {plen} chars")
        print(f"\nDry run complete. Would call {golden_model} for {len(pending)} trace(s).")
        return

    TRACES_DIR.mkdir(parents=True, exist_ok=True)
    success_count = 0

    for i, trace in enumerate(pending):
        session_id = trace.get("session_id", "?")
        model_repo = trace.get("input", {}).get("model_repo_id", "?")
        print(f"\n[{i + 1}/{len(pending)}] session={session_id} model={model_repo}")

        system_prompt = trace.get("system_prompt", "")
        user_prompt = trace.get("user_prompt", "")

        if not user_prompt:
            print("  SKIP — no user_prompt in trace")
            continue

        try:
            if args.backend == "anthropic":
                golden_text, latency_ms = call_anthropic(
                    api_key=api_key,
                    model=golden_model,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                )
            else:
                golden_text, latency_ms = call_openai_compat(
                    base_url=settings.openai_base_url,
                    api_key=api_key,
                    model=golden_model,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    verify_ssl=settings.verify_ssl,
                )
        except Exception as exc:
            print(f"  ERROR — {exc}")
            continue

        golden_record = {
            "trace_id": trace.get("trace_id", ""),
            "session_id": session_id,
            "timestamp": datetime.now(UTC).isoformat(),
            "original_model": trace.get("model_used", ""),
            "golden_model": golden_model,
            "input": trace.get("input", {}),
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "original_output": trace.get("output", ""),
            "golden_output": golden_text,
            "golden_latency_ms": round(latency_ms, 2),
        }

        with open(OUTPUT_JSONL, "a", encoding="utf-8") as f:
            f.write(json.dumps(golden_record, ensure_ascii=False, default=str) + "\n")

        success_count += 1
        print(f"  OK — {len(golden_text)} chars, {latency_ms:.0f}ms")

    print(f"\nDone. {success_count}/{len(pending)} golden response(s) written to {OUTPUT_JSONL}")


if __name__ == "__main__":
    main()
