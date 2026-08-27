#!/usr/bin/env python3
"""Run evaluation comparing production design suggestions against golden references.

Scoring: 100-point scale measuring similarity to golden output.

Reads:
    data/traces/golden_suggestions.jsonl   (must exist — run generate_golden.py first)

Usage:
    uv run python scripts/run_eval.py --no-mlflow
    uv run python scripts/run_eval.py [--experiment NAME]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

TRACES_DIR = PROJECT_ROOT / "data" / "traces"
GOLDEN_JSONL = TRACES_DIR / "golden_suggestions.jsonl"

REQUIRED_SECTIONS = [
    "Architecture Direction",
    "Key Considerations",
    "Risk Factors",
    "Alternative Approaches",
]

WEIGHT_STRUCTURE = 10
WEIGHT_KEYWORD = 20
WEIGHT_CONTENT = 25
WEIGHT_FACTUAL = 15
WEIGHT_LENGTH = 10
WEIGHT_FORECAST_REF = 20


# ---------------------------------------------------------------------------
# Scoring rubric (100-point scale, golden-centric)
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    """Lowercase word tokenization for content comparison."""
    return re.findall(r"[a-z0-9][\w\-\.]*", text.lower())


def _ngrams(tokens: list[str], n: int) -> Counter:
    """Generate n-gram counter from token list."""
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def _extract_technical_terms(text: str) -> set[str]:
    """Extract technical terms: uppercase-starting words, flags, numbers with units."""
    terms: set[str] = set()
    terms.update(re.findall(r"\b[A-Z][A-Za-z0-9_\-]{2,}\b", text))
    terms.update(re.findall(r"--[\w\-]+", text))
    terms.update(re.findall(r"\b(?:TP|PP|TP=\d+|PP=\d+)\b", text))
    terms.update(re.findall(r"\b\d+[×x]\s*\w+", text))
    terms.update(re.findall(r"\b\d+\s*(?:GB|MB|ms|tok/s|Gbps)\b", text))
    return terms


def _extract_key_decisions(text: str) -> dict[str, str | None]:
    """Extract key architectural decisions for factual alignment."""
    decisions: dict[str, str | None] = {}

    tp_match = re.search(r"TP[=\s]*(\d+)", text)
    decisions["tp_size"] = tp_match.group(1) if tp_match else None

    pp_match = re.search(r"PP[=\s]*(\d+)", text)
    decisions["pp_size"] = pp_match.group(1) if pp_match else None

    gpu_match = re.search(r"(\d+)[×x]\s*(?:H100|H200|A100|MI300X|B200)", text)
    decisions["gpu_count"] = gpu_match.group(1) if gpu_match else None

    max_model_len_match = re.search(r"max[_-]model[_-]len[\"']?\s*(?:to\s+)?(\d+)", text)
    decisions["max_model_len"] = max_model_len_match.group(1) if max_model_len_match else None

    decisions["multi_node"] = "yes" if re.search(
        r"multi[_-]?node|PP[=\s]*[2-9]|pipeline.parallelism.across.nodes?", text, re.IGNORECASE
    ) else "no"

    decisions["chunked_prefill"] = "yes" if re.search(r"chunked.prefill", text, re.IGNORECASE) else "no"
    decisions["prefix_caching"] = "yes" if re.search(r"prefix.cach", text, re.IGNORECASE) else "no"

    return decisions


def score_structure(prod: str, golden: str) -> float:
    """Structure score: sections present in prod that are also in golden (0-1)."""
    golden_sections = [s for s in REQUIRED_SECTIONS if s.lower() in golden.lower()]
    if not golden_sections:
        return 1.0
    found = sum(1 for s in golden_sections if s.lower() in prod.lower())
    return found / len(golden_sections)


def score_keyword_similarity(prod: str, golden: str) -> float:
    """Technical keyword Jaccard similarity against golden (0-1)."""
    prod_kw = _extract_technical_terms(prod)
    gold_kw = _extract_technical_terms(golden)
    if not gold_kw:
        return 1.0
    if not prod_kw:
        return 0.0
    intersection = prod_kw & gold_kw
    union = prod_kw | gold_kw
    return len(intersection) / len(union)


def score_content_similarity(prod: str, golden: str) -> float:
    """Content similarity via weighted unigram + bigram F1 against golden (0-1)."""
    prod_tokens = _tokenize(prod)
    gold_tokens = _tokenize(golden)

    if not gold_tokens:
        return 1.0
    if not prod_tokens:
        return 0.0

    uni_prod = Counter(prod_tokens)
    uni_gold = Counter(gold_tokens)
    uni_overlap = sum((uni_prod & uni_gold).values())
    uni_precision = uni_overlap / sum(uni_prod.values()) if uni_prod else 0
    uni_recall = uni_overlap / sum(uni_gold.values()) if uni_gold else 0
    uni_f1 = (2 * uni_precision * uni_recall / (uni_precision + uni_recall)) if (uni_precision + uni_recall) > 0 else 0

    bi_prod = _ngrams(prod_tokens, 2)
    bi_gold = _ngrams(gold_tokens, 2)
    bi_overlap = sum((bi_prod & bi_gold).values())
    bi_precision = bi_overlap / sum(bi_prod.values()) if bi_prod else 0
    bi_recall = bi_overlap / sum(bi_gold.values()) if bi_gold else 0
    bi_f1 = (2 * bi_precision * bi_recall / (bi_precision + bi_recall)) if (bi_precision + bi_recall) > 0 else 0

    return 0.4 * uni_f1 + 0.6 * bi_f1


def score_factual_alignment(prod: str, golden: str) -> float:
    """Factual alignment: key architectural decisions match golden (0-1)."""
    prod_decisions = _extract_key_decisions(prod)
    gold_decisions = _extract_key_decisions(golden)

    total = 0
    matched = 0
    for key, gold_val in gold_decisions.items():
        if gold_val is None:
            continue
        total += 1
        prod_val = prod_decisions.get(key)
        if prod_val == gold_val:
            matched += 1

    if total == 0:
        return 1.0
    return matched / total


def score_length_appropriateness(prod: str, golden: str) -> float:
    """Length appropriateness: penalize if significantly shorter or longer than golden (0-1)."""
    prod_len = len(prod)
    gold_len = len(golden)
    if gold_len == 0:
        return 1.0
    ratio = prod_len / gold_len
    if 0.7 <= ratio <= 1.4:
        return 1.0
    elif ratio < 0.7:
        return max(0.0, ratio / 0.7)
    else:
        return max(0.0, 1.0 - (ratio - 1.4) / 1.6)


def score_forecast_reference(prod: str, golden: str, input_ctx: dict | None = None) -> float:
    """Forecast cross-reference: checks if key performance numbers are correctly cited (0-1).

    Verifies that the design suggestion correctly references:
    - Decode throughput (tok/s)
    - TPOT estimate (ms)
    - TTFT estimate (ms)
    - Ridge batch size
    - GPU memory utilization / fit status
    - GPU configuration
    """
    if not input_ctx:
        return 1.0

    checks = []

    # Check decode throughput reference
    decode_tps = input_ctx.get("decode_tps", "")
    if decode_tps:
        tps_val = str(decode_tps).split(".")[0]
        if len(tps_val) >= 2:
            checks.append(_num_referenced(prod, float(decode_tps), tolerance=0.1))

    # Check TPOT reference
    tpot = input_ctx.get("estimated_tpot_ms", "")
    if tpot:
        checks.append(_num_referenced(prod, float(tpot), tolerance=0.2))

    # Check TTFT reference
    ttft = input_ctx.get("estimated_ttft_ms", "")
    if ttft:
        checks.append(_num_referenced(prod, float(ttft), tolerance=0.2))

    # Check ridge batch reference
    ridge = input_ctx.get("ridge_batch", "")
    if ridge:
        checks.append(_num_referenced(prod, float(ridge), tolerance=0.15))

    # Check utilization/fit awareness
    fits = input_ctx.get("fits", "")
    utilization = input_ctx.get("utilization_pct", "")
    if fits == "No":
        memory_aware = any(
            term in prod.lower()
            for term in ["exceed", "insufficient", "doesn't fit", "not fit",
                         "overflow", "deficit", "shortfall", "exceed",
                         "tp=2", "tp=4", "tp=8", "multi-gpu", "multi-node",
                         "pipeline parallelism", "additional gpu", "more gpu"]
        )
        checks.append(1.0 if memory_aware else 0.0)
    elif utilization and float(utilization) < 50:
        underutil_aware = any(
            term in prod.lower()
            for term in ["underutiliz", "mig", "multi-instance", "larger model",
                         "fp16 instead", "higher precision", "co-locate"]
        )
        checks.append(1.0 if underutil_aware else 0.0)

    # Check GPU config reference
    gpu_type = input_ctx.get("gpu_type", "")
    gpu_count = input_ctx.get("gpu_count", "")
    if gpu_type and gpu_count:
        gpu_short = gpu_type.split("-")[0]
        gpu_mentioned = gpu_short.lower() in prod.lower()
        count_mentioned = gpu_count in prod
        checks.append(1.0 if (gpu_mentioned and count_mentioned) else 0.5 if gpu_mentioned else 0.0)

    # Check target awareness (does it discuss whether targets are met?)
    tpot_target = input_ctx.get("tpot_target_ms", "")
    if tpot_target and tpot:
        target_discussed = any(
            term in prod.lower()
            for term in ["target", "budget", "meets", "within", "below",
                         "exceed", "margin", "headroom", "tight"]
        )
        checks.append(1.0 if target_discussed else 0.3)

    if not checks:
        return 1.0
    return sum(checks) / len(checks)


def _num_referenced(text: str, value: float, tolerance: float = 0.1) -> float:
    """Check if a numeric value is referenced in the text (within tolerance).

    Returns 1.0 if found, 0.5 if approximately found, 0.0 if absent.
    """
    import re

    numbers = re.findall(r"[\d,]+\.?\d*", text)
    for num_str in numbers:
        try:
            num = float(num_str.replace(",", ""))
            if value == 0:
                continue
            ratio = abs(num - value) / abs(value)
            if ratio < 0.01:
                return 1.0
            elif ratio < tolerance:
                return 0.5
        except ValueError:
            continue
    return 0.0


def compute_total_score(row: pd.Series) -> float:
    """Compute weighted total score (0-100) for a single row."""
    prod = str(row.get("output", ""))
    golden = str(row.get("golden_output", ""))
    input_ctx = row.get("input") if isinstance(row.get("input"), dict) else None

    s_structure = score_structure(prod, golden)
    s_keyword = score_keyword_similarity(prod, golden)
    s_content = score_content_similarity(prod, golden)
    s_factual = score_factual_alignment(prod, golden)
    s_length = score_length_appropriateness(prod, golden)
    s_forecast = score_forecast_reference(prod, golden, input_ctx)

    total = (
        s_structure * WEIGHT_STRUCTURE
        + s_keyword * WEIGHT_KEYWORD
        + s_content * WEIGHT_CONTENT
        + s_factual * WEIGHT_FACTUAL
        + s_length * WEIGHT_LENGTH
        + s_forecast * WEIGHT_FORECAST_REF
    )
    return round(total, 1)


def compute_subscores(row: pd.Series) -> dict[str, float]:
    """Compute all subscores for a single row."""
    prod = str(row.get("output", ""))
    golden = str(row.get("golden_output", ""))
    input_ctx = row.get("input") if isinstance(row.get("input"), dict) else None
    return {
        "structure": round(score_structure(prod, golden) * WEIGHT_STRUCTURE, 1),
        "keyword_sim": round(score_keyword_similarity(prod, golden) * WEIGHT_KEYWORD, 1),
        "content_sim": round(score_content_similarity(prod, golden) * WEIGHT_CONTENT, 1),
        "factual_align": round(score_factual_alignment(prod, golden) * WEIGHT_FACTUAL, 1),
        "length_approp": round(score_length_appropriateness(prod, golden) * WEIGHT_LENGTH, 1),
        "forecast_ref": round(score_forecast_reference(prod, golden, input_ctx) * WEIGHT_FORECAST_REF, 1),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_golden(path: Path) -> pd.DataFrame:
    if not path.exists():
        print(f"ERROR: {path} not found. Run generate_golden.py first.")
        sys.exit(1)

    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    if not records:
        print("ERROR: golden file is empty.")
        sys.exit(1)

    print(f"Loaded {len(records)} golden record(s)")
    return pd.DataFrame(records)


def compute_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Add score columns to the DataFrame."""
    if "original_output" in df.columns:
        df.rename(columns={"original_output": "output"}, inplace=True, errors="ignore")

    df["total_score"] = df.apply(compute_total_score, axis=1)

    subscores = df.apply(compute_subscores, axis=1, result_type="expand")
    df = pd.concat([df, subscores], axis=1)
    return df


def print_results(df: pd.DataFrame, *, mlflow_info: dict | None = None) -> None:
    """Print evaluation results to stdout."""
    n = len(df)
    avg_score = df["total_score"].mean()

    print(f"\n{'=' * 60}")
    print(f"  Golden Similarity Evaluation ({n} samples)")
    print(f"{'=' * 60}")
    print(f"  TOTAL SCORE:    {avg_score:.1f} / 100")
    print(f"{'─' * 60}")
    print(f"  Breakdown (avg):")
    print(f"    Structure      ({WEIGHT_STRUCTURE:2d} pts): {df['structure'].mean():5.1f}")
    print(f"    Keyword Sim    ({WEIGHT_KEYWORD:2d} pts): {df['keyword_sim'].mean():5.1f}")
    print(f"    Content Sim    ({WEIGHT_CONTENT:2d} pts): {df['content_sim'].mean():5.1f}")
    print(f"    Factual Align  ({WEIGHT_FACTUAL:2d} pts): {df['factual_align'].mean():5.1f}")
    print(f"    Length Approp  ({WEIGHT_LENGTH:2d} pts): {df['length_approp'].mean():5.1f}")
    print(f"    Forecast Ref   ({WEIGHT_FORECAST_REF:2d} pts): {df['forecast_ref'].mean():5.1f}")
    print(f"{'─' * 60}")

    for _, row in df.iterrows():
        repo = row.get("input", {}).get("model_repo_id", "?") if isinstance(row.get("input"), dict) else "?"
        print(f"  [{row['total_score']:5.1f}] {repo}")

    print(f"{'=' * 60}")
    if mlflow_info:
        print(f"  MLflow run:       {mlflow_info['run_id']}")
        print(f"  Experiment:       {mlflow_info['experiment']}")
        print(f"  Tracking URI:     {mlflow_info['uri']}")
    else:
        print(f"  Mode:             offline (--no-mlflow)")
    print(f"{'=' * 60}")


def save_results(df: pd.DataFrame) -> Path:
    """Save per-sample results to JSON and return the path."""
    per_sample = []
    for _, row in df.iterrows():
        per_sample.append({
            "session_id": row.get("session_id", ""),
            "model_repo_id": (
                row.get("input", {}).get("model_repo_id", "")
                if isinstance(row.get("input"), dict)
                else ""
            ),
            "total_score": row["total_score"],
            "structure": row["structure"],
            "keyword_sim": row["keyword_sim"],
            "content_sim": row["content_sim"],
            "factual_align": row["factual_align"],
            "length_approp": row["length_approp"],
            "forecast_ref": row["forecast_ref"],
        })

    results_path = TRACES_DIR / "eval_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(per_sample, f, indent=2, ensure_ascii=False)
    return results_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate design suggestions against golden references (100-pt scale)")
    parser.add_argument("--experiment", type=str, default=None, help="MLflow experiment name override")
    parser.add_argument("--no-mlflow", action="store_true", help="Run offline without MLflow logging")
    args = parser.parse_args()

    df = load_golden(GOLDEN_JSONL)
    df = compute_metrics(df)
    results_path = save_results(df)

    if args.no_mlflow:
        print_results(df)
        print(f"\n  Results saved to: {results_path}")
        return

    from backend.config import get_settings

    settings = get_settings()

    try:
        import mlflow
    except ImportError:
        print("ERROR: mlflow not installed. Run: uv pip install mlflow")
        sys.exit(1)

    if settings.mlflow_tracking_insecure_tls:
        os.environ.setdefault("MLFLOW_TRACKING_INSECURE_TLS", "true")
    if settings.mlflow_tracking_token:
        os.environ.setdefault("MLFLOW_TRACKING_TOKEN", settings.mlflow_tracking_token)
    if settings.mlflow_workspace:
        os.environ.setdefault("MLFLOW_WORKSPACE", settings.mlflow_workspace)

    os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "10")
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "1")

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    experiment_name = args.experiment or f"{settings.mlflow_experiment_name}-eval"

    try:
        mlflow.set_experiment(experiment_name)
    except Exception as exc:
        print(f"ERROR: MLflow unreachable — {exc}")
        print("Hint: use --no-mlflow to run offline, or check MLFLOW_TRACKING_URI in .env")
        sys.exit(1)

    with mlflow.start_run(run_name="design-suggestion-eval") as run:
        golden_model = df["golden_model"].iloc[0] if "golden_model" in df.columns else "unknown"
        prod_model = df["original_model"].iloc[0] if "original_model" in df.columns else "unknown"
        mlflow.log_param("golden_model", golden_model)
        mlflow.log_param("production_model", prod_model)
        mlflow.log_param("num_samples", len(df))

        mlflow.log_metric("total_score", df["total_score"].mean())
        mlflow.log_metric("structure", df["structure"].mean())
        mlflow.log_metric("keyword_sim", df["keyword_sim"].mean())
        mlflow.log_metric("content_sim", df["content_sim"].mean())
        mlflow.log_metric("factual_align", df["factual_align"].mean())
        mlflow.log_metric("length_approp", df["length_approp"].mean())
        mlflow.log_metric("forecast_ref", df["forecast_ref"].mean())

        mlflow.log_artifact(str(results_path))
        mlflow.log_artifact(str(GOLDEN_JSONL))

        print_results(df, mlflow_info={
            "run_id": run.info.run_id,
            "experiment": experiment_name,
            "uri": settings.mlflow_tracking_uri,
        })


if __name__ == "__main__":
    main()
