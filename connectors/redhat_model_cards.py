"""Red Hat AI model-card connector.

Fetches RedHatAI model cards from Hugging Face and parses evaluation tables.
Accuracy evaluations are QUALITY evidence, not serving-performance evidence.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
from datetime import UTC, datetime
from functools import partial
from typing import Any

import httpx
from huggingface_hub import HfApi, ModelInfo

from domain.evidence import EvidenceItem

logger = logging.getLogger(__name__)

_REDHAT_ORG = "RedHatAI"
_HF_CARD_URL = "https://huggingface.co/{repo_id}"
_QUANT_SUFFIXES_RE = re.compile(
    r"[-_](?:FP8[-_]?(?:dynamic|block|static)?|FP4|NVFP4|GPTQ|AWQ|INT8|INT4|"
    r"W4A16|W8A8|quantized\.\w+)",
    re.IGNORECASE,
)
_MARKDOWN_TABLE_RE = re.compile(
    r"^\|(?P<row>.+)\|$",
    re.MULTILINE,
)
_SERVING_METRICS = frozenset({
    "ttft", "tpot", "throughput", "tokens/s", "tok/s",
    "latency", "requests/s", "req/s", "p50", "p90", "p99",
    "time to first token", "time per output token",
    "itl", "inter-token latency",
})


def _excerpt_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _strip_quant_suffix(name: str) -> str:
    return _QUANT_SUFFIXES_RE.sub("", name)


def _base_model_name(repo_id: str) -> str:
    """Extract the model name after the org prefix and strip quantization."""
    _, _, name = repo_id.rpartition("/")
    return _strip_quant_suffix(name).lower()


class RedHatModelCardConnector:
    """Fetches and parses Red Hat AI model cards from Hugging Face."""

    def __init__(self, token: str | None = None) -> None:
        self._api = HfApi(token=token or None)
        self._token = token or None

    async def find_redhat_evidence(self, repo_id: str) -> list[EvidenceItem]:
        """Find benchmark/evaluation evidence from model cards.

        Search strategy:
        1. Fetch the original model's own card (may have benchmarks)
        2. Search ``RedHatAI/<model>`` variants for evaluation data
        3. Red Hat quantized variant whose base matches

        Returns an EvidenceItem list (may be empty).
        """
        evidence: list[EvidenceItem] = []

        # First: try the original model's own card for benchmarks
        card_text = await self._fetch_model_card(repo_id)
        if card_text:
            evidence.extend(self._parse_model_card(card_text, repo_id))

        # Second: search for RedHatAI variants
        try:
            redhat_repos = await self._search_redhat_models(repo_id)
        except Exception:
            logger.warning("HuggingFace search for RedHatAI models failed", exc_info=True)
            return evidence

        for rh_repo in redhat_repos[:3]:
            if rh_repo == repo_id:
                continue
            rh_card = await self._fetch_model_card(rh_repo)
            if rh_card:
                evidence.extend(self._parse_model_card(rh_card, rh_repo))

        return evidence

    # ------------------------------------------------------------------
    # HuggingFace search
    # ------------------------------------------------------------------

    async def _search_redhat_models(self, repo_id: str) -> list[str]:
        """Search for matching ``RedHatAI/*`` models on HuggingFace.

        The HfApi.list_models call is synchronous, so we run it in an
        executor to avoid blocking the event loop.
        """
        base_name = _base_model_name(repo_id)
        _, _, short_name = repo_id.rpartition("/")

        loop = asyncio.get_running_loop()
        search_fn = partial(
            self._api.list_models,
            author=_REDHAT_ORG,
            search=short_name,
            sort="downloads",
            limit=20,
        )
        try:
            results: list[ModelInfo] = await loop.run_in_executor(None, lambda: list(search_fn()))
        except Exception:
            logger.warning("HuggingFace API search failed for %s", short_name, exc_info=True)
            return []

        exact: list[str] = []
        fuzzy: list[str] = []

        for model in results:
            model_repo = model.modelId or ""
            if not model_repo.startswith(f"{_REDHAT_ORG}/"):
                continue
            rh_base = _base_model_name(model_repo)
            if rh_base == base_name:
                exact.append(model_repo)
            elif base_name in rh_base or rh_base in base_name:
                fuzzy.append(model_repo)

        return exact or fuzzy

    async def _fetch_model_card(self, redhat_repo_id: str) -> str | None:
        """Fetch the README.md (model card) content via HTTP."""
        url = f"https://huggingface.co/{redhat_repo_id}/raw/main/README.md"
        headers = {}
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"

        async with httpx.AsyncClient(timeout=httpx.Timeout(30.0)) as client:
            try:
                resp = await client.get(url, headers=headers)
                resp.raise_for_status()
                return resp.text
            except (httpx.HTTPStatusError, httpx.RequestError) as exc:
                logger.warning("Failed to fetch model card for %s: %s", redhat_repo_id, exc)
                return None

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    def _parse_model_card(self, card_text: str, redhat_repo_id: str) -> list[EvidenceItem]:
        """Parse a Red Hat model card into evidence items."""
        now = datetime.now(UTC)
        evidence: list[EvidenceItem] = []
        source_url = _HF_CARD_URL.format(repo_id=redhat_repo_id)

        common: dict[str, Any] = {
            "category": "redhat_evaluation",
            "source_url": source_url,
            "source_domain": "huggingface.co",
            "publisher": "Red Hat AI",
            "retrieved_at": now,
            "source_tier": "official_secondary",
            "verification_level": "verified",
        }

        vllm_version = self._extract_vllm_version(card_text)
        hardware_sig = self._extract_hardware(card_text)
        if vllm_version:
            common["vllm_version"] = vllm_version
        if hardware_sig:
            common["hardware_signature"] = hardware_sig

        sections = self._split_sections(card_text)

        for heading, body in sections:
            tables = self._extract_tables(body)
            for table_text in tables:
                rows = self._parse_evaluation_table(table_text)
                if not rows:
                    continue

                ev_type = self._classify_evidence_type(heading, rows)

                claim_type = "serving_performance" if ev_type == "serving_performance" else "accuracy"

                summary_lines: list[str] = []
                for row in rows:
                    cols = " | ".join(f"{k}: {v}" for k, v in row.items() if v)
                    summary_lines.append(cols)

                table_summary = "\n".join(summary_lines[:20])
                if len(summary_lines) > 20:
                    table_summary += f"\n... and {len(summary_lines) - 20} more rows"

                evidence.append(
                    EvidenceItem(
                        claim_type=claim_type,
                        title=f"{heading} — {redhat_repo_id}",
                        summary=table_summary,
                        raw_excerpt_hash=_excerpt_hash(table_text),
                        **common,
                    )
                )

        launch_cmd = self._extract_launch_command(card_text)
        if launch_cmd:
            evidence.append(
                EvidenceItem(
                    claim_type="compatibility",
                    title=f"Evaluation launch command — {redhat_repo_id}",
                    summary=launch_cmd,
                    raw_excerpt_hash=_excerpt_hash(launch_cmd),
                    **common,
                )
            )

        quant_method = self._extract_quantization_method(card_text)
        if quant_method:
            evidence.append(
                EvidenceItem(
                    claim_type="architecture",
                    title=f"Quantization method — {redhat_repo_id}",
                    summary=quant_method,
                    raw_excerpt_hash=_excerpt_hash(quant_method),
                    **common,
                )
            )

        limitations = self._extract_section_text(card_text, r"(?:Limitation|Intended Use|Disclaimer)")
        if limitations:
            evidence.append(
                EvidenceItem(
                    claim_type="limitation",
                    title=f"Limitations — {redhat_repo_id}",
                    summary=limitations[:2000],
                    raw_excerpt_hash=_excerpt_hash(limitations),
                    **common,
                )
            )

        return evidence

    # ------------------------------------------------------------------
    # Table extraction and parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _split_sections(text: str) -> list[tuple[str, str]]:
        """Split markdown into ``(heading, body)`` pairs."""
        parts = re.split(r"^(#{1,4}\s+.+)$", text, flags=re.MULTILINE)
        sections: list[tuple[str, str]] = []
        current_heading = "Preamble"
        for part in parts:
            stripped = part.strip()
            if re.match(r"^#{1,4}\s+", stripped):
                current_heading = re.sub(r"^#+\s*", "", stripped)
            else:
                if stripped:
                    sections.append((current_heading, stripped))
        return sections

    @staticmethod
    def _extract_tables(body: str) -> list[str]:
        """Pull contiguous markdown table blocks from a section body."""
        tables: list[str] = []
        current_table_lines: list[str] = []
        for line in body.splitlines():
            stripped = line.strip()
            if stripped.startswith("|") and stripped.endswith("|"):
                current_table_lines.append(stripped)
            else:
                if current_table_lines:
                    tables.append("\n".join(current_table_lines))
                    current_table_lines = []
        if current_table_lines:
            tables.append("\n".join(current_table_lines))
        return tables

    @staticmethod
    def _parse_evaluation_table(table_text: str) -> list[dict[str, str]]:
        """Parse a Markdown table into a list of row dicts."""
        lines = [ln.strip() for ln in table_text.strip().splitlines() if ln.strip()]
        if len(lines) < 2:
            return []

        def split_row(line: str) -> list[str]:
            line = line.strip("|")
            return [cell.strip() for cell in line.split("|")]

        headers = split_row(lines[0])

        separator_idx = 1
        separator_re = r"^[\|\s\-:]+$"
        data_start = 2 if separator_idx < len(lines) and re.match(separator_re, lines[separator_idx]) else 1

        rows: list[dict[str, str]] = []
        for line in lines[data_start:]:
            cells = split_row(line)
            row: dict[str, str] = {}
            for i, cell in enumerate(cells):
                key = headers[i] if i < len(headers) else f"col_{i}"
                row[key] = cell
            rows.append(row)

        return rows

    def _classify_evidence_type(self, section_heading: str, table_data: list[dict[str, str]]) -> str:
        """Classify whether evidence is accuracy or serving_performance.

        Serving performance requires explicit metrics like TTFT, TPOT,
        throughput, tokens/s. Everything else is accuracy/quality evidence.
        """
        check_text = section_heading.lower()
        for row in table_data:
            for val in row.values():
                check_text += " " + val.lower()

        for metric in _SERVING_METRICS:
            if metric in check_text:
                return "serving_performance"

        return "accuracy"

    # ------------------------------------------------------------------
    # Metadata extraction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_vllm_version(card_text: str) -> str | None:
        """Extract vLLM version from launch commands or explicit mentions."""
        patterns = [
            r"vllm[=<>!]+(\d+\.\d+(?:\.\d+)?)",
            r"vllm\s+(?:version\s+)?v?(\d+\.\d+(?:\.\d+)?)",
            r"vLLM\s+v?(\d+\.\d+(?:\.\d+)?)",
        ]
        for pat in patterns:
            m = re.search(pat, card_text, re.IGNORECASE)
            if m:
                return m.group(1)
        return None

    @staticmethod
    def _extract_hardware(card_text: str) -> str | None:
        """Extract hardware/GPU info from the card."""
        patterns = [
            r"(?:tensor_parallel_size|tp)\s*[=:]\s*(\d+)",
            r"(\d+)\s*[×x]\s*(A100|H100|H200|B200|B300|L40S?|MI\d+\w*|RTX\s*\w+)",
            r"(A100|H100|H200|B200|B300|L40S?|MI\d+\w*|RTX\s*\w+)\s*[×x]\s*(\d+)",
        ]
        parts: list[str] = []
        for pat in patterns:
            for m in re.finditer(pat, card_text, re.IGNORECASE):
                parts.append(m.group(0))
        return "; ".join(parts) if parts else None

    @staticmethod
    def _extract_launch_command(card_text: str) -> str | None:
        """Extract the first vLLM-related shell command from code blocks."""
        code_blocks = re.findall(r"```(?:\w*)\n(.*?)```", card_text, re.DOTALL)
        for block in code_blocks:
            if "vllm" in block.lower() or "lm_eval" in block.lower() or "lighteval" in block.lower():
                trimmed = block.strip()
                if len(trimmed) > 2000:
                    trimmed = trimmed[:2000] + "..."
                return trimmed
        return None

    @staticmethod
    def _extract_quantization_method(card_text: str) -> str | None:
        """Extract quantization method description."""
        patterns = [
            r"(?:quantiz(?:ed|ation)\s+(?:method|scheme|using|with|via))\s*[:\-]?\s*(.+?)(?:\.|$)",
            r"(?:FP8|INT8|INT4|W4A16|W8A8|GPTQ|AWQ)\s+quantiz\w+\s+(?:using|with|via)\s+(.+?)(?:\.|$)",
        ]
        for pat in patterns:
            m = re.search(pat, card_text, re.IGNORECASE | re.MULTILINE)
            if m:
                return m.group(0).strip()

        for keyword in ("FP8-dynamic", "FP8-block", "FP8-static", "GPTQ", "AWQ", "W4A16", "W8A8"):
            if keyword.lower() in card_text.lower():
                return keyword

        return None

    @staticmethod
    def _extract_section_text(card_text: str, heading_pattern: str) -> str | None:
        """Extract the text body under a heading matching *heading_pattern*."""
        m = re.search(
            rf"^#{1,4}\s+.*{heading_pattern}.*$\n(.*?)(?=^#{1,4}\s|\Z)",
            card_text,
            re.MULTILINE | re.DOTALL | re.IGNORECASE,
        )
        if m:
            text = m.group(1).strip()
            return text if text else None
        return None
