"use client";

import { useEffect, useState, useCallback } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import {
  CheckCircle2,
  AlertTriangle,
  ArrowLeft,
  GitCompare,
  Loader2,
  Trash2,
  Cpu,
  Copy,
  Check,
} from "lucide-react";
import { cn } from "@/lib/utils";

interface RecommendationItem {
  session_id: string;
  title: string | null;
  status: string;
  model_repo_id: string | null;
  completed_at: string | null;
  created_at: string;
  gpu_config: string | null;
  memory_utilization: string | null;
  fits: boolean | null;
}

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:7001";

export default function RecommendationsPage() {
  const router = useRouter();
  const [items, setItems] = useState<RecommendationItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [error, setError] = useState<string | null>(null);
  const [copiedId, setCopiedId] = useState<string | null>(null);

  useEffect(() => {
    fetchRecommendations();
  }, []);

  async function fetchRecommendations() {
    setLoading(true);
    try {
      const res = await fetch(`${API_BASE}/api/v1/recommendations`);
      if (!res.ok) throw new Error(`Failed to fetch: ${res.status}`);
      const data = await res.json();
      setItems(data.items || []);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load");
    } finally {
      setLoading(false);
    }
  }

  function toggleSelect(id: string) {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(id)) {
        next.delete(id);
      } else {
        if (next.size >= 2) return prev;
        next.add(id);
      }
      return next;
    });
  }

  async function handleDelete(id: string) {
    if (!confirm("Delete this recommendation?")) return;
    try {
      await fetch(`${API_BASE}/api/v1/designs/${id}`, { method: "DELETE" });
      setItems((prev) => prev.filter((item) => item.session_id !== id));
      setSelected((prev) => {
        const next = new Set(prev);
        next.delete(id);
        return next;
      });
    } catch {
      // silent fail
    }
  }

  const handleCopyReport = useCallback(async (id: string) => {
    try {
      const res = await fetch(`${API_BASE}/api/v1/designs/${id}`);
      if (!res.ok) throw new Error("Failed to fetch detail");
      const detail = await res.json();
      const reportText = buildExportPrompt(detail);
      await navigator.clipboard.writeText(reportText);
      setCopiedId(id);
      setTimeout(() => setCopiedId(null), 2500);
    } catch (err) {
      console.error("Failed to copy report:", err);
    }
  }, []);

  function handleCompare() {
    const ids = Array.from(selected);
    if (ids.length !== 2) return;
    router.push(`/recommendations/compare?a=${ids[0]}&b=${ids[1]}`);
  }

  const selectedArr = Array.from(selected);

  return (
    <main className="min-h-screen">
      <header className="border-b border-[var(--border)] px-6 py-4">
        <div className="flex items-center gap-3">
          <Link href="/" className="text-[var(--muted-foreground)] hover:text-[var(--foreground)] transition-colors">
            <ArrowLeft className="h-5 w-5" />
          </Link>
          <h1 className="text-xl font-semibold">Saved Recommendations</h1>
          <span className="rounded-full bg-[var(--accent)] px-2 py-0.5 text-xs text-[var(--primary)]">
            {items.length}
          </span>
        </div>
        <p className="mt-1 ml-8 text-sm text-[var(--muted-foreground)]">
          Select two recommendations to compare side-by-side
        </p>
      </header>

      <div className="mx-auto max-w-6xl px-6 py-8">
        {/* Compare Action Bar */}
        <div className={cn(
          "mb-6 flex items-center justify-between rounded-xl border p-4 transition-all",
          selected.size === 2
            ? "border-[var(--primary)]/30 bg-[var(--primary)]/5"
            : "border-[var(--border)] bg-[var(--muted)]/30"
        )}>
          <div className="text-sm">
            {selected.size === 0 && (
              <span className="text-[var(--muted-foreground)]">Select 2 items to compare</span>
            )}
            {selected.size === 1 && (
              <span className="text-[var(--muted-foreground)]">Select 1 more item to compare</span>
            )}
            {selected.size === 2 && (
              <span className="font-medium text-[var(--primary)]">Ready to compare</span>
            )}
          </div>
          <button
            onClick={handleCompare}
            disabled={selected.size !== 2}
            className={cn(
              "flex items-center gap-2 rounded-lg px-4 py-2.5 text-sm font-medium transition-all",
              selected.size === 2
                ? "bg-[var(--primary)] text-[var(--primary-foreground)] hover:opacity-90"
                : "bg-[var(--muted)] text-[var(--muted-foreground)] cursor-not-allowed"
            )}
          >
            <GitCompare className="h-4 w-4" />
            Compare
          </button>
        </div>

        {/* Loading */}
        {loading && (
          <div className="flex items-center justify-center py-20">
            <Loader2 className="h-6 w-6 animate-spin text-[var(--primary)]" />
          </div>
        )}

        {/* Error */}
        {error && (
          <div className="rounded-xl border border-red-500/30 bg-red-500/5 p-4 text-sm text-red-600">
            {error}
          </div>
        )}

        {/* Empty */}
        {!loading && !error && items.length === 0 && (
          <div className="flex flex-col items-center justify-center py-20 text-center">
            <Cpu className="mb-4 h-12 w-12 text-[var(--muted-foreground)]/50" />
            <h3 className="text-lg font-medium">No saved recommendations</h3>
            <p className="mt-1 text-sm text-[var(--muted-foreground)]">
              Complete a design session and save the recommendation to see it here.
            </p>
            <Link
              href="/"
              className="mt-4 rounded-lg bg-[var(--primary)] px-4 py-2 text-sm font-medium text-[var(--primary-foreground)] hover:opacity-90"
            >
              Start New Design
            </Link>
          </div>
        )}

        {/* List */}
        {!loading && items.length > 0 && (
          <div className="space-y-3">
            {items.map((item) => {
              const isSelected = selected.has(item.session_id);
              return (
                <div
                  key={item.session_id}
                  onClick={() => toggleSelect(item.session_id)}
                  className={cn(
                    "group flex cursor-pointer items-center gap-4 rounded-xl border p-4 transition-all",
                    isSelected
                      ? "border-[var(--primary)] bg-[var(--primary)]/5 ring-1 ring-[var(--primary)]/30"
                      : "border-[var(--border)] hover:border-[var(--primary)]/30 hover:bg-[var(--muted)]/30"
                  )}
                >
                  {/* Selection indicator */}
                  <div className={cn(
                    "flex h-6 w-6 shrink-0 items-center justify-center rounded-full border-2 transition-all",
                    isSelected
                      ? "border-[var(--primary)] bg-[var(--primary)]"
                      : "border-[var(--border)] group-hover:border-[var(--primary)]/50"
                  )}>
                    {isSelected && <CheckCircle2 className="h-4 w-4 text-white" />}
                  </div>

                  {/* Content */}
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2">
                      <h3 className="text-sm font-semibold truncate">
                        {item.model_repo_id || item.title || "Unnamed"}
                      </h3>
                      {item.fits === true && (
                        <CheckCircle2 className="h-3.5 w-3.5 shrink-0 text-green-500" />
                      )}
                      {item.fits === false && (
                        <AlertTriangle className="h-3.5 w-3.5 shrink-0 text-red-500" />
                      )}
                    </div>
                    <div className="mt-1 flex items-center gap-3 text-xs text-[var(--muted-foreground)]">
                      {item.gpu_config && (
                        <span className="rounded bg-[var(--muted)] px-1.5 py-0.5">{item.gpu_config}</span>
                      )}
                      {item.memory_utilization && (
                        <span>Memory: {item.memory_utilization}</span>
                      )}
                      <span>
                        {item.completed_at
                          ? new Date(item.completed_at).toLocaleDateString("ko-KR", {
                              year: "numeric",
                              month: "short",
                              day: "numeric",
                              hour: "2-digit",
                              minute: "2-digit",
                            })
                          : new Date(item.created_at).toLocaleDateString("ko-KR")}
                      </span>
                    </div>
                  </div>

                  {/* Chip showing selection order */}
                  {isSelected && (
                    <span className="shrink-0 rounded-full bg-[var(--primary)] px-2 py-0.5 text-[10px] font-bold text-white">
                      {selectedArr.indexOf(item.session_id) === 0 ? "A" : "B"}
                    </span>
                  )}

                  {/* Copy Report */}
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      handleCopyReport(item.session_id);
                    }}
                    title="Copy report prompt"
                    className={cn(
                      "shrink-0 rounded p-1.5 transition-all",
                      copiedId === item.session_id
                        ? "text-green-600 opacity-100"
                        : "text-[var(--muted-foreground)] opacity-0 hover:bg-[var(--primary)]/10 hover:text-[var(--primary)] group-hover:opacity-100"
                    )}
                  >
                    {copiedId === item.session_id ? (
                      <Check className="h-4 w-4" />
                    ) : (
                      <Copy className="h-4 w-4" />
                    )}
                  </button>

                  {/* Delete */}
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      handleDelete(item.session_id);
                    }}
                    className="shrink-0 rounded p-1.5 text-[var(--muted-foreground)] opacity-0 transition-all hover:bg-red-500/10 hover:text-red-500 group-hover:opacity-100"
                  >
                    <Trash2 className="h-4 w-4" />
                  </button>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </main>
  );
}

// --- Export helper ---

function buildExportPrompt(detail: {
  session_id: string;
  model_repo_id: string | null;
  result_snapshot: {
    view_model: {
      sections: Array<{ id: string; data: Record<string, unknown> }>;
    };
  } | null;
}): string {
  const sections = detail.result_snapshot?.view_model?.sections || [];
  const getSection = (id: string) => sections.find((s) => s.id === id)?.data || {};

  const model = getSection("model_summary");
  const mem = getSection("memory_estimate");
  const dep = getSection("deployment_config");
  const perf = getSection("performance_forecast");
  const cost = getSection("cost_estimate");
  const evidence = getSection("evidence_summary");
  const suggestion = getSection("design_suggestion");

  const v = (val: unknown, fallback = "?") => (val != null && val !== "" ? String(val) : fallback);
  const precLabel = ({ 4: "INT4", 8: "FP8", 16: "FP16", 32: "FP32" } as Record<number, string>)[
    Number(mem.precision_bits) || 16
  ] ?? "FP16";
  const useCases = (dep.use_case_presets as string[] | undefined)?.join(", ") || "General inference";

  const evidenceLines: string[] = [];
  const recipe = evidence.vllm_recipe as Record<string, unknown> | undefined;
  const evals = evidence.evaluations as Record<string, unknown> | undefined;
  const compat = evidence.rhoai_compatibility as Record<string, unknown> | undefined;
  if (recipe) evidenceLines.push(`- [vLLM Recipe] ${v(recipe.count)} items, vLLM ≥ ${v(recipe.min_vllm_version)}`);
  if (evals) evidenceLines.push(`- [Benchmarks] ${v(evals.accuracy_benchmarks)} accuracy, ${v(evals.performance_benchmarks)} perf`);
  if (compat) evidenceLines.push(`- [RHOAI] ${compat.is_validated ? "Validated" : "Not validated"}${compat.matrix_version ? ` (v${v(compat.matrix_version)})` : ""}`);

  const costSummary = cost.type === "cloud"
    ? `Cloud (${v(cost.platform)}): $${v(cost.on_demand_hourly_usd)}/hr on-demand, $${Number(cost.monthly_on_demand_usd || 0).toLocaleString()}/mo`
    : `On-Prem TCO: $${Number(cost.monthly_total_usd || 0).toLocaleString()}/mo (${v(dep.gpu_count)}× ${v(dep.gpu_type)})`;

  const systemPrompt = `You are a senior AI infrastructure architect specializing in GPU inference deployments on Red Hat OpenShift AI with vLLM. You provide concise, structured, and actionable deployment architecture recommendations based on quantitative evidence.

Write in clear, professional English. Be direct and specific. Focus on architectural decisions that matter for production inference workloads.`;

  const userPrompt = `Based on the following deployment context, provide a concise inference architecture recommendation.

## Model
- Repository: ${v(model.repo_id)}
- Architecture: ${v(model.architecture_type)}
- Parameters: ${v(model.parameters_display)}
- Context length: ${v(model.context_length)} tokens
- Precision: ${precLabel}

## Hardware Configuration
- Platform: ${v(dep.platform || dep.environment_type, "on-premise")}
- GPU: ${v(dep.gpu_count)}× ${v(dep.gpu_type)}
- Total VRAM: ${v(mem.total_available_gb)} GB

## Memory Analysis
- Model weights: ${v(mem.model_weights_gb)} GB
- KV cache (est.): ${v(mem.kv_cache_gb)} GB
- Total required: ${v(mem.total_required_min_gb ?? mem.total_required_gb)} GB / ${v(mem.total_available_gb)} GB available
- Utilization: ${v(mem.utilization_pct)}%
- Fits: ${mem.fits ? "Yes" : "No"}

## Performance Targets
- Use cases: ${useCases}
- Target users: ${v(dep.target_end_users)}
- Max concurrent requests: ${v(dep.max_concurrent_requests)}
- TTFT target: ${v(dep.ttft_target_ms)}ms
- TPOT target: ${v(dep.tpot_target_ms)}ms

## Performance Forecast (Roofline)
- Theoretical decode throughput: ${v(perf.theoretical_decode_tps)} tok/s (batch=1)
- Estimated TPOT: ${v(perf.estimated_tpot_ms)}ms
- Estimated TTFT: ${v(perf.estimated_ttft_ms)}ms
- Ridge batch size: ${v(perf.ridge_batch_size)}
- Max batch at target TPOT: ${v(perf.max_batch_at_target_tpot)}

## Monthly Cost
${costSummary}

## Evidence Collected
${evidenceLines.length > 0 ? evidenceLines.join("\n") : "No evidence collected"}

---

Provide your recommendation in the following structure:

### Architecture Direction
A 2-3 sentence summary of the recommended deployment architecture, including parallelism strategy (tensor parallelism, pipeline parallelism, or both) and whether distributed inference (llm-d) is warranted.

### Key Considerations
3-5 bullet points covering:
- Memory headroom and quantization tradeoffs
- Parallelism strategy rationale (TP vs PP vs single-GPU)
- Latency vs throughput tradeoff given the target use cases
- vLLM configuration recommendations (e.g., chunked prefill, prefix caching)
- Any scaling or capacity concerns

### Risk Factors
2-3 bullet points on risks or limitations to watch for.

### Alternative Approaches
1-2 bullet points suggesting alternative configurations if requirements change (e.g., different GPU count, quantization, or model variant).`;

  const inputContext: Record<string, string> = {
    model_repo_id: v(model.repo_id),
    architecture_type: v(model.architecture_type),
    parameters_display: v(model.parameters_display),
    context_length: v(model.context_length),
    precision_label: precLabel,
    precision_bits: v(mem.precision_bits, "16"),
    platform: v(dep.platform || dep.environment_type, "on-premise"),
    gpu_type: v(dep.gpu_type),
    gpu_count: v(dep.gpu_count),
    total_vram_gb: v(mem.total_available_gb),
    model_weights_gb: v(mem.model_weights_gb),
    kv_cache_gb: v(mem.kv_cache_gb),
    total_required_gb: v(mem.total_required_min_gb ?? mem.total_required_gb),
    total_available_gb: v(mem.total_available_gb),
    utilization_pct: v(mem.utilization_pct),
    fits: mem.fits ? "Yes" : "No",
    use_cases: useCases,
    target_users: v(dep.target_end_users),
    max_concurrent: v(dep.max_concurrent_requests),
    ttft_target_ms: v(dep.ttft_target_ms),
    tpot_target_ms: v(dep.tpot_target_ms),
    decode_tps: v(perf.theoretical_decode_tps),
    estimated_tpot_ms: v(perf.estimated_tpot_ms),
    estimated_ttft_ms: v(perf.estimated_ttft_ms),
    ridge_batch: v(perf.ridge_batch_size),
    max_batch_at_target: v(perf.max_batch_at_target_tpot),
    cost_summary: costSummary,
    evidence_summary: evidenceLines.join("; "),
  };

  const goldenRecord = {
    trace_id: "",
    session_id: detail.session_id,
    timestamp: new Date().toISOString(),
    original_model: v(suggestion.model_used, ""),
    golden_model: "<MODEL_NAME>",
    input: inputContext,
    system_prompt: systemPrompt,
    user_prompt: userPrompt,
    original_output: (suggestion.content as string) || "",
    golden_output: "<PASTE_RESPONSE_HERE>",
    golden_latency_ms: 0,
  };

  const jsonlLine = JSON.stringify(goldenRecord);

  return [
    "=== PROMPT (paste into ChatGPT / Claude) ===",
    "",
    `[System]\n${systemPrompt}`,
    "",
    `[User]\n${userPrompt}`,
    "",
    "",
    "=== GOLDEN JSONL (replace <PASTE_RESPONSE_HERE> with the response, then append to data/traces/golden_suggestions.jsonl) ===",
    "",
    jsonlLine,
  ].join("\n");
}
