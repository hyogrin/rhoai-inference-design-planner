"use client";

import { useState, useCallback, useEffect } from "react";
import { useRouter } from "next/navigation";
import {
  Cpu,
  MemoryStick,
  DollarSign,
  TrendingUp,
  AlertTriangle,
  CheckCircle2,
  BookOpen,
  ShieldCheck,
  Copy, Check,
  Loader2,
  ExternalLink,
  Lightbulb,
  Server,
  List,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { useI18n } from "@/lib/i18n";
import type { AgentState } from "@/lib/use-agent-stream";

interface RecommendationStepProps {
  agentState: AgentState;
}

export function RecommendationStep({ agentState }: RecommendationStepProps) {
  const { t } = useI18n();
  const router = useRouter();
  const isRunning = agentState.status === "running";
  const [copied, setCopied] = useState(false);
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);

  const viewModel = extractViewModel(agentState);
  const rec: Array<{ id: string; title: string; data: Record<string, unknown> }> = viewModel?.sections || [];
  const verification = viewModel?.verification as { warnings: string[]; status: string } | undefined;

  const modelSummary = rec.find((s) => s.id === "model_summary")?.data;
  const evidenceSummary = rec.find((s) => s.id === "evidence_summary")?.data;
  const deploymentConfig = rec.find((s) => s.id === "deployment_config")?.data;
  const memoryEstimate = rec.find((s) => s.id === "memory_estimate")?.data;
  const costEstimate = rec.find((s) => s.id === "cost_estimate")?.data;
  const perfForecast = rec.find((s) => s.id === "performance_forecast")?.data;
  const designSuggestion = rec.find((s) => s.id === "design_suggestion")?.data;

  const handleExport = useCallback(() => {
    const text = buildExportPrompt({
      modelSummary,
      memoryEstimate,
      deploymentConfig,
      costEstimate,
      perfForecast,
      evidenceSummary,
      designSuggestion,
    });
    navigator.clipboard.writeText(text).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2500);
    });
  }, [modelSummary, memoryEstimate, deploymentConfig, costEstimate, perfForecast, evidenceSummary, designSuggestion]);

  const handleSave = useCallback(async () => {
    if (!viewModel || saving || saved) return;
    setSaving(true);
    try {
      const apiBase = process.env.NEXT_PUBLIC_API_URL || "http://localhost:7001";
      const res = await fetch(`${apiBase}/api/v1/recommendations`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          recommendation: { verification },
          view_model: viewModel,
        }),
      });
      if (!res.ok) throw new Error(`Save failed: ${res.status}`);
      setSaved(true);
    } catch (err) {
      console.error("Failed to save recommendation:", err);
      setSaving(false);
    }
  }, [viewModel, saving, saved, verification]);

  // Auto-save when recommendation is complete
  useEffect(() => {
    if (viewModel && !isRunning && !saving && !saved) {
      handleSave();
    }
  }, [viewModel, isRunning]); // eslint-disable-line react-hooks/exhaustive-deps

  if (isRunning || !viewModel) {
    return (
      <div className="space-y-6">
        <div>
          <h2 className="text-lg font-semibold">{t("step5.title")}</h2>
          <p className="mt-1 text-sm text-[var(--muted-foreground)]">
            {t("step5.generating")}
          </p>
        </div>
        <div className="flex items-center gap-3 rounded-xl border border-[var(--primary)]/20 bg-[var(--primary)]/5 p-6">
          <Loader2 className="h-5 w-5 animate-spin text-[var(--primary)]" />
          <span className="text-sm font-medium text-[var(--primary)]">
            {t("step5.computing")}
          </span>
        </div>
        <SkeletonPanels />
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-lg font-semibold">{t("step5.title")}</h2>
          <p className="mt-1 text-sm text-[var(--muted-foreground)]">
            {t("step5.subtitle", { model: String(modelSummary?.repo_id || "your model") })}
          </p>
        </div>
        <div className="flex items-center gap-2">
          {/* Auto-save status indicator */}
          <span className={cn(
            "flex items-center gap-1.5 text-xs font-medium",
            saved ? "text-green-600" : saving ? "text-[var(--muted-foreground)]" : ""
          )}>
            {saved ? (
              <>
                <Check className="h-3.5 w-3.5" />
                {t("step5.autoSaved")}
              </>
            ) : saving ? (
              <>
                <Loader2 className="h-3.5 w-3.5 animate-spin" />
                {t("step5.saving")}
              </>
            ) : null}
          </span>
          <button
            onClick={() => router.push("/recommendations")}
            className="flex items-center gap-2 rounded-lg border border-[var(--primary)]/30 bg-[var(--primary)]/5 px-4 py-2.5 text-sm font-medium text-[var(--primary)] hover:bg-[var(--primary)]/10 transition-all"
          >
            <List className="h-4 w-4" />
            {t("step5.viewAll")}
          </button>
          <button
            onClick={handleExport}
            className={cn(
              "flex items-center gap-2 rounded-lg border px-4 py-2.5 text-sm font-medium transition-all",
              copied
                ? "border-green-500/30 bg-green-500/10 text-green-700"
                : "border-[var(--border)] hover:bg-[var(--muted)]"
            )}
          >
            {copied ? (
              <>
                <Check className="h-4 w-4" />
                {t("step5.copiedClipboard")}
              </>
            ) : (
              <>
                <Copy className="h-4 w-4" />
                {t("step5.exportPrompt")}
              </>
            )}
          </button>
        </div>
      </div>

      {/* Verification Warnings */}
      {verification && verification.warnings && verification.warnings.length > 0 && (
        <div className="rounded-xl border border-amber-500/30 bg-amber-500/5 p-4">
          <div className="mb-2 flex items-center gap-2 text-amber-600">
            <AlertTriangle className="h-4 w-4" />
            <span className="text-sm font-medium">{t("step5.attention")}</span>
          </div>
          <ul className="space-y-1 text-xs text-amber-700">
            {verification.warnings.map((w, i) => (
              <li key={i}>• {w}</li>
            ))}
          </ul>
        </div>
      )}

      {/* Section 1: Model Overview + Evidence Summary */}
      <div className="grid gap-4 sm:grid-cols-2">
        <ModelOverviewCard data={modelSummary} />
        <EvidenceSummaryCard data={evidenceSummary} />
      </div>

      {/* Section 2: Deployment + Memory + Cost */}
      <div className="grid gap-4 sm:grid-cols-3">
        <DeploymentCard data={deploymentConfig} />
        <MemoryCard data={memoryEstimate} />
        <CostCard data={costEstimate} />
      </div>

      {/* Section 2.5: Use Case Distribution */}
      {deploymentConfig && <UseCaseChart data={deploymentConfig} />}

      {/* Section 3: Performance Forecast (Graph) */}
      {perfForecast && <PerformanceForecastChart data={perfForecast} />}

      {/* Section 4: Design Suggestion (LLM) */}
      {designSuggestion && <DesignSuggestionCard data={designSuggestion} />}
    </div>
  );
}

// --- Sub-components ---

function ModelOverviewCard({ data }: { data: Record<string, unknown> | undefined }) {
  const { t } = useI18n();
  if (!data) return null;
  return (
    <div className="rounded-xl border border-[var(--border)] p-5">
      <div className="mb-4 flex items-center gap-2">
        <Cpu className="h-4 w-4 text-[var(--primary)]" />
        <h3 className="text-sm font-semibold">{t("step5.modelOverview")}</h3>
      </div>
      <div className="space-y-2 text-xs">
        <Row label="Model" value={data.repo_id as string} />
        <Row label="Architecture" value={data.architecture_type as string} />
        <Row label="Parameters" value={data.parameters_display as string} />
        <Row label="Context Length" value={data.context_length ? `${Number(data.context_length).toLocaleString()} tokens` : "—"} />
        <Row label="Layers" value={String(data.num_layers || "—")} />
        <Row label="KV Heads" value={String(data.num_kv_heads || "—")} />
        {data.license ? <Row label="License" value={String(data.license)} /> : null}
      </div>
    </div>
  );
}

function EvidenceSummaryCard({ data }: { data: Record<string, unknown> | undefined }) {
  const { t } = useI18n();
  if (!data) return null;
  const recipe = data.vllm_recipe as Record<string, unknown> | undefined;
  const evals = data.evaluations as Record<string, unknown> | undefined;
  const compat = data.rhoai_compatibility as Record<string, unknown> | undefined;

  return (
    <div className="rounded-xl border border-[var(--border)] p-5">
      <div className="mb-4 flex items-center gap-2">
        <BookOpen className="h-4 w-4 text-[var(--primary)]" />
        <h3 className="text-sm font-semibold">{t("step5.collectedEvidence")}</h3>
        <span className="ml-auto rounded-full bg-[var(--accent)] px-2 py-0.5 text-[10px] font-bold">
          {String(data.total_items || 0)} items
        </span>
      </div>
      <div className="space-y-3 text-xs">
        {recipe && (
          <div>
            <span className="font-medium text-[var(--foreground)]">vLLM Recipe</span>
            <span className="ml-2 text-[var(--muted-foreground)]">
              {recipe.count as number} items • vLLM ≥ {(recipe.min_vllm_version as string) || "?"}
            </span>
            {(recipe.features as string[] | undefined)?.length ? (
              <div className="mt-1 flex flex-wrap gap-1">
                {(recipe.features as string[]).slice(0, 3).map((f, i) => (
                  <span key={i} className="rounded bg-[var(--muted)] px-1.5 py-0.5 text-[10px]">{f.split("'")[1] || f}</span>
                ))}
              </div>
            ) : null}
          </div>
        )}
        {evals && (
          <div>
            <span className="font-medium text-[var(--foreground)]">Benchmarks</span>
            <span className="ml-2 text-[var(--muted-foreground)]">
              {evals.accuracy_benchmarks as number} accuracy, {evals.performance_benchmarks as number} perf
            </span>
          </div>
        )}
        {compat && (
          <div>
            <div className="flex items-center gap-1">
              <ShieldCheck className={cn("h-3 w-3", compat.is_validated ? "text-green-500" : "text-amber-500")} />
              <span className="font-medium text-[var(--foreground)]">RHOAI</span>
              <span className="ml-1 text-[var(--muted-foreground)]">
                {compat.is_validated ? "Validated" : "Not in validated matrix"}
                {compat.matrix_version ? ` (v${compat.matrix_version as string})` : ""}
              </span>
            </div>
            {compat.is_validated && (compat.verified_accelerators as string[] | undefined)?.length ? (
              <div className="mt-1.5">
                {compat.matched_model ? (
                  <div className="mb-1 text-[10px] text-[var(--muted-foreground)]">
                    {String(compat.matched_model)}
                    {compat.min_vram_gb ? ` · min ${String(compat.min_vram_gb)} GB` : ""}
                  </div>
                ) : null}
                <div className="flex flex-wrap gap-1">
                  {(compat.verified_accelerators as string[]).slice(0, 8).map((gpu, i) => (
                    <span key={i} className="rounded bg-green-500/10 px-1.5 py-0.5 text-[10px] font-medium text-green-700">
                      {gpu}
                    </span>
                  ))}
                  {(compat.verified_accelerators as string[]).length > 8 && (
                    <span className="rounded bg-[var(--muted)] px-1.5 py-0.5 text-[10px] text-[var(--muted-foreground)]">
                      +{(compat.verified_accelerators as string[]).length - 8} more
                    </span>
                  )}
                </div>
              </div>
            ) : null}
          </div>
        )}
      </div>
    </div>
  );
}

function UseCaseChart({ data }: { data: Record<string, unknown> }) {
  const { t } = useI18n();
  const useCases = (data.use_case_presets as string[] | undefined) || [];
  if (useCases.length === 0) return null;

  const USE_CASE_WEIGHTS: Record<string, { weight: number; label: string; color: string }> = {
    chatbot: { weight: 35, label: "Chatbot", color: "#3b82f6" },
    rag: { weight: 30, label: "RAG", color: "#10b981" },
    coding: { weight: 20, label: "Coding", color: "#f59e0b" },
    summarization: { weight: 10, label: "Summarization", color: "#8b5cf6" },
    translation: { weight: 10, label: "Translation", color: "#ec4899" },
    classification: { weight: 5, label: "Classification", color: "#06b6d4" },
    extraction: { weight: 8, label: "Extraction", color: "#f97316" },
    agents: { weight: 25, label: "Agents", color: "#6366f1" },
    agentic: { weight: 25, label: "Agentic", color: "#6366f1" },
    batch: { weight: 15, label: "Batch", color: "#a855f7" },
  };

  const FALLBACK_COLORS = ["#64748b", "#0ea5e9", "#d946ef", "#84cc16", "#ef4444"];

  const items = useCases.map((uc, idx) => {
    const preset = USE_CASE_WEIGHTS[uc.toLowerCase()] || {
      weight: 15,
      label: uc,
      color: FALLBACK_COLORS[idx % FALLBACK_COLORS.length],
    };
    return { id: uc, ...preset };
  });

  const totalWeight = items.reduce((sum, item) => sum + item.weight, 0);
  const normalized = items.map((item) => ({
    ...item,
    pct: Math.round((item.weight / totalWeight) * 100),
  }));

  const USE_CASE_CHARACTERISTICS: Record<string, { latency: string; throughput: string; context: string }> = {
    chatbot: { latency: "Low TTFT", throughput: "Medium", context: "Short-Medium" },
    rag: { latency: "Medium TTFT", throughput: "High", context: "Long (retrieval)" },
    coding: { latency: "Low TPOT", throughput: "Medium", context: "Medium-Long" },
    summarization: { latency: "Flexible", throughput: "High", context: "Very Long input" },
    translation: { latency: "Low", throughput: "High", context: "Short" },
    classification: { latency: "Very Low", throughput: "Very High", context: "Short" },
    extraction: { latency: "Medium", throughput: "High", context: "Medium" },
    agents: { latency: "Low TPOT", throughput: "Medium", context: "Multi-turn" },
    agentic: { latency: "Low TPOT", throughput: "Medium", context: "Multi-turn" },
    batch: { latency: "Flexible", throughput: "Very High", context: "Variable" },
  };

  return (
    <div className="rounded-xl border border-[var(--border)] p-6">
      <div className="mb-4 flex items-center gap-2">
        <svg className="h-4 w-4 text-[var(--primary)]" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <path d="M21 12a9 9 0 01-9 9 9 9 0 01-9-9 9 9 0 019-9 9 9 0 019 9z" />
          <path d="M12 3v9l6 3" />
        </svg>
        <h3 className="text-sm font-semibold">{t("step5.useCaseDistribution")}</h3>
        <span className="ml-auto text-[10px] text-[var(--muted-foreground)]">
          Estimated request mix by workload type
        </span>
      </div>

      <div className="flex gap-6">
        {/* Horizontal stacked bar */}
        <div className="flex-1">
          <div className="mb-3 flex h-8 w-full overflow-hidden rounded-lg">
            {normalized.map((item, i) => (
              <div
                key={item.id}
                className="flex items-center justify-center text-[9px] font-bold text-white transition-all"
                style={{
                  width: `${item.pct}%`,
                  backgroundColor: item.color,
                  borderRight: i < normalized.length - 1 ? "2px solid var(--background)" : undefined,
                }}
              >
                {item.pct >= 15 ? `${item.pct}%` : ""}
              </div>
            ))}
          </div>

          {/* Legend + Characteristics */}
          <div className="space-y-2">
            {/* Column headers for characteristics */}
            <div className="flex items-center gap-3 mb-1">
              <div className="min-w-[100px]" />
              <span className="text-xs font-bold tabular-nums w-[30px]" />
              <div className="flex gap-2 text-[10px] text-[var(--muted-foreground)] font-medium">
                <span className="rounded px-1.5 py-0.5 w-[72px] text-center">Latency</span>
                <span className="rounded px-1.5 py-0.5 w-[72px] text-center">Throughput</span>
                <span className="rounded px-1.5 py-0.5 w-[90px] text-center">Context Length</span>
              </div>
            </div>
            {normalized.map((item) => {
              const chars = USE_CASE_CHARACTERISTICS[item.id.toLowerCase()];
              return (
                <div key={item.id} className="flex items-center gap-3">
                  <div className="flex items-center gap-2 min-w-[100px]">
                    <div className="h-2.5 w-2.5 rounded-sm" style={{ backgroundColor: item.color }} />
                    <span className="text-xs font-medium">{item.label}</span>
                  </div>
                  <span className="text-xs font-bold tabular-nums w-[30px]" style={{ color: item.color }}>
                    {item.pct}%
                  </span>
                  {chars && (
                    <div className="flex gap-2 text-[10px] text-[var(--muted-foreground)]">
                      <span className="rounded bg-[var(--muted)] px-1.5 py-0.5 w-[72px] text-center">{chars.latency}</span>
                      <span className="rounded bg-[var(--muted)] px-1.5 py-0.5 w-[72px] text-center">{chars.throughput}</span>
                      <span className="rounded bg-[var(--muted)] px-1.5 py-0.5 w-[90px] text-center">{chars.context}</span>
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      </div>

      {/* Insight */}
      <div className="mt-4 rounded-lg border border-dashed border-[var(--border)] bg-[var(--muted)]/10 px-4 py-3">
        <p className="text-[10px] text-[var(--muted-foreground)]">
          <span className="font-medium text-[var(--foreground)]">Optimization priority: </span>
          {normalized[0] && normalized[0].pct > 40
            ? `Heavily optimized for ${normalized[0].label} — consider dedicated vLLM config tuning for this pattern.`
            : `Mixed workload — enable chunked prefill and prefix caching for best multi-use-case performance.`}
        </p>
      </div>
    </div>
  );
}

function DeploymentCard({ data }: { data: Record<string, unknown> | undefined }) {
  const { t } = useI18n();
  if (!data) return null;
  return (
    <div className="rounded-xl border border-[var(--border)] p-5">
      <div className="mb-3 flex items-center gap-2">
        <Server className="h-4 w-4 text-[var(--primary)]" />
        <h3 className="text-sm font-semibold">{t("step5.deployment")}</h3>
      </div>
      <div className="space-y-2 text-xs">
        <Row label="GPU" value={`${data.gpu_count}× ${data.gpu_type}`} />
        <Row label="RHOAI" value={`v${data.rhoai_version} (vLLM ${data.vllm_version})`} />
        <Row label="Use Case" value={(data.use_case_presets as string[] | undefined)?.join(", ") || "—"} />
        <Row label="Users" value={String(data.target_end_users || "—")} />
        <Row label="Concurrent" value={String(data.max_concurrent_requests || "—")} />
        <Row label="TTFT Target" value={`${data.ttft_target_ms}ms`} />
        <Row label="TPOT Target" value={`${data.tpot_target_ms}ms`} />
      </div>
    </div>
  );
}

function MemoryCard({ data }: { data: Record<string, unknown> | undefined }) {
  const { t } = useI18n();
  if (!data) return null;
  const fits = data.fits as boolean;
  const util = data.utilization_pct as number;

  return (
    <div className={cn("rounded-xl border p-5", fits ? "border-[var(--border)]" : "border-red-500/30")}>
      <div className="mb-3 flex items-center gap-2">
        <MemoryStick className="h-4 w-4 text-[var(--primary)]" />
        <h3 className="text-sm font-semibold">{t("step5.memory")}</h3>
        {fits ? (
          <CheckCircle2 className="ml-auto h-4 w-4 text-green-500" />
        ) : (
          <AlertTriangle className="ml-auto h-4 w-4 text-red-500" />
        )}
      </div>
      <div className="space-y-2 text-xs">
        <Row label="Model Weights" value={`${data.model_weights_gb} GB (${precisionLabel(data.precision_bits as number)})`} />
        <Row label="KV Cache" value={`${data.kv_cache_gb} GB`} />
        <Row label="Overhead" value={`${data.overhead_min_total_gb ?? data.overhead_gb ?? "~5–12"} GB`} />
        <div className="border-t border-dashed border-[var(--border)] pt-2">
          <Row label="Total Required" value={`${data.total_required_min_gb ?? data.total_required_gb ?? "—"} GB`} bold />
          <Row label="Available" value={`${data.total_available_gb} GB`} />
        </div>
        {/* Utilization bar */}
        <div className="pt-1">
          <div className="flex justify-between text-[10px]">
            <span>Utilization</span>
            <span className={cn("font-bold", util > 95 ? "text-red-500" : util > 80 ? "text-amber-500" : "text-green-500")}>
              {util}%
            </span>
          </div>
          <div className="mt-1 h-2 w-full overflow-hidden rounded-full bg-[var(--muted)]">
            <div
              className={cn("h-full rounded-full transition-all", util > 95 ? "bg-red-500" : util > 80 ? "bg-amber-500" : "bg-green-500")}
              style={{ width: `${Math.min(util, 100)}%` }}
            />
          </div>
        </div>
      </div>
    </div>
  );
}

function CostCard({ data }: { data: Record<string, unknown> | undefined }) {
  const { t } = useI18n();
  if (!data) return null;
  const isCloud = data.type === "cloud";

  return (
    <div className="rounded-xl border border-[var(--border)] p-5">
      <div className="mb-3 flex items-center gap-2">
        <DollarSign className="h-4 w-4 text-[var(--primary)]" />
        <h3 className="text-sm font-semibold">{t("step5.cost")}</h3>
        <span className="ml-auto rounded bg-[var(--muted)] px-1.5 py-0.5 text-[9px] font-medium">
          {isCloud ? String(data.platform).toUpperCase() : "On-Prem TCO"}
        </span>
      </div>

      {isCloud ? (
        <div className="space-y-2 text-xs">
          {data.instance ? <Row label="Instance" value={String(data.instance)} /> : null}
          <Row label="On-demand" value={`$${data.on_demand_hourly_usd}/hr`} />
          {Number(data.spot_hourly_usd) > 0 ? <Row label="Spot" value={`$${data.spot_hourly_usd}/hr`} /> : null}
          {Number(data.reserved_1yr_hourly_usd) > 0 ? <Row label="Reserved (1yr)" value={`$${data.reserved_1yr_hourly_usd}/hr`} /> : null}
          <div className="border-t border-dashed border-[var(--border)] pt-2">
            <div className="flex items-baseline justify-between">
              <span className="text-[var(--muted-foreground)]">Monthly (on-demand)</span>
              <span className="text-lg font-bold text-[var(--primary)]">
                ${Number(data.monthly_on_demand_usd).toLocaleString()}
              </span>
            </div>
            {Number(data.monthly_spot_usd) > 0 ? (
              <div className="mt-1 flex items-baseline justify-between text-[10px]">
                <span className="text-[var(--muted-foreground)]">Monthly (spot)</span>
                <span className="font-medium text-green-600">${Number(data.monthly_spot_usd).toLocaleString()}</span>
              </div>
            ) : null}
          </div>
          {data.source_url ? (
            <a href={String(data.source_url)} target="_blank" rel="noreferrer"
              className="mt-2 flex items-center gap-1 text-[10px] text-[var(--primary)] hover:underline">
              <ExternalLink className="h-3 w-3" /> Pricing page
            </a>
          ) : null}
        </div>
      ) : (
        <div className="space-y-2 text-xs">
          <Row label="Hardware (3yr dep.)" value={`$${Number(data.monthly_hardware_usd || 0).toLocaleString()}/mo`} />
          <Row label="Power & Cooling" value={`$${Number(data.monthly_power_usd || 0).toLocaleString()}/mo`} />
          <Row label="Colocation" value={`$${Number(data.monthly_colocation_usd || 0).toLocaleString()}/mo`} />
          <Row label="Staffing (1 FTE)" value={`$${Number(data.monthly_staffing_usd || 0).toLocaleString()}/mo`} />
          <Row label="RH AI Inference" value={`$${Number(data.monthly_rh_subscription_usd || 0).toLocaleString()}/mo`} />
          <div className="border-t border-dashed border-[var(--border)] pt-2">
            <div className="flex items-baseline justify-between">
              <span className="text-[var(--muted-foreground)]">Monthly TCO</span>
              <span className="text-lg font-bold text-[var(--primary)]">
                ${Number(data.monthly_total_usd).toLocaleString()}
              </span>
            </div>
            <p className="mt-1 text-[10px] text-[var(--muted-foreground)]">
              {String(data.gpu_count)}× {String(data.gpu_type)}
            </p>
          </div>
        </div>
      )}
    </div>
  );
}

function PerformanceForecastChart({ data }: { data: Record<string, unknown> }) {
  const { t } = useI18n();
  const chartData = (data.chart_data as { batch_size: number; throughput_tokens_per_sec: number; latency_per_token_ms: number }[]) || [];
  const maxThroughput = Math.max(...chartData.map((d) => d.throughput_tokens_per_sec), 1);
  const explanation = data.explanation as Record<string, string> | undefined;

  return (
    <div className="rounded-xl border border-[var(--border)] p-6">
      <div className="mb-4 flex items-center gap-2">
        <TrendingUp className="h-4 w-4 text-[var(--primary)]" />
        <h3 className="text-sm font-semibold">{t("step5.performanceForecast")}</h3>
        <span className="ml-auto text-[10px] text-[var(--muted-foreground)]">
          Theoretical roofline model (memory-bandwidth bound)
        </span>
      </div>

      {/* Key metrics */}
      <div className="mb-5 grid grid-cols-3 gap-4">
        <MetricBox label="Decode Throughput" value={`${Number(data.theoretical_decode_tps).toLocaleString()} tok/s`} />
        <MetricBox label="Est. TPOT" value={`${data.estimated_tpot_ms}ms`} />
        <MetricBox label="Est. TTFT" value={`${data.estimated_ttft_ms}ms`} />
      </div>

      {/* Bar chart */}
      <div className="rounded-lg border border-[var(--border)] bg-[var(--muted)]/20 p-4">
        <div className="mb-2 flex items-center justify-between text-[10px] text-[var(--muted-foreground)]">
          <span>Throughput vs Batch Size</span>
          <span>tokens/sec</span>
        </div>
        <div className="flex items-end gap-2" style={{ height: 140 }}>
          {chartData.map((point) => {
            const barHeight = Math.max((point.throughput_tokens_per_sec / maxThroughput) * 100, 4);
            return (
              <div key={point.batch_size} className="flex h-full flex-1 flex-col items-center justify-end gap-1">
                <span className="text-[9px] font-medium tabular-nums">
                  {point.throughput_tokens_per_sec > 1000
                    ? `${(point.throughput_tokens_per_sec / 1000).toFixed(1)}k`
                    : Math.round(point.throughput_tokens_per_sec)}
                </span>
                <div
                  className="w-full rounded-t bg-[var(--primary)] transition-all"
                  style={{ height: `${barHeight}px` }}
                />
                <span className="text-[9px] text-[var(--muted-foreground)]">
                  {point.batch_size}
                </span>
              </div>
            );
          })}
        </div>
        <div className="mt-2 text-center text-[10px] text-[var(--muted-foreground)]">
          Batch Size
        </div>
      </div>

      {/* Calculation Methodology */}
      <div className="mt-4 rounded-lg border border-dashed border-[var(--border)] bg-[var(--muted)]/10 p-4">
        <h4 className="mb-2 text-[11px] font-semibold text-[var(--muted-foreground)] uppercase tracking-wide">
          Calculation Method
        </h4>

        {explanation && (
          <div className="mb-3 grid grid-cols-2 gap-x-6 gap-y-1 text-[10px] sm:grid-cols-3">
            <div><span className="text-[var(--muted-foreground)]">Model:</span> <span className="font-medium">{String(explanation.model)}</span></div>
            <div><span className="text-[var(--muted-foreground)]">Hardware:</span> <span className="font-medium">{String(explanation.hardware)}</span></div>
            <div><span className="text-[var(--muted-foreground)]">Bandwidth:</span> <span className="font-medium">{String(explanation.bandwidth)}</span></div>
            <div><span className="text-[var(--muted-foreground)]">Compute:</span> <span className="font-medium">{String(explanation.compute)}</span></div>
            <div className="sm:col-span-2"><span className="text-[var(--muted-foreground)]">Ridge Point:</span> <span className="font-medium">Batch {String(data.ridge_batch_size)}</span></div>
          </div>
        )}

        <div className="space-y-2 text-[10px] text-[var(--muted-foreground)]">
          <div>
            <span className="font-medium text-[var(--foreground)]">Decode (TPOT)</span> — Memory-bandwidth bound
            <div className="mt-0.5 rounded bg-[var(--muted)]/50 px-2 py-1 font-mono text-[9px]">
              throughput(batch=1) = HBM_bandwidth / model_weight_size = {String(data.theoretical_decode_tps)} tok/s
            </div>
          </div>
          <div>
            <span className="font-medium text-[var(--foreground)]">Batched Decode</span> — Roofline
            <div className="mt-0.5 rounded bg-[var(--muted)]/50 px-2 py-1 font-mono text-[9px]">
              throughput(batch=B) = min(B × {String(data.theoretical_decode_tps)}, {String(data.compute_ceiling_tps)} TFLOPS ceiling)
            </div>
          </div>
          <div>
            <span className="font-medium text-[var(--foreground)]">Prefill (TTFT)</span> — Compute-bound
            <div className="mt-0.5 rounded bg-[var(--muted)]/50 px-2 py-1 font-mono text-[9px]">
              TTFT = (2 × params × input_tokens) / total_FLOPS = {String(data.estimated_ttft_ms)}ms (@ 512 tokens)
            </div>
          </div>
        </div>

        {explanation && (
          <div className="mt-3 flex gap-4 text-[9px]">
            <span className="rounded bg-blue-500/10 px-2 py-0.5 text-blue-600">
              {String(explanation.memory_bound)}
            </span>
            <span className="rounded bg-amber-500/10 px-2 py-0.5 text-amber-600">
              {String(explanation.compute_bound)}
            </span>
          </div>
        )}

        <p className="mt-3 text-[9px] italic text-[var(--muted-foreground)]">
          * Theoretical upper bounds. Actual throughput is typically 50–70% of roofline due to
          KV cache memory pressure, attention overhead, scheduling latency, and tensor parallelism communication cost.
        </p>
      </div>
    </div>
  );
}

function DesignSuggestionCard({ data }: { data: Record<string, unknown> }) {
  const { t } = useI18n();
  const content = data.content as string | undefined;
  const source = data.source as string | undefined;
  const modelUsed = data.model_used as string | undefined;

  if (!content) return null;

  const sections = parseMarkdownSections(content);

  return (
    <div className="rounded-xl border border-[var(--primary)]/20 bg-gradient-to-br from-[var(--primary)]/5 to-transparent p-6">
      <div className="mb-4 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Lightbulb className="h-4 w-4 text-[var(--primary)]" />
          <h3 className="text-sm font-semibold">{t("step5.designSuggestion")}</h3>
        </div>
        <span className="rounded bg-[var(--muted)] px-1.5 py-0.5 text-[9px] font-medium text-[var(--muted-foreground)]">
          {source === "llm" ? `AI · ${modelUsed}` : "Deterministic"}
        </span>
      </div>

      <div className="space-y-4">
        {sections.map((section, i) => (
          <div key={i}>
            {section.heading && (
              <h4 className="mb-1.5 text-xs font-semibold text-[var(--foreground)]">
                {section.heading}
              </h4>
            )}
            <div className="space-y-1 text-xs leading-relaxed text-[var(--muted-foreground)]">
              {section.lines.map((line, j) => {
                if (line.startsWith("- ") || line.startsWith("• ")) {
                  return (
                    <div key={j} className="flex gap-2">
                      <span className="mt-0.5 shrink-0 text-[var(--primary)]">•</span>
                      <span>{line.replace(/^[-•]\s*/, "")}</span>
                    </div>
                  );
                }
                return line.trim() ? <p key={j}>{line}</p> : null;
              })}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function parseMarkdownSections(text: string): Array<{ heading: string | null; lines: string[] }> {
  const lines = text.split("\n");
  const sections: Array<{ heading: string | null; lines: string[] }> = [];
  let current: { heading: string | null; lines: string[] } = { heading: null, lines: [] };

  for (const line of lines) {
    const headingMatch = line.match(/^#{1,4}\s+(.+)/);
    if (headingMatch) {
      if (current.heading || current.lines.length > 0) {
        sections.push(current);
      }
      current = { heading: headingMatch[1], lines: [] };
    } else {
      current.lines.push(line);
    }
  }
  if (current.heading || current.lines.length > 0) {
    sections.push(current);
  }
  return sections;
}

// --- Export ---

function buildExportPrompt({
  modelSummary,
  memoryEstimate,
  deploymentConfig,
  costEstimate,
  perfForecast,
  evidenceSummary,
  designSuggestion,
}: {
  modelSummary?: Record<string, unknown>;
  memoryEstimate?: Record<string, unknown>;
  deploymentConfig?: Record<string, unknown>;
  costEstimate?: Record<string, unknown>;
  perfForecast?: Record<string, unknown>;
  evidenceSummary?: Record<string, unknown>;
  designSuggestion?: Record<string, unknown>;
}): string {
  const v = (val: unknown, fallback = "?") => (val != null && val !== "" ? String(val) : fallback);
  const mem = memoryEstimate || {};
  const dep = deploymentConfig || {};
  const perf = perfForecast || {};
  const cost = costEstimate || {};
  const model = modelSummary || {};
  const evidence = evidenceSummary || {};

  const precLabel = precisionLabel(mem.precision_bits as number | undefined);
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
- Overhead: ${v(mem.overhead_min_total_gb ?? mem.overhead_gb)} GB
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
1-2 bullet points suggesting alternative configurations if requirements change (e.g., different GPU count, quantization, or model variant).

IMPORTANT: If GPU memory utilization is below 50%, explicitly recommend:
- Using NVIDIA MIG (Multi-Instance GPU) to partition the GPU and co-locate multiple models or serving replicas on a single device (applicable to A100, H100, H200, B200)
- Deploying a larger or higher-precision variant of the model to better utilize the available VRAM (e.g., FP16 instead of FP8, or a bigger model in the same family)`;

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
    overhead_gb: v(mem.overhead_min_total_gb ?? mem.overhead_gb),
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

  const suggestion = designSuggestion || {};
  const goldenRecord = {
    trace_id: "",
    session_id: "",
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

// --- Helpers ---

function precisionLabel(bits: number | undefined): string {
  if (!bits) return "FP16";
  const map: Record<number, string> = { 4: "INT4", 8: "FP8", 16: "FP16", 32: "FP32" };
  return map[bits] ?? `${bits}-bit`;
}

function Row({ label, value, bold }: { label: string; value: string | number | null | undefined; bold?: boolean }) {
  return (
    <div className="flex items-center justify-between">
      <span className="text-[var(--muted-foreground)]">{label}</span>
      <span className={cn("text-right", bold ? "font-bold" : "font-medium")}>{String(value ?? "—")}</span>
    </div>
  );
}

function MetricBox({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg border border-[var(--border)] bg-[var(--background)] p-3 text-center">
      <div className="text-[10px] text-[var(--muted-foreground)]">{label}</div>
      <div className="mt-0.5 text-sm font-bold text-[var(--primary)]">{value}</div>
    </div>
  );
}

function SkeletonPanels() {
  return (
    <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
      {[1, 2, 3, 4, 5, 6].map((i) => (
        <div key={i} className="rounded-xl border border-[var(--border)] p-5">
          <div className="mb-3 h-4 w-2/3 animate-pulse rounded bg-[var(--muted)]" />
          <div className="space-y-2">
            <div className="h-3 w-full animate-pulse rounded bg-[var(--muted)]" />
            <div className="h-3 w-5/6 animate-pulse rounded bg-[var(--muted)]" />
            <div className="h-3 w-4/6 animate-pulse rounded bg-[var(--muted)]" />
          </div>
        </div>
      ))}
    </div>
  );
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function extractViewModel(agentState: AgentState): any | null {
  return agentState.viewModel || null;
}
