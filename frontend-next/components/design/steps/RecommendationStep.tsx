"use client";

import {
  Cpu,
  MemoryStick,
  DollarSign,
  TrendingUp,
  AlertTriangle,
  CheckCircle2,
  BookOpen,
  ShieldCheck,
  Download,
  Loader2,
  ExternalLink,
} from "lucide-react";
import { cn } from "@/lib/utils";
import type { AgentState } from "@/lib/use-agent-stream";

interface RecommendationStepProps {
  agentState: AgentState;
}

export function RecommendationStep({ agentState }: RecommendationStepProps) {
  const isRunning = agentState.status === "running";

  // Extract view_model from the last step or state
  const viewModel = extractViewModel(agentState);
  const rec: Array<{ id: string; title: string; data: Record<string, unknown> }> = viewModel?.sections || [];
  const verification = viewModel?.verification as { warnings: string[]; status: string } | undefined;

  const modelSummary = rec.find((s) => s.id === "model_summary")?.data;
  const evidenceSummary = rec.find((s) => s.id === "evidence_summary")?.data;
  const deploymentConfig = rec.find((s) => s.id === "deployment_config")?.data;
  const memoryEstimate = rec.find((s) => s.id === "memory_estimate")?.data;
  const costEstimate = rec.find((s) => s.id === "cost_estimate")?.data;
  const perfForecast = rec.find((s) => s.id === "performance_forecast")?.data;

  if (isRunning || !viewModel) {
    return (
      <div className="space-y-6">
        <div>
          <h2 className="text-lg font-semibold">Recommendation</h2>
          <p className="mt-1 text-sm text-[var(--muted-foreground)]">
            Generating evidence-backed deployment recommendation...
          </p>
        </div>
        <div className="flex items-center gap-3 rounded-xl border border-[var(--primary)]/20 bg-[var(--primary)]/5 p-6">
          <Loader2 className="h-5 w-5 animate-spin text-[var(--primary)]" />
          <span className="text-sm font-medium text-[var(--primary)]">
            Computing sizing, cost, and performance forecast...
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
          <h2 className="text-lg font-semibold">Deployment Recommendation</h2>
          <p className="mt-1 text-sm text-[var(--muted-foreground)]">
            Evidence-backed sizing for {String(modelSummary?.repo_id || "your model")}
          </p>
        </div>
        <button className="flex items-center gap-2 rounded-lg border border-[var(--border)] px-4 py-2.5 text-sm font-medium transition-colors hover:bg-[var(--muted)]">
          <Download className="h-4 w-4" />
          Export
        </button>
      </div>

      {/* Verification Warnings */}
      {verification && verification.warnings && verification.warnings.length > 0 && (
        <div className="rounded-xl border border-amber-500/30 bg-amber-500/5 p-4">
          <div className="mb-2 flex items-center gap-2 text-amber-600">
            <AlertTriangle className="h-4 w-4" />
            <span className="text-sm font-medium">Attention</span>
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

      {/* Section 3: Performance Forecast (Graph) */}
      {perfForecast && <PerformanceForecastChart data={perfForecast} />}
    </div>
  );
}

// --- Sub-components ---

function ModelOverviewCard({ data }: { data: Record<string, unknown> | undefined }) {
  if (!data) return null;
  return (
    <div className="rounded-xl border border-[var(--border)] p-5">
      <div className="mb-4 flex items-center gap-2">
        <Cpu className="h-4 w-4 text-[var(--primary)]" />
        <h3 className="text-sm font-semibold">Model Overview</h3>
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
  if (!data) return null;
  const recipe = data.vllm_recipe as Record<string, unknown> | undefined;
  const evals = data.evaluations as Record<string, unknown> | undefined;
  const compat = data.rhoai_compatibility as Record<string, unknown> | undefined;

  return (
    <div className="rounded-xl border border-[var(--border)] p-5">
      <div className="mb-4 flex items-center gap-2">
        <BookOpen className="h-4 w-4 text-[var(--primary)]" />
        <h3 className="text-sm font-semibold">Collected Evidence</h3>
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
          <div className="flex items-center gap-1">
            <ShieldCheck className={cn("h-3 w-3", compat.is_validated ? "text-green-500" : "text-amber-500")} />
            <span className="font-medium text-[var(--foreground)]">RHOAI</span>
            <span className="ml-1 text-[var(--muted-foreground)]">
              {compat.is_validated ? "Validated" : "Not in validated matrix"}
            </span>
          </div>
        )}
      </div>
    </div>
  );
}

function DeploymentCard({ data }: { data: Record<string, unknown> | undefined }) {
  if (!data) return null;
  return (
    <div className="rounded-xl border border-[var(--border)] p-5">
      <h3 className="mb-3 text-sm font-semibold">Deployment</h3>
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
  if (!data) return null;
  const fits = data.fits as boolean;
  const util = data.utilization_pct as number;

  return (
    <div className={cn("rounded-xl border p-5", fits ? "border-[var(--border)]" : "border-red-500/30")}>
      <div className="mb-3 flex items-center gap-2">
        <MemoryStick className="h-4 w-4 text-[var(--primary)]" />
        <h3 className="text-sm font-semibold">Memory</h3>
        {fits ? (
          <CheckCircle2 className="ml-auto h-4 w-4 text-green-500" />
        ) : (
          <AlertTriangle className="ml-auto h-4 w-4 text-red-500" />
        )}
      </div>
      <div className="space-y-2 text-xs">
        <Row label="Model Weights" value={`${data.model_weights_gb} GB (FP${data.precision_bits})`} />
        <Row label="KV Cache" value={`${data.kv_cache_gb} GB`} />
        <Row label="Overhead" value={`${data.overhead_gb} GB`} />
        <div className="border-t border-dashed border-[var(--border)] pt-2">
          <Row label="Total Required" value={`${data.total_required_gb} GB`} bold />
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
  if (!data) return null;
  const isCloud = data.type === "cloud";

  return (
    <div className="rounded-xl border border-[var(--border)] p-5">
      <div className="mb-3 flex items-center gap-2">
        <DollarSign className="h-4 w-4 text-[var(--primary)]" />
        <h3 className="text-sm font-semibold">Cost</h3>
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
          <Row label="GPU Unit Price" value={`$${Number(data.gpu_unit_price_usd).toLocaleString()}`} />
          <Row label="Hardware Total" value={`$${Number(data.hardware_total_usd).toLocaleString()}`} />
          <Row label="Depreciation" value={`${data.depreciation_years}yr → $${Number(data.monthly_depreciation_usd).toLocaleString()}/mo`} />
          <Row label="Power ({data.power_kw} kW)" value={`$${Number(data.monthly_power_usd).toLocaleString()}/mo`} />
          <div className="border-t border-dashed border-[var(--border)] pt-2">
            <div className="flex items-baseline justify-between">
              <span className="text-[var(--muted-foreground)]">Monthly TCO</span>
              <span className="text-lg font-bold text-[var(--primary)]">
                ${Number(data.monthly_total_usd).toLocaleString()}
              </span>
            </div>
            <p className="mt-1 text-[10px] text-[var(--muted-foreground)]">
              {String(data.gpu_count)}× {String(data.gpu_type)} (depreciation + power)
            </p>
          </div>
        </div>
      )}
    </div>
  );
}

function PerformanceForecastChart({ data }: { data: Record<string, unknown> }) {
  const chartData = (data.chart_data as { batch_size: number; throughput_tokens_per_sec: number; latency_per_token_ms: number }[]) || [];
  const maxThroughput = Math.max(...chartData.map((d) => d.throughput_tokens_per_sec), 1);
  const explanation = data.explanation as Record<string, string> | undefined;

  return (
    <div className="rounded-xl border border-[var(--border)] p-6">
      <div className="mb-4 flex items-center gap-2">
        <TrendingUp className="h-4 w-4 text-[var(--primary)]" />
        <h3 className="text-sm font-semibold">Performance Forecast</h3>
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
        <div className="flex items-end gap-2" style={{ height: 120 }}>
          {chartData.map((point) => {
            const height = (point.throughput_tokens_per_sec / maxThroughput) * 100;
            return (
              <div key={point.batch_size} className="flex flex-1 flex-col items-center gap-1">
                <span className="text-[9px] font-medium tabular-nums">
                  {point.throughput_tokens_per_sec > 1000
                    ? `${(point.throughput_tokens_per_sec / 1000).toFixed(1)}k`
                    : Math.round(point.throughput_tokens_per_sec)}
                </span>
                <div
                  className="w-full rounded-t bg-[var(--primary)] transition-all"
                  style={{ height: `${Math.max(height, 4)}%` }}
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

// --- Helpers ---

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
