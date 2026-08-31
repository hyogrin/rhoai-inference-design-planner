"use client";

import { useState, useMemo } from "react";
import {
  ShieldCheck,
  CircleCheck,
  CircleX,
  CircleMinus,
  Lightbulb,
  Loader2,
  Calculator,
  Sparkles,
} from "lucide-react";
import { useI18n } from "@/lib/i18n";
import type { AgentState } from "@/lib/use-agent-stream";

function StatusIcon({ status }: { status: "pass" | "fail" | "warn" }) {
  switch (status) {
    case "pass":
      return <CircleCheck className="h-4 w-4 text-[var(--success)]" />;
    case "fail":
      return <CircleX className="h-4 w-4 text-[var(--destructive)]" />;
    case "warn":
      return <CircleMinus className="h-4 w-4 text-[var(--warning)]" />;
  }
}

const GPU_MEMORY_GB: Record<string, number> = {
  "B200-192GB": 192,
  "H200-141GB": 141,
  "H100-80GB": 80,
  "MI300X-192GB": 192,
  "A100-80GB": 80,
  "A100-40GB": 40,
  "L40S-48GB": 48,
  "A10G-24GB": 24,
  "L4-24GB": 24,
  "T4-16GB": 16,
};

interface ReadinessGateStepProps {
  agentState: AgentState;
  selectedGpu: string | null;
  gpuCount: number;
  modelRepoId?: string;
}

// --- Model Analysis (from LLM, editable by user) ---
interface ModelAnalysis {
  weight_precision: string;
  effective_bits: number;
  kv_layout: "standard_gqa" | "mla" | "hybrid_sliding";
  kv_cache_bytes_per_element: number;
  explanation: string;
}

// --- Architecture raw values (from HF connector, for deterministic calc) ---
interface ArchSummary {
  parameters?: number;
  max_context?: number;
  num_hidden_layers?: number;
  hidden_size?: number;
  num_attention_heads?: number;
  num_key_value_heads?: number;
  head_dim?: number;
  sliding_window?: number;
  sliding_attention_layers?: number;
  full_attention_layers?: number;
  global_head_dim?: number;
  num_global_kv_heads?: number;
  kv_lora_rank?: number;
  qk_rope_head_dim?: number;
  checkpoint_size_bytes?: number;
}

interface MemoryEstimate {
  modelWeightsGb: number;
  weightSource: "checkpoint" | "param_count";
  kvCacheGb: number;
  totalRequiredMin: number;
  totalRequiredMax: number;
  totalAvailableGb: number;
  utilizationPct: number;
  fits: boolean;
  concurrencyLow: number;
  concurrencyHigh: number;
  maxConcurrent: number;
  numLayers: number;
  numKvHeads: number;
  headDim: number;
  kvLoraRank?: number;
  qkRopeHeadDim?: number;
  slidingLayers?: number;
  fullLayers?: number;
  slidingWindow?: number;
  globalHeadDim?: number;
  numGlobalKvHeads?: number;
}

/**
 * Deterministic memory calculation using user-confirmed model analysis + raw arch values.
 * NO precision detection logic — uses values directly.
 */
function calculateMemory(
  analysis: ModelAnalysis,
  arch: ArchSummary,
  gpuId: string,
  gpuCount: number,
): MemoryEstimate | null {
  const perGpuGb = GPU_MEMORY_GB[gpuId];
  if (!perGpuGb) return null;

  const paramCount = arch.parameters;

  // Weight memory: checkpoint preferred, else param_count × effective_bits
  let modelWeightsGb: number;
  let weightSource: "checkpoint" | "param_count";
  if (arch.checkpoint_size_bytes) {
    modelWeightsGb = arch.checkpoint_size_bytes / (1024 ** 3);
    weightSource = "checkpoint";
  } else if (paramCount) {
    modelWeightsGb = (paramCount * analysis.effective_bits / 8) / (1024 ** 3);
    weightSource = "param_count";
  } else {
    return null;
  }

  const numLayers = arch.num_hidden_layers ?? 32;
  const hiddenSize = arch.hidden_size ?? 4096;
  const numKvHeads = arch.num_key_value_heads ?? 8;
  const numAttentionHeads = arch.num_attention_heads ?? 32;
  const headDim = arch.head_dim ?? Math.floor(hiddenSize / Math.max(numAttentionHeads, 1));
  const seqLen = 4096;
  const maxConcurrent = 32;

  const kvBytes = analysis.kv_cache_bytes_per_element;
  const kvLoraRank = arch.kv_lora_rank;
  const qkRopeHeadDim = arch.qk_rope_head_dim;
  const slidingLayers = arch.sliding_attention_layers;
  const fullLayers = arch.full_attention_layers;
  const slidingWindow = arch.sliding_window;
  const globalHeadDim = arch.global_head_dim;
  const numGlobalKvHeads = arch.num_global_kv_heads;

  // KV cache per request (bytes) based on layout
  let kvPerRequestBytes: number;
  let kvReplicatedAcrossTp = false;

  if (analysis.kv_layout === "mla" && kvLoraRank && qkRopeHeadDim) {
    const elementsPerTokenPerLayer = kvLoraRank + qkRopeHeadDim;
    kvPerRequestBytes = numLayers * elementsPerTokenPerLayer * kvBytes * seqLen;
    kvReplicatedAcrossTp = true;
  } else if (analysis.kv_layout === "hybrid_sliding" && slidingLayers && fullLayers && slidingWindow) {
    const slidingTokens = Math.min(seqLen, slidingWindow);
    const slidingKv = 2 * slidingLayers * numKvHeads * headDim * kvBytes * slidingTokens;
    const fHeadDim = globalHeadDim ?? headDim;
    const fKvHeads = numGlobalKvHeads ?? numKvHeads;
    const fullKv = 2 * fullLayers * fKvHeads * fHeadDim * kvBytes * seqLen;
    kvPerRequestBytes = slidingKv + fullKv;
  } else {
    kvPerRequestBytes = 2 * numLayers * numKvHeads * headDim * kvBytes * seqLen;
  }

  // Per-GPU budget
  const overheadMinPerGpu = 2.5;
  const overheadMaxPerGpu = 6.0;
  const totalAvailableGb = perGpuGb * gpuCount;
  const perGpuBudget = perGpuGb * 0.90;
  const modelMemPerGpu = modelWeightsGb / Math.max(gpuCount, 1);

  const kvPerRequestPerGpu = kvReplicatedAcrossTp
    ? kvPerRequestBytes
    : kvPerRequestBytes / Math.max(gpuCount, 1);

  const availableOptimistic = perGpuBudget - modelMemPerGpu - overheadMinPerGpu;
  const availableConservative = perGpuBudget - modelMemPerGpu - overheadMaxPerGpu;

  const concurrencyHigh = availableOptimistic > 0
    ? Math.max(1, Math.floor((availableOptimistic * (1024 ** 3)) / kvPerRequestPerGpu))
    : 0;
  const concurrencyLow = availableConservative > 0
    ? Math.max(1, Math.floor((availableConservative * (1024 ** 3)) / kvPerRequestPerGpu))
    : 0;

  const effectiveConcurrent = Math.min(maxConcurrent, concurrencyHigh);
  const kvCacheGb = kvReplicatedAcrossTp
    ? (effectiveConcurrent * kvPerRequestBytes * gpuCount) / (1024 ** 3)
    : (effectiveConcurrent * kvPerRequestBytes) / (1024 ** 3);

  const overheadMinTotal = overheadMinPerGpu * gpuCount;
  const overheadMaxTotal = overheadMaxPerGpu * gpuCount;
  const totalRequiredMin = modelWeightsGb + kvCacheGb + overheadMinTotal;
  const totalRequiredMax = modelWeightsGb + kvCacheGb + overheadMaxTotal;
  const utilizationPct = (totalRequiredMin / totalAvailableGb) * 100;
  const fits = (modelMemPerGpu + overheadMaxPerGpu) <= perGpuGb * 0.95 && concurrencyLow >= 1;

  return {
    modelWeightsGb, weightSource, kvCacheGb,
    totalRequiredMin, totalRequiredMax, totalAvailableGb,
    utilizationPct, fits, concurrencyLow, concurrencyHigh, maxConcurrent,
    numLayers, numKvHeads, headDim,
    kvLoraRank, qkRopeHeadDim,
    slidingLayers, fullLayers, slidingWindow, globalHeadDim, numGlobalKvHeads,
  };
}

// --- Model Analysis Editor (editable pre-filled fields) ---
function ModelAnalysisEditor({
  analysis,
  onChange,
}: {
  analysis: ModelAnalysis;
  onChange: (updated: ModelAnalysis) => void;
}) {
  const { t } = useI18n();
  return (
    <div className="rounded-xl border border-[var(--primary)]/20 bg-[var(--primary)]/5 p-5">
      <div className="mb-4 flex items-center gap-2">
        <Sparkles className="h-4 w-4 text-[var(--primary)]" />
        <h3 className="text-sm font-semibold">{t("step3.modelAnalysis")}</h3>
        <span className="ml-auto rounded bg-[var(--muted)] px-2 py-0.5 text-[10px] text-[var(--muted-foreground)]">
          editable
        </span>
      </div>

      <div className="grid grid-cols-2 gap-4 text-sm">
        <div>
          <label className="mb-1 block text-xs text-[var(--muted-foreground)]">Weight Precision</label>
          <input
            type="text"
            value={analysis.weight_precision}
            onChange={(e) => onChange({ ...analysis, weight_precision: e.target.value })}
            className="w-full rounded border border-[var(--border)] bg-[var(--background)] px-3 py-1.5 text-sm"
          />
        </div>
        <div>
          <label className="mb-1 block text-xs text-[var(--muted-foreground)]">Effective Bits</label>
          <select
            value={analysis.effective_bits}
            onChange={(e) => onChange({ ...analysis, effective_bits: Number(e.target.value) })}
            className="w-full rounded border border-[var(--border)] bg-[var(--background)] px-3 py-1.5 text-sm"
          >
            <option value={4}>4-bit</option>
            <option value={8}>8-bit</option>
            <option value={16}>16-bit</option>
          </select>
        </div>
        <div>
          <label className="mb-1 block text-xs text-[var(--muted-foreground)]">KV Layout</label>
          <select
            value={analysis.kv_layout}
            onChange={(e) => onChange({ ...analysis, kv_layout: e.target.value as ModelAnalysis["kv_layout"] })}
            className="w-full rounded border border-[var(--border)] bg-[var(--background)] px-3 py-1.5 text-sm"
          >
            <option value="standard_gqa">Standard GQA/MHA</option>
            <option value="mla">MLA (Multi-head Latent Attention)</option>
            <option value="hybrid_sliding">Hybrid (Sliding + Full)</option>
          </select>
        </div>
        <div>
          <label className="mb-1 block text-xs text-[var(--muted-foreground)]">KV Cache dtype</label>
          <select
            value={analysis.kv_cache_bytes_per_element}
            onChange={(e) => onChange({ ...analysis, kv_cache_bytes_per_element: Number(e.target.value) })}
            className="w-full rounded border border-[var(--border)] bg-[var(--background)] px-3 py-1.5 text-sm"
          >
            <option value={1}>FP8 (1 byte)</option>
            <option value={2}>BF16 (2 bytes)</option>
          </select>
        </div>
      </div>

      {analysis.explanation && (
        <div className="mt-3 rounded bg-[var(--muted)]/50 px-3 py-2 text-[11px] text-[var(--muted-foreground)]">
          {analysis.explanation}
        </div>
      )}
    </div>
  );
}

// --- Memory Breakdown Display ---
function MemoryBreakdown({
  estimate,
  analysis,
  arch,
  gpuCount,
  gpuId,
  modelRepoId,
}: {
  estimate: MemoryEstimate;
  analysis: ModelAnalysis;
  arch: ArchSummary;
  gpuCount: number;
  gpuId: string;
  modelRepoId?: string;
}) {
  const { t } = useI18n();
  const perGpuGb = GPU_MEMORY_GB[gpuId] || 0;

  return (
    <div className="rounded-xl border border-[var(--border)] p-6">
      <div className="mb-4 flex items-center gap-2">
        <Calculator className="h-5 w-5 text-[var(--primary)]" />
        <h3 className="font-medium">{t("step3.memoryEstimation")}</h3>
      </div>

      <div className="mb-4 rounded-lg bg-[var(--muted)]/50 p-4 font-mono text-xs">
        {modelRepoId && (
          <div className="mb-3 rounded bg-[var(--primary)]/5 border border-[var(--primary)]/20 px-3 py-2 text-[11px]">
            <span className="text-[var(--muted-foreground)]">Model: </span>
            <span className="font-semibold text-[var(--primary)]">{modelRepoId}</span>
          </div>
        )}

        {/* Weight Memory */}
        <div className="mb-3 text-[var(--muted-foreground)]">
          Model Weight Memory ({estimate.weightSource === "checkpoint" ? "Checkpoint" : analysis.weight_precision})
        </div>
        <div className="mb-1 flex items-center gap-2">
          {estimate.weightSource === "checkpoint" ? (
            <span className="text-[var(--foreground)]">Checkpoint-based ({analysis.weight_precision})</span>
          ) : (
            <span className="text-[var(--foreground)]">
              {((arch.parameters ?? 0) / 1e9).toFixed(1)}B params × {analysis.effective_bits / 8} bytes/param
            </span>
          )}
          <span className="text-[var(--muted-foreground)]">=</span>
          <span className="font-semibold text-[var(--primary)]">≈ {estimate.modelWeightsGb.toFixed(1)} GiB</span>
        </div>

        {/* KV Cache */}
        <div className="mb-3 mt-4 text-[var(--muted-foreground)]">
          KV Cache Working Set ({analysis.kv_cache_bytes_per_element === 1 ? "FP8" : "BF16"})
          {analysis.kv_layout === "mla" && (
            <span className="ml-1 rounded bg-purple-500/10 border border-purple-500/20 px-1.5 py-0.5 text-[10px] text-purple-700">MLA</span>
          )}
          {analysis.kv_layout === "hybrid_sliding" && (
            <span className="ml-1 rounded bg-blue-500/10 border border-blue-500/20 px-1.5 py-0.5 text-[10px] text-blue-700">Hybrid</span>
          )}
        </div>

        {analysis.kv_layout === "mla" && estimate.kvLoraRank && estimate.qkRopeHeadDim ? (
          <div className="space-y-1">
            <div className="text-[10px] text-[var(--muted-foreground)]">
              MLA latent (kv_lora_rank={estimate.kvLoraRank}, rope_dim={estimate.qkRopeHeadDim}):
            </div>
            <div className="flex items-center gap-2">
              <span className="text-[var(--foreground)]">
                {estimate.numLayers} × {estimate.kvLoraRank + estimate.qkRopeHeadDim} × {analysis.kv_cache_bytes_per_element}B × {estimate.concurrencyHigh} × 4096
              </span>
              <span className="text-[var(--muted-foreground)]">=</span>
              <span className="font-semibold text-[var(--primary)]">≈ {estimate.kvCacheGb.toFixed(1)} GiB</span>
            </div>
            <div className="text-[10px] text-[var(--muted-foreground)]">Replicated across TP GPUs (not sharded)</div>
          </div>
        ) : analysis.kv_layout === "hybrid_sliding" && estimate.slidingLayers && estimate.fullLayers ? (
          <div className="space-y-2">
            <div className="text-[10px] text-[var(--muted-foreground)]">
              Sliding ({estimate.slidingLayers} layers, window={estimate.slidingWindow}) + Full ({estimate.fullLayers} layers)
            </div>
            <div className="flex items-center gap-2">
              <span className="text-[var(--foreground)]">Total KV</span>
              <span className="text-[var(--muted-foreground)]">=</span>
              <span className="font-semibold text-[var(--primary)]">≈ {estimate.kvCacheGb.toFixed(1)} GiB</span>
            </div>
          </div>
        ) : (
          <div className="flex items-center gap-2">
            <span className="text-[var(--foreground)]">
              2 × {estimate.numLayers} × {estimate.numKvHeads} × {estimate.headDim} × {analysis.kv_cache_bytes_per_element}B × {estimate.concurrencyHigh} × 4096
            </span>
            <span className="text-[var(--muted-foreground)]">=</span>
            <span className="font-semibold text-[var(--primary)]">≈ {estimate.kvCacheGb.toFixed(1)} GiB</span>
          </div>
        )}

        {/* Overhead */}
        <div className="mb-3 mt-4 text-[var(--muted-foreground)]">Runtime Overhead (CUDA, activations)</div>
        <div className="mb-1">
          <span className="font-semibold text-[var(--primary)]">
            2.5–6 GiB/GPU × {gpuCount} = {(2.5 * gpuCount).toFixed(1)}–{(6 * gpuCount).toFixed(1)} GiB
          </span>
        </div>

        {/* Total + Concurrency */}
        <div className="mt-4 border-t border-[var(--border)] pt-3 space-y-2">
          <div className="flex items-center justify-between">
            <span className="font-semibold">Estimated Aggregate</span>
            <span className="font-bold">≈ {estimate.totalRequiredMin.toFixed(1)}–{estimate.totalRequiredMax.toFixed(1)} GiB</span>
          </div>
          <div className="flex items-center justify-between text-[11px]">
            <span className="text-[var(--muted-foreground)]">Memory-feasible Concurrency</span>
            <span className={`font-semibold ${estimate.concurrencyHigh >= estimate.maxConcurrent ? "text-[var(--success)]" : "text-amber-600"}`}>
              {estimate.concurrencyLow === estimate.concurrencyHigh
                ? `${estimate.concurrencyHigh} sequences`
                : `${estimate.concurrencyLow}–${estimate.concurrencyHigh} sequences`}
            </span>
          </div>
        </div>

        {estimate.concurrencyHigh < estimate.maxConcurrent && (
          <div className="mt-3 rounded bg-amber-500/10 border border-amber-500/20 px-3 py-2 text-[11px] text-amber-700">
            Requested {estimate.maxConcurrent} concurrent — consider more GPUs, shorter seq_len, or lower concurrency.
          </div>
        )}
      </div>

      {/* GPU Capacity Bar */}
      <div className="rounded-lg border border-[var(--border)] p-4">
        <div className="mb-2 flex items-center gap-2 font-mono text-xs">
          <span>{gpuCount} GPU{gpuCount > 1 ? "s" : ""} × {perGpuGb} GiB</span>
          <span className="text-[var(--muted-foreground)]">=</span>
          <span className="font-semibold text-[var(--primary)]">{estimate.totalAvailableGb} GiB</span>
        </div>
        <div className="relative h-6 w-full overflow-hidden rounded-md bg-[var(--muted)]">
          <div
            className={`h-full rounded-md transition-all duration-500 ${
              estimate.fits ? "bg-[var(--success)]" : estimate.utilizationPct <= 110 ? "bg-[var(--warning)]" : "bg-[var(--destructive)]"
            }`}
            style={{ width: `${Math.min(estimate.utilizationPct, 100)}%` }}
          />
          <div className="absolute inset-0 flex items-center justify-center">
            <span className="text-[11px] font-bold text-white drop-shadow-sm">{estimate.utilizationPct.toFixed(0)}%</span>
          </div>
        </div>
        <div className="mt-2 flex items-center gap-2">
          {estimate.fits ? <CircleCheck className="h-3.5 w-3.5 text-[var(--success)]" /> : <CircleX className="h-3.5 w-3.5 text-[var(--destructive)]" />}
          <span className={`text-xs font-medium ${estimate.fits ? "text-[var(--success)]" : "text-[var(--destructive)]"}`}>
            {estimate.fits
              ? `Fits (${(estimate.totalAvailableGb - estimate.totalRequiredMin).toFixed(1)} GiB headroom)`
              : `Exceeds by ${(estimate.totalRequiredMin - estimate.totalAvailableGb).toFixed(1)} GiB`}
          </span>
        </div>
      </div>
    </div>
  );
}

export function ReadinessGateStep({ agentState, selectedGpu, gpuCount, modelRepoId }: ReadinessGateStepProps) {
  const { t } = useI18n();
  const { status, nodeStatuses, interrupt } = agentState;
  const validationDone = nodeStatuses["validate_discovery"] === "done";
  const modelAnalysisDone = nodeStatuses["interpret_model_config"] === "done";
  const isRunning = status === "running" && !validationDone;

  // LLM-generated model analysis (editable)
  const rawAnalysis = interrupt?.model_analysis as ModelAnalysis | undefined;
  const defaultAnalysis: ModelAnalysis = { weight_precision: "BF16", effective_bits: 16, kv_layout: "standard_gqa", kv_cache_bytes_per_element: 2, explanation: "" };
  const [analysis, setAnalysis] = useState<ModelAnalysis>(rawAnalysis || defaultAnalysis);
  const [synced, setSynced] = useState(false);

  // Sync once when LLM analysis arrives
  if (rawAnalysis && !synced) {
    setAnalysis(rawAnalysis);
    setSynced(true);
  }

  const archSummary = interrupt?.architecture_summary as ArchSummary | undefined;

  // Deterministic calculation using confirmed analysis + raw arch
  const memoryEstimate = useMemo(() => {
    if (!archSummary || !selectedGpu) return null;
    return calculateMemory(analysis, archSummary, selectedGpu, gpuCount);
  }, [analysis, archSummary, selectedGpu, gpuCount]);

  const checks: { label: string; status: "pass" | "fail" | "warn" }[] = validationDone
    ? [
        { label: "Model weights accessible", status: "pass" },
        { label: "Architecture supported by vLLM", status: "pass" },
        { label: "Evidence collected from multiple sources", status: "pass" },
        {
          label: `GPU memory sufficient for ${analysis.weight_precision} inference`,
          status: memoryEstimate
            ? memoryEstimate.fits ? "pass" : memoryEstimate.utilizationPct <= 110 ? "warn" : "fail"
            : "warn",
        },
      ]
    : [];

  const overallStatus = checks.some((c) => c.status === "fail")
    ? "blocked"
    : checks.some((c) => c.status === "warn")
      ? "limited"
      : validationDone ? "ready" : "pending";

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-lg font-semibold">{t("step3.title")}</h2>
        <p className="mt-1 text-sm text-[var(--muted-foreground)]">
          {t("step3.description")}
        </p>
      </div>

      {/* Status */}
      <div className="rounded-xl border border-[var(--border)] p-6">
        {isRunning ? (
          <div className="flex items-center gap-4">
            <div className="flex h-12 w-12 items-center justify-center rounded-full bg-[var(--muted)]">
              <Loader2 className="h-6 w-6 animate-spin text-[var(--primary)]" />
            </div>
            <div>
              <h3 className="text-lg font-semibold text-[var(--muted-foreground)]">{t("step3.validating")}</h3>
              <p className="text-sm text-[var(--muted-foreground)]">{t("step3.validatingDesc")}</p>
            </div>
          </div>
        ) : (
          <div className="flex items-center gap-4">
            <div className={`flex h-12 w-12 items-center justify-center rounded-full ${
              overallStatus === "ready" ? "bg-[var(--success)]/10"
                : overallStatus === "limited" ? "bg-[var(--warning)]/10"
                  : overallStatus === "blocked" ? "bg-[var(--destructive)]/10" : "bg-[var(--muted)]"
            }`}>
              <ShieldCheck className={`h-6 w-6 ${
                overallStatus === "ready" ? "text-[var(--success)]"
                  : overallStatus === "limited" ? "text-[var(--warning)]"
                    : overallStatus === "blocked" ? "text-[var(--destructive)]" : "text-[var(--muted-foreground)]"
              }`} />
            </div>
            <div>
              <h3 className={`text-lg font-semibold ${
                overallStatus === "ready" ? "text-[var(--success)]"
                  : overallStatus === "limited" ? "text-[var(--warning)]"
                    : overallStatus === "blocked" ? "text-[var(--destructive)]" : "text-[var(--muted-foreground)]"
              }`}>
                {overallStatus === "ready" ? t("step3.ready") : overallStatus === "limited" ? t("step3.limited") : overallStatus === "blocked" ? t("step3.blocked") : t("step3.awaitingValidation")}
              </h3>
              <p className="text-sm text-[var(--muted-foreground)]">
                {overallStatus === "ready" ? t("step3.readyDesc")
                  : overallStatus === "limited" ? t("step3.limitedDesc")
                    : overallStatus === "blocked" ? t("step3.blockedDesc")
                      : t("step3.awaitingValidationDesc")}
              </p>
            </div>
          </div>
        )}
      </div>

      {/* Validation Checks */}
      {checks.length > 0 && (
        <div className="rounded-xl border border-[var(--border)] p-6">
          <h3 className="mb-4 font-medium">{t("step3.validationChecks")}</h3>
          <div className="space-y-3">
            {checks.map((check) => (
              <div key={check.label} className="flex items-center gap-3 rounded-lg border border-[var(--border)] px-4 py-3">
                <StatusIcon status={check.status} />
                <span className="text-sm">{check.label}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Model Analysis (editable) */}
      {(modelAnalysisDone || rawAnalysis) && (
        <ModelAnalysisEditor analysis={analysis} onChange={setAnalysis} />
      )}

      {/* Memory Breakdown (recalculates when analysis changes) */}
      {memoryEstimate && archSummary && selectedGpu && (
        <MemoryBreakdown
          estimate={memoryEstimate}
          analysis={analysis}
          arch={archSummary}
          gpuCount={gpuCount}
          gpuId={selectedGpu}
          modelRepoId={modelRepoId}
        />
      )}

      {/* Recommendations when doesn't fit */}
      {memoryEstimate && !memoryEstimate.fits && selectedGpu && (
        <div className="rounded-xl border border-dashed border-[var(--destructive)]/30 bg-[var(--destructive)]/5 p-6">
          <div className="mb-3 flex items-center gap-2">
            <Lightbulb className="h-5 w-5 text-[var(--destructive)]" />
            <h3 className="font-medium text-[var(--destructive)]">{t("step3.recommendations")}</h3>
          </div>
          <ul className="space-y-2 text-sm text-[var(--muted-foreground)]">
            <li className="flex items-start gap-2">
              <span className="mt-1.5 block h-1.5 w-1.5 rounded-full bg-[var(--destructive)]" />
              Increase GPU count: need at least{" "}
              {Math.ceil(memoryEstimate.modelWeightsGb / ((GPU_MEMORY_GB[selectedGpu] || 80) * 0.90 - 6))} GPUs
            </li>
            {analysis.effective_bits === 16 && (
              <li className="flex items-start gap-2">
                <span className="mt-1.5 block h-1.5 w-1.5 rounded-full bg-[var(--destructive)]" />
                Use FP8 quantization to reduce weight memory by ~50%
              </li>
            )}
            <li className="flex items-start gap-2">
              <span className="mt-1.5 block h-1.5 w-1.5 rounded-full bg-[var(--destructive)]" />
              Consider higher-memory GPUs (H200 141GB, B200 192GB, MI300X 192GB)
            </li>
          </ul>
        </div>
      )}
    </div>
  );
}
