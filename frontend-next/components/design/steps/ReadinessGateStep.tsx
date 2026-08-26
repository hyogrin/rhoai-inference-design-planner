"use client";

import {
  ShieldCheck,
  CircleCheck,
  CircleX,
  CircleMinus,
  Lightbulb,
  Loader2,
  Calculator,
} from "lucide-react";
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
}

interface MemoryEstimate {
  modelWeightsGb: number;
  kvCacheGb: number;
  overheadGb: number;
  totalRequiredGb: number;
  totalAvailableGb: number;
  utilizationPct: number;
  fits: boolean;
}

function estimateMemory(
  paramsBillions: number | null,
  gpuId: string | null,
  gpuCount: number
): MemoryEstimate | null {
  if (!paramsBillions || !gpuId) return null;

  const perGpuGb = GPU_MEMORY_GB[gpuId];
  if (!perGpuGb) return null;

  // FP16: 2 bytes per param
  const modelWeightsGb = (paramsBillions * 1e9 * 2) / (1024 ** 3);
  // KV cache estimate: ~10-15% of model weights for typical serving
  const kvCacheGb = modelWeightsGb * 0.15;
  // Runtime overhead: CUDA contexts, activations, etc.
  const overheadGb = 2.5;

  const totalRequiredGb = modelWeightsGb + kvCacheGb + overheadGb;
  const totalAvailableGb = perGpuGb * gpuCount;
  const utilizationPct = (totalRequiredGb / totalAvailableGb) * 100;
  const fits = totalRequiredGb <= totalAvailableGb * 0.95; // 5% safety margin

  return {
    modelWeightsGb,
    kvCacheGb,
    overheadGb,
    totalRequiredGb,
    totalAvailableGb,
    utilizationPct,
    fits,
  };
}

function MemoryBreakdown({ estimate, paramsBillions, gpuId, gpuCount }: {
  estimate: MemoryEstimate;
  paramsBillions: number;
  gpuId: string;
  gpuCount: number;
}) {
  const perGpuGb = GPU_MEMORY_GB[gpuId] || 0;

  return (
    <div className="rounded-xl border border-[var(--border)] p-6">
      <div className="mb-4 flex items-center gap-2">
        <Calculator className="h-5 w-5 text-[var(--primary)]" />
        <h3 className="font-medium">Memory Estimation Breakdown</h3>
      </div>

      {/* Formula */}
      <div className="mb-4 rounded-lg bg-[var(--muted)]/50 p-4 font-mono text-xs">
        <div className="mb-3 text-[var(--muted-foreground)]">
          Model Weight Memory (FP16)
        </div>
        <div className="mb-1 flex items-center gap-2">
          <span className="text-[var(--foreground)]">
            {paramsBillions}B params × 2 bytes/param
          </span>
          <span className="text-[var(--muted-foreground)]">=</span>
          <span className="font-semibold text-[var(--primary)]">
            {estimate.modelWeightsGb.toFixed(1)} GB
          </span>
        </div>

        <div className="mb-3 mt-4 text-[var(--muted-foreground)]">
          KV Cache (estimated at 15% of weights)
        </div>
        <div className="mb-1 flex items-center gap-2">
          <span className="text-[var(--foreground)]">
            {estimate.modelWeightsGb.toFixed(1)} GB × 0.15
          </span>
          <span className="text-[var(--muted-foreground)]">=</span>
          <span className="font-semibold text-[var(--primary)]">
            {estimate.kvCacheGb.toFixed(1)} GB
          </span>
        </div>

        <div className="mb-3 mt-4 text-[var(--muted-foreground)]">
          Runtime Overhead (CUDA, activations)
        </div>
        <div className="mb-1 flex items-center gap-2">
          <span className="font-semibold text-[var(--primary)]">
            ~{estimate.overheadGb.toFixed(1)} GB
          </span>
        </div>

        <div className="mt-4 border-t border-[var(--border)] pt-3">
          <div className="flex items-center justify-between">
            <span className="text-[var(--foreground)] font-semibold">Total Required</span>
            <span className="font-bold text-[var(--foreground)]">
              {estimate.totalRequiredGb.toFixed(1)} GB
            </span>
          </div>
        </div>
      </div>

      {/* GPU Capacity */}
      <div className="rounded-lg border border-[var(--border)] p-4">
        <div className="mb-3 text-xs font-medium text-[var(--muted-foreground)] uppercase tracking-wide">
          Available GPU Memory
        </div>
        <div className="mb-2 flex items-center gap-2 font-mono text-xs">
          <span>{gpuCount} GPU{gpuCount > 1 ? "s" : ""} × {perGpuGb} GB</span>
          <span className="text-[var(--muted-foreground)]">=</span>
          <span className="font-semibold text-[var(--primary)]">
            {estimate.totalAvailableGb} GB
          </span>
        </div>

        {/* Visual bar */}
        <div className="mt-3">
          <div className="mb-1 flex items-center justify-between text-[10px] text-[var(--muted-foreground)]">
            <span>0 GB</span>
            <span>{estimate.totalAvailableGb} GB</span>
          </div>
          <div className="relative h-6 w-full overflow-hidden rounded-md bg-[var(--muted)]">
            <div
              className={`h-full rounded-md transition-all duration-500 ${
                estimate.fits
                  ? "bg-[var(--success)]"
                  : estimate.utilizationPct <= 110
                    ? "bg-[var(--warning)]"
                    : "bg-[var(--destructive)]"
              }`}
              style={{ width: `${Math.min(estimate.utilizationPct, 100)}%` }}
            />
            {estimate.utilizationPct > 100 && (
              <div
                className="absolute right-0 top-0 h-full bg-[var(--destructive)]/30"
                style={{ width: `${Math.min(estimate.utilizationPct - 100, 30)}%` }}
              />
            )}
            <div className="absolute inset-0 flex items-center justify-center">
              <span className="text-[11px] font-bold text-white drop-shadow-sm">
                {estimate.utilizationPct.toFixed(0)}% utilization
              </span>
            </div>
          </div>

          <div className="mt-2 flex items-center gap-2">
            {estimate.fits ? (
              <CircleCheck className="h-3.5 w-3.5 text-[var(--success)]" />
            ) : (
              <CircleX className="h-3.5 w-3.5 text-[var(--destructive)]" />
            )}
            <span className={`text-xs font-medium ${
              estimate.fits ? "text-[var(--success)]" : "text-[var(--destructive)]"
            }`}>
              {estimate.fits
                ? `Fits within GPU memory (${(estimate.totalAvailableGb - estimate.totalRequiredGb).toFixed(1)} GB headroom)`
                : `Exceeds available memory by ${(estimate.totalRequiredGb - estimate.totalAvailableGb).toFixed(1)} GB`}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}

export function ReadinessGateStep({ agentState, selectedGpu, gpuCount }: ReadinessGateStepProps) {
  const { status, nodeStatuses, interrupt } = agentState;
  const validationDone = nodeStatuses["validate_discovery"] === "done";
  const isRunning = status === "running" && !validationDone;

  // Extract model parameters from interrupt data
  const paramsBillions = interrupt?.architecture_summary?.parameters
    ? interrupt.architecture_summary.parameters / 1e9
    : null;

  const memoryEstimate = estimateMemory(paramsBillions, selectedGpu, gpuCount);

  const checks: { label: string; status: "pass" | "fail" | "warn" }[] = validationDone
    ? [
        { label: "Model weights accessible", status: "pass" },
        { label: "Architecture supported by vLLM", status: "pass" },
        { label: "Evidence collected from multiple sources", status: "pass" },
        {
          label: "GPU memory sufficient for FP16 inference",
          status: memoryEstimate
            ? memoryEstimate.fits
              ? "pass"
              : memoryEstimate.utilizationPct <= 110
                ? "warn"
                : "fail"
            : "warn",
        },
      ]
    : [];

  const overallStatus = checks.some((c) => c.status === "fail")
    ? "blocked"
    : checks.some((c) => c.status === "warn")
      ? "limited"
      : validationDone
        ? "ready"
        : "pending";

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-lg font-semibold">Readiness Gate</h2>
        <p className="mt-1 text-sm text-[var(--muted-foreground)]">
          Validating deployment feasibility before generating recommendations.
        </p>
      </div>

      {/* Status Indicator */}
      <div className="rounded-xl border border-[var(--border)] p-6">
        {isRunning ? (
          <div className="flex items-center gap-4">
            <div className="flex h-12 w-12 items-center justify-center rounded-full bg-[var(--muted)]">
              <Loader2 className="h-6 w-6 animate-spin text-[var(--primary)]" />
            </div>
            <div>
              <h3 className="text-lg font-semibold text-[var(--muted-foreground)]">
                Validating...
              </h3>
              <p className="text-sm text-[var(--muted-foreground)]">
                Running readiness checks on collected evidence.
              </p>
            </div>
          </div>
        ) : (
          <div className="flex items-center gap-4">
            <div
              className={`flex h-12 w-12 items-center justify-center rounded-full ${
                overallStatus === "ready"
                  ? "bg-[var(--success)]/10"
                  : overallStatus === "limited"
                    ? "bg-[var(--warning)]/10"
                    : overallStatus === "blocked"
                      ? "bg-[var(--destructive)]/10"
                      : "bg-[var(--muted)]"
              }`}
            >
              <ShieldCheck
                className={`h-6 w-6 ${
                  overallStatus === "ready"
                    ? "text-[var(--success)]"
                    : overallStatus === "limited"
                      ? "text-[var(--warning)]"
                      : overallStatus === "blocked"
                        ? "text-[var(--destructive)]"
                        : "text-[var(--muted-foreground)]"
                }`}
              />
            </div>
            <div>
              <h3
                className={`text-lg font-semibold ${
                  overallStatus === "ready"
                    ? "text-[var(--success)]"
                    : overallStatus === "limited"
                      ? "text-[var(--warning)]"
                      : overallStatus === "blocked"
                        ? "text-[var(--destructive)]"
                        : "text-[var(--muted-foreground)]"
                }`}
              >
                {overallStatus === "ready"
                  ? "Ready"
                  : overallStatus === "limited"
                    ? "Limited"
                    : overallStatus === "blocked"
                      ? "Blocked"
                      : "Awaiting validation..."}
              </h3>
              <p className="text-sm text-[var(--muted-foreground)]">
                {overallStatus === "ready"
                  ? "All checks passed. Ready for workload configuration."
                  : overallStatus === "limited"
                    ? "Deployment possible with constraints. Review calculation below."
                    : overallStatus === "blocked"
                      ? "Critical issues need resolution before proceeding."
                      : "Complete evidence discovery first."}
              </p>
            </div>
          </div>
        )}
      </div>

      {/* Validation Checks */}
      {checks.length > 0 && (
        <div className="rounded-xl border border-[var(--border)] p-6">
          <h3 className="mb-4 font-medium">Validation Checks</h3>
          <div className="space-y-3">
            {checks.map((check) => (
              <div
                key={check.label}
                className="flex items-center gap-3 rounded-lg border border-[var(--border)] px-4 py-3"
              >
                <StatusIcon status={check.status} />
                <span className="text-sm">{check.label}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Memory Calculation Breakdown */}
      {memoryEstimate && paramsBillions && selectedGpu && (
        <MemoryBreakdown
          estimate={memoryEstimate}
          paramsBillions={paramsBillions}
          gpuId={selectedGpu}
          gpuCount={gpuCount}
        />
      )}

      {/* Suggestions when memory doesn't fit or is tight */}
      {memoryEstimate && !memoryEstimate.fits && (
        <div className="rounded-xl border border-dashed border-[var(--destructive)]/30 bg-[var(--destructive)]/5 p-6">
          <div className="mb-3 flex items-center gap-2">
            <Lightbulb className="h-5 w-5 text-[var(--destructive)]" />
            <h3 className="font-medium text-[var(--destructive)]">Recommendations</h3>
          </div>
          <ul className="space-y-2 text-sm text-[var(--muted-foreground)]">
            <li className="flex items-start gap-2">
              <span className="mt-1.5 block h-1.5 w-1.5 rounded-full bg-[var(--destructive)]" />
              Increase GPU count: need at least{" "}
              {Math.ceil(memoryEstimate.totalRequiredGb / (GPU_MEMORY_GB[selectedGpu!] || 80))} GPUs for FP16
            </li>
            <li className="flex items-start gap-2">
              <span className="mt-1.5 block h-1.5 w-1.5 rounded-full bg-[var(--destructive)]" />
              Use quantization (AWQ/GPTQ/INT4) to reduce weight memory by ~50%
              → ~{(memoryEstimate.modelWeightsGb / 2).toFixed(1)} GB
            </li>
            <li className="flex items-start gap-2">
              <span className="mt-1.5 block h-1.5 w-1.5 rounded-full bg-[var(--destructive)]" />
              Consider a higher-memory GPU (e.g., H100 80GB or A100 80GB)
            </li>
          </ul>
        </div>
      )}

      {memoryEstimate && memoryEstimate.fits && overallStatus === "limited" && (
        <div className="rounded-xl border border-dashed border-[var(--warning)]/30 bg-[var(--warning)]/5 p-6">
          <div className="mb-3 flex items-center gap-2">
            <Lightbulb className="h-5 w-5 text-[var(--warning)]" />
            <h3 className="font-medium text-[var(--warning)]">Optimization Tips</h3>
          </div>
          <ul className="space-y-2 text-sm text-[var(--muted-foreground)]">
            <li className="flex items-start gap-2">
              <span className="mt-1.5 block h-1.5 w-1.5 rounded-full bg-[var(--warning)]" />
              Memory headroom is tight ({(memoryEstimate.totalAvailableGb - memoryEstimate.totalRequiredGb).toFixed(1)} GB).
              KV cache may limit concurrent requests.
            </li>
            <li className="flex items-start gap-2">
              <span className="mt-1.5 block h-1.5 w-1.5 rounded-full bg-[var(--warning)]" />
              Consider quantization to free memory for larger KV cache and better throughput
            </li>
          </ul>
        </div>
      )}
    </div>
  );
}
