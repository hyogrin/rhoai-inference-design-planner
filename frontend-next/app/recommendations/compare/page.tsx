"use client";

import { useEffect, useState, Suspense } from "react";
import { useSearchParams } from "next/navigation";
import Link from "next/link";
import {
  ArrowLeft,
  CheckCircle2,
  AlertTriangle,
  MemoryStick,
  Cpu,
  DollarSign,
  TrendingUp,
  Loader2,
} from "lucide-react";
import { cn } from "@/lib/utils";

interface DesignDetail {
  session_id: string;
  title: string | null;
  model_repo_id: string | null;
  completed_at: string | null;
  result_snapshot: {
    view_model: {
      sections: Array<{ id: string; title: string; data: Record<string, unknown> }>;
      verification?: { warnings: string[]; status: string };
    };
  } | null;
}

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:7001";

function CompareContent() {
  const searchParams = useSearchParams();
  const idA = searchParams.get("a");
  const idB = searchParams.get("b");

  const [designA, setDesignA] = useState<DesignDetail | null>(null);
  const [designB, setDesignB] = useState<DesignDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!idA || !idB) {
      setError("Two recommendation IDs are required");
      setLoading(false);
      return;
    }
    Promise.all([
      fetch(`${API_BASE}/api/v1/designs/${idA}`).then((r) => r.json()),
      fetch(`${API_BASE}/api/v1/designs/${idB}`).then((r) => r.json()),
    ])
      .then(([a, b]) => {
        setDesignA(a);
        setDesignB(b);
      })
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false));
  }, [idA, idB]);

  if (loading) {
    return (
      <div className="flex items-center justify-center py-20">
        <Loader2 className="h-6 w-6 animate-spin text-[var(--primary)]" />
      </div>
    );
  }

  if (error || !designA || !designB) {
    return (
      <div className="mx-auto max-w-4xl px-6 py-8">
        <div className="rounded-xl border border-red-500/30 bg-red-500/5 p-4 text-sm text-red-600">
          {error || "Failed to load recommendations"}
        </div>
      </div>
    );
  }

  const sectionsA = designA.result_snapshot?.view_model?.sections || [];
  const sectionsB = designB.result_snapshot?.view_model?.sections || [];

  const getSection = (sections: typeof sectionsA, id: string) =>
    sections.find((s) => s.id === id)?.data || {};

  const modelA = getSection(sectionsA, "model_summary");
  const modelB = getSection(sectionsB, "model_summary");
  const memA = getSection(sectionsA, "memory_estimate");
  const memB = getSection(sectionsB, "memory_estimate");
  const deployA = getSection(sectionsA, "deployment_config");
  const deployB = getSection(sectionsB, "deployment_config");
  const costA = getSection(sectionsA, "cost_estimate");
  const costB = getSection(sectionsB, "cost_estimate");
  const perfA = getSection(sectionsA, "performance_forecast");
  const perfB = getSection(sectionsB, "performance_forecast");
  const suggA = getSection(sectionsA, "design_suggestion");
  const suggB = getSection(sectionsB, "design_suggestion");

  return (
    <div className="mx-auto max-w-7xl px-6 py-8 space-y-6">
      {/* Column Headers */}
      <div className="grid grid-cols-[1fr_1fr] gap-4">
        <ColumnHeader label="A" design={designA} />
        <ColumnHeader label="B" design={designB} />
      </div>

      {/* Model Overview */}
      <CompareSection title="Model Overview" icon={<Cpu className="h-4 w-4" />}>
        <CompareRow label="Model" a={str(modelA.repo_id)} b={str(modelB.repo_id)} />
        <CompareRow label="Architecture" a={str(modelA.architecture_type)} b={str(modelB.architecture_type)} />
        <CompareRow label="Parameters" a={str(modelA.parameters_display)} b={str(modelB.parameters_display)} />
        <CompareRow label="Context Length" a={fmtNum(modelA.context_length)} b={fmtNum(modelB.context_length)} suffix="tokens" />
      </CompareSection>

      {/* Memory */}
      <CompareSection title="Memory" icon={<MemoryStick className="h-4 w-4" />}>
        <CompareRow label="Model Weights" a={`${str(memA.model_weights_gb)} GB`} b={`${str(memB.model_weights_gb)} GB`} />
        <CompareRow label="KV Cache" a={`${str(memA.kv_cache_gb)} GB`} b={`${str(memB.kv_cache_gb)} GB`} />
        <CompareRow label="Overhead" a={`${str(memA.overhead_min_total_gb ?? memA.overhead_gb ?? "—")} GB`} b={`${str(memB.overhead_min_total_gb ?? memB.overhead_gb ?? "—")} GB`} />
        <CompareRow label="Total Required" a={`${str(memA.total_required_min_gb ?? memA.total_required_gb ?? "—")} GB`} b={`${str(memB.total_required_min_gb ?? memB.total_required_gb ?? "—")} GB`} />
        <CompareRow label="Available" a={`${str(memA.total_available_gb)} GB`} b={`${str(memB.total_available_gb)} GB`} />
        <CompareRow label="Utilization" a={`${str(memA.utilization_pct)}%`} b={`${str(memB.utilization_pct)}%`} highlight="lower" />
        <CompareRow label="Fits" a={memA.fits === true ? "Yes" : memA.fits === false ? "No" : "—"} b={memB.fits === true ? "Yes" : memB.fits === false ? "No" : "—"} highlightGood="Yes" />
      </CompareSection>

      {/* Deployment Config */}
      <CompareSection title="Deployment" icon={<Cpu className="h-4 w-4" />}>
        <CompareRow label="GPU" a={formatGpuWithPlatform(deployA)} b={formatGpuWithPlatform(deployB)} />
        <CompareRow label="RHOAI" a={`v${str(deployA.rhoai_version)}`} b={`v${str(deployB.rhoai_version)}`} />
        <CompareRow label="vLLM" a={str(deployA.vllm_version || deployA.min_vllm_version)} b={str(deployB.vllm_version || deployB.min_vllm_version)} />
      </CompareSection>

      {/* Workload & Targets — placed right before Performance Forecast for context */}
      <CompareSection title="Workload & Targets" icon={<TrendingUp className="h-4 w-4" />}>
        <CompareRow label="Use Cases" a={(deployA.use_case_presets as string[] | undefined)?.join(", ") || "—"} b={(deployB.use_case_presets as string[] | undefined)?.join(", ") || "—"} />
        <CompareRow label="Max Concurrent" a={str(deployA.max_concurrent_requests)} b={str(deployB.max_concurrent_requests)} />
        <CompareRow label="TTFT Target" a={`${str(deployA.ttft_target_ms)}ms`} b={`${str(deployB.ttft_target_ms)}ms`} />
        <CompareRow label="TPOT Target" a={`${str(deployA.tpot_target_ms)}ms`} b={`${str(deployB.tpot_target_ms)}ms`} />
      </CompareSection>

      {/* Performance */}
      <CompareSection title="Performance Forecast" icon={<TrendingUp className="h-4 w-4" />}>
        <CompareRow label="Decode Throughput" a={`${fmtNum(perfA.theoretical_decode_tps)} tok/s`} b={`${fmtNum(perfB.theoretical_decode_tps)} tok/s`} highlight="higher" />
        <CompareRow label="Est. TPOT" a={`${str(perfA.estimated_tpot_ms)}ms`} b={`${str(perfB.estimated_tpot_ms)}ms`} highlight="lower" />
        <CompareRow label="Est. TTFT" a={`${str(perfA.estimated_ttft_ms)}ms`} b={`${str(perfB.estimated_ttft_ms)}ms`} highlight="lower" />
        <CompareRow label="Ridge Batch" a={str(perfA.ridge_batch_size)} b={str(perfB.ridge_batch_size)} />
      </CompareSection>

      {/* Cost */}
      <CompareSection title="Cost" icon={<DollarSign className="h-4 w-4" />}>
        {costA.type === "cloud" || costB.type === "cloud" ? (
          <>
            <CompareRow label="Type" a={str(costA.type)} b={str(costB.type)} />
            <CompareRow label="Monthly" a={`$${fmtNum(costA.monthly_on_demand_usd || costA.monthly_total_usd)}`} b={`$${fmtNum(costB.monthly_on_demand_usd || costB.monthly_total_usd)}`} highlight="lower" />
          </>
        ) : (
          <>
            <CompareRow label="Monthly TCO" a={`$${fmtNum(costA.monthly_total_usd)}`} b={`$${fmtNum(costB.monthly_total_usd)}`} highlight="lower" />
          </>
        )}
      </CompareSection>

      {/* Design Suggestion */}
      {(suggA.content || suggB.content) ? (
        <div className="rounded-xl border border-[var(--border)] overflow-hidden">
          <div className="border-b border-[var(--border)] bg-[var(--muted)]/30 px-5 py-3">
            <h3 className="text-sm font-semibold">Design Suggestion</h3>
          </div>
          <div className="grid grid-cols-2 divide-x divide-[var(--border)]">
            <div className="p-5">
              <SuggestionContent content={str(suggA.content)} />
            </div>
            <div className="p-5">
              <SuggestionContent content={str(suggB.content)} />
            </div>
          </div>
        </div>
      ) : null}
    </div>
  );
}

export default function ComparePage() {
  return (
    <main className="min-h-screen">
      <header className="border-b border-[var(--border)] px-6 py-4">
        <div className="flex items-center gap-3">
          <Link href="/recommendations" className="text-[var(--muted-foreground)] hover:text-[var(--foreground)] transition-colors">
            <ArrowLeft className="h-5 w-5" />
          </Link>
          <h1 className="text-xl font-semibold">Compare Recommendations</h1>
        </div>
        <p className="mt-1 ml-8 text-sm text-[var(--muted-foreground)]">
          Side-by-side comparison with highlighted differences
        </p>
      </header>

      <Suspense fallback={
        <div className="flex items-center justify-center py-20">
          <Loader2 className="h-6 w-6 animate-spin text-[var(--primary)]" />
        </div>
      }>
        <CompareContent />
      </Suspense>
    </main>
  );
}

// --- Helpers ---

function str(val: unknown): string {
  if (val == null || val === "") return "—";
  return String(val);
}

function fmtNum(val: unknown): string {
  if (val == null || val === "") return "—";
  const n = Number(val);
  if (isNaN(n)) return String(val);
  return n.toLocaleString();
}

function formatPlatformLabel(platform: unknown): string {
  if (!platform) return "On-Premise";
  const p = String(platform).toLowerCase();
  if (p === "on-premise" || p === "on_prem") return "On-Premise";
  if (p === "aws") return "AWS";
  if (p === "azure") return "Azure";
  if (p === "gcp") return "GCP";
  return String(platform);
}

function formatGpuWithPlatform(deploy: Record<string, unknown>): string {
  const gpu = `${str(deploy.gpu_count)}× ${str(deploy.gpu_type)}`;
  const platform = formatPlatformLabel(deploy.platform || deploy.environment_type);
  return `${gpu} (${platform})`;
}

function ColumnHeader({ label, design }: { label: string; design: DesignDetail }) {
  return (
    <div className="rounded-xl border border-[var(--border)] bg-[var(--muted)]/20 p-4">
      <div className="flex items-center gap-2">
        <span className="flex h-6 w-6 items-center justify-center rounded-full bg-[var(--primary)] text-xs font-bold text-white">
          {label}
        </span>
        <span className="text-sm font-semibold truncate">
          {design.model_repo_id || design.title || "Unnamed"}
        </span>
      </div>
      {design.completed_at && (
        <p className="mt-1 ml-8 text-xs text-[var(--muted-foreground)]">
          {new Date(design.completed_at).toLocaleDateString("ko-KR", {
            year: "numeric",
            month: "short",
            day: "numeric",
            hour: "2-digit",
            minute: "2-digit",
          })}
        </p>
      )}
    </div>
  );
}

function CompareSection({
  title,
  icon,
  children,
}: {
  title: string;
  icon: React.ReactNode;
  children: React.ReactNode;
}) {
  return (
    <div className="rounded-xl border border-[var(--border)] overflow-hidden">
      <div className="flex items-center gap-2 border-b border-[var(--border)] bg-[var(--muted)]/30 px-5 py-3">
        <span className="text-[var(--primary)]">{icon}</span>
        <h3 className="text-sm font-semibold">{title}</h3>
      </div>
      <div className="divide-y divide-[var(--border)]">{children}</div>
    </div>
  );
}

function CompareRow({
  label,
  a,
  b,
  suffix,
  highlight,
  highlightGood,
}: {
  label: string;
  a: string;
  b: string;
  suffix?: string;
  highlight?: "higher" | "lower";
  highlightGood?: string;
}) {
  const isDifferent = a !== b;
  let betterSide: "a" | "b" | null = null;

  if (isDifferent && highlight) {
    const numA = parseFloat(a.replace(/[^0-9.-]/g, ""));
    const numB = parseFloat(b.replace(/[^0-9.-]/g, ""));
    if (!isNaN(numA) && !isNaN(numB)) {
      betterSide = highlight === "higher"
        ? (numA > numB ? "a" : "b")
        : (numA < numB ? "a" : "b");
    }
  }

  if (isDifferent && highlightGood) {
    if (a === highlightGood && b !== highlightGood) betterSide = "a";
    else if (b === highlightGood && a !== highlightGood) betterSide = "b";
  }

  const aVal = suffix && a !== "—" ? `${a} ${suffix}` : a;
  const bVal = suffix && b !== "—" ? `${b} ${suffix}` : b;

  return (
    <div className="grid grid-cols-[140px_1fr_1fr] items-center gap-4 px-5 py-2.5">
      <span className="text-xs text-[var(--muted-foreground)]">{label}</span>
      <span className={cn(
        "text-xs font-medium",
        isDifferent && "font-semibold",
        betterSide === "a" && "text-green-600",
        betterSide === "b" && "text-[var(--foreground)]"
      )}>
        {aVal}
        {betterSide === "a" && <CheckCircle2 className="ml-1 inline h-3 w-3" />}
        {isDifferent && betterSide === "b" && <AlertTriangle className="ml-1 inline h-3 w-3 text-amber-500" />}
      </span>
      <span className={cn(
        "text-xs font-medium",
        isDifferent && "font-semibold",
        betterSide === "b" && "text-green-600",
        betterSide === "a" && "text-[var(--foreground)]"
      )}>
        {bVal}
        {betterSide === "b" && <CheckCircle2 className="ml-1 inline h-3 w-3" />}
        {isDifferent && betterSide === "a" && <AlertTriangle className="ml-1 inline h-3 w-3 text-amber-500" />}
      </span>
    </div>
  );
}

function SuggestionContent({ content }: { content: string }) {
  if (!content || content === "—") {
    return <p className="text-xs text-[var(--muted-foreground)] italic">No suggestion</p>;
  }

  const lines = content.split("\n");
  return (
    <div className="space-y-1 text-xs leading-relaxed text-[var(--muted-foreground)]">
      {lines.map((line, i) => {
        if (line.match(/^#{1,4}\s+/)) {
          return (
            <h4 key={i} className="mt-2 text-xs font-semibold text-[var(--foreground)]">
              {line.replace(/^#{1,4}\s+/, "")}
            </h4>
          );
        }
        if (line.startsWith("- ") || line.startsWith("• ")) {
          return (
            <div key={i} className="flex gap-2">
              <span className="mt-0.5 shrink-0 text-[var(--primary)]">•</span>
              <span>{line.replace(/^[-•]\s*/, "")}</span>
            </div>
          );
        }
        return line.trim() ? <p key={i}>{line}</p> : null;
      })}
    </div>
  );
}
