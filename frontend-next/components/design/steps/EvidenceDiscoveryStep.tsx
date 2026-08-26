"use client";

import { FileSearch, BookOpen, FlaskConical, Globe, ShieldCheck, Loader2, ExternalLink } from "lucide-react";
import { cn } from "@/lib/utils";
import type { AgentState, NodeStatus } from "@/lib/use-agent-stream";

const EVIDENCE_SOURCES = [
  {
    nodeKey: "fetch_huggingface_metadata",
    icon: FileSearch,
    title: "Model Architecture",
    description: "Parsing model config, tokenizer, and architecture details",
  },
  {
    nodeKey: "discover_vllm_recipe",
    icon: BookOpen,
    title: "vLLM Recipe",
    description: "Searching for official vLLM serving configurations",
  },
  {
    nodeKey: "discover_redhat_evaluations",
    icon: FlaskConical,
    title: "Red Hat Evaluations",
    description: "HuggingFace model card benchmarks and evaluation results",
  },
  {
    nodeKey: "check_rhoai_compatibility",
    icon: ShieldCheck,
    title: "RHOAI Compatibility",
    description: "Validated models matrix and accelerator compatibility",
  },
  {
    nodeKey: "fetch_pricing",
    icon: Globe,
    title: "Pricing Data",
    description: "Fetching GPU pricing and cost estimation data",
  },
];

function QualityScore({ score, hasError }: { score: number; hasError: boolean }) {
  if (hasError || score === 0) {
    return (
      <span className="flex items-center gap-1 text-[11px] font-bold text-[var(--destructive)]">
        <span className="flex gap-0.5">
          {[1, 2, 3, 4, 5].map((i) => (
            <span
              key={i}
              className="h-1.5 w-3 rounded-sm bg-[var(--destructive)]/20"
            />
          ))}
        </span>
        0/5
      </span>
    );
  }

  const color = score >= 4
    ? "var(--success)"
    : score >= 3
      ? "var(--primary)"
      : score >= 2
        ? "var(--warning)"
        : "var(--destructive)";

  return (
    <span className="flex items-center gap-1 text-[11px] font-bold" style={{ color }}>
      <span className="flex gap-0.5">
        {[1, 2, 3, 4, 5].map((i) => (
          <span
            key={i}
            className="h-1.5 w-3 rounded-sm transition-colors"
            style={{
              backgroundColor: i <= score ? color : `color-mix(in srgb, ${color} 20%, transparent)`,
            }}
          />
        ))}
      </span>
      {score}/5
    </span>
  );
}

function StatusBadge({ status }: { status: NodeStatus }) {
  switch (status) {
    case "done":
      return (
        <span className="flex items-center gap-1.5 rounded-full bg-[var(--success)]/10 px-2 py-0.5 text-xs font-medium text-[var(--success)]">
          <span className="h-1.5 w-1.5 rounded-full bg-[var(--success)]" />
          Done
        </span>
      );
    case "running":
      return (
        <span className="flex items-center gap-1.5 rounded-full bg-[var(--primary)]/10 px-2 py-0.5 text-xs font-medium text-[var(--primary)]">
          <Loader2 className="h-3 w-3 animate-spin" />
          Running
        </span>
      );
    case "error":
      return (
        <span className="flex items-center gap-1.5 rounded-full bg-[var(--destructive)]/10 px-2 py-0.5 text-xs font-medium text-[var(--destructive)]">
          <span className="h-1.5 w-1.5 rounded-full bg-[var(--destructive)]" />
          Error
        </span>
      );
    default:
      return (
        <span className="flex items-center gap-1.5 rounded-full bg-[var(--muted)] px-2 py-0.5 text-xs font-medium text-[var(--muted-foreground)]">
          <span className="h-1.5 w-1.5 rounded-full bg-[var(--muted-foreground)]" />
          Pending
        </span>
      );
  }
}

interface EvidenceDiscoveryStepProps {
  agentState: AgentState;
}

export function EvidenceDiscoveryStep({ agentState }: EvidenceDiscoveryStepProps) {
  const { status, nodeStatuses, steps, error } = agentState;

  const completedCount = EVIDENCE_SOURCES.filter(
    (s) => nodeStatuses[s.nodeKey] === "done"
  ).length;
  const totalCount = EVIDENCE_SOURCES.length;
  const progressPct = totalCount > 0 ? (completedCount / totalCount) * 100 : 0;

  const isRunning = status === "running";
  const isCompleted = status === "completed";
  const isError = status === "error";

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-lg font-semibold">Evidence Discovery</h2>
        <p className="mt-1 text-sm text-[var(--muted-foreground)]">
          Gathering deployment evidence from multiple sources to inform
          recommendations.
        </p>
      </div>

      {/* Progress Timeline */}
      <div className="rounded-xl border border-[var(--border)] p-6">
        <div className="mb-4 flex items-center justify-between">
          <h3 className="font-medium">Discovery Progress</h3>
          <span
            className={cn(
              "rounded-full px-2.5 py-1 text-xs font-medium",
              isCompleted
                ? "bg-[var(--success)]/10 text-[var(--success)]"
                : isError
                  ? "bg-[var(--destructive)]/10 text-[var(--destructive)]"
                  : isRunning
                    ? "bg-[var(--primary)]/10 text-[var(--primary)]"
                    : "bg-[var(--muted)] text-[var(--muted-foreground)]"
            )}
          >
            {isError
              ? "Error"
              : isCompleted
                ? "Complete"
                : isRunning
                  ? `${completedCount} / ${totalCount} complete`
                  : "Waiting..."}
          </span>
        </div>

        <div className="h-2 w-full overflow-hidden rounded-full bg-[var(--muted)]">
          <div
            className={cn(
              "h-full rounded-full transition-all duration-700 ease-out",
              isError ? "bg-[var(--destructive)]" : "bg-[var(--primary)]"
            )}
            style={{ width: `${isCompleted ? 100 : progressPct}%` }}
          />
        </div>

        {isRunning && (
          <div className="mt-3 flex items-center gap-2 text-xs text-[var(--muted-foreground)]">
            <Loader2 className="h-3 w-3 animate-spin" />
            Analyzing model metadata and collecting evidence...
          </div>
        )}

        {error && (
          <div className="mt-3 rounded-lg border border-[var(--destructive)]/30 bg-[var(--destructive)]/5 p-3 text-xs text-[var(--destructive)]">
            {error}
          </div>
        )}
      </div>

      {/* Evidence Source Cards */}
      <div className="grid gap-4 sm:grid-cols-2">
        {EVIDENCE_SOURCES.map((source) => {
          const nodeStatus = nodeStatuses[source.nodeKey] || "pending";
          const isDone = nodeStatus === "done";

          const stepData = steps.find((s) => s.node === source.nodeKey);
          const hasError = nodeStatus === "error" || !!stepData?.error;
          const qualityScore = stepData?.qualityScore || 0;
          const isLowScore = isDone && !hasError && qualityScore <= 2;

          return (
            <div
              key={source.nodeKey}
              className={cn(
                "rounded-xl border p-5 transition-all",
                hasError
                  ? "border-[var(--destructive)]/30 bg-[var(--destructive)]/5"
                  : isDone && !isLowScore
                    ? "border-[var(--success)]/30 bg-[var(--success)]/5"
                    : isDone && isLowScore
                      ? "border-[var(--warning)]/30 bg-[var(--warning)]/5"
                      : nodeStatus === "running"
                        ? "border-[var(--primary)]/30 bg-[var(--primary)]/5"
                        : "border-[var(--border)]"
              )}
            >
                <div className="mb-3 flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <div
                    className={cn(
                      "flex h-9 w-9 items-center justify-center rounded-lg",
                      hasError
                        ? "bg-[var(--destructive)]/10"
                        : isDone && !isLowScore
                          ? "bg-[var(--success)]/10"
                          : isDone && isLowScore
                            ? "bg-[var(--warning)]/10"
                            : "bg-[var(--muted)]"
                    )}
                  >
                    <source.icon
                      className={cn(
                        "h-4 w-4",
                        hasError
                          ? "text-[var(--destructive)]"
                          : isDone && !isLowScore
                            ? "text-[var(--success)]"
                            : isDone && isLowScore
                              ? "text-[var(--warning)]"
                              : "text-[var(--muted-foreground)]"
                      )}
                    />
                  </div>
                  <div>
                    <h4 className="text-sm font-medium">{source.title}</h4>
                    <p className="text-xs text-[var(--muted-foreground)]">
                      {source.description}
                    </p>
                  </div>
                </div>
                <StatusBadge status={nodeStatus} />
              </div>

              {nodeStatus === "pending" && isRunning && (
                <div className="space-y-2 pt-2">
                  <div className="h-3 w-full animate-pulse rounded bg-[var(--muted)]" />
                  <div className="h-3 w-4/5 animate-pulse rounded bg-[var(--muted)]" />
                </div>
              )}

              {(isDone || nodeStatus === "error") && (() => {
                const stepData = steps.find((s) => s.node === source.nodeKey);
                const score = stepData?.qualityScore || 0;
                const urls = stepData?.sourceUrls || [];
                const hasError = !!stepData?.error;

                return (
                  <div className="space-y-2 pt-2">
                    {/* Quality Score */}
                    <div className="flex items-center gap-2">
                      <QualityScore score={score} hasError={hasError} />
                      {stepData?.evidenceCount && stepData.evidenceCount > 0 && (
                        <span className="text-[11px] text-[var(--muted-foreground)]">
                          {stepData.evidenceCount} item{stepData.evidenceCount > 1 ? "s" : ""}
                        </span>
                      )}
                    </div>

                    {/* Error message */}
                    {hasError && (
                      <div className="text-[11px] text-[var(--destructive)]">
                        {stepData?.error}
                      </div>
                    )}

                    {/* Source citations */}
                    {urls.length > 0 && (
                      <div className="space-y-0.5">
                        {urls.slice(0, 2).map((url, i) => (
                          <a
                            key={i}
                            href={url}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="flex items-center gap-1 text-[10px] text-[var(--primary)] hover:underline truncate"
                          >
                            <ExternalLink className="h-2.5 w-2.5 shrink-0" />
                            <span className="truncate">{new URL(url).hostname}{new URL(url).pathname.slice(0, 40)}</span>
                          </a>
                        ))}
                        {urls.length > 2 && (
                          <span className="text-[10px] text-[var(--muted-foreground)]">
                            +{urls.length - 2} more source{urls.length - 2 > 1 ? "s" : ""}
                          </span>
                        )}
                      </div>
                    )}
                  </div>
                );
              })()}
            </div>
          );
        })}
      </div>

      {/* Event Log */}
      {steps.length > 0 && (
        <div className="rounded-xl border border-[var(--border)] p-6">
          <div className="mb-3 flex items-center justify-between">
            <h3 className="text-sm font-medium">Activity Log</h3>
            {agentState.totalEvidence > 0 && (
              <span className="rounded-full bg-[var(--primary)]/10 px-2.5 py-0.5 text-xs font-medium text-[var(--primary)]">
                {agentState.totalEvidence} evidence items collected
              </span>
            )}
          </div>
          <div className="max-h-52 space-y-2 overflow-y-auto">
            {steps.map((step, i) => (
              <div
                key={i}
                className={cn(
                  "flex items-start gap-2.5 rounded-lg border px-3 py-2",
                  step.error
                    ? "border-[var(--destructive)]/30 bg-[var(--destructive)]/5"
                    : "border-[var(--border)] bg-[var(--muted)]/30"
                )}
              >
                <span className={cn(
                  "mt-1.5 h-2 w-2 shrink-0 rounded-full",
                  step.error ? "bg-[var(--destructive)]" : "bg-[var(--success)]"
                )} />
                <div className="min-w-0 flex-1">
                  <div className="flex items-center justify-between gap-2">
                    <span className="text-xs font-medium text-[var(--foreground)]">
                      {step.detail || step.node}
                    </span>
                    <div className="flex items-center gap-2 shrink-0">
                      {step.qualityScore !== undefined && (
                        <span className={cn(
                          "text-[10px] font-bold",
                          step.qualityScore >= 4 ? "text-[var(--success)]"
                            : step.qualityScore >= 3 ? "text-[var(--primary)]"
                              : step.qualityScore >= 2 ? "text-[var(--warning)]"
                                : "text-[var(--destructive)]"
                        )}>
                          {step.qualityScore}/5
                        </span>
                      )}
                      <span className="text-[10px] tabular-nums text-[var(--muted-foreground)]">
                        {new Date(step.timestamp).toLocaleTimeString([], {
                          hour: "2-digit",
                          minute: "2-digit",
                          second: "2-digit",
                        })}
                      </span>
                    </div>
                  </div>
                  {step.error && (
                    <span className="mt-0.5 inline-block text-[10px] text-[var(--destructive)]">
                      {step.error}
                    </span>
                  )}
                  {step.sourceUrls && step.sourceUrls.length > 0 && (
                    <div className="mt-0.5 flex flex-wrap gap-x-3 gap-y-0.5">
                      {step.sourceUrls.slice(0, 2).map((url, j) => (
                        <a
                          key={j}
                          href={url}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="flex items-center gap-0.5 text-[10px] text-[var(--primary)] hover:underline"
                        >
                          <ExternalLink className="h-2 w-2" />
                          {(() => { try { return new URL(url).hostname; } catch { return url.slice(0, 30); } })()}
                        </a>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
