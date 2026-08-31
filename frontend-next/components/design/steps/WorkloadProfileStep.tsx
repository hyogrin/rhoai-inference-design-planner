"use client";

import { useState } from "react";
import { cn } from "@/lib/utils";
import { useI18n } from "@/lib/i18n";
import { MessageSquare, Database, Bot, Layers, Code, Send } from "lucide-react";
import type { WorkloadInterrupt } from "@/lib/use-agent-stream";

const PRESETS = [
  { id: "chatbot", icon: MessageSquare, label: "Chatbot", description: "Interactive dialogue" },
  { id: "rag", icon: Database, label: "RAG", description: "Retrieval augmented" },
  { id: "coding", icon: Code, label: "Coding Assistant", description: "Code generation & completion" },
  { id: "agentic", icon: Bot, label: "Agentic", description: "Tool-calling agents" },
  { id: "batch", icon: Layers, label: "Batch", description: "Offline processing" },
] as const;

type PresetId = (typeof PRESETS)[number]["id"];

const RHOAI_VERSIONS = [
  { version: "3.4", vllm: "0.18", label: "3.4" },
  { version: "3.5", vllm: "0.24", label: "3.5" },
] as const;

interface WorkloadProfileStepProps {
  interrupt: WorkloadInterrupt | null;
  onSubmit: (data: WorkloadConfig) => void;
  gpu: string | null;
  gpuCount: number;
  platform: string | null;
}

export interface WorkloadConfig {
  use_case_presets: PresetId[];
  use_case_allocations: Record<string, number>;
  target_end_users: number;
  max_concurrent_requests: number;
  ttft_ms: number;
  tpot_ms: number;
  rhoai_version: string;
  vllm_version: string;
  gpu_type: string | null;
  gpu_count: number;
  platform: string | null;
}

export function WorkloadProfileStep({
  interrupt,
  onSubmit,
  gpu,
  gpuCount,
  platform,
}: WorkloadProfileStepProps) {
  const [selectedPresets, setSelectedPresets] = useState<PresetId[]>([]);
  const [allocations, setAllocations] = useState<Record<PresetId, number>>({} as Record<PresetId, number>);
  const [targetUsers, setTargetUsers] = useState(100);
  const [concurrency, setConcurrency] = useState(32);
  const [ttftMs, setTtftMs] = useState(500);
  const [tpotMs, setTpotMs] = useState(30);
  const [rhoaiVersion, setRhoaiVersion] = useState("3.5");
  const { t } = useI18n();

  const selectedRhoai = RHOAI_VERSIONS.find((v) => v.version === rhoaiVersion)!;

  const togglePreset = (id: PresetId) => {
    setSelectedPresets((prev) => {
      const next = prev.includes(id)
        ? prev.filter((p) => p !== id)
        : [...prev, id];

      if (next.length > 0) {
        const even = Math.floor(100 / next.length);
        const remainder = 100 - even * next.length;
        const newAlloc: Record<string, number> = {};
        next.forEach((p, i) => {
          newAlloc[p] = even + (i === 0 ? remainder : 0);
        });
        setAllocations(newAlloc as Record<PresetId, number>);
      } else {
        setAllocations({} as Record<PresetId, number>);
      }
      return next;
    });
  };

  const updateAllocation = (id: PresetId, value: number) => {
    setAllocations((prev) => {
      const updated = { ...prev, [id]: value };
      const others = selectedPresets.filter((p) => p !== id);
      const remaining = 100 - value;
      const othersSum = others.reduce((s, p) => s + (prev[p] || 0), 0);

      if (othersSum > 0 && others.length > 0) {
        others.forEach((p) => {
          updated[p] = Math.round(((prev[p] || 0) / othersSum) * remaining);
        });
        const newSum = Object.values(updated).reduce((s, v) => s + v, 0);
        if (newSum !== 100 && others.length > 0) {
          updated[others[0]] += 100 - newSum;
        }
      } else if (others.length > 0) {
        const each = Math.floor(remaining / others.length);
        others.forEach((p, i) => {
          updated[p] = each + (i === 0 ? remaining - each * others.length : 0);
        });
      }
      return updated as Record<PresetId, number>;
    });
  };

  const handleSubmit = () => {
    const config: WorkloadConfig = {
      use_case_presets: selectedPresets,
      use_case_allocations: allocations,
      target_end_users: targetUsers,
      max_concurrent_requests: concurrency,
      ttft_ms: ttftMs,
      tpot_ms: tpotMs,
      rhoai_version: selectedRhoai.version,
      vllm_version: selectedRhoai.vllm,
      gpu_type: gpu,
      gpu_count: gpuCount,
      platform,
    };
    onSubmit(config);
  };

  const allocationSum = Object.values(allocations).reduce((s, v) => s + v, 0);

  if (!interrupt) {
    return (
      <div className="space-y-6">
        <div>
          <h2 className="text-lg font-semibold">{t("step4.title")}</h2>
          <p className="mt-1 text-sm text-[var(--muted-foreground)]">
            {t("step4.waitingDesc")}
          </p>
        </div>
        <div className="flex items-center justify-center rounded-xl border border-dashed border-[var(--border)] p-12 text-sm text-[var(--muted-foreground)]">
          {t("step4.waitingPlaceholder")}
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-lg font-semibold">{t("step4.title")}</h2>
        <p className="mt-1 text-sm text-[var(--muted-foreground)]">
          {t("step4.description")}
        </p>
      </div>

      {/* Model Summary */}
      <div className="rounded-xl border border-[var(--primary)]/20 bg-[var(--accent)] p-4">
        <div className="flex items-center justify-between text-sm">
          <span className="font-medium">{interrupt.model_repo_id}</span>
          <span className="text-xs text-[var(--muted-foreground)]">
            {interrupt.architecture_summary?.parameters
              ? `${(interrupt.architecture_summary.parameters / 1e9).toFixed(0)}B params`
              : "Unknown size"}{" "}
            • {interrupt.evidence_collected} evidence items
          </span>
        </div>
      </div>

      {/* Use Case Presets */}
      <div className="rounded-xl border border-[var(--border)] p-6">
        <div className="mb-1 flex items-center justify-between">
          <h3 className="text-sm font-medium">{t("step4.useCasePattern")}</h3>
          <span className="text-xs text-[var(--muted-foreground)]">{t("step4.multiSelect")}</span>
        </div>
        <p className="mb-4 text-xs text-[var(--muted-foreground)]">
          {t("step4.useCaseHelp")}
        </p>

        <div className="grid grid-cols-2 gap-3 sm:grid-cols-5">
          {PRESETS.map((preset) => {
            const isActive = selectedPresets.includes(preset.id);
            return (
              <button
                key={preset.id}
                onClick={() => togglePreset(preset.id)}
                className={cn(
                  "relative flex flex-col items-center gap-2 rounded-lg border p-4 text-center transition-all",
                  isActive
                    ? "border-[var(--primary)] bg-[var(--accent)] shadow-sm"
                    : "border-[var(--border)] hover:border-[var(--primary)]/50 hover:bg-[var(--muted)]"
                )}
              >
                {isActive && (
                  <span className="absolute right-2 top-2 flex h-4 w-4 items-center justify-center rounded-full bg-[var(--primary)] text-[9px] font-bold text-[var(--primary-foreground)]">
                    ✓
                  </span>
                )}
                <preset.icon className={cn("h-5 w-5", isActive ? "text-[var(--primary)]" : "text-[var(--muted-foreground)]")} />
                <span className="text-xs font-medium">{preset.label}</span>
                <span className="text-[10px] leading-tight text-[var(--muted-foreground)]">
                  {preset.description}
                </span>
              </button>
            );
          })}
        </div>

        {selectedPresets.length > 1 && (
          <div className="mt-5 rounded-lg border border-[var(--border)] bg-[var(--muted)]/30 p-4">
            <div className="mb-3 flex items-center justify-between">
              <span className="text-xs font-medium">{t("step4.trafficDistribution")}</span>
              <span className={cn(
                "text-xs font-bold",
                allocationSum === 100 ? "text-[var(--success)]" : "text-[var(--warning)]"
              )}>
                Total: {allocationSum}%
              </span>
            </div>
            <div className="space-y-3">
              {selectedPresets.map((id) => {
                const preset = PRESETS.find((p) => p.id === id)!;
                const pct = allocations[id] || 0;
                return (
                  <div key={id} className="flex items-center gap-3">
                    <span className="w-28 shrink-0 text-xs font-medium">{preset.label}</span>
                    <input
                      type="range"
                      min={0}
                      max={100}
                      value={pct}
                      onChange={(e) => updateAllocation(id, Number(e.target.value))}
                      className="h-2 flex-1 cursor-pointer appearance-none rounded-full bg-[var(--muted)] accent-[var(--primary)]"
                    />
                    <span className="w-10 text-right text-xs font-bold tabular-nums">{pct}%</span>
                  </div>
                );
              })}
            </div>
          </div>
        )}
      </div>

      {/* Configuration Details */}
      <div className="rounded-xl border border-[var(--border)] p-6">
        <h3 className="mb-4 font-medium">{t("step4.serviceRequirements")}</h3>

        <div className="grid gap-5 sm:grid-cols-2">
          {/* Target End Users */}
          <div>
            <label className="mb-1.5 block text-sm font-medium">
              {t("step4.targetUsers")}
            </label>
            <input
              type="number"
              min={1}
              value={targetUsers}
              onChange={(e) => setTargetUsers(Number(e.target.value))}
              className="w-full rounded-lg border border-[var(--border)] bg-[var(--background)] px-4 py-2.5 text-sm focus:border-[var(--primary)] focus:outline-none focus:ring-2 focus:ring-[var(--primary)]/20"
            />
            <p className="mt-1 text-[10px] text-[var(--muted-foreground)]">
              {t("step4.targetUsersHelp")}
            </p>
          </div>

          {/* Max Concurrent Requests */}
          <div>
            <label className="mb-1.5 block text-sm font-medium">
              {t("step4.maxConcurrent")}
            </label>
            <input
              type="number"
              min={1}
              value={concurrency}
              onChange={(e) => setConcurrency(Number(e.target.value))}
              className="w-full rounded-lg border border-[var(--border)] bg-[var(--background)] px-4 py-2.5 text-sm focus:border-[var(--primary)] focus:outline-none focus:ring-2 focus:ring-[var(--primary)]/20"
            />
            <p className="mt-1 text-[10px] text-[var(--muted-foreground)]">
              {t("step4.maxConcurrentHelp")}
            </p>
          </div>

          {/* TTFT */}
          <div>
            <label className="mb-1.5 block text-sm font-medium">
              {t("step4.ttft")}
            </label>
            <div className="flex items-center gap-2">
              <input
                type="range"
                min={50}
                max={3000}
                step={50}
                value={ttftMs}
                onChange={(e) => setTtftMs(Number(e.target.value))}
                className="h-2 flex-1 cursor-pointer appearance-none rounded-full bg-[var(--muted)] accent-[var(--primary)]"
              />
              <span className="w-16 rounded-md border border-[var(--border)] bg-[var(--background)] px-2 py-1 text-center text-xs font-mono font-medium">
                {ttftMs}ms
              </span>
            </div>
            <p className="mt-1 text-[10px] text-[var(--muted-foreground)]">
              {t("step4.ttftHelp")}
            </p>
          </div>

          {/* TPOT */}
          <div>
            <label className="mb-1.5 block text-sm font-medium">
              {t("step4.tpot")}
            </label>
            <div className="flex items-center gap-2">
              <input
                type="range"
                min={5}
                max={200}
                step={5}
                value={tpotMs}
                onChange={(e) => setTpotMs(Number(e.target.value))}
                className="h-2 flex-1 cursor-pointer appearance-none rounded-full bg-[var(--muted)] accent-[var(--primary)]"
              />
              <span className="w-16 rounded-md border border-[var(--border)] bg-[var(--background)] px-2 py-1 text-center text-xs font-mono font-medium">
                {tpotMs}ms
              </span>
            </div>
            <p className="mt-1 text-[10px] text-[var(--muted-foreground)]">
              {t("step4.tpotHelp")}
            </p>
          </div>

          {/* RHOAI Version */}
          <div className="sm:col-span-2">
            <label className="mb-1.5 block text-sm font-medium">
              {t("step4.rhoaiVersion")}
            </label>
            <div className="flex gap-3">
              {RHOAI_VERSIONS.map((v) => (
                <button
                  key={v.version}
                  onClick={() => setRhoaiVersion(v.version)}
                  className={cn(
                    "flex flex-col items-center rounded-lg border px-6 py-3 transition-all",
                    rhoaiVersion === v.version
                      ? "border-[var(--primary)] bg-[var(--primary)] text-[var(--primary-foreground)]"
                      : "border-[var(--border)] hover:border-[var(--primary)]/50 hover:bg-[var(--muted)]"
                  )}
                >
                  <span className="text-sm font-bold">{v.label}</span>
                  <span className={cn(
                    "text-[10px]",
                    rhoaiVersion === v.version ? "text-[var(--primary-foreground)]/70" : "text-[var(--muted-foreground)]"
                  )}>
                    vLLM {v.vllm}
                  </span>
                </button>
              ))}
            </div>
            <p className="mt-1.5 text-[10px] text-[var(--muted-foreground)]">
              {t("step4.rhoaiVersionHelp", { version: selectedRhoai.version, vllm: selectedRhoai.vllm })}
            </p>
          </div>
        </div>
      </div>

      {/* Summary */}
      <div className="rounded-lg border border-dashed border-[var(--primary)]/30 bg-[var(--accent)] p-4">
        <h4 className="mb-2 text-xs font-medium text-[var(--muted-foreground)] uppercase tracking-wide">
          {t("step4.configSummary")}
        </h4>
        <div className="grid grid-cols-2 gap-x-6 gap-y-1 text-xs sm:grid-cols-3">
          <div>
            <span className="text-[var(--muted-foreground)]">Use Case:</span>{" "}
            <span className="font-medium">
              {selectedPresets.length > 0
                ? selectedPresets.map((p) => PRESETS.find((x) => x.id === p)?.label).join(" + ")
                : "—"}
            </span>
          </div>
          <div>
            <span className="text-[var(--muted-foreground)]">Users / Concurrent:</span>{" "}
            <span className="font-medium">{targetUsers} / {concurrency}</span>
          </div>
          <div>
            <span className="text-[var(--muted-foreground)]">TTFT / TPOT:</span>{" "}
            <span className="font-medium">{ttftMs}ms / {tpotMs}ms</span>
          </div>
          <div>
            <span className="text-[var(--muted-foreground)]">Platform:</span>{" "}
            <span className="font-medium">RHOAI {selectedRhoai.version} (vLLM {selectedRhoai.vllm})</span>
          </div>
          <div>
            <span className="text-[var(--muted-foreground)]">GPU:</span>{" "}
            <span className="font-medium">{gpuCount}× {gpu || "—"}</span>
          </div>
        </div>
      </div>

      {/* Submit */}
      <div className="flex justify-end">
        <button
          onClick={handleSubmit}
          disabled={selectedPresets.length === 0}
          className={cn(
            "flex items-center gap-2 rounded-lg px-6 py-3 text-sm font-medium transition-opacity",
            selectedPresets.length > 0
              ? "bg-[var(--primary)] text-[var(--primary-foreground)] hover:opacity-90"
              : "cursor-not-allowed bg-[var(--muted)] text-[var(--muted-foreground)]"
          )}
        >
          <Send className="h-4 w-4" />
          {t("step4.generateRecommendation")}
        </button>
      </div>
    </div>
  );
}
