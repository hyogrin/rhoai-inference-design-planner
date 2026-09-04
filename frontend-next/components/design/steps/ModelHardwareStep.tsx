"use client";

import { useEffect } from "react";
import { cn } from "@/lib/utils";
import { useI18n } from "@/lib/i18n";
import {
  Cpu,
  Server,
  Cloud,
  Building2,
  Monitor,
  ChevronRight,
  Link2,
  Unlink,
} from "lucide-react";

const PLATFORMS = [
  {
    id: "on-premise",
    label: "On-Premise",
    description: "Self-managed OpenShift cluster",
    icon: Building2,
  },
  {
    id: "aws",
    label: "AWS",
    description: "ROSA / self-managed on EC2",
    icon: Cloud,
  },
  {
    id: "azure",
    label: "Azure",
    description: "ARO / self-managed on Azure",
    icon: Monitor,
  },
  {
    id: "gcp",
    label: "GCP",
    description: "Self-managed on GCE",
    icon: Cloud,
  },
] as const;

type PlatformId = (typeof PLATFORMS)[number]["id"];

const GPU_OPTIONS: Record<
  string,
  { id: string; label: string; memory: string; tier: string }[]
> = {
  "on-premise": [
    { id: "B300-288GB", label: "NVIDIA B300", memory: "288 GB HBM3e", tier: "flagship" },
    { id: "GB300-288GB", label: "NVIDIA GB300 (NVL72)", memory: "288 GB HBM3e/GPU", tier: "flagship" },
    { id: "B200-192GB", label: "NVIDIA B200", memory: "192 GB HBM3e", tier: "flagship" },
    { id: "GB200-192GB", label: "NVIDIA GB200 (NVL72)", memory: "192 GB HBM3e/GPU", tier: "flagship" },
    { id: "H200-141GB", label: "NVIDIA H200", memory: "141 GB HBM3e", tier: "flagship" },
    { id: "H100-80GB", label: "NVIDIA H100", memory: "80 GB HBM3", tier: "flagship" },
    { id: "MI300X-192GB", label: "AMD MI300X", memory: "192 GB HBM3", tier: "flagship" },
    { id: "A100-80GB", label: "NVIDIA A100 (80GB)", memory: "80 GB HBM2e", tier: "high" },
    { id: "A100-40GB", label: "NVIDIA A100 (40GB)", memory: "40 GB HBM2e", tier: "high" },
    { id: "RTX-PRO-6000-96GB", label: "NVIDIA RTX PRO 6000", memory: "96 GB GDDR7", tier: "high" },
    { id: "L40S-48GB", label: "NVIDIA L40S", memory: "48 GB GDDR6X", tier: "mid" },
    { id: "L4-24GB", label: "NVIDIA L4", memory: "24 GB GDDR6", tier: "mid" },
    { id: "T4-16GB", label: "NVIDIA T4", memory: "16 GB GDDR6", tier: "entry" },
  ],
  aws: [
    { id: "B300-288GB", label: "NVIDIA B300", memory: "288 GB HBM3e", tier: "flagship" },
    { id: "B200-192GB", label: "NVIDIA B200", memory: "192 GB HBM3e", tier: "flagship" },
    { id: "GB200-192GB", label: "NVIDIA GB200 (NVL72)", memory: "192 GB HBM3e/GPU", tier: "flagship" },
    { id: "H200-141GB", label: "NVIDIA H200", memory: "141 GB HBM3e", tier: "flagship" },
    { id: "H100-80GB", label: "NVIDIA H100", memory: "80 GB HBM3", tier: "flagship" },
    { id: "A100-80GB", label: "NVIDIA A100 (80GB)", memory: "80 GB HBM2e", tier: "high" },
    { id: "RTX-PRO-6000-96GB", label: "NVIDIA RTX PRO 6000", memory: "96 GB GDDR7", tier: "high" },
    { id: "A10G-24GB", label: "NVIDIA A10G", memory: "24 GB GDDR6X", tier: "mid" },
    { id: "L4-24GB", label: "NVIDIA L4", memory: "24 GB GDDR6", tier: "mid" },
    { id: "T4-16GB", label: "NVIDIA T4", memory: "16 GB GDDR6", tier: "entry" },
  ],
  azure: [
    { id: "B300-288GB", label: "NVIDIA B300", memory: "288 GB HBM3e", tier: "flagship" },
    { id: "B200-192GB", label: "NVIDIA B200", memory: "192 GB HBM3e", tier: "flagship" },
    { id: "H200-141GB", label: "NVIDIA H200", memory: "141 GB HBM3e", tier: "flagship" },
    { id: "H100-80GB", label: "NVIDIA H100", memory: "80 GB HBM3", tier: "flagship" },
    { id: "A100-80GB", label: "NVIDIA A100 (80GB)", memory: "80 GB HBM2e", tier: "high" },
    { id: "RTX-PRO-6000-96GB", label: "NVIDIA RTX PRO 6000", memory: "96 GB GDDR7", tier: "high" },
    { id: "T4-16GB", label: "NVIDIA T4", memory: "16 GB GDDR6", tier: "entry" },
  ],
  gcp: [
    { id: "B300-288GB", label: "NVIDIA B300", memory: "288 GB HBM3e", tier: "flagship" },
    { id: "B200-192GB", label: "NVIDIA B200", memory: "192 GB HBM3e", tier: "flagship" },
    { id: "H200-141GB", label: "NVIDIA H200", memory: "141 GB HBM3e", tier: "flagship" },
    { id: "H100-80GB", label: "NVIDIA H100", memory: "80 GB HBM3", tier: "flagship" },
    { id: "A100-80GB", label: "NVIDIA A100 (80GB)", memory: "80 GB HBM2e", tier: "high" },
    { id: "RTX-PRO-6000-96GB", label: "NVIDIA RTX PRO 6000", memory: "96 GB GDDR7", tier: "high" },
    { id: "L4-24GB", label: "NVIDIA L4", memory: "24 GB GDDR6", tier: "mid" },
    { id: "T4-16GB", label: "NVIDIA T4", memory: "16 GB GDDR6", tier: "entry" },
  ],
};

// Cloud instances: platform → gpu_id → available instance SKUs
interface CloudInstanceOption {
  id: string;
  label: string;
  gpu_count: number;
  note: string;
  hourly: number;
}

const CLOUD_INSTANCES: Record<string, Record<string, CloudInstanceOption[]>> = {
  aws: {
    "B300-288GB": [
      { id: "p6-b300.48xlarge", label: "p6-b300.48xlarge", gpu_count: 8, note: "8× B300 SXM, NVLink 5", hourly: 142.42 },
    ],
    "B200-192GB": [
      { id: "p6-b200.48xlarge", label: "p6-b200.48xlarge", gpu_count: 8, note: "8× B200 SXM, NVLink 5", hourly: 113.93 },
    ],
    "GB200-192GB": [
      { id: "p6e-gb200.36xlarge", label: "p6e-gb200.36xlarge", gpu_count: 4, note: "4× GB200 (NVL72 node)", hourly: 49.42 },
    ],
    "H200-141GB": [
      { id: "p5e.48xlarge", label: "p5e.48xlarge", gpu_count: 8, note: "8× H200 SXM", hourly: 120.00 },
    ],
    "H100-80GB": [
      { id: "p5.48xlarge", label: "p5.48xlarge", gpu_count: 8, note: "8× H100 SXM", hourly: 98.32 },
    ],
    "A100-80GB": [
      { id: "p4d.24xlarge", label: "p4d.24xlarge", gpu_count: 8, note: "8× A100 80GB", hourly: 32.77 },
      { id: "p4de.24xlarge", label: "p4de.24xlarge", gpu_count: 8, note: "8× A100 80GB (UltraClusters)", hourly: 40.96 },
    ],
    "RTX-PRO-6000-96GB": [
      { id: "g7e.2xlarge", label: "g7e.2xlarge", gpu_count: 1, note: "1× RTX PRO 6000", hourly: 3.36 },
      { id: "g7e.12xlarge", label: "g7e.12xlarge", gpu_count: 2, note: "2× RTX PRO 6000", hourly: 8.29 },
      { id: "g7e.24xlarge", label: "g7e.24xlarge", gpu_count: 4, note: "4× RTX PRO 6000", hourly: 16.57 },
      { id: "g7e.48xlarge", label: "g7e.48xlarge", gpu_count: 8, note: "8× RTX PRO 6000", hourly: 33.14 },
    ],
    "A10G-24GB": [
      { id: "g5.xlarge", label: "g5.xlarge", gpu_count: 1, note: "1× A10G", hourly: 1.01 },
      { id: "g5.12xlarge", label: "g5.12xlarge", gpu_count: 4, note: "4× A10G", hourly: 5.67 },
      { id: "g5.48xlarge", label: "g5.48xlarge", gpu_count: 8, note: "8× A10G", hourly: 16.29 },
    ],
    "L4-24GB": [
      { id: "g6.xlarge", label: "g6.xlarge", gpu_count: 1, note: "1× L4", hourly: 0.80 },
      { id: "g6.48xlarge", label: "g6.48xlarge", gpu_count: 8, note: "8× L4", hourly: 13.35 },
    ],
    "T4-16GB": [
      { id: "g4dn.xlarge", label: "g4dn.xlarge", gpu_count: 1, note: "1× T4", hourly: 0.53 },
      { id: "g4dn.12xlarge", label: "g4dn.12xlarge", gpu_count: 4, note: "4× T4", hourly: 3.91 },
    ],
  },
  azure: {
    "B300-288GB": [
      { id: "Standard_ND128isr_GB300_v6", label: "ND128isr GB300 v6", gpu_count: 4, note: "4× B300 (NVL72 node)", hourly: 112.48 },
    ],
    "B200-192GB": [
      { id: "Standard_ND128isr_NDR_GB200_v6", label: "ND128isr GB200 v6", gpu_count: 4, note: "4× B200 (NVL72 node)", hourly: 108.16 },
    ],
    "H200-141GB": [
      { id: "Standard_ND96isr_H200_v5", label: "ND96isr H200 v5", gpu_count: 8, note: "8× H200 SXM", hourly: 118.50 },
    ],
    "H100-80GB": [
      { id: "Standard_ND96isr_H100_v5", label: "ND96isr H100 v5", gpu_count: 8, note: "8× H100 SXM", hourly: 98.32 },
    ],
    "A100-80GB": [
      { id: "Standard_NC24ads_A100_v4", label: "NC24ads A100 v4", gpu_count: 1, note: "1× A100 80GB", hourly: 3.67 },
      { id: "Standard_ND96amsr_A100_v4", label: "ND96amsr A100 v4", gpu_count: 8, note: "8× A100 80GB", hourly: 32.77 },
    ],
    "RTX-PRO-6000-96GB": [
      { id: "Standard_NC144ds_xl_RTXPRO6000BSE_v6", label: "NC144ds RTX PRO v6", gpu_count: 1, note: "1× RTX PRO 6000", hourly: 6.38 },
      { id: "Standard_NC288ds_xl_RTXPRO6000BSE_v6", label: "NC288ds RTX PRO v6", gpu_count: 2, note: "2× RTX PRO 6000", hourly: 11.00 },
    ],
    "T4-16GB": [
      { id: "Standard_NC4as_T4_v3", label: "NC4as T4 v3", gpu_count: 1, note: "1× T4", hourly: 0.53 },
      { id: "Standard_NC64as_T4_v3", label: "NC64as T4 v3", gpu_count: 4, note: "4× T4", hourly: 4.35 },
    ],
  },
  gcp: {
    "B300-288GB": [
      { id: "a4x-maxgpu-4g-metal", label: "a4x-maxgpu-4g-metal", gpu_count: 4, note: "4× B300 (NVL72 node)", hourly: 68.00 },
    ],
    "B200-192GB": [
      { id: "a4x-highgpu-4g", label: "a4x-highgpu-4g", gpu_count: 4, note: "4× B200 (NVL72 node)", hourly: 54.00 },
      { id: "a4-highgpu-8g", label: "a4-highgpu-8g", gpu_count: 8, note: "8× B200 HGX", hourly: 128.88 },
    ],
    "H200-141GB": [
      { id: "a3-ultragpu-8g", label: "a3-ultragpu-8g", gpu_count: 8, note: "8× H200 SXM", hourly: 86.76 },
    ],
    "H100-80GB": [
      { id: "a3-highgpu-8g", label: "a3-highgpu-8g", gpu_count: 8, note: "8× H100 SXM", hourly: 87.83 },
    ],
    "A100-80GB": [
      { id: "a2-highgpu-1g", label: "a2-highgpu-1g", gpu_count: 1, note: "1× A100 80GB", hourly: 5.12 },
      { id: "a2-ultragpu-8g", label: "a2-ultragpu-8g", gpu_count: 8, note: "8× A100 80GB", hourly: 40.97 },
    ],
    "RTX-PRO-6000-96GB": [
      { id: "g4-standard-48", label: "g4-standard-48", gpu_count: 1, note: "1× RTX PRO 6000", hourly: 4.50 },
      { id: "g4-standard-96", label: "g4-standard-96", gpu_count: 2, note: "2× RTX PRO 6000", hourly: 9.00 },
      { id: "g4-standard-192", label: "g4-standard-192", gpu_count: 4, note: "4× RTX PRO 6000", hourly: 18.00 },
      { id: "g4-standard-384", label: "g4-standard-384", gpu_count: 8, note: "8× RTX PRO 6000", hourly: 36.00 },
    ],
    "L4-24GB": [
      { id: "g2-standard-4", label: "g2-standard-4", gpu_count: 1, note: "1× L4", hourly: 0.84 },
      { id: "g2-standard-96", label: "g2-standard-96", gpu_count: 8, note: "8× L4", hourly: 13.07 },
    ],
    "T4-16GB": [
      { id: "n1-standard-4-t4", label: "n1-standard-4-t4", gpu_count: 1, note: "1× T4", hourly: 0.45 },
    ],
  },
};

const GPU_COUNTS = [1, 2, 4, 8, 16, 32, 64, 72] as const;

// NVL72 rack-scale GPUs (pricing is estimated/contact-sales)
const NVL72_GPUS = new Set(["GB200-192GB", "GB300-288GB"]);

// GPUs that support NVLink natively
const NVLINK_GPUS = new Set(["B300-288GB", "GB300-288GB", "B200-192GB", "GB200-192GB", "H200-141GB", "H100-80GB", "A100-80GB", "A100-40GB", "MI300X-192GB"]);
// GPUs typically deployed with InfiniBand in multi-node setups
const IB_GPUS = new Set(["B300-288GB", "GB300-288GB", "B200-192GB", "GB200-192GB", "H200-141GB", "H100-80GB", "A100-80GB", "MI300X-192GB", "RTX-PRO-6000-96GB"]);

const isCloudPlatform = (p: string | null): boolean =>
  p === "aws" || p === "azure" || p === "gcp";

interface ModelHardwareStepProps {
  modelRepoId: string;
  onModelRepoIdChange: (v: string) => void;
  platform: string | null;
  onPlatformChange: (v: string) => void;
  selectedGpu: string | null;
  onGpuChange: (v: string) => void;
  gpuCount: number;
  onGpuCountChange: (v: number) => void;
  nvlink: boolean;
  onNvlinkChange: (v: boolean) => void;
  infiniband: boolean;
  onInfinibandChange: (v: boolean) => void;
  selectedInstance: string | null;
  onInstanceChange: (v: string | null) => void;
}

export function ModelHardwareStep({
  modelRepoId,
  onModelRepoIdChange,
  platform,
  onPlatformChange,
  selectedGpu,
  onGpuChange,
  gpuCount,
  onGpuCountChange,
  nvlink,
  onNvlinkChange,
  infiniband,
  onInfinibandChange,
  selectedInstance,
  onInstanceChange,
}: ModelHardwareStepProps) {
  const { t } = useI18n();
  const gpuOptions = platform ? (GPU_OPTIONS[platform] || []) : [];
  const isCloud = isCloudPlatform(platform);
  const instanceOptions = (isCloud && platform && selectedGpu)
    ? (CLOUD_INSTANCES[platform]?.[selectedGpu] || [])
    : [];

  // Auto-detect interconnect based on GPU type and count
  useEffect(() => {
    if (!selectedGpu) return;
    const supportsNvlink = NVLINK_GPUS.has(selectedGpu) && gpuCount >= 2;
    const supportsIb = IB_GPUS.has(selectedGpu) && gpuCount >= 8;
    onNvlinkChange(supportsNvlink);
    onInfinibandChange(supportsIb);
  }, [selectedGpu, gpuCount]); // eslint-disable-line react-hooks/exhaustive-deps

  // When instance is selected (cloud), lock GPU count
  useEffect(() => {
    if (!isCloud || !selectedInstance) return;
    const inst = instanceOptions.find((i) => i.id === selectedInstance);
    if (inst) onGpuCountChange(inst.gpu_count);
  }, [selectedInstance]); // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-lg font-semibold">{t("step1.title")}</h2>
        <p className="mt-1 text-sm text-[var(--muted-foreground)]">
          {t("step1.description")}
        </p>
      </div>

      {/* Model Selection Card */}
      <div className="rounded-xl border border-[var(--border)] p-6">
        <div className="mb-4 flex items-center gap-2">
          <Cpu className="h-5 w-5 text-[var(--primary)]" />
          <h3 className="font-medium">{t("step1.modelIdentity")}</h3>
        </div>

        <div>
          <label className="mb-1.5 block text-sm font-medium">
            {t("step1.hfModelId")}
          </label>
          <input
            type="text"
            value={modelRepoId}
            onChange={(e) => onModelRepoIdChange(e.target.value)}
            placeholder={t("step1.hfModelIdPlaceholder")}
            className="w-full rounded-lg border border-[var(--border)] bg-[var(--background)] px-4 py-2.5 text-sm placeholder:text-[var(--muted-foreground)] focus:border-[var(--primary)] focus:outline-none focus:ring-2 focus:ring-[var(--primary)]/20"
          />
          <p className="mt-1 text-xs text-[var(--muted-foreground)]">
            {t("step1.hfModelIdHelp")}
          </p>
        </div>
      </div>

      {/* Hardware Pool Card */}
      <div className="rounded-xl border border-[var(--border)] p-6">
        <div className="mb-4 flex items-center gap-2">
          <Server className="h-5 w-5 text-[var(--primary)]" />
          <h3 className="font-medium">{t("step1.hardwarePool")}</h3>
        </div>

        {/* Step 1: Platform Selection */}
        <div className="mb-6">
          <div className="mb-3 flex items-center gap-2">
            <span className="flex h-5 w-5 items-center justify-center rounded-full bg-[var(--primary)] text-[10px] font-bold text-[var(--primary-foreground)]">
              1
            </span>
            <span className="text-sm font-medium">{t("step1.deploymentPlatform")}</span>
          </div>

          <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
            {PLATFORMS.map((p) => (
              <button
                key={p.id}
                onClick={() => {
                  onPlatformChange(p.id);
                  onGpuChange("");
                  onInstanceChange(null);
                }}
                className={cn(
                  "flex flex-col items-center gap-2 rounded-xl border p-4 text-center transition-all",
                  platform === p.id
                    ? "border-[var(--primary)] bg-[var(--accent)] shadow-sm"
                    : "border-[var(--border)] hover:border-[var(--primary)]/50 hover:bg-[var(--muted)]"
                )}
              >
                <p.icon
                  className={cn(
                    "h-6 w-6",
                    platform === p.id
                      ? "text-[var(--primary)]"
                      : "text-[var(--muted-foreground)]"
                  )}
                />
                <span className="text-sm font-medium">{p.label}</span>
                <span className="text-[11px] leading-tight text-[var(--muted-foreground)]">
                  {p.description}
                </span>
              </button>
            ))}
          </div>
        </div>

        {/* Step 2: GPU Selection */}
        {platform && (
          <div className="animate-in fade-in slide-in-from-top-2 duration-300">
            <div className="mb-3 flex items-center gap-2">
              <span className="flex h-5 w-5 items-center justify-center rounded-full bg-[var(--primary)] text-[10px] font-bold text-[var(--primary-foreground)]">
                2
              </span>
              <span className="text-sm font-medium">{t("step1.gpuConfig")}</span>
            </div>

            {/* GPU Type */}
            <div className="mb-4">
              <label className="mb-2 block text-xs font-medium text-[var(--muted-foreground)] uppercase tracking-wide">
                {t("step1.gpuType")}
              </label>
              <div className="space-y-2">
                {gpuOptions.map((gpu) => (
                  <button
                    key={gpu.id}
                    onClick={() => {
                      onGpuChange(gpu.id);
                      onInstanceChange(null);
                    }}
                    className={cn(
                      "flex w-full items-center justify-between rounded-lg border px-4 py-3 text-left transition-all",
                      selectedGpu === gpu.id
                        ? "border-[var(--primary)] bg-[var(--accent)]"
                        : "border-[var(--border)] hover:border-[var(--primary)]/50 hover:bg-[var(--muted)]"
                    )}
                  >
                    <div className="flex items-center gap-3">
                      <div
                        className={cn(
                          "h-2.5 w-2.5 rounded-full",
                          gpu.tier === "flagship"
                            ? "bg-[var(--success)]"
                            : gpu.tier === "high"
                              ? "bg-[var(--primary)]"
                              : "bg-[var(--muted-foreground)]"
                        )}
                      />
                      <div>
                        <span className="text-sm font-medium">{gpu.label}</span>
                        <span className="ml-2 text-xs text-[var(--muted-foreground)]">
                          {gpu.memory}
                        </span>
                      </div>
                    </div>
                    {selectedGpu === gpu.id && (
                      <ChevronRight className="h-4 w-4 text-[var(--primary)]" />
                    )}
                  </button>
                ))}
              </div>
              {selectedGpu && NVL72_GPUS.has(selectedGpu) && (
                <div className="mt-2 flex items-start gap-2 rounded-lg border border-amber-500/30 bg-amber-500/5 px-3 py-2">
                  <span className="mt-0.5 shrink-0 text-amber-500">⚠</span>
                  <p className="text-[10px] leading-relaxed text-amber-700">
                    {t("step1.nvl72Notice")}
                  </p>
                </div>
              )}
            </div>

            {/* Cloud: Instance Picker / On-premise: GPU Count */}
            {selectedGpu && (
              <div className="animate-in fade-in slide-in-from-top-2 duration-200">
                {isCloud && instanceOptions.length > 0 ? (
                  <>
                    <label className="mb-2 block text-xs font-medium text-[var(--muted-foreground)] uppercase tracking-wide">
                      {t("step1.instanceType")}
                    </label>
                    <div className="space-y-2">
                      {instanceOptions.map((inst) => (
                        <button
                          key={inst.id}
                          onClick={() => onInstanceChange(inst.id)}
                          className={cn(
                            "flex w-full items-center justify-between rounded-lg border px-4 py-3 text-left transition-all",
                            selectedInstance === inst.id
                              ? "border-[var(--primary)] bg-[var(--accent)]"
                              : "border-[var(--border)] hover:border-[var(--primary)]/50 hover:bg-[var(--muted)]"
                          )}
                        >
                          <div className="flex items-center gap-3">
                            <div className="h-2.5 w-2.5 rounded-full bg-[var(--primary)]" />
                            <div>
                              <span className="text-sm font-medium">{inst.label}</span>
                              <span className="ml-2 text-xs text-[var(--muted-foreground)]">
                                {inst.note}
                              </span>
                            </div>
                          </div>
                          <span className="text-xs font-medium text-[var(--muted-foreground)]">
                            ${inst.hourly.toFixed(2)}/hr
                          </span>
                        </button>
                      ))}
                    </div>
                    {selectedInstance && (
                      <p className="mt-2 text-xs text-[var(--muted-foreground)]">
                        {t("step1.gpuCountLocked")}: {gpuCount}×{" "}
                        {gpuOptions.find((g) => g.id === selectedGpu)?.memory ?? "—"}
                      </p>
                    )}
                  </>
                ) : (
                  <>
                    <label className="mb-2 block text-xs font-medium text-[var(--muted-foreground)] uppercase tracking-wide">
                      {t("step1.gpuCount")}
                    </label>
                    <div className="flex flex-wrap gap-2">
                      {GPU_COUNTS.map((count) => (
                        <button
                          key={count}
                          onClick={() => onGpuCountChange(count)}
                          className={cn(
                            "flex h-11 w-14 items-center justify-center rounded-lg border text-sm font-semibold transition-all",
                            gpuCount === count
                              ? "border-[var(--primary)] bg-[var(--primary)] text-[var(--primary-foreground)] shadow-sm"
                              : "border-[var(--border)] text-[var(--foreground)] hover:border-[var(--primary)]/50 hover:bg-[var(--muted)]"
                          )}
                        >
                          {count}
                        </button>
                      ))}
                    </div>
                    <p className="mt-2 text-xs text-[var(--muted-foreground)]">
                      {t("step1.totalGpuMemory")}: {gpuCount}×{" "}
                      {gpuOptions.find((g) => g.id === selectedGpu)?.memory ?? "—"}
                    </p>
                  </>
                )}

                {/* Interconnect Options */}
                {gpuCount >= 2 && (
                  <div className="mt-4 rounded-lg border border-[var(--border)] bg-[var(--muted)]/30 p-4">
                    <label className="mb-3 block text-xs font-medium text-[var(--muted-foreground)] uppercase tracking-wide">
                      {t("step1.interconnect")}
                    </label>
                    <div className="flex flex-wrap gap-3">
                      {/* NVLink Toggle */}
                      <button
                        onClick={() => onNvlinkChange(!nvlink)}
                        disabled={!NVLINK_GPUS.has(selectedGpu)}
                        className={cn(
                          "flex items-center gap-2 rounded-lg border px-4 py-2.5 text-sm font-medium transition-all",
                          nvlink
                            ? "border-[var(--primary)] bg-[var(--primary)]/10 text-[var(--primary)]"
                            : "border-[var(--border)] text-[var(--muted-foreground)] hover:border-[var(--primary)]/50",
                          !NVLINK_GPUS.has(selectedGpu) && "cursor-not-allowed opacity-40"
                        )}
                      >
                        {nvlink ? <Link2 className="h-4 w-4" /> : <Unlink className="h-4 w-4" />}
                        NVLink
                        {nvlink && <span className="ml-1 text-[10px] font-bold uppercase">On</span>}
                      </button>

                      {/* InfiniBand Toggle */}
                      <button
                        onClick={() => onInfinibandChange(!infiniband)}
                        disabled={!IB_GPUS.has(selectedGpu)}
                        className={cn(
                          "flex items-center gap-2 rounded-lg border px-4 py-2.5 text-sm font-medium transition-all",
                          infiniband
                            ? "border-[var(--primary)] bg-[var(--primary)]/10 text-[var(--primary)]"
                            : "border-[var(--border)] text-[var(--muted-foreground)] hover:border-[var(--primary)]/50",
                          !IB_GPUS.has(selectedGpu) && "cursor-not-allowed opacity-40"
                        )}
                      >
                        {infiniband ? <Link2 className="h-4 w-4" /> : <Unlink className="h-4 w-4" />}
                        InfiniBand
                        {infiniband && <span className="ml-1 text-[10px] font-bold uppercase">On</span>}
                      </button>
                    </div>
                    <p className="mt-2 text-[10px] text-[var(--muted-foreground)]">
                      {!NVLINK_GPUS.has(selectedGpu)
                        ? "This GPU does not support NVLink"
                        : nvlink && infiniband
                          ? "NVLink for intra-node GPU↔GPU, InfiniBand for inter-node communication"
                          : nvlink
                            ? "NVLink enables high-bandwidth GPU↔GPU tensor parallelism"
                            : gpuCount >= 8
                              ? "Consider enabling InfiniBand for multi-node deployments"
                              : "Enable NVLink for faster tensor parallelism between GPUs"}
                    </p>
                  </div>
                )}
              </div>
            )}
          </div>
        )}

        {/* Summary */}
        {platform && selectedGpu && (isCloud ? selectedInstance : true) && (
          <div className="mt-6 rounded-lg border border-dashed border-[var(--primary)]/30 bg-[var(--accent)] p-4">
            <h4 className="mb-1 text-xs font-medium text-[var(--muted-foreground)] uppercase tracking-wide">
              {t("step1.configSummary")}
            </h4>
            <p className="text-sm font-medium">
              {gpuCount}× {gpuOptions.find((g) => g.id === selectedGpu)?.label} on{" "}
              {PLATFORMS.find((p) => p.id === platform)?.label}
              {isCloud && selectedInstance && (
                <span className="ml-1 text-xs text-[var(--muted-foreground)]">
                  ({instanceOptions.find((i) => i.id === selectedInstance)?.label})
                </span>
              )}
              {(nvlink || infiniband) && (
                <span className="ml-2 text-xs text-[var(--muted-foreground)]">
                  ({[nvlink && "NVLink", infiniband && "InfiniBand"].filter(Boolean).join(" + ")})
                </span>
              )}
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
