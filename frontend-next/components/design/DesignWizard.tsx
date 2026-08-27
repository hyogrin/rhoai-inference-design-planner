"use client";

import { useState } from "react";
import { cn } from "@/lib/utils";
import { ChevronLeft, ChevronRight, Check, Loader2 } from "lucide-react";
import { useAgentStream } from "@/lib/use-agent-stream";
import { ModelHardwareStep } from "./steps/ModelHardwareStep";
import { EvidenceDiscoveryStep } from "./steps/EvidenceDiscoveryStep";
import { ReadinessGateStep } from "./steps/ReadinessGateStep";
import { WorkloadProfileStep } from "./steps/WorkloadProfileStep";
import { RecommendationStep } from "./steps/RecommendationStep";

const STEPS = [
  { id: 1, name: "Model & Hardware" },
  { id: 2, name: "Evidence Discovery" },
  { id: 3, name: "Readiness Gate" },
  { id: 4, name: "Workload Profile" },
  { id: 5, name: "Recommendation" },
] as const;

export function DesignWizard() {
  const [currentStep, setCurrentStep] = useState(1);
  const [modelRepoId, setModelRepoId] = useState("");
  const [platform, setPlatform] = useState<string | null>(null);
  const [selectedGpu, setSelectedGpu] = useState<string | null>(null);
  const [gpuCount, setGpuCount] = useState(1);
  const [nvlink, setNvlink] = useState(false);
  const [infiniband, setInfiniband] = useState(false);

  const { state: agentState, startDiscovery, resumeWithWorkload } =
    useAgentStream();


  const handleStartDiscovery = () => {
    if (!modelRepoId.trim()) return;
    setCurrentStep(2);
    startDiscovery(modelRepoId.trim(), "main", {
      platform: platform || "on-premise",
      gpuType: selectedGpu || "",
      gpuCount,
      nvlink,
      infiniband,
    });
  };

  const handleSubmitWorkload = (workloadData: Record<string, unknown> | import("./steps/WorkloadProfileStep").WorkloadConfig) => {
    resumeWithWorkload(workloadData as Record<string, unknown>);
    setCurrentStep(5);
  };

  const goNext = () => {
    if (currentStep === 1) {
      handleStartDiscovery();
      return;
    }
    setCurrentStep((s) => Math.min(s + 1, 5));
  };

  const goBack = () => setCurrentStep((s) => Math.max(s - 1, 1));

  const isDiscovering = agentState.status === "running" && currentStep === 2;
  const canProceedFromStep2 =
    agentState.status === "completed" && agentState.interrupt !== null;

  return (
    <div className="mx-auto max-w-6xl px-6 py-8">
      {/* Step Indicator */}
      <nav className="mb-8">
        <ol className="flex items-center gap-2">
          {STEPS.map((step, idx) => (
            <li key={step.id} className="flex items-center gap-2">
              <button
                onClick={() => {
                  if (step.id <= currentStep) setCurrentStep(step.id);
                }}
                disabled={step.id > currentStep}
                className={cn(
                  "flex items-center gap-2 rounded-lg px-3 py-2 text-sm font-medium transition-colors",
                  currentStep === step.id
                    ? "bg-[var(--primary)] text-[var(--primary-foreground)]"
                    : currentStep > step.id
                      ? "bg-[var(--accent)] text-[var(--primary)] cursor-pointer"
                      : "bg-[var(--muted)] text-[var(--muted-foreground)] cursor-not-allowed"
                )}
              >
                <span
                  className={cn(
                    "flex h-6 w-6 items-center justify-center rounded-full text-xs font-bold",
                    currentStep > step.id
                      ? "bg-[var(--primary)] text-[var(--primary-foreground)]"
                      : "bg-white/20"
                  )}
                >
                  {currentStep > step.id ? (
                    <Check className="h-3.5 w-3.5" />
                  ) : (
                    step.id
                  )}
                </span>
                <span className="hidden sm:inline">{step.name}</span>
              </button>
              {idx < STEPS.length - 1 && (
                <div
                  className={cn(
                    "h-px w-6 lg:w-10",
                    currentStep > step.id
                      ? "bg-[var(--primary)]"
                      : "bg-[var(--border)]"
                  )}
                />
              )}
            </li>
          ))}
        </ol>
      </nav>

      {/* Step Content */}
      <div className="min-h-[500px]">
        {currentStep === 1 && (
          <ModelHardwareStep
            modelRepoId={modelRepoId}
            onModelRepoIdChange={setModelRepoId}
            platform={platform}
            onPlatformChange={setPlatform}
            selectedGpu={selectedGpu}
            onGpuChange={setSelectedGpu}
            gpuCount={gpuCount}
            onGpuCountChange={setGpuCount}
            nvlink={nvlink}
            onNvlinkChange={setNvlink}
            infiniband={infiniband}
            onInfinibandChange={setInfiniband}
          />
        )}
        {currentStep === 2 && (
          <EvidenceDiscoveryStep agentState={agentState} />
        )}
        {currentStep === 3 && (
          <ReadinessGateStep
            agentState={agentState}
            selectedGpu={selectedGpu}
            gpuCount={gpuCount}
            modelRepoId={modelRepoId}
          />
        )}
        {currentStep === 4 && (
          <WorkloadProfileStep
            interrupt={agentState.interrupt}
            onSubmit={handleSubmitWorkload}
            gpu={selectedGpu}
            gpuCount={gpuCount}
            platform={platform}
          />
        )}
        {currentStep === 5 && <RecommendationStep agentState={agentState} />}
      </div>

      {/* Navigation */}
      <div className="mt-8 flex items-center justify-between border-t border-[var(--border)] pt-6">
        <button
          onClick={goBack}
          disabled={currentStep === 1 || isDiscovering}
          className={cn(
            "flex items-center gap-2 rounded-lg px-4 py-2.5 text-sm font-medium transition-colors",
            currentStep === 1 || isDiscovering
              ? "cursor-not-allowed text-[var(--muted-foreground)] opacity-50"
              : "border border-[var(--border)] hover:bg-[var(--muted)]"
          )}
        >
          <ChevronLeft className="h-4 w-4" />
          Back
        </button>

        <span className="text-sm text-[var(--muted-foreground)]">
          Step {currentStep} of {STEPS.length}
        </span>

        {currentStep >= 4 ? (
          <div className="w-[88px]" />
        ) : (
          <button
            onClick={goNext}
            disabled={
              isDiscovering ||
              (currentStep === 1 && !modelRepoId.trim()) ||
              (currentStep === 2 && !canProceedFromStep2)
            }
            className={cn(
              "flex items-center gap-2 rounded-lg px-4 py-2.5 text-sm font-medium transition-colors",
              isDiscovering ||
                (currentStep === 1 && !modelRepoId.trim()) ||
                (currentStep === 2 && !canProceedFromStep2)
                ? "cursor-not-allowed bg-[var(--muted)] text-[var(--muted-foreground)] opacity-50"
                : "bg-[var(--primary)] text-[var(--primary-foreground)] hover:opacity-90"
            )}
          >
            {isDiscovering ? (
              <>
                <Loader2 className="h-4 w-4 animate-spin" />
                Analyzing...
              </>
            ) : currentStep === 1 ? (
              <>
                Start Analysis
                <ChevronRight className="h-4 w-4" />
              </>
            ) : (
              <>
                Next
                <ChevronRight className="h-4 w-4" />
              </>
            )}
          </button>
        )}
      </div>
    </div>
  );
}
