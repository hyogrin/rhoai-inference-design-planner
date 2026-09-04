"use client";

import { useCallback, useRef, useState } from "react";

export type DiscoveryStatus = "idle" | "running" | "completed" | "error";

export type NodeStatus = "pending" | "running" | "done" | "error";

export interface StepEvent {
  id: string;
  node: string;
  phase: string;
  timestamp: number;
  evidenceCount?: number;
  detail?: string;
  qualityScore?: number;
  sourceUrls?: string[];
  error?: string | null;
}

export interface WorkloadInterrupt {
  type: string;
  model_repo_id: string;
  architecture_summary: {
    type: string;
    family: string | null;
    parameters: number | null;
    max_context: number | null;
  };
  evidence_collected: number;
  required_fields: Array<{
    field: string;
    label: string;
    type: string;
    options?: string[];
    min?: number;
    max?: number;
    default?: number | string;
  }>;
  message: string;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  model_analysis?: Record<string, any>;
}

export interface AgentState {
  status: DiscoveryStatus;
  steps: StepEvent[];
  nodeStatuses: Record<string, NodeStatus>;
  interrupt: WorkloadInterrupt | null;
  error: string | null;
  threadId: string | null;
  totalEvidence: number;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  viewModel: any | null;
}

const AGENT_URL =
  process.env.NEXT_PUBLIC_AGENT_URL || "http://localhost:7001/agent";

const DISCOVERY_NODES = [
  "fetch_huggingface_metadata",
  "discover_vllm_recipe",
  "discover_redhat_evaluations",
  "check_rhoai_compatibility",
  "fetch_pricing",
  "validate_discovery",
  "interpret_model_config",
];

export function useAgentStream() {
  const [state, setState] = useState<AgentState>({
    status: "idle",
    steps: [],
    nodeStatuses: {},
    interrupt: null,
    error: null,
    threadId: null,
    totalEvidence: 0,
    viewModel: null,
  });

  const abortRef = useRef<AbortController | null>(null);

  const startDiscovery = useCallback(
    async (
      modelRepoId: string,
      modelRevision = "main",
      hardwareConfig?: {
        platform: string;
        gpuType: string;
        gpuCount: number;
        nvlink?: boolean;
        infiniband?: boolean;
        instanceType?: string;
      },
      language?: string,
    ) => {
      abortRef.current?.abort();
      const controller = new AbortController();
      abortRef.current = controller;

      const threadId = crypto.randomUUID();
      const runId = crypto.randomUUID();

      setState({
        status: "running",
        steps: [],
        nodeStatuses: Object.fromEntries(
          DISCOVERY_NODES.map((n) => [n, "pending" as NodeStatus])
        ),
        interrupt: null,
        error: null,
        threadId,
        totalEvidence: 0,
        viewModel: null,
      });

      try {
        const response = await fetch(AGENT_URL, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          signal: controller.signal,
          body: JSON.stringify({
            runId,
            threadId,
            messages: [{ role: "user", content: modelRepoId }],
            forwardedProps: {
              model_repo_id: modelRepoId,
              model_revision: modelRevision,
              ...(hardwareConfig && {
                platform: hardwareConfig.platform,
                gpu_type: hardwareConfig.gpuType,
                gpu_count: hardwareConfig.gpuCount,
                nvlink: hardwareConfig.nvlink,
                infiniband: hardwareConfig.infiniband,
                ...(hardwareConfig.instanceType && { instance_type: hardwareConfig.instanceType }),
              }),
              ...(language && { language }),
            },
          }),
        });

        if (!response.ok || !response.body) {
          throw new Error(`HTTP ${response.status}`);
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split("\n");
          buffer = lines.pop() || "";

          for (const line of lines) {
            if (!line.startsWith("data: ")) continue;
            const jsonStr = line.slice(6).trim();
            if (!jsonStr || jsonStr === "[DONE]") continue;

            try {
              const event = JSON.parse(jsonStr);
              handleEvent(event, setState);
            } catch {
              // skip malformed JSON
            }
          }
        }
      } catch (err: unknown) {
        if (err instanceof Error && err.name === "AbortError") return;
        setState((prev) => ({
          ...prev,
          status: "error",
          error: err instanceof Error ? err.message : "Unknown error",
        }));
      }
    },
    []
  );

  const resumeWithWorkload = useCallback(
    async (workloadData: Record<string, unknown>) => {
      if (!state.threadId) return;

      const controller = new AbortController();
      abortRef.current = controller;

      setState((prev) => ({
        ...prev,
        status: "running",
        interrupt: null,
      }));

      try {
        const response = await fetch(AGENT_URL, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          signal: controller.signal,
          body: JSON.stringify({
            runId: crypto.randomUUID(),
            threadId: state.threadId,
            messages: [{ role: "user", content: "workload_config" }],
            state: workloadData,
            forwardedProps: {},
          }),
        });

        if (!response.ok || !response.body) {
          throw new Error(`HTTP ${response.status}`);
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split("\n");
          buffer = lines.pop() || "";

          for (const line of lines) {
            if (!line.startsWith("data: ")) continue;
            const jsonStr = line.slice(6).trim();
            if (!jsonStr || jsonStr === "[DONE]") continue;

            try {
              const event = JSON.parse(jsonStr);
              handleEvent(event, setState);
            } catch {
              // skip
            }
          }
        }
      } catch (err: unknown) {
        if (err instanceof Error && err.name === "AbortError") return;
        setState((prev) => ({
          ...prev,
          status: "error",
          error: err instanceof Error ? err.message : "Unknown error",
        }));
      }
    },
    [state.threadId]
  );

  const stop = useCallback(() => {
    abortRef.current?.abort();
  }, []);

  return { state, startDiscovery, resumeWithWorkload, stop };
}

const NODE_LABELS: Record<string, string> = {
  normalize_intake: "Initialized model intake",
  fetch_huggingface_metadata: "Fetched HuggingFace model metadata",
  discover_vllm_recipe: "Searched vLLM serving recipes",
  discover_redhat_evaluations: "Looked up HuggingFace model card benchmarks",
  check_rhoai_compatibility: "Checked RHOAI validated models matrix",
  fetch_pricing: "Retrieved GPU pricing data",
  validate_discovery: "Validated discovery results",
};

function handleEvent(
  event: Record<string, unknown>,
  setState: React.Dispatch<React.SetStateAction<AgentState>>
) {
  const type = event.type as string;

  switch (type) {
    case "CUSTOM": {
      const name = event.name as string;
      if (name === "step") {
        const step = event.value as StepEvent;
        step.detail = NODE_LABELS[step.node] || step.node;
        // Map quality_score from backend
        if ((event.value as Record<string, unknown>).quality_score !== undefined) {
          step.qualityScore = (event.value as Record<string, unknown>).quality_score as number;
        }
        if ((event.value as Record<string, unknown>).source_urls) {
          step.sourceUrls = (event.value as Record<string, unknown>).source_urls as string[];
        }
        if ((event.value as Record<string, unknown>).error) {
          step.error = (event.value as Record<string, unknown>).error as string;
        }
        setState((prev) => {
          const newSteps = [...prev.steps, step];
          const newStatuses = { ...prev.nodeStatuses };
          newStatuses[step.node] = step.error ? "error" : "done";
          return { ...prev, steps: newSteps, nodeStatuses: newStatuses };
        });
      } else if (name === "workload_interrupt") {
        const interrupt = event.value as WorkloadInterrupt;
        setState((prev) => ({
          ...prev,
          status: "completed",
          interrupt,
          totalEvidence: interrupt.evidence_collected || prev.totalEvidence,
        }));
      } else if (name === "view_model") {
        setState((prev) => ({
          ...prev,
          viewModel: event.value,
        }));
      }
      break;
    }
    case "STATE_SNAPSHOT": {
      const snapshot = event.snapshot as Record<string, unknown> | undefined;
      if (snapshot && typeof snapshot.evidence_count === "number") {
        setState((prev) => {
          const updated = { ...prev, totalEvidence: snapshot.evidence_count as number };
          // Enrich the latest step with evidence count
          if (updated.steps.length > 0) {
            const lastStep = { ...updated.steps[updated.steps.length - 1] };
            lastStep.evidenceCount = snapshot.evidence_count as number;
            updated.steps = [...updated.steps.slice(0, -1), lastStep];
          }
          return updated;
        });
      }
      break;
    }
    case "RUN_FINISHED":
      setState((prev) => ({
        ...prev,
        status: "completed",
      }));
      break;
    case "RUN_ERROR":
      setState((prev) => ({
        ...prev,
        status: "error",
        error: (event.message as string) || "Unknown error",
      }));
      break;
  }
}
