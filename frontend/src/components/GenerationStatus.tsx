"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { checkStatus, type JobStatus } from "@/lib/api";

const STEP_LABELS: Record<string, string> = {
  queued: "Waiting to start...",
  analyzing: "Analyzing garment...",
  sewing_patterns: "Generating sewing patterns...",
  draping: "Simulating 3D draping...",
  converting: "Preparing 3D model...",
  done: "Complete!",
  error: "Generation failed",
};

const STEP_ORDER = ["queued", "analyzing", "sewing_patterns", "draping", "converting", "done"];

interface GenerationStatusProps {
  jobId: string | null;
  onComplete: (jobId: string) => void;
  onError: (error: string) => void;
  onCancel: () => void;
}

export default function GenerationStatus({
  jobId,
  onComplete,
  onError,
  onCancel,
}: GenerationStatusProps) {
  const [status, setStatus] = useState<JobStatus | null>(null);
  const intervalRef = useRef<ReturnType<typeof setInterval>>(undefined);

  const poll = useCallback(async () => {
    if (!jobId) return;
    try {
      const s = await checkStatus(jobId);
      setStatus(s);
      if (s.status === "complete") {
        clearInterval(intervalRef.current);
        onComplete(jobId);
      } else if (s.status === "failed") {
        clearInterval(intervalRef.current);
        onError(s.error || "Unknown error");
      }
    } catch {
      // Network error — keep polling
    }
  }, [jobId, onComplete, onError]);

  useEffect(() => {
    if (!jobId) {
      setStatus(null);
      return;
    }

    poll();
    intervalRef.current = setInterval(poll, 2000);
    return () => clearInterval(intervalRef.current);
  }, [jobId, poll]);

  if (!jobId || !status) return null;

  const currentStep = status.step || "queued";
  const stepIndex = STEP_ORDER.indexOf(currentStep);
  const progress = status.progress || (stepIndex >= 0 ? (stepIndex / (STEP_ORDER.length - 1)) * 100 : 0);

  return (
    <div className="flex flex-col gap-3 rounded-xl bg-neutral-50 p-4">
      {/* Step label */}
      <div className="flex items-center justify-between">
        <p className="text-sm font-medium text-neutral-700">
          {STEP_LABELS[currentStep] || currentStep}
        </p>
        <span className="text-xs tabular-nums text-neutral-400">
          {Math.round(progress)}%
        </span>
      </div>

      {/* Progress bar */}
      <div className="h-1.5 w-full overflow-hidden rounded-full bg-neutral-200">
        <div
          className="h-full rounded-full bg-neutral-900 transition-all duration-500 ease-out"
          style={{ width: `${progress}%` }}
        />
      </div>

      {/* Step indicators */}
      <div className="flex gap-1">
        {STEP_ORDER.slice(0, -1).map((step, i) => (
          <div
            key={step}
            className={`h-1 flex-1 rounded-full transition-colors duration-300 ${
              i <= stepIndex ? "bg-neutral-900" : "bg-neutral-200"
            }`}
          />
        ))}
      </div>

      {/* Cancel */}
      {status.status !== "complete" && status.status !== "failed" && (
        <button
          onClick={onCancel}
          className="mt-1 self-start text-xs text-neutral-400 transition-colors hover:text-neutral-600"
        >
          Cancel
        </button>
      )}

      {/* Error */}
      {status.status === "failed" && (
        <p className="text-sm text-red-500">{status.error}</p>
      )}
    </div>
  );
}
