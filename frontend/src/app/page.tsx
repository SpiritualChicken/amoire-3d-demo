"use client";

import { useCallback, useEffect, useState } from "react";
import GarmentViewer from "@/components/GarmentViewer";
import ImageUpload from "@/components/ImageUpload";
import GenerationStatus from "@/components/GenerationStatus";
import GarmentGallery from "@/components/GarmentGallery";
import {
  checkHealth,
  generateGarment,
  getGallery,
  getMesh,
  type GalleryItem,
} from "@/lib/api";
import { DEMO_GARMENTS } from "@/lib/demo-data";

export default function Home() {
  const [demoMode, setDemoMode] = useState(true);
  const [meshUrl, setMeshUrl] = useState<string | null>(null);
  const [jobId, setJobId] = useState<string | null>(null);
  const [generating, setGenerating] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [gallery, setGallery] = useState<GalleryItem[]>([]);
  const [activeGalleryId, setActiveGalleryId] = useState<string | null>(null);
  const [backendStatus, setBackendStatus] = useState<string | null>(null);

  // Check backend health on mount
  useEffect(() => {
    checkHealth()
      .then((health) => {
        setDemoMode(false);
        setBackendStatus(
          health.mock_mode ? "mock" : health.model_loaded ? "ready" : "loading",
        );
        // Fetch gallery
        getGallery()
          .then(setGallery)
          .catch(() => {});
      })
      .catch(() => {
        setDemoMode(true);
        setBackendStatus(null);
      });
  }, []);

  // Handle image upload → start generation
  const handleGenerate = useCallback(
    async (file: File) => {
      setError(null);
      setGenerating(true);
      setMeshUrl(null);
      setActiveGalleryId(null);

      if (demoMode) {
        // Demo mode: fake delay then load random sample
        setTimeout(() => {
          const sample =
            DEMO_GARMENTS[Math.floor(Math.random() * DEMO_GARMENTS.length)];
          setMeshUrl(sample.meshUrl);
          setActiveGalleryId(sample.id);
          setGenerating(false);
        }, 3000);
        return;
      }

      try {
        const result = await generateGarment(file);
        if (result.cache_hit && result.status === "complete") {
          // Immediate result from cache
          const blob = await getMesh(result.job_id);
          const url = URL.createObjectURL(blob);
          setMeshUrl(url);
          setGenerating(false);
        } else {
          setJobId(result.job_id);
        }
      } catch (err) {
        setError(err instanceof Error ? err.message : "Failed to start generation");
        setGenerating(false);
      }
    },
    [demoMode],
  );

  // Handle job completion
  const handleJobComplete = useCallback(async (completedJobId: string) => {
    try {
      const blob = await getMesh(completedJobId);
      const url = URL.createObjectURL(blob);
      setMeshUrl(url);
      setJobId(null);
      setGenerating(false);

      // Refresh gallery
      getGallery()
        .then(setGallery)
        .catch(() => {});
    } catch {
      setError("Failed to load generated mesh");
      setGenerating(false);
    }
  }, []);

  // Handle job error
  const handleJobError = useCallback((errorMsg: string) => {
    setError(errorMsg);
    setJobId(null);
    setGenerating(false);
  }, []);

  // Handle gallery item selection
  const handleGallerySelect = useCallback(
    (url: string) => {
      setError(null);

      // If it's a demo mesh URL (local path), use directly
      if (url.startsWith("/")) {
        setMeshUrl(url);
        const demoItem = DEMO_GARMENTS.find((g) => g.meshUrl === url);
        setActiveGalleryId(demoItem?.id ?? null);
        return;
      }

      // API mesh URL — fetch the blob
      setGenerating(true);
      fetch(url)
        .then((res) => res.blob())
        .then((blob) => {
          setMeshUrl(URL.createObjectURL(blob));
          setGenerating(false);
        })
        .catch(() => {
          setError("Failed to load mesh");
          setGenerating(false);
        });
    },
    [],
  );

  return (
    <div className="flex min-h-screen flex-col">
      {/* Header */}
      <header className="flex items-center justify-between border-b border-neutral-100 px-6 py-4">
        <div className="flex items-baseline gap-3">
          <h1 className="text-lg font-semibold tracking-tight text-neutral-900">
            Amoire 3D Canvas
          </h1>
          {backendStatus && (
            <span
              className={`rounded px-1.5 py-0.5 text-[10px] font-medium ${
                backendStatus === "ready"
                  ? "bg-green-50 text-green-600"
                  : backendStatus === "mock"
                    ? "bg-blue-50 text-blue-600"
                    : "bg-amber-50 text-amber-600"
              }`}
            >
              {backendStatus === "ready"
                ? "GPU ready"
                : backendStatus === "mock"
                  ? "Mock mode"
                  : "Model loading..."}
            </span>
          )}
        </div>
        {demoMode && (
          <span className="text-xs text-neutral-400">
            Running in demo mode
          </span>
        )}
      </header>

      {/* Main content */}
      <main className="flex flex-1 flex-col lg:flex-row">
        {/* Left panel */}
        <aside className="flex w-full flex-col gap-6 border-b border-neutral-100 p-6 lg:w-[340px] lg:border-b-0 lg:border-r">
          <ImageUpload onGenerate={handleGenerate} disabled={generating} />

          {jobId && (
            <GenerationStatus
              jobId={jobId}
              onComplete={handleJobComplete}
              onError={handleJobError}
              onCancel={() => {
                setJobId(null);
                setGenerating(false);
              }}
            />
          )}

          {error && !jobId && (
            <div className="animate-fade-in rounded-xl bg-red-50 p-4">
              <p className="text-sm text-red-600">{error}</p>
              <button
                onClick={() => setError(null)}
                className="mt-2 text-xs text-red-400 hover:text-red-600"
              >
                Dismiss
              </button>
            </div>
          )}

          {/* Info section */}
          <div className="mt-auto hidden text-xs text-neutral-400 lg:block">
            <p>
              Upload a garment image to generate an interactive 3D model using
              AI-powered sewing pattern simulation.
            </p>
            <p className="mt-2">
              Orbit: drag &middot; Zoom: scroll &middot; Reset: double-click
            </p>
          </div>
        </aside>

        {/* Right panel — 3D Viewer */}
        <section className="flex flex-1 flex-col" style={{ minHeight: 400 }}>
          <div className="flex-1">
            <GarmentViewer
              meshUrl={meshUrl}
              loading={generating}
              error={error}
              onRetry={() => setError(null)}
            />
          </div>
        </section>
      </main>

      {/* Gallery */}
      <footer className="border-t border-neutral-100 p-4">
        <GarmentGallery
          items={gallery}
          demoMode={demoMode}
          onSelect={handleGallerySelect}
          activeId={activeGalleryId}
        />
      </footer>
    </div>
  );
}
