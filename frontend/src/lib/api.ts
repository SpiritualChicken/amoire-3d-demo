const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export interface HealthStatus {
  status: string;
  model_loaded: boolean;
  mock_mode: boolean;
  gpu: { name: string; memory_total_mb: number; memory_used_mb: number } | null;
  active_jobs: number;
}

export interface JobStatus {
  job_id: string;
  status: "queued" | "processing" | "complete" | "failed";
  progress: number;
  step: string;
  error?: string;
  processing_time?: number;
  garment_type?: string;
}

export interface GalleryItem {
  id: string;
  thumbnail_url: string | null;
  mesh_url: string;
  created_at: string;
  garment_type: string;
  processing_time?: number;
}

class ApiError extends Error {
  constructor(
    message: string,
    public status: number,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function request<T>(
  path: string,
  options?: RequestInit,
  retries = 2,
): Promise<T> {
  const url = `${API_BASE}${path}`;
  let lastError: Error | null = null;

  for (let attempt = 0; attempt <= retries; attempt++) {
    try {
      const res = await fetch(url, {
        ...options,
        signal: AbortSignal.timeout(10000),
      });
      if (!res.ok) {
        const body = await res.text();
        throw new ApiError(body || res.statusText, res.status);
      }
      return (await res.json()) as T;
    } catch (err) {
      lastError = err as Error;
      if (err instanceof ApiError && err.status < 500) throw err;
      if (attempt < retries) await new Promise((r) => setTimeout(r, 1000));
    }
  }
  throw lastError!;
}

export async function generateGarment(
  image: File,
): Promise<{ job_id: string; status: string; cache_hit?: boolean }> {
  const form = new FormData();
  form.append("image", image);

  const url = `${API_BASE}/api/generate-3d`;
  const res = await fetch(url, { method: "POST", body: form });

  if (!res.ok) {
    const body = await res.text();
    throw new ApiError(body || res.statusText, res.status);
  }

  return res.json();
}

export async function checkStatus(jobId: string): Promise<JobStatus> {
  return request<JobStatus>(`/api/status/${jobId}`);
}

export async function getMesh(jobId: string): Promise<Blob> {
  const url = `${API_BASE}/api/mesh/${jobId}`;
  const res = await fetch(url);
  if (!res.ok) throw new ApiError("Failed to fetch mesh", res.status);
  return res.blob();
}

export async function getGallery(): Promise<GalleryItem[]> {
  const data = await request<{ items: GalleryItem[] }>("/api/gallery");
  return data.items;
}

export async function checkHealth(): Promise<HealthStatus> {
  return request<HealthStatus>("/api/health", undefined, 0);
}

export function meshUrl(jobId: string): string {
  return `${API_BASE}/api/mesh/${jobId}`;
}

export function thumbnailUrl(path: string): string {
  if (path.startsWith("http")) return path;
  return `${API_BASE}${path}`;
}
