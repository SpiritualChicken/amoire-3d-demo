"use client";

import { useCallback, useRef, useState } from "react";

const ACCEPTED_TYPES = ["image/jpeg", "image/png", "image/webp"];
const MAX_SIZE_MB = 20;
const MAX_SIZE_BYTES = MAX_SIZE_MB * 1024 * 1024;

interface ImageUploadProps {
  onGenerate: (file: File) => void;
  disabled?: boolean;
}

export default function ImageUpload({ onGenerate, disabled }: ImageUploadProps) {
  const [preview, setPreview] = useState<string | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const validateAndSet = useCallback((f: File) => {
    setError(null);

    if (!ACCEPTED_TYPES.includes(f.type)) {
      setError("Please upload a JPG, PNG, or WebP image.");
      return;
    }
    if (f.size > MAX_SIZE_BYTES) {
      setError(`File too large. Maximum size is ${MAX_SIZE_MB}MB.`);
      return;
    }

    setFile(f);
    const url = URL.createObjectURL(f);
    setPreview(url);
  }, []);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragOver(false);
      const f = e.dataTransfer.files[0];
      if (f) validateAndSet(f);
    },
    [validateAndSet],
  );

  const handleFileChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const f = e.target.files?.[0];
      if (f) validateAndSet(f);
    },
    [validateAndSet],
  );

  const handleClear = useCallback(() => {
    setFile(null);
    setPreview(null);
    setError(null);
    if (inputRef.current) inputRef.current.value = "";
  }, []);

  return (
    <div className="flex flex-col gap-4">
      {/* Drop zone */}
      <div
        onDragOver={(e) => {
          e.preventDefault();
          setDragOver(true);
        }}
        onDragLeave={() => setDragOver(false)}
        onDrop={handleDrop}
        onClick={() => inputRef.current?.click()}
        className={`
          relative flex cursor-pointer flex-col items-center justify-center
          rounded-xl border-2 border-dashed transition-all duration-200
          ${preview ? "aspect-[3/4] overflow-hidden p-0" : "min-h-[240px] p-8"}
          ${dragOver ? "border-neutral-900 bg-neutral-50" : "border-neutral-300 hover:border-neutral-400 hover:bg-neutral-50/50"}
          ${disabled ? "pointer-events-none opacity-50" : ""}
        `}
      >
        {preview ? (
          <>
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src={preview}
              alt="Uploaded garment"
              className="h-full w-full object-cover"
            />
            <button
              onClick={(e) => {
                e.stopPropagation();
                handleClear();
              }}
              className="absolute right-2 top-2 flex h-7 w-7 items-center justify-center rounded-full bg-white/80 text-sm text-neutral-600 shadow-sm backdrop-blur-sm transition-colors hover:bg-white"
            >
              &times;
            </button>
          </>
        ) : (
          <>
            <div className="mb-3 text-3xl text-neutral-300">
              <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                <polyline points="17 8 12 3 7 8" />
                <line x1="12" y1="3" x2="12" y2="15" />
              </svg>
            </div>
            <p className="text-sm font-medium text-neutral-700">
              Drop a garment image here
            </p>
            <p className="mt-1 text-xs text-neutral-400">
              or click to browse &middot; JPG, PNG, WebP &middot; max {MAX_SIZE_MB}MB
            </p>
          </>
        )}

        <input
          ref={inputRef}
          type="file"
          accept=".jpg,.jpeg,.png,.webp"
          capture="environment"
          onChange={handleFileChange}
          className="hidden"
        />
      </div>

      {/* Error */}
      {error && (
        <p className="text-sm text-red-500">{error}</p>
      )}

      {/* Generate button */}
      {file && (
        <button
          onClick={() => onGenerate(file)}
          disabled={disabled}
          className="
            w-full rounded-lg bg-neutral-900 px-6 py-3 text-sm font-medium text-white
            transition-all duration-200 hover:bg-neutral-800 active:scale-[0.98]
            disabled:cursor-not-allowed disabled:opacity-50
          "
        >
          Generate 3D Garment
        </button>
      )}
    </div>
  );
}
