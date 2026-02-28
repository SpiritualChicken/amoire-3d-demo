"use client";

import { type GalleryItem } from "@/lib/api";
import { DEMO_GARMENTS, type DemoGarment } from "@/lib/demo-data";

interface GarmentGalleryProps {
  items: GalleryItem[];
  demoMode: boolean;
  onSelect: (meshUrl: string) => void;
  activeId?: string | null;
}

function DemoCard({
  garment,
  active,
  onClick,
}: {
  garment: DemoGarment;
  active: boolean;
  onClick: () => void;
}) {
  return (
    <button
      onClick={onClick}
      className={`
        group flex flex-shrink-0 flex-col overflow-hidden rounded-lg border transition-all duration-200
        ${active ? "border-neutral-900 shadow-md" : "border-neutral-200 hover:border-neutral-400 hover:shadow-sm"}
      `}
      style={{ width: 120 }}
    >
      <div className="flex h-[100px] items-center justify-center bg-neutral-100 p-2">
        {/* Placeholder icon for demo items */}
        <div className="text-3xl text-neutral-300">
          {garment.type === "tshirt" && (
            <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
              <path d="M8 2l-5 5 3 2v11h12V9l3-2-5-5-3 2h-2L8 2z" />
            </svg>
          )}
          {garment.type === "skirt" && (
            <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
              <path d="M8 4h8l2 16H6L8 4z" />
            </svg>
          )}
          {garment.type === "pants" && (
            <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
              <path d="M6 4h12v6l-2 10h-3l-1-10-1 10H8L6 10V4z" />
            </svg>
          )}
          {garment.type === "dress" && (
            <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
              <path d="M8 2l-1 4-3 2 2 2-2 10h16L18 8l2-2-3-2-1-4H8z" />
            </svg>
          )}
        </div>
      </div>
      <div className="px-2 py-1.5">
        <p className="truncate text-xs font-medium text-neutral-700">{garment.name}</p>
        <p className="text-[10px] text-neutral-400">{garment.type}</p>
      </div>
    </button>
  );
}

function ApiCard({
  item,
  active,
  onClick,
}: {
  item: GalleryItem;
  active: boolean;
  onClick: () => void;
}) {
  return (
    <button
      onClick={onClick}
      className={`
        group flex flex-shrink-0 flex-col overflow-hidden rounded-lg border transition-all duration-200
        ${active ? "border-neutral-900 shadow-md" : "border-neutral-200 hover:border-neutral-400 hover:shadow-sm"}
      `}
      style={{ width: 120 }}
    >
      <div className="flex h-[100px] items-center justify-center overflow-hidden bg-neutral-100">
        {item.thumbnail_url ? (
          /* eslint-disable-next-line @next/next/no-img-element */
          <img
            src={item.thumbnail_url}
            alt={item.garment_type}
            className="h-full w-full object-cover"
          />
        ) : (
          <div className="text-3xl text-neutral-300">?</div>
        )}
      </div>
      <div className="px-2 py-1.5">
        <p className="truncate text-xs font-medium capitalize text-neutral-700">
          {item.garment_type}
        </p>
        {item.processing_time && (
          <p className="text-[10px] text-neutral-400">
            {item.processing_time.toFixed(1)}s
          </p>
        )}
      </div>
    </button>
  );
}

export default function GarmentGallery({
  items,
  demoMode,
  onSelect,
  activeId,
}: GarmentGalleryProps) {
  const showDemo = demoMode || items.length === 0;

  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-center justify-between px-1">
        <h3 className="text-xs font-medium uppercase tracking-wider text-neutral-400">
          {showDemo ? "Try these examples" : "Previously generated"}
        </h3>
        {demoMode && (
          <span className="rounded bg-amber-50 px-1.5 py-0.5 text-[10px] font-medium text-amber-600">
            Demo mode
          </span>
        )}
      </div>

      <div className="flex gap-3 overflow-x-auto pb-2 scrollbar-thin">
        {showDemo
          ? DEMO_GARMENTS.map((g) => (
              <DemoCard
                key={g.id}
                garment={g}
                active={activeId === g.id}
                onClick={() => onSelect(g.meshUrl)}
              />
            ))
          : items.map((item) => (
              <ApiCard
                key={item.id}
                item={item}
                active={activeId === item.id}
                onClick={() => onSelect(item.mesh_url)}
              />
            ))}
      </div>
    </div>
  );
}
