"use client";

import { useCallback, useEffect, useRef, useState } from "react";

type ViewerState = "empty" | "loading" | "loaded" | "error";

interface GarmentViewerProps {
  meshUrl: string | null;
  loading?: boolean;
  error?: string | null;
  onRetry?: () => void;
}

/**
 * Babylon.js 3D garment viewer with PBR materials and studio lighting.
 *
 * Renders a .glb mesh in an interactive scene with orbit camera.
 * All Babylon imports are dynamic to avoid SSR issues.
 */
export default function GarmentViewer({
  meshUrl,
  loading = false,
  error = null,
  onRetry,
}: GarmentViewerProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const engineRef = useRef<InstanceType<typeof import("@babylonjs/core").Engine> | null>(null);
  const sceneRef = useRef<InstanceType<typeof import("@babylonjs/core").Scene> | null>(null);
  const cameraRef = useRef<InstanceType<typeof import("@babylonjs/core").ArcRotateCamera> | null>(null);
  const currentMeshesRef = useRef<InstanceType<typeof import("@babylonjs/core").AbstractMesh>[]>([]);
  const defaultCameraStateRef = useRef<{ alpha: number; beta: number; radius: number; target: import("@babylonjs/core").Vector3 } | null>(null);

  const [viewerState, setViewerState] = useState<ViewerState>("empty");
  const [wireframe, setWireframe] = useState(false);
  const [isFullscreen, setIsFullscreen] = useState(false);

  // Initialize Babylon scene
  useEffect(() => {
    if (!canvasRef.current) return;

    let disposed = false;

    async function init() {
      const BABYLON = await import("@babylonjs/core");
      await import("@babylonjs/loaders/glTF");

      if (disposed || !canvasRef.current) return;

      // Engine
      const engine = new BABYLON.Engine(canvasRef.current, true, {
        preserveDrawingBuffer: true,
        stencil: true,
        antialias: true,
      });
      engineRef.current = engine;

      // Scene
      const scene = new BABYLON.Scene(engine);
      scene.clearColor = new BABYLON.Color4(0.96, 0.96, 0.96, 1); // #f5f5f5
      scene.ambientColor = new BABYLON.Color3(0.15, 0.15, 0.15);
      sceneRef.current = scene;

      // Camera — ArcRotate, 3/4 view from slightly above-right
      const camera = new BABYLON.ArcRotateCamera(
        "camera",
        -Math.PI / 4,       // alpha: slightly right
        Math.PI / 3,        // beta: ~60 degrees from above
        2.5,                // radius
        BABYLON.Vector3.Zero(),
        scene,
      );
      camera.attachControl(canvasRef.current, true);
      camera.lowerRadiusLimit = 0.5;
      camera.upperRadiusLimit = 8;
      camera.lowerBetaLimit = 0.1;
      camera.upperBetaLimit = Math.PI - 0.1;
      camera.inertia = 0.85;
      camera.wheelDeltaPercentage = 0.02;
      camera.panningSensibility = 200;
      camera.pinchDeltaPercentage = 0.01;
      camera.minZ = 0.01;
      cameraRef.current = camera;

      // Save default camera state for reset
      defaultCameraStateRef.current = {
        alpha: camera.alpha,
        beta: camera.beta,
        radius: camera.radius,
        target: camera.target.clone(),
      };

      // Double-click to reset camera
      scene.onPointerObservable.add((pointerInfo) => {
        if (pointerInfo.type === BABYLON.PointerEventTypes.POINTERDOUBLETAP) {
          resetCamera();
        }
      });

      // Lighting — fashion studio style
      // Main key light from upper-left
      const keyLight = new BABYLON.DirectionalLight(
        "keyLight",
        new BABYLON.Vector3(-0.5, -1, 0.5),
        scene,
      );
      keyLight.intensity = 1.8;
      keyLight.diffuse = new BABYLON.Color3(1, 0.98, 0.95); // warm white

      // Fill light from right — softer
      const fillLight = new BABYLON.DirectionalLight(
        "fillLight",
        new BABYLON.Vector3(0.5, -0.5, -0.3),
        scene,
      );
      fillLight.intensity = 0.6;
      fillLight.diffuse = new BABYLON.Color3(0.95, 0.97, 1); // cool white

      // Hemisphere ambient fill
      const hemiLight = new BABYLON.HemisphericLight(
        "hemiLight",
        new BABYLON.Vector3(0, 1, 0),
        scene,
      );
      hemiLight.intensity = 0.5;
      hemiLight.diffuse = new BABYLON.Color3(1, 1, 1);
      hemiLight.groundColor = new BABYLON.Color3(0.8, 0.8, 0.82);

      // Shadow-catching ground plane (invisible but catches shadows)
      const ground = BABYLON.MeshBuilder.CreateGround(
        "ground",
        { width: 10, height: 10 },
        scene,
      );
      const groundMat = new BABYLON.PBRMaterial("groundMat", scene);
      groundMat.albedoColor = new BABYLON.Color3(0.96, 0.96, 0.96);
      groundMat.metallic = 0;
      groundMat.roughness = 1;
      groundMat.alpha = 0.0; // invisible
      ground.material = groundMat;
      ground.receiveShadows = true;
      ground.position.y = -0.01;

      // Shadow generator on key light
      const shadowGen = new BABYLON.ShadowGenerator(1024, keyLight);
      shadowGen.useBlurExponentialShadowMap = true;
      shadowGen.blurKernel = 32;
      shadowGen.setDarkness(0.6);

      // Render loop
      engine.runRenderLoop(() => {
        scene.render();
      });

      // Resize
      const onResize = () => engine.resize();
      window.addEventListener("resize", onResize);

      return () => {
        window.removeEventListener("resize", onResize);
      };
    }

    init();

    return () => {
      disposed = true;
      engineRef.current?.dispose();
      engineRef.current = null;
      sceneRef.current = null;
    };
  }, []);

  // Load mesh when URL changes
  useEffect(() => {
    if (!meshUrl || !sceneRef.current) return;

    let cancelled = false;

    async function loadMesh() {
      const BABYLON = await import("@babylonjs/core");
      const scene = sceneRef.current!;

      // Remove previous meshes
      for (const m of currentMeshesRef.current) {
        m.dispose();
      }
      currentMeshesRef.current = [];

      setViewerState("loading");

      try {
        const result = await BABYLON.SceneLoader.ImportMeshAsync(
          "",
          "",
          meshUrl!,
          scene,
          undefined,
          ".glb",
        );

        if (cancelled) {
          result.meshes.forEach((m) => m.dispose());
          return;
        }

        const meshes = result.meshes;
        currentMeshesRef.current = meshes;

        // Compute overall bounding box
        let min = new BABYLON.Vector3(Infinity, Infinity, Infinity);
        let max = new BABYLON.Vector3(-Infinity, -Infinity, -Infinity);

        for (const mesh of meshes) {
          if (!mesh.getBoundingInfo) continue;
          const bi = mesh.getBoundingInfo();
          min = BABYLON.Vector3.Minimize(min, bi.boundingBox.minimumWorld);
          max = BABYLON.Vector3.Maximize(max, bi.boundingBox.maximumWorld);
        }

        const center = BABYLON.Vector3.Center(min, max);
        const extent = max.subtract(min);
        const maxDim = Math.max(extent.x, extent.y, extent.z);

        // Normalize: scale mesh to fit in ~1.5 unit sphere, center at origin
        if (maxDim > 0) {
          const scale = 1.5 / maxDim;
          const root = meshes[0];
          if (root) {
            root.scaling = new BABYLON.Vector3(scale, scale, scale);
            root.position = center.negate().scale(scale);
            // Adjust so bottom sits near ground
            root.position.y += (extent.y * scale) / 2;
          }
        }

        // Apply PBR material to meshes without materials
        for (const mesh of meshes) {
          if (!mesh.material && mesh.getTotalVertices() > 0) {
            const mat = new BABYLON.PBRMaterial(`fabric_${mesh.name}`, scene);
            mat.albedoColor = new BABYLON.Color3(0.88, 0.86, 0.84); // warm off-white
            mat.metallic = 0;
            mat.roughness = 0.65; // fabric-like
            mat.backFaceCulling = false;
            mesh.material = mat;
          }

          // Enable shadows
          mesh.receiveShadows = true;
          const shadowGen = scene.lights[0]?.getShadowGenerator?.() as
            | InstanceType<typeof BABYLON.ShadowGenerator>
            | undefined;
          if (shadowGen) {
            shadowGen.addShadowCaster(mesh);
          }
        }

        // Reset camera to frame the garment
        if (cameraRef.current) {
          cameraRef.current.target = BABYLON.Vector3.Zero();
          cameraRef.current.alpha = -Math.PI / 4;
          cameraRef.current.beta = Math.PI / 3;
          cameraRef.current.radius = 2.5;
        }

        setViewerState("loaded");
      } catch (err) {
        console.error("Mesh loading failed:", err);
        if (!cancelled) setViewerState("error");
      }
    }

    loadMesh();
    return () => {
      cancelled = true;
    };
  }, [meshUrl]);

  // Set viewer to loading state when loading prop changes
  useEffect(() => {
    if (loading) setViewerState("loading");
  }, [loading]);

  // Set viewer to error state when error prop changes
  useEffect(() => {
    if (error) setViewerState("error");
  }, [error]);

  // Wireframe toggle
  useEffect(() => {
    const scene = sceneRef.current;
    if (!scene) return;
    for (const mesh of currentMeshesRef.current) {
      if (mesh.material) {
        mesh.material.wireframe = wireframe;
      }
    }
  }, [wireframe]);

  const resetCamera = useCallback(() => {
    const cam = cameraRef.current;
    const def = defaultCameraStateRef.current;
    if (!cam || !def) return;

    // Smooth animation using Babylon's built-in animation
    import("@babylonjs/core").then((BABYLON) => {
      const scene = sceneRef.current;
      if (!scene) return;

      BABYLON.Animation.CreateAndStartAnimation("camAlpha", cam, "alpha", 60, 30, cam.alpha, def.alpha, 0);
      BABYLON.Animation.CreateAndStartAnimation("camBeta", cam, "beta", 60, 30, cam.beta, def.beta, 0);
      BABYLON.Animation.CreateAndStartAnimation("camRadius", cam, "radius", 60, 30, cam.radius, def.radius, 0);
    });
  }, []);

  const toggleFullscreen = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const container = canvas.parentElement;
    if (!container) return;

    if (!document.fullscreenElement) {
      container.requestFullscreen();
      setIsFullscreen(true);
    } else {
      document.exitFullscreen();
      setIsFullscreen(false);
    }
  }, []);

  // Resize engine when fullscreen changes
  useEffect(() => {
    const onFsChange = () => {
      setIsFullscreen(!!document.fullscreenElement);
      setTimeout(() => engineRef.current?.resize(), 100);
    };
    document.addEventListener("fullscreenchange", onFsChange);
    return () => document.removeEventListener("fullscreenchange", onFsChange);
  }, []);

  return (
    <div className="relative flex h-full w-full flex-col overflow-hidden rounded-xl bg-[#f5f5f5]">
      {/* Canvas */}
      <canvas
        ref={canvasRef}
        className="h-full w-full touch-none outline-none"
        style={{ display: "block" }}
      />

      {/* Overlay states */}
      {viewerState === "empty" && (
        <div className="absolute inset-0 flex items-center justify-center">
          <p className="max-w-[200px] text-center text-sm text-neutral-400">
            Upload an image to generate a 3D garment
          </p>
        </div>
      )}

      {viewerState === "loading" && (
        <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 bg-[#f5f5f5]/80 backdrop-blur-sm">
          {/* Spinner */}
          <div className="h-8 w-8 animate-spin rounded-full border-2 border-neutral-300 border-t-neutral-800" />
          <p className="text-sm text-neutral-500">Generating 3D garment...</p>
        </div>
      )}

      {viewerState === "error" && (
        <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 bg-[#f5f5f5]/80 backdrop-blur-sm">
          <p className="text-sm text-red-500">{error || "Failed to load mesh"}</p>
          {onRetry && (
            <button
              onClick={onRetry}
              className="rounded-lg bg-neutral-900 px-4 py-2 text-xs text-white transition-colors hover:bg-neutral-800"
            >
              Retry
            </button>
          )}
        </div>
      )}

      {/* Controls overlay */}
      {viewerState === "loaded" && (
        <div className="absolute bottom-3 left-1/2 flex -translate-x-1/2 gap-1.5 rounded-lg bg-white/80 p-1.5 shadow-sm backdrop-blur-sm">
          {/* Reset camera */}
          <button
            onClick={resetCamera}
            title="Reset camera (or double-click)"
            className="flex h-8 w-8 items-center justify-center rounded-md text-neutral-500 transition-colors hover:bg-neutral-100 hover:text-neutral-700"
          >
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8" />
              <path d="M3 3v5h5" />
            </svg>
          </button>

          {/* Wireframe toggle */}
          <button
            onClick={() => setWireframe((w) => !w)}
            title="Toggle wireframe"
            className={`flex h-8 w-8 items-center justify-center rounded-md transition-colors hover:bg-neutral-100 ${
              wireframe ? "bg-neutral-200 text-neutral-800" : "text-neutral-500 hover:text-neutral-700"
            }`}
          >
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <rect x="3" y="3" width="7" height="7" />
              <rect x="14" y="3" width="7" height="7" />
              <rect x="14" y="14" width="7" height="7" />
              <rect x="3" y="14" width="7" height="7" />
            </svg>
          </button>

          {/* Fullscreen */}
          <button
            onClick={toggleFullscreen}
            title="Toggle fullscreen"
            className="flex h-8 w-8 items-center justify-center rounded-md text-neutral-500 transition-colors hover:bg-neutral-100 hover:text-neutral-700"
          >
            {isFullscreen ? (
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M8 3v3a2 2 0 0 1-2 2H3m18 0h-3a2 2 0 0 1-2-2V3m0 18v-3a2 2 0 0 1 2-2h3M3 16h3a2 2 0 0 1 2 2v3" />
              </svg>
            ) : (
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M8 3H5a2 2 0 0 0-2 2v3m18 0V5a2 2 0 0 0-2-2h-3m0 18h3a2 2 0 0 0 2-2v-3M3 16v3a2 2 0 0 0 2 2h3" />
              </svg>
            )}
          </button>
        </div>
      )}
    </div>
  );
}
