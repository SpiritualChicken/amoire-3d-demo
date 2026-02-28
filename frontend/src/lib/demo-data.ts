/**
 * Demo/fallback data used when the backend is not available.
 * Provides sample meshes and images so the frontend works standalone.
 */

export interface DemoGarment {
  id: string;
  name: string;
  type: string;
  meshUrl: string;
  imageUrl: string;
}

export const DEMO_GARMENTS: DemoGarment[] = [
  {
    id: "demo-tshirt",
    name: "Classic T-Shirt",
    type: "tshirt",
    meshUrl: "/sample-meshes/tshirt.glb",
    imageUrl: "/sample-images/tshirt.png",
  },
  {
    id: "demo-skirt",
    name: "A-Line Skirt",
    type: "skirt",
    meshUrl: "/sample-meshes/skirt.glb",
    imageUrl: "/sample-images/skirt.png",
  },
  {
    id: "demo-pants",
    name: "Straight Pants",
    type: "pants",
    meshUrl: "/sample-meshes/pants.glb",
    imageUrl: "/sample-images/pants.png",
  },
  {
    id: "demo-dress",
    name: "Summer Dress",
    type: "dress",
    meshUrl: "/sample-meshes/dress.glb",
    imageUrl: "/sample-images/dress.png",
  },
];
