"""Probe BoxMesh API to discover how to extract vertices/faces."""
import sys
sys.path.insert(0, '/workspace/amoire-3d-demo/ml/GarmentCodeRC')

from pygarment.meshgen.boxmeshgen import BoxMesh
import pygarment.data_config as dc

props = dc.Properties(
    '/workspace/amoire-3d-demo/ml/GarmentCodeRC/assets/Sim_props/default_sim_props.yaml'
)
res = props['sim']['config']['resolution_scale']
print(f'Resolution scale: {res}')

spec = '/workspace/amoire-3d-demo/data/outputs/a623de74a5db/valid_garment_upper/valid_garment_upper_specification.json'
bm = BoxMesh(spec, res)
bm.load()

print('\n=== All attributes ===')
attrs = [a for a in dir(bm) if not a.startswith('_')]
print(attrs)

print('\n=== Mesh-related attributes ===')
candidates = [
    'vertices', 'verts', 'faces', 'mesh', 'V', 'F',
    'panels', 'triangles', 'tri_faces', 'vert_data',
    'pattern', 'name', 'spec', 'panel_order',
    'sim_mesh', 'box_mesh', 'edge_list', 'stitch_list',
]
for a in candidates:
    if hasattr(bm, a):
        val = getattr(bm, a)
        if hasattr(val, 'shape'):
            print(f'  {a}: {type(val).__name__} shape={val.shape}')
        elif hasattr(val, '__len__'):
            print(f'  {a}: {type(val).__name__} len={len(val)}')
        else:
            print(f'  {a}: {type(val).__name__} = {repr(val)[:100]}')

# Try to find mesh data through the pattern
print('\n=== Checking pattern attribute ===')
if hasattr(bm, 'pattern'):
    pat = bm.pattern
    print(f'pattern type: {type(pat).__name__}')
    pat_attrs = [a for a in dir(pat) if not a.startswith('_')]
    print(f'pattern attrs: {pat_attrs}')

# Try to find panel meshes
print('\n=== Checking for panel data ===')
if hasattr(bm, 'panels'):
    panels = bm.panels
    if isinstance(panels, dict):
        for k, v in panels.items():
            print(f'  panel "{k}": {type(v).__name__}')
            if hasattr(v, '__dict__'):
                for vk, vv in v.__dict__.items():
                    if not vk.startswith('_'):
                        info = f'{type(vv).__name__}'
                        if hasattr(vv, 'shape'):
                            info += f' shape={vv.shape}'
                        elif hasattr(vv, '__len__'):
                            info += f' len={len(vv)}'
                        print(f'    {vk}: {info}')
    elif isinstance(panels, list):
        print(f'  {len(panels)} panels')
        if panels:
            p = panels[0]
            print(f'  first panel type: {type(p).__name__}')

# Check if trimesh export is possible
print('\n=== Trying trimesh export ===')
try:
    import trimesh
    if hasattr(bm, 'vertices') and hasattr(bm, 'faces'):
        m = trimesh.Trimesh(vertices=bm.vertices, faces=bm.faces)
        print(f'Direct: {len(m.vertices)} verts, {len(m.faces)} faces')
    elif hasattr(bm, 'mesh') and isinstance(bm.mesh, trimesh.Trimesh):
        print(f'Has trimesh: {len(bm.mesh.vertices)} verts, {len(bm.mesh.faces)} faces')
    else:
        print('No direct trimesh conversion found')
except Exception as e:
    print(f'Error: {e}')
