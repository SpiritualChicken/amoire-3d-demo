"""Patch boxmeshgen.py to skip UV texture generation when in_uv_config is None."""
import sys

f = '/workspace/amoire-3d-demo/ml/GarmentCodeRC/pygarment/meshgen/boxmeshgen.py'
with open(f) as fh:
    lines = fh.readlines()

# Find the line with uv_config.update(in_uv_config)
target = '        uv_config.update(in_uv_config)\n'
idx = None
for i, line in enumerate(lines):
    if line == target:
        idx = i
        break

if idx is None:
    print('ERROR: Could not find target line. Already patched?')
    sys.exit(1)

# Insert the None check before that line
patch_lines = [
    '        if in_uv_config is None:\n',
    '            save_obj(\n',
    '                self.paths.g_box_mesh,\n',
    '                self.vertices,\n',
    '                self.faces_with_texture,\n',
    '                None,\n',
    '                vert_normals=self.eval_vertex_normals() if with_normals else None,\n',
    '            )\n',
    '            return\n',
]

lines[idx:idx] = patch_lines

with open(f, 'w') as fh:
    fh.writelines(lines)

print('Patched successfully!')
