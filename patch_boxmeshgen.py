"""Patch boxmeshgen.py to skip UV texture generation when in_uv_config is None."""
import sys

f = '/workspace/amoire-3d-demo/ml/GarmentCodeRC/pygarment/meshgen/boxmeshgen.py'
with open(f) as fh:
    lines = fh.readlines()

# Find the patched None check (from previous patch) and fix it to pass [] instead of None
old_patch = '                None,\n'
found = False
for i, line in enumerate(lines):
    if line == '        if in_uv_config is None:\n':
        # Find the None argument in the save_obj call and replace with []
        for j in range(i, min(i + 10, len(lines))):
            if lines[j] == old_patch:
                lines[j] = '                [],\n'
                found = True
                break
        break

if not found:
    print('ERROR: Could not find previous patch to fix. Was it applied?')
    sys.exit(1)

with open(f, 'w') as fh:
    fh.writelines(lines)

print('Patched: save_obj now gets [] instead of None for UVs')
