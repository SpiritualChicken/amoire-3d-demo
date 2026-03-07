"""Patch simulation.py to fix signal-in-thread and skip render_images."""
import sys

f = '/workspace/amoire-3d-demo/ml/GarmentCodeRC/pygarment/meshgen/simulation.py'
with open(f) as fh:
    code = fh.read()

count = 0

# Fix 1: Wrap signal calls in try/except to handle non-main-thread usage
old1 = '                signal.signal(signal.SIGALRM, alarm_handler)\n                signal.alarm(frame_timeout_after)'
new1 = '                try:\n                    signal.signal(signal.SIGALRM, alarm_handler)\n                    signal.alarm(frame_timeout_after)\n                except ValueError:\n                    pass  # signal only works in main thread'
if old1 in code:
    code = code.replace(old1, new1)
    count += 1

# Also fix the alarm(0) reset
old1b = '                    signal.alarm(0)'
new1b = '                    try:\n                        signal.alarm(0)\n                    except ValueError:\n                        pass'
if old1b in code:
    code = code.replace(old1b, new1b)
    count += 1

# Fix 2: Skip render_images (we don't need rendered images, and it fails without UVs)
old2 = '    render_images(paths, garment.v_body, garment.f_body, render_props[\'config\'])'
new2 = '    # render_images(paths, garment.v_body, garment.f_body, render_props[\'config\'])  # Skipped: not needed for API'
if old2 in code:
    code = code.replace(old2, new2)
    count += 1

if count == 0:
    print('ERROR: No patches applied. Already patched?')
    sys.exit(1)

with open(f, 'w') as fh:
    fh.write(code)

print(f'Patched simulation.py: {count} changes applied')
