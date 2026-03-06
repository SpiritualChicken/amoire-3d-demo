"""One-shot patch: fix list vs numpy array return types in triangulation_utils."""
import pathlib

p = pathlib.Path("/workspace/amoire-3d-demo/ml/GarmentCodeRC/pygarment/meshgen/triangulation_utils.py")
text = p.read_text()
text = text.replace("keep.tolist()", "list(keep)")
text = text.replace(
    "return result_verts[:len_b]",
    "return [np.array(v) for v in result_verts[:len_b]]",
)
p.write_text(text)
print("Patched successfully")
