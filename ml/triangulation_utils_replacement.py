"""Helper functions for the triangulation of the panels

Drop-in replacement using the 'triangle' package instead of CGAL bindings.
Provides the same public API consumed by boxmeshgen.py.
"""

import numpy as np
import matplotlib.pyplot as plt

try:
    import triangle as tr
except ImportError:
    raise ImportError("Install the 'triangle' package: pip install triangle")


# ---------------------------------------------------------------------------
# Wrapper classes that mimic the CGAL CDT interface used by boxmeshgen.py
# ---------------------------------------------------------------------------

class _VertexHandle:
    """Lightweight stand-in for a CGAL vertex handle."""
    __slots__ = ("idx", "pt")

    def __init__(self, idx, pt):
        self.idx = idx
        self.pt = list(pt)

    def point(self):
        return _Point2(self.pt[0], self.pt[1])

    def set_point(self, p):
        self.pt = [p.x(), p.y()]


class _Point2:
    """Minimal stand-in for CGAL Point_2."""
    __slots__ = ("_x", "_y")

    def __init__(self, x, y):
        self._x = float(x)
        self._y = float(y)

    def x(self):
        return self._x

    def y(self):
        return self._y

    def __iter__(self):
        return iter((self._x, self._y))

    def __getitem__(self, i):
        return (self._x, self._y)[i]


# Expose as module-level name so tri_utils.Point_2 works
Point_2 = _Point2


class _CDT:
    """Wraps the 'triangle' library to provide the subset of the CGAL CDT
    interface that boxmeshgen.py actually uses."""

    def __init__(self):
        self._points = []          # list of [x, y]
        self._segments = []        # list of [i, j]
        self._result = None        # output of triangle.triangulate
        self._extra_boundary = 0   # count of extra points inserted by constraints

    # -- point / constraint insertion ------------------------------------------

    def insert(self, pt):
        """Insert a point and return a vertex handle."""
        idx = len(self._points)
        if isinstance(pt, _Point2):
            self._points.append([pt.x(), pt.y()])
        else:
            self._points.append([float(pt[0]), float(pt[1])])
        return _VertexHandle(idx, self._points[-1])

    def insert_constraint(self, vh_start, vh_end):
        """Add a segment constraint between two vertex handles."""
        self._segments.append([vh_start.idx, vh_end.idx])

    def number_of_vertices(self):
        return len(self._points)

    # -- triangulation ---------------------------------------------------------

    def _triangulate(self, flags="p"):
        """Run triangle and cache the result."""
        verts = np.array(self._points, dtype=np.float64)
        segs = (
            np.array(self._segments, dtype=np.int32)
            if self._segments
            else np.empty((0, 2), dtype=np.int32)
        )
        tri_input = dict(vertices=verts, segments=segs)
        self._result = tr.triangulate(tri_input, flags)

    def refine(self, max_area):
        """Constrained Delaunay + quality refinement."""
        flags = "pq30a{:.10f}".format(max_area)
        self._triangulate(flags)

    def triangulate_no_refine(self):
        """Constrained Delaunay without adding interior points."""
        self._triangulate("p")

    # -- query helpers used by get_keep_vertices / get_face_v_ids --------------

    def result_vertices(self):
        if self._result is not None:
            return self._result["vertices"].tolist()
        return list(self._points)

    def result_faces(self):
        if self._result is not None and "triangles" in self._result:
            return self._result["triangles"].tolist()
        return []


# Module-level names that boxmeshgen.py accesses via tri_utils.*
Mesh_2_Constrained_Delaunay_triangulation_2 = _CDT
Constrained_Delaunay_triangulation_2 = _CDT


class _MeshCriteria:
    """Stand-in for Delaunay_mesh_size_criteria_2(aspect_bound, size_bound)."""

    def __init__(self, aspect_bound, size_bound):
        self.aspect_bound = aspect_bound
        self.size_bound = size_bound


Delaunay_mesh_size_criteria_2 = _MeshCriteria


class _CGAL_Mesh_2_Module:
    """Namespace that provides refine_Delaunay_mesh_2 as boxmeshgen expects."""

    @staticmethod
    def refine_Delaunay_mesh_2(cdt, criteria):
        # Convert size_bound (max edge length) to max triangle area.
        # For an equilateral triangle: area = (sqrt(3)/4) * edge^2
        max_edge = criteria.size_bound
        max_area = (np.sqrt(3) / 4.0) * max_edge ** 2
        cdt.refine(max_area)


CGAL_Mesh_2 = _CGAL_Mesh_2_Module()


# ---------------------------------------------------------------------------
# Public functions matching the original API
# ---------------------------------------------------------------------------

def get_edge_vert_ids(edges):
    """Return pairs of vertex indices describing panel boundary segments."""
    zipped_array = np.empty((0, 2))
    for edge in edges:
        edge_verts_ids = edge.vertex_range
        rolled_list = np.roll(edge_verts_ids, 1, axis=0)
        zipped_array_edge = np.stack((rolled_list, edge_verts_ids), axis=1)[1:]
        zipped_array = np.concatenate((zipped_array, zipped_array_edge), axis=0)
    return zipped_array.astype(int)


def create_cdt_points(cdt, points):
    """Insert points into CDT, return list of vertex handles."""
    cdt_points = []
    for p in points:
        v = cdt.insert(p)
        cdt_points.append(v)
    return cdt_points


def cdt_insert_constraints(cdt, cdt_points, edge_verts_ids):
    """Insert boundary segment constraints into CDT.

    Returns a dict mapping any extra-inserted vertex indices to the boundary
    vertex they should be merged with.  With the triangle library, segment
    constraints never insert extra vertices, so this always returns {}.
    """
    for s_id, e_id in edge_verts_ids:
        cdt.insert_constraint(cdt_points[s_id], cdt_points[e_id])
    return {}


def get_keep_vertices(cdt, len_b):
    """After refinement, return interior + original boundary vertices,
    filtering out any extra boundary points the mesher may have added."""
    result_verts = cdt.result_vertices()
    result_faces = cdt.result_faces()

    if not result_faces:
        return result_verts[:len_b]

    faces = np.array(result_faces)
    all_verts = np.array(result_verts)

    # Find boundary edges (edges appearing in only one face)
    edges = np.concatenate([faces[:, :2], faces[:, 1:], faces[:, ::2]])
    edges_sorted = np.sort(edges, axis=1)
    unique_edges, counts = np.unique(edges_sorted, axis=0, return_counts=True)
    boundary_edges = unique_edges[counts == 1]
    boundary_v_ids = np.unique(boundary_edges.flatten())

    # New boundary vertices are those on the boundary but beyond original count
    new_bdry = boundary_v_ids[boundary_v_ids >= len_b]

    keep = np.delete(all_verts, new_bdry, axis=0)
    return keep.tolist()


def mark_domain(cdt):
    """No-op compatibility stub -- triangle already returns only interior faces."""
    return {}


def plot_triangulation(cdt, face_info):
    """Simple visualization using triangle's result."""
    result_verts = np.array(cdt.result_vertices())
    result_faces = cdt.result_faces()
    if not result_faces:
        return
    faces = np.array(result_faces)
    plt.triplot(result_verts[:, 0], result_verts[:, 1], faces, "b-", linewidth=0.5)

    # Draw boundary segments in red
    if cdt._segments:
        pts = np.array(cdt._points)
        for s in cdt._segments:
            plt.plot(pts[s, 0], pts[s, 1], "r-", linewidth=1.5)
    plt.gca().set_aspect("equal")
    plt.show()


def get_face_v_ids(cdt, points, new_points, check=False, plot=False):
    """Triangulate (without refinement) and return face vertex indices.

    This is the second-pass CDT that uses the refined-and-filtered vertices
    to produce the final triangulation.
    """
    cdt.triangulate_no_refine()
    faces = cdt.result_faces()

    if not faces:
        return np.empty((0, 3), dtype=int)

    f = np.array(faces, dtype=int)

    # Handle new_points remapping (should be empty with triangle, but keep for safety)
    if new_points:
        for i in range(len(f)):
            for j in range(3):
                if f[i, j] in new_points:
                    f[i, j] = new_points[f[i, j]]
        # Remove degenerate faces
        mask = ~(
            (f[:, 0] == f[:, 1]) | (f[:, 1] == f[:, 2]) | (f[:, 0] == f[:, 2])
        )
        f = f[mask]

    if plot:
        plot_triangulation(cdt, {})

    return f


def get_faces_sorted(cdt):
    """Return sorted face indices and vertex coordinates."""
    cdt.triangulate_no_refine()
    verts = cdt.result_vertices()
    faces = cdt.result_faces()
    if not faces:
        return np.empty((0, 3), dtype=int), verts
    f = np.array([sorted(face) for face in faces], dtype=int)
    return f, verts


def is_manifold(face_v_ids: np.ndarray, points: np.ndarray, tol=1e-2):
    """Check if all face triangles are proper (not degenerate)."""
    faces = points[face_v_ids]
    face_side_1 = np.linalg.norm(faces[:, 0] - faces[:, 1], axis=1)
    face_side_2 = np.linalg.norm(faces[:, 1] - faces[:, 2], axis=1)
    face_side_3 = np.linalg.norm(faces[:, 0] - faces[:, 2], axis=1)
    side_lengths = np.stack([face_side_1, face_side_2, face_side_3], axis=-1)
    return np.all(side_lengths.sum(axis=1) > 2 * side_lengths.max(axis=1) + tol)
