# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "gpytoolbox==0.3.3",
#     "libigl>=2.6.2",
#     "meshmash>=0.1.0",
#     "numpy",
#     "point-cloud-utils>=0.30.0",
#     "pyvista[all]>=0.48.4",
#     "trimesh>=4.12.1",
# ]
# ///

# %%
# ---------------------------------------------------------------------------
# Watertight remeshing via generalized-winding-number iso-surfacing on an
# ADAPTIVE octree (libigl `lipschitz_octree` -> `unique_sparse_voxel_corners`
# -> `marching_cubes`).
#
# Strategy (orientation-DEPENDENT): once every connected component is consistently
# OUTWARD-oriented, a buried interior blob contributes +1 to the winding number,
# so the whole interior (blob included) has WN >= 1. The WN = 0.5 isosurface then
# treats interior junk as solid and FILLS it, rather than carving a hole. This is
# why the orientation sign-fix (reused from robust_inside_outside.py) is a
# prerequisite. The isosurface regenerates a fresh closed boundary => watertight.
#
# The "octree" is libigl's `lipschitz_octree`: driven by an EXACT (1-Lipschitz)
# unsigned distance-to-triangles function, it refines only cells the surface can
# pass through, yielding a sparse adaptive narrow band of leaf cells. Using an
# exact distance (never an over-estimate) is essential: an over-estimating udf
# (e.g. distance to nearest vertex) prunes surface-containing cells and leaves
# holes. The finest cell size is controlled by `voxel_size` (swept below).
# ---------------------------------------------------------------------------
import time

import igl
import numpy as np
import point_cloud_utils as pcu
import pyvista as pv
import trimesh
from gpytoolbox import fast_winding_number

from meshmash import fetch_sample_mesh

mesh = fetch_sample_mesh("microns_neuron_sample")
vertices = mesh[0].astype(np.float64)
faces = mesh[1].astype(np.int32)
print(f"Input mesh: {len(vertices)} vertices, {len(faces)} faces")

# %%
# --- preclean: dedup, orient within components, drop degenerate faces ---
vertices, faces, _, _ = pcu.deduplicate_mesh_vertices(
    vertices, faces, epsilon=5, return_index=True
)
faces, _ = pcu.orient_mesh_faces(faces)
areas = pcu.mesh_face_areas(vertices, faces)
faces = faces[areas > 1e-2]
print(f"After preclean: {len(vertices)} vertices, {len(faces)} faces")

# %%
# ---------------------------------------------------------------------------
# GLOBAL outward orientation fix (per-component, via each component's own
# self-winding-number). See robust_inside_outside.py for the full rationale:
# judging each component on its OWN winding field is robust to mis-orientation
# elsewhere, and comparing |WN| magnitudes (not signed) makes the fix idempotent.
# ---------------------------------------------------------------------------
cv, nv, cf, nf = pcu.connected_components(vertices, faces)

timer = time.time()
oriented_faces = faces.copy()
flip_eps = 25.0  # nm, offset for the sign-probe points
n_flipped = 0
for comp_id in np.unique(cf):
    comp_face_mask = cf == comp_id
    comp_faces = oriented_faces[comp_face_mask]
    comp_normals = pcu.estimate_mesh_face_normals(vertices, comp_faces)
    comp_ctrs = vertices[comp_faces].mean(axis=1)
    probe_out = comp_ctrs + comp_normals * flip_eps
    probe_in = comp_ctrs - comp_normals * flip_eps
    w_out = fast_winding_number(probe_out, vertices, comp_faces)
    w_in = fast_winding_number(probe_in, vertices, comp_faces)
    if np.mean(np.abs(w_out)) - np.mean(np.abs(w_in)) > 0:
        oriented_faces[comp_face_mask] = comp_faces[:, ::-1]
        n_flipped += 1
faces = oriented_faces
print(
    f"Flipped {n_flipped} / {len(np.unique(cf))} components outward "
    f"in {time.time() - timer:.2f}s"
)

# %%
# exact (1-Lipschitz) unsigned distance to the triangle soup, batched. This is
# what drives octree refinement; libigl calls it once per depth level on the
# current active cell centers, so an exact point-to-mesh distance is affordable.
faces64 = faces.astype(np.int64)


def udf(query_points):
    """Exact unsigned distance from each query point to the mesh surface."""
    sqr_d, _, _ = igl.point_mesh_squared_distance(
        np.ascontiguousarray(query_points, dtype=np.float64), vertices, faces64
    )
    return np.sqrt(sqr_d)


# %%
def winding_isosurface(vertices, faces64, voxel_size, udf, pad=1.5):
    """Adaptive-octree winding-number iso-surface.

    1. `igl.lipschitz_octree` refines a root cell (side `h0`, fully bracketing
       the mesh) down to `max_depth` wherever the exact udf says the surface
       could pass, giving a sparse set of leaf cells (`ijk`). The finest leaf
       side length is ~`voxel_size`.
    2. `igl.unique_sparse_voxel_corners` de-duplicates the shared corners into a
       corner list `GV` plus a per-cell 8-corner index array `J`.
    3. Batch-evaluate the accurate generalized winding number at the corners in
       ONE call, centered so the surface is the 0 level set (WN - 0.5).
    4. `igl.marching_cubes` extracts the watertight isosurface on the sparse
       cell grid.
    """
    bbmin = vertices.min(axis=0)
    bbmax = vertices.max(axis=0)
    center = (bbmin + bbmax) / 2.0
    h0 = float((bbmax - bbmin).max()) * pad  # root cell brackets the whole mesh
    origin = center - h0 / 2.0
    max_depth = int(np.ceil(np.log2(h0 / voxel_size)))
    cell_size = h0 / (2**max_depth)

    ijk = igl.lipschitz_octree(origin, h0, max_depth, udf)
    _, corner_idx, corner_pts = igl.unique_sparse_voxel_corners(
        origin, h0, max_depth, ijk
    )
    wn = igl.fast_winding_number(vertices, faces64, corner_pts) - 0.5
    sv, sf = igl.marching_cubes(wn, corner_pts, corner_idx)
    return np.asarray(sv), np.asarray(sf, dtype=np.int32), cell_size, len(ijk)


# %%
# sweep the finest voxel size (nm). smaller -> finer spines, more cells, slower.
voxel_sweep = [100.0, 50.0, 25.0]

results = {}
for voxel_size in voxel_sweep:
    t0 = time.time()
    sv, sf, cell_size, n_cells = winding_isosurface(vertices, faces64, voxel_size, udf)
    elapsed = time.time() - t0
    watertight = trimesh.Trimesh(sv, sf, process=False).is_watertight
    results[voxel_size] = (sv, sf, elapsed, watertight)
    print(
        f"voxel={voxel_size:>6}nm (cell={cell_size:6.1f}nm, cells={n_cells:>8})  "
        f"verts={len(sv):>8}  faces={len(sf):>8}  "
        f"time={elapsed:6.2f}s  watertight={watertight}"
    )

# %%
# write every swept result to outs/ as .ply for downstream use / inspection
# (done before the blocking interactive window so results are never lost)
from pathlib import Path

try:
    _here = Path(__file__).resolve().parent
except NameError:  # interactive cell execution
    _here = Path.cwd()
out_dir = _here.parent / "outs"
out_dir.mkdir(parents=True, exist_ok=True)

for voxel_size in voxel_sweep:
    sv, sf, _, watertight = results[voxel_size]
    out_path = out_dir / f"watertight_winding_voxel_{int(voxel_size)}nm.ply"
    trimesh.Trimesh(sv, sf, process=False).export(out_path)
    print(f"Wrote {out_path}  (watertight={watertight})")

# %%
# visualize each swept result overlaid on the original mesh
orig_poly = pv.make_tri_mesh(vertices, faces)

plotter = pv.Plotter(shape=(1, len(voxel_sweep)))
for i, voxel_size in enumerate(voxel_sweep):
    sv, sf, elapsed, watertight = results[voxel_size]
    plotter.subplot(0, i)
    plotter.add_text(
        f"voxel={voxel_size}nm\n{len(sf)} faces\nwatertight={watertight}", font_size=8
    )
    plotter.add_mesh(orig_poly, color="lightgray", opacity=0.25)
    plotter.add_mesh(pv.make_tri_mesh(sv, sf), color="cornflowerblue", opacity=0.85)
plotter.link_views()
plotter.enable_fly_to_right_click()
plotter.show()

# %%
