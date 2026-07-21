# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "meshmash>=0.1.0",
#     "numpy",
#     "point-cloud-utils>=0.30.0",
#     "pymeshlab>=2023.12",
#     "pyvista[all]>=0.48.4",
#     "trimesh>=4.12.1",
# ]
# ///

# %%
# ---------------------------------------------------------------------------
# Watertight remeshing via volumetric (SDF) uniform resampling (pymeshlab).
#
# Strategy (orientation-FREE): build a signed-distance grid around the input
# surface and marching-cubes it back out. This produces a watertight, manifold
# mesh and -- because we keep only the largest connected component -- drops
# interior blobs and floating junk automatically, without needing consistent
# normal orientation. Resolution (cellsize) is exposed as a sweepable parameter.
# ---------------------------------------------------------------------------
import time

import numpy as np
import point_cloud_utils as pcu
import pymeshlab as ml
import pyvista as pv
import trimesh
from meshmash import fetch_sample_mesh

mesh = fetch_sample_mesh("microns_neuron_sample")
vertices = mesh[0].astype(np.float64)
faces = mesh[1].astype(np.int32)
print(f"Input mesh: {len(vertices)} vertices, {len(faces)} faces")

# %%
# light preclean: merge coincident vertices (matches robust_inside_outside.py)
vertices, faces, _, _ = pcu.deduplicate_mesh_vertices(
    vertices, faces, epsilon=5, return_index=True
)
print(f"After dedup: {len(vertices)} vertices, {len(faces)} faces")


# %%
def uniform_resample(vertices, faces, cellsize_pct, offset_pct=0.0):
    """Volumetric uniform resampling (SDF grid -> marching cubes) via pymeshlab.

    cellsize_pct / offset_pct are percentages of the mesh bounding-box diagonal.
    Returns (out_vertices, out_faces).
    """
    ms = ml.MeshSet()
    ms.add_mesh(ml.Mesh(vertex_matrix=vertices, face_matrix=faces))
    ms.generate_resampled_uniform_mesh(
        cellsize=ml.PercentageValue(cellsize_pct),
        offset=ml.PercentageValue(offset_pct),
        mergeclosevert=True,
    )
    out = ms.current_mesh()
    return out.vertex_matrix(), out.face_matrix().astype(np.int32)


def keep_largest_component(vertices, faces):
    """Drop interior blobs / floating junk by keeping the largest component."""
    tm = trimesh.Trimesh(vertices, faces, process=False)
    comps = tm.split(only_watertight=False)
    if len(comps) == 0:
        return vertices, faces
    largest = max(comps, key=lambda c: len(c.faces))
    return np.asarray(largest.vertices), np.asarray(largest.faces, dtype=np.int32)


# %%
# sweep resolution: cellsize as a percentage of the bounding-box diagonal.
# smaller cellsize -> finer detail (spines), slower, larger output.
cellsize_sweep = [2.0, 1.0, 0.5]

results = {}
for cellsize_pct in cellsize_sweep:
    t0 = time.time()
    rv, rf = uniform_resample(vertices, faces, cellsize_pct)
    rv, rf = keep_largest_component(rv, rf)
    elapsed = time.time() - t0

    watertight = trimesh.Trimesh(rv, rf, process=False).is_watertight
    results[cellsize_pct] = (rv, rf, elapsed, watertight)
    print(
        f"cellsize={cellsize_pct:>4}%  "
        f"verts={len(rv):>8}  faces={len(rf):>8}  "
        f"time={elapsed:6.2f}s  watertight={watertight}"
    )

# %%
# visualize each swept result overlaid on the original mesh
orig_poly = pv.make_tri_mesh(vertices, faces)

plotter = pv.Plotter(shape=(1, len(cellsize_sweep)))
for i, cellsize_pct in enumerate(cellsize_sweep):
    rv, rf, elapsed, watertight = results[cellsize_pct]
    plotter.subplot(0, i)
    plotter.add_text(
        f"cellsize={cellsize_pct}%\n{len(rf)} faces\nwatertight={watertight}",
        font_size=8,
    )
    plotter.add_mesh(orig_poly, color="lightgray", opacity=0.25)
    plotter.add_mesh(pv.make_tri_mesh(rv, rf), color="tomato", opacity=0.85)
plotter.link_views()
plotter.enable_fly_to_right_click()
plotter.show()

# %%
# write the finest watertight result to outs/ as .ply for downstream use
finest = min(cellsize_sweep)
rv, rf, _, _ = results[finest]
out_path = f"../outs/watertight_meshlab_cellsize_{finest}pct.ply"
trimesh.Trimesh(rv, rf, process=False).export(out_path)
print(f"Wrote {out_path}")

# %%
