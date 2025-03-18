# %%
from caveclient import CAVEclient
from cloudvolume import CloudVolume

client = CAVEclient("minnie65_phase3_v1")
cv = client.info.segmentation_cloudvolume(progress=False)
cv = CloudVolume(
    "precomputed://gs://iarpa_microns/minnie/minnie65/seg_m943", progress=False
)
# cv.cloudpath =
# root_id = 864691135501563458
root_id = 864691135855890478  # l 23 with large axon
root_id = 864691135572094189  # chandelier cell
raw_mesh = cv.mesh.get(root_id)[root_id]
# raw_mesh.deduplicate_chunk_boundaries()
raw_mesh.deduplicate_vertices(True)
# raw_mesh.deduplicate
raw_mesh = (raw_mesh.vertices, raw_mesh.faces)

# %%
import numpy as np

row = client.materialize.query_view(
    "nucleus_detection_lookup_v1",
    filter_equal_dict={"pt_root_id": root_id},
    materialization_version=943,
    desired_resolution=[1, 1, 1],
).iloc[0]
center = np.array(row["pt_position"])

# %%

from fast_simplification import simplify

simple_mesh = simplify(*raw_mesh, target_reduction=0.9, agg=9)

simple_mesh = (simple_mesh[0].astype("float32"), simple_mesh[1])

new_vertices = simple_mesh[0].copy()
# recenter to soma
new_vertices -= center
# flip z and y
new_vertices = new_vertices[:, [0, 2, 1]]
# new y needs to be flipped
new_vertices[:, 2] = new_vertices[:, 2] * -1

simple_mesh = (new_vertices, simple_mesh[1])

import pyvista as pv

poly = pv.make_tri_mesh(*simple_mesh)

poly["x"] = poly.points[:, 0] > poly.points[:, 0].mean()

plotter = pv.Plotter()

plotter.add_mesh(poly, color="black")

plotter.camera.focal_point = (0, 0, 0)

# plotter.show()

plotter.export_gltf(
    f"{root_id}.gltf", inline_data=True, rotate_scene=True, save_normals=False
)
# python3 -m http.server 9000
# %%
from meshio import Mesh

Mesh(simple_mesh[0].astype("float32"), {"triangle": simple_mesh[1]}).write(
    f"{root_id}.vtu"
)
