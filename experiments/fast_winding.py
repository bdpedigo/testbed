# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#     "caveclient[cv]==7.6.0",
#     "gpytoolbox==0.3.3",
#     "pyvista[all]==0.44.2",
# ]
# ///


# %%
import time

import numpy as np
import pyvista as pv
from caveclient import CAVEclient
from gpytoolbox import fast_winding_number

client = CAVEclient("minnie65_phase3_v1", version=1300)

cv = client.info.segmentation_cloudvolume()

# %%
# root_id = 864691135662877004
root_id = 864691135279120289

leaves = client.chunkedgraph.get_leaves(root_id, stop_layer=3)

node = leaves[0]
node = root_id
mesh = cv.mesh.get(
    node, remove_duplicate_vertices=True, deduplicate_chunk_boundaries=False
)[node]

mesh = (mesh.vertices, mesh.faces)

# %%

bounds = np.array((mesh[0].min(axis=0), mesh[0].max(axis=0)))

n_samples = 1_000_000

indices = np.random.choice(len(mesh[0]), n_samples, replace=True)
points = mesh[0][indices].copy()
points += np.random.normal(0, 100, points.shape)

# %%

currtime = time.time()

out = fast_winding_number(points, mesh[0], mesh[1])
print(f"{time.time() - currtime:.3f} seconds elapsed.")

# %%

plotter = pv.Plotter(shape=(1, 2))

plotter.subplot(0, 0)
plotter.add_mesh(pv.make_tri_mesh(*mesh), opacity=0.5)

plotter.add_points(
    points,
    scalars=out,
    render_points_as_spheres=True,
    point_size=2,
    cmap="coolwarm",
    clim=[0, 1],
    show_scalar_bar=False,
)

plotter.subplot(0, 1)

plotter.add_mesh(pv.make_tri_mesh(*mesh), opacity=0.5)

plotter.add_points(
    points[out > 0.5],
    scalars=out[out > 0.5],
    render_points_as_spheres=True,
    point_size=2,
    cmap="coolwarm",
    clim=[0, 1],
    show_scalar_bar=False,
)

plotter.link_views()
plotter.enable_fly_to_right_click()

# plotter.export_html("fast_winding.html")
plotter.show()

# %%
