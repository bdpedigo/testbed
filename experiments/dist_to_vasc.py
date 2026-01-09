# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "caveclient",
#     "cloud-volume",
#     "numpy",
#     "point-cloud-utils",
#     "pyvista",
#     "scikit-learn",
#     "seaborn",
#     "skel-features",
# ]
#
# [tool.uv.sources]
# skel-features = { git = "https://github.com/AllenInstitute/em_skeleton_feature_extraction.git" }
# ///

# %%
import time
from typing import Any, Tuple, Union

import numpy as np
import point_cloud_utils as pcu
import pyvista as pv
import seaborn as sns
import skel_features as sf
from caveclient import CAVEclient
from cloudvolume import CloudVolume
from sklearn.metrics import pairwise_distances_argmin_min

type Mesh = Union[tuple[np.ndarray, np.ndarray], Any]


def interpret_mesh(mesh) -> Tuple[np.ndarray, np.ndarray]:
    if isinstance(mesh, tuple):
        return mesh
    elif hasattr(mesh, "vertices") and hasattr(mesh, "faces"):
        return mesh.vertices, mesh.faces
    else:
        raise ValueError(
            "Mesh should be tuple of vertices and faces or object with `vertices` and `faces` attributes"
        )


def edges_to_lines(edges: np.ndarray) -> np.ndarray:
    lines = np.column_stack((np.full((len(edges), 1), 2), edges))
    return lines


def poly_to_mesh(poly: pv.PolyData) -> Mesh:
    vertices = np.asarray(poly.points)
    faces = poly.faces.reshape(-1, 4)[:, 1:]
    return vertices, faces


def subset_mesh_by_indices(mesh: Mesh, indices: np.ndarray) -> Mesh:
    if indices.dtype == bool:
        indices = np.where(indices)[0]

    vertices, faces = interpret_mesh(mesh)
    new_vertices = vertices[indices]
    index_mapping = dict(zip(indices, np.arange(len(indices))))

    # use numpy to get faces for which all indices are in the subset
    face_mask = np.all(np.isin(faces, indices), axis=1)

    new_faces = np.vectorize(index_mapping.get, otypes=[faces.dtype])(faces[face_mask])
    return new_vertices, new_faces


bv_cv = CloudVolume(
    "precomputed://https://rhoana.rc.fas.harvard.edu/ng/EM_lowres/mouse/bv",
    progress=False,
)

currtime = time.time()

raw_mesh = bv_cv.mesh.get(1)
bv_mesh = (raw_mesh.vertices, raw_mesh.faces)

print(f"{time.time() - currtime:.3f} seconds elapsed to download vasculature mesh.")

# %%

bv_poly = pv.make_tri_mesh(*bv_mesh)
bv_poly = bv_poly.extract_largest()
bv_mesh = poly_to_mesh(bv_poly)

# %%

skel_dir = "gs://csm-skel/skel-test"

version = 1412
client = CAVEclient("minnie65_public", version=version)
cell_info = client.materialize.query_view("aibs_cell_info")
cell_info = cell_info.query("cell_type_source == 'allen_v1_column_types_slanted_ref'")
root_ids = cell_info.query("broad_type.isin(['excitatory', 'inhibitory'])")[
    "pt_root_id"
].unique()

nrn = sf.io_utils.load_root_id(root_ids[0], skel_dir)

nrn.reset_mask()

# nrn.apply_mask(nrn.anno["is_dendrite"].mesh_mask | nrn.anno["is_soma"].mesh_mask)
nrn.apply_mask(nrn.anno["is_axon"].mesh_mask)

# %%

skel_edges = nrn.skeleton.edges
skel_verts = nrn.skeleton.vertices.astype("float32")

lines = edges_to_lines(skel_edges)

line_mesh = pv.PolyData(skel_verts, lines=lines)

plotter = pv.Plotter()

plotter.add_mesh(line_mesh, color="black", line_width=3)
plotter.add_mesh(bv_poly, color="red", opacity=0.1)

plotter.enable_fly_to_right_click()
plotter.show()

# %%

currtime = time.time()
dists, fid, bc = pcu.closest_points_on_mesh(skel_verts, *bv_mesh)

print(f"{time.time() - currtime:.3f} seconds elapsed to compute closest points on mesh.")

closest_pts = pcu.interpolate_barycentric_coords(bv_mesh[1], fid, bc, bv_mesh[0])

# %%
drawing_pts = np.concatenate([skel_verts, closest_pts], axis=0)
drawing_edges = np.arange(len(drawing_pts)).reshape(-1, 2, order="F")
drawing_lines = edges_to_lines(drawing_edges)
drawing_mesh = pv.PolyData(drawing_pts, lines=drawing_lines)

plotter = pv.Plotter()
plotter.add_mesh(line_mesh, line_width=10, scalars=dists, cmap="summer_r")
plotter.add_mesh(drawing_mesh, color="grey", line_width=1)
plotter.add_mesh(bv_poly, color="red", opacity=0.1)
plotter.enable_fly_to_right_click()

plotter.show()

# %%

# just for visualization, minimum distance to skeleton in nm to be shown
threshold = 50_000

currtime = time.time()
_, mesh_dists = pairwise_distances_argmin_min(bv_mesh[0], skel_verts)

mask = mesh_dists < threshold

subset_mesh = subset_mesh_by_indices(bv_mesh, mask)

print(f"{time.time() - currtime:.3f} seconds elapsed to mask vasculature mesh.")

# %%
plotter = pv.Plotter()
plotter.add_mesh(line_mesh, line_width=10, scalars=dists, cmap="summer_r")
plotter.add_mesh(drawing_mesh, color="grey", line_width=1)
plotter.add_mesh(pv.make_tri_mesh(*subset_mesh), color="red", opacity=0.1)
plotter.enable_fly_to_right_click()

plotter.show()

# %%

sns.histplot(x=dists, log_scale=False)
