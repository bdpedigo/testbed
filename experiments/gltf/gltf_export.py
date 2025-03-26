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
# root_id = 864691135572094189  # chandelier cell
raw_mesh = cv.mesh.get(root_id)[root_id]
# raw_mesh.deduplicate_chunk_boundaries()
raw_mesh.deduplicate_vertices(True)
# raw_mesh.deduplicate
raw_mesh = (raw_mesh.vertices, raw_mesh.faces)

# %%
import numpy as np

cut_pos = np.array([143461, 121963, 24039])
cut_pos *= np.array([4, 4, 40])

bounds = np.array([cut_pos, cut_pos])
cut_width = 3000
bounds[0] -= np.array([cut_width, 300, cut_width])
bounds[1] += np.array([cut_width, 300, cut_width])

import pyvista as pv
from fast_simplification import simplify

simple_mesh = simplify(*raw_mesh, target_reduction=0.9, agg=9)

simple_mesh = (simple_mesh[0].astype("float32"), simple_mesh[1])

# %%
skel = client.skeleton.get_skeleton(root_id, output_format="dict")

# %%
skeleton_vertices = skel["vertices"]
skeleton_edges = skel["edges"]


def edges_to_lines(edges: np.ndarray) -> np.ndarray:
    lines = np.column_stack((np.full((len(edges), 1), 2), edges))
    return lines


lines = edges_to_lines(skeleton_edges)

skel_poly = pv.PolyData(skeleton_vertices, lines=lines)
skel_poly["compartment"] = skel["compartment"]

from sklearn.neighbors import NearestNeighbors

nbrs = NearestNeighbors(n_neighbors=1).fit(skeleton_vertices)
distances, indices = nbrs.kneighbors(simple_mesh[0])

# %%
skel_compartment = skel["compartment"].copy()
skel_compartment[skel_compartment == 1] = 3  # soma to dendrite
mesh_compartments = skel_compartment[indices.flatten()]

# %%

from scipy.sparse import csr_array, eye_array
from scipy.sparse.csgraph import laplacian
from scipy.sparse.linalg import spsolve


def mesh_to_poly(mesh) -> pv.PolyData:
    if isinstance(mesh, pv.PolyData):
        return mesh
    elif isinstance(mesh, tuple):
        return pv.make_tri_mesh(*mesh)
    elif hasattr(mesh, "polydata"):
        return mesh.polydata
    elif hasattr(mesh, "vertices") and hasattr(mesh, "faces"):
        return pv.make_tri_mesh(mesh.vertices, mesh.faces)
    else:
        raise ValueError("Invalid mesh input.")


def mesh_to_adjacency(mesh) -> csr_array:
    # TODO only use here because this is faster than numpy unique for unique extracting
    # edges, should be some other way to do this
    poly = mesh_to_poly(mesh)
    edge_data = poly.extract_all_edges(use_all_points=True, clear_data=True)
    lines = edge_data.lines
    edges = lines.reshape(-1, 3)[:, 1:]
    vertices = poly.points
    n_vertices = len(vertices)

    edge_lengths = np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1)

    # was having some issue here with dijkstra, so casting to intc
    # REF: https://github.com/scipy/scipy/issues/20904
    # REF: related https://github.com/scipy/scipy/issues/20817
    adj = csr_array(
        (edge_lengths, (edges[:, 0].astype(np.intc), edges[:, 1].astype(np.intc))),
        shape=(n_vertices, n_vertices),
    )

    return adj


def label_smoothing_analytic(laplacian, Y, alpha=0.9):
    # one hot encode labels

    I = eye_array(laplacian.shape[0])
    invertee = I - alpha * laplacian

    F = spsolve(invertee, Y)

    F = (1 - alpha) * F

    return F


adj = mesh_to_adjacency(simple_mesh)

lap = laplacian(adj, normed=True, return_diag=False, symmetrized=True)

# %%

Y = np.zeros((len(simple_mesh[0]), 2))
Y[mesh_compartments == 3, 0] = 1
Y[mesh_compartments == 2, 1] = 1

import time

currtime = time.time()

smoothed_labels = label_smoothing_analytic(lap, Y)
smoothed_labels = np.argmax(smoothed_labels, axis=1)

print(f"{time.time() - currtime:.3f} seconds elapsed.")

# %%
# box = pv.Box(bounds=bounds.flatten(order="F"))

simple_poly = pv.make_tri_mesh(*simple_mesh)
# simple_poly["compartment"] = mesh_compartments


# cut_simple_poly = simple_poly.clip_surface(box, invert=False)
# cut_simple_poly = cut_simple_poly.connectivity()
import seaborn as sns

colors = sns.color_palette("husl", 2)
# colors = ["coral", "lightblue"]

from matplotlib.colors import to_rgb

# colors to rgb
# pal = {0: colors[0], 1: colors[1]}
pal = dict(zip(np.unique(mesh_compartments), colors))
pal = {2: to_rgb("white"), 3: to_rgb("lightgrey")}
rgb = np.array([pal[i] for i in mesh_compartments])
rgb = rgb * 255
rgb = rgb.astype(np.uint8)
# rgb = rgb.astype(float)

plotter = pv.Plotter()

# plotter.add_mesh(pv.make_tri_mesh(*simple_mesh), color="black")

plotter.add_mesh(
    simple_poly,
    # opacity=0.3,
    # scalars=rgb,
    # scalars=mesh_compartments.astype("float32"),
    scalars=rgb,
    rgb=True,
    # rgb=True,
    # color = 'blue',
    show_scalar_bar=False,
    interpolate_before_map=False,
)

# plotter.add_mesh(
#     skel_poly,
#     scalars="compartment",
#     show_scalar_bar=False,
#     cmap="tab10",
#     interpolate_before_map=False,
# )

# plotter.add_mesh(box, color="red", opacity=0.1)

plotter.camera.focal_point = cut_pos

plotter.show()

plotter.export_gltf(
    f"{root_id}.gltf", inline_data=True, rotate_scene=True, save_normals=False
)


# %%
row = client.materialize.query_view(
    "nucleus_detection_lookup_v1",
    filter_equal_dict={"pt_root_id": root_id},
    materialization_version=943,
    desired_resolution=[1, 1, 1],
).iloc[0]
center = np.array(row["pt_position"])

#%%

new_vertices = simple_mesh[0].copy()
# recenter to soma
new_vertices -= center
# flip z and y
new_vertices = new_vertices[:, [0, 2, 1]]
# new y needs to be flipped
new_vertices[:, 2] = new_vertices[:, 2] * -1

mesh = (new_vertices, simple_mesh[1])

import pyvista as pv

simple_poly = pv.make_tri_mesh(*mesh)

axon = simple_poly.extract_points(mesh_compartments == 2).extract_surface()
dendrite = simple_poly.extract_points(mesh_compartments == 3).extract_surface()

plotter = pv.Plotter()

plotter.add_mesh(axon, color="white")
# plotter.add_mesh(dendrite, color="dimgrey")
plotter.add_mesh(dendrite, color="#3d3d3d")

plotter.camera.focal_point = cut_pos
plotter.set_background("black")

# plotter.show()
plotter.export_gltf(
    f"{root_id}.gltf", inline_data=True, rotate_scene=True, save_normals=False
)


# %%


# poly["x"] = poly.points[:, 0] > poly.points[:, 0].mean()
poly["compartment"] = simple_poly["compartment"]

plotter = pv.Plotter()

plotter.add_mesh(poly, scalars=rgb, show_scalar_bar=False, rgb=True)

plotter.camera.focal_point = (0, 0, 0)

# plotter.show()

plotter.export_gltf(
    f"{root_id}.gltf", inline_data=False, rotate_scene=True, save_normals=False
)
# python3 -m http.server 9000
# # %%
# from meshio import Mesh

# Mesh(simple_mesh[0].astype("float32"), {"triangle": simple_mesh[1]}).write(
#     f"{root_id}.vtu"
# )

# %%

pre_syns = client.materialize.synapse_query(
    pre_ids=root_id, desired_resolution=[1, 1, 1], split_positions=True
)
post_syns = client.materialize.synapse_query(
    post_ids=root_id, desired_resolution=[1, 1, 1], split_positions=True
)

pre_points = pre_syns[
    plotter.export_gltf(
        f"{root_id}.gltf", inline_data=True, rotate_scene=True, save_normals=False
    )["ctr_pt_position_x", "ctr_pt_position_y", "ctr_pt_position_z"]
].values
post_points = post_syns[
    ["ctr_pt_position_x", "ctr_pt_position_y", "ctr_pt_position_z"]
].values


def transform_points(points):
    points = points.copy()
    points -= center
    points = points[:, [0, 2, 1]]
    points[:, 2] = points[:, 2] * -1
    return points


pre_points = transform_points(pre_points)
post_points = transform_points(post_points)

# %%

pre_spheres = []
post_spheres = []

size = 400
for point in pre_points:
    pre_spheres.append(
        pv.Sphere(radius=size, center=point, theta_resolution=10, phi_resolution=10)
    )

for point in post_points:
    post_spheres.append(
        pv.Sphere(radius=size, center=point, theta_resolution=10, phi_resolution=10)
    )

pre_spheres = pv.MultiBlock(pre_spheres)
post_spheres = pv.MultiBlock(post_spheres)
pre_spheres = pre_spheres.combine().triangulate().extract_surface()
post_spheres = post_spheres.combine().triangulate().extract_surface()

# %%
plotter = pv.Plotter()

plotter.add_mesh(poly, scalars=rgb, show_scalar_bar=False, rgb=True)

plotter.add_mesh(
    pre_spheres, color="coral", opacity=1, emissive=0.9, specular=0.1, diffuse=0.1
)
plotter.add_mesh(
    post_spheres, color="lightblue", opacity=1, emissive=0.9, specular=0.1, diffuse=0.1
)
# plotter.add_points(pre_points, point_size=100, color="red")
# plotter.add_points(
#     post_points,  point_size=100, color="blue"
# )

plotter.camera.focal_point = (0, 0, 0)

# plotter.show()
plotter.export_gltf(
    f"{root_id}.gltf", inline_data=True, rotate_scene=True, save_normals=False
)

# %%
