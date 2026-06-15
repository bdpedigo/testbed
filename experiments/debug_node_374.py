# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "caveclient>=8.0.1",
#     "cloud-volume>=12.13.1",
#     "meshio>=5.3.5",
#     "meshmash>=0.1.0",
#     "numpy",
#     "scipy>=1.15.3",
#     "trimesh>=4.6.0",
#     "triangle>=20250106",
# ]
# ///
"""Debug node 374427724082633558 — segfaults during capping after dedup."""

from pathlib import Path

import meshio
import numpy as np
import trimesh as tm
from caveclient import CAVEclient
from meshmash import mesh_to_poly, poly_to_mesh
from scipy.spatial import cKDTree

mesh_path = Path("/Users/ben.pedigo/code/testbed/data/meshes")

client = CAVEclient("minnie65_public", version=1718)
cv = client.info.segmentation_cloudvolume(progress=False)

node_id = 374427724082633558

mesh = meshio.read(mesh_path / f"{node_id}.ply")
mesh = (mesh.points, mesh.cells_dict["triangle"])


# --- Inline needed functions ---


def clean_mesh(mesh, tolerance=5):
    poly = mesh_to_poly(mesh)
    poly.clean(
        tolerance=tolerance, absolute=True, lines_to_points=True, point_merging=True
    )
    return poly_to_mesh(poly)


def remove_degenerate_faces(mesh):
    verts, faces = mesh
    degenerate = (
        (faces[:, 0] == faces[:, 1])
        | (faces[:, 1] == faces[:, 2])
        | (faces[:, 0] == faces[:, 2])
    )
    if degenerate.any():
        faces = faces[~degenerate]
    return (verts, faces)


def get_chunk_grid_params(cv):
    chunk_size = np.array(cv.meta.graph_chunk_size) * np.array(cv.mip_resolution(0))
    draco_grid_size = cv.meta.get_draco_grid_size(0)
    if getattr(cv.meta, "chunks_start_at_voxel_offset", False):
        offset = np.array(cv.meta.voxel_offset(0)) * np.array(cv.mip_resolution(0))
    else:
        offset = np.array([0, 0, 0], dtype=float)
    return chunk_size, draco_grid_size, offset


def deduplicate_chunk_boundaries(mesh, chunk_size, draco_grid_size, offset):
    vertices, faces = mesh
    shifted = vertices - offset
    dist_behind = np.mod(shifted, chunk_size)
    dist_ahead = chunk_size - dist_behind
    is_on_behind = np.any(dist_behind < (draco_grid_size / 2), axis=1)
    is_on_ahead = np.any(dist_ahead <= (draco_grid_size / 2), axis=1)
    is_chunk_aligned = is_on_behind | is_on_ahead

    _, unique_inverse, counts = np.unique(
        vertices, return_inverse=True, return_counts=True, axis=0
    )
    only_double = np.where(counts == 2)[0]
    is_doubled = np.isin(unique_inverse, only_double)

    do_merge = is_doubled & is_chunk_aligned

    if not np.any(do_merge):
        return mesh

    n = len(vertices)
    tag_col = np.arange(n, dtype=np.float64).reshape(-1, 1)
    tag_col[do_merge] = -1
    tagged = np.hstack([vertices, tag_col])

    face_verts = tagged[faces.flatten()]
    new_verts_4d, new_faces_flat = np.unique(face_verts, return_inverse=True, axis=0)
    new_vertices = new_verts_4d[:, :3].astype(vertices.dtype)
    new_faces = new_faces_flat.astype(np.uint32).reshape(-1, 3)

    return (new_vertices, new_faces)


def classify_vertices_to_chunk_planes(vertices, chunk_size, draco_grid_size, offset):
    tol = draco_grid_size / 2
    shifted = vertices - offset
    dist_behind = np.mod(shifted, chunk_size)
    dist_ahead = chunk_size - dist_behind
    on_behind = dist_behind < tol
    on_ahead = dist_ahead <= tol
    on_plane = on_behind | on_ahead
    plane_behind = shifted - dist_behind + offset
    plane_ahead = shifted + dist_ahead + offset
    plane_values = np.where(dist_behind < dist_ahead, plane_behind, plane_ahead)
    return on_plane, plane_values


# --- Process the node ---

chunk_size, draco_grid_size, offset = get_chunk_grid_params(cv)
print(f"chunk_size={chunk_size}, draco_grid_size={draco_grid_size}")

mesh = remove_degenerate_faces(mesh)
print(f"Original: {mesh[0].shape[0]} verts, {mesh[1].shape[0]} faces")

mesh = deduplicate_chunk_boundaries(mesh, chunk_size, draco_grid_size, offset)
mesh = clean_mesh(mesh, tolerance=5)
print(f"After dedup+clean: {mesh[0].shape[0]} verts, {mesh[1].shape[0]} faces")

# Find boundary loops
trimesh_mesh = tm.Trimesh(*mesh, process=False)
boundary_edges = trimesh_mesh.edges[
    tm.grouping.group_rows(trimesh_mesh.edges_sorted, require_count=1)
]
print(f"Boundary edges: {len(boundary_edges)}")

if len(boundary_edges) == 0:
    print("No boundary — mesh is already closed!")
    import sys

    sys.exit(0)

# Trace loops
from collections import defaultdict

adjacency = defaultdict(list)
for e in boundary_edges:
    adjacency[e[0]].append(e[1])
    adjacency[e[1]].append(e[0])

visited_edges = set()
loops = []
for start in adjacency:
    if start in visited_edges:
        continue
    loop = [start]
    visited_edges.add(start)
    current = start
    while True:
        neighbors = adjacency[current]
        next_node = None
        for n in neighbors:
            if n not in visited_edges:
                next_node = n
                break
        if next_node is None:
            break
        loop.append(next_node)
        visited_edges.add(next_node)
        current = next_node
    loops.append(loop)

print(f"\nFound {len(loops)} boundary loops")

vertices = mesh[0]
on_plane, plane_values = classify_vertices_to_chunk_planes(
    vertices, chunk_size, draco_grid_size, offset
)

for i, loop in enumerate(loops):
    loop_verts = vertices[loop]
    loop_on_plane = on_plane[loop]
    loop_plane_vals = plane_values[loop]

    # Which planes are represented?
    planes_in_loop = set()
    for vi, (on, pv) in enumerate(zip(loop_on_plane, loop_plane_vals)):
        for axis in range(3):
            if on[axis]:
                planes_in_loop.add((axis, pv[axis]))

    n_on = loop_on_plane.any(axis=1).sum()
    print(f"\nLoop {i}: {len(loop)} verts, {n_on} on chunk planes")
    print(f"  Planes: {planes_in_loop}")

    # Check for pinching (duplicate 2D positions after projection)
    for axis, pval in planes_in_loop:
        # Get verts on this plane
        mask = loop_on_plane[:, axis]
        plane_loop_verts = loop_verts[mask]
        # Project to 2D
        axes_2d = [a for a in range(3) if a != axis]
        pts_2d = plane_loop_verts[:, axes_2d]
        if len(pts_2d) < 3:
            continue
        tree = cKDTree(pts_2d)
        pairs = tree.query_pairs(r=1e-6)
        if pairs:
            print(
                f"  PINCHED on axis={axis}, plane={pval}: {len(pairs)} duplicate 2D pairs out of {len(pts_2d)} verts"
            )
        else:
            print(
                f"  OK on axis={axis}, plane={pval}: {len(pts_2d)} verts, no duplicates"
            )
