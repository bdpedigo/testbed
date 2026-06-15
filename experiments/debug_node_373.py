# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "caveclient>=8.0.1",
#     "cloud-volume>=12.13.1",
#     "meshio>=5.3.5",
#     "meshmash>=0.1.0",
#     "numpy",
#     "pyvista[all]>=0.47.3",
#     "scipy>=1.15.3",
#     "triangle>=20250106",
# ]
# ///
"""Debug node 373864774129237031 — understand why _split_loop_by_plane produces
a pinched polygon that segfaults triangle."""

from pathlib import Path

import meshio
import numpy as np
from caveclient import CAVEclient
from meshmash import mesh_to_poly, poly_to_mesh
from scipy.spatial import cKDTree

mesh_path = Path("/Users/ben.pedigo/code/testbed/data/meshes")

client = CAVEclient("minnie65_public", version=1718)
cv = client.info.segmentation_cloudvolume(progress=False)

node_id = 373864774129237031

mesh = meshio.read(mesh_path / f"{node_id}.ply")
mesh = (mesh.points, mesh.cells_dict["triangle"])


# --- Inline the needed functions to avoid importing the whole module ---


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


def classify_vertex_to_chunk_plane(vertex, chunk_size, draco_grid_size, offset):
    tol = draco_grid_size / 2
    shifted = vertex - offset
    dist_behind = np.mod(shifted, chunk_size)
    dist_ahead = chunk_size - dist_behind
    best_axis = None
    best_dist = np.inf
    best_plane_value = None
    for axis in range(3):
        d_behind = dist_behind[axis]
        d_ahead = dist_ahead[axis]
        if d_behind < tol and d_behind < best_dist:
            best_dist = d_behind
            best_axis = axis
            best_plane_value = vertex[axis] - d_behind
        if d_ahead <= tol and d_ahead < best_dist:
            best_dist = d_ahead
            best_axis = axis
            best_plane_value = vertex[axis] + d_ahead
    if best_axis is None:
        return None
    return (best_axis, best_plane_value)


def _extract_boundary_loops(faces):
    from collections import defaultdict

    edge_count = defaultdict(list)
    for fi, face in enumerate(faces):
        for i in range(3):
            e = tuple(sorted((face[i], face[(i + 1) % 3])))
            edge_count[e].append(fi)
    boundary_edges = {e for e, flist in edge_count.items() if len(flist) == 1}
    if not boundary_edges:
        return []
    adj = defaultdict(set)
    for a, b in boundary_edges:
        adj[a].add(b)
        adj[b].add(a)
    visited_edges = set()
    loops = []
    for start in adj:
        if all(tuple(sorted((start, n))) in visited_edges for n in adj[start]):
            continue
        loop = [start]
        prev = start
        current = None
        for n in adj[start]:
            if tuple(sorted((start, n))) not in visited_edges:
                current = n
                break
        if current is None:
            continue
        visited_edges.add(tuple(sorted((prev, current))))
        loop.append(current)
        while current != start:
            next_v = None
            for n in adj[current]:
                if n == prev:
                    continue
                if tuple(sorted((current, n))) not in visited_edges:
                    next_v = n
                    break
            if next_v is None:
                break
            visited_edges.add(tuple(sorted((current, next_v))))
            prev = current
            current = next_v
            if current != start:
                loop.append(current)
        if current == start and len(loop) >= 3:
            loops.append(loop)
    return loops


def _intersect_segment_plane(v1, v2, axis, plane_value):
    d = v2[axis] - v1[axis]
    if abs(d) < 1e-12:
        return None
    t = (plane_value - v1[axis]) / d
    if t < 0 or t > 1:
        return None
    return v1 + t * (v2 - v1)


def _split_loop_by_plane(loop_vertices, chunk_size, draco_grid_size, offset):
    n = len(loop_vertices)
    classifications = [
        classify_vertex_to_chunk_plane(
            loop_vertices[i], chunk_size, draco_grid_size, offset
        )
        for i in range(n)
    ]
    non_none = [c for c in classifications if c is not None]
    if not non_none:
        return {}
    if len(set(non_none)) == 1 and all(c is not None for c in classifications):
        return {non_none[0]: [loop_vertices.copy()]}

    plane_segments = {}
    current_plane = None
    current_segment = []

    def _flush_segment():
        nonlocal current_segment, current_plane
        if current_plane is not None and len(current_segment) >= 1:
            if current_plane not in plane_segments:
                plane_segments[current_plane] = []
            plane_segments[current_plane].append(current_segment)
        current_segment = []

    start_offset = 0
    for i in range(n):
        if classifications[i] is not None:
            start_offset = i
            break
    else:
        return {}

    order = [(start_offset + i) % n for i in range(n)]
    current_plane = classifications[order[0]]
    current_segment = [loop_vertices[order[0]]]

    for idx in range(len(order)):
        i = order[idx]
        next_idx = (idx + 1) % n
        j = order[next_idx]
        this_class = classifications[i]
        next_class = classifications[j]

        if this_class is None:
            continue
        if next_class is None:
            _flush_segment()
            current_plane = None
            continue
        if this_class == next_class:
            if next_idx != 0:
                current_segment.append(loop_vertices[j])
            continue

        v1 = loop_vertices[i]
        v2 = loop_vertices[j]
        this_axis, this_pv = this_class
        next_axis, next_pv = next_class

        if this_axis == next_axis:
            mid = (v1 + v2) / 2
            current_segment.append(mid)
            _flush_segment()
            current_plane = next_class
            current_segment = [mid]
            if next_idx != 0:
                current_segment.append(loop_vertices[j])
        else:
            cross_next = _intersect_segment_plane(v1, v2, next_axis, next_pv)
            cross_this = _intersect_segment_plane(v1, v2, this_axis, this_pv)

            if cross_this is not None and cross_next is not None:
                d_this = np.linalg.norm(cross_this - v1)
                d_next = np.linalg.norm(cross_next - v1)
                if d_this < d_next:
                    corner = np.copy(cross_this)
                    corner[next_axis] = next_pv
                    current_segment.append(cross_this)
                    current_segment.append(corner)
                    _flush_segment()
                    current_plane = next_class
                    current_segment = [corner, cross_next]
                else:
                    corner = np.copy(cross_next)
                    corner[this_axis] = this_pv
                    current_segment.append(cross_next)
                    current_segment.append(corner)
                    _flush_segment()
                    current_plane = next_class
                    current_segment = [corner, cross_this]
            else:
                cross = cross_next if cross_next is not None else cross_this
                if cross is None:
                    cross = (v1 + v2) / 2
                current_segment.append(cross)
                _flush_segment()
                current_plane = next_class
                current_segment = [cross]

            if next_idx != 0:
                current_segment.append(loop_vertices[j])

    _flush_segment()

    merged = {}
    for plane_key, segments in plane_segments.items():
        all_pts = np.concatenate([np.array(s) for s in segments], axis=0)
        diffs = np.linalg.norm(np.diff(all_pts, axis=0), axis=1)
        keep = np.concatenate([[True], diffs > 1e-6])
        all_pts = all_pts[keep]
        if len(all_pts) > 1 and np.linalg.norm(all_pts[0] - all_pts[-1]) < 1e-6:
            all_pts = all_pts[:-1]
        if len(all_pts) >= 3:
            merged[plane_key] = [all_pts]
    return merged


# Prepare mesh same as pipeline
mesh = remove_degenerate_faces(mesh)
chunk_size, draco_grid_size, offset = get_chunk_grid_params(cv)
mesh = clean_mesh(mesh, tolerance=draco_grid_size / 2)
vertices, faces = mesh

tol = draco_grid_size / 2

# Classify all vertices
on_plane, plane_values = classify_vertices_to_chunk_planes(
    vertices, chunk_size, draco_grid_size, offset
)
on_any_plane = np.any(on_plane, axis=1)

# Extract boundary loops
loops = _extract_boundary_loops(faces)
print(f"Found {len(loops)} boundary loops")

for loop_idx, loop in enumerate(loops):
    loop_arr = np.array(loop)
    on_plane_count = np.sum(on_any_plane[loop_arr])
    frac = on_plane_count / len(loop_arr)

    if frac < 0.5:
        continue

    print(
        f"\nLoop {loop_idx}: {len(loop_arr)} verts, {on_plane_count} on-plane ({frac:.1%})"
    )

    loop_verts = vertices[loop_arr]

    # Classify each vertex
    classifications = [
        classify_vertex_to_chunk_plane(
            loop_verts[i], chunk_size, draco_grid_size, offset
        )
        for i in range(len(loop_verts))
    ]

    # Show classification summary
    planes_seen = {}
    none_count = 0
    for i, c in enumerate(classifications):
        if c is None:
            none_count += 1
        else:
            key = (c[0], round(c[1], 1))
            if key not in planes_seen:
                planes_seen[key] = []
            planes_seen[key].append(i)

    print(f"  None (off-plane): {none_count} vertices")
    for key, indices in sorted(planes_seen.items()):
        axis_name = "xyz"[key[0]]
        print(f"  {axis_name}={key[1]}: {len(indices)} vertices")

    # Now split and check for pinched polygons
    plane_polygons = _split_loop_by_plane(
        loop_verts, chunk_size, draco_grid_size, offset
    )

    for (axis, pv_val), polygons in plane_polygons.items():
        for poly_idx, polygon_3d in enumerate(polygons):
            # Simulate what _triangulate_face_polygon does
            polygon_3d = polygon_3d.copy()
            polygon_3d[:, axis] = pv_val

            axes_2d = [a for a in range(3) if a != axis]
            pts_2d = polygon_3d[:, axes_2d]

            n = len(pts_2d)
            # Deduplicate consecutive
            keep = [0]
            for i in range(1, n):
                if np.linalg.norm(pts_2d[i] - pts_2d[keep[-1]]) > 1e-6:
                    keep.append(i)
            if (
                len(keep) > 1
                and np.linalg.norm(pts_2d[keep[-1]] - pts_2d[keep[0]]) < 1e-6
            ):
                keep = keep[:-1]
            keep = np.array(keep)
            pts_2d_clean = pts_2d[keep]
            n_clean = len(pts_2d_clean)

            if n_clean < 3:
                continue

            # Check for non-consecutive near-duplicates
            tree = cKDTree(pts_2d_clean)
            dup_pairs = tree.query_pairs(r=1.0)
            non_consec_dups = []
            for i, j in dup_pairs:
                if (
                    abs(i - j) != 1
                    and not (i == 0 and j == n_clean - 1)
                    and not (j == 0 and i == n_clean - 1)
                ):
                    non_consec_dups.append((i, j))

            axis_name = "xyz"[axis]
            if non_consec_dups:
                print(
                    f"\n  *** PINCHED POLYGON on {axis_name}={pv_val:.1f} ({n_clean} pts after dedup) ***"
                )
                print("      Non-consecutive duplicate pairs (dist < 1.0):")
                for i, j in non_consec_dups[:10]:
                    dist = np.linalg.norm(pts_2d_clean[i] - pts_2d_clean[j])
                    print(f"        pts[{i}] vs pts[{j}]: dist={dist:.6f}")
                    print(f"          pts[{i}] = {pts_2d_clean[i]}")
                    print(f"          pts[{j}] = {pts_2d_clean[j]}")

                # Show what the original 3D vertices look like for those points
                print("      Corresponding 3D polygon vertices:")
                for i, j in non_consec_dups[:5]:
                    orig_i = keep[i]
                    orig_j = keep[j]
                    print(f"        3D[{orig_i}] = {polygon_3d[orig_i]}")
                    print(f"        3D[{orig_j}] = {polygon_3d[orig_j]}")
                    # What were their classifications?
            else:
                print(
                    f"  Polygon on {axis_name}={pv_val:.1f}: {n_clean} pts, no pinch detected"
                )

print("\nDone.")

# %% Visualize Loop 2 (the problematic multi-plane loop)
import pyvista as pvista

# Re-extract loop 2 data
loop_idx = 2
loop = loops[loop_idx]
loop_arr = np.array(loop)
loop_verts = vertices[loop_arr]

# Classify each vertex
classifications = [
    classify_vertex_to_chunk_plane(loop_verts[i], chunk_size, draco_grid_size, offset)
    for i in range(len(loop_verts))
]

# Color by classification
colors = []
for c in classifications:
    if c is None:
        colors.append("white")
    elif c[0] == 0:  # x-plane
        colors.append("red")
    elif c[0] == 1:  # y-plane
        colors.append("blue")
    else:  # z-plane
        colors.append("green")

# Also detect which vertices are on 2+ planes simultaneously
multi_plane_indices = []
for i, v in enumerate(loop_verts):
    shifted = v - offset
    dist_behind = np.mod(shifted, chunk_size)
    dist_ahead = chunk_size - dist_behind
    n_planes = 0
    for ax in range(3):
        if dist_behind[ax] < tol or dist_ahead[ax] <= tol:
            n_planes += 1
    if n_planes >= 2:
        multi_plane_indices.append(i)

print(
    f"\nLoop 2: {len(multi_plane_indices)} vertices on 2+ planes (chunk edges/corners)"
)
for i in multi_plane_indices:
    shifted = loop_verts[i] - offset
    dist_behind = np.mod(shifted, chunk_size)
    dist_ahead = chunk_size - dist_behind
    on_axes = []
    for ax in range(3):
        if dist_behind[ax] < tol or dist_ahead[ax] <= tol:
            on_axes.append("xyz"[ax])
    print(
        f"  vert[{i}] = {loop_verts[i]} on axes: {on_axes}, classified as: {classifications[i]}"
    )

plotter = pvista.Plotter()
plotter.add_mesh(pvista.make_tri_mesh(vertices, faces), color="lightgray", opacity=0.3)

# Add loop vertices colored by plane
for i, (v, color) in enumerate(zip(loop_verts, colors)):
    plotter.add_points(v.reshape(1, 3), color=color, point_size=12)

# Highlight multi-plane vertices in yellow (larger)
if multi_plane_indices:
    plotter.add_points(loop_verts[multi_plane_indices], color="yellow", point_size=18)

# Draw edges of the loop as lines
for i in range(len(loop_verts)):
    j = (i + 1) % len(loop_verts)
    line = pvista.Line(loop_verts[i], loop_verts[j])
    plotter.add_mesh(line, color="black", line_width=2)

plotter.enable_fly_to_right_click()
plotter.camera.focal_point = loop_verts.mean(axis=0)
plotter.show()
