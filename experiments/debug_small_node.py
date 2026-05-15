# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "caveclient>=8.0.1",
#     "cloud-volume>=12.13.1",
#     "meshio>=5.3.5",
#     "meshmash>=0.1.0",
#     "pymeshfix>=0.18.1",
#     "pyvista[all]>=0.47.3",
#     "scipy>=1.15.3",
#     "triangle>=20250106",
#     "trimesh>=4.12.1",
# ]
# ///

"""
Diagnostic script for debugging small border node meshes that fail
to become watertight. Reads meshes from local PLY cache.
"""

# %%

import sys
from pathlib import Path

sys.path.insert(0, ".")  # ensure local imports work

from collections import defaultdict

import meshio
import numpy as np
import pyvista as pv
import trimesh
from caveclient import CAVEclient
from meshmash import mesh_to_poly, poly_to_mesh

mesh_path = Path("/Users/ben.pedigo/code/testbed/data/meshes")


def clean_mesh(mesh, tolerance=5):
    poly = mesh_to_poly(mesh)
    poly.clean(
        tolerance=tolerance, absolute=True, lines_to_points=True, point_merging=True
    )
    return poly_to_mesh(poly)


def node_bbox(node_id, cv, adjust_draco=True):
    chunk_grid = np.array(cv.meta.decode_chunk_position(node_id))
    layer = cv.meta.decode_layer_id(node_id)
    if getattr(cv.meta, "chunks_start_at_voxel_offset", False):
        base_location = cv.meta.voxel_offset(0) * cv.mip_resolution(0)
    else:
        base_location = np.array([0, 0, 0])
    layer_scale = 2 ** (layer - 2)
    chunk_start = (
        base_location
        + chunk_grid * cv.meta.graph_chunk_size * cv.mip_resolution(0) * layer_scale
    )
    chunk_dims = cv.meta.graph_chunk_size * cv.mip_resolution(0) * layer_scale
    chunk_end = chunk_start + chunk_dims
    if adjust_draco:
        draco_size = cv.meta.get_draco_grid_size(0)
        chunk_start = draco_size * np.ceil(chunk_start / draco_size)
        chunk_end = draco_size * np.floor(chunk_end / draco_size)
    return np.stack([chunk_start, chunk_end])


def vertices_on_bbox_surface(vertices, bbox, cv):
    draco_grid_size = cv.meta.get_draco_grid_size(0)
    tol = draco_grid_size / 2
    inside = np.all((vertices >= bbox[0] - tol) & (vertices <= bbox[1] + tol), axis=1)
    on_any_face = np.zeros(len(vertices), dtype=bool)
    for axis in range(3):
        dist_to_min = np.abs(vertices[:, axis] - bbox[0, axis])
        dist_to_max = np.abs(vertices[:, axis] - bbox[1, axis])
        on_any_face |= (dist_to_min < tol) | (dist_to_max < tol)
    return np.flatnonzero(inside & on_any_face)


def _extract_boundary_loops(faces):
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


def _classify_vertex_to_face(vertex, bbox, tol):
    best_axis, best_side, best_dist = 0, 0, np.inf
    for axis in range(3):
        d_min = abs(vertex[axis] - bbox[0, axis])
        d_max = abs(vertex[axis] - bbox[1, axis])
        if d_min < best_dist:
            best_dist, best_axis, best_side = d_min, axis, 0
        if d_max < best_dist:
            best_dist, best_axis, best_side = d_max, axis, 1
    return (best_axis, best_side)


def _bbox_edge_point(v1, v2, bbox, axis, side):
    plane_val = bbox[side, axis]
    d = v2[axis] - v1[axis]
    if abs(d) < 1e-12:
        return None
    t = (plane_val - v1[axis]) / d
    if t < 0 or t > 1:
        return None
    return v1 + t * (v2 - v1)


def _split_loop_by_face(loop_vertices, bbox, tol):
    n = len(loop_vertices)
    classifications = [
        _classify_vertex_to_face(loop_vertices[i], bbox, tol) for i in range(n)
    ]
    if len(set(classifications)) == 1:
        return {classifications[0]: [loop_vertices]}
    face_segments = {}
    current_face = classifications[0]
    current_segment = [loop_vertices[0]]
    for i in range(n):
        next_i = (i + 1) % n
        this_face = classifications[i]
        next_face = classifications[next_i]
        if this_face == next_face:
            if next_i != 0:
                current_segment.append(loop_vertices[next_i])
            continue
        v1 = loop_vertices[i]
        v2 = loop_vertices[next_i]
        if this_face[0] == next_face[0]:
            mid = (v1 + v2) / 2
            current_segment.append(mid)
            face_segments.setdefault(current_face, []).append(np.array(current_segment))
            current_face = next_face
            current_segment = [mid]
            if next_i != 0:
                current_segment.append(loop_vertices[next_i])
        else:
            cross_a = _bbox_edge_point(v1, v2, bbox, next_face[0], next_face[1])
            cross_b = _bbox_edge_point(v1, v2, bbox, this_face[0], this_face[1])
            if cross_a is not None and cross_b is not None:
                d_a = np.linalg.norm(cross_a - v1)
                d_b = np.linalg.norm(cross_b - v1)
                if d_b < d_a:
                    corner = np.copy(cross_b)
                    corner[next_face[0]] = bbox[next_face[1], next_face[0]]
                    current_segment.append(cross_b)
                    current_segment.append(corner)
                    face_segments.setdefault(current_face, []).append(
                        np.array(current_segment)
                    )
                    current_face = next_face
                    current_segment = [corner, cross_a]
                else:
                    corner = np.copy(cross_a)
                    corner[this_face[0]] = bbox[this_face[1], this_face[0]]
                    current_segment.append(cross_a)
                    current_segment.append(corner)
                    face_segments.setdefault(current_face, []).append(
                        np.array(current_segment)
                    )
                    current_face = next_face
                    current_segment = [corner, cross_b]
            else:
                cross = cross_a if cross_a is not None else cross_b
                if cross is None:
                    cross = (v1 + v2) / 2
                current_segment.append(cross)
                face_segments.setdefault(current_face, []).append(
                    np.array(current_segment)
                )
                current_face = next_face
                current_segment = [cross]
            if next_i != 0:
                current_segment.append(loop_vertices[next_i])
    if len(current_segment) > 0:
        face_segments.setdefault(current_face, []).append(np.array(current_segment))
    merged = {}
    for face_key, segments in face_segments.items():
        all_pts = np.concatenate(segments, axis=0)
        diffs = np.linalg.norm(np.diff(all_pts, axis=0), axis=1)
        keep = np.concatenate([[True], diffs > 1e-6])
        all_pts = all_pts[keep]
        if len(all_pts) > 1 and np.linalg.norm(all_pts[0] - all_pts[-1]) < 1e-6:
            all_pts = all_pts[:-1]
        if len(all_pts) >= 3:
            merged[face_key] = [all_pts]
    return merged


def _triangulate_face_polygon(polygon_3d, axis):
    import triangle as tr

    axes_2d = [a for a in range(3) if a != axis]
    pts_2d = polygon_3d[:, axes_2d]
    n = len(pts_2d)
    if n < 3:
        return np.empty((0, 3), dtype=int)
    keep = [0]
    for i in range(1, n):
        if np.linalg.norm(pts_2d[i] - pts_2d[keep[-1]]) > 1e-6:
            keep.append(i)
    if len(keep) > 1 and np.linalg.norm(pts_2d[keep[-1]] - pts_2d[keep[0]]) < 1e-6:
        keep = keep[:-1]
    keep = np.array(keep)
    if len(keep) < 3:
        return np.empty((0, 3), dtype=int)
    pts_2d = pts_2d[keep]
    n = len(pts_2d)

    # Check for non-consecutive duplicate points (pinched polygon).
    # The triangle C library segfaults on these.
    from scipy.spatial import cKDTree

    tree = cKDTree(pts_2d)
    dup_pairs = tree.query_pairs(r=1e-6)
    if len(dup_pairs) > 0:
        for i, j in dup_pairs:
            pts_2d[j] += np.array([1e-4, 1e-4])

    segments = np.array([[i, (i + 1) % n] for i in range(n)])
    tri_input = {"vertices": pts_2d.astype(np.float64), "segments": segments}
    tri_output = tr.triangulate(tri_input, "p")
    if "triangles" not in tri_output:
        return np.empty((0, 3), dtype=int)
    return keep[tri_output["triangles"]]


def cap_mesh_at_bbox(mesh, bbox, cv, verbose=False):
    vertices, faces = mesh
    draco_grid_size = cv.meta.get_draco_grid_size(0)
    tol = draco_grid_size / 2
    surface_mask = np.zeros(len(vertices), dtype=bool)
    surface_inds = vertices_on_bbox_surface(vertices, bbox, cv)
    surface_mask[surface_inds] = True
    loops = _extract_boundary_loops(faces)
    if verbose:
        print(f"  cap_mesh_at_bbox: found {len(loops)} boundary loops")
    new_verts_list = [vertices]
    new_faces_list = [faces]
    next_vert_idx = len(vertices)
    for li, loop in enumerate(loops):
        loop_arr = np.array(loop)
        on_surface_count = np.sum(surface_mask[loop_arr])
        frac = on_surface_count / len(loop_arr) if len(loop_arr) > 0 else 0
        if verbose:
            loop_verts = vertices[loop_arr]
            extent = loop_verts.max(axis=0) - loop_verts.min(axis=0)
            print(
                f"  Loop {li}: {len(loop)} verts, {on_surface_count} on surface ({frac * 100:.0f}%), extent={extent}"
            )
        if on_surface_count < len(loop_arr) * 0.5:
            if verbose:
                print("    SKIPPED (< 50% on surface)")
            continue
        loop_verts = vertices[loop_arr]
        face_polygons = _split_loop_by_face(loop_verts, bbox, tol)
        if verbose:
            print(f"    Split into {len(face_polygons)} face groups:")
            for (axis, side), polygons in face_polygons.items():
                for p in polygons:
                    print(f"      axis={axis}, side={side}: {len(p)} points")
        for (axis, side), polygons in face_polygons.items():
            for polygon_3d in polygons:
                if len(polygon_3d) < 3:
                    continue
                polygon_3d = polygon_3d.copy()
                polygon_3d[:, axis] = bbox[side, axis]
                tri_indices = _triangulate_face_polygon(polygon_3d, axis)
                if verbose:
                    print(
                        f"      Triangulated: {len(tri_indices)} triangles from {len(polygon_3d)} verts"
                    )
                if len(tri_indices) == 0:
                    continue
                _poly_to_mesh_map = np.empty(len(polygon_3d), dtype=int)
                for pi, pv_coord in enumerate(polygon_3d):
                    dists = np.linalg.norm(vertices[loop_arr] - pv_coord, axis=1)
                    min_dist_idx = np.argmin(dists)
                    if dists[min_dist_idx] < tol:
                        _poly_to_mesh_map[pi] = loop_arr[min_dist_idx]
                    else:
                        new_verts_list.append(pv_coord.reshape(1, 3))
                        _poly_to_mesh_map[pi] = next_vert_idx
                        next_vert_idx += 1
                cap_faces = _poly_to_mesh_map[tri_indices]
                new_faces_list.append(cap_faces)
    new_vertices = np.concatenate(new_verts_list, axis=0).astype(np.float32)
    new_faces = np.concatenate(new_faces_list, axis=0).astype(np.uint32)
    return new_vertices, new_faces


def fill_holes(mesh, hole_size=10000):
    poly = mesh_to_poly(mesh)
    filled = poly.fill_holes(hole_size)
    return poly_to_mesh(filled)


def fix_mesh(mesh):
    import pymeshfix

    mf = pymeshfix.MeshFix(mesh[0].astype(np.float64), mesh[1].astype(np.int32))
    mf.clean()
    out = poly_to_mesh(mf.mesh)
    if mesh[0].dtype == np.float32:
        out = (out[0].astype(np.float32), out[1])
    return out


def is_watertight(mesh):
    tmesh = trimesh.Trimesh(*mesh, process=False)
    return tmesh.is_watertight


# %%
# Setup
client = CAVEclient("minnie65_public", version=1718)
cv = client.info.segmentation_cloudvolume(
    parallel=1, progress=False, green_threads=False
)

node_id = 376120971989420649

print("Loading mesh from local cache...")
mesh_file = meshio.read(mesh_path / f"{node_id}.ply")
mesh = (mesh_file.points.astype(np.float32), mesh_file.cells_dict["triangle"])
print("Done.")
bbox = node_bbox(node_id, cv)
bbox_raw = node_bbox(node_id, cv, adjust_draco=False)
draco_grid_size = cv.meta.get_draco_grid_size(0)

# %%
# Basic info
print("=" * 60)
print(f"Node ID: {node_id}")
print(f"Layer: {cv.meta.decode_layer_id(node_id)}")
print(f"Chunk position: {cv.meta.decode_chunk_position(node_id)}")
print(f"Draco grid size: {draco_grid_size}")
print()
print(f"Raw mesh: {mesh[0].shape[0]} vertices, {mesh[1].shape[0]} faces")
print(f"Vertex dtype: {mesh[0].dtype}")
print()
print("Bbox (draco-adjusted):")
print(f"  min: {bbox[0]}")
print(f"  max: {bbox[1]}")
print(f"  size: {bbox[1] - bbox[0]}")
print("Bbox (raw):")
print(f"  min: {bbox_raw[0]}")
print(f"  max: {bbox_raw[1]}")
print(f"  size: {bbox_raw[1] - bbox_raw[0]}")
print()

# Mesh extent vs bbox
mesh_min = mesh[0].min(axis=0)
mesh_max = mesh[0].max(axis=0)
mesh_extent = mesh_max - mesh_min
print("Mesh extent:")
print(f"  min: {mesh_min}")
print(f"  max: {mesh_max}")
print(f"  size: {mesh_extent}")
print()

# How much of the bbox does the mesh occupy?
bbox_size = bbox[1] - bbox[0]
print(f"Mesh occupies {mesh_extent / bbox_size * 100} % of bbox (per axis)")
print()

# Check boundary vertices
surface_inds = vertices_on_bbox_surface(mesh[0], bbox, cv)
print(f"Vertices on bbox surface: {len(surface_inds)} / {len(mesh[0])}")
print(f"  That's {len(surface_inds) / len(mesh[0]) * 100:.1f}% of all vertices")
print()

# %%
# Step-by-step pipeline with diagnostics


def report_mesh(mesh, label):
    verts, faces = mesh
    print(f"--- {label} ---")
    print(f"  Vertices: {verts.shape[0]}, Faces: {faces.shape[0]}")

    # Boundary edges
    poly = pv.make_tri_mesh(verts, faces)
    boundary = poly.extract_feature_edges(
        boundary_edges=True,
        non_manifold_edges=False,
        feature_edges=False,
        manifold_edges=False,
    )
    nonmanifold = poly.extract_feature_edges(
        boundary_edges=False,
        non_manifold_edges=True,
        feature_edges=False,
        manifold_edges=False,
    )
    print(f"  Boundary edges: {boundary.n_lines}, points: {boundary.n_points}")
    print(
        f"  Non-manifold edges: {nonmanifold.n_lines}, points: {nonmanifold.n_points}"
    )

    # Boundary loops
    loops = _extract_boundary_loops(faces)
    print(f"  Boundary loops: {len(loops)}")
    for i, loop in enumerate(loops):
        loop_arr = np.array(loop)
        loop_verts = verts[loop_arr]
        extent = loop_verts.max(axis=0) - loop_verts.min(axis=0)
        print(f"    Loop {i}: {len(loop)} vertices, extent={extent}")

    # Watertight check
    tmesh = trimesh.Trimesh(verts, faces, process=False)
    print(f"  Watertight: {tmesh.is_watertight}")
    print(f"  Volume (if watertight): {tmesh.volume if tmesh.is_watertight else 'N/A'}")
    print()


print("=" * 60)
print("PIPELINE STEPS")
print("=" * 60)

# Step 0: Raw
report_mesh(mesh, "0. Raw mesh")

# Step 1: Clean
mesh_1 = clean_mesh(mesh, tolerance=5)
report_mesh(mesh_1, "1. After clean_mesh(tolerance=5)")

# %%
# Analyze the polygon that will be sent to triangle.triangulate()
# (this is what segfaults — inspect it WITHOUT calling triangulate)
print("=" * 60)
print("SEGFAULT ANALYSIS: polygon sent to triangle.triangulate()")
print("=" * 60)

verts_1, faces_1 = mesh_1
draco_grid_size = cv.meta.get_draco_grid_size(0)
tol = draco_grid_size / 2

surface_inds_1 = vertices_on_bbox_surface(verts_1, bbox, cv)
surface_mask = np.zeros(len(verts_1), dtype=bool)
surface_mask[surface_inds_1] = True

loops = _extract_boundary_loops(faces_1)
print(f"Boundary loops found: {len(loops)}")

for li, loop in enumerate(loops):
    loop_arr = np.array(loop)
    on_surface_count = np.sum(surface_mask[loop_arr])
    frac = on_surface_count / len(loop_arr) if len(loop_arr) > 0 else 0
    loop_verts = verts_1[loop_arr]
    extent = loop_verts.max(axis=0) - loop_verts.min(axis=0)
    print(
        f"\nLoop {li}: {len(loop)} verts, {on_surface_count} on surface ({frac * 100:.0f}%), extent={extent}"
    )

    if on_surface_count < len(loop_arr) * 0.5:
        print("  SKIPPED (< 50% on surface)")
        continue

    # Split by face
    face_polygons = _split_loop_by_face(loop_verts, bbox, tol)
    print(f"  Split into {len(face_polygons)} face groups:")

    for (axis, side), polygons in face_polygons.items():
        for pi, polygon_3d in enumerate(polygons):
            print(f"  axis={axis}, side={side}: {len(polygon_3d)} points")

            # Project to 2D (same as _triangulate_face_polygon)
            polygon_3d_snapped = polygon_3d.copy()
            polygon_3d_snapped[:, axis] = bbox[side, axis]

            axes_2d = [a for a in range(3) if a != axis]
            pts_2d = polygon_3d_snapped[:, axes_2d]

            n = len(pts_2d)
            # Dedup
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
            m = len(pts_2d_clean)
            print(f"    After dedup: {m} points (was {n})")

            if m < 3:
                print("    Too few points, would skip")
                continue

            # Check for self-intersecting segments
            def segments_intersect(p1, p2, p3, p4):
                d1 = p2 - p1
                d2 = p4 - p3
                cross_val = d1[0] * d2[1] - d1[1] * d2[0]
                if abs(cross_val) < 1e-10:
                    return False
                t = ((p3[0] - p1[0]) * d2[1] - (p3[1] - p1[1]) * d2[0]) / cross_val
                u = ((p3[0] - p1[0]) * d1[1] - (p3[1] - p1[1]) * d1[0]) / cross_val
                return 0 < t < 1 and 0 < u < 1

            n_intersections = 0
            for i in range(m):
                for j in range(i + 2, m):
                    if i == 0 and j == m - 1:
                        continue  # adjacent
                    if segments_intersect(
                        pts_2d_clean[i],
                        pts_2d_clean[(i + 1) % m],
                        pts_2d_clean[j],
                        pts_2d_clean[(j + 1) % m],
                    ):
                        n_intersections += 1
                        if n_intersections <= 5:
                            print(
                                f"    INTERSECTION: seg {i}-{(i + 1) % m} x seg {j}-{(j + 1) % m}"
                            )
                            print(
                                f"      {pts_2d_clean[i]} -> {pts_2d_clean[(i + 1) % m]}"
                            )
                            print(
                                f"      {pts_2d_clean[j]} -> {pts_2d_clean[(j + 1) % m]}"
                            )

            print(f"    Total self-intersections: {n_intersections}")

            # Check for duplicate points (non-consecutive)
            from scipy.spatial import cKDTree

            tree = cKDTree(pts_2d_clean)
            pairs = tree.query_pairs(r=1e-6)
            print(f"    Duplicate point pairs (dist < 1e-6): {len(pairs)}")
            for p in list(pairs)[:5]:
                print(
                    f"      pts {p[0]} and {p[1]}: {pts_2d_clean[p[0]]} vs {pts_2d_clean[p[1]]}"
                )

            # Check polygon area and winding
            signed_area = 0
            for i in range(m):
                j = (i + 1) % m
                signed_area += pts_2d_clean[i][0] * pts_2d_clean[j][1]
                signed_area -= pts_2d_clean[j][0] * pts_2d_clean[i][1]
            signed_area /= 2
            print(f"    Polygon signed area: {signed_area:.2f}")
            print(f"    Winding: {'CCW' if signed_area > 0 else 'CW'}")

            # Min/max edge lengths
            edge_lengths = np.array(
                [
                    np.linalg.norm(pts_2d_clean[(i + 1) % m] - pts_2d_clean[i])
                    for i in range(m)
                ]
            )
            print(
                f"    Edge lengths: min={edge_lengths.min():.4f}, max={edge_lengths.max():.4f}, median={np.median(edge_lengths):.4f}"
            )
            zero_edges = (edge_lengths < 1e-6).sum()
            print(f"    Zero-length edges: {zero_edges}")

print("\nDone — no triangulate() call, no segfault.")

# %%
# Analyze which vertices are truly on the y-max face vs off it
print("\n" + "=" * 60)
print("FACE CLASSIFICATION ANALYSIS")
print("=" * 60)

loop_0 = _extract_boundary_loops(faces_1)[0]
loop_0_arr = np.array(loop_0)
loop_0_verts = verts_1[loop_0_arr]
proj_axis = 1  # y
y_max = bbox[1, 1]
draco_grid_size = cv.meta.get_draco_grid_size(0)
tol = draco_grid_size / 2

y_dists = np.abs(loop_0_verts[:, proj_axis] - y_max)
on_face = y_dists < tol
off_face = ~on_face

print(f"Loop 0: {len(loop_0)} vertices")
print(f"  y-max face value: {y_max}")
print(f"  On y-max face (dist < {tol}): {on_face.sum()}")
print(f"  Off y-max face: {off_face.sum()}")
print()

# Show all vertices with their y-distance from the face
print("All loop vertices (with y-distance from y-max):")
for i in range(len(loop_0)):
    marker = ""
    if off_face[i]:
        marker = f" ← OFF FACE (y_dist={y_dists[i]:.0f})"
        # What face would this vertex belong to instead?
        v = loop_0_verts[i]
        best_axis, best_side, best_dist = 0, 0, np.inf
        for ax in range(3):
            d_min = abs(v[ax] - bbox[0, ax])
            d_max = abs(v[ax] - bbox[1, ax])
            if d_min < best_dist:
                best_dist, best_axis, best_side = d_min, ax, 0
            if d_max < best_dist:
                best_dist, best_axis, best_side = d_max, ax, 1
        face_name = f"axis={best_axis} side={best_side}"
        marker += f" (closest face: {face_name}, dist={best_dist:.0f})"
    print(f"  [{i:2d}] {loop_0_verts[i]}  y_dist={y_dists[i]:6.0f}{marker}")

# Show the classification for _classify_vertex_to_face for each vertex
print("\nFace classifications via _classify_vertex_to_face:")
from collections import Counter

classifs = [
    _classify_vertex_to_face(loop_0_verts[i], bbox, tol) for i in range(len(loop_0))
]
counts = Counter(classifs)
for face, count in counts.most_common():
    print(f"  axis={face[0]}, side={face[1]}: {count} vertices")

print()
print("Vertices NOT classified to y-max (axis=1, side=1):")
for i, c in enumerate(classifs):
    if c != (1, 1):
        print(f"  [{i:2d}] {loop_0_verts[i]} → axis={c[0]}, side={c[1]}")
