# /// script
# requires-python = ">=3.10,<3.11"
# dependencies = [
#     "caveclient>=8.0.1",
#     "cloud-volume>=12.13.1",
#     "ipykernel>=7.2.0",
#     "ipywidgets>=8.1.8",
#     "pygamer>=2.0.7",
#     "pymeshlab>=2025.7.post1",
#     "pyvista[all]>=0.47.3",
#     "scipy>=1.15.3",
#     "triangle>=20250106",
#     "trimesh>=4.12.1",
# ]
# ///

# NOTE: To sync this script's environment, run:
#   CMAKE_ARGS="-DCMAKE_POLICY_VERSION_MINIMUM=3.5" uv sync --script mesh_cleanup.py
#
# The CMAKE_ARGS flag is required because pygamer's bundled tetgen uses an old
# cmake_minimum_required that CMake 4.3+ no longer accepts. Python must be <3.11
# because pygamer's bundled pybind11 is incompatible with the opaque PyFrameObject
# introduced in Python 3.11.


# %%

import numpy as np
import pyvista as pv
from caveclient import CAVEclient

client = CAVEclient("minnie65_public", version=1718)

# %%
root_id = 864691136438529438

selection = "all"
node_level = 5
if selection == "synapse_neighborhood":
    pt = np.array([179959, 122568, 21903]) * np.array([4, 4, 40])

    synapses = client.materialize.synapse_query(
        post_ids=[root_id],
        desired_resolution=[1, 1, 1],
        split_positions=True,
    )

    synapse_pts = (
        synapses[["ctr_pt_position_x", "ctr_pt_position_y", "ctr_pt_position_z"]]
        .to_numpy()
        .astype(np.float64)
    )

    dists = np.linalg.norm(synapse_pts - pt, axis=1)

    sv_id = synapses.iloc[np.argsort(dists)[100]]["post_pt_supervoxel_id"]
    node_id = client.chunkedgraph.get_roots(sv_id, stop_layer=5)[0]
    node_ids = [node_id]
else:
    node_ids = client.chunkedgraph.get_leaves(root_id, stop_layer=node_level)

# %%

cv = client.info.segmentation_cloudvolume(
    parallel=20, progress=True, green_threads=False
)

# %%
meshes = cv.mesh.get(
    node_ids[:50],
    deduplicate_chunk_boundaries=False,
    remove_duplicate_vertices=False,
    fuse=False,
)


# %%


def node_bbox(node_id, cv, adjust_draco=True):
    """Get the bounding box in nm for a node ID in the chunkedgraph.

    Accounts for dataset voxel offset, mip resolution, chunk size,
    and draco grid quantization.

    Parameters
    ----------
    node_id : int
        A node ID in the chunkedgraph (e.g. a level-2 or level-4 ID).
    cv : cloudvolume.CloudVolume
        CloudVolume object for the segmentation.
    adjust_draco : bool
        Whether to snap bounds to the draco grid.

    Returns
    -------
    bbox_start : np.ndarray
        (3,) array of bounding box start in nm.
    bbox_end : np.ndarray
        (3,) array of bounding box end in nm.
    """
    # Decode chunk grid position from the node ID
    chunk_grid = np.array(cv.meta.decode_chunk_position(node_id))
    layer = cv.meta.decode_layer_id(node_id)

    # Convert chunk grid coords to nm, accounting for voxel offset and resolution
    # chunks_start_at_voxel_offset controls whether chunk grid starts at voxel_offset
    if getattr(cv.meta, "chunks_start_at_voxel_offset", False):
        base_location = cv.meta.voxel_offset(0) * cv.mip_resolution(0)
    else:
        base_location = np.array([0, 0, 0])
    # At higher layers, each chunk grid unit covers 2^(layer-2) base chunks
    layer_scale = 2 ** (layer - 2)
    chunk_start = (
        base_location
        + chunk_grid * cv.meta.graph_chunk_size * cv.mip_resolution(0) * layer_scale
    )

    # Chunk dimensions in nm (scaled for the layer)
    chunk_dims = cv.meta.graph_chunk_size * cv.mip_resolution(0) * layer_scale
    chunk_end = chunk_start + chunk_dims

    # Snap to draco quantization grid.
    # Draco rounds vertices UP, so:
    #  - chunk_start snaps UP (ceil) to where the first boundary vertices land
    #  - chunk_end snaps DOWN (floor) to where the last boundary vertices land
    if adjust_draco:
        draco_size = cv.meta.get_draco_grid_size(0)
        chunk_start = draco_size * np.ceil(chunk_start / draco_size)
        chunk_end = draco_size * np.floor(chunk_end / draco_size)

    return np.stack([chunk_start, chunk_end])


def vertices_on_bbox_surface(vertices, bbox, cv):
    """Find indices of mesh vertices that lie on the surface of a bounding box.

    Uses cloudvolume's chunk-alignment logic: the mesh is quantized to a draco
    grid, so vertices near a chunk boundary are snapped to the nearest draco
    grid point. A vertex is "on" a face if it is within draco_grid_size/2 of
    that face's plane coordinate.

    Matches cloudvolume's asymmetry: the "behind" (min) face uses strict <
    and the "ahead" (max) face uses <=, because draco rounds up.

    Parameters
    ----------
    vertices : np.ndarray
        (N, 3) array of mesh vertex positions in nm.
    bbox : np.ndarray
        (2, 3) array where bbox[0] is the min corner and bbox[1] is the max corner.
    cv : cloudvolume.CloudVolume
        CloudVolume object, used to get the draco grid size.

    Returns
    -------
    np.ndarray
        1D array of indices of vertices on the bbox surface.
    """
    draco_grid_size = cv.meta.get_draco_grid_size(0)
    # With correct ceil/floor in node_bbox, bbox faces align with draco grid.
    # Use a small epsilon for floating point comparison.
    tol = draco_grid_size / 2

    # Vertex must be inside the bbox (within tolerance)
    inside = np.all((vertices >= bbox[0] - tol) & (vertices <= bbox[1] + tol), axis=1)

    # For each axis, check if vertex is near the min or max face
    on_any_face = np.zeros(len(vertices), dtype=bool)
    for axis in range(3):
        dist_to_min = np.abs(vertices[:, axis] - bbox[0, axis])
        dist_to_max = np.abs(vertices[:, axis] - bbox[1, axis])
        on_min_face = dist_to_min < tol
        on_max_face = dist_to_max < tol
        on_any_face |= on_min_face | on_max_face

    return np.flatnonzero(inside & on_any_face)


def vertices_on_chunk_boundaries(vertices, bbox, cv):
    """Find indices of mesh vertices on ANY chunk boundary within the bbox.

    Higher-layer meshes span multiple layer-2 chunks, and the mesh has seams
    at internal chunk boundaries too. This detects all such vertices.

    Parameters
    ----------
    vertices : np.ndarray
        (N, 3) array of mesh vertex positions in nm.
    bbox : np.ndarray
        (2, 3) bounding box.
    cv : cloudvolume.CloudVolume
        CloudVolume object.

    Returns
    -------
    np.ndarray
        1D array of indices of vertices on any chunk boundary.
    """
    draco_grid_size = cv.meta.get_draco_grid_size(0)
    tol = draco_grid_size / 2

    # The base chunk size in nm
    chunk_size_nm = np.array(cv.meta.graph_chunk_size) * np.array(cv.mip_resolution(0))

    # Compute the offset: chunk boundaries start at base_location
    if getattr(cv.meta, "chunks_start_at_voxel_offset", False):
        offset = np.array(cv.meta.voxel_offset(0)) * np.array(cv.mip_resolution(0))
    else:
        offset = np.array([0, 0, 0], dtype=float)

    # Vertex must be inside the bbox (within tolerance)
    inside = np.all((vertices >= bbox[0] - tol) & (vertices <= bbox[1] + tol), axis=1)

    # Use modular distance to chunk boundaries (same as cloudvolume)
    shifted = vertices - offset
    dist_behind = np.mod(shifted, chunk_size_nm)
    dist_ahead = chunk_size_nm - dist_behind

    # Draco rounds up: behind uses <, ahead uses <=
    on_behind = np.any(dist_behind < tol, axis=1)
    on_ahead = np.any(dist_ahead <= tol, axis=1)

    return np.flatnonzero(inside & (on_behind | on_ahead))


def find_planar_boundary_loops(vertices, faces, bbox):
    """Find boundary loops that are approximately axis-aligned (planar).

    Uses PyVista to extract boundary edges, then groups connected boundary
    vertices into loops and checks planarity.

    Parameters
    ----------
    vertices : np.ndarray
        (N, 3) mesh vertex positions.
    faces : np.ndarray
        (M, 3) mesh triangle indices.
    bbox : np.ndarray
        (2, 3) bounding box.

    Returns
    -------
    list of dict
        Each dict has:
        - 'loop_indices': ndarray of original vertex indices in the loop
        - 'axis': int (which axis it's most planar on)
        - 'spread': float (spread along the planar axis, i.e. max-min)
        - 'plane_value': float (mean coordinate along the planar axis)
        - 'dist_to_bbox_min': float (distance from plane to bbox min face)
        - 'dist_to_bbox_max': float (distance from plane to bbox max face)
        - 'loop_length': int
    """
    pv_mesh = pv.make_tri_mesh(vertices, faces)

    # Extract boundary edges
    boundary = pv_mesh.extract_feature_edges(
        boundary_edges=True,
        non_manifold_edges=False,
        feature_edges=False,
        manifold_edges=False,
    )

    if boundary.n_points == 0:
        return []

    # Split into connected components (each is a loop)
    connected = boundary.connectivity(extraction_mode="all")
    region_ids = connected.point_data["RegionId"]
    n_regions = region_ids.max() + 1

    results = []
    for region_id in range(n_regions):
        region_mask = region_ids == region_id
        loop_pts = np.asarray(connected.points[region_mask])

        if len(loop_pts) < 3:
            continue

        # Find original vertex indices
        # boundary.points are a subset of mesh points; find their indices
        region_point_ids = np.flatnonzero(region_mask)
        # Map back to original mesh via point coordinates
        # (PyVista preserves point IDs through extract_feature_edges via vtkOriginalPointIds)
        if "vtkOriginalPointIds" in connected.point_data:
            orig_ids = connected.point_data["vtkOriginalPointIds"][region_mask]
        else:
            # Fallback: match by coordinates
            from scipy.spatial import cKDTree

            tree = cKDTree(vertices)
            _, orig_ids = tree.query(loop_pts)

        # Check planarity along each axis
        best_axis = None
        best_spread = np.inf
        for axis in range(3):
            spread = loop_pts[:, axis].max() - loop_pts[:, axis].min()
            if spread < best_spread:
                best_spread = spread
                best_axis = axis

        plane_value = loop_pts[:, best_axis].mean()
        dist_to_min = abs(plane_value - bbox[0, best_axis])
        dist_to_max = abs(plane_value - bbox[1, best_axis])

        results.append(
            {
                "loop_indices": np.asarray(orig_ids),
                "axis": best_axis,
                "spread": best_spread,
                "plane_value": plane_value,
                "dist_to_bbox_min": dist_to_min,
                "dist_to_bbox_max": dist_to_max,
                "loop_length": len(loop_pts),
            }
        )

    # Sort by spread (most planar first)
    results.sort(key=lambda x: x["spread"])
    return results


def _extract_boundary_loops(faces):
    """Extract ordered boundary loops from a triangle mesh.

    Boundary edges are edges that appear in exactly one face.

    Returns a list of loops, each a list of vertex indices in order.
    """
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


def _classify_vertex_to_face(vertex, bbox, tol):
    """Classify a vertex to its closest bbox face.

    Returns (axis, side) where axis is 0,1,2 (x,y,z) and side is 0 (min) or 1 (max).
    """
    best_axis = 0
    best_side = 0
    best_dist = np.inf
    for axis in range(3):
        d_min = abs(vertex[axis] - bbox[0, axis])
        d_max = abs(vertex[axis] - bbox[1, axis])
        if d_min < best_dist:
            best_dist = d_min
            best_axis = axis
            best_side = 0
        if d_max < best_dist:
            best_dist = d_max
            best_axis = axis
            best_side = 1
    return (best_axis, best_side)


def _bbox_edge_point(v1, v2, bbox, axis, side):
    """Compute where the segment v1->v2 crosses the plane bbox[side, axis].

    Returns the 3D intersection point, or None if parallel or out of range.
    """
    plane_val = bbox[side, axis]
    d = v2[axis] - v1[axis]
    if abs(d) < 1e-12:
        return None
    t = (plane_val - v1[axis]) / d
    if t < 0 or t > 1:
        return None
    return v1 + t * (v2 - v1)


def _split_loop_by_face(loop_vertices, bbox, tol):
    """Split a boundary loop into per-face segments with bbox geometry inserted.

    Returns dict mapping (axis, side) -> list of (N_i, 3) vertex arrays.
    Each array is an ordered polygon to be triangulated on that face.
    """
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
            # Same axis, different side
            mid = (v1 + v2) / 2
            current_segment.append(mid)
            if current_face not in face_segments:
                face_segments[current_face] = []
            face_segments[current_face].append(np.array(current_segment))
            current_face = next_face
            current_segment = [mid]
            if next_i != 0:
                current_segment.append(loop_vertices[next_i])
        else:
            # Different axes — crossing a bbox edge
            cross_a = _bbox_edge_point(v1, v2, bbox, next_face[0], next_face[1])
            cross_b = _bbox_edge_point(v1, v2, bbox, this_face[0], this_face[1])

            if cross_a is not None and cross_b is not None:
                # Two crossings — going around a corner
                d_a = np.linalg.norm(cross_a - v1)
                d_b = np.linalg.norm(cross_b - v1)

                if d_b < d_a:
                    corner = np.copy(cross_b)
                    corner[next_face[0]] = bbox[next_face[1], next_face[0]]
                    current_segment.append(cross_b)
                    current_segment.append(corner)
                    if current_face not in face_segments:
                        face_segments[current_face] = []
                    face_segments[current_face].append(np.array(current_segment))
                    current_face = next_face
                    current_segment = [corner, cross_a]
                else:
                    corner = np.copy(cross_a)
                    corner[this_face[0]] = bbox[this_face[1], this_face[0]]
                    current_segment.append(cross_a)
                    current_segment.append(corner)
                    if current_face not in face_segments:
                        face_segments[current_face] = []
                    face_segments[current_face].append(np.array(current_segment))
                    current_face = next_face
                    current_segment = [corner, cross_b]
            else:
                cross = cross_a if cross_a is not None else cross_b
                if cross is None:
                    cross = (v1 + v2) / 2
                current_segment.append(cross)
                if current_face not in face_segments:
                    face_segments[current_face] = []
                face_segments[current_face].append(np.array(current_segment))
                current_face = next_face
                current_segment = [cross]

            if next_i != 0:
                current_segment.append(loop_vertices[next_i])

    if len(current_segment) > 0:
        if current_face not in face_segments:
            face_segments[current_face] = []
        face_segments[current_face].append(np.array(current_segment))

    # Merge segments on the same face into closed polygons
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
    """Triangulate a polygon on a bbox face using constrained Delaunay.

    Projects to 2D by dropping the normal axis, triangulates, returns indices.
    """
    import triangle as tr

    axes_2d = [a for a in range(3) if a != axis]
    pts_2d = polygon_3d[:, axes_2d]

    n = len(pts_2d)
    if n < 3:
        return np.empty((0, 3), dtype=int)

    segments = np.array([[i, (i + 1) % n] for i in range(n)])

    tri_input = {"vertices": pts_2d.astype(np.float64), "segments": segments}
    tri_output = tr.triangulate(tri_input, "p")

    if "triangles" not in tri_output:
        return np.empty((0, 3), dtype=int)

    return tri_output["triangles"]


def cap_mesh_at_bbox(vertices, faces, bbox, cv):
    """Cap open boundary holes at chunk bounding box faces.

    Detects boundary loops on the bbox surface, splits them by bbox face
    (inserting corner geometry as needed), and fills each sub-polygon with
    a constrained Delaunay triangulation.

    Parameters
    ----------
    vertices : np.ndarray
        (N, 3) mesh vertex positions in nm.
    faces : np.ndarray
        (M, 3) mesh triangle indices.
    bbox : np.ndarray
        (2, 3) bounding box (non-draco-adjusted recommended).
    cv : cloudvolume.CloudVolume
        CloudVolume object, used for draco grid size.

    Returns
    -------
    new_vertices : np.ndarray
        (N', 3) updated vertex array (may have new vertices from
        edge/corner insertions).
    new_faces : np.ndarray
        (M', 3) updated face array including cap triangles.
    """
    draco_grid_size = cv.meta.get_draco_grid_size(0)
    tol = draco_grid_size / 2

    surface_mask = np.zeros(len(vertices), dtype=bool)
    surface_inds = vertices_on_bbox_surface(vertices, bbox, cv)
    surface_mask[surface_inds] = True

    loops = _extract_boundary_loops(faces)

    new_verts_list = [vertices]
    new_faces_list = [faces]
    next_vert_idx = len(vertices)

    for loop in loops:
        loop_arr = np.array(loop)

        on_surface_count = np.sum(surface_mask[loop_arr])
        if on_surface_count < len(loop_arr) * 0.5:
            continue

        loop_verts = vertices[loop_arr]

        face_polygons = _split_loop_by_face(loop_verts, bbox, tol)

        for (axis, side), polygons in face_polygons.items():
            for polygon_3d in polygons:
                if len(polygon_3d) < 3:
                    continue

                polygon_3d = polygon_3d.copy()
                polygon_3d[:, axis] = bbox[side, axis]

                tri_indices = _triangulate_face_polygon(polygon_3d, axis)
                if len(tri_indices) == 0:
                    continue

                poly_to_mesh = np.empty(len(polygon_3d), dtype=int)
                for pi, pv_coord in enumerate(polygon_3d):
                    dists = np.linalg.norm(vertices[loop_arr] - pv_coord, axis=1)
                    min_dist_idx = np.argmin(dists)
                    if dists[min_dist_idx] < tol:
                        poly_to_mesh[pi] = loop_arr[min_dist_idx]
                    else:
                        new_verts_list.append(pv_coord.reshape(1, 3))
                        poly_to_mesh[pi] = next_vert_idx
                        next_vert_idx += 1

                cap_faces = poly_to_mesh[tri_indices]
                new_faces_list.append(cap_faces)

    new_vertices = np.concatenate(new_verts_list, axis=0).astype(np.float32)
    new_faces = np.concatenate(new_faces_list, axis=0).astype(np.uint32)

    return new_vertices, new_faces


draco = cv.meta.get_draco_grid_size(0)
print(f"draco_grid_size: {draco}, half: {draco / 2}")

for node_id, mesh in meshes.items():
    bbox = node_bbox(node_id, cv)
    mesh = (mesh.vertices, mesh.faces)

    loop_info = find_planar_boundary_loops(mesh[0], mesh[1], bbox)

    print(f"Node ID: {node_id}, found {len(loop_info)} boundary loops:")
    for info in loop_info:
        print(
            f"len={info['loop_length']}, axis={info['axis']}, "
            f"spread={info['spread']:.1f}, plane={info['plane_value']:.1f}, "
            f"d_min={info['dist_to_bbox_min']:.1f}, d_max={info['dist_to_bbox_max']:.1f}"
        )
    print()

# %%

node_id = 377814219896229972
bbox = node_bbox(node_id, cv)
mesh = meshes[node_id]
mesh = (mesh.vertices, mesh.faces)

plotter = pv.Plotter()
plotter.add_mesh(pv.make_tri_mesh(*mesh), color="lightgray")

loop_info = find_planar_boundary_loops(mesh[0], mesh[1], bbox)
for res in loop_info:
    # if res["spread"] > 100:
    loop_pts = mesh[0][res["loop_indices"]]
    plotter.add_points(loop_pts, color="red", point_size=10)

plotter.enable_fly_to_right_click()
plotter.show()


# %%

box = pv.Box(bounds=bbox.T.flatten().tolist())

bbox = node_bbox(node_id, cv)
vertices_on_surface = vertices_on_bbox_surface(mesh[0], bbox, cv)

plotter = pv.Plotter()
plotter.add_mesh(pv.make_tri_mesh(*mesh), color="lightgray")
plotter.add_mesh(box, color="black", opacity=1, style="wireframe")
plotter.add_points(mesh[0][vertices_on_surface], color="red", point_size=10)
plotter.enable_fly_to_right_click()
plotter.show()

# %%

capped_mesh = cap_mesh_at_bbox(mesh[0], mesh[1], bbox, cv)

# %%

capped_pv_mesh = pv.make_tri_mesh(*capped_mesh)

plotter = pv.Plotter()
plotter.add_mesh(capped_pv_mesh, color="lightgray")
plotter.add_mesh(box, color="black", opacity=1, style="wireframe")
plotter.add_points(mesh[0][vertices_on_surface], color="red", point_size=10)
plotter.enable_fly_to_right_click()
plotter.show()


# %%

# highlight non-manifold edges in red

poly1 = capped_pv_mesh.extract_feature_edges(
    boundary_edges=True,
    non_manifold_edges=True,
    feature_edges=False,
    manifold_edges=False,
)
plotter = pv.Plotter()
plotter.add_mesh(capped_pv_mesh, color="lightgray")
plotter.add_mesh(poly1, color="red", line_width=5)
plotter.add_mesh(box, color="black", opacity=1, style="wireframe")
plotter.enable_fly_to_right_click()
plotter.show()
# %%

capped_filled_pv_mesh = capped_pv_mesh.fill_holes(10000)

plotter = pv.Plotter()
plotter.add_mesh(capped_filled_pv_mesh, color="lightgray")
plotter.add_mesh(box, color="black", opacity=1, style="wireframe")
plotter.enable_fly_to_right_click()
plotter.show()


# %%

poly2 = capped_filled_pv_mesh.extract_feature_edges(
    boundary_edges=True,
    non_manifold_edges=True,
    feature_edges=False,
    manifold_edges=False,
)
plotter = pv.Plotter()
plotter.add_mesh(capped_filled_pv_mesh, color="lightgray")
plotter.add_mesh(poly2, color="red", line_width=5)
plotter.add_mesh(box, color="black", opacity=1, style="wireframe")
plotter.enable_fly_to_right_click()
plotter.show()

# %%

capped_filled_cleaned_pv_mesh = capped_filled_pv_mesh.clean(
    tolerance=1e-5,  # Merge vertices within this distance
    absolute=True,
    lines_to_points=True,
)

# %%
poly3 = capped_filled_cleaned_pv_mesh.extract_feature_edges(
    boundary_edges=True,
    non_manifold_edges=True,
    feature_edges=False,
    manifold_edges=False,
)
print(poly3.n_points, "non-manifold edges after cleaning")

# %%


def poly_to_mesh(poly: pv.PolyData):
    """Convert a [PolyData][pyvista.PolyData] to a ``(vertices, faces)`` tuple.

    Parameters
    ----------
    poly :
        Triangle surface mesh as a [PolyData][pyvista.PolyData].

    Returns
    -------
    vertices :
        Array of vertex positions, shape ``(V, 3)``.
    faces :
        Array of triangle face indices, shape ``(F, 3)``.
    """
    vertices = np.asarray(poly.points)
    faces = poly.faces.reshape(-1, 4)[:, 1:]
    return vertices, faces


clean_mesh = poly_to_mesh(capped_filled_pv_mesh)


# %%


# from pymeshlab import Mesh

# meshlab_mesh = Mesh(capped_mesh[0], capped_mesh[1])

# %%

import trimesh

trimesh_mesh = trimesh.Trimesh(*clean_mesh, process=False)
trimesh_mesh.is_watertight

# %%
