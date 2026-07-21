# script2
# requires-python = ">=3.12"
# dependencies = [
#     "caveclient[cv]>=8.1.0",
#     "ipykernel>=7.3.0",
#     "ipywidgets>=8.1.8",
#     "meshmash>=0.1.0",
#     "point-cloud-utils>=0.34.0",
#     "pyvista[all]>=0.48.4",
#     "robust-laplacian>=1.1.0",
#     "scikit-learn>=1.9.0",
#     "seaborn>=0.13.2",
#     "skeletor>=1.6.0",
# ]
# ///

# %%
import time

import numpy as np
import point_cloud_utils as pcu
import pyvista as pv
from gpytoolbox import fast_winding_number
from meshmash import fetch_sample_mesh
from sklearn.metrics import pairwise_distances_argmin

mesh = fetch_sample_mesh("microns_neuron_sample")
vertices = mesh[0].astype(np.float64)
faces = mesh[1].astype(np.int32)


# %%
# merge duplicate vertices
vertices, faces, old_to_new, new_to_old = pcu.deduplicate_mesh_vertices(
    vertices, faces, epsilon=22, return_index=True
)

# %%
from fast_simplification import simplify

vertices, faces = simplify(vertices, faces, target_reduction=0.7)

# %%

# orient normals
faces, components = pcu.orient_mesh_faces(faces)


# %%
# remove any degenerate faces

areas = pcu.mesh_face_areas(vertices, faces)
mask = areas > 1e-2
faces = faces[mask]

# %%
# fix slivers/slits
poly = (
    pv.make_tri_mesh(vertices, faces)
    .clean(absolute=True, tolerance=50, lines_to_points=True, point_merging=True)
    .fill_holes(10000.0)
)
from meshmash import poly_to_mesh

vertices, faces = poly_to_mesh(poly)


# %%
def remove_fins(vertices, faces):
    """Remove triangles hanging on by a vertex that only one face references,
    then drop the now-unreferenced vertices and reindex."""
    vcount = np.bincount(faces.ravel(), minlength=len(vertices))
    faces = faces[~np.any(vcount[faces] == 1, axis=1)]
    used = np.unique(faces)
    remap = -np.ones(len(vertices), dtype=np.int64)
    remap[used] = np.arange(len(used))
    return vertices[used], remap[faces]


vertices, faces = remove_fins(vertices, faces)


# %%
def stitch_t_junctions(vertices, faces, tol=50.0, max_rounds=10):
    """Repair T-junctions locally without requiring a globally manifold mesh.

    A T-junction is a boundary edge A-B (used by one face) whose "partner" is an
    interior vertex M sitting on the segment A-B (M's own edges are shared, so M
    is not on the boundary). We split the single triangle holding A-B into
    (A,M,C)+(M,B,C), which turns A-M and M-B into shared edges and closes the
    slit. Vertices are unchanged (M already lies on the segment), so geometry is
    preserved exactly. Iterated because one triangle can carry several such edges.
    """
    from scipy.spatial import cKDTree

    faces = faces.astype(np.int64, copy=True)
    vtree = cKDTree(vertices)  # vertices never move, build once
    total = 0
    for _ in range(max_rounds):
        nf = len(faces)
        tri_edges = np.sort(
            np.concatenate(
                [faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0
            ),
            axis=1,
        )
        row_face = np.tile(np.arange(nf), 3)
        uniq, first_idx, counts = np.unique(
            tri_edges, axis=0, return_index=True, return_counts=True
        )
        bmask = counts == 1  # boundary edges: used by exactly one face
        bedges = uniq[bmask]
        bface = row_face[first_idx[bmask]]

        splits = []  # (face_idx, a, b, m)
        used_faces = set()  # one split per face per round to avoid conflicts
        for (a, b), fi in zip(bedges, bface):
            fi = int(fi)
            if fi in used_faces:
                continue
            pa, pb = vertices[a], vertices[b]
            ab = pb - pa
            L2 = float(ab @ ab)
            if L2 == 0:
                continue
            L = np.sqrt(L2)
            best_m, best_d = -1, tol
            for c in vtree.query_ball_point(0.5 * (pa + pb), 0.5 * L + tol):
                if c == a or c == b:
                    continue
                t = ((vertices[c] - pa) @ ab) / L2
                if t <= 0.1 or t >= 0.9:  # must land on the segment INTERIOR
                    continue
                d = float(np.linalg.norm(vertices[c] - (pa + t * ab)))
                if d < best_d:
                    best_d, best_m = d, int(c)
            if best_m >= 0:
                splits.append((fi, int(a), int(b), best_m))
                used_faces.add(fi)

        if not splits:
            break

        keep = np.ones(nf, dtype=bool)
        keep[[s[0] for s in splits]] = False
        new_faces = [faces[keep]]
        for fi, a, b, m in splits:
            tri = faces[fi]
            for i in range(3):
                j = (i + 1) % 3
                if {int(tri[i]), int(tri[j])} == {a, b}:
                    k = (i + 2) % 3
                    new_faces.append(
                        np.array([[tri[i], m, tri[k]], [m, tri[j], tri[k]]])
                    )
                    break
        faces = np.concatenate(new_faces, axis=0)
        total += len(splits)

    print(f"stitched {total} T-junctions")
    return vertices, faces


vertices, faces = stitch_t_junctions(vertices, faces, tol=50.0)

# %%
plotter = pv.Plotter()

poly = pv.make_tri_mesh(vertices, faces)
plotter.add_mesh(poly, color="grey", opacity=0.5)
boundary = poly.extract_feature_edges(
    boundary_edges=True,
    feature_edges=False,
    manifold_edges=False,
    non_manifold_edges=False,
)
if boundary.n_points > 0:
    plotter.add_mesh(boundary, color="red", line_width=3)
plotter.enable_fly_to_right_click()


pos = (1206870.0, 467733.0, 711207.0)
plotter.camera.focal_point = pos

plotter.show()

#%%
# import pymeshlab

# ms = pymeshlab.MeshSet()
# ms.add_mesh(pymeshlab.Mesh(vertices, faces))
# ms.meshing_remove_t_vertices(method="Edge Flip", threshold=40.0, repeat=True)
# ms.meshing_remove_unreferenced_vertices()
# m = ms.current_mesh()
# vertices, faces = m.vertex_matrix(), m.face_matrix()

# %%
# ---------------------------------------------------------------------------
# Diagnose each open-boundary GROUP (connected component of boundary edges)
# ---------------------------------------------------------------------------
# A clean manifold-with-boundary has boundary edges that chain into CLOSED LOOPS
# (every boundary vertex has exactly 2 boundary edges). If instead we see many
# tiny groups that are single unpaired edges with two degree-1 ends, the mesh
# has CRACKS: an edge A-B used by one face whose partner edge A'-B' exists on
# DUPLICATE, unmerged vertices (A'~A, B'~B). The two halves fall into separate
# groups, so the useful quantity is the GAP between a boundary vertex and the
# nearest boundary vertex in a DIFFERENT group -- i.e. the weld distance needed
# to close the crack. A boundary vertex with NO nearby partner is a real hole.
from scipy.sparse import coo_array
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

poly = pv.make_tri_mesh(vertices, faces)
boundary = poly.extract_feature_edges(
    boundary_edges=True,
    feature_edges=False,
    manifold_edges=False,
    non_manifold_edges=False,
)

bpts = boundary.points
# feature edges are 2-point line cells: [2, i, j, 2, k, l, ...]
bedges = boundary.lines.reshape(-1, 3)[:, 1:]

# connected components of the boundary-edge graph
n_bpts = len(bpts)
adj = coo_array(
    (np.ones(len(bedges)), (bedges[:, 0], bedges[:, 1])),
    shape=(n_bpts, n_bpts),
)
n_groups, group_of_pt = connected_components(adj, directed=False)
degree = np.bincount(bedges.ravel(), minlength=n_bpts)

# --- cross-group weld gap: nearest boundary vertex in a DIFFERENT group -----
tree = cKDTree(bpts)
# query enough neighbors to step past same-group points, then take first that
# belongs to another boundary group
k = min(n_bpts, 8)
dists, idxs = tree.query(bpts, k=k)
vgap = np.full(n_bpts, np.inf)
partner_group = np.full(n_bpts, -1)
for col in range(1, k):
    other = group_of_pt[idxs[:, col]] != group_of_pt
    take = other & np.isinf(vgap)
    vgap[take] = dists[take, col]
    partner_group[take] = group_of_pt[idxs[take, col]]

WELD_TOL_NM = 75.0  # a partner within this distance => a weldable crack

# --- T-junction test: does an INTERIOR mesh vertex straddle a boundary edge? -
# For a T-junction the partner vertex M sits in the middle of the open edge A-B
# but is interior (its own edges are shared), so it is NOT in the boundary point
# set and the weld-gap test above cannot see it. Here we look at ALL mesh
# vertices near each boundary edge segment and check for one that projects onto
# the segment INTERIOR at ~zero perpendicular distance -> that is the straddling
# vertex M, i.e. a T-junction the crack is silently closed against.
STRADDLE_TOL_NM = 50.0  # perpendicular dist of a straddling vertex to the edge
vtree = cKDTree(vertices)
edge_straddle = np.full(len(bedges), np.inf)
for ei, (ia, ib) in enumerate(bedges):
    pa, pb = bpts[ia], bpts[ib]
    ab = pb - pa
    L = np.linalg.norm(ab)
    if L == 0:
        continue
    mid = 0.5 * (pa + pb)
    cand = vtree.query_ball_point(mid, 0.5 * L + STRADDLE_TOL_NM)
    if not cand:
        continue
    P = vertices[cand]
    t = ((P - pa) @ ab) / (L * L)
    perp = np.linalg.norm(P - (pa + t[:, None] * ab), axis=1)
    interior = (t > 0.1) & (t < 0.9)  # projects strictly between the endpoints
    if interior.any():
        edge_straddle[ei] = perp[interior].min()

rows = []
cls_of_pt = np.empty(n_bpts, dtype=object)
for g in range(n_groups):
    idx = np.flatnonzero(group_of_pt == g)
    edge_mask = (group_of_pt[bedges[:, 0]] == g) & (group_of_pt[bedges[:, 1]] == g)
    n_branch = int((degree[idx] >= 3).sum())
    n_ends = int((degree[idx] == 1).sum())
    is_loop = n_ends == 0 and n_branch == 0

    seg = np.linalg.norm(
        bpts[bedges[edge_mask, 0]] - bpts[bedges[edge_mask, 1]], axis=1
    )
    perimeter = float(seg.sum())
    group_gap = float(np.min(vgap[idx]))  # closest approach to another group
    group_straddle = float(np.min(edge_straddle[edge_mask]))  # nearest straddling vtx

    if n_branch > 0:
        cls = "branchy"
    elif np.isfinite(group_straddle) and group_straddle <= STRADDLE_TOL_NM:
        cls = "T-junction"
    elif np.isfinite(group_gap) and group_gap <= WELD_TOL_NM:
        cls = "weldable crack"
    elif is_loop:
        cls = "hole (loop)"
    else:
        cls = "open/unknown"
    cls_of_pt[idx] = cls

    rows.append(
        dict(
            group=g,
            cls=cls,
            n_verts=len(idx),
            n_edges=int(edge_mask.sum()),
            loop=is_loop,
            gap_nm=round(group_gap, 1),
            straddle_nm=round(group_straddle, 1),
            perim_nm=round(perimeter, 1),
        )
    )

import pandas as pd

diag = pd.DataFrame(rows).sort_values(["cls", "gap_nm"])
print(f"{n_groups} boundary groups")
print(diag["cls"].value_counts().to_string())
print(diag.to_string(index=False))

# %%
# color each boundary group by its diagnosed defect class
class_codes = {
    "T-junction": 0,
    "weldable crack": 1,
    "hole (loop)": 2,
    "open/unknown": 3,
    "branchy": 4,
}
class_cmap = ["red", "orange", "blue", "purple", "green"]
boundary.point_data["cls"] = np.array(
    [class_codes[c] for c in cls_of_pt], dtype=int
)

plotter = pv.Plotter()
plotter.add_mesh(poly, color="grey", opacity=0.3)
plotter.add_mesh(
    boundary,
    scalars="cls",
    cmap=class_cmap,
    clim=[0, 4],
    line_width=4,
    show_scalar_bar=False,
)
plotter.add_text(
    "red=T-junction  orange=weldable crack  blue=hole  purple=open  green=branchy"
)
plotter.enable_fly_to_right_click()
plotter.show()



# %%
# ---------------------------------------------------------------------------
# Consistent GLOBAL normal orientation fix
# ---------------------------------------------------------------------------
# `pcu.orient_mesh_faces` already makes normals consistent WITHIN each connected
# component (adjacency propagation), but each component's overall SIGN is still
# arbitrary -- which is exactly how the buried blob ended up with inward normals.
#
# We fix each component's sign independently using its OWN self-winding-number.
# Judging a component on its own geometry (not the full-mesh WN field) is the
# robust choice: a single flipped component corrupts the global field via
# cancellation, so a global-field test could be fooled. A component's self-WN is
# unaffected by how the rest of the mesh is oriented.
#
# The inside region (|self-WN| ~ 1) vs outside (|self-WN| ~ 0) is orientation-
# INDEPENDENT; flipping a component only negates the SIGN of its field. So we
# decide inside/outside by MAGNITUDE, then make the normal point toward the
# low-|WN| (outside) side. This is what makes the fix idempotent: a SIGNED
# comparison co-rotates with the normal (both n and the field flip together),
# so it would flip the same component on every run and never converge.
#
# For a correctly outward-oriented closed component:
#   - +normal (outward) side -> |self-WN| ~ 0
#   - -normal (inward)  side -> |self-WN| ~ 1
# If instead the +normal side has the LARGER |WN|, it points inward -> flip it.
from tqdm import tqdm

cv, nv, cf, nf = pcu.connected_components(vertices, faces)

timer = time.time()
timers = {
    "select_component": 0,
    "estimate_normals": 0,
    "compute_winding": 0,
    "orient_components": 0,
}
oriented_faces = faces.copy()
flip_eps = 25.0  # nm, offset for the sign-probe points

n_flipped = 0
flipped_info = []  # (comp_id, n_faces, margin) for each flipped component
for comp_id in tqdm(np.unique(cf), desc="Orienting components"):
    t0 = time.time()
    comp_face_mask = cf == comp_id
    if comp_face_mask.sum() > 10_000:
        continue  # skip huge components (e.g. the main neuron) to save time
    comp_faces = oriented_faces[comp_face_mask]
    timers["select_component"] += time.time() - t0

    t0 = time.time()
    comp_normals = pcu.estimate_mesh_face_normals(vertices, comp_faces)
    timers["estimate_normals"] += time.time() - t0

    t0 = time.time()
    comp_ctrs = vertices[comp_faces].mean(axis=1)
    probe_out = comp_ctrs + comp_normals * flip_eps  # +normal side (expect OUTSIDE)
    probe_in = comp_ctrs - comp_normals * flip_eps  # -normal side (expect INSIDE)

    # self-winding-number: only this component's faces contribute, so the result
    # is independent of any (mis)orientation elsewhere in the mesh
    w_out = fast_winding_number(probe_out, vertices, comp_faces)
    w_in = fast_winding_number(probe_in, vertices, comp_faces)
    timers["compute_winding"] += time.time() - t0

    t0 = time.time()
    # inside side has the larger |WN|; outward normal should point to the smaller
    # |WN| (outside) side. Compare MAGNITUDES so the test does not co-rotate with
    # the normal -> idempotent. Flip when +normal points into the |WN|~1 region.
    # `margin` near 0 means an AMBIGUOUS component (e.g. an open sheet) whose
    # orientation is not well defined -> prone to flip-flopping.
    margin = np.mean(np.abs(w_out)) - np.mean(np.abs(w_in))
    if margin > 0:
        oriented_faces[comp_face_mask] = comp_faces[:, ::-1]
        n_flipped += 1
        flipped_info.append((int(comp_id), int(comp_face_mask.sum()), float(margin)))
    timers["orient_components"] += time.time() - t0


print(f"Flipped {n_flipped} / {len(np.unique(cf))} components to outward orientation")
print(f"Orientation fix took {time.time() - timer:.2f} seconds")
# report flipped components sorted by how ambiguous the decision was (small
# |margin| = near-tie = likely open/ambiguous, at risk of flip-flopping)
for comp_id, n_faces, margin in sorted(flipped_info, key=lambda x: abs(x[2])):
    print(f"  comp {comp_id}: {n_faces} faces, margin={margin:+.4f}")
# # adopt the globally-consistent orientation
faces = oriented_faces
normals = pcu.estimate_mesh_face_normals(vertices, faces)


# %%

# construct sample points that are +/- epsilon away from each mesh facet
normals = pcu.estimate_mesh_face_normals(vertices, faces)
lengths = np.linalg.norm(normals, axis=1)
assert np.allclose(lengths, 1.0, atol=1e-6)

# %%

ctrs = vertices[faces].mean(axis=1)

sampling_strategy = "edge_adaptive"  # "constant" or "edge_adaptive"
if sampling_strategy == "constant":
    epsilon = 25  # units are in nm
    points_outside = ctrs + normals * epsilon
    points_inside = ctrs - normals * epsilon
elif sampling_strategy == "edge_adaptive":
    tri = vertices[faces]
    e = np.linalg.norm(tri - np.roll(tri, -1, axis=1), axis=2)
    edge_face = e.mean(1)
    edge_scale = 0.5
    epsilon = edge_face * edge_scale
    points_outside = ctrs + normals * epsilon[:, None]
    points_inside = ctrs - normals * epsilon[:, None]


all_query_points = np.concatenate([points_outside, points_inside], axis=0)

timer = time.time()
numbers = fast_winding_number(all_query_points[:], vertices, faces)
print(f"Fast winding number took {time.time() - timer:.2f} seconds")

# %%
numbers_outside = numbers[: len(points_outside)]
numbers_inside = numbers[len(points_outside) :]
# %%
table = np.concatenate([numbers_outside[:, None], numbers_inside[:, None]], axis=1)

labels, inv, counts = np.unique(
    table > 0.5, axis=0, return_inverse=True, return_counts=True
)  # should be (array([False, True]), array([1000, 1000]))

print(f"Labels: {labels}")
print(f"Counts: {counts}")

# %%

# is_boundary = np.isclose(numbers_outside, 0.5) & np.isclose(numbers_inside, 0.5)

# %%

mesh_poly = pv.make_tri_mesh(vertices, faces)

# create a face-feature, inv

# mesh_poly.cell_data["inside_outside"] = inv
mesh_poly.cell_data["is_border"] = np.isclose(numbers_outside, 0.0, atol=1e-3)
# plotter = pv.Plotter()
# # show with a categorical colormap
# plotter.add_mesh(
#     mesh_poly,
#     scalars="inside_outside",
#     cmap=["red", "blue", "green", "purple"],
#     opacity=0.7,
# )
# plotter.enable_fly_to_right_click()
# plotter.show()

# %%
# (false, false) -> mostly salt and pepper faces, often on thin bits, keep
# (false, true) -> outside faces, keep
# (true, false) -> salt and pepper, some concentration at weird interfaces
# (true, true) -> mostly inside faces, remove
# %%
# make plots of each color separately
plotter = pv.Plotter()
# i = 3
plotter.add_mesh(
    mesh_poly.extract_cells(inv == 3),
    # scalars="inside_outside",
    color="red",
    opacity=0.9,
)
plotter.add_mesh(
    mesh_poly,
    color="grey",
    opacity=0.1,
)
plotter.enable_fly_to_right_click()
plotter.show()

# %%
keep_mask = (numbers_outside <= 0.5) | (
    numbers_inside <= 0.5
)  # this criterion might change

# %%
# ---------------------------------------------------------------------------
# Graph-cut smoothing of the keep/remove labeling
# ---------------------------------------------------------------------------
# The per-face `keep_mask` above is a hard, independent threshold, so it yields
# salt-and-pepper misclassifications wherever the winding number sits near 0.5
# (open holes, thin sheets, ambiguous interfaces). A binary graph cut on the
# face-adjacency (dual) graph regularizes this: it finds the globally optimal
# keep/remove labeling that balances (a) agreement with the winding-number
# evidence [data/unary term] against (b) label agreement between adjacent faces
# [smoothness/pairwise term]. Coherent buried blobs get removed while isolated
# flips are suppressed.
#
# Library choice: PyMaxflow (`import maxflow`) — a Cython wrapper around the
# Boykov-Kolmogorov max-flow/min-cut solver, the de-facto standard for binary
# vision-style cuts; it returns the node partition directly via get_segment().
# NOTE: it is GPL (the underlying BK code is research-only). Fully permissive
# fallback: scipy.sparse.csgraph.maximum_flow (BSD) + a residual-graph BFS to
# recover the cut, at the cost of extra bookkeeping.
import maxflow

data_weight = 0.1  # strength of the winding-number evidence (unary)
pairwise_weight = 1  # strength of the coherence prior (smoothness)

n_faces = len(faces)

# --- data term (unary) --------------------------------------------------
# m = min(w_out, w_in): a face is "buried" (remove) only when BOTH probes read
# inside (m > 0.5), and is kept when either probe is outside (m < 0.5). At
# pairwise_weight=0 this exactly reproduces the hard `keep_mask` above.
m = np.minimum(numbers_outside, numbers_inside)
d_keep = data_weight * np.maximum(0.0, m - 0.5)  # cost of KEEPING a buried face
d_remove = data_weight * np.maximum(0.0, 0.5 - m)  # cost of REMOVING a surface face

# --- pairwise term (smoothness) ----------------------------------------
# Face adjacency from shared (manifold) edges, each dual edge weighted by its
# shared-edge length so cuts prefer to run along short edges / creases.
edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0)
edges_sorted = np.sort(edges, axis=1)
face_of_edge = np.tile(np.arange(n_faces), 3)

order = np.lexsort((edges_sorted[:, 1], edges_sorted[:, 0]))
edges_sorted = edges_sorted[order]
face_of_edge = face_of_edge[order]

# consecutive identical (sorted) edges are shared by two faces -> a dual edge
same = np.all(edges_sorted[1:] == edges_sorted[:-1], axis=1)
adj_a = face_of_edge[:-1][same]
adj_b = face_of_edge[1:][same]
shared_edge = edges_sorted[:-1][same]
edge_len = np.linalg.norm(
    vertices[shared_edge[:, 0]] - vertices[shared_edge[:, 1]], axis=1
)
valid = adj_a != adj_b
adj_a, adj_b, edge_len = adj_a[valid], adj_b[valid], edge_len[valid]
w_pair = pairwise_weight * (edge_len / edge_len.mean())

# --- build and solve the min-cut ---------------------------------------
timer = time.time()
g = maxflow.Graph[float](n_faces, len(adj_a))
node_ids = g.add_nodes(n_faces)
# source = KEEP (label 0), sink = REMOVE (label 1); get_segment==0 -> keep.
# add_grid_tedges(node, source_cap, sink_cap): source_cap=d_remove, sink_cap=d_keep
g.add_grid_tedges(node_ids, d_remove, d_keep)
for i, j, w in zip(adj_a, adj_b, w_pair):
    g.add_edge(int(i), int(j), float(w), float(w))
g.maxflow()

seg = np.fromiter(
    (g.get_segment(i) for i in range(n_faces)), dtype=np.int8, count=n_faces
)
smoothed_keep_mask = seg == 0  # source side = keep
print(f"Graph cut took {time.time() - timer:.2f} seconds")

n_changed = int((smoothed_keep_mask != keep_mask).sum())
print(
    f"Kept {smoothed_keep_mask.sum()} / {n_faces} faces "
    f"({keep_mask.sum()} before smoothing, {n_changed} labels changed)"
)

# %%
# compare original vs raw-threshold vs graph-cut-smoothed labeling side by side,
# highlighting open boundary edges / holes in red for each
panels = [
    ("original", np.ones(n_faces, dtype=bool)),
    ("raw threshold", keep_mask),
    ("graph-cut smoothed", smoothed_keep_mask),
]

plotter = pv.Plotter(shape=(1, len(panels)))
for col, (title, mask) in enumerate(panels):
    plotter.subplot(0, col)
    poly = pv.make_tri_mesh(vertices, faces[mask])
    plotter.add_mesh(poly, color="grey", opacity=0.5)
    boundary = poly.extract_feature_edges(
        boundary_edges=True,
        feature_edges=False,
        manifold_edges=False,
        non_manifold_edges=False,
    )
    if boundary.n_points > 0:
        plotter.add_mesh(boundary, color="red", line_width=3)
    plotter.add_text(f"{title} ({boundary.n_lines} boundary edges)")
plotter.link_views()
plotter.enable_fly_to_right_click()
plotter.show()

# %%

# filtered_mesh = (vertices, faces[keep_mask])
# filtered_mesh_poly = pv.make_tri_mesh(*filtered_mesh)

# plotter = pv.Plotter()

# plotter.add_mesh(
#     filtered_mesh_poly,
#     color="grey",
#     opacity=0.3,
# )
# plotter.enable_fly_to_right_click()
# plotter.show()


# %%
if False:
    query_pt = np.array([1206492.0, 549990.0, 717507.0])

    # cv, nv, cf, nf = pcu.connected_components(vertices, faces)

    # get the closest point on the mesh to the query point,
    # then get the connected component of the mesh that contains that point, and then compute the winding number for that component only
    pt_idx = pairwise_distances_argmin(query_pt[None, :], vertices)[0]

    plotter = pv.Plotter()
    plotter.add_mesh(
        mesh_poly,
        color="grey",
        opacity=0.5,
    )
    plotter.add_mesh(
        mesh_poly.extract_points(cv == cv[pt_idx], adjacent_cells=True),
        color="red",
        opacity=0.9,
    )
    plotter.enable_fly_to_right_click()
    plotter.show()

    # get the winding numbers for the connected component that contains the query point
    vertex_component_mask = cv == cv[pt_idx]
    face_component_mask = cf == cv[pt_idx]
    mask_numbers_inside = numbers_inside[face_component_mask]
    # print(mask_numbers_inside)
    mask_numbers_outside = numbers_outside[face_component_mask]
    # print(mask_numbers_outside)

    table = np.concatenate(
        [mask_numbers_outside[:, None], mask_numbers_inside[:, None]], axis=1
    )

    labels, inv, counts = np.unique(
        table > 0.5, axis=0, return_inverse=True, return_counts=True
    )  # should be (array([False, True]), array([1000, 1000]))

    print(f"Labels: {labels}")
    print(f"Counts: {counts}")

    comp = cv[pt_idx]
    fmask = cf == comp
    c = vertices[np.unique(faces[fmask])].mean(0)

    noise = np.random.normal(scale=300, size=(10000, 3))
    noise += c

    wns = fast_winding_number(noise, vertices, faces)

    wns

    plotter = pv.Plotter()
    plotter.add_mesh(
        mesh_poly,
        color="grey",
        opacity=0.5,
    )
    plotter.add_points(
        noise,
        scalars=wns,
        cmap="coolwarm",
        # render_points_as_spheres=True,
        point_size=10,
        clim=[0, 1],
    )
    plotter.enable_fly_to_right_click()
    plotter.show()
