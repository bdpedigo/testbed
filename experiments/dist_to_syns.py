# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "caveclient",
#     "cloud-volume",
#     "deltalake>=1.6.0",
#     "ipykernel>=7.2.0",
#     "ipywidgets>=8.1.8",
#     "meshmash>=0.1.0",
#     "numpy",
#     "point-cloud-utils",
#     "polars>=1.41.2",
#     "pyvista[all]",
#     "scikit-learn",
#     "seaborn",
# ]
# ///

# %%
import time
from typing import Any, Union

import matplotlib.pyplot as plt
import numpy as np
import point_cloud_utils as pcu
import polars as pl
import pyvista as pv
import seaborn as sns
from cloudvolume import CloudVolume
from meshmash import poly_to_mesh

type Mesh = Union[tuple[np.ndarray, np.ndarray], Any]


bv_cv = CloudVolume(
    "precomputed://s3://bossdb-open-data/wei2024/minnie/bv",
    progress=False,
    use_https=True,
)

currtime = time.time()

raw_mesh = bv_cv.mesh.get(1)[1]
bv_mesh = (raw_mesh.vertices, raw_mesh.faces)

print(f"{time.time() - currtime:.3f} seconds elapsed to download vasculature mesh.")

# %%

bv_poly = pv.make_tri_mesh(*bv_mesh)
# bv_poly = bv_poly.extract_largest()
bv_mesh = poly_to_mesh(bv_poly)

# %%


local_path = (
    "/Users/ben.pedigo/code/meshrep/meshrep/data/synapses_pni_2_v1412_deltalake"
)

# PLEASE only do this download once and save the result somewhere!
# cloud_path = (
#     "gs://allen-minnie-phase3/mat_deltalakes/v1412/synapses_pni_2_v1412_deltalake"
# )

synapses = pl.scan_delta(local_path)

synapses = synapses.select(
    pl.col("ctr_pt_position_x") * 4,
    pl.col("ctr_pt_position_y") * 4,
    pl.col("ctr_pt_position_z") * 40,
)
timer = time.time()
pts = synapses.collect().to_numpy().astype("float32")
print(f"Collected synapse positions in {time.time() - timer:.2f} seconds")

# %%

# get a bounding box of middle 80
x_min, x_max = np.percentile(pts[:, 0], [10, 90])
y_min, y_max = np.percentile(pts[:, 1], [10, 90])
z_min, z_max = np.percentile(pts[:, 2], [10, 90])

pts = pts[
    (pts[:, 0] >= x_min)
    & (pts[:, 0] <= x_max)
    & (pts[:, 1] >= y_min)
    & (pts[:, 1] <= y_max)
    & (pts[:, 2] >= z_min)
    & (pts[:, 2] <= z_max)
]

# %%

bounds = np.stack([np.min(pts, axis=0), np.max(pts, axis=0)])

# cubic boxes in nm
step_size = 2500
grid = np.mgrid[
    bounds[0, 0] : bounds[1, 0] : step_size,
    bounds[0, 1] : bounds[1, 1] : step_size,
    bounds[0, 2] : bounds[1, 2] : step_size,
]

# group points into their boxes
grid_indices = np.floor((pts - bounds[0]) / step_size).astype(int)
grid_indices = np.clip(grid_indices, 0, grid.shape[1] - 1)
del pts

# faster O(n) alternative using ravel_multi_index + bincount
n_bins = np.array(grid.shape[1:])
linear_indices = np.ravel_multi_index(grid_indices.T, n_bins)
counts_flat = np.bincount(linear_indices, minlength=int(np.prod(n_bins)))

# 3D count array (if needed)
counts_3d = counts_flat.reshape(n_bins)

# %%

# crop bv poly to bounds of synapses
# xmin, xmax, ymin, ymax, zmin, zmax = bounds.flatten()
bv_poly_crop = bv_poly.clip_box(bounds=bounds.T.flatten(), invert=False)

# %%

plotter = pv.Plotter()

count_vol = pv.ImageData(
    dimensions=n_bins,
    origin=bounds[0],
    spacing=(step_size, step_size, step_size),
)
count_vol["counts"] = counts_3d.ravel(order="F")

plotter.add_volume(count_vol, cmap="viridis", opacity="sigmoid_6")
plotter.add_mesh(bv_poly_crop, color="red", opacity=1)
plotter.enable_fly_to_right_click()
plotter.show()

# %%

# get the midpoint of each box
x_centers = bounds[0, 0] + (np.arange(n_bins[0]) + 0.5) * step_size
y_centers = bounds[0, 1] + (np.arange(n_bins[1]) + 0.5) * step_size
z_centers = bounds[0, 2] + (np.arange(n_bins[2]) + 0.5) * step_size

midpoints = (
    np.array(np.meshgrid(x_centers, y_centers, z_centers, indexing="ij"))
    .reshape(3, -1)
    .T
)
# %%

currtime = time.time()
dists, fid, bc = pcu.closest_points_on_mesh(
    midpoints.astype(np.float32, order="F"),
    bv_mesh[0].astype(np.float32, order="F"),
    bv_mesh[1].astype(np.int32, order="F"),
)

print(
    f"{time.time() - currtime:.3f} seconds elapsed to compute closest points on mesh."
)

closest_pts = pcu.interpolate_barycentric_coords(bv_mesh[1], fid, bc, bv_mesh[0])

# %%

results_df = pl.DataFrame(
    dict(
        x=midpoints[:, 0],
        y=midpoints[:, 1],
        z=midpoints[:, 2],
        dist_to_bv=dists,
        closest_x=closest_pts[:, 0],
        closest_y=closest_pts[:, 1],
        closest_z=closest_pts[:, 2],
        synapse_count=counts_flat,
    )
)


fig, ax = plt.subplots(figsize=(6, 6))
sns.scatterplot(
    data=results_df.sample(10000),
    x="dist_to_bv",
    y="synapse_count",
    alpha=0.05,
    linewidth=0,
    ax=ax,
)
ax.set_xlabel("Distance to Blood Vessel")
ax.set_ylabel("Synapse Count")
plt.show()

# %%

sns.jointplot(
    data=results_df.sample(100000),
    x="dist_to_bv",
    y="synapse_count",
    alpha=0.1,
    linewidth=0,
)

# %%

synapses = pl.scan_delta(local_path)

synapses = synapses.select(
    pl.col("ctr_pt_position_x") * 4,
    pl.col("ctr_pt_position_y") * 4,
    pl.col("ctr_pt_position_z") * 40,
    pl.col("size"),
    pl.col("synapse_id"),
)
synapses = synapses.filter(
    (pl.col("ctr_pt_position_x") >= x_min)
    & (pl.col("ctr_pt_position_x") <= x_max)
    & (pl.col("ctr_pt_position_y") >= y_min)
    & (pl.col("ctr_pt_position_y") <= y_max)
    & (pl.col("ctr_pt_position_z") >= z_min)
    & (pl.col("ctr_pt_position_z") <= z_max)
)

synapses = synapses.collect()

# %%
pts = (
    synapses.select(
        pl.col("ctr_pt_position_x"),
        pl.col("ctr_pt_position_y"),
        pl.col("ctr_pt_position_z"),
    )
    .to_numpy()
    .astype("float32")
)

# %%
batch_size = 10_000_000
timer = time.time()
verts_f = bv_mesh[0].astype(np.float32, order="F")
faces_i = bv_mesh[1].astype(np.int32, order="F")

dists_batches = []
for i in range(0, len(pts), batch_size):
    batch = pts[i : i + batch_size].astype(np.float32, order="F")
    d, _, _ = pcu.closest_points_on_mesh(batch, verts_f, faces_i)
    dists_batches.append(d)
    print(f"  Batch {i // batch_size + 1}: {len(batch)} points done")

dists = np.concatenate(dists_batches)
dists = pl.Series(name="dist_to_bv", values=dists)
print(f"{time.time() - timer:.3f} seconds elapsed to compute closest points on mesh.")

del pts, dists_batches
# %%
synapses = synapses.with_columns(dists)

# %%


fig, ax = plt.subplots(figsize=(6, 6))
sns.scatterplot(
    synapses.sample(100000), x="dist_to_bv", y="size", alpha=0.1, linewidth=0, ax=ax
)
ax.set(yscale="log")


from scipy.stats import spearmanr

sample = synapses.sample(1_000_000)
corr, p_value = spearmanr(sample["dist_to_bv"], sample["size"])
print(f"Spearman correlation: {corr:.4f}, p-value: {p_value:.4e}")

# %%

fig, ax = plt.subplots(figsize=(6, 6))
sns.histplot(sample, x="dist_to_bv", log_scale=True, ax=ax)
ax.set_xlabel("Distance to Blood Vessel (nm)")
