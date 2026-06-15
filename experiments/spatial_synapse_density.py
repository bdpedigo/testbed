# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "caveclient>=8.0.1",
#     "deltalake>=1.5.0",
#     "ipykernel>=7.2.0",
#     "ipywidgets>=8.1.8",
#     "kdepy>=1.1.12",
#     "polars>=1.39.3",
#     "pyarrow>=23.0.1",
#     "pyvista>=0.48.4",
#     "scipy>=1.17.1",
#     "seaborn>=0.13.2",
#     "tqdm>=4.67.3",
# ]
# ///
# %%

import time

import numpy as np
import polars as pl
import pyvista as pv
from caveclient import CAVEclient

client = CAVEclient("minnie65_phase3_v1", version=1718)


column_types = client.materialize.tables.allen_v1_column_types_slanted_ref().query()

column_types = pl.from_pandas(column_types).lazy()

local_path = (
    "/Users/ben.pedigo/code/meshrep/meshrep/data/synapses_pni_2_v1412_deltalake"
)
cloud_path = (
    "gs://allen-minnie-phase3/mat_deltalakes/v1412/synapses_pni_2_v1412_deltalake"
)
synapses = pl.scan_delta(cloud_path)


# %%
synapses = synapses.select(
    pl.col("ctr_pt_position_x") * 4,
    pl.col("ctr_pt_position_y") * 4,
    pl.col("ctr_pt_position_z") * 40,
)
timer = time.time()
pts = synapses.collect().to_numpy().astype("float32")
print(f"Collected synapse positions in {time.time() - timer:.2f} seconds")

# %%
from KDEpy import FFTKDE

sample_pts = pts[np.random.choice(pts.shape[0], size=1_000_000, replace=False)]
kde = FFTKDE("triweight", bw=10000).fit(sample_pts)


# %%

# bandwidth = 5_000  # in nm
# kde = gaussian_kde(pts.T, bw_method=bandwidth)

# %%

bounds = np.stack([np.min(pts, axis=0), np.max(pts, axis=0)])
step_size = 20_000  # in nm
grid = np.mgrid[
    bounds[0, 0] : bounds[1, 0] : step_size,
    bounds[0, 1] : bounds[1, 1] : step_size,
    bounds[0, 2] : bounds[1, 2] : step_size,
]
grid = grid.reshape(-1, 3)

print(grid.shape)

# %%
timer = time.time()
x, y = kde.evaluate(100)
print(f"Evaluated KDE on grid in {time.time() - timer:.2f} seconds")

# %%

kde_values = kde(grid_points)
kde_values = kde_values.reshape(grid.shape[1:])
print(kde_values.shape)

# %%


plotter = pv.Plotter()

plotter.add_volume(kde_values, cmap="viridis", opacity="sigmoid_6")
plotter.show()


# # %%
# bounds = synapses.select(
#     pl.col("ctr_pt_position_x").min().alias("min_x"),
#     pl.col("ctr_pt_position_y").min().alias("min_y"),
#     pl.col("ctr_pt_position_z").min().alias("min_z"),
#     pl.col("ctr_pt_position_x").max().alias("max_x"),
#     pl.col("ctr_pt_position_y").max().alias("max_y"),
#     pl.col("ctr_pt_position_z").max().alias("max_z"),
# ).collect()
# # %%
# # want to compute a 3d KDE of synapse density across the volume in a streaming
# # fashion. We can do this by binning the synapses into a 3d grid and then applying a
# # Gaussian filter to the resulting density map. But we need to handle the borders
# # between grid cells in a way that doesn't create artifacts
# # Define the grid size and bin the synapses into a 3D histogram

# kde_bandwidth = 5_000
# grid_size = 50_000  # in nm
# x_bins = pl.arange(bounds["min_x"][0], bounds["max_x"][0] + grid_size, grid_size)
# y_bins = pl.arange(bounds["min_y"][0], bounds["max_y"][0] + grid_size, grid_size)
# z_bins = pl.arange(bounds["min_z"][0], bounds["max_z"][0] + grid_size, grid_size)

# synapses = synapses.with_columns(
#     pl.col("ctr_pt_position_x").bin(x_bins).alias("x_bin"),
#     pl.col("ctr_pt_position_y").bin(y_bins).alias("y_bin"),
#     pl.col("ctr_pt_position_z").bin(z_bins).alias("z_bin"),
# )

# density = synapses.group_by("x_bin", "y_bin", "z_bin").agg(pl.count()).collect()
