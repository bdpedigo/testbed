# %%
import time

import matplotlib.pyplot as plt
import numpy as np
import ossify
import pyvista as pv
import seaborn as sns
from caveclient import CAVEclient

root_id = 864691135134585248

# %%
# 1. What is the synaptic density per unit length of dendrite for subclasses of VIP
# interneurons? How does this vary with distance from the soma?
#

client = CAVEclient("minnie65_phase3_v1")

# %%

currtime = time.time()

ts = client.chunkedgraph.get_root_timestamps([root_id], latest=True)[0]
cell = ossify.load_cell_from_client(
    root_id,
    client,
    synapses=True,
    restore_graph=False,
    restore_properties=True,
    synapse_spatial_point="ctr_pt_psition",
    timestamp=ts,
)
print(f"{time.time() - currtime:.3f} seconds elapsed.")

# %%

fig, ax = plt.subplots(figsize=(8, 6))
ossify.plot.plot_morphology_2d(
    cell,
    color="compartment",
    projection="xy",
    palette="coolwarm_r",
    ax=ax,
    linewidth="radius",
    linewidth_norm=(100, 500),  # Radius range for normalization
    widths=(0.05, 3),  # Final line width range
)
ax.axis("off")

# %%

skeleton_labels = cell.skeleton.get_label("compartment")
dendrite = cell.apply_mask("skeleton", skeleton_labels == 3)

syn_density = (
    dendrite.skeleton.map_annotations_to_label(
        "post_syn", distance_threshold=10000, agg="density", validate=True
    )
    * 1000
)
sns.set_context("talk")
fig, ax = plt.subplots(figsize=(8, 6))
sns.histplot(syn_density, ax=ax)
ax.set_xlabel("Post-synaptic density (synapses/um)")

# %%
distance_to_root = dendrite.skeleton.distance_to_root()
dendrite.skeleton.nodes["distance_to_nucleus"] = distance_to_root
dendrite.skeleton.nodes["post_syn_density"] = syn_density
# %%
fig, ax = plt.subplots(figsize=(6, 6))
nodes = dendrite.skeleton.nodes
sns.scatterplot(
    data=nodes,
    x="distance_to_nucleus",
    y="post_syn_density",
    ax=ax,
    s=10,
)
ax.set_xlabel("Distance to nucleus (nm)")
ax.set_ylabel("Post-synaptic density (synapses/um)")


# %%
dendrite.skeleton.map_annotations_to_label(
    "post_syn",
    distance_threshold=20000,
)

# %%
x = dendrite.skeleton
import pandas as pd

result = x.map_annotations_to_label(
    "post_syn",
    20000,
    agg="count",
    chunk_size=1000,
)
count_len_df = pd.concat(
    (
        pd.Series(
            data=x.half_edge_length,
            index=x.vertex_index,
            name="net_length",
        ),
        result,
    ),
    axis=1,
)

# %%
x.csgraph_undirected

# %%


fig, ax = plt.subplots(figsize=(8, 6))
ossify.plot.plot_morphology_2d(
    dendrite,
    projection="xy",
    ax=ax,
    linewidth="radius",
    linewidth_norm=(100, 500),  # Radius range for normalization
    widths=(0.05, 3),  # Final line width range
)
ax.axis("off")

# %%


def edges_to_lines(edges):
    """Convert edges to pyvista lines format."""
    n_edges = edges.shape[0]
    lines = np.hstack(
        [np.full((n_edges, 1), 2), edges]
    )  # Each line has 2 points, so prepend with 2
    return lines.flatten()


plotter = pv.Plotter()
points = dendrite.skeleton.nodes[["x", "y", "z"]].to_numpy()
edges = dendrite.skeleton.edges_positional
lines = edges_to_lines(edges)
skeleton = pv.PolyData(points, lines=lines)

plotter.add_mesh(skeleton, color="black", line_width=2)

plotter.show()

# %%
from matplotlib.collections import LineCollection

edge_colors = "gray"
edge_kws = dict(alpha=1, linewidth=2)


def plot_graph(nodes, edges, x, y, ax):
    nodes = nodes.reset_index()
    start_xs = nodes.loc[edges[:, 0]][x]
    end_xs = nodes.loc[edges[:, 1]][x]
    start_ys = -nodes.loc[edges[:, 0]][y]
    end_ys = -nodes.loc[edges[:, 1]][y]
    pre_coords = list(zip(start_xs, start_ys))
    post_coords = list(zip(end_xs, end_ys))
    coords = list(zip(pre_coords, post_coords))
    lc = LineCollection(
        segments=coords,
        colors=edge_colors,
        **edge_kws,
    )
    ax.add_collection(lc)


nodes = dendrite.skeleton.nodes
edges = dendrite.skeleton.edges_positional

fig, ax = plt.subplots(figsize=(8, 6))
plot_graph(nodes, edges, "x", "y", ax)
ax.autoscale()


# %%

import seaborn as sns

# %%
dendrite.skeleton.nodes.drop("post_syn_density", axis=1, inplace=True, errors="ignore")
dendrite.skeleton.add_label(syn_density, "post_syn_density")

# %%

fig, ax = plt.subplots(figsize=(8, 6))
ossify.plot.plot_morphology_2d(
    dendrite,
    projection="xy",
    color="post_syn_density",
    color_norm=(0, 0.1),
    ax=ax,
    linewidth="radius",
    linewidth_norm=(100, 500),  # Radius range for normalization
    widths=(0.05, 3),  # Final line width range
)
ax.axis("off")


# %%
# 2. Compare density of synapses allocated to different dendritic compartments
#
# 3. What is the distribution of mono/polysynaptic spines for multipolar VIP neurons?
# Are they E-E, E-I, I-I?
#
# 4. What is the distribution of the number synapses from single axons? Are
# polysynaptic axons targeting specific compartments on multipolar VIP neurons?
# Is this cell-type specific?
#
# 5. Of axons making synapses onto VIP neurons that arise within the volume, where are
# their somas located?
#
