# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "caveclient",
#     "deltalake",
#     "matplotlib",
#     "ossify",
#     "polars",
#     "seaborn",
# ]
#
# [tool.uv.sources]
# ossify = { git = "https://github.com/bdpedigo/ossify.git", rev = "69348ca" }
# ///

# %%
import time

import matplotlib.pyplot as plt
import ossify  # https://github.com/bdpedigo/ossify.git@69348ca
import polars as pl
import seaborn as sns
from caveclient import CAVEclient
from matplotlib.collections import LineCollection

root_id = 864691135654097346  # bVIP
root_id = 864691135938202165  # bVIP
root_id = 864691135136756761  # mVIP
client = CAVEclient("minnie65_phase3_v1")

# %%
# 1. What is the synaptic density per unit length of dendrite for subclasses of VIP
# interneurons? How does this vary with distance from the soma? Compare density of
# synapses allocated to different dendritic compartments

# %%

currtime = time.time()

ts = client.chunkedgraph.get_root_timestamps([root_id], latest=True)[0]
cell = ossify.load_cell_from_client(
    root_id,
    client,
    synapses=True,
    restore_graph=False,
    restore_properties=True,
    include_partner_root_id=True,
    timestamp=ts,
)
print(f"{time.time() - currtime:.3f} seconds elapsed.")

# %%

# prediction_path = Path(
#     "/Users/ben.pedigo/code/meshrep/meshrep/data/auburn_elk_detour_predictions_deltalake"
# )
prediction_path = "gs://bdp-ssa/minnie65_phase3_v1/absolute-solo-yak/1412/auburn-elk-detour-synapse_hks_model/post-synapse-predictions-deltalake"
synapse_ids = cell.annotations.post_syn.nodes.index
currtime = time.time()
postsyn_predictions = (
    pl.scan_delta(prediction_path)
    .filter(pl.col("synapse_id").is_in(synapse_ids))
    .select(["synapse_id", "label"])
    .collect()
    .to_pandas()
    .set_index("synapse_id")
)
print(f"{time.time() - currtime:.3f} seconds elapsed.")

# %%
cell.annotations.post_syn.nodes["tag"] = postsyn_predictions["label"]
cell.annotations.post_syn.nodes["tag"] = cell.annotations.post_syn.nodes["tag"].fillna(
    "unknown"
)

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

skeleton_labels = cell.skeleton.get_feature("compartment")
distance_to_nucleus = cell.skeleton.distance_to_root()
cell.skeleton.add_feature(distance_to_nucleus, "distance_to_nucleus", overwrite=True)

# %%
dendrite = cell.apply_mask("skeleton", skeleton_labels == 3)

# %%

fig, ax = plt.subplots(1, 1, figsize=(8, 8))
ossify.plot.plot_morphology_2d(
    dendrite,
    color="distance_to_nucleus",
    projection="xy",
    palette="cool",
    ax=ax,
    linewidth="radius",
    linewidth_norm=(100, 500),  # Radius range for normalization
    widths=(0.05, 3),  # Final line width range
)
ax.set_title("Distance to nucleus ($\\mu$m)")
ax.set_xticks([])
ax.set_yticks([])
ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
# %%
if False:
    # TODO don't trust this part yet, something about distance to tip calculation seems off
    dists = cell.skeleton.distance_between(
        cell.skeleton.nodes.index, cell.skeleton.end_points, directed=True
    )
    min_dist_to_tip = dists.min(axis=1)
    cell.skeleton.add_feature(min_dist_to_tip, "min_dist_to_tip", overwrite=True)
    end_nodes = cell.skeleton.nodes.loc[cell.skeleton.end_points]

    fig, ax = plt.subplots(figsize=(8, 6))
    ossify.plot.plot_morphology_2d(
        dendrite,
        color="min_dist_to_tip",
        projection="xy",
        palette="coolwarm",
        ax=ax,
        linewidth="radius",
        linewidth_norm=(100, 500),  # Radius range for normalization
        widths=(0.05, 3),  # Final line width range
    )
    sns.scatterplot(data=end_nodes, x="x", y="y", color="black", s=10, ax=ax, zorder=2)
    ax.set_title("Min distance to tip ($\\mu$m)")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines[["top", "right", "left", "bottom"]].set_visible(False)

# %%
aggregation_distance = 20_000
syn_density = (
    dendrite.skeleton.map_annotations_to_feature(
        "post_syn",
        distance_threshold=aggregation_distance,
        agg="density",
        validate=True,
        chunk_size=2000,
    )
    * 1000
)
dendrite.skeleton.add_feature(syn_density.values, "post_syn_density", overwrite=True)

spine_syn_density = (
    dendrite.skeleton.map_annotations_to_feature(
        "post_syn",
        distance_threshold=aggregation_distance,
        agg="density",
        validate=True,
        chunk_size=2000,
        query="tag == 'spine'",
    )
    * 1000
)
dendrite.skeleton.add_feature(
    spine_syn_density.values, "spine_syn_density", overwrite=True
)

non_spine_syn_density = (
    dendrite.skeleton.map_annotations_to_feature(
        "post_syn",
        distance_threshold=aggregation_distance,
        agg="density",
        validate=True,
        chunk_size=2000,
        query="tag != 'spine'",
    )
    * 1000
)
dendrite.skeleton.add_feature(
    non_spine_syn_density.values, "non_spine_syn_density", overwrite=True
)


# %%
sns.set_context("talk")
fig, ax = plt.subplots(figsize=(8, 6))
sns.histplot(syn_density, ax=ax)
ax.set_xlabel("Post-synaptic density (synapses/um)")
sns.histplot(spine_syn_density, color="C1", ax=ax)
sns.histplot(non_spine_syn_density, color="C2", ax=ax)
ax.legend(
    ["All post-synaptic", "Spine post-synaptic", "Non-spine post-synaptic"],
    frameon=False,
)

# %%


def plot_graph(nodes, edges, x, y, ax, hue=None, vmin="min", vmax="max", **edge_kws):
    start_xs = nodes.loc[edges[:, 0]][x].values
    end_xs = nodes.loc[edges[:, 1]][x].values
    start_ys = nodes.loc[edges[:, 0]][y].values
    end_ys = nodes.loc[edges[:, 1]][y].values
    if hue is None:
        edge_colors = "gray"
    else:
        start_hues = nodes.loc[edges[:, 0]][hue].values
        end_hues = nodes.loc[edges[:, 1]][hue].values
        edge_hues = (start_hues + end_hues) / 2
        edge_colors = sns.color_palette("RdBu_r", as_cmap=True)
        if vmin == "min":
            vmin = nodes[hue].min()
        if vmax == "max":
            vmax = nodes[hue].max()
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        edge_colors = edge_colors(norm(edge_hues))

    pre_coords = list(zip(start_xs, start_ys))
    post_coords = list(zip(end_xs, end_ys))
    coords = list(zip(pre_coords, post_coords))

    lc = LineCollection(
        segments=coords,
        colors=edge_colors,
        **edge_kws,
    )
    ax.add_collection(lc)
    ax.autoscale()


nodes = dendrite.skeleton.nodes.copy()
nodes["distance_to_nucleus"] = nodes["distance_to_nucleus"] / 1000  # convert to microns
edges = dendrite.skeleton.edges

fig, axs = plt.subplots(
    1, 3, figsize=(15, 5), sharey=True, sharex=True, layout="constrained"
)
for i, y in enumerate(
    ["post_syn_density", "spine_syn_density", "non_spine_syn_density"]
):
    ax = axs[i]
    sns.scatterplot(
        data=nodes,
        x="distance_to_nucleus",
        y=y,
        ax=ax,
        s=10,
    )
    plot_graph(nodes, edges, x="distance_to_nucleus", y=y, ax=ax, alpha=0.2)
    ax.set_xlabel("Distance to nucleus ($\\mu$m)")
    ax.set_ylabel("Post-synapses / $\\mu$m")
    ax.set_title(
        y.replace("_", " ")
        .replace("syn", "synapse")
        .replace("density", "")
        .capitalize()
    )

# %%

fig, ax = plt.subplots(figsize=(10, 6))
plot_graph(
    dendrite.skeleton.nodes,
    dendrite.skeleton.edges,
    x="x",
    y="y",
    hue="post_syn_density",
    # hue="distance_to_nucleus",
    linewidth=3,
    ax=ax,
    vmin=0.2,
    vmax=0.6,
)
sns.scatterplot(
    data=dendrite.annotations.post_syn.nodes,
    x="ctr_pt_position_x",
    y="ctr_pt_position_y",
    color="black",
    s=1,
    ax=ax,
    zorder=2,
    linewidth=0.1,
    edgecolor="white",
)
# add a color bar
sm = plt.cm.ScalarMappable(
    cmap=sns.color_palette("RdBu_r", as_cmap=True),
    norm=plt.Normalize(vmin=0.2, vmax=0.6),
)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, shrink=0.5)
cbar.set_label("Post-synapses / $\\mu$m")
ax.invert_yaxis()
ax.axis("equal")
ax.axis("off")

# %%
# What is the distribution of the number synapses from single axons?
# Are polysynaptic axons targeting specific compartments on multipolar VIP neurons?
# Is this cell-type specific?

partner_counts = cell.annotations.post_syn.nodes["pre_pt_root_id"].value_counts()
fig, ax = plt.subplots(figsize=(8, 6))
sns.histplot(partner_counts, ax=ax, discrete=True)
ax.set_xlabel("Number of synapses from object")
ax.set_ylabel("Number of objects")
ax.set_yscale("log")

# %%
# Another question I have: Of axons making synapses onto VIP neurons that arise within
# the volume, where are their somas located?

cell_df = (
    client.materialize.query_view(
        "aibs_cell_info", split_positions=True, desired_resolution=[1, 1, 1]
    )
    .drop_duplicates("pt_root_id", keep=False)
    .set_index("pt_root_id")
)
cell_df["axon_cleaned"] = cell_df["axon_cleaned"].fillna("f") == "t"

# %%
partner_df = cell_df.loc[cell_df.index.intersection(partner_counts.index)]
fig, ax = plt.subplots(figsize=(8, 8))
sns.scatterplot(
    data=partner_df,
    x="pt_position_x",
    y="pt_position_z",
    hue="axon_cleaned",
    ax=ax,
    s=50,
)
ax.scatter(
    cell.skeleton.root_location[0], cell.skeleton.root_location[2], color="red", s=100
)

# %%
# TODO both of these are hard, for a few reasons... I can get them to you but not this week
# What is the distribution of mono/polysynaptic spines for multipolar VIP neurons?
# Are they E-E, E-I, I-I?

# %%
