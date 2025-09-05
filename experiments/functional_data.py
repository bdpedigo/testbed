# %%
import numpy as np
import pandas as pd
from caveclient import CAVEclient
from scipy.sparse import csr_array

rf_properties = pd.read_csv(
    "https://raw.githubusercontent.com/AllenInstitute/allen_v1dd/refs/heads/main/data_frames/rf_metrics_M409828.csv"
)
rf_properties["volume"] = pd.to_numeric(
    rf_properties["volume"], errors="coerce"
).astype("Int64")

# REPLACE PATH HERE
coreg_path = "/Users/ben.pedigo/code/testbed/data/swdb2025/coregistration_1196.feather"
coreg_table = pd.read_feather(coreg_path)

merged_table = pd.merge(
    coreg_table, rf_properties, how="inner", on=["column", "volume", "plane", "roi"]
)
# %%
merged_table["has_rf_on_or_off"].sum()
# %%


client = CAVEclient("v1dd_public", version=1196)

root_ids = coreg_table["pt_root_id"].unique()

# %%
cell_table = (
    client.materialize.query_table(
        "nucleus_detection_v0", desired_resolution=[1, 1, 1], split_positions=True
    )
    .drop_duplicates("pt_root_id")
    .set_index("pt_root_id")
)
# %%
label_table = (
    client.materialize.query_table("cell_type_multifeature_v1")
    .drop_duplicates("pt_root_id")
    .set_index("pt_root_id")
)
# %%
proofread_table = client.materialize.query_table(
    "proofreading_status_and_strategy"
).set_index("pt_root_id")
# %%
cell_table["broad_type"] = label_table["classification_system"]
cell_table["is_functional"] = cell_table.index.isin(root_ids)
cell_table["is_axon_proofread"] = proofread_table["strategy_axon"].isin(
    ["axon_partially_extended", "axon_fully_extended"]
)
cell_table["is_axon_proofread"] = (
    cell_table["is_axon_proofread"].fillna(False).astype(bool)
)
# func_cell_table = cell_table.loc[root_ids]

# %%
inhibitory_root_ids = cell_table.query(
    "broad_type == 'inhibitory' & is_axon_proofread"
).index.values
query_root_ids = np.union1d(root_ids, inhibitory_root_ids)

# %%


def get_chunk(chunk_ids):
    synapse_chunk = client.materialize.synapse_query(
        pre_ids=chunk_ids,
        post_ids=query_root_ids,
        desired_resolution=[1, 1, 1],
        split_positions=True,
    )
    return synapse_chunk


from joblib import Parallel, delayed

synapse_chunks = Parallel(n_jobs=-1, verbose=True)(
    delayed(get_chunk)(chunk) for chunk in np.array_split(root_ids, 20)
)
synapses = pd.concat(synapse_chunks)

# %%
index = pd.Index(query_root_ids)

edgelist_by_id = (
    synapses.groupby(["pre_pt_root_id", "post_pt_root_id"])
    .size()
    .index.to_frame(index=False)
)
edgelist = edgelist_by_id.apply(index.get_indexer).values

adj = csr_array(
    (np.ones(edgelist.shape[0]), (edgelist[:, 0], edgelist[:, 1])),
    shape=(len(index), len(index)),
)


# %%
import matplotlib.pyplot as plt
import seaborn as sns

fig, ax = plt.subplots(1, 1, figsize=(6, 6))

sns.scatterplot(
    data=cell_table,
    x="pt_position_x",
    y="pt_position_z",
    hue="broad_type",
    ax=ax,
    s=1,
)

# %%

from matplotlib.collections import LineCollection

fig, ax = plt.subplots(1, 1, figsize=(6, 6))

sns.scatterplot(
    data=cell_table.query("is_functional"),
    x="pt_position_x",
    y="pt_position_z",
    s=15,
    ax=ax,
    color="tab:orange",
)
sns.scatterplot(
    data=cell_table.query("is_axon_proofread & broad_type == 'inhibitory'"),
    x="pt_position_x",
    y="pt_position_z",
    s=15,
    ax=ax,
    color="tab:purple",
    linewidth=0,
)
ax.axis("equal")
ax.autoscale(False)


edgelist_by_id["pre_x"] = cell_table.loc[
    edgelist_by_id["pre_pt_root_id"], "pt_position_x"
].values
edgelist_by_id["pre_y"] = cell_table.loc[
    edgelist_by_id["pre_pt_root_id"], "pt_position_z"
].values
edgelist_by_id["post_x"] = cell_table.loc[
    edgelist_by_id["post_pt_root_id"], "pt_position_x"
].values
edgelist_by_id["post_y"] = cell_table.loc[
    edgelist_by_id["post_pt_root_id"], "pt_position_z"
].values

pre_edgelist = edgelist_by_id

pre_coords = list(zip(edgelist_by_id["pre_x"], edgelist_by_id["pre_y"]))
post_coords = list(zip(edgelist_by_id["post_x"], edgelist_by_id["post_y"]))
coords = list(zip(pre_coords, post_coords))
lc = LineCollection(
    segments=coords,
    alpha=0.1,
    linewidths=0.5,
    colors="black",
    zorder=0,
)

sns.scatterplot(
    data=cell_table.query("~is_functional"),
    x="pt_position_x",
    y="pt_position_z",
    s=0.5,
    ax=ax,
    alpha=0.5,
    color="tab:blue",
    linewidth=0,
    zorder=-1,
)
ax.add_collection(lc)

# %%
one_hop_adj = adj @ adj

functional_sub_indices = index.get_indexer(root_ids)
functional_one_hop_adj = one_hop_adj[functional_sub_indices][:, functional_sub_indices]
functional_one_hop_connections = np.nonzero(functional_one_hop_adj)
functional_one_hop_connections = np.stack(functional_one_hop_connections).T
functional_one_hop_connections = index.values[functional_one_hop_connections]
functional_one_hop_connections = pd.DataFrame(
    functional_one_hop_connections, columns=["source", "target"]
)
functional_one_hop_connections = functional_one_hop_connections.query(
    "source != target"
)

# %%
sns.heatmap(data=functional_one_hop_adj.toarray())

# %%
ssi = pd.read_csv(
    "https://raw.githubusercontent.com/AllenInstitute/allen_v1dd/refs/heads/main/data_frames/surround_supression_index_M409828.csv"
)
ssi["volume"] = pd.to_numeric(ssi["volume"], errors="coerce").astype("Int64")
ssi = pd.merge(coreg_table, ssi, how="inner", on=["column", "volume", "plane", "roi"])
ssi = ssi.drop_duplicates("pt_root_id")
ssi = ssi.set_index("pt_root_id")
# %%
sorted_index = ssi.sort_values("ssi").index

# %%
functional_one_hop_adj_df = pd.DataFrame(
    functional_one_hop_adj.toarray(), index=root_ids, columns=root_ids
)
functional_one_hop_adj_df = functional_one_hop_adj_df.reindex(
    index=sorted_index, columns=sorted_index
)

fig, ax = plt.subplots(1, 1, figsize=(10, 10))
sns.heatmap(data=functional_one_hop_adj_df, cbar=False, square=True, ax=ax)

# %%
positions = cell_table.loc[index, ["pt_position_x", "pt_position_z"]].values
from sklearn.metrics import pairwise_distances

distances = pairwise_distances(positions)

from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform

linkage_matrix = linkage(squareform(distances), method="ward")
ordered_indices = leaves_list(linkage_matrix)
ordered_index = index[ordered_indices]
distances = pd.DataFrame(distances, index=index, columns=index)
distances = distances.reindex(index=ordered_index, columns=ordered_index)

fig, axs = plt.subplots(1, 2, figsize=(20, 10))

ax = axs[0]
sns.heatmap(distances, ax=ax, xticklabels=False, yticklabels=False, cbar=False)

ax = axs[1]
sns.heatmap(
    functional_one_hop_adj_df.reindex(index=ordered_index, columns=ordered_index),
    ax=ax,
    xticklabels=False,
    yticklabels=False,
    cbar=False,
)

# %%
