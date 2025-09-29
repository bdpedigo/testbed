# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "caveclient>=7.11.0",
#     "joblib",
#     "seaborn",
# ]
# ///

# %%
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from caveclient import CAVEclient
from joblib import Parallel, delayed

# %%
seg_version = 1507

client = CAVEclient("minnie65_phase3_v1")

# %%

query_kwargs = dict(desired_resolution=[1, 1, 1], split_positions=True)
cell_info = client.materialize.views.aibs_cell_info().query(
    materialization_version=seg_version, **query_kwargs
)
cell_info.drop_duplicates(subset=["pt_root_id"], inplace=True, keep=False)
cell_info.set_index("pt_root_id", inplace=True)

# %%

# NOTE table not available at 1507, getting now and mapping root IDs back
timestamp = client.materialize.get_timestamp(seg_version)
new_cell_types = client.materialize.tables.cell_type_multifeature_v1_fine().query(
    **query_kwargs
)
old_root_ids = client.chunkedgraph.get_roots(
    new_cell_types["pt_supervoxel_id"].to_list(), timestamp=timestamp
)
new_cell_types["pt_root_id"] = old_root_ids
new_cell_types.dropna(subset=["pt_root_id"], inplace=True)
new_cell_types.drop_duplicates(subset=["pt_root_id"], inplace=True, keep=False)
new_cell_types.set_index("pt_root_id", inplace=True)
# filling in E/I with Casey's new table where possible
cell_info["broad_type"] = (
    new_cell_types["classification_system"]
    .reindex(cell_info.index)
    .combine_first(cell_info["broad_type"])
)
cell_info["cell_type_fine"] = new_cell_types["cell_type"].reindex(cell_info.index)


# %%
proofread_cell_info = cell_info.query("axon_cleaned == 't'").copy()
proofread_cell_info["axon_cleaned"] = True
proofread_cell_info["axon_cleaned"] = proofread_cell_info["axon_cleaned"].astype(bool)


proofreading_info = client.materialize.query_table(
    "proofreading_status_and_strategy",
    materialization_version=seg_version,
    **query_kwargs,
)
thalamic_info = (
    proofreading_info.query("status_dendrite != 't'").set_index("pt_root_id").copy()
)
thalamic_info["cell_type"] = "thalamic"
thalamic_info["broad_type"] = "thalamic"
thalamic_info["cell_type_fine"] = "thalamic"
thalamic_info["axon_cleaned"] = True

# append thalamic rows to cell_info
proofread_cell_info = pd.concat([proofread_cell_info, thalamic_info], axis=0)

# %%

post_fine_type = "ITC_c"

query_roots = cell_info.query("cell_type_fine == @post_fine_type").index.to_list()

root_chunks = np.array_split(np.array(query_roots), len(query_roots) // 10)


# %%


def get_synapse_chunk(chunk):
    syns = client.materialize.synapse_query(
        post_ids=chunk, timestamp=timestamp, **query_kwargs
    )
    syns.set_index("id", inplace=True)
    return syns


currtime = time.time()

synapse_chunks = Parallel(n_jobs=16)(
    delayed(get_synapse_chunk)(chunk) for chunk in root_chunks
)
synapses = pd.concat(synapse_chunks)
print(f"{time.time() - currtime:.3f} seconds elapsed.")


# %%

# for pre synapses, only trust the typing info if proofread
synapses["pre_broad_type"] = (
    synapses["pre_pt_root_id"].map(proofread_cell_info["broad_type"]).fillna("unknown")
)
synapses["pre_cell_type"] = (
    synapses["pre_pt_root_id"].map(proofread_cell_info["cell_type"]).fillna("unknown")
)

synapses["post_broad_type"] = (
    synapses["post_pt_root_id"].map(cell_info["broad_type"]).fillna("unknown")
)
synapses["post_cell_type"] = (
    synapses["post_pt_root_id"].map(proofread_cell_info["cell_type"]).fillna("unknown")
)
synapses["post_cell_type_fine"] = (
    synapses["post_pt_root_id"].map(new_cell_types["cell_type"]).fillna("unknown")
)
synapses["post_nucleus_depth"] = synapses["post_pt_root_id"].map(
    cell_info["pt_position_y"]
)

# note that depth is in nm, converting to um here
# negative sign is because depth is more negative in our data
synapses["delta_depth"] = (
    -(synapses["ctr_pt_position_y"] - synapses["post_nucleus_depth"]) / 1000
)

# %%

sns.set_context("talk")
fig, ax = plt.subplots(figsize=(6, 8))

sns.histplot(
    data=synapses,
    y="delta_depth",
    kde=True,
    hue="pre_broad_type",
    ax=ax,
    stat="proportion",
    hue_order=["inhibitory", "excitatory", "thalamic"],
)
ax.axhline(0, color="k", linestyle="--")
ax.set_ylabel("Synapse depth relative to nucleus (μm)")
ax.set_title(r"Proofread axons $\rightarrow$ " + post_fine_type.replace("_", ""))
sns.move_legend(ax, "upper right", title="Presynaptic type")


# # %%
# all_syns["pre_broad_type"] = query_synapses["pre_broad_type"]
# all_syns["pre_broad_type"] = all_syns["pre_broad_type"].fillna("unknown")
# all_syns["post_nucleus_depth"] = all_syns["post_pt_root_id"].map(
#     cell_info["pt_position_y"]
# )
# all_syns["delta_depth"] = (
#     -(all_syns["ctr_pt_position_y"] - all_syns["post_nucleus_depth"]) / 1000
# )

# %%
fig, ax = plt.subplots(figsize=(6, 8))

sns.histplot(
    data=synapses,
    y="delta_depth",
    kde=True,
    hue="pre_broad_type",
    ax=ax,
    stat="proportion",
    hue_order=["inhibitory", "excitatory", "thalamic", "unknown"],
    common_norm=False,
    bins=50,
)
ax.axhline(0, color="k", linestyle="--")
ax.set_ylabel("Synapse depth relative to nucleus (μm)")
ax.set_title(r"Proofread axons $\rightarrow$ " + post_fine_type.replace("_", ""))
sns.move_legend(ax, "upper right", title="Presynaptic type")

# %%

synapses["delta_depth_bin"] = pd.cut(
    synapses["delta_depth"], bins=np.arange(-300, 300, 25)
)


# %%
bin_proportions = (
    synapses.groupby(["delta_depth_bin", "pre_broad_type"], observed=True)
    .size()
    .unstack()
    .apply(lambda x: x / x.sum(), axis=1)
    .fillna(0.0)
)
bin_proportions["delta_depth_bin_center"] = bin_proportions.index.map(
    lambda x: x.mid
).astype(float)

# %%
fig, ax = plt.subplots(figsize=(6, 8))

sns.lineplot(
    data=bin_proportions,
    y="delta_depth_bin_center",
    x="excitatory",
    ax=ax,
    orient="y",
    label="excitatory",
)
sns.lineplot(
    data=bin_proportions,
    y="delta_depth_bin_center",
    x="inhibitory",
    ax=ax,
    orient="y",
    label="inhibitory",
)
sns.lineplot(
    data=bin_proportions,
    y="delta_depth_bin_center",
    x="thalamic",
    ax=ax,
    orient="y",
    label="thalamic",
)
ax.set(xlabel="Proportion of total synapses")
ax.axhline(0.0, color="k", linestyle="--")
ax.set_ylabel("Synapse depth relative to nucleus (μm)")
sns.move_legend(ax, "upper right", title="Presynaptic type")
ax.set_title(r"Proofread axons $\rightarrow$ " + post_fine_type.replace("_", ""))

# %%
