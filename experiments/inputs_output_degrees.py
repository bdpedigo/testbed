# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#     "caveclient>=7.6.0",
#     "ipykernel",
#     "pandas>=2.2.3",
#     "seaborn>=0.13.2",
# ]
# ///

# %%
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from caveclient import CAVEclient

# %%
connections = pd.read_csv(
    "/Users/ben.pedigo/code/testbed/data/connections_with_nuclei.csv.gz", header=None
)

# %%
connections.columns = [
    "pre_pt_root_id",
    "post_pt_root_id",
    "n_syn",
    "sum_size",
    "pre_nuc_id",
    "post_nuc_id",
]

# %%
connections.query("pre_pt_root_id != 0 and post_pt_root_id != 0", inplace=True)

# %%

client = CAVEclient("minnie65_phase3_v1", version=1300)
cells = client.materialize.query_view("aibs_cell_info")

# %%

cells.query("broad_type == 'excitatory' and dendrite_cleaned== 't'", inplace=True)


# %%
out_degrees = connections.groupby("pre_pt_root_id").size()

# %%


n_cells = 40
root_ids = cells.sample(n_cells)["pt_root_id"].values
input_output_degrees_by_root = []
for root_id in root_ids:
    inputs = (
        connections.query("post_pt_root_id == @root_id")
        .groupby("pre_pt_root_id")
        .size()
        .index
    )
    inputs_output_degrees = (
        out_degrees.loc[inputs].rename("output_n_synapses").to_frame().reset_index()
    )

    inputs_output_degrees["post_pt_root_id"] = root_id
    input_output_degrees_by_root.append(inputs_output_degrees)

degree_info = pd.concat(input_output_degrees_by_root)


# %%
sns.set_context("talk", font_scale=1)
sns.set_style("whitegrid")

fig, ax = plt.subplots(figsize=(10, 6))

palette = sns.color_palette("husl", n_cells)
palette = dict(zip(root_ids, palette))
log_scale = True
cumulative = False
bins = 100
sns.histplot(
    data=degree_info,
    x="output_n_synapses",
    ax=ax,
    log_scale=log_scale,
    element="poly",
    hue="post_pt_root_id",
    palette=palette,
    legend=False,
    common_norm=False,
    bins=bins,
    stat="proportion",
    fill=False,
    cumulative=cumulative,
    linewidth=1,
)
sns.histplot(
    data=degree_info,
    x="output_n_synapses",
    ax=ax,
    log_scale=log_scale,
    element="poly",
    legend=False,
    common_norm=False,
    bins=bins,
    stat="proportion",
    fill=False,
    cumulative=cumulative,
    linewidth=1,
    color="black",
)
# ax.set_yscale("log")
ax.set_xlabel("# total output synapses for presynaptic object")
ax.set_ylabel("Proportion of presynaptic objects")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# %%
import numpy as np

bins = np.geomspace(0.5, 1e4, 1000)

degree_info["log_output_n_synapses"] = np.log10(degree_info["output_n_synapses"])

degree_info["bin"] = pd.cut(
    degree_info["output_n_synapses"], bins=bins, labels=bins[:-1]
)
degree_hist = (
    degree_info.groupby("post_pt_root_id", sort=True)["bin"]
    .value_counts(normalize=True)
    .reset_index()
)


fig, ax = plt.subplots()

sns.lineplot(
    data=degree_hist,
    x="bin",
    y="proportion",
    hue="post_pt_root_id",
    palette=palette,
    legend=False,
)

ax.set_yscale("log")
ax.set_xscale("log")
