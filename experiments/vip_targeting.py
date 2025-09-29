# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "cortical-tools",
#     "seaborn",
# ]
# ///

# %%
import matplotlib.pyplot as plt
import seaborn as sns
from cortical_tools.datasets.microns_prod import MicronsProdClient

mpc = MicronsProdClient()
mpc.set_export_cloudpath("gs://mat_dbs/public/minnie65_phase3_v1")
# %%
seg_version = 1507
synapses = mpc.exports.get_table("synapses_with_axon_proofreading", version=seg_version)
synapses["ctr_pt_position_x"] = synapses["ctr_pt_position_x"] * 4
synapses["ctr_pt_position_y"] = synapses["ctr_pt_position_y"] * 4
synapses["ctr_pt_position_z"] = synapses["ctr_pt_position_z"] * 40

# %%

query_kwargs = dict(desired_resolution=[1, 1, 1], split_positions=True)
cell_info = mpc.views.aibs_cell_info().query(
    materialization_version=seg_version, **query_kwargs
)
cell_info.drop_duplicates(subset=["pt_root_id"], inplace=True, keep=False)
cell_info.set_index("pt_root_id", inplace=True)
cell_info

# %%

# NOTE table not available at 1507, getting now and mapping root IDs back
timestamp = mpc.cave.materialize.get_timestamp(seg_version)
new_cell_types = mpc.tables.cell_type_multifeature_v1_fine().query(**query_kwargs)
old_root_ids = mpc.cave.chunkedgraph.get_roots(
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
# %%

synapses["pre_broad_type"] = synapses["pre_pt_root_id"].map(cell_info["broad_type"])
synapses["post_broad_type"] = synapses["post_pt_root_id"].map(cell_info["broad_type"])
synapses["pre_cell_type"] = synapses["pre_pt_root_id"].map(cell_info["cell_type"])
synapses["post_cell_type"] = synapses["post_pt_root_id"].map(cell_info["cell_type"])
synapses["post_cell_type_fine"] = synapses["post_pt_root_id"].map(
    new_cell_types["cell_type"]
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

post_fine_type = "ITC_c"
query_synapses = synapses.query("post_cell_type_fine == @post_fine_type")

# %%

sns.set_context("talk")
fig, ax = plt.subplots(figsize=(6, 8))

sns.histplot(
    data=query_synapses,
    y="delta_depth",
    kde=True,
    hue="pre_broad_type",
    ax=ax,
    stat="proportion",
    hue_order=["inhibitory", "excitatory"],
)
ax.axhline(0, color="k", linestyle="--")
ax.set_ylabel("Synapse depth relative to nucleus (μm)")
ax.set_title(r"Proofread axons $\rightarrow$ " + post_fine_type.replace("_", ""))
sns.move_legend(ax, "upper right", title="Presynaptic type")


