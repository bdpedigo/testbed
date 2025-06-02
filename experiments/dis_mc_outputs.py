# %%

import time

import numpy as np
from caveclient import CAVEclient
from nglui import site_utils, statebuilder

client = CAVEclient("minnie65_phase3_v1", version=1300)

cell_info = client.materialize.query_view("aibs_cell_info")


# %%
nuc_ids = [
    260541,
    260505,
    296398,
    298937,
    262678,
    301239,
    303149,
    260622,
    296735,
    264649,
]

source_cell_info = cell_info.set_index("id").loc[nuc_ids]

# %%

query_kwargs = {
    "desired_resolution": [1, 1, 1],
    "split_positions": True,
}
source_syns = client.materialize.tables.multi_input_spine_predictions_ssa(
    pre_pt_root_id=source_cell_info["pt_root_id"].values
).query(**query_kwargs)


# %%

target_types = ["L3a", "L3b"]
target_cell_info = cell_info.query(
    "(mtype_source == 'allen_column_mtypes_v2') and (mtype in @target_types)"
)

# %%
source_syns.query(
    "post_pt_root_id.isin(@target_cell_info.pt_root_id.values)", inplace=True
)

# %%
all_post_root_ids = source_syns["post_pt_root_id"].unique()
syn_group_ids = source_syns["group_id"].unique()

# %%

# note: this does not scale well, I think because there is not an index on group_id
# so it is probably more efficient to do this in chunks by post_pt_root_id and post-hoc
# filter
currtime = time.time()
coincident_syns = client.materialize.tables.multi_input_spine_predictions_ssa(
    group_id=syn_group_ids,
    post_pt_root_id=all_post_root_ids,
).query(**query_kwargs)
print(f"{time.time() - currtime:.3f} seconds elapsed.")

# %%
coincident_syns["root_ids"] = list(
    zip(coincident_syns["post_pt_root_id"], coincident_syns["pre_pt_root_id"])
)
# %%
# client.materialize.query_view("multi_input_spine_predictions_ssa").query(
#     filter_in_dict={"post_pt_root_id": all_post_root_ids[:100]}, **query_kwargs
# )
synapse_group_table = (
    coincident_syns.groupby("group_id")
    .agg(
        {
            "ctr_pt_position_x": "mean",
            "ctr_pt_position_y": "mean",
            "ctr_pt_position_z": "mean",
            "post_pt_root_id": "unique",
            "pre_pt_root_id": "unique",
            "target_id": "unique",
        }
    )
    .rename(
        columns={
            "ctr_pt_position_x": "mean_x",
            "ctr_pt_position_y": "mean_y",
            "ctr_pt_position_z": "mean_z",
            "post_pt_root_id": "post_root_ids",
            "pre_pt_root_id": "pre_root_ids",
            "target_id": "synapse_ids",
        }
    )
)
synapse_group_table["root_ids"] = [
    synapse_group_table["post_root_ids"].values[i].tolist()
    + synapse_group_table["pre_root_ids"].values[i].tolist()
    for i in range(len(synapse_group_table))
]

synapse_group_table["synapse_ids_str"] = synapse_group_table["synapse_ids"].apply(str)

mc_ids = []
for i in range(len(synapse_group_table)):
    pre_root_ids = synapse_group_table["pre_root_ids"].values[i].tolist()
    mc_mask = np.isin(
        pre_root_ids,
        source_cell_info["pt_root_id"].values,
    )
    mc_id = np.array(pre_root_ids)[mc_mask][0]
    mc_ids.append(mc_id)

synapse_group_table["mc_id"] = mc_ids

# %%
synapse_group_table.to_csv(
    "/Users/ben.pedigo/code/testbed/outs/synapse_group_table.csv"
)
coincident_syns.to_csv("/Users/ben.pedigo/code/testbed/outs/coincident_syns.csv")
# %%


fixed_ids = source_cell_info["pt_root_id"].values.tolist()
fixed_id_colors = len(source_cell_info) * ["#FFA200"]

fixed_ids += coincident_syns["post_pt_root_id"].unique().tolist()
fixed_id_colors += len(coincident_syns) * ["#00FF00"]

# NOTE coloring the pre-synaptic ids was not working, maybe NGL issue?
# non_mc_pre_ids = coincident_syns["pre_pt_root_id"].unique()
# non_mc_pre_ids = np.setdiff1d(
#     non_mc_pre_ids,
#     source_cell_info["pt_root_id"].values,
# )
# fixed_ids += non_mc_pre_ids.tolist()
# fixed_id_colors += len(non_mc_pre_ids) * ["#FF0000"]

site_utils.set_default_config("spelunker")


img, seg = statebuilder.from_client(client)

ann = statebuilder.AnnotationLayerConfig(
    "synapses",
    linked_segmentation_layer="seg",
    mapping_rules=statebuilder.PointMapper(
        # "ctr_pt_position",
        "mean",
        description_column="synapse_ids_str",
        # group_column="group_id",
        split_positions=True,
        # tag_column="tag",
        linked_segmentation_column="root_ids",
    ),
    data_resolution=[1, 1, 1],
    # tags=["soma", "shaft", "spine", "basal", "apical"],
)

state_dict = statebuilder.StateBuilder([img, seg, ann]).render_state(
    synapse_group_table.sort_values("mc_id"),
    return_as="dict",
    client=client,
    target_site="spelunker",
)

state_dict["layers"][1]["segmentColors"] = dict(
    zip([str(x) for x in fixed_ids], fixed_id_colors)
)

state_dict["layout"] = "3d"
state_dict["showAxisLines"] = True

site = "https://spelunker.cave-explorer.org/"
target_site = "spelunker"

state_id = client.state.upload_state_json(state_dict)
client.state.build_neuroglancer_url(state_id, site, target_site=target_site)
