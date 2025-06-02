# %%
from caveclient import CAVEclient

datastack = "minnie65_public"
version = 1300
client = CAVEclient(datastack, version=version)
query_params = dict(split_positions=True, desired_resolution=[1, 1, 1])
cell_table = client.materialize.query_view("aibs_cell_info", **query_params)

# %%
cell_table.to_csv(
    "/Users/ben.pedigo/code/testbed/data/capocaccia/cell_info.csv", index=False
)
cell_table.to_csv(
    "/Users/ben.pedigo/code/testbed/data/capocaccia/cell_info.csv.gz", index=False
)
# %%
column_cell_table = cell_table.query(
    "broad_type_source == 'allen_v1_column_types_slanted_ref'"
)
column_cell_table.to_csv(
    "/Users/ben.pedigo/code/testbed/data/capocaccia/cell_info_column.csv", index=False
)
column_cell_table.to_csv(
    "/Users/ben.pedigo/code/testbed/data/capocaccia/cell_info_column.csv.gz",
    index=False,
)

# %%
column_root_ids = column_cell_table["pt_root_id"].unique()
column_synapse_table = client.materialize.synapse_query(
    pre_ids=column_root_ids, post_ids=column_root_ids, **query_params
)

# %%
column_synapse_table = column_synapse_table.drop(
    columns=["created", "superceded_id", "valid"]
)

# %%
column_synapse_table.to_csv(
    "/Users/ben.pedigo/code/testbed/data/capocaccia/synapses_column.csv", index=False
)
column_synapse_table.to_csv(
    "/Users/ben.pedigo/code/testbed/data/capocaccia/synapses_column.csv.gz", index=False
)

# %%
