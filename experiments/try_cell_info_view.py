# %%
from caveclient import CAVEclient

client = CAVEclient("minnie65_phase3_v1")

out = client.materialize.query_view(
    "aibs_cell_info",
    materialization_version=1376,
    filter_equal_dict={"broad_type": "inhibitory"},
)


# %%
