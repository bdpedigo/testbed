# %%
from caveclient import CAVEclient

minnie_client = CAVEclient("minnie65_public", version=1507)

proofreading_table = minnie_client.materialize.query_table(
    "proofreading_status_and_strategy"
)
proofreading_table.query("status_dendrite == 'f'", inplace=True)

root_ids = proofreading_table["pt_root_id"].unique().tolist()
# %%
