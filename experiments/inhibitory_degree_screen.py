# %%
from caveclient import CAVEclient

client = CAVEclient("minnie65_phase3_v1")

annotation_version = 1556
cell_type_df = client.materialize.tables.cell_type_multifeature_v1().query(
    desired_resolution=[1, 1, 1],
    split_positions=True,
    materialization_version=annotation_version,
)

seg_version = 1412
cell_info = (
    client.materialize.views.aibs_cell_info()
    .query(materialization_version=seg_version)
    .set_index("id")
)
cell_info["cell_type_multifeature"] = cell_type_df.set_index("id")["cell_type"]
cell_info["broad_type_multifeature"] = cell_type_df.set_index("id")[
    "classification_system"
]

# %%
query_cell_info = cell_info.query("broad_type_multifeature == 'inhibitory'")

