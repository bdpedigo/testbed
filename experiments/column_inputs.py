# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "caveclient>=8.0.1",
#     "deltalake>=1.5.0",
#     "ipykernel>=7.2.0",
#     "ipywidgets>=8.1.8",
#     "polars>=1.39.3",
#     "pyarrow>=23.0.1",
#     "seaborn>=0.13.2",
#     "tqdm>=4.67.3",
# ]
# ///
# %%
import time

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns
from caveclient import CAVEclient

client = CAVEclient("minnie65_phase3_v1", version=1412)


column_types = client.materialize.tables.allen_v1_column_types_slanted_ref().query()

column_types = pl.from_pandas(column_types).lazy()

local_path = (
    "/Users/ben.pedigo/code/meshrep/meshrep/data/synapses_pni_2_v1412_deltalake"
)
cloud_path = (
    "gs://allen-minnie-phase3/mat_deltalakes/v1412/synapses_pni_2_v1412_deltalake"
)
synapses = pl.scan_delta(cloud_path)

synapses = synapses.join(
    column_types.select(
        pl.col("pt_root_id").alias("post_pt_root_id"),
        pl.col("cell_type").alias("post_cell_type"),
    ),
    on="post_pt_root_id",
    how="left",
).join(
    column_types.select(
        pl.col("pt_root_id").alias("pre_pt_root_id"),
        pl.col("cell_type").alias("pre_cell_type"),
    ),
    on="pre_pt_root_id",
    how="left",
)
synapses = synapses.with_columns(
    pl.col("pre_cell_type").is_not_null().alias("pre_is_column"),
    pl.col("post_cell_type").is_not_null().alias("post_is_column"),
)

grouping = (
    synapses.filter(pl.col("post_is_column"))
    .group_by("pre_is_column", "post_pt_root_id", "post_cell_type")
    .agg(pl.len())
)

currtime = time.time()
result = grouping.collect(engine="streaming")
print(f"{time.time() - currtime:.3f} seconds elapsed.")
# %%

cell_type_info = (
    result.pivot(
        index=["post_pt_root_id", "post_cell_type"], on="pre_is_column", values="len"
    )
    .fill_null(0)
    .rename(
        {"true": "pre_is_column", "false": "pre_not_column"},
    )
    .with_columns(
        (
            pl.col("pre_is_column")
            / (pl.col("pre_is_column") + pl.col("pre_not_column"))
        ).alias("p_pre_is_column")
    )
)


CELL_TYPE_CATEGORIES = [
    "23P",
    "4P",
    "5P-IT",
    "5P-PT",
    "5P-NP",
    "6P-IT",
    "6P-CT",
    "BC",
    "BPC",
    "MC",
    "NGC",
]

cell_type_info = cell_type_info.filter(
    pl.col("post_cell_type").is_in(CELL_TYPE_CATEGORIES)
)
cell_type_info = cell_type_info.with_columns(
    pl.col("post_cell_type").cast(pl.Enum(categories=CELL_TYPE_CATEGORIES))
)


sns.set_context("talk")
fig, ax = plt.subplots(figsize=(8, 6))

sns.stripplot(
    data=cell_type_info,
    x="post_cell_type",
    y="p_pre_is_column",
    ax=ax,
)
ax.set_xticks(ax.get_xticks())
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
ax.set_ylabel("Proportion of input synapses\nfrom column cells")
ax.set_xlabel("Post-synaptic cell type")

ax.spines[["top", "right"]].set_visible(False)
