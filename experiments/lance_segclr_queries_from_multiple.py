# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "caveclient[cv]>=8.1.0",
#     "deltalake>=1.6.0",
#     "ipykernel>=7.3.0",
#     "ipywidgets>=8.1.8",
#     "mmh3>=5.2.1",
#     "nglui>=4.7.4",
#     "polars>=1.41.2",
#     "pylance>=7.0.0",
#     "pyvista[all]>=0.48.4",
#     "scikit-learn>=1.9.0",
#     "seaborn>=0.13.2",
#     "umap-learn>=0.5.12",
# ]
# ///
# %%
import time
from concurrent.futures import ThreadPoolExecutor

import deltalake
import lance
import mmh3
import numpy as np
import polars as pl
import pyarrow as pa
import seaborn as sns
from caveclient import CAVEclient
from nglui.parser import StateParser
from nglui.statebuilder import ViewerState
from sklearn.metrics.pairwise import cosine_distances

LANCE_EMBEDDING_PATH = "gs://bdp-ssa/segclr/lance_embeddings"
DELTA_EMBEDDING_PATH = "gs://bdp-ssa/segclr/embeddings"
DELTA_MAPPING_PATH = "gs://bdp-ssa/segclr/condensation_maps"


delta_table = deltalake.DeltaTable(str(DELTA_EMBEDDING_PATH))
client = CAVEclient("minnie65_phase3_v1")

# %%

# this state holds my initial seed set of a handful of clean objects
state = client.state.get_state_json(5906902461448192)
parser = StateParser(state)
segs = parser.selection_dataframe().query("visible")["id"].to_list()

# TODO really should map these backward in time


# %%

N_SHARDS = 4096


def mmh3_shard(segment_id, n_shards, bytewidth=8):
    return (
        mmh3.hash(int(segment_id).to_bytes(bytewidth, "little"), signed=False)
        % n_shards
    )


query_df = pl.DataFrame(
    {
        "root_id": segs,
        "root_id_partition": [mmh3_shard(root_id, N_SHARDS) for root_id in segs],
    }
)
query_df = query_df.slice(0, 14)

#%%


# %%
print(f"Querying {len(query_df)} objects from {DELTA_EMBEDDING_PATH}")
sample_df = (
    pl.scan_delta(DELTA_EMBEDDING_PATH)
    .filter(
        pl.col("root_id_partition").is_in(query_df["root_id_partition"].to_list()),
        pl.col("root_id").is_in(query_df["root_id"].to_list()),
    )
    .join(
        query_df.lazy(),
        on=["root_id_partition", "root_id"],
        how="semi",
    )
    .collect(engine="streaming")
)
print(len(sample_df), "embeddings found", len(query_df), "objects queried")

# %%
subsample_df = sample_df.group_by("root_id").map_groups(
    lambda df: df.sample(n=min(10, len(df)))
)

# %%
res = np.array([4, 4, 40])
ts = client.materialize.get_timestamp(943)
# %%
cell_table = client.materialize.query_table("aibs_metamodel_celltypes_v661")

# %%

ds = lance.dataset(LANCE_EMBEDDING_PATH)
k = 100
query_vectors = [row for row in subsample_df["embedding"].to_list()]


def query_one(vec):
    return ds.to_table(
        nearest={
            "column": "embedding",
            "q": vec,
            "k": k,
            "metric": "cosine",
            "maximum_nprobes": 10,
        },
    )


timer = time.time()
with ThreadPoolExecutor(max_workers=16) as pool:
    results = list(pool.map(query_one, query_vectors))
tbl = pa.concat_tables(results)
timer = time.time() - timer
print(f"Batch query time ({len(query_vectors)} vectors): {timer:.3f}s")

# %%
nearest_neighbors_df = pl.from_arrow(tbl)
nearest_neighbors_df = nearest_neighbors_df.filter(
    ~pl.col("root_id").is_in(cell_table["pt_root_id"].to_list())
)

# %%

# compute distance from nearest neighbors to sample_df
known_vectors = np.vstack(
    sample_df.select(["embedding"]).to_pandas()["embedding"].to_numpy()
)
found_vectors = np.vstack(
    nearest_neighbors_df.select(["embedding"]).to_pandas()["embedding"].to_numpy()
)


distances = cosine_distances(found_vectors, known_vectors)

median_dists = np.median(distances, axis=1)

sns.histplot(median_dists, bins=100)

nearest_neighbors_df = nearest_neighbors_df.with_columns(
    pl.Series("median_distance", median_dists),
)

# %%
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
# from sklearn.manifold import TSNE
from umap import UMAP

pca = PCA(n_components=64)
pca_result = pca.fit_transform(distances)

fig, axs = plt.subplots(1, 2, figsize=(12, 6))
sns.scatterplot(
    x=pca_result[:, 0],
    y=pca_result[:, 1],
    hue=median_dists,
    palette="viridis",
    ax=axs[0],
    s=10,
    linewidth=0,
)

umap = UMAP(n_components=2, min_dist=0.7)
umap_result = umap.fit_transform(distances)

sns.scatterplot(
    x=umap_result[:, 0],
    y=umap_result[:, 1],
    hue=median_dists,
    palette="viridis",
    ax=axs[1],
    s=10,
    linewidth=0,
)


# %%

root_agg_distance = nearest_neighbors_df.group_by("root_id").agg(
    pl.col("median_distance").mean().alias("mean_median_distance")
)
root_agg_distance = root_agg_distance.join(
    nearest_neighbors_df.sort("median_distance").group_by("root_id").first(),
    on="root_id",
).sort("mean_median_distance")


# %%
# nearest_neighbors_df = nearest_neighbors_df.to_pandas()

# one_per_object = True
# if one_per_object:
#     print(len(nearest_neighbors_df), "nearest neighbors found")
#     nearest_neighbors_df_to_show = (
#         nearest_neighbors_df.sort_values("_distance")
#         .groupby("root_id", as_index=False)
#         .first()
#     )
#     print(
#         len(nearest_neighbors_df_to_show),
#         "nearest neighbors after filtering to one per object",
#     )
nearest_neighbors_df_to_show = (
    root_agg_distance.to_pandas()
)  # .sort_values("_distance")

# %%

vs = (
    ViewerState(client=client)
    .add_layers_from_client()
    .add_points(
        nearest_neighbors_df_to_show.query("mean_median_distance < 0.05"),
        point_column=["x", "y", "z"],
        data_resolution=[1, 1, 1],
        segment_column="root_id",
    )
)
vs.to_browser(browser="firefox", shorten=True)
# %%

# vs = (
#     ViewerState(client=client)
#     .add_layers_from_client()
#     .add_segments(
#         nearest_neighbors_df["root_id"].tolist(),
#     )
#     .add_points(
#         nearest_neighbors_df,
#         point_column=["x", "y", "z"],
#         data_resolution=[1, 1, 1],
#         segment_column="root_id",
#         swap_visible_segments_on_move=False,
#     )
# )
# vs.to_browser()

# %%

test_root = 864691135146960293
query_df = pl.DataFrame(
    {
        "root_id": [test_root],
        "root_id_partition": [mmh3_shard(test_root, N_SHARDS)],
    }
)
root_points = (
    pl.scan_delta(DELTA_MAPPING_PATH)
    .filter(
        pl.col("root_id_partition").is_in(query_df["root_id_partition"].to_list()),
        pl.col("root_id").is_in(query_df["root_id"].to_list()),
    )
    .join(
        query_df.lazy(),
        on=["root_id_partition", "root_id"],
        how="semi",
    )
    .collect(engine="streaming")
)

rep_points = (
    pl.scan_delta(DELTA_EMBEDDING_PATH)
    .filter(
        pl.col("root_id") == test_root,
        pl.col("root_id_partition") == mmh3_shard(test_root, N_SHARDS),
    )
    .collect(engine="streaming")
)

# %%
vs = (
    ViewerState(client=client)
    .add_layers_from_client()
    .add_points(
        root_points.to_pandas(),
        point_column=["x", "y", "z"],
        data_resolution=[1, 1, 1],
        segment_column="root_id",
    )
    .add_points(
        rep_points.to_pandas(),
        point_column=["x", "y", "z"],
        data_resolution=[1, 1, 1],
        segment_column="root_id",
        name="rep",
    )
)
vs.to_browser(browser="firefox", shorten=True)
