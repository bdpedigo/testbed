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
# ]
# ///
# %%
import time

import deltalake
import lance
import mmh3
import numpy as np
import polars as pl
from caveclient import CAVEclient
from nglui.statebuilder import ViewerState

LANCE_EMBEDDING_PATH = "gs://bdp-ssa/segclr/lance_embeddings"
DELTA_EMBEDDING_PATH = "gs://bdp-ssa/segclr/embeddings"
DELTA_MAPPING_PATH = "gs://bdp-ssa/segclr/condensation_maps"


delta_table = deltalake.DeltaTable(str(DELTA_EMBEDDING_PATH))

# %%
res = np.array([4, 4, 40])


queries = {
    "myelinated_axon": {
        "root_id": 864691136143864983,
        "pt": None,
    },
    "pnn": {"root_id": 864691135338484243, "pt": None},
    "hypertargeting_bouton": {
        "root_id": 864691135891799433,
        "pt": [318531, 187183, 19369],
    },
    "mega_pancake": {"root_id": 864691135760577870, "pt": [334114, 123538, 25528]},
    "bowtie": {"root_id": 864691136330810602, "pt": [315067, 125502, 19647]},
    "cilia": {"root_id": 864691135856595758, "pt": [280587, 198356, 20138]},
    "soma": {"root_id": 864691135856595758, "pt": [278082, 200220, 20232]},
    "astrocyte": {"root_id": 864691136196481868, "pt": [206006, 206445, 18091]},
}

client = CAVEclient("minnie65_phase3_v1")
ts = client.materialize.get_timestamp(943)

query = queries["pnn"]

root_id = query["root_id"]
pt = query["pt"]

if pt is not None:
    pt = np.array(pt) * res

    if pt is not None:
        cv = client.info.segmentation_cloudvolume()
        out = cv.scattered_points(
            [pt], agglomerate=True, coord_resolution=[1, 1, 1], timestamp=ts
        )
        root_id = list(out.values())[0]
        print(root_id)


# %%

N_SHARDS = 4096


def mmh3_shard(segment_id, n_shards, bytewidth=8):
    return (
        mmh3.hash(int(segment_id).to_bytes(bytewidth, "little"), signed=False)
        % n_shards
    )


root_id = 864691134072866404
query_df = pl.DataFrame(
    {
        "root_id": [root_id],
        "root_id_partition": [mmh3_shard(root_id, N_SHARDS)],
    }
)

sample_df = (
    pl.scan_delta(DELTA_EMBEDDING_PATH)
    .filter(
        pl.col("root_id_partition") == mmh3_shard(root_id, N_SHARDS),
        pl.col("root_id") == root_id,
    ).join(
        query_df.lazy(),
        on=["root_id_partition", "root_id"],
        how="anti",
    )
    .collect()
)
print(len(sample_df), "embeddings found for root_id", root_id)

# %%

if pt is None:
    row = sample_df.row(0, named=True)
    vec = row["embedding"]
else:
    # find the closest x,y,z to the point
    sample_df = sample_df.with_columns(
        (
            (pl.col("x") - pt[0]) ** 2
            + (pl.col("y") - pt[1]) ** 2
            + (pl.col("z") - pt[2]) ** 2
        ).alias("dist")
    )

    row = sample_df.sort("dist").row(0, named=True)
    vec = row["embedding"]


# %%

# use lance and the ivf_sq index to query the embedding


ds = lance.dataset(LANCE_EMBEDDING_PATH)
k = 500
timer = time.time()
tbl = ds.to_table(
    nearest={
        "column": "embedding",
        "q": vec,
        "k": k,
        "metric": "cosine",
        "maximum_nprobes": 10,
    },
    # limit=k,
    # fast_search=True,
)
timer = time.time() - timer
print(f"Query time: {timer:.3f}s")

nearest_neighbors_df = pl.from_arrow(tbl).to_pandas()

# %%
one_per_object = True
if one_per_object:
    print(len(nearest_neighbors_df), "nearest neighbors found")
    nearest_neighbors_df = (
        nearest_neighbors_df.sort_values("_distance")
        .groupby("root_id", as_index=False)
        .first()
    )
    print(
        len(nearest_neighbors_df), "nearest neighbors after filtering to one per object"
    )
nearest_neighbors_df = nearest_neighbors_df.sort_values("_distance")

# %%

vs = (
    ViewerState(client=client)
    .add_layers_from_client()
    .add_points(
        nearest_neighbors_df,
        point_column=["x", "y", "z"],
        data_resolution=[1, 1, 1],
        segment_column="root_id",
    )
)
vs.to_browser()

# %%

vs = (
    ViewerState(client=client)
    .add_layers_from_client()
    .add_segments(
        nearest_neighbors_df["root_id"].tolist(),
    )
    .add_points(
        nearest_neighbors_df,
        point_column=["x", "y", "z"],
        data_resolution=[1, 1, 1],
        segment_column="root_id",
        swap_visible_segments_on_move=False,
    )
)
vs.to_browser()
