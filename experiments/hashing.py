# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "caveclient>=8.0.1",
#     "deltalake>=1.5.0",
#     "ipykernel>=7.2.0",
#     "ipywidgets>=8.1.8",
#     "mmh3>=5.2.1",
#     "polars>=1.39.3",
#     "pyarrow>=23.0.1",
#     "pyvista>=0.48.4",
#     "scipy>=1.17.1",
#     "seaborn>=0.13.2",
#     "tqdm>=4.67.3",
# ]
# ///
# %%


import polars as pl

local_path = (
    "/Users/ben.pedigo/code/meshrep/meshrep/data/synapses_pni_2_v1412_deltalake"
)
synapses = pl.scan_delta(local_path)

# %%

import mmh3


def mmh3_shard(segment_id, n_shards, bytewidth=8):
    return mmh3.hash(segment_id.to_bytes(bytewidth, "little"), signed=False) % n_shards

# %%

post_ids = synapses.select(pl.col("post_pt_root_id")).collect().to_series().to_list()

# %%
n_shards = 64

shards = [mmh3_shard(pid, n_shards=n_shards) for pid in post_ids]


# %%

import numpy as np

shards = np.array(shards)
import matplotlib.pyplot as plt

plt.hist(shards, bins=n_shards)

#%%
_, counts = np.unique(shards, return_counts=True)

# cv of counts
cv = np.std(counts) / np.mean(counts)
print(f"CV of shard counts: {cv:.4f}")

#%%

post_ids
