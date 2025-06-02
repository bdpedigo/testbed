# %%
import pandas as pd

link = "https://raw.githubusercontent.com/AllenInstitute/Perisomatic_Based_CellTyping/refs/heads/main/data/microns_SomaData_AllCells_v661.csv"

df = pd.read_csv(link, engine="python", index_col=0)

# %%
from caveclient import CAVEclient

client = CAVEclient("minnie65_phase3_v1")

cells = client.materialize.query_view("aibs_cell_info")

# %%

import seaborn as sns

sns.kdeplot(
    data=cells.query("broad_type != 'nonneuron'"),
    x="nuc_volume",
    hue="broad_type",
    common_norm=False,
)
sns.kdeplot(
    data=cells.query("cell_type_source == 'allen_v1_column_types_slanted_ref'"),
    x="nuc_volume",
    hue="broad_type",
    common_norm=False,
    linestyle="--",
)


# %%
root_id = cells["pt_root_id"].values[0]
client.materialize.synapse_query(pre_ids=root_id)

#%%
from cloudvolume import CloudVolume

bv_cv = CloudVolume(
    "precomputed://https://rhoana.rc.fas.harvard.edu/ng/EM_lowres/mouse/bv"
)
raw_mesh = bv_cv.mesh.get(1)