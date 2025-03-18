# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#     "caveclient[cv]==7.6.0",
#     "gpytoolbox==0.3.3",
#     "pyvista[all]==0.44.2",
#     "scikit-learn",
# ]
# ///
#%%
from caveclient import CAVEclient

client = CAVEclient("minnie65_phase3_v1", version=1300)

# %%

cv = client.info.segmentation_cloudvolume()

# %%
root_id = 864691137021245294

raw_mesh = cv.mesh.get(
    root_id, remove_duplicate_vertices=True, deduplicate_chunk_boundaries=False
)[root_id]
raw_mesh = (raw_mesh.vertices, raw_mesh.faces)
# %%

from meshmash import read_condensed_features

condensed_features, mesh_labels = read_condensed_features(
    "gs://bdp-ssa/minnie/foggy-forest-call/features/864691137021245294.npz"
)

# %%
from joblib import load

model = load(
    "/Users/ben.pedigo/code/meshrep/meshrep/experiments/foggy-forest-call/model.joblib"
)

X = condensed_features.drop(columns=["x", "y", "z"], index=-1)
# %%
import pandas as pd

y = model.predict(X)
y = pd.Series(y, index=X.index)
y = y.reindex(condensed_features.index)

# %%
mesh_y = y[mesh_labels]

# %%
import pyvista as pv

plotter = pv.Plotter()

plotter.add_mesh(pv.make_tri_mesh(*raw_mesh), scalars=mesh_y)

plotter.enable_fly_to_right_click()

plotter.show()


# %%
from meshmash import get_label_components

components = get_label_components(raw_mesh, mesh_y)

# %%
