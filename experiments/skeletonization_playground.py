# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "caveclient[cv]>=8.1.0",
#     "ipykernel>=7.3.0",
#     "ipywidgets>=8.1.8",
#     "meshmash>=0.1.0",
#     "pyvista[all]>=0.48.4",
#     "robust-laplacian>=1.1.0",
#     "scikit-learn>=1.9.0",
#     "seaborn>=0.13.2",
#     "skeletor>=1.6.0",
# ]
# ///


# %%
import time

from meshmash import fetch_sample_mesh
from skeletor.skeletonize import by_wavefront

mesh = fetch_sample_mesh("microns_neuron_sample")

timer = time.time()
out = by_wavefront(mesh)
print(f"Skeletonization took {time.time() - timer:.2f} seconds")
#%%

