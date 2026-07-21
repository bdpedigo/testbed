# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "meshmash>=0.1.0",
#     "numpy",
#     "pyvista[all]>=0.48.4",
#     "trimesh>=4.12.1",
# ]
# ///

# %%
# Overlay the exported 200nm watertight winding iso-surface on the original
# neuron mesh (rendered transparent) and save a screenshot.
from pathlib import Path

import numpy as np
import pyvista as pv
import trimesh
from meshmash import fetch_sample_mesh

try:
    _here = Path(__file__).resolve().parent
except NameError:
    _here = Path.cwd()
out_dir = _here.parent / "outs"

# original mesh
mesh = fetch_sample_mesh("microns_neuron_sample")
orig = pv.make_tri_mesh(mesh[0].astype(np.float64), mesh[1].astype(np.int32))

# exported 200nm watertight result
wt = trimesh.load(out_dir / "watertight_winding_voxel_200nm.ply", process=False)
wt_poly = pv.make_tri_mesh(np.asarray(wt.vertices), np.asarray(wt.faces))

# interactive overlay: solid 200nm watertight mesh inside the transparent original
p = pv.Plotter(window_size=(1600, 1200))
p.add_mesh(orig, color="darkgray", opacity=0.35, label="original")
p.add_mesh(wt_poly, color="cornflowerblue", opacity=1.0, label="200nm watertight")
p.add_legend()
p.add_text("200nm watertight winding iso-surface vs original", font_size=10)
p.enable_fly_to_right_click()
p.show()
