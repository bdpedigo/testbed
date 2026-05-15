#%%
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "cloud-volume==12.12.0",
# ]
# ///

import traceback

import cloudvolume

print("\n\n\n\n\n")

print(cloudvolume.__version__)
print("\n\n\n\n\n")

back = cloudvolume.from_cloudpath("precomputed://https://syconn.esc.mpcdf.mpg.de/notebook/j0251/72_seg_20210127_agglo2_syn_20220811_celltypes_20230822/sv")
print(back)
print(dir(back))
print(back.get(16264492))
quit()


#%%



# %%
try:
    cv = cloudvolume.CloudVolume(
        "precomputed://https://syconn.esc.mpcdf.mpg.de/notebook/j0251/72_seg_20210127_agglo2_syn_20220811_celltypes_20230822/sv",
    )
except Exception as e:
    print(f"Error: {e}")
    traceback.print_exc()

print("\n\n\n\n\n")
# %%

try:
    cv = cloudvolume.CloudVolume(
        "precomputed://https://syconn.esc.mpcdf.mpg.de/notebook/j0251/72_seg_20210127_agglo2_syn_20220811_celltypes_20230822",
        mesh_dir="sv",
    )
except Exception as e:
    print(f"Error: {e}")
    traceback.print_exc()
print("\n\n\n\n\n")

#%%

import requests
import struct
import numpy as np


def decode_mesh(mesh_data):
    num_mesh_vertices = struct.unpack('<I', mesh_data[0:4])[0]
    vertices = np.frombuffer(mesh_data, dtype='<f4', count=num_mesh_vertices * 3, offset=4).reshape(-1, 3)

    faces_offset = 4 + num_mesh_vertices * 12
    faces = np.frombuffer(mesh_data[faces_offset:], dtype='<u4').reshape(-1, 3)
    return vertices, faces

def get_songbird_mesh(neuron_id) -> tuple[np.ndarray, np.ndarray]:
    mesh_url = f"https://syconn.esc.mpcdf.mpg.de/notebook/j0251/72_seg_20210127_agglo2_syn_20220811_celltypes_20230822/sv/{neuron_id}:0:{neuron_id}_mesh"
    mesh_response = requests.get(mesh_url)

    if mesh_response.status_code == 200:
        mesh_data = mesh_response.content
        return decode_mesh(mesh_data)
    else:
        raise Exception(f"Failed to fetch mesh for neuron ID {neuron_id}. Status code: {mesh_response.status_code}")

neuron_id = 16264492

mesh = get_songbird_mesh(neuron_id)

#%%

import pyvista as pv 

plotter = pv.Plotter()

plotter.add_mesh(pv.make_tri_mesh(*mesh), color='darkgrey')
plotter.enable_fly_to_right_click()
plotter.show()
