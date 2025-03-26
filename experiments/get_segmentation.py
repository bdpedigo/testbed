# %%
# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = [
#     "caveclient[cv]",
#     "matplotlib",
# ]
# ///
# %%

import matplotlib.pyplot as plt
import numpy as np
from caveclient import CAVEclient
from cloudvolume import Bbox

client = CAVEclient("minnie65_public", version=1300)

# can pass progress=False if you don't want progress bars at download time
cv = client.info.segmentation_cloudvolume() 

# this is the name of the object in the segmentation, i.e. the name of the node
# at the highest level in the octree
root_id = 864691135463085725

layer = 5  # set layer in the chunkedgraph (octree) hierarchy, higher is larger chunks
level_ids = client.chunkedgraph.get_leaves(root_id, stop_layer=layer)

# %%
# select a chunk as an example
chunk_node_id = level_ids[0]

# %%

# download current mesh for that chunk
mesh = cv.mesh.get(
    chunk_node_id, remove_duplicate_vertices=True, deduplicate_chunk_boundaries=False
)[chunk_node_id]

print(len(mesh.vertices))

print(len(mesh.faces))

# %%

# download segmentation data at some mip level
mip = 0

bounds = np.array([mesh.vertices.min(axis=0), mesh.vertices.max(axis=0)])
bbox = Bbox(bounds[0], bounds[1], unit="nm")

# size of voxels, note they are aniostropic
voxel_size = np.array(cv.info["scales"][mip]["resolution"])

seg = cv.download(bbox, mip=mip, agglomerate=False, segids=[chunk_node_id])

seg = np.array(seg.squeeze())

mask = seg > 0

print(mask.shape)

# %%

# plot the mask in xy, xz, and zy planes

fig, axs = plt.subplots(2, 2, figsize=(10, 10), sharey="row")

# xy plane
ratio = mask.shape[0] * voxel_size[0] / (mask.shape[1] * voxel_size[1])
axs[0, 0].imshow(mask[:, :, :].sum(axis=2), cmap="binary", aspect=ratio)

# xz plane
ratio = mask.shape[0] * voxel_size[0] / (mask.shape[2] * voxel_size[2])
axs[0, 1].imshow(mask[:, :, :].sum(axis=1), cmap="binary", aspect=ratio)

# zy plane
ratio = mask.shape[2] * voxel_size[2] / (mask.shape[1] * voxel_size[1])
axs[1, 0].imshow(mask[:, :, :].sum(axis=0).T, cmap="binary", aspect=ratio)

for ax in axs.flat:
    ax.invert_yaxis()
    ax.invert_xaxis()

axs[1, 1].axis("off")

plt.show()

# %%
fig, axs = plt.subplots(2, 2, figsize=(10, 10))

axs[0, 0].plot(mesh.vertices[:, 1], mesh.vertices[:, 0], "o", markersize=1)

axs[0, 1].plot(mesh.vertices[:, 2], mesh.vertices[:, 0], "o", markersize=1)

axs[1, 0].plot(mesh.vertices[:, 1], mesh.vertices[:, 2], "o", markersize=1)

axs[1, 1].axis("off")

plt.show()