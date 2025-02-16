# %%
# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = [
#     "caveclient[cv]",
# ]
# ///
# %%

from caveclient import CAVEclient

client = CAVEclient("minnie65_public", version=1300)

cv = client.info.segmentation_cloudvolume()

root_id = 864691135307809094

mesh = cv.mesh.get(
    root_id, remove_duplicate_vertices=True, deduplicate_chunk_boundaries=False
)[root_id]

print(len(mesh.vertices))

print(len(mesh.faces))
