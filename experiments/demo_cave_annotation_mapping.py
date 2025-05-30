# I have only tested this with Python 3.12
# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#     "cave-mapper",
#     "nglui",
# ]
# ///

import logging
from pathlib import Path

import urllib3
from cave_mapper import map_points_via_mesh
from caveclient import CAVEclient
from nglui import parser

logging.getLogger("urllib3").setLevel(logging.CRITICAL)

client = CAVEclient("minnie65_public")

path = Path("/Users/ben.pedigo/code/meshrep/meshrep/data/hand_label_mapping")

ngl_link = "https://spelunker.cave-explorer.org/#!middleauth+https://global.daf-apis.com/nglstate/api/v1/5259593667051520"
ngl_state = int(ngl_link.split("/")[-1].split("+")[-1])

state_info = client.state.get_state_json(ngl_state)

label_df = parser.annotation_dataframe(
    state_info, expand_tags=True, point_resolution=[1, 1, 1], split_points=True
)
print("Found annotations:", len(label_df))

object_df = parser.selection_dataframe(state_info)
object_df.query("visible", inplace=True)

if len(object_df) != 1:
    raise ValueError(
        "Expected exactly one object in the selection, but found multiple or none."
    )
root_id = object_df["id"].values[0]

print("Found object:", root_id)
# %%

points = label_df[["point_x", "point_y", "point_z"]].values

mapping_info = map_points_via_mesh(
    points, root_id, client, max_distance=4, verbose=True
)

assert mapping_info["mesh_voxel_distance_nm"].max() < 150

print("Mapping complete")
print(mapping_info.head())

# %%

cv = client.info.segmentation_cloudvolume()
seg_voxel_resolution = cv.meta.resolution(0)


quit()  # comment this out to keep going, but you'd need to fill in the description, table name, etc.

# %%
table_name = "bdp_point_compartment_labels"
description = "Manual labels put down on mesh surfaces to describe neuron parts. Currently, categories are only 'soma', 'shaft', or 'spine'. Table is a work in progress and may be updated or moved into other tables in the future."
client.annotation.create_table(
    table_name,
    schema_name="bound_tag",
    voxel_resolution=seg_voxel_resolution,
    description=description,
)

# %%

stager = client.annotation.stage_annotations(table_name)

upload_df = mapping_info[["voxel_pt_x", "voxel_pt_y", "voxel_pt_z", "tag"]].rename(
    columns={
        "voxel_pt_x": "pt_position_x",
        "voxel_pt_y": "pt_position_y",
        "voxel_pt_z": "pt_position_z",
    }
)
upload_df["pt_position"] = upload_df.apply(
    lambda x: [x["pt_position_x"], x["pt_position_y"], x["pt_position_z"]],
    axis=1,
)
upload_df = upload_df.drop(["pt_position_x", "pt_position_y", "pt_position_z"], axis=1)
stager.add_dataframe(upload_df)

client.annotation.upload_staged_annotations(stager)


# %%
count = client.annotation.get_annotation_count(table_name)

count
