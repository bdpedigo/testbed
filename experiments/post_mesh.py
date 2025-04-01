# %%
import numpy as np
from meshio import read

mesh_path = "/Users/ben.pedigo/code/testbed/data/frog/files/BoredFrog.stl"

mesh = read(mesh_path)

vertices = mesh.points
faces = mesh.cells_dict["triangle"]

bounds = np.array([vertices.min(axis=0), vertices.max(axis=0)])

vertices = vertices - bounds[0]
vertices = vertices / (bounds[1] - bounds[0])
vertices = vertices[:, [0, 2, 1]]
vertices[:, 1] = -vertices[:, 1]
# vertices[:, 2] = -vertices[:, 2]
vertices = vertices * 8000

pos = np.array([263944, 162511, 24770]) * np.array([4, 4, 40])

vertices = vertices + pos
vertices = vertices.astype(np.float32)

# %%
from caveclient import CAVEclient
from cloudvolume import CloudVolume

client = CAVEclient("minnie65_phase3_v1")

base_cv = client.info.segmentation_cloudvolume()

cv = CloudVolume(
    "gs://allen-minnie-phase3/bdp-skeletons/ribbit",
    info={
        "app": {"supported_api_versions": [0, 1]},
        "chunks_start_at_voxel_offset": True,
        # "data_dir": "gs://minnie65_pcg/ws",
        "data_type": "uint64",
        # "graph": {
        #     "bounding_box": [2048, 2048, 512],
        #     "chunk_size": [256, 256, 512],
        #     "cv_mip": 0,
        #     "n_bits_for_layer_id": 8,
        #     "n_layers": 12,
        #     "spatial_bit_masks": {
        #         "1": 10,
        #         "2": 10,
        #         "3": 9,
        #         "4": 8,
        #         "5": 7,
        #         "6": 6,
        #         "7": 5,
        #         "8": 4,
        #         "9": 3,
        #         "10": 2,
        #         "11": 1,
        #         "12": 1,
        #     },
        # },
        "mesh": "mesh",
        "mesh_dir": "mesh",
        # "mesh_metadata": {
        #     "uniform_draco_grid_size": 21.0,
        #     "unsharded_mesh_dir": "dynamic",
        # },
        "num_channels": 1,
        "scales": [
            {
                "chunk_sizes": [[256, 256, 32]],
                "compressed_segmentation_block_size": [8, 8, 8],
                "encoding": "compressed_segmentation",
                "key": "8_8_40",
                "locked": True,
                "resolution": [1, 1, 1],
                # "size": [192424, 131051, 13008],
                # "voxel_offset": [26385, 30308, 14850],
            },
        ],
        "sharded_mesh": False,
        "type": "segmentation",
        "verify_mesh": False,
    },
    mesh_dir="mesh",
)

info = cv.create_new_info(
    num_channels=1,
    layer_type="segmentation",
    data_type="uint8",
    encoding="raw",
    resolution=[1, 1, 1],
    voxel_offset=[1, 1, 1],
    volume_size=[10_000_000, 10_000_000, 10_000_000],
    mesh="mesh",
)

cv = CloudVolume(
    "gs://allen-minnie-phase3/bdp-skeletons/ribbit",
    info=info,
    mesh_dir="mesh",
)

cv.commit_info()

from cloudvolume import Mesh

cv_mesh = Mesh(vertices=vertices, faces=faces, segid=99)
cv.mesh.put(
    cv_mesh,
)

# %%
