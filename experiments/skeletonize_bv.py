# %%
import pyvista as pv
from cloudvolume import CloudVolume

bv_cv = CloudVolume(
    "precomputed://https://rhoana.rc.fas.harvard.edu/ng/EM_lowres/mouse/bv"
)
# bv_cv = CloudVolume("s3://bossdb-open-data/wei2024/minnie/bv|precomputed")

raw_mesh = bv_cv.mesh.get(1)

# %%


poly = pv.make_tri_mesh(raw_mesh.vertices, raw_mesh.faces)
poly = poly.extract_largest()

plotter = pv.Plotter()

plotter.add_mesh(poly, color="red")

plotter.show()


# %%
import numpy as np

vertices = np.asarray(poly.points)
faces = poly.faces.reshape(-1, 4)[:, 1:]
mesh = (vertices, faces)


# %%
# def edges_to_lines(edges: np.ndarray) -> np.ndarray:
#     lines = np.column_stack((np.full((len(edges), 1), 2), edges))
#     return lines


# skeleton_poly = pv.PolyData(skeleton.vertices, lines=edges_to_lines(skeleton.edges))

# plotter = pv.Plotter()

# plotter.add_mesh(poly, color="red", opacity=0.2)

# plotter.add_mesh(skeleton_poly, color="black", line_width=2)

# plotter.show()

# %%
# %%
from typing import Optional

import cloudvolume
import numpy as np
import pandas as pd
from caveclient import CAVEclient
from skeletor.skeletonize import by_wavefront

skeleton = by_wavefront(mesh, waves=10, step_size=1)

#%%

#%%

def create_skeleton_bucket(
    bucket_path: str, client: CAVEclient, vertex_attributes: list[str]
):
    """
    Generates a bucket with info files for storing precomputed skeletons.

    Parameters
    ----------
    bucket_path :
        The path to the bucket where the skeletons will be stored. Follows the
        cloudvolume conventions, so will likely look like
        "gs://bucket-name/path/to/skeletons".
    client :
        The client to use for getting the base info.
    vertex_attributes :
        The list of attributes to store on the vertices of the skeleton. Radius is
        automatically included. Attributes will be added in the order provided.

    Returns
    -------
    :
        The cloudvolume object for writing skeletons to the bucket.
    :
        The attribute info to be used for each skeleton.
    """
    base_cv = client.info.segmentation_cloudvolume()

    info = base_cv.info.copy()

    info["skeletons"] = "skeletons"

    cv = cloudvolume.CloudVolume(
        "precomputed://" + bucket_path,
        info=info,
        compress=False,
    )
    cv.commit_info()

    sk_info = cv.skeleton.meta.default_info()

    attribute_info = [{"id": "radius", "data_type": "float32", "num_components": 1}]
    for attribute in vertex_attributes:
        attribute_info.append(
            {
                "id": attribute,
                "data_type": "float32",
                "num_components": 1,
            }
        )
    sk_info["vertex_attributes"] = attribute_info
    cv.skeleton.meta.info = sk_info
    cv.skeleton.meta.commit_info()
    return cv, attribute_info


def create_skeleton(
    vertices: np.ndarray,
    edges: np.ndarray,
    segid: Optional[int] = None,
    vertex_attributes: Optional[pd.DataFrame] = None,
    attribute_info: dict = None,
):
    """
    Creates a skeleton object from the provided vertices and edges, and optional
    attributes.

    Parameters
    ----------
    vertices :
        The vertices of the skeleton, provided as an (n,3) array of coordinates.
    edges :
        The edges of the skeleton, provided as an (e,2) array of vertex indices.
    segid :
        The segid to associate with the skeleton.
    vertex_attributes :
        The attributes to store on the vertices of the skeleton. If provided, the
        attribute_info must also be provided.
    attribute_info :
        The information about the attributes to be stored on the vertices.

    Returns
    -------
    :
        The skeleton object.
    """
    skeleton = cloudvolume.Skeleton(
        vertices=vertices.astype(np.float32),
        edges=edges,
        radii=np.ones(len(vertices), dtype=np.float32),
        segid=segid,
        vertex_types=None,
        extra_attributes=attribute_info,
    )
    if vertex_attributes is not None and attribute_info is not None:
        for attribute in attribute_info[1:]:
            skeleton.__setattr__(
                attribute["id"],
                vertex_attributes[attribute["id"]].values.astype(np.float32),
            )
    return skeleton


client = CAVEclient("minnie65_public")
bucket_path = "gs://allen-minnie-phase3/vasculature-skeletons"
cv, attribute_info = create_skeleton_bucket(bucket_path, client, vertex_attributes=[])

skeleton = create_skeleton(
    skeleton.vertices, skeleton.edges, segid=3, attribute_info=attribute_info
)

cv.skeleton.upload(skeleton)

# %%
