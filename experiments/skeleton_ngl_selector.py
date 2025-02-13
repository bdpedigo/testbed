# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = [
#    "nglui>=3.7.1",
#    "pcg-skel>=1.1.0",
#    "pyvista[all]>=0.44.1",
# ]
# ///
# %%

import numpy as np
import pandas as pd
import pyvista as pv
from caveclient import CAVEclient
from joblib import Parallel, delayed
from nglui.segmentprops import SegmentProperties
from nglui.statebuilder import StateBuilder, from_client, site_utils
from pcg_skel import pcg_meshwork
from pcg_skel.features import add_is_axon_annotation

PLOT = True
# %%
client = CAVEclient("minnie65_phase3_v1", version=1300)

type_df = client.materialize.query_table(
    "allen_v1_column_types_slanted_ref", desired_resolution=[1, 1, 1]
)
type_df.query("classification_system== 'aibs_coarse_inhibitory'", inplace=True)
root_ids = type_df["pt_root_id"].values
type_df = type_df.set_index("pt_root_id")

# %%
root_id = type_df.index[0]
nuc_loc = type_df.loc[root_id, "pt_position"]
print("Skeletonizing...")
skel = pcg_meshwork(
    root_id,
    client=client,
    root_point=nuc_loc,
    root_point_resolution=[1, 1, 1],
    collapse_soma=True,
    synapses=True,
)
add_is_axon_annotation(skel, "pre_syn", "post_syn")

# %%


if PLOT:
    vertices = np.array(skel.skeleton.vertices)
    edges = skel.skeleton.edges
    lines = np.column_stack([np.full(len(edges), 2), edges]).ravel()

    skel_poly = pv.PolyData(vertices, lines=lines)

    pv.set_jupyter_backend("client")
    plotter = pv.Plotter()
    plotter.add_mesh(skel_poly, color="black", line_width=10)
    plotter.show()


# %%
compartment_labels = np.empty(len(skel.skeleton.vertices), dtype="object")

axon_mask = skel.mesh_property_to_skeleton(
    skel.anno.is_axon.mesh_mask, aggfunc="median"
).astype(bool)

compartment_labels[axon_mask] = "axon"
compartment_labels[~axon_mask] = "dendrite"
compartment_labels[skel.skeleton.root] = "soma"

# %%
if PLOT:
    plotter = pv.Plotter()
    plotter.add_mesh(
        skel_poly,
        scalars=pd.Series(compartment_labels)
        .map({"axon": 1, "soma": 0, "dendrite": -1})
        .astype(float),
        line_width=5,
        interpolate_before_map=False,
        cmap="coolwarm",
    )
    plotter.show()

# %%
lvl2_ids = skel.anno["lvl2_ids"]["lvl2_id"]

mesh_to_skel = skel.skeleton.mesh_to_skel_map_base

lvl2_to_skel = dict(zip(lvl2_ids, mesh_to_skel.tolist()))

# %%
print("Computing branch order...")
skeleton = skel.skeleton
branch_points = skel.skeleton.branch_points
branch_order = np.zeros(len(skeleton.vertices), dtype=int)
include_non_axon = set()
for i in range(len(skeleton.vertices)):
    path = skeleton.path_to_root(i)
    # added this bit in to also keep parts of putative dendrite if they get passed over
    # on the way back to the soma
    if compartment_labels[i] == "axon":
        path_labels = compartment_labels[path]
        include_non_axon.update(path[path_labels != "axon"].tolist())
    branch_order[i] = np.isin(branch_points, path).sum().item() - 1
include_non_axon = np.array(list(include_non_axon))

# %%
if PLOT:
    pv.set_jupyter_backend("client")
    plotter = pv.Plotter()
    plotter.add_mesh(skel_poly, scalars=branch_order.astype(float), line_width=10)
    plotter.show()

# %%

level2_data = pd.DataFrame(index=lvl2_ids)
level2_data["skeleton_index"] = [lvl2_to_skel[i] for i in lvl2_ids]
level2_data["compartment"] = compartment_labels[level2_data["skeleton_index"]].tolist()
level2_data["branch_order"] = branch_order[level2_data["skeleton_index"]].tolist()
level2_data["include"] = level2_data["compartment"].isin(
    ["axon", "soma"]
) | level2_data["skeleton_index"].isin(include_non_axon)


# %%

print("Computing minimal covering nodes...")


def compute_cover_for_order(order):
    level2_ids = level2_data.query("include and branch_order <= @order").index
    covering_nodes = client.chunkedgraph.get_minimal_covering_nodes(level2_ids)
    return covering_nodes


max_order = 10

level2_axon_ids_by_order = Parallel(n_jobs=-1)(
    delayed(compute_cover_for_order)(order) for order in range(max_order)
)


# %%

print("Generating Neuroglancer link...")
site_utils.set_default_config(target_site="spelunker")

order_df = pd.DataFrame(
    {"order": range(max_order), "axon_ids": level2_axon_ids_by_order}
)
order_df = order_df.explode("axon_ids")

seg_prop = SegmentProperties.from_dataframe(
    order_df, tag_value_cols="order", id_col="axon_ids"
)
prop_id = client.state.upload_property_json(seg_prop.to_dict())

ngl_url = "https://spelunker.cave-explorer.org/"
prop_url = client.state.build_neuroglancer_url(
    prop_id, ngl_url=ngl_url, format_properties=True
)

img, seg = from_client(client, use_skeleton_service=True)
seg.color = "red"
_, seg2 = from_client(client, segmentation_name="branches")

seg.add_selection_map(fixed_ids=[root_id])

seg2.add_segment_propeties(prop_url)
seg2.add_selection_map(
    fixed_ids=order_df["axon_ids"].tolist(),
    fixed_id_colors=len(order_df) * ["#ff9333"],
)

statebuilder = StateBuilder([img, seg, seg2], client=client)
state_dict = statebuilder.render_state([], client=client, return_as="dict")
state_dict["layers"][2]["segments"].clear()
state_dict["layers"][1]["objectAlpha"] = 0.1
state_dict["layers"][1]["segmentColors"] = {
    f"{root_id}": "#ff9333",
}

url = StateBuilder(
    [],
    base_state=state_dict,
    view_kws={"position": nuc_loc / np.array([4, 4, 40]), "layout": "3d"},
).render_state([], client=client, return_as="short")

print(url)
print()


######

# %%

max_order = 10


def run_for_root(root_id):
    nuc_loc = type_df.loc[root_id, "pt_position"]
    skel = pcg_meshwork(
        root_id,
        client=client,
        root_point=nuc_loc,
        root_point_resolution=[1, 1, 1],
        collapse_soma=True,
        synapses=True,
    )
    add_is_axon_annotation(skel, "pre_syn", "post_syn")

    compartment_labels = np.empty(len(skel.skeleton.vertices), dtype="object")

    axon_mask = skel.mesh_property_to_skeleton(
        skel.anno.is_axon.mesh_mask, aggfunc="median"
    ).astype(bool)

    compartment_labels[axon_mask] = "axon"
    compartment_labels[~axon_mask] = "dendrite"
    compartment_labels[skel.skeleton.root] = "soma"

    lvl2_ids = skel.anno["lvl2_ids"]["lvl2_id"]

    mesh_to_skel = skel.skeleton.mesh_to_skel_map_base

    lvl2_to_skel = dict(zip(lvl2_ids, mesh_to_skel.tolist()))

    skeleton = skel.skeleton
    branch_points = skel.skeleton.branch_points
    branch_order = np.zeros(len(skeleton.vertices), dtype=int)
    include_non_axon = set()
    for i in range(len(skeleton.vertices)):
        path = skeleton.path_to_root(i)
        # added this bit in to also keep parts of putative dendrite if they get passed over
        # on the way back to the soma
        if compartment_labels[i] == "axon":
            path_labels = compartment_labels[path]
            include_non_axon.update(path[path_labels != "axon"].tolist())
        branch_order[i] = np.isin(branch_points, path).sum().item() - 1
    include_non_axon = np.array(list(include_non_axon))

    level2_data = pd.DataFrame(index=lvl2_ids)
    level2_data["skeleton_index"] = [lvl2_to_skel[i] for i in lvl2_ids]
    level2_data["compartment"] = compartment_labels[
        level2_data["skeleton_index"]
    ].tolist()
    level2_data["branch_order"] = branch_order[level2_data["skeleton_index"]].tolist()
    level2_data["include"] = level2_data["compartment"].isin(
        ["axon", "soma"]
    ) | level2_data["skeleton_index"].isin(include_non_axon)

    # level2_axon_ids_by_order = Parallel(n_jobs=1)(
    #     delayed(compute_cover_for_order)(order) for order in range(max_order)
    # )
    def compute_cover_for_order(order):
        level2_ids = level2_data.query("include and branch_order <= @order").index
        covering_nodes = client.chunkedgraph.get_minimal_covering_nodes(level2_ids)
        return covering_nodes

    level2_axon_ids_by_order = [
        compute_cover_for_order(order) for order in range(max_order)
    ]

    site_utils.set_default_config(target_site="spelunker")

    order_df = pd.DataFrame(
        {"order": range(max_order), "axon_ids": level2_axon_ids_by_order}
    )
    order_df = order_df.explode("axon_ids")

    seg_prop = SegmentProperties.from_dataframe(
        order_df, tag_value_cols="order", id_col="axon_ids"
    )
    prop_id = client.state.upload_property_json(seg_prop.to_dict())

    ngl_url = "https://spelunker.cave-explorer.org/"
    prop_url = client.state.build_neuroglancer_url(
        prop_id, ngl_url=ngl_url, format_properties=True
    )

    img, seg = from_client(client, use_skeleton_service=True)
    seg.color = "red"
    _, seg2 = from_client(client, segmentation_name="branches")

    seg.add_selection_map(fixed_ids=[root_id])

    seg2.add_segment_propeties(prop_url)
    seg2.add_selection_map(
        fixed_ids=order_df["axon_ids"].tolist(),
        fixed_id_colors=len(order_df) * ["#ff9333"],
    )

    statebuilder = StateBuilder([img, seg, seg2], client=client)
    state_dict = statebuilder.render_state([], client=client, return_as="dict")
    state_dict["layers"][2]["segments"].clear()
    state_dict["layers"][1]["objectAlpha"] = 0.1
    state_dict["layers"][1]["segmentColors"] = {
        f"{root_id}": "#ff9333",
    }

    url = StateBuilder(
        [],
        base_state=state_dict,
        view_kws={"position": nuc_loc / np.array([4, 4, 40]), "layout": "3d"},
    ).render_state([], client=client, return_as="short")

    return url


urls_by_root = Parallel(n_jobs=-1, verbose=10)(
    delayed(run_for_root)(root_id) for root_id in root_ids
)
# %%
url_df = pd.DataFrame({"root_id": root_ids, "url": urls_by_root})
url_df["nucleus_id"] = url_df["root_id"].map(type_df["target_id"])
url_df["cell_type"] = url_df["root_id"].map(type_df["cell_type"])
url_df.rename(columns={"root_id": "root_id_at_1300"}, inplace=True)
url_df = url_df.set_index("nucleus_id")
url_df.to_csv("agnes_ngl_urls.csv")
# %%
