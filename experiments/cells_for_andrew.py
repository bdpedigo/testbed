# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = [
#    "nglui>=3.7.1",
#    "pcg-skel>=1.1.0",
#    "pyvista[all]>=0.44.1",
# ]
# ///

# %%
from caveclient import CAVEclient
from nglui.site_utils import set_default_config
from nglui.statebuilder import (
    AnnotationLayerConfig,
    PointMapper,
    StateBuilder,
    from_client,
)

client_1300 = CAVEclient("minnie65_phase3_v1", version=1300)

# %%
# IDs were at a previous materialization
root_ids_at_943 = [
    864691134989474682,
    864691135122575655,
    864691135375445234,
    864691135494056080,
    864691135615759977,
    864691135906091230,
    864691135924057859,
    864691136002229896,
    864691136238392892,
    864691136369600127,
    864691136595487394,
    864691136718105206,
]

client_generic = CAVEclient("minnie65_phase3_v1")
# %%
# mapping them to materialization 1300 where we have spine/shaft/soma predictions
new_roots_map = {}
for root_id in root_ids_at_943:
    new_roots = client_generic.chunkedgraph.get_latest_roots(
        root_id, timestamp=client_1300.timestamp
    )
    if len(new_roots) > 1:
        nucs = client_1300.materialize.views.nucleus_detection_lookup_v1(
            pt_root_id=new_roots
        ).query(desired_resolution=[1, 1, 1], split_positions=True)
        nuc_idx = nucs["pt_position_y"].idxmin()
        nuc = nucs.loc[[nuc_idx]]
        new_roots = [nuc["pt_root_id"]]
    new_roots_map[root_id] = new_roots[0].item()

root_ids_at_1300 = list(new_roots_map.values())

# %%
urls = []
for root_id in root_ids_at_1300:
    print(root_id)
    post_synapses = client_1300.materialize.tables.synapse_target_predictions_ssa(
        post_pt_root_id=root_id
    ).query(desired_resolution=[1, 1, 1], split_positions=True)

    set_default_config(target_site="spelunker", caveclient=client_1300)
    img, seg = from_client(client_1300)

    seg.add_selection_map(fixed_ids=root_id)

    syn_anno = AnnotationLayerConfig(
        name="post_synapses",
        mapping_rules=PointMapper(
            point_column="ctr_pt_position",
            description_column="id",
            tag_column="tag",
            split_positions=True,
        ),
        tags=["soma", "shaft", "spine"],
        data_resolution=[1, 1, 1],
    )

    sb = StateBuilder(layers=[img, seg, syn_anno])
    state_dict = sb.render_state(post_synapses, return_as="dict")

    shader = """
        void main() {
        int is_soma = int(prop_tag0());
        int is_shaft = int(prop_tag1());
        int is_spine = int(prop_tag2());
            
        if ((is_soma + is_shaft + is_spine) == 0) {
            setColor(vec3(0.0, 0.0, 0.0));
        } else if ((is_soma + is_shaft + is_spine) > 1) {
            setColor(vec3(1.0, 1.0, 1.0));
        } else if (is_soma > 0) {
            setColor(vec3(0, 0.890196, 1.0));
        } else if (is_shaft > 0) {
            setColor(vec3(0.9372549, 0.90196078, 0.27058824));
        } else if (is_spine > 0) {
            setColor(vec3(0.91372549, 0.20784314, 0.63137255));
        }
        setPointMarkerSize(10.0);
        }
        """
    state_dict["layers"][-1]["shader"] = shader
    state_dict["layers"][1]["objectAlpha"] = 0.4
    state_dict["layout"] = "3d"

    annotation_id = state_dict["layers"][-1]["annotations"][0]["id"]
    state_dict["selection"] = {
        "layers": {
            "labeled_synapses": {
                "annotationId": annotation_id,
                "annotationSource": 0,
                "annotationSubsource": "default",
            }
        }
    }
    state_id = client_generic.state.upload_state_json(state_dict)
    base_url = "https://spelunker.cave-explorer.org/#!middleauth+https://global.daf-apis.com/nglstate/api/v1/"
    url = base_url + str(state_id)
    print(url)
    urls.append(url)

    print()

# %%

for url in urls:
    print(url)
    print()
# %%

