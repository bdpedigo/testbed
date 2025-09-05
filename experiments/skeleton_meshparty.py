# %%
from caveclient import CAVEclient
from meshparty.skeleton import Skeleton

client = CAVEclient("v1dd")
root_id = 864691132661526697
skeleton_dict = client.skeleton.get_skeleton(root_id)
# %%

skel = Skeleton(
    skeleton_dict["vertices"],
    skeleton_dict["edges"],
    root=skeleton_dict["root"],
    radius=skeleton_dict["radius"],
)

# %%
print("branch points", skel.n_branch_points)
print("end points (tips)", skel.n_end_points)
print("path length (nm)", skel.path_length())
print("mean radius (nm)", skel.radius.mean())
print("max radius (nm)", skel.radius.max())
print("min radius (nm)", skel.radius.min())
