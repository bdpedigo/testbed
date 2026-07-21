# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "numpy>=1.26",
#     "pyvista>=0.44",
#     "gpytoolbox>=0.3.3",
# ]
# ///
"""
Nested-sphere sanity test for the fast winding number (FWN).

WHAT THIS DEMONSTRATES
----------------------
The generalized winding number at a point p equals the number of properly
CLOSED, CONSISTENTLY-ORIENTED shells that enclose p. A single closed mesh
therefore only ever produces w in {0, 1}. To see 2, 3, 4 you need genuinely
nested closed shells -- which is exactly what concentric spheres give you.
This is the direct counterpart to the real-mesh observation "I only ever see
values near 0 or 1": that is the correct signature of ONE closed surface.

ASSUMPTIONS (each is enforced by an assertion below)
----------------------------------------------------
A1. pyvista's Sphere is a closed, outward-oriented triangle mesh, so FWN
    reads +1 (not -1) inside a single sphere. -> test_orientation
A2. A single closed sphere yields a field that only ever takes the values
    ~0 (outside) and ~1 (inside) -- never 2. -> test_single_sphere_is_binary
A3. N concentric outward spheres of decreasing radius give a field that steps
    0 -> 1 -> 2 -> ... -> N as you move inward across each shell. This is the
    whole point: nesting is what produces integers above 1. -> test_nested_field
A4. Under the per-facet +/-eps straddle test, ONLY the outermost sphere's
    facets separate exterior (w<0.5) from interior (w>=0.5); every inner
    sphere's facets are buried (both samples >=0.5) and get dropped. This is
    "extract only the exterior surface" working on a clean nested case.
    -> test_facet_straddle_keeps_only_outer

The straddle classification uses min/max of the two samples, so it does NOT
depend on which way the facet normal happens to point -- only that +/-eps
lands on opposite sides of the surface.
"""

import numpy as np
import pyvista as pv
from gpytoolbox import fast_winding_number

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
RADII = [40.0, 30.0, 20.0, 10.0]        # strictly decreasing, all centered at origin
THETA_RES, PHI_RES = 60, 60             # fine tessellation -> small facets
EPS = 2.0                                # offset distance for the +/-eps facet test
TOL = 0.1                                # tolerance for "is w near this integer"

# EPS must be small enough not to jump between shells (min gap here is 10, so
# an offset of 2 stays well within a shell) and large enough to clear the
# near-field of the facets (edge length ~ 2*pi*r/res; for r=10 that is ~1).


# ----------------------------------------------------------------------------
# Mesh construction
# ----------------------------------------------------------------------------
def make_sphere(radius):
    """One closed, triangulated, outward-oriented sphere as (V, F)."""
    s = pv.Sphere(
        radius=radius, theta_resolution=THETA_RES, phi_resolution=PHI_RES
    ).triangulate()
    V = np.asarray(s.points, dtype=np.float64)
    F = s.faces.reshape(-1, 4)[:, 1:].astype(np.int64)  # drop the leading "3"
    return V, F


def make_nested(radii):
    """Concatenate several spheres into one triangle soup.

    Returns V, F, and `ranges`: a list of (start, end) face-index intervals,
    one per sphere in the order given (so ranges[0] is the OUTERMOST sphere).
    """
    V_parts, F_parts, ranges = [], [], []
    v_offset = 0
    f_offset = 0
    for r in radii:
        v, f = make_sphere(r)
        F_parts.append(f + v_offset)
        ranges.append((f_offset, f_offset + len(f)))
        V_parts.append(v)
        v_offset += len(v)
        f_offset += len(f)
    return np.concatenate(V_parts), np.concatenate(F_parts), ranges


# ----------------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------------
def test_orientation():
    """A1: FWN at the center of a single sphere is +1 (outward orientation)."""
    V, F = make_sphere(RADII[0])
    w_center = fast_winding_number(np.array([[0.0, 0.0, 0.0]]), V, F)[0]
    print(f"[A1] w at center of single sphere = {w_center:+.4f} (expect +1)")
    assert abs(w_center - 1.0) < TOL, (
        f"expected +1 inside an outward sphere, got {w_center:.4f}. "
        "If this is ~-1, pyvista handed back inward-oriented faces."
    )


def test_single_sphere_is_binary():
    """A2: a single closed mesh only ever produces w ~ 0 or ~ 1, never 2."""
    V, F = make_sphere(RADII[0])
    R = RADII[0]
    rng = np.random.default_rng(0)
    dirs = rng.normal(size=(2000, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    # clearly-inside points (r = 0.5R -> w~1) and clearly-outside (r = 1.5R -> w~0),
    # both kept well away from the surface band to avoid on-surface ambiguity.
    inside_pts = dirs[:1000] * (0.5 * R)
    outside_pts = dirs[1000:] * (1.5 * R)
    w = fast_winding_number(np.concatenate([inside_pts, outside_pts]), V, F)
    w_in, w_out = w[:1000], w[1000:]
    print(f"[A2] inside  w range = [{w_in.min():.3f}, {w_in.max():.3f}] (expect ~1)")
    print(f"[A2] outside w range = [{w_out.min():.3f}, {w_out.max():.3f}] (expect ~0)")
    print(f"[A2] global max |w|  = {np.abs(w).max():.3f} (expect < 1.1, i.e. never 2)")
    assert np.all(np.abs(w_in - 1.0) < TOL)
    assert np.all(np.abs(w_out - 0.0) < TOL)
    # the teaching assertion: a single shell can NEVER reach 2.
    assert np.abs(w).max() < 1.0 + TOL


def test_nested_field():
    """A3: N nested shells step the field 0,1,2,...,N from outside to center."""
    V, F, _ = make_nested(RADII)
    N = len(RADII)
    # query midway through each shell (and the exterior, and the center)
    query_radii = [RADII[0] + 5.0]                       # outside all -> 0
    query_radii += [(RADII[i] + RADII[i + 1]) / 2 for i in range(N - 1)]  # 1..N-1
    query_radii += [0.0]                                  # center -> N
    q = np.array([[r, 0.0, 0.0] for r in query_radii])
    w = fast_winding_number(q, V, F)
    expected = np.arange(N + 1)                           # [0, 1, 2, ..., N]
    for r, wi, ei in zip(query_radii, w, expected):
        print(f"[A3] r={r:6.1f}  w={wi:+.4f}  expect {ei}")
    assert np.allclose(w, expected, atol=TOL), (
        f"nested field wrong: got {np.round(w, 3)}, expected {expected}"
    )
    # explicit: the deepest shell reaches exactly N -- higher integers DO appear.
    assert abs(w[-1] - N) < TOL


def test_facet_straddle_keeps_only_outer():
    """A4: per-facet +/-eps straddle keeps ONLY the outer sphere; inner buried."""
    V, F, ranges = make_nested(RADII)
    tri = V[F]                                            # (n_faces, 3, 3)
    centroids = tri.mean(axis=1)
    # face normals via edge cross product, then NORMALIZE (constant offset depth).
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    L = np.linalg.norm(n, axis=1, keepdims=True)
    assert np.all(L > 0), "degenerate face slipped through"
    n /= L

    p_plus = centroids + n * EPS
    p_minus = centroids - n * EPS
    w_plus = fast_winding_number(p_plus, V, F)
    w_minus = fast_winding_number(p_minus, V, F)

    lo = np.minimum(w_plus, w_minus)
    hi = np.maximum(w_plus, w_minus)
    boundary = (lo < 0.5) & (hi >= 0.5)                  # exterior boundary -> KEEP
    buried = lo >= 0.5                                    # both interior -> DROP

    outer_start, outer_end = ranges[0]
    outer = np.zeros(len(F), dtype=bool)
    outer[outer_start:outer_end] = True

    kept_frac_outer = boundary[outer].mean()
    kept_frac_inner = boundary[~outer].mean()
    buried_frac_inner = buried[~outer].mean()
    print(f"[A4] outer-sphere facets classified KEEP:   {kept_frac_outer:6.2%} (expect ~100%)")
    print(f"[A4] inner-sphere facets classified KEEP:   {kept_frac_inner:6.2%} (expect ~0%)")
    print(f"[A4] inner-sphere facets classified BURIED: {buried_frac_inner:6.2%} (expect ~100%)")

    # Only the outermost shell should carry the 0->1 transition.
    assert boundary[outer].all(), "some outer-sphere facets were NOT kept"
    assert not boundary[~outer].any(), "some inner-sphere facets were wrongly kept"
    assert buried[~outer].all(), "some inner-sphere facets were not marked buried"


if __name__ == "__main__":
    for fn in (
        test_orientation,
        test_single_sphere_is_binary,
        test_nested_field,
        test_facet_straddle_keeps_only_outer,
    ):
        print(f"\n=== {fn.__name__} ===")
        fn()
        print(f"    PASSED")
    print("\nAll assertions passed.")