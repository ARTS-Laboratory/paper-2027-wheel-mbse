"""
=============================================================================
  COMPLIANT WHEEL — PARAMETRIC STRUCTURED MESH
=============================================================================
Node positions are a closed-form, differentiable function of the 14 genes; element
connectivity is a compile-time constant.  No external mesher, no topology change
between designs, and nothing data-dependent inside the traced function — which is what
makes `jax.grad` through the whole FEA cheap and stable.

    coords = spoke_block_coords(genes, cfg)        # traced, differentiable
    conn   = spoke_block_connectivity(cfg)         # numpy constant, built once

WHY A STRUCTURED BLOCK AND NOT A MESHER
---------------------------------------
The spoke is already an offset-band parameterisation — `x(s, eta) = C(s) + eta*(t(s)/2)*N(s)`
is exactly what `wheel_geometry.offset_band` computes, and it is exactly what
`thicken_3taper_curve` has always computed for the exported outline.  Adding a
transverse index turns the outline into a mesh.  So the meshed part and the exported
part are the same surface by construction rather than by two pipelines agreeing.

Running gmsh instead would put a combinatorial, non-differentiable step in the middle
of the gradient path, and remeshing between optimizer iterations would make the
objective discontinuous.

SCOPE: THE SPOKE BLOCK ONLY, FOR NOW
------------------------------------
This module currently meshes ONE spoke, hub-root cross-section to rim-tip
cross-section.  That is deliberate and it is what the single-spoke verification needs:
the beam model being checked against clamps at the root and guides the tip
(`generalized_spoke_mechanics`, wheel_fea.py:308), so the FE model must have exactly
the same boundary, with no junction, collar, or rim ring in the way.

The full 360 deg wheel needs three more block types (hub junction, hub collar, rim
band) and they are the hard part of this project's geometry, because the spoke arrives
almost tangentially:

    arrival angle at hub : 10.5 deg from tangent
    arrival angle at rim :  6.1 deg from tangent
    root cross-section   : straddles the hub circle — top flank 1.17 mm INSIDE
                           r=12.7, bottom flank 1.17 mm outside

That straddle is why `wheel_step_export._embed` exists, and why its comment records
that a straight tangent extension aimed 3 mm inside the hub never gets there at all
(closest approach 11.26 mm).  Those blocks are built once there is a solver to validate
them against, not before.

ELEMENT ORDER
-------------
Nodes are generated on a single fine grid; Q4 and Q9 connectivity are both read off it.
For a tensor grid, "add midside and centre nodes" is just "double the node density",
so one grid serves both and the Q4 mesh is a strict subset of the Q9 nodes.  Q9 is the
production choice — a Q4 mesh shear-locks in bending, and a 2 mm flexure is exactly the
case where that silently reports a design as stiffer and safer than it is.
=============================================================================
"""

import numpy as np

import wheel_geometry as _geom

# Local frame: the spoke centerline runs from (0,0) to (span,0); the global frame shifts
# x by the hub radius and rotates into a sector.  Kept local here — the FEA applies the
# placement, so a single-spoke verification run never pays for it.


class MeshConfig:
    """Element counts and element order for one spoke block.

    `n_span`   elements along the centerline (hub -> rim)
    `n_thick`  elements through the thickness
    `order`    1 for Q4 bilinear, 2 for Q9 biquadratic

    The counts here are a starting bracket, NOT a converged choice — they are replaced
    by measured numbers from the mesh-refinement study.  `n_curve` is the number of
    centerline samples the geometry kernel evaluates before the block is resampled onto
    its grid; it only needs to be comfortably finer than `n_span`.
    """

    __slots__ = ("name", "n_span", "n_thick", "order", "n_curve")

    def __init__(self, name, n_span, n_thick, order=2, n_curve=600):
        if order not in (1, 2):
            raise ValueError(f"order must be 1 (Q4) or 2 (Q9), got {order}")
        if n_span < 1 or n_thick < 1:
            raise ValueError("n_span and n_thick must be >= 1")
        self.name = name
        self.n_span = n_span
        self.n_thick = n_thick
        self.order = order
        self.n_curve = n_curve

    # Node-grid dimensions.  Q9 needs a node at every element midside and centre, which
    # on a tensor grid is simply twice the element count per direction.
    @property
    def n_node_span(self):
        return self.order * self.n_span + 1

    @property
    def n_node_thick(self):
        return self.order * self.n_thick + 1

    @property
    def n_nodes(self):
        return self.n_node_span * self.n_node_thick

    @property
    def n_elements(self):
        return self.n_span * self.n_thick

    @property
    def nodes_per_element(self):
        return 4 if self.order == 1 else 9

    def __repr__(self):
        return (f"MeshConfig({self.name!r}, {self.n_span}x{self.n_thick} "
                f"{'Q4' if self.order == 1 else 'Q9'}, "
                f"{self.n_elements} elem, {self.n_nodes} nodes)")


# Standard configurations.  `smoke` is for unit tests and CI, `coarse` for CPU
# development and the optimizer trajectory, `medium`/`fine` for checkpoints and the
# refinement study.
CONFIGS = {
    "smoke":  MeshConfig("smoke",  24,  3, order=2, n_curve=600),
    "coarse": MeshConfig("coarse", 64,  6, order=2, n_curve=600),
    "medium": MeshConfig("medium", 128, 10, order=2, n_curve=1200),
    "fine":   MeshConfig("fine",   256, 16, order=2, n_curve=2400),
}


def get_config(cfg):
    """Accept a name or a MeshConfig, so callers can pass either."""
    if isinstance(cfg, MeshConfig):
        return cfg
    try:
        return CONFIGS[cfg]
    except KeyError:
        raise KeyError(f"unknown mesh config {cfg!r}; have {sorted(CONFIGS)}") from None


# ---------------------------------------------------------------------------
# NODE COORDINATES  (traced / differentiable)
# ---------------------------------------------------------------------------

def band_sampler(cx1, cy1, cx2, cy2, cx3, cy3, cx4, cy4,
                 t0, t1, t2, t3, n_curve, span_mm, xp=np):
    """THE evaluator for the spoke band: `sample(s, eta) -> [..., 2]`.

        x(s, eta) = C(s) + eta * (t(s)/2) * N_hat(s),   s and eta in arc-length and
                                                        [-1, +1] respectively

    Every block that touches the band goes through this one function — the spoke block
    itself and both junction patches — and that is not tidiness, it is what makes the
    seams EXACT.  Two constructions that are mathematically identical but differ in
    which quantity gets interpolated disagree at O(h_curve^2): measured, the junction
    and the spoke block once put their shared cross-section 3.9e-4 mm apart, which
    leaves the assembled mesh with a real kink at every one of the 24 junctions and
    makes a 1e-10 seam assertion impossible to state honestly.

    The centerline is laid out by ARC LENGTH, not by Bezier parameter: the parameter is
    not proportional to arc length on a curved centerline, so gridding on it would bunch
    elements where the curve happens to be parameterised densely rather than where the
    geometry needs resolution.

    Normals come from the ANALYTIC hodograph, not from differencing the sampled curve.
    `xp.gradient` is one-sided at the two ends, so the flank position there carries O(h)
    error and the GEOMETRY ITSELF MOVES as the mesh refines: measured, the top-flank tip
    walked 8.9 um between n_span 32 and 64, halving on every refinement.  A convergence
    study on that mesh measures the geometry settling down rather than the
    discretization.  The exact tangents are sampled on the dense uniform-parameter grid
    and interpolated onto the arc-length grid; interpolating the unnormalised derivative
    and normalising afterwards keeps the direction right even where the two grids
    disagree most.

    Returns `(sample, s_dense, curve)`.  `s` and `eta` may be any broadcastable shapes.
    """
    curve, ctrl = _geom.bezier_centerline(cx1, cy1, cx2, cy2, cx3, cy3, cx4, cy4,
                                          span_mm=span_mm, num_points=n_curve, xp=xp)
    s_dense = _geom.arc_fractions(curve, xp=xp, at="nodes")
    d_dense = _geom.bezier_tangent(ctrl, n_curve, xp=xp, normalize=False)

    def sample(s, eta):
        s = xp.asarray(s, dtype=float)
        eta = xp.asarray(eta, dtype=float)
        cx = xp.interp(s, s_dense, curve[:, 0])
        cy = xp.interp(s, s_dense, curve[:, 1])
        tx = xp.interp(s, s_dense, d_dense[:, 0])
        ty = xp.interp(s, s_dense, d_dense[:, 1])
        # Same +90 degree rotation and same 1e-12 guard as
        # `_geom.normals_from_tangents`, spelled out here so it works on any shape
        # rather than only on [N, 2].
        nx, ny = -ty, tx
        mag = xp.sqrt(nx * nx + ny * ny) + 1e-12
        half = _geom.thickness_at_arc_length(s, t0, t1, t2, t3, xp=xp) / 2.0
        return xp.stack([cx + eta * half * nx / mag,
                         cy + eta * half * ny / mag], axis=-1)

    return sample, s_dense, curve


def spoke_block_coords(cx1, cy1, cx2, cy2, cx3, cy3, cx4, cy4,
                       t0, t1, t2, t3, cfg, span_mm, xp=np, s_range=(0.0, 1.0)):
    """Node positions of one spoke block, shape [n_node_span, n_node_thick, 2].

    Smooth in every gene, so `jax.grad` flows straight through to the node coordinates
    and from there to element energies.  Thin wrapper over `band_sampler` — see there
    for why the arc-length layout and the analytic normals are both load-bearing.

    `s_range` trims the block to a sub-interval of arc length.  The full-wheel mesh
    needs it because the raw band's two ends lie INSIDE the hub disk and the rim band
    respectively (the root cross-section straddles r=12.7: top flank 11.53, bottom flank
    13.87), so that material is already owned by those rings and meshing both would
    double-count it.
    """
    sample, _, _ = band_sampler(cx1, cy1, cx2, cy2, cx3, cy3, cx4, cy4,
                                t0, t1, t2, t3, cfg.n_curve, span_mm, xp=xp)
    s_grid = xp.linspace(s_range[0], s_range[1], cfg.n_node_span)
    eta = xp.linspace(-1.0, 1.0, cfg.n_node_thick)
    return sample(s_grid[:, None], eta[None, :])


def spoke_block_coords_from_vector(vec, cfg, span_mm, xp=np, s_range=(0.0, 1.0)):
    """Same, taking the ordered 14-vector the GA and SGD work in.

    Only the first 12 genes reach the mesh — R_hub and R_rim are the junction fillet
    radii, and the full-wheel mesh does not build fillets (see `wheel_wheel.py` for the
    measurement that says the shipped part's fillets are worth 0.29 mm2 of 2521).
    """
    cfg = get_config(cfg)
    return spoke_block_coords(*[vec[i] for i in range(12)],
                              cfg=cfg, span_mm=span_mm, xp=xp, s_range=s_range)


# ---------------------------------------------------------------------------
# CONNECTIVITY  (compile-time constant)
# ---------------------------------------------------------------------------

def grid_connectivity(n_node_i, n_node_j, order, base=0):
    """Element-to-node map for ANY structured [n_node_i, n_node_j] node grid.

    Node ids are row-major over the grid, i.e. `nid(i, j) = base + i * n_node_j + j`.
    `base` lets a block's elements be emitted directly in global numbering, which is
    what the full-wheel assembly needs.

    Vertex ordering is COUNTER-CLOCKWISE in the (i, j) index space, the standard
    isoparametric convention.  Whether that is counter-clockwise in PHYSICAL space
    depends on the block's orientation, and is checked separately by the Jacobian sign —
    an element numbered the wrong way round has a negative Jacobian and would silently
    contribute negative stiffness.  The full-wheel builder therefore flips whole blocks
    rather than trusting each one to have been laid out the right way round.

    Q9 ordering follows the usual convention and MUST match `wheel_fem._NODE_IJ`:
    4 corners, then 4 midsides (bottom, right, top, left), then the centre.
    """
    p = order
    if (n_node_i - 1) % p or (n_node_j - 1) % p:
        raise ValueError(f"grid {n_node_i}x{n_node_j} is not a whole number of "
                         f"order-{p} elements")
    ni, nj = (n_node_i - 1) // p, (n_node_j - 1) // p

    def nid(i, j):
        return base + i * n_node_j + j

    elems = []
    for ei in range(ni):
        for ej in range(nj):
            i0, j0 = p * ei, p * ej
            if p == 1:
                elems.append([nid(i0, j0), nid(i0 + 1, j0),
                              nid(i0 + 1, j0 + 1), nid(i0, j0 + 1)])
            else:
                elems.append([
                    nid(i0, j0), nid(i0 + 2, j0), nid(i0 + 2, j0 + 2), nid(i0, j0 + 2),
                    nid(i0 + 1, j0), nid(i0 + 2, j0 + 1),
                    nid(i0 + 1, j0 + 2), nid(i0, j0 + 1),
                    nid(i0 + 1, j0 + 1),
                ])
    return np.asarray(elems, dtype=np.int32)


def spoke_block_connectivity(cfg):
    """The spoke block's element-to-node map.  Thin wrapper over `grid_connectivity`."""
    cfg = get_config(cfg)
    return grid_connectivity(cfg.n_node_span, cfg.n_node_thick, cfg.order)


def boundary_nodes(cfg):
    """The node id sets the FEA applies boundary conditions to.

    `root` and `tip` are the two end cross-sections — the surfaces the beam model
    clamps and guides — and `flank_top` / `flank_bot` are the free surfaces.
    """
    cfg = get_config(cfg)
    ns, nt = cfg.n_node_span, cfg.n_node_thick
    ids = np.arange(ns * nt).reshape(ns, nt)
    return {
        "root": ids[0, :].copy(),
        "tip": ids[-1, :].copy(),
        "flank_bot": ids[:, 0].copy(),
        "flank_top": ids[:, -1].copy(),
    }


def flatten(coords):
    """[n_span, n_thick, 2] grid -> [n_nodes, 2], matching `nid(i,j)` row-major."""
    return coords.reshape(-1, 2)


# ---------------------------------------------------------------------------
# QUALITY METRICS
# ---------------------------------------------------------------------------

def _corner_coords(nodes_xy, conn):
    """The 4 corner vertices of every element, [n_elem, 4, 2].

    Q9 stores its corners first, so the same slice serves both orders.
    """
    return nodes_xy[conn[:, :4]]


def scaled_jacobian(nodes_xy, conn, xp=np):
    """Minimum scaled Jacobian per element, in [-1, 1].

    At each of the 4 corners, the Jacobian of the bilinear map normalised by the
    lengths of the two edges meeting there — i.e. the sine of the corner angle.  The
    element's score is the worst corner.

    1.0 is a perfect rectangle.  Values near 0 are degenerate (a corner has collapsed
    to a sliver); NEGATIVE means the element is inverted and the FE formulation is
    meaningless there.

    Note this measures SHAPE, not aspect ratio: a long thin rectangle scores 1.0.  That
    is the right property for validity, and the reason aspect ratio is reported
    separately.

    `xp=jax.numpy` makes this traceable, which is what M8's mesh-validity barrier needs:
    the barrier has to be differentiable in the genes, and it reaches them through
    `wheel_wheel.mesh_coords`.  M2b's committed report and every caller of the numpy path
    are unaffected — `study_objective.py` gate 6 pins the two paths to 1e-12.  `conn` stays
    a numpy index array either way; it is topology, and topology is frozen.
    """
    P = _corner_coords(nodes_xy, conn)
    out = []
    for k in range(4):
        a = P[:, (k + 1) % 4] - P[:, k]        # outgoing edge
        b = P[:, (k - 1) % 4] - P[:, k]        # incoming edge
        cross = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
        denom = xp.linalg.norm(a, axis=1) * xp.linalg.norm(b, axis=1)
        out.append(cross / xp.where(denom > 0, denom, xp.inf))
    # Corners are traversed consistently, so a valid element has all four the same
    # sign; take the signed extreme nearest zero.
    stacked = xp.stack(out, axis=1)
    return xp.where(stacked.min(axis=1) < 0,
                    stacked.min(axis=1),
                    xp.abs(stacked).min(axis=1))


def aspect_ratio(nodes_xy, conn):
    """Longest over shortest edge of each element."""
    P = _corner_coords(nodes_xy, conn)
    edges = np.linalg.norm(P[:, [1, 2, 3, 0]] - P, axis=2)
    return edges.max(axis=1) / np.maximum(edges.min(axis=1), 1e-30)


def element_areas(nodes_xy, conn):
    """Signed area of each element by the shoelace formula on its 4 corners.

    Signed on purpose: a negative area is an inverted element, and summing absolute
    values would hide exactly the defect this is meant to expose.
    """
    P = _corner_coords(nodes_xy, conn)
    x, y = P[:, :, 0], P[:, :, 1]
    return 0.5 * (x * np.roll(y, -1, axis=1) - np.roll(x, -1, axis=1) * y).sum(axis=1)


def quality_report(nodes_xy, conn):
    """Everything the design-space sweep records for one genome."""
    sj = scaled_jacobian(nodes_xy, conn)
    ar = aspect_ratio(nodes_xy, conn)
    area = element_areas(nodes_xy, conn)
    return {
        "min_scaled_jacobian": float(sj.min()),
        "mean_scaled_jacobian": float(sj.mean()),
        "n_inverted": int((area <= 0).sum()),
        "max_aspect_ratio": float(ar.max()),
        "total_area_mm2": float(area.sum()),
        "min_element_area_mm2": float(np.abs(area).min()),
        "n_elements": int(len(conn)),
    }
