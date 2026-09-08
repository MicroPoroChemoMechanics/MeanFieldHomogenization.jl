# =============================================================================
#  train_supershape.jl — the two surrogates of a superspherical **cavity**,
#  committed under `src/NeuralInclusions/models/`.
#
#  Maintenance script, run **by hand**. Unlike `train_models.jl`, whose teacher
#  is an analytic Hill tensor, this one solves a finite-element cell for every
#  sample. That is affordable only because the cell is an octant: one eighth of
#  the domain, about a second per sample at level 3 where the full cell took the
#  better part of a minute.
#
#      julia scripts/nn/train_supershape.jl                 # both
#      julia scripts/nn/train_supershape.jl conduction      # one only
#
#  ## Why the two physics have different inputs, and it is not arbitrary
#
#  A cavity has no material contrast, so the shape is nearly the whole input.
#  What saves the elastic model from being a one-input curve is that the
#  response of a cavity still depends on the reference **Poisson ratio** through
#  the Eshelby tensor — and `ν₀` is exactly what an iterative scheme moves at
#  every iteration, which is what makes a surrogate over it worth having. In
#  transport `𝑨_∇∇` is scale-free in `k₀`, so the shape alone determines it.
#
#  ## And why transport needs only one number
#
#  A second-order tensor invariant under the octahedral group is **isotropic**.
#  So a supersphere, which needs three constants in elasticity, needs one here:
#  hence `GradLocISO2` rather than the three-component cubic class.
# =============================================================================

import Pkg
Pkg.activate(@__DIR__; io = devnull)
Pkg.instantiate(; io = devnull)

using MeanFieldHomogenization
using TensND
using Printf
using LinearAlgebra

import Lux, Optimisers, Zygote           # loads MeanFieldHomogenizationLuxExt
import Ferrite, FerriteGmsh, Gmsh        # loads MeanFieldHomogenizationFerriteExt

const NI = MeanFieldHomogenization.NeuralInclusions
const OUT = NI.MODEL_DIR

const SECTIONS = isempty(ARGS) ? nothing : Set(ARGS)
want(name) = SECTIONS === nothing || name in SECTIONS

# ─── The teacher, and the budget it runs under ───────────────────────────────
#
# `max_dofs` and `min_free_gb` are the point, not decoration: this loop meshes
# and factorizes a hundred times, and an out-of-memory kill takes the session
# with it. The octant keeps every one of them small enough that the guard never
# has to fire — which is the right way round.

const LEVEL = 3
const CELL = FECellMeshOptions(;
    level = LEVEL, outer_level = 3, radius_ratio = 3.0, relax = 20,
    octant = true, max_dofs = 120_000, min_free_gb = 4.0,
)

# The lower bound is where the *teacher* stops being trustworthy, not where the
# physics stops being interesting. Below p ≈ 0.35 two things degrade together:
# the isotropy residual of `𝑨_∇∇` passes 1e-3 and stops improving with
# refinement, the field at the conical points being singular; and the boundary
# snapping backs off on a growing number of nodes for the same reason. Training
# there would fit mesh error. The surrogate's domain guard then refuses `p`
# below it rather than extrapolating silently.
const P_LO, P_HI = 0.35, 2.50
const NU_LO, NU_HI = 0.0, 0.45

pore_geometry(x) = FESupershapePore(Supersphere(1.0, x[1]); opts = CELL)

elastic_response(g, C₀) = strain_strain_loc(g, C₀, C₀)
transport_response(g, K₀) = gradient_gradient_loc(g, K₀, K₀)

function train_and_save(
        name, spec, box, geometry, response, teacher, n, nval, opts;
        notes = "", atol::Real
    )
    println("\n", "="^78)
    println("training `", name, "` — ", n, " + ", nval, " finite-element solves")
    println("="^78)
    flush(stdout)
    # `atol` is the class residual a label may carry. The default 1e-8 suits an
    # analytic teacher; this one solves a cell, so its labels are only in the
    # class to the accuracy of the mesh. Measured on the isotropy residual of
    # `𝑨_∇∇`: about 2e-6 for a convex shape at level 4, but ~1e-3 at p = 0.30 --
    # and there it does **not** improve with refinement (9.7e-4 at level 4
    # against 1.1e-3 at level 3), because what limits it is the singular field
    # at the conical points, not the mesh density. So the threshold is set by
    # the concave end, and 3e-3 is still three orders below what a wrong class
    # would leave.
    t = @elapsed train, val = NI.generate_dataset(
        geometry, response, spec, box, n; nvalidation = nval, atol
    )
    @printf "dataset: %.1f s for %d samples (%.2f s each)\n" t (n + nval) t / (n + nval)
    flush(stdout)
    s = NI.train_surrogate(spec, box, train, val; options = opts, teacher_name = teacher, notes)
    NI.report_surrogate(s, val; labels = NI.component_labels(NI.hill_class(spec)))
    @printf "headline (worst block relative error): %.3e\n" NI.worst_error(s.provenance)
    path = NI.save_surrogate(joinpath(OUT, name * ".json"), s)
    println("wrote ", path)
    flush(stdout)
    return s
end

# ─── 1. Transport: one input, one output ─────────────────────────────────────

want("conduction") && train_and_save(
    "supershape_pore_conduction",
    NI.DimensionlessHill(GradLocISO2()),
    NI.SampleBox([:p], [P_LO], [P_HI]),
    pore_geometry, transport_response,
    "gradient_gradient_loc(FESupershapePore, TensISO{2}) — octant cell, level $LEVEL",
    120, 40,
    NI.TrainingOptions(; hidden = [24, 24], epochs = 12_000, batchsize = 32),
    # Measured isotropy residual of `𝑨_∇∇`: ~2e-6 for a convex shape at level 4,
    # ~1e-3 at the concave end -- and there it does **not** improve with
    # refinement (9.7e-4 at level 4 against 1.1e-3 at level 3), the field at the
    # conical points being singular. Still two orders below what a wrong class
    # would leave.
    atol = 6.0e-3,
    notes = "𝑨_∇∇ of a superspherical cavity; isotropic because a 2nd-order " *
        "tensor invariant under the octahedral group is; scale-free in k₀",
)

# ─── 2. Elasticity: two inputs, three outputs ────────────────────────────────

want("elastic") && train_and_save(
    "supershape_pore_elastic",
    NI.DimensionlessHill(StrainLocCubic()),
    NI.SampleBox([:p, :nu0], [P_LO, NU_LO], [P_HI, NU_HI]),
    pore_geometry, elastic_response,
    "strain_strain_loc(FESupershapePore, TensISO{4}) — octant cell, level $LEVEL",
    480, 140,
    NI.TrainingOptions(; hidden = [48, 48], epochs = 20_000, batchsize = 64),
    # A fourth-order tensor departs from its class by more than a second-order
    # one does: the measured cubic residual is 1e-4 to 1e-3 for p ≥ 0.7 but
    # reaches 1.5e-2 at p = 0.35. That is real mesh error in the concave range,
    # and the surrogate inherits it -- which is why the reported accuracy below
    # is the honest figure and not a target to be tuned toward.
    atol = 2.5e-2,
    notes = "𝑨_εε of a superspherical cavity on (𝕁, 𝔼, 𝕋); dimensionless, so " *
        "the inputs are the shape exponent and the reference Poisson ratio",
)
