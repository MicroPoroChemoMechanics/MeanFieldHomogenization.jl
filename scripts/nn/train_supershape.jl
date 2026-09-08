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

import SHA                              # keys the dataset cache
import Serialization                    # and stores it

import Lux, Optimisers, Zygote           # loads MeanFieldHomogenizationLuxExt
import Ferrite, FerriteGmsh, Gmsh        # loads MeanFieldHomogenizationFerriteExt

const NI = MeanFieldHomogenization.NeuralInclusions
const OUT = NI.MODEL_DIR

# Where the finite-element datasets are kept between runs. A scratch directory,
# so nothing measurement-shaped is left inside the repository; `MFH_NN_CACHE`
# overrides it and `MFH_NN_FORCE=1` regenerates regardless.
const CACHE_DIR = get(
    ENV, "MFH_NN_CACHE", joinpath(tempdir(), "mfh_nn_datasets"),
)
const FORCE = get(ENV, "MFH_NN_FORCE", "0") == "1"

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

# ─── The axisymmetric family ─────────────────────────────────────────────────
#
# A two-dimensional solve per Fourier mode, so `radius_ratio = 6` is affordable
# and worth it: what the corrected condition leaves is truncation, and the sweep
# in `89_fe_concave_pores.jl` measures it at 1.1e-3 for `R/a = 4` against
# 1.5e-4 for 6. The teacher's own accuracy is the ceiling on the surrogate's.
#
# `nradial` is modest because the profile grading does the work. Uniform
# elements left `R₃₃` — the mode-0 answer, which loads across the equatorial
# wedge of a concave shape — not converging at all: increments of 4.4e-3 then
# 3.7e-3 refining 12 → 20 → 32. Graded, the same sequence gives 3.8e-4 then
# 1.9e-4, and `nradial = 14` already sits 4e-4 from the converged value. A
# surrogate cannot be better than its teacher, and the first version of this
# model was not: 1.37e-2 worst case, all of it inherited.
const AXI = FEAxiMeshOptions(; nradial = 14, radius_ratio = 6.0, tip_refine = 16.0)

# `a` is not a feature: the localization of a cavity is scale-free, so only the
# aspect ratio `c/a` and the exponent matter. The shape is built at `a = 1` and
# the features are `(c, p)`.
# The shape features arrive as logarithms, which is what makes the sampling
# geometric; `a = 1` because a cavity's localization is scale-free.
axi_geometry(x) = FEAxiSupershapePore(
    Superspheroid(1.0, exp(x[1]), exp(x[2])); opts = AXI
)

const C_LO, C_HI = 0.5, 2.0
# The axisymmetric box stops at `p = 1.5`: past it `R₃₃` moves by a couple of
# per cent, so sampling further spends the budget where nothing happens. It
# reaches down to 0.25, which is inside the range Sevostianov et al. study —
# and the teacher is measured to converge there, increments of 8e-4 at
# `p = 0.20` with a minimum mesh angle of 35°, so there is no floor to respect.
const AXI_P_LO, AXI_P_HI = 0.25, 1.5
const AXI_LP_LO, AXI_LP_HI = log(AXI_P_LO), log(AXI_P_HI)
const AXI_LC_LO, AXI_LC_HI = log(C_LO), log(C_HI)

elastic_response(g, C₀) = strain_strain_loc(g, C₀, C₀)
transport_response(g, K₀) = gradient_gradient_loc(g, K₀, K₀)

function train_and_save(
        name, spec, box, geometry, response, teacher, n, nval, opts;
        notes = "", atol::Real, reference = nothing
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
    # The dataset is finite-element solves — 56 minutes for the axisymmetric
    # elastic model — so it is cached beside the run and reused. The key is
    # everything the labels depend on; change the box, the sample count, the
    # teacher string or the tolerance and a stale cache is not read. `CACHE_DIR`
    # is a scratch directory, never the repository: these files are measurement
    # residue, not artifacts.
    key = bytes2hex(
        SHA.sha1(
            string(
                name, "|", n, "|", nval, "|", atol, "|", teacher, "|",
                box.names, "|", box.lo, "|", box.hi, "|",
                reference === nothing ? "none" : "ref",
            ),
        ),
    )[1:12]
    cache = joinpath(CACHE_DIR, "$(name)_$(key).jls")
    local train, val
    if isfile(cache) && !FORCE
        t = @elapsed ((train, val) = Serialization.deserialize(cache))
        @printf "dataset: reused cache (%s) in %.1f s\n" basename(cache) t
    else
        t = @elapsed ((train, val) = NI.generate_dataset(
            geometry, response, spec, box, n; nvalidation = nval, atol, reference
        ))
        @printf "dataset: %.1f s for %d samples (%.2f s each)\n" t (n + nval) t / (n + nval)
        mkpath(CACHE_DIR)
        Serialization.serialize(cache, (train, val))
        println("cached to ", cache)
    end
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

# ─── 3. The axisymmetric cavity, transport ───────────────────────────────────
#
#  Two components, `R₁₁` and `R₃₃`, because a body of revolution distinguishes
#  its axis — where the supersphere's transport localization is isotropic and
#  needs one.

want("axi_conduction") && train_and_save(
    "axi_supershape_pore_conduction",
    NI.DimensionlessHill(GradLocTI2()),
    NI.SampleBox([:log_aspect, :log_p], [AXI_LC_LO, AXI_LP_LO], [AXI_LC_HI, AXI_LP_HI]),
    axi_geometry, transport_response,
    "gradient_gradient_loc(FEAxiSupershapePore, TensISO{2}) — Fourier axi, " *
        "nradial = $(AXI.nradial), R/a = $(AXI.radius_ratio)",
    # `R₃₃` is the demanding component: it runs over a factor of nine across
    # this box and steepens in the flat-and-concave corner, where a thin oblate
    # cavity with conical points is nearly a crack pierced by a needle. `R₁₁`
    # barely moves. So the sample count is set by the harder of the two — and so
    # is the sampling law, which is why the features are `:log_aspect` and
    # `:log_p` rather than `:c` and `:p`. A `SampleBox` is linear, so a uniform
    # box in `p` spends most of its budget where nothing happens: measured,
    # 7.1e-3 worst case uniform against 3.7e-3 geometric, same everything else.
    500, 150,
    # `log_threshold = 5`: `R₃₃` runs from 1.66 at `p = 0.6` to 15.0 at
    # `p = 0.20` — a factor of nine, and a near-divergence as the body tends to
    # a crack pierced by a needle. The default of 30 declines a log fit there,
    # and with identity scaling the loss is carried by the large values while
    # the relative error at the other end is whatever remains. Measured: 1.4e-2
    # worst case without it.
    NI.TrainingOptions(;
        hidden = [48, 48], epochs = 25_000, batchsize = 64, log_threshold = 5.0
    ),
    # The class residual of the axisymmetric teacher is far smaller than the
    # three-dimensional one's: transverse isotropy is *structural* here — the
    # Fourier modes decode straight onto the Kelvin basis — so the projection
    # residual is round-off, and what `atol` guards against is a wrong axis.
    atol = 1.0e-6,
    notes = "𝑨_∇∇ of a superspheroidal cavity, TI about the revolution axis; " *
        "scale-free in k₀, so the features are the aspect ratio and the exponent",
)

# ─── 4. The axisymmetric cavity, elasticity ──────────────────────────────────
#
#  Six components, not five: a localization tensor has no major symmetry, and
#  `StrainLocTI` carries the six Walpole coefficients for exactly that reason.
#
#  This is the class whose reference medium the package deliberately refuses to
#  guess, because the same class also serves *heterogeneous* morphologies that
#  carry their constituents inside themselves. A cavity does not, and is of
#  degree 0 in the reference — so the caller states it, which is what the
#  `reference` keyword is for.

want("axi_elastic") && train_and_save(
    "axi_supershape_pore_elastic",
    NI.DimensionlessHill(StrainLocTI()),
    NI.SampleBox(
        [:log_aspect, :log_p, :nu0],
        [AXI_LC_LO, AXI_LP_LO, NU_LO], [AXI_LC_HI, AXI_LP_HI, NU_HI],
    ),
    axi_geometry, elastic_response,
    "strain_strain_loc(FEAxiSupershapePore, TensISO{4}) — Fourier axi, " *
        "nradial = $(AXI.nradial), R/a = $(AXI.radius_ratio)",
    # A three-dimensional box, and `ℓ₁` runs from 1.4 to 63 across it — a factor
    # of 46, near-diverging in the flat-and-concave corner where the cavity is
    # nearly a crack pierced by a needle. 600 samples is 8.4 points per
    # dimension, and it showed: 4.0e-2 worst block error where the transport
    # model on a two-dimensional box reaches 3.7e-3. The teacher is not the
    # limit — its own mesh convergence is 1.5e-3 between `nradial` 8 and 20,
    # forty times below the fit — so the samples are.
    1400, 400,
    NI.TrainingOptions(;
        hidden = [64, 64], epochs = 20_000, batchsize = 64, log_threshold = 5.0
    ),
    atol = 1.0e-6,
    reference = (box, x) -> NI._iso_ref(x[NI.feature_index(box, :nu0)]),
    notes = "𝔸_εε of a superspheroidal cavity on the six Walpole coefficients " *
        "about the revolution axis; the reference is stated rather than guessed",
)
