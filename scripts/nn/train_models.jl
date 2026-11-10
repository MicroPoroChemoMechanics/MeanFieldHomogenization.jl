# =============================================================================
#  train_models.jl — regenerate the surrogates committed under
#  `src/NeuralInclusions/models/`.
#
#  Maintenance script, run **by hand**, never at test or documentation time: the
#  tests and script 84 load the committed JSON, which is what keeps the suite
#  deterministic and the doc build free of Lux.
#
#      julia scripts/nn/train_models.jl                # all three
#      julia scripts/nn/train_models.jl conduction     # one model only
#
#  Model names: elastic, conduction, triaxial, affine.
#
#  Teachers are the *analytic* Hill tensors, so the labels are exact and the
#  only error in the chain is the fit itself.  This is the purpose of the
#  ellipsoid pilot: it is the one morphology where a surrogate can be held to a
#  closed form.
# =============================================================================

import Pkg
Pkg.activate(@__DIR__; io = devnull)
Pkg.instantiate(; io = devnull)

using MeanFieldHomogenization
using TensND
using Printf
using Plots

gr()
default(;
    fontfamily = "sans-serif", framestyle = :box, grid = true, legendfontsize = 8,
    left_margin = 5Plots.mm, bottom_margin = 5Plots.mm,
)

import Lux, Optimisers, Zygote           # loads MeanFieldHomogenizationLuxExt

const NI = MeanFieldHomogenization.NeuralInclusions
const OUT = NI.MODEL_DIR

const SECTIONS = isempty(ARGS) ? nothing : Set(ARGS)
want(name) = SECTIONS === nothing || name in SECTIONS

mkpath(OUT)

# The sampled range of aspect ratios: two decades either side of the sphere.
# Deliberately short of the crack and needle limits, where several Walpole
# components diverge — a surrogate that claimed them would be extrapolating and
# the domain guard is there to say so.
const RMAX = 20.0
const LOGR = log(RMAX)

# Poisson ratio of the reference medium.  The upper bound stops short of ½:
# the bulk modulus diverges there and with it one of the two affine
# coefficients.
const NU_LO, NU_HI = 0.0, 0.49

# A triaxial ellipsoid must stay *genuinely* triaxial: on the faces of the box
# two semi-axes coincide, the analytic teacher returns a `TensTI` instead of a
# `TensOrtho`, and the label extraction refuses it (rightly — the component
# count changes).  A 5 % minimum separation keeps the whole box orthotropic.
const LOG_SEP = log(1.05)

report = IOBuffer()

function record(name, s, val, labels)
    println(report, "## `$name`\n")
    println(report, "- features: `", join(s.features, "`, `"), "`")
    println(report, "- network: ", s.net)
    println(report, "- samples: $(s.provenance.nsamples) train, $(s.provenance.nvalidation) held out")
    @printf(
        report,
        "- worst held-out error, relative to the tensor magnitude: **%.3e**\n",
        NI.worst_error(s.provenance)
    )
    println(
        report,
        "\nPer-component diagnostics (a structurally vanishing component such as ",
        "`ℓ₃(𝕎)` is scored against the whole block, not against itself):\n"
    )
    println(report, "| component | max rel. err | rms rel. err |")
    println(report, "| --- | ---: | ---: |")
    for (i, l) in enumerate(labels)
        @printf(report, "| `%s` | %.3e | %.3e |\n", l, val.max_rel_error[i], val.rms_rel_error[i])
    end
    println(report)
    return nothing
end

# The learning curves, kept so the documentation can show one without fitting
# anything at build time.
const CURVES = Dict{String, Vector{Any}}()

function train_and_save(
        name, spec, box, geometry, response, teacher_name, n, nval, opts; notes = ""
    )
    println("\n", "="^78)
    println("training `$name`")
    println("="^78)
    train, val = NI.generate_dataset(geometry, response, spec, box, n; nvalidation = nval)
    history = Any[]
    s = NI.train_surrogate(
        spec, box, train, val;
        options = opts, teacher_name, notes, history
    )
    CURVES[name] = history
    v = NI.report_surrogate(s, val; labels = NI.component_labels(spec))
    path = NI.save_surrogate(joinpath(OUT, name * ".json"), s)
    println("wrote ", path)
    record(name, s, v, NI.component_labels(spec))
    return s
end

# ─── 1. Spheroid, elasticity — the reference model ───────────────────────────
#
#  Two features, five outputs.  `ω = distinct/equal`, so the box covers prolate
#  (positive logarithm) and oblate (negative) in one sweep, with the sphere at
#  the origin.

# The geometry callback: shape features → an `Ellipsoid`.  `Ellipsoid` sorts its
# semi-axes descending and permutes the basis, and `NeuralHillInclusion` applies
# the very same canonicalization, so the frame the labels are read in is the
# frame the surrogate writes into.
spheroid_geometry(x) = _spheroid(exp(x[1]))

function _spheroid(ω)
    # ω > 1: prolate, the distinct axis is the long one.  ω < 1: oblate.
    return ω ≥ 1 ? Ellipsoid(ω, 1.0, 1.0) : Ellipsoid(1.0, 1.0, ω)
end

# The response callback: what is to be learned.  For the pilot the analytic Hill
# tensor; for a complex morphology this is the only line that changes.
hill_response(geom, P₀) = hill_tensor(geom, P₀)

want("elastic") && train_and_save(
    "spheroid_hill_iso_elastic",
    NI.DimensionlessHill(NI.HillTI()),
    NI.SampleBox([:log_aspect, :nu0], [-LOGR, NU_LO], [LOGR, NU_HI]),
    spheroid_geometry, hill_response,
    "hill_tensor(Ellipsoid spheroid, TensISO{4}) — analytic",
    6000, 1500,
    NI.TrainingOptions(; hidden = [48, 48], epochs = 6000, batchsize = 256),
    notes = "2μ₀ℙ of a spheroid in an isotropic matrix; ω = distinct/equal " *
        "semi-axis, ν₀ the matrix Poisson ratio",
)

# ─── 2. Spheroid, transport — one feature, and materially exact ──────────────
#
#  The 2nd-order Hill tensor is exactly `𝕍ᴬ/k₀`, so the conductivity is divided
#  out and the surrogate is a function of the shape alone.  One input, two
#  outputs: the easiest of the three, and the sharpest check that the decode is
#  right.

want("conduction") && train_and_save(
    "spheroid_hill_iso_conduction",
    NI.DimensionlessHill(NI.HillTI2()),
    NI.SampleBox([:log_aspect], [-LOGR], [LOGR]),
    spheroid_geometry, hill_response,
    "hill_tensor(Ellipsoid spheroid, TensISO{2}) — analytic",
    3000, 800,
    NI.TrainingOptions(; hidden = [32, 32], epochs = 6000, batchsize = 128),
    notes = "k₀ℙ_K of a spheroid; no material feature — the 1/k₀ scaling is exact",
)

# ─── 3. Triaxial ellipsoid, elasticity — nine components ─────────────────────

function triaxial_geometry(x)
    r2 = exp(x[1])            # a₂/a₁
    r3 = r2 * exp(x[2])       # a₃/a₁ = (a₂/a₁)·(a₃/a₂)
    return Ellipsoid(1.0, r2, r3)
end

want("triaxial") && train_and_save(
    "triaxial_hill_iso_elastic",
    NI.DimensionlessHill(NI.HillOrtho()),
    NI.SampleBox(
        [:log_r2, :log_r32, :nu0],
        [-LOGR, -LOGR, NU_LO],
        [-LOG_SEP, -LOG_SEP, NU_HI],
    ),
    triaxial_geometry, hill_response,
    "hill_tensor(Ellipsoid triaxial, TensISO{4}) — analytic",
    12000, 3000,
    NI.TrainingOptions(; hidden = [64, 64], epochs = 8000, batchsize = 256),
    notes = "2μ₀ℙ of a triaxial ellipsoid; (a₂/a₁, a₃/a₂) with axes sorted " *
        "descending, so the admissible set is a box",
)

# ─── 4. Spheroid, elasticity, affine factorization ───────────────────────────
#
#  The same physics as model 1, but ν₀ is removed from the inputs and the
#  network predicts the two shape tensors 𝕌ᴬ and 𝕍ᴬ instead — ten outputs from
#  one input.  The material dependence then costs nothing and is exact.
#  Trained here to be compared against model 1 in script 84.

want("affine") && train_and_save(
    "spheroid_hill_iso_affine",
    NI.AffineHill(NI.HillTI()),
    NI.SampleBox([:log_aspect], [-LOGR], [LOGR]),
    spheroid_geometry, hill_response,
    "hill_tensor(Ellipsoid spheroid, TensISO{4}) — analytic, two reference media",
    6000, 1500,
    NI.TrainingOptions(; hidden = [48, 48], epochs = 6000, batchsize = 256),
    notes = "shape tensors 𝕌ᴬ and 𝕍ᴬ of a spheroid; ν₀ is not an input — the " *
        "affine decomposition ℙ = d·𝕌ᴬ + (1/μ₀)·𝕍ᴬ is applied exactly at decode",
)

# ─── The learning curve, for the tutorial page ───────────────────────────────
#
#  Committed as a PNG on purpose: the tutorial shows how training goes without
#  running a single epoch at documentation-build time.

if length(CURVES) ≥ 2
    ASSET = normpath(joinpath(@__DIR__, "..", "..", "docs", "src", "assets", "nn"))
    mkpath(ASSET)
    plt = plot(;
        xlabel = "epoch", ylabel = "mean squared error (standardized targets)",
        yscale = :log10, legend = :topright, size = (760, 460),
        title = "Learning curves of two shipped surrogates\n" *
            "spheroid, isotropic matrix; Adam, batch 256, 48+48 hidden, tanh",
        titlefontsize = 10,
    )
    for (name, col) in (
            ("spheroid_hill_iso_elastic", :steelblue),
            ("spheroid_hill_iso_affine", :firebrick),
        )
        h = get(CURVES, name, nothing)
        h === nothing && continue
        ep = [r.epoch for r in h]
        short = replace(name, "spheroid_hill_iso_" => "")
        plot!(plt, ep, [r.train for r in h]; lc = col, lw = 1.6, label = "$short — train")
        plot!(
            plt, ep, [r.validation for r in h];
            lc = col, lw = 1.6, ls = :dash, label = "$short — held out"
        )
    end
    savefig(plt, joinpath(ASSET, "training_curve.png"))
    println("\nwrote ", joinpath(ASSET, "training_curve.png"))
end

# ─── The table, for the manual page ──────────────────────────────────────────

if SECTIONS === nothing
    path = joinpath(OUT, "training_report.md")
    open(path, "w") do io
        println(io, "<!-- Generated by scripts/nn/train_models.jl — do not edit by hand. -->\n")
        println(io, "# Shipped surrogates\n")
        write(io, String(take!(report)))
    end
    println("\nwrote ", path)
else
    @info "partial run ($(join(ARGS, ", "))) — training_report.md left untouched"
end
