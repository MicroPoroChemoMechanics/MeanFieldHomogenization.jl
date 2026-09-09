# =============================================================================
#  train_layered_spheroid.jl — the two surrogates of an `N`-layer confocal
#  spheroid, taught by the axisymmetric Fourier finite-element cell.
#
#      julia --project=scripts/nn scripts/nn/train_layered_spheroid.jl [ntrain nval]
#
#  Writes `layered_spheroid_strain.json` and `layered_spheroid_stress.json` into
#  `src/NeuralInclusions/models/`, plus the learning curves the tutorial embeds.
#
#  ## Why the teacher is the finite-element cell and not the closed form
#
#  `LayeredSpheroid` solves this body in closed form, and on this slice it is
#  exact — so it is tempting to label from it and get a free dataset. It cannot
#  supply the labels this needs: a heterogeneous inclusion enters through gate B
#  with **both** localization tensors, and the analytic type implements the
#  strain side only. Its `stress_strain_loc` is refused rather than answered
#  (the generic `ℂ₁ : 𝔸_εε` presupposes a uniform stiffness a multilayer does
#  not have), so half the pair simply does not exist there.
#
#  The finite-element cell returns both, from one solve. So the closed form is
#  what *validates* the teacher — the tutorial shows that comparison — and the
#  cell is what teaches. That ordering is also what makes the surrogate useful
#  beyond this slice: the same object meshes nests of spheroids with freely
#  chosen semi-axes, where no closed form exists at all.
#
#  ## The box, and why the features are contrast ratios
#
#  A heterogeneous morphology carries its constituents inside itself, so `𝔸`
#  depends on the reference medium only through the **contrast**. Scaling `ℂ₀`
#  and every constituent together leaves `𝔸_εε` unchanged and multiplies
#  `𝔸_σε` by the factor — so the features are `log Eₖ/E₀`, never absolute
#  moduli, and the reference medium is just a unit of stress.
#
#  Every Poisson ratio is fixed at 0.2 and stated here rather than left
#  implicit: with them free the box would have six dimensions, and 750 solves
#  spread over six dimensions is 3 points per dimension.
#
#  ## Prolate only, and that is the confocal family's own restriction
#
#  `confocal_layer_radii` refuses `ω ≈ 1`: a sphere has no focal distance, so
#  the confocal family does not contain one and the box cannot straddle it.
#  Prolate and oblate are therefore two separate boxes — the analytic solution
#  splits into two branches there too, real `q` against `q = iτ`. This ships the
#  prolate one; `ARGS[3] == "oblate"` trains the mirror box, and the tutorial
#  says so rather than implying the model covers both.
# =============================================================================

import Pkg
Pkg.activate(@__DIR__; io = devnull)
Pkg.instantiate(; io = devnull)

using MeanFieldHomogenization
using TensND
using Printf
using LinearAlgebra

import Lux, Optimisers, Zygote           # loads MeanFieldHomogenizationLuxExt
import SHA                               # keys the checkpoint file
import Ferrite, FerriteGmsh, Gmsh        # loads MeanFieldHomogenizationFerriteExt

const NI = MeanFieldHomogenization.NeuralInclusions
const OUT = NI.MODEL_DIR
const ASSET = normpath(joinpath(@__DIR__, "..", "..", "docs", "src", "assets", "nn"))
mkpath(ASSET)

const NTRAIN = length(ARGS) ≥ 1 ? parse(Int, ARGS[1]) : 600
const NVAL = length(ARGS) ≥ 2 ? parse(Int, ARGS[2]) : 150
const FAMILY = length(ARGS) ≥ 3 ? ARGS[3] : "prolate"
FAMILY in ("prolate", "oblate") ||
    error("the third argument is \"prolate\" or \"oblate\", got \"$FAMILY\"")

const NU = 0.2
const E0 = 1.0
ce(E, ν) = iso_stiffness(E / (3 * (1 - 2ν)), E / (2 * (1 + ν)))
const C0 = ce(E0, NU)

# `ω` away from 1 on the side asked for. The inner bound is 1.25 rather than
# 1.05 because that is where the analytic branch this is validated against is
# itself trustworthy; see the tutorial's near-sphere section.
const ΩLO, ΩHI = FAMILY == "prolate" ? (1.25, 3.0) : (0.30, 0.80)

const BOX = NI.SampleBox(
    [:log_aspect, :core_fraction, :log_mu_ratio_1, :log_mu_ratio_2],
    [log(ΩLO), 0.20, log(0.5), log(0.5)],
    [log(ΩHI), 0.70, log(4.0), log(4.0)],
)

const MESH = (; nradial = 14, radius_ratio = 5.0)

"The inclusion at one point of the box, and its two constituents."
function geometry(x)
    ω, w = exp(x[1]), x[2]
    C1, C2 = ce(exp(x[3]) * E0, NU), ce(exp(x[4]) * E0, NU)
    ar, dr = confocal_layer_radii(ω, 1.0, (w, 1 - w))
    return FEAxiLayeredSpheroid(
        ar, dr, (C1, C2); opts = FEAxiMeshOptions(; MESH...)
    ), (C1, C2)
end

# `MFH_NN_SPEC=anchored` learns `𝕄 = 𝔸_b⁻¹ : 𝔸` against the homogeneous spheroid
# at the layers' mean modulus, which is exact on the whole face `r₁ = r₂` of the
# box. Measured on these very labels, that removes a factor of 1.8 from the
# spread the strain side must cover and 5.6 from the stress side.
const SPEC_KIND = get(ENV, "MFH_NN_SPEC", "dimensionless")
SPEC_KIND in ("dimensionless", "anchored") ||
    error("MFH_NN_SPEC is \"dimensionless\" or \"anchored\", got \"$SPEC_KIND\"")
_spec(class) = SPEC_KIND == "anchored" ?
    NI.AnchoredHill(class, :layered_spheroid) : NI.DimensionlessHill(class)
const SPEC_A = _spec(NI.StrainLocTI())
const SPEC_B = _spec(NI.StressLocTI())

# The revolution axis of the cell, which is the frame every label is expressed
# in — `_class_frame` of a localization class reads column 3.
const FRAME = (0.0, 0.0, 1.0)

# Deliberately not through `generate_dataset`: one solve returns **both**
# tensors, and the generic path would mesh and factorize a second time per
# ─── Checkpointing, because a lost hour is a lost hour ───────────────────────
#
# One solve costs several seconds, so a full dataset is more than an hour of
# uninterrupted finite elements, and an interrupted run used to lose every
# sample it had already paid for. Each label is therefore appended to a
# checkpoint file the moment it exists, and a restart recomputes only what is
# missing.
#
# The index is a sound name for a sample because the Halton sample is
# deterministic: point `j` depends on `j` and the base alone, never on `n`. So a
# resume cannot mix two designs, and enlarging the dataset keeps every sample
# already computed.
#
# `MFH_NN_MAX_NEW` bounds the number of *new* solves one process performs per data
# set (so a run that finishes the training set still gets a full budget for the
# held-out one), which
# is the actual protection: it keeps each invocation short instead of trusting a
# memory cap to survive eighty minutes of meshing. When the budget runs out the
# script says what remains and stops without training on a partial set.

const CKPT_DIR = get(ENV, "MFH_NN_CKPT", joinpath(tempdir(), "mfh_nn_datasets"))
const MAX_NEW = parse(Int, get(ENV, "MFH_NN_MAX_NEW", string(typemax(Int))))

# Everything that changes a label goes into the key, and nothing that does not:
# the network, the epochs and the output specification are all downstream of it.
const CKPT_KEY = bytes2hex(
    SHA.sha1(
        string(
            "v1|", FAMILY, "|", BOX.names, "|", BOX.lo, "|", BOX.hi,
            "|", NU, "|", E0, "|", MESH,
        )
    )
)[1:12]

ckpt_path(tag) = joinpath(CKPT_DIR, "layered_spheroid_$(tag)_$(CKPT_KEY).csv")

"Labels already on disk, by sample index. A line torn by a hard kill is skipped."
function read_ckpt(tag)
    done = Dict{Int, Tuple{Vector{Float64}, Vector{Float64}}}()
    path = ckpt_path(tag)
    isfile(path) || return done
    for line in eachline(path)
        f = split(strip(line), ',')
        length(f) == 13 || continue
        j = tryparse(Int, f[1])
        v = map(x -> tryparse(Float64, x), f[2:13])
        (j === nothing || any(isnothing, v)) && continue
        done[j] = (collect(v[1:6]), collect(v[7:12]))
    end
    return done
end

function append_ckpt(tag, j, za, zb)
    mkpath(CKPT_DIR)
    open(ckpt_path(tag), "a") do io
        println(io, join((j, za..., zb...), ','))
    end
    return nothing
end

# surrogate for nothing.
function label(X, tag)
    n = size(X, 2)
    done = read_ckpt(tag)
    todo = [j for j in 1:n if !haskey(done, j)]
    @printf(
        "    %d / %d on disk, %d to go — %s\n",
        n - length(todo), n, length(todo), ckpt_path(tag)
    )
    flush(stdout)
    budget = min(length(todo), MAX_NEW)
    t0 = time()
    for (i, j) in enumerate(todo)
        i > budget && break
        geom, _ = geometry(collect(view(X, :, j)))
        frame = NI._class_frame(NI.StrainLocTI(), geom)
        A, B = fe_axi_localization(geom, C0)
        # `components` measures the projection residual, so a wrong axis or a
        # response that is not transversely isotropic about it fails here
        # instead of training quietly on corrupted labels.
        za = collect(NI.components(NI.StrainLocTI(), A, frame; atol = 1.0e-6)) .*
            NI.dimensionless_scale(NI.StrainLocTI(), C0)
        zb = collect(NI.components(NI.StressLocTI(), B, frame; atol = 1.0e-6)) .*
            NI.dimensionless_scale(NI.StressLocTI(), C0)
        append_ckpt(tag, j, za, zb)          # durable before anything else
        done[j] = (za, zb)
        if i % 10 == 0 || i == budget
            el = time() - t0
            @printf(
                "    %4d / %4d new   %5.1f s elapsed, %5.1f s left\n",
                i, budget, el, el * (budget - i) / i
            )
            flush(stdout)
        end
    end
    length(done) == n || return nothing      # incomplete: the caller stops
    ZA = Matrix{Float64}(undef, 6, n)
    ZB = Matrix{Float64}(undef, 6, n)
    for j in 1:n
        ZA[:, j] .= done[j][1]
        ZB[:, j] .= done[j][2]
    end
    return ZA, ZB
end

println("="^78)
println("finite-element dataset: $NTRAIN training + $NVAL held-out solves")
@printf(
    "  box: ω ∈ [%.2f, %.2f] (%s), w ∈ [0.2, 0.7], E₁/E₀ and E₂/E₀ ∈ [0.5, 4]\n",
    ΩLO, ΩHI, FAMILY
)
println("  fixed: ν = $NU everywhere, nradial = $(MESH.nradial), R/a = $(MESH.radius_ratio)")
println("="^78)
flush(stdout)

Xt = NI.sample_box(BOX, NTRAIN)
# The held-out offset is a fixed constant, not `NTRAIN`: a Halton point depends
# on its index alone, so this keeps the two sets disjoint *and* keeps the
# held-out set — and its checkpoint — valid when the training set is enlarged.
Xv = NI.sample_box(BOX, NVAL; offset = 100_000)

println("  training set:")
const LT = label(Xt, "train")
println("  held-out set:")
const LV = LT === nothing ? nothing : label(Xv, "val")

if LT === nothing || LV === nothing
    println("\n", "="^78)
    println("dataset incomplete — the per-set solve budget MFH_NN_MAX_NEW = $MAX_NEW ran out.")
    println("Every label computed is on disk. Run the same command again to continue;")
    println("nothing is recomputed, and training starts on the run that completes it.")
    println("="^78)
    exit(0)
end
const ZAt, ZBt = LT
const ZAv, ZBv = LV

const FE_SECONDS = let
    geom, _ = geometry([log(2.0), 0.4, log(2.0), log(1.0)])
    fe_axi_mesh_report(geom)                 # mesh out of the timing
    @elapsed fe_axi_localization(geom, ce(1.01 * E0, NU))
end
@printf("\none cold finite-element evaluation: %.2f s\n", FE_SECONDS)

# The checkpoint holds the **dimensionless** components, which at fixed `ℂ₀`
# represent `𝔸` faithfully and invertibly. So changing the output specification
# re-encodes the labels already on disk rather than re-solving for them: the 700
# finite-element solves serve both specifications.
function reencode(Z, X, class, spec)
    spec isa NI.DimensionlessHill && return Z
    sc = NI.dimensionless_scale(class, C0)
    out = similar(Z)
    for j in axes(Z, 2)
        t = NI.build(class, collect(view(Z, :, j)) ./ sc, FRAME)
        out[:, j] .= NI.encode(spec, t, C0, FRAME, collect(view(X, :, j)), BOX.names)
    end
    return out
end

function fit(name, spec, ZT, ZV; notes)
    println("\n", "="^78)
    println("training `$name`")
    println("="^78)
    flush(stdout)
    train = NI.Dataset(Xt, ZT, copy(BOX.names))
    val = NI.Dataset(Xv, ZV, copy(BOX.names))
    history = Any[]
    s = NI.train_surrogate(
        spec, BOX, train, val;
        options = NI.TrainingOptions(;
            hidden = [64, 64], epochs = 8000, batchsize = 128, verbose = true
        ),
        teacher_name = "fe_axi_localization(FEAxiLayeredSpheroid, TensISO{4}) — " *
            "axisymmetric Fourier finite elements",
        notes, history,
    )
    NI.report_surrogate(s, val; labels = NI.component_labels(spec))
    println("wrote ", NI.save_surrogate(joinpath(OUT, name * ".json"), s))
    flush(stdout)
    return s, history, val
end

const SUFFIX = (FAMILY == "prolate" ? "" : "_oblate") *
    (SPEC_KIND == "anchored" ? "_anchored" : "")

sA, hA, valA = fit(
    "layered_spheroid_strain" * SUFFIX, SPEC_A,
    reencode(ZAt, Xt, NI.StrainLocTI(), SPEC_A),
    reencode(ZAv, Xv, NI.StrainLocTI(), SPEC_A);
    notes = "𝔸_εε of a two-layer confocal spheroid, from axisymmetric Fourier " *
        "finite elements; features (log ω, w, log E₁/E₀, log E₂/E₀) at ν = 0.2, " *
        "$FAMILY branch",
)
sB, hB, valB = fit(
    "layered_spheroid_stress" * SUFFIX, SPEC_B,
    reencode(ZBt, Xt, NI.StressLocTI(), SPEC_B),
    reencode(ZBv, Xv, NI.StressLocTI(), SPEC_B);
    notes = "𝔸_σε of the same morphology, divided by 2μ₀ — it is of degree +1 " *
        "in the moduli where 𝔸_εε is of degree 0. The analytic type supplies " *
        "no counterpart: this half of gate B exists only through the cell",
)

# The speed-up, which is the whole practical point of a surrogate. The frame is
# the revolution axis, and `guard = :none` keeps the domain check out of the
# timing — it is a bounds test, not part of the evaluation.
let x = [log(2.0), 0.4, log(2.0), 0.0], frame = (0.0, 0.0, 1.0)
    sA(x, C0, frame; guard = :none)                       # warm up
    t = @elapsed for _ in 1:1000
        sA(x, C0, frame; guard = :none)
    end
    per = t / 1000
    @printf("\none surrogate evaluation: %.1f µs\n", 1.0e6 * per)
    @printf("speed-up over one finite-element evaluation: about %.0f×\n", FE_SECONDS / per)
end
