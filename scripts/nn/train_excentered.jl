# =============================================================================
#  train_excentered.jl — train a neural surrogate on the finite-element
#  localization tensors of `FEExcenteredSphere`, and commit it under
#  `src/NeuralInclusions/models/`.
#
#  Maintenance script, run **by hand**: it performs one axisymmetric
#  finite-element solve per sample, so a full run costs tens of minutes. Nothing
#  here happens at test or documentation-build time — the tutorial
#  `scripts/85_neural_excentered_sphere.jl` loads the committed JSON.
#
#      julia scripts/nn/train_excentered.jl            # the shipped dataset
#      julia scripts/nn/train_excentered.jl 40 10      # a quick smoke run
#
#  This is the case the whole surrogate machinery exists for. The ellipsoid
#  pilot proves the pipeline against a closed form; here there *is* no closed
#  form, the teacher costs ~2 s per evaluation, and an iterative scheme asks for
#  it again at every iteration.
# =============================================================================

import Pkg
Pkg.activate(@__DIR__; io = devnull)
Pkg.instantiate(; io = devnull)

using MeanFieldHomogenization
using TensND
using Printf
using Plots

import Ferrite, FerriteGmsh, Gmsh          # the teacher
import Lux, Optimisers, Zygote             # the optimizer

gr()
default(;
    fontfamily = "sans-serif", framestyle = :box, grid = true, legendfontsize = 8,
    left_margin = 5Plots.mm, bottom_margin = 5Plots.mm,
)

const NI = MeanFieldHomogenization.NeuralInclusions
const OUT = NI.MODEL_DIR
const ASSET = normpath(joinpath(@__DIR__, "..", "..", "docs", "src", "assets", "nn"))
mkpath(OUT)
mkpath(ASSET)

const NTRAIN = length(ARGS) ≥ 1 ? parse(Int, ARGS[1]) : 600
const NVAL = length(ARGS) ≥ 2 ? parse(Int, ARGS[2]) : 150

# ─── The parameter box ───────────────────────────────────────────────────────
#
#  Fixed, and stated here rather than left implicit: all Poisson ratios 0.2 and
#  a stiff old aggregate at E₁/E₀ = 3.5, as in Adessina et al. (2017). What
#  varies is the eccentricity, the core fraction, and the quality of the adhered
#  mortar.
#
#  The features are **contrast ratios**, never absolute moduli: a heterogeneous
#  morphology carries its constituents inside itself, so `𝔸` depends on the
#  reference medium only through the contrast. Scaling `ℂ₀` and every constituent
#  together leaves `𝔸_εε` unchanged and multiplies `𝔸_σε` by the factor — exact
#  to ~10⁻¹⁴. The reference medium below is therefore just a unit of stress.

const NU = 0.2
const E0 = 1.0
const MU1_RATIO = 3.5                       # old aggregate / fresh paste

ce(E, ν) = iso_stiffness(E / (3 * (1 - 2ν)), E / (2 * (1 + ν)))
const C0 = ce(E0, NU)
const C1 = ce(MU1_RATIO * E0, NU)

const BOX = NI.SampleBox(
    [:eccentricity, :core_fraction, :log_mu_ratio_2],
    [0.0, 0.20, log(0.10)],
    [0.8, 0.70, log(2.00)],
)

const MESH = (; nradial = 18, radius_ratio = 4.0)

geometry(x) = FEExcenteredSphere(
    1.0, (C1, ce(exp(x[3]) * E0, NU));
    core_fraction = x[2], eccentricity = x[1], MESH...
)

# ─── Labeling ────────────────────────────────────────────────────────────────
#
#  Deliberately *not* through `generate_dataset`: one finite-element solve
#  returns **both** localization tensors, and going through the generic path
#  twice — once per surrogate — would mesh and factorize everything a second
#  time for nothing. The two label matrices are filled from one pass.

const SPEC_A = NI.DimensionlessHill(NI.StrainLocTI())
const SPEC_B = NI.DimensionlessHill(NI.StressLocTI())

function label(X)
    n = size(X, 2)
    ZA = Matrix{Float64}(undef, 6, n)
    ZB = Matrix{Float64}(undef, 6, n)
    t0 = time()
    for j in 1:n
        x = collect(view(X, :, j))
        geom = geometry(x)
        frame = NI._class_frame(NI.StrainLocTI(), geom)
        A, B = fe_axi_localization(geom, C0)
        ## `components` checks the projection residual, so a wrong axis or a
        ## response that is not transversely isotropic about it fails here rather
        ## than training quietly on corrupted labels.
        ZA[:, j] .= collect(NI.components(NI.StrainLocTI(), A, frame; atol = 1.0e-6)) .*
            NI.dimensionless_scale(NI.StrainLocTI(), C0)
        ZB[:, j] .= collect(NI.components(NI.StressLocTI(), B, frame; atol = 1.0e-6)) .*
            NI.dimensionless_scale(NI.StressLocTI(), C0)
        if j % 25 == 0 || j == n
            el = time() - t0
            @printf(
                "    %4d / %4d   %5.1f s elapsed, %5.1f s left\n",
                j, n, el, el * (n - j) / j
            )
            flush(stdout)
        end
    end
    return ZA, ZB
end

println("="^78)
println("finite-element dataset: $(NTRAIN) training + $(NVAL) held-out solves")
println("  box: α ∈ [0, 0.8], w ∈ [0.2, 0.7], E₂/E₀ ∈ [0.1, 2]")
println("  fixed: ν = $NU everywhere, E₁/E₀ = $MU1_RATIO, nradial = $(MESH.nradial), R/a = $(MESH.radius_ratio)")
println("="^78)

Xt = NI.sample_box(BOX, NTRAIN)
Xv = NI.sample_box(BOX, NVAL; offset = NTRAIN)

println("  training set:")
ZAt, ZBt = label(Xt)
println("  held-out set:")
ZAv, ZBv = label(Xv)

const FE_SECONDS = let
    geom = geometry([0.4, 0.5, log(0.4)])
    fe_axi_mesh_report(geom)                 # mesh out of the timing
    @elapsed fe_axi_localization(geom, ce(1.01 * E0, NU))
end
@printf("\none cold finite-element evaluation: %.2f s\n", FE_SECONDS)

# ─── Fit ─────────────────────────────────────────────────────────────────────

function fit(name, spec, ZT, ZV; notes)
    println("\n", "="^78)
    println("training `$name`")
    println("="^78)
    train = NI.Dataset(Xt, ZT, copy(BOX.names))
    val = NI.Dataset(Xv, ZV, copy(BOX.names))
    history = Any[]
    s = NI.train_surrogate(
        spec, BOX, train, val;
        options = NI.TrainingOptions(;
            hidden = [64, 64], epochs = 6000, batchsize = 128, verbose = true
        ),
        teacher_name = "fe_axi_localization(FEExcenteredSphere, TensISO{4}) — " *
            "axisymmetric Fourier finite elements",
        notes, history,
    )
    NI.report_surrogate(s, val; labels = NI.component_labels(spec))
    println("wrote ", NI.save_surrogate(joinpath(OUT, name * ".json"), s))
    return s, history, val
end

sA, hA, valA = fit(
    "excentered_sphere_strain", SPEC_A, ZAt, ZAv;
    notes = "𝔸_εε of a sphere with an off-center core, from axisymmetric Fourier " *
        "finite elements; features (α, w, log E₂/E₀) at ν = 0.2 and E₁/E₀ = 3.5",
)
sB, hB, valB = fit(
    "excentered_sphere_stress", SPEC_B, ZBt, ZBv;
    notes = "𝔸_σε of the same morphology, divided by 2μ₀ — it is of degree +1 in " *
        "the moduli where 𝔸_εε is of degree 0",
)

# ─── The learning curves, for the tutorial ───────────────────────────────────

plt = plot(;
    xlabel = "epoch", ylabel = "mean squared error (standardized targets)",
    yscale = :log10, legend = :topright, size = (760, 460),
    title = "Learning curves, surrogate of the finite-element solve\n" *
        "$(NTRAIN) train / $(NVAL) held out, Adam, batch 128, 64+64 hidden, tanh",
    titlefontsize = 10,
)
for (h, nm, col) in ((hA, "𝔸_εε", :steelblue), (hB, "𝔸_σε", :firebrick))
    ep = [r.epoch for r in h]
    plot!(plt, ep, [r.train for r in h]; lc = col, lw = 1.6, label = "$nm — train")
    plot!(
        plt, ep, [r.validation for r in h];
        lc = col, lw = 1.6, ls = :dash, label = "$nm — held out"
    )
end
savefig(plt, joinpath(ASSET, "excentered_training_curve.png"))
println("\nwrote ", joinpath(ASSET, "excentered_training_curve.png"))

# ─── The report, for the tutorial page ───────────────────────────────────────

open(joinpath(OUT, "excentered_report.md"), "w") do io
    println(io, "<!-- Generated by scripts/nn/train_excentered.jl — do not edit by hand. -->\n")
    println(io, "# Surrogate of the finite-element eccentric sphere\n")
    println(io, "- teacher: `fe_axi_localization`, `nradial = $(MESH.nradial)`, `R/a = $(MESH.radius_ratio)`")
    @printf(io, "- one cold finite-element evaluation: **%.2f s**\n", FE_SECONDS)
    println(io, "- box: `α ∈ [0, 0.8]`, `w ∈ [0.2, 0.7]`, `E₂/E₀ ∈ [0.1, 2]`; fixed `ν = $NU`, `E₁/E₀ = $MU1_RATIO`")
    println(io, "- $(NTRAIN) training + $(NVAL) held-out solves\n")
    println(io, "| surrogate | network | worst held-out error |")
    println(io, "| --- | --- | ---: |")
    for (nm, s) in (("`𝔸_εε`", sA), ("`𝔸_σε`", sB))
        @printf(
            io, "| %s | %s | %.3e |\n",
            nm, join(NI.layer_widths(s.net), "→"), NI.worst_error(s.provenance)
        )
    end
end
println("wrote ", joinpath(OUT, "excentered_report.md"))
