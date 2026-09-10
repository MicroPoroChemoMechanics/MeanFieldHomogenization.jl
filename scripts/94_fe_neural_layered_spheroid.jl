# =============================================================================
#  94_fe_neural_layered_spheroid.jl — the whole computation for an N-layer
#  spheroid: geometry, mesh, finite elements against both closed forms, the
#  trained surrogate, the schemes, and the sensitivity the mesh cannot give.
#
#      julia --project=scripts/fe scripts/94_fe_neural_layered_spheroid.jl
#
#  Run by hand, never at documentation-build time. The static page
#  `docs/src/tutorials/axi_layered_spheroid.md` embeds the figures produced by
#  `scripts/fe/make_layered_spheroid_figures.jl` and quotes the numbers printed
#  here; re-meshing and re-training during a build would cost hours.
#
#  Numbered 94 because the 30-39 and 80-89 blocks are full. Like
#  `96_nano_spheroids.jl`, this is not an N-body script despite the range.
#
#  §5 onwards needs the two committed surrogates,
#  `layered_spheroid_strain.json` and `layered_spheroid_stress.json`, produced
#  by `scripts/nn/train_layered_spheroid.jl` (1300 finite-element solves, close
#  to three hours, and checkpointed so it need not be one sitting). Without them
#  the script still runs and says which sections it skipped.
# =============================================================================

import Pkg
Pkg.activate(joinpath(@__DIR__, "fe"); io = devnull)

using MeanFieldHomogenization
using TensND
using LinearAlgebra
using Printf

import Ferrite, FerriteGmsh, Gmsh

const NI = MeanFieldHomogenization.NeuralInclusions
const CAN = TensND.CanonicalBasis{3, Float64}()

arr(t) = Array(TensND.get_array(TensND.change_tens(t, CAN)))
reldiff(a, b) = norm(arr(a) - arr(b)) / norm(arr(b))
ce(E, ν) = iso_stiffness(E / (3 * (1 - 2ν)), E / (2 * (1 + ν)))

const NU = 0.2
const C0 = ce(1.0, NU)
const K0 = TensISO{3}(1.0)
banner(t) = (println("\n", "="^78); println(t); println("="^78); flush(stdout))

# ─── §1 The geometry, and the two ways in ────────────────────────────────────

banner("§1  The geometry: confocal for calibration, free radii for the rest")

println("""
Layers are given by their semi-axes, per layer, ascending — exactly the
(axis_radii, disk_radii) pair `LayeredSpheroid` takes. One call produces them
and both inclusions consume them, so a cross-check compares two solvers on one
body instead of on two transcriptions of one body.""")

const AR, DR = confocal_layer_radii(2.0, 1.0, (0.3, 0.7))
const CS = (ce(4.0, NU), ce(1.5, NU))

@printf(
    "\nconfocal, c/a = 2, core 30 %% of the volume:\n  axis_radii = (%.6f, %.6f)\n  disk_radii = (%.6f, %.6f)\n",
    AR[1], AR[2], DR[1], DR[2]
)
@printf(
    "  focal distances agree to %.2e — that is what confocal means\n",
    abs((AR[1]^2 - DR[1]^2) - (AR[2]^2 - DR[2]^2))
)

fe = FEAxiLayeredSpheroid(AR, DR, CS)
@printf(
    "  layers %d, volume fractions %s\n", layer_count(fe),
    string(round.(layer_fractions(fe); digits = 4))
)
println("  → the fractions come back as the ones requested, which checks the")
println("    confocal bisection against the closed-form layer volume")

println("\nfree radii — an oblate core inside a prolate shell. Nested,")
println("axisymmetric, and outside every confocal family:")
let g = FEAxiLayeredSpheroid((0.4, 1.4), (0.9, 1.0), CS)
    @printf("  layer volumes %s\n", string(round.(layer_volumes(g); digits = 5)))
    @printf("  fractions     %s\n", string(round.(layer_fractions(g); digits = 4)))
end

println("\nand the nesting guard, which is two strict comparisons per layer:")
try
    FEAxiLayeredSpheroid((0.5, 1.4), (1.1, 1.0), CS)
    println("  NOT REFUSED — that is a bug")
catch e
    println("  refused: ", first(split(sprint(showerror, e), '\n')))
end

# ─── §2 The mesh ─────────────────────────────────────────────────────────────

banner("§2  The mesh, and the check that exists for any geometry")

println("""
Each layer's meshed volume of revolution against the closed form 4πa²c/3 — a
closed form for *any* semi-axes, confocal or not, so this is the one check
available on a geometry with no analytic solution at all.""")

@printf("\n%-32s %8s %8s %14s\n", "geometry", "layers", "cells", "volume error")
for (tag, ar, dr) in (
        ("confocal prolate, 2 layers", AR, DR),
        ("confocal oblate, 3 layers", confocal_layer_radii(0.5, 1.0, (0.25, 0.35, 0.4))...),
        ("free radii, aspect reverses", (0.4, 1.4), (0.9, 1.0)),
        ("concentric spheres, 3 layers", (0.5, 0.8, 1.0), (0.5, 0.8, 1.0)),
    )
    n = length(ar)
    g = FEAxiLayeredSpheroid(
        ar, dr, ntuple(_ -> TensISO{3}(1.0), n);
        opts = FEAxiMeshOptions(; nradial = 16, radius_ratio = 5.0),
    )
    r = fe_axi_mesh_report(g)
    @printf("%-32s %8d %8d %14.1e\n", tag, n, r.ncells, r.volume_error)
    flush(stdout)
end

# ─── §3 Against the two closed forms ─────────────────────────────────────────

banner("§3  The cell against the two exact families that cross this space")

println("""
Confocal layers, where `LayeredSpheroid` is the closed form — both branches of
it, since the oblate one runs in complex arithmetic — and concentric spheres of
*arbitrary* radii, where `LayeredSphere` is. The two families meet only at the
equal-radii sphere, so between them they pin the mesher, the per-layer material
map, the N-region average and the dipole correction.""")

@printf("\n%-32s %14s %14s\n", "slice", "conduction", "elasticity")
for (tag, ω, fr) in (
        ("confocal prolate, 2 layers", 2.0, (0.3, 0.7)),
        ("confocal oblate, 2 layers", 0.5, (0.4, 0.6)),
        ("confocal prolate, 3 layers", 1.8, (0.2, 0.3, 0.5)),
    )
    ar, dr = confocal_layer_radii(ω, 1.0, fr)
    n = length(fr)
    ks = ntuple(i -> TensISO{3}(1.0 + 3.0i), n)
    cs = ntuple(i -> ce(1.0 + i, NU), n)
    opts = FEAxiMeshOptions(; nradial = 14, radius_ratio = 6.0)
    d2 = reldiff(
        gradient_gradient_loc(FEAxiLayeredSpheroid(ar, dr, ks; opts), K0, K0),
        gradient_gradient_loc(LayeredSpheroid(ar, dr, ks), K0, K0),
    )
    d4 = reldiff(
        strain_strain_loc(FEAxiLayeredSpheroid(ar, dr, cs; opts), C0, C0),
        strain_strain_loc(LayeredSpheroid(ar, dr, cs), C0, C0),
    )
    @printf("%-32s %14.2e %14.2e\n", tag, d2, d4)
    flush(stdout)
end
for (tag, radii) in (
        ("spheres, free radii, 2 layers", (0.6, 1.0)),
        ("spheres, free radii, 3 layers", (0.4, 0.7, 1.0)),
    )
    n = length(radii)
    ks = ntuple(i -> TensISO{3}(1.0 + 3.0i), n)
    cs = ntuple(i -> ce(1.0 + i, NU), n)
    opts = FEAxiMeshOptions(; nradial = 16, radius_ratio = 6.0)
    d2 = reldiff(
        gradient_gradient_loc(FEAxiLayeredSpheroid(radii, radii, ks; opts), K0, K0),
        gradient_gradient_loc(LayeredSphere(radii, ks), K0, K0),
    )
    d4 = reldiff(
        strain_strain_loc(FEAxiLayeredSpheroid(radii, radii, cs; opts), C0, C0),
        strain_strain_loc(LayeredSphere(radii, cs), C0, C0),
    )
    @printf("%-32s %14.2e %14.2e\n", tag, d2, d4)
    flush(stdout)
end

println("""
\nAnd the stress side, which is not derivable from the strain side: the inclusion
is heterogeneous, so 𝔸_σε ≠ ℂ₁ : 𝔸_εε for any single ℂ₁. It comes out of the
same solve. Only `LayeredSphere` supplies a counterpart — the analytic spheroid
does not, and its generic is now refused rather than silently answered.""")
let radii = (0.6, 1.0), cs = (ce(3.0, NU), ce(1.5, NU))
    opts = FEAxiMeshOptions(; nradial = 16, radius_ratio = 6.0)
    @printf(
        "  𝔸_σε against LayeredSphere: %.2e\n",
        reldiff(
            stress_strain_loc(FEAxiLayeredSpheroid(radii, radii, cs; opts), C0, C0),
            stress_strain_loc(LayeredSphere(radii, cs), C0, C0),
        )
    )
    try
        stress_strain_loc(LayeredSpheroid(AR, DR, cs), C0, C0)
        println("  the analytic spheroid answered — that is a bug")
    catch e
        println("  the analytic spheroid: ", first(split(sprint(showerror, e), '\n')))
    end
end

# ─── §4 The near-sphere limit of the analytic branch ─────────────────────────

banner("§4  What this comparison found inside the analytic solution")

println("""
A single confocal layer *is* a homogeneous spheroid, so the closed-form Eshelby
result is its answer and the confocal machinery must reproduce it at every
aspect ratio. It did not, until its linear solve equilibrated the columns: they
are amplitudes of Papkovich-Neuber potentials, and those span q^(2n+1) as the
focal distance goes to zero, so the matrix loses its rank in floating point long
before the geometry degenerates. Transport on the identical chart was always
exact, which is what said the geometry was innocent.""")

@printf("\n%-12s %14s %14s\n", "|1 - c/a|", "elasticity", "transport")
for ω in (0.9999, 0.999, 0.99, 0.97, 0.95, 0.9, 0.7, 1.05, 1.3)
    ar, dr = confocal_layer_radii(ω, 1.0, (1.0,))
    c1, k1 = ce(4.0, NU), TensISO{3}(5.0)
    de = reldiff(
        strain_strain_loc(LayeredSpheroid(ar, dr, (c1,)), C0, C0),
        strain_strain_loc(Spheroid(ω), c1, C0),
    )
    dc = reldiff(
        gradient_gradient_loc(LayeredSpheroid(ar, dr, (k1,)), K0, K0),
        gradient_gradient_loc(Spheroid(ω), k1, K0),
    )
    @printf("%-12.4f %14.2e %14.2e\n", abs(1 - ω), de, dc)
    flush(stdout)
end
println("""
The residual limit is |1 - c/a| ≈ 1e-4, where the warning still fires. A
near-sphere is a sphere: `LayeredSphere` is the answer there, and it is exact.""")

# ─── §5 The surrogate ────────────────────────────────────────────────────────

banner("§5  The surrogate: both sides of gate B, one solve feeding each")

const MODELS = ("layered_spheroid_strain", "layered_spheroid_stress")

if !all(in(NI.shipped_models()), MODELS)
    println("""
The two models are not committed, so §5 to §7 are skipped. Produce them with

    julia --project=scripts/nn scripts/nn/train_layered_spheroid.jl 1200 100
""")
else
    strain_s = load_surrogate(model_path(MODELS[1]))
    stress_s = load_surrogate(model_path(MODELS[2]))

    println("features: ", strain_s.features)
    @printf(
        "𝔸_εε : %s, worst held-out error %.2e\n",
        join(NI.layer_widths(strain_s.net), "→"), worst_error(strain_s.provenance)
    )
    @printf(
        "𝔸_σε : %s, worst held-out error %.2e\n",
        join(NI.layer_widths(stress_s.net), "→"), worst_error(stress_s.provenance)
    )
    println("""
The features are **contrast ratios**, never absolute moduli. A heterogeneous
morphology carries its constituents inside itself, so its localization depends
on the reference medium only through the contrasts — which is also why the same
pair of models serves any ℂ₀ of the trained aspect range.

The teacher is the finite-element cell, not the closed form: `LayeredSpheroid`
supplies no stress side, so half of gate B exists only through the cell.""")

    # The surrogate-backed inclusion. `shape_params` are the features the
    # sensitivity API differentiates; `properties` back both the contrast
    # features and the Voigt/Reuss bounds.
    function nn_incl(ω, w, r1, r2)
        return NeuralLocalizationInclusion(
            (1.0, 1.0, ω);
            strain = strain_s, stress = stress_s,
            shape_params = (; core_fraction = w),
            fractions = (w, 1 - w),
            properties = (ce(r1, NU), ce(r2, NU)),
            guard = :error,
        )
    end

    println()
    check_inclusion_interface(nn_incl(2.0, 0.4, 2.0, 1.0))

    banner("§6  Surrogate against the cell, on the same bodies")
    println("\nconfocal, ν = 0.2 throughout, moduli relative to the matrix\n")
    @printf(
        "%-6s %-6s %-8s %-8s %12s %12s %10s\n",
        "c/a", "w", "E₁/E₀", "E₂/E₀", "𝔸_εε err", "𝔸_σε err", "speed-up"
    )
    for (ω, w, r1, r2) in (
            (1.4, 0.30, 3.0, 1.0), (2.0, 0.40, 2.0, 0.8),
            (2.6, 0.55, 1.0, 3.0), (1.7, 0.65, 0.7, 2.5),
        )
        ar, dr = confocal_layer_radii(ω, 1.0, (w, 1 - w))
        cell = FEAxiLayeredSpheroid(
            ar, dr, (ce(r1, NU), ce(r2, NU));
            opts = FEAxiMeshOptions(; nradial = 14, radius_ratio = 5.0),
        )
        tfe = @elapsed A, B = fe_axi_localization(cell, C0)
        nn = nn_incl(ω, w, r1, r2)
        strain_strain_loc(nn, C0, C0)                      # warm the compilation
        tnn = @elapsed strain_strain_loc(nn, C0, C0)
        @printf(
            "%-6.2f %-6.2f %-8.1f %-8.1f %12.2e %12.2e %10.0f×\n", ω, w, r1, r2,
            reldiff(strain_strain_loc(nn, C0, C0), A),
            reldiff(stress_strain_loc(nn, C0, C0), B), tfe / tnn
        )
        flush(stdout)
    end

    banner("§7  In a scheme, and the sensitivity the cell refuses")

    # Two builders, because the two questions below are not licit under the same
    # one. `IsoSymmetrize()` averages the inclusion over orientations, which is
    # **required** by the iterative schemes: they re-evaluate the inclusion in
    # their own running estimate, and an oriented spheroid makes that estimate
    # transversely isotropic, while the dipole boundary condition is the
    # closed-form *isotropic* field — the cell refuses an anisotropic reference.
    function rve_iso(incl, f)
        r = RVE()
        add_phase!(r, :matrix, Ellipsoid(1.0), Dict(:C => C0); fraction = :rest)
        add_phase!(
            r, :incl, incl, Dict(:C => C0);
            fraction = f, symmetrize = IsoSymmetrize()
        )
        return r
    end

    # Aligned, for the sensitivity: the effective medium is then transversely
    # isotropic and `C₁₁₁₁` is a component of it rather than a projection, so
    # the derivative below is the derivative of something well defined.
    function rve_aligned(incl, f)
        r = RVE()
        add_phase!(r, :matrix, Ellipsoid(1.0), Dict(:C => C0); fraction = :rest)
        add_phase!(r, :incl, incl, Dict(:C => C0); fraction = f)
        return r
    end
    young(C) = (kk = k_mu(isotropify(C)); 9kk[1] * kk[2] / (3kk[1] + kk[2]))

    let ω = 2.0, w = 0.4, r1 = 2.0, r2 = 0.8, f = 0.3
        ar, dr = confocal_layer_radii(ω, 1.0, (w, 1 - w))
        cell = FEAxiLayeredSpheroid(
            ar, dr, (ce(r1, NU), ce(r2, NU));
            opts = FEAxiMeshOptions(; nradial = 14, radius_ratio = 5.0),
        )
        nn = nn_incl(ω, w, r1, r2)
        println("\n  c/a = 2, w = 0.4, E₁/E₀ = 2, E₂/E₀ = 0.8, f = 0.3")
        println("  effective Young modulus E_eff/E₀, inclusions averaged over orientations\n")
        @printf("  %-16s %14s %14s\n", "scheme", "cell", "surrogate")
        for sch in (Dilute(), MoriTanaka(), SelfConsistent())
            @printf(
                "  %-16s %14.6f %14.6f\n", string(nameof(typeof(sch))),
                young(homogenize(rve_iso(cell, f), sch, :C)),
                young(homogenize(rve_iso(nn, f), sch, :C))
            )
            flush(stdout)
        end
        println("""
  The self-consistent row is the one that matters: it changes its reference
  medium at every iteration, so the cell never hits its cache and pays a full
  mesh and factorization per iteration, while the surrogate is unaffected.""")

        println("""
\n  And the derivative by the morphology, which is most of the reason to train
  one. The cell solves in Float64 and memoizes on the reference medium alone, so
  a derivative through it would come back a silent zero — it raises instead.""")
        idx = C -> Array(get_array(C))[1, 1, 1, 1]
        @printf("\n  %-10s %16s %16s\n", "w", "AD", "finite differences")
        for w0 in (0.30, 0.45, 0.60)
            d = derivative(
                rve_aligned(nn_incl(ω, w0, r1, r2), f), MoriTanaka(),
                geometry(:incl, :shape_params, 1); indexer = idx
            )
            h = 1.0e-5
            fd = (
                idx(homogenize(rve_aligned(nn_incl(ω, w0 + h, r1, r2), f), MoriTanaka())) -
                    idx(homogenize(rve_aligned(nn_incl(ω, w0 - h, r1, r2), f), MoriTanaka()))
            ) / 2h
            @printf("  %-10.2f %16.6f %16.6f\n", w0, d, fd)
            flush(stdout)
        end
        try
            derivative(
                rve_aligned(cell, f), MoriTanaka(),
                geometry(:incl, :shape_params, 1); indexer = idx
            )
            println("\n  the cell answered — that is a bug")
        catch e
            println("\n  the cell: ", first(split(sprint(showerror, e), '\n')))
        end
    end
end
