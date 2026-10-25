# =============================================================================
#  quasibrittle_strength.jl — shared three-scale quasi-brittle strength model.
#
#    Reference: pichler2011 in docs/src/references.bib — "Upscaling quasi-brittle strength of
#    cement paste and mortar : a multi-scale engineering mechanics model",
#    Cement and Concrete Research 41, 467-476.
#    https://doi.org/10.1016/j.cemconres.2011.01.010
#
#  Included by `scripts/41_multiscale_strength.jl` (plots / tables) and
#  `scripts/bench_echoes/benchmark_strength.jl` (cross-validation against the
#  echoes Python reference).  Built ENTIRELY on the public MeanFieldHomogenization API :
#
#  * Hydrate Foam (HF)  — Self-Consistent over NTHETA needle families
#    (`Spheroid(ω; euler_angles = (θᵢ, 0, 0))`), each with
#    `symmetrize = TISymmetrize((0,0,1))` : the EXACT azimuthal average about
#    the global axis (echoes' per-bin `symmetrize=[TI]`), preserving the
#    non-major-symmetric content of the concentration tensors
#    (`TensTI{4,T,8}`), plus water and air.  The reference medium of each
#    family is taken through the same average about the axis: the SC estimate
#    built from such families is TI about it, so the average leaves it
#    unchanged — echoes computes the per-bin concentration in the running
#    estimate likewise.
#  * Cement Paste (CP)  — Mori-Tanaka (HF matrix + clinker).
#  * Mortar (MO)        — Mori-Tanaka (CP matrix + sand).
#  * Strength           — Pichler & Hellmich (2011): the quadratic average
#    √⟨½ s:s⟩ of the deviatoric stress in the hydrates (the article's
#    normalization, ½ under the root) is bounded by σ_ult. With the pull-back
#    `M = S_mo : ∂C_mo/∂μ_hyd : S_mo`, `fc/σ_ult = 1/√(M₃₃₃₃ · μ² / f_θ)`;
#    `∂C_mo/∂μ_hyd(bin θ=0)` comes from ONE ForwardDiff pass through the
#    whole three-scale chain (multi-bin SC included) — no hand-rolled IFT, no
#    Mandel bypass. The code passes ∂/∂(2μ) with 2μ², the same number.
# =============================================================================

using MeanFieldHomogenization
using ForwardDiff
using TensND
using LinearAlgebra

# ── Physical constants (quasi-brittle strength model) ───────────────────────────
const ρ_w = 1.0
const ρ_clin = 3.15;  const d_clin = ρ_clin / ρ_w
const ρ_hyd = 2.073; const d_hyd = ρ_hyd / ρ_w
const ρ_san = 2.648; const d_san = ρ_san / ρ_w

const K_clin, μ_clin = 116.7, 53.8
const K_hyd_ref, μ_hyd_ref = 18.7, 11.8
const K_san, μ_san = 37.8, 44.3

# Water / air stiffness regularization : echoes uses an exact zero; the small
# positive TINY selects the physically relevant (percolating) SC branch while
# keeping the iteration smooth (documented deliberate deviation).
const TINY = 1.0e-3

# Needle aspect ratio and angular discretization (main CCR2011 script; the
# companion iso variant uses ω = 100 — see `4x_cementpaste_iso.jl`).
const NTHETA = 20
const ω_aspect = 1.0e4

# ── Powers-model volume fractions ──────────────────────────────────────────
f_clin(wc, α) = (1 - α) / (1 + d_clin * wc)
f_w(wc, α) = d_clin * (wc - 0.42 * α) / (1 + d_clin * wc)
f_hyd(wc, α) = 1.42 * d_clin / d_hyd * α / (1 + d_clin * wc)
fh_san(wc, sc) = sc / d_san / (1 / d_clin + wc + sc / d_san)
αmax(wc) = min(1.0, wc / 0.42)

# ── Hydrate foam — multi-bin Self-Consistent ───────────────────────────────
#
# Two starting points for the self-consistent iteration of the needle
# families. `start = :iso` (the default) first solves the foam with ONE
# isotropically averaged needle family, whose Hill tensor in the isotropic
# running estimate is a closed form, and starts the discrete families from that
# estimate: echoes' route, and a few iterations instead of some twenty.
# `start = :direct` starts them from the hydrate stiffness itself. Both converge
# to the same fixed point; their agreement, and that of the isotropic estimate,
# validates the angular discretization.
#
# `sc_tol` are the stopping tolerances of the self-consistent iteration.
const SC_TOL = (; abstol = 1.0e-8)
#
# `μ_b0` is the shear modulus of the FIRST family (θ = 0, axis ‖ ez); it may
# be a `ForwardDiff.Dual` — the derivative of the whole chain with respect to
# it is the strength sensitivity.  All other families stay at
# `μ_hyd_ref`.
# Fractions of hydrates, water and air within the foam.
function foam_fractions(wc, α_p)
    fclin = f_clin(wc, α_p)
    fw = f_w(wc, α_p)
    fhyd = f_hyd(wc, α_p)
    fair = max(0.0, 1 - fclin - fw - fhyd)
    return fhyd / (1 - fclin), fw / (1 - fclin), fair / (1 - fclin)
end

# The foam with one isotropically averaged needle family: a closed form at
# every iteration.
function build_hf_iso(wc, α_p; ω::Real = ω_aspect, sc_tol = SC_TOL)
    fthyd_t, ftw_t, ftair_t = foam_fractions(wc, α_p)
    C_hyd = TensISO{3}(3 * K_hyd_ref, 2 * μ_hyd_ref)
    rve = RVE()
    add_phase!(rve, :CP, Ellipsoid(1.0), Dict(:C => C_hyd); fraction = :rest)
    add_phase!(rve, :HYD, Spheroid(ω), Dict(:C => C_hyd); fraction = fthyd_t, symmetrize = :iso)
    C_tiny = TensISO{3}(3 * TINY, 2 * TINY)
    add_phase!(rve, :W, Ellipsoid(1.0), Dict(:C => C_tiny); fraction = ftw_t)
    add_phase!(rve, :AIR, Ellipsoid(1.0), Dict(:C => C_tiny); fraction = ftair_t)
    return homogenize(rve, SelfConsistent(; sc_tol..., maxiters = 1000), :C; select_best = true)
end

# The starting point of the discrete families, in the element type `T` of the
# chain.
_hf_init(::Val{:iso}, wc, α_p, ω, sc_tol, ::Type{T}) where {T} =
    TensISO{3}(map(x -> convert(T, x), TensND.get_data(build_hf_iso(wc, α_p; ω, sc_tol)))...)
_hf_init(::Val{:direct}, wc, α_p, ω, sc_tol, ::Type) = nothing

function build_hf(
        wc, α_p, μ_b0; N::Int = NTHETA, ω::Real = ω_aspect,
        start::Symbol = :iso, sc_tol = SC_TOL
    )
    fthyd_t, ftw_t, ftair_t = foam_fractions(wc, α_p)

    T = typeof(μ_b0)
    ez = (0.0, 0.0, 1.0)
    rve = RVE(; T = T)
    # Zero-volume matrix phase = SC seed only (Σ inclusion fractions = 1).
    # The spheres (seed, water, air) are not averaged: an isotropic sphere
    # contributes with the symmetry of the reference already, which is
    # transversely isotropic about e₃ here, not isotropic.
    add_phase!(rve, :CP, Ellipsoid(1.0), Dict(:C => TensISO{3}(convert(T, 3 * K_hyd_ref), convert(T, 2 * μ_hyd_ref))); fraction = :rest)
    for (i, bin) in enumerate(polar_orientation_bins(N))
        μ_h = i == 1 ? μ_b0 : convert(T, μ_hyd_ref)
        add_phase!(
            rve, Symbol(:HYD, i),
            Spheroid(ω; euler_angles = (bin.θ, 0.0, 0.0)),
            Dict(:C => TensISO{3}(convert(T, 3 * K_hyd_ref), 2 * μ_h));
            fraction = fthyd_t * bin.weight,
            symmetrize = TISymmetrize(ez)
        )
    end
    C_tiny = TensISO{3}(convert(T, 3 * TINY), convert(T, 2 * TINY))
    add_phase!(
        rve, :W, Ellipsoid(1.0), Dict(:C => C_tiny);
        fraction = ftw_t
    )
    add_phase!(
        rve, :AIR, Ellipsoid(1.0), Dict(:C => C_tiny);
        fraction = ftair_t
    )
    init = _hf_init(Val(start), wc, α_p, ω, sc_tol, T)
    return homogenize(
        rve, SelfConsistent(; init, sc_tol..., maxiters = 1000),
        :C; select_best = true
    )
end

# ── Cement paste : MT(HF, clinker) ─────────────────────────────────────────
function build_cp(wc, α_p, C_hf::TensND.AbstractTens)
    fclin = f_clin(wc, α_p)
    T = eltype(C_hf)
    rve = RVE(; T = T)
    add_phase!(rve, :CP, Ellipsoid(1.0), Dict(:C => C_hf); fraction = :rest)
    add_phase!(
        rve, :CLIN, Ellipsoid(1.0),
        Dict(:C => TensISO{3}(convert(T, 3 * K_clin), convert(T, 2 * μ_clin)));
        fraction = fclin
    )
    return homogenize(rve, MoriTanaka(), :C)
end

# ── Mortar : MT(CP, sand) ──────────────────────────────────────────────────
function build_mo(wc, sc, C_cp::TensND.AbstractTens)
    fsan = fh_san(wc, sc)
    T = eltype(C_cp)
    rve = RVE(; T = T)
    add_phase!(rve, :CP, Ellipsoid(1.0), Dict(:C => C_cp); fraction = :rest)
    add_phase!(
        rve, :SAN, Ellipsoid(1.0),
        Dict(:C => TensISO{3}(convert(T, 3 * K_san), convert(T, 2 * μ_san)));
        fraction = fsan
    )
    return homogenize(rve, MoriTanaka(), :C)
end

# ── Full three-scale chain : scalar μ_b0 in, C_mo array out ────────────────
#
# The converged HF estimate is typed `TensTI{4,T,8}` but is major-symmetric
# up to O(binning + convergence) noise (and so is its derivative w.r.t.
# μ_b0 : perturbing the θ=0 family preserves the TI(ez) major symmetry).
# `best_fit_ti` extracts the physical 5-parameter TI stiffness — echoes does
# the same with `tensor(C_hf.array, TI)` before the CP scale — keeping the
# CP / MO Mori-Tanaka stages on the analytical TI-coaxial Hill branch.
function multiscale_C_mo(wc, α_p, sc, μ_b0; N::Int = NTHETA, ω::Real = ω_aspect, kw...)
    C_hf = best_fit_ti(build_hf(wc, α_p, μ_b0; N = N, ω = ω, kw...), (0.0, 0.0, 1.0))
    C_cp = build_cp(wc, α_p, C_hf)
    C_mo = build_mo(wc, sc, C_cp)
    return Array(components_canon(C_mo))
end

# ── Iso bulk / shear moduli of a (nearly) iso 4-tensor ─────────────────────
# `proj_tens(:ISO, ...)` is TensND's paramsym-style best-fit projection
# (echoes' `.paramsym(sym=ISO)` analog); `k_mu` is MeanFieldHomogenization's
# stiffness-role interpretation of the resulting (α,β) = (3k,2μ) coefficients.
extract_kμ(arr::AbstractArray) = k_mu(TensND.proj_tens(Val(:ISO), arr)[1])

# ── Strength criterion (quasi-brittle model) ───────────────────────────
#
# `M = S_mo : dC : S_mo` (compliance pull-back of the sensitivity), and
# `fc = 1/√(M₃₃₃₃ · 2μ²/f_θ)` where `f_θ` is the volume fraction of the
# perturbed hydrate family (bin θ = 0) in the mortar. `dC` is the derivative
# with respect to 2μ, so that this is `1/√(M₃₃₃₃ · μ²/f_θ)` with ∂/∂μ.
function quasibrittle_strength(
        arr_C_mo::AbstractArray, arr_dC::AbstractArray,
        μh::Real, f_θ::Real
    )
    # Intrinsic TensND algebra: full effective compliance and the double
    # contraction `S ⊡ dC ⊡ S` — no index loops, no iso approximation.
    S = inv(Tens(arr_C_mo))
    M = S ⊡ Tens(arr_dC) ⊡ S
    return 1 / sqrt(abs(M[3, 3, 3, 3]) * 2 * μh^2 / f_θ)
end

# ── One (wc, α) point : elastic moduli + strength ──────────────────────────
#
# The derivative `dC_mo/dμ_b0` is a SINGLE ForwardDiff pass through the
# complete chain (multi-bin SC + MT + MT); value and derivative come from
# one Dual evaluation.
function compute_point(wc, α_p; sc = 0.0, N::Int = NTHETA, ω::Real = ω_aspect, kw...)
    TagT = typeof(ForwardDiff.Tag(multiscale_C_mo, Float64))
    μ_dual = ForwardDiff.Dual{TagT}(float(μ_hyd_ref), 1.0)
    arr_dual = multiscale_C_mo(wc, α_p, sc, μ_dual; N = N, ω = ω, kw...)
    arr_C_mo = ForwardDiff.value.(arr_dual)
    arr_dC_mo = ForwardDiff.partials.(arr_dual, 1)

    K_mo, μ_mo = extract_kμ(arr_C_mo)
    E_mo, _ = E_nu(iso_stiffness(K_mo, μ_mo))

    fhyd_in_mortar = f_hyd(wc, α_p) * (1 - fh_san(wc, sc))
    f_θ = fhyd_in_mortar * polar_orientation_bins(N)[1].weight
    # The criterion is expressed in terms of the iso stiffness parameter
    # `2μ` : `d/d(2μ) = (1/2)·d/dμ`.
    fc = quasibrittle_strength(arr_C_mo, arr_dC_mo ./ 2, μ_hyd_ref, f_θ)
    return (; K_mo, μ_mo, E_mo, fc)
end
