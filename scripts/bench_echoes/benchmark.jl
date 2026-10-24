# =============================================================================
#  scripts/bench_echoes/benchmark.jl
#
#  Side-by-side benchmark of MeanFieldHomogenization.jl (Julia) vs Echoes (C++ via
#  PythonCall).
#
#  Sections:
#   § 1  Hill tensor P (4th order, elasticity)
#   § 2  Crack compliance ΔS at ε=1
#   § 3  Hill tensor P (2nd order, conductivity) — echoes.hill dispatches on
#        the rank of the stiffness tensor
#   § 4  Hill derivative ∂P/∂C_{ij} — ForwardDiff (Julia) vs
#        echoes.hill_derivative (Echoes); elasticity only
#   § 5  Summary tables
#
#  Echoes analytical fast path: whenever the stiffness is isotropic,
#  Echoes bypasses the declared algorithm and uses closed-form formulas.
#  For iso cases we therefore display a single "analytical" row (no
#  algorithm name).  Aniso cases keep the RESIDUES + NUMINT3D comparison.
#
#  Echoes must be importable as `import echoes` from the Python
#  interpreter pointed to by PythonCall (see README.md).
#
#  Run from the MeanFieldHomogenization.jl package root:
#    julia --project=scripts/bench_echoes scripts/bench_echoes/benchmark.jl
# =============================================================================

import Pkg
Pkg.activate(@__DIR__; io = devnull)

# Which Python holds Echoes: chosen before PythonCall is loaded
include(joinpath(@__DIR__, "echoes_python.jl"))

using MeanFieldHomogenization
using TensND
using LinearAlgebra
using Printf
using BenchmarkTools
using ForwardDiff
using PythonCall
using JSON

# `jl_method = :decuhr` is benchmarked below (§ 1), and that backend lives in a
# package extension: without these two imports the extension never loads and the
# run dies with "the `:decuhr` backend requires the DECUHR extension". Both are
# dependencies of this environment (`Project.toml`), DECUHR taken from the
# sibling checkout like TensND. Same requirement as `test/runtests.jl`.
import DECUHR, Integrals

BenchmarkTools.DEFAULT_PARAMETERS.samples = 20
BenchmarkTools.DEFAULT_PARAMETERS.seconds = 1.0

# ─── Echoes side: `kernels_echoes.py`, next to this file ─────────────────────
#
# The media, the cases and the Echoes calls are defined there once; run on its
# own, that module times the Echoes side alone.

pyimport("sys").path.insert(0, @__DIR__)
include(joinpath(@__DIR__, "pyjl.jl"))
const KE = pyimport("kernels_echoes")
const py_hill = KE.py_hill
const py_crack = KE.py_crack
const py_crack_ell = KE.py_crack_ell
const py_hill_derivative_iso = KE.py_hill_derivative_iso
const py_hill_derivative_ortho = KE.py_hill_derivative_ortho

# ─── Helpers ─────────────────────────────────────────────────────────────────

"""
    km_index_to_linear(i, j) -> Int

Convert a (i, j) Kelvin–Mandel upper-triangle pair (1 ≤ i ≤ j ≤ 6) to
the 0-based linear index used by Echoes' 21-component stiffness
constructor and by `hill_derivative(..., index)`.
"""
function km_index_to_linear(i::Int, j::Int)
    @assert 1 ≤ i ≤ j ≤ 6
    offsets = (0, 6, 11, 15, 18, 20)
    return offsets[i] + (j - i)
end

"""
    compare(A, B) -> (maxabs, maxrel)

Componentwise max absolute / max relative error between two arrays of
identical shape.  Relative error is normalized by
`max(|A|, |B|, 1e-300)`.
"""
function compare(A::AbstractArray, B::AbstractArray)
    @assert size(A) == size(B)
    scale = max(maximum(abs, A), maximum(abs, B), 1.0e-300)
    maxabs = maximum(abs.(A .- B))
    return (maxabs, maxabs / scale)
end

to_jlmat(x) = pymat(x)

fmt_time(t) =
    t < 1.0e-6 ? (@sprintf "%7.2f ns" (t * 1.0e9)) :
    t < 1.0e-3 ? (@sprintf "%7.2f µs" (t * 1.0e6)) :
    t < 1.0 ? (@sprintf "%7.2f ms" (t * 1.0e3)) :
    (@sprintf "%7.2f  s" t)

# ─── Reference matrices ──────────────────────────────────────────────────────

# 1. Isotropic steel: E = 210 GPa, ν = 0.3
const E_iso = pyconvert(Float64, KE.E_ISO)
const ν_iso = pyconvert(Float64, KE.NU_ISO)
const λ_iso = E_iso * ν_iso / ((1 + ν_iso) * (1 - 2ν_iso))
const μ_iso = E_iso / (2 * (1 + ν_iso))
const k_iso = λ_iso + 2μ_iso / 3
const C_iso = TensISO{3}(3k_iso, 2μ_iso)

# 2. Cubic crystal: C11=250, C12=100, C44=80 GPa
const C11c, C12c, C44c = pyconvert.(Float64, (KE.C11C, KE.C12C, KE.C44C))
const C_cubic = TensND.TensOrtho(
    C11c, C11c, C11c, C12c, C12c, C12c, C44c, C44c, C44c,
    TensND.CanonicalBasis{3, Float64}(),
)

# 3. Triclinic
const C_tric_KM = pyconvert(Matrix{Float64}, KE.C_TRIC_KM)
const C_tric = from_mandel(C_tric_KM, CanonicalBasis{3, Float64}())

const MEDIA_PY = KE.MEDIA
const C_iso_py = MEDIA_PY["iso"]
const C_cubic_py = MEDIA_PY["cubic"]
const C_tric_py = MEDIA_PY["triclinic"]

# 4. Conductivity tensors (2nd order)
const k_cond = pyconvert(Float64, KE.K_COND)
const K_iso = TensISO{3}(k_cond)
const K_iso_mat = [k_cond 0.0 0.0; 0.0 k_cond 0.0; 0.0 0.0 k_cond]
const K_iso_py = MEDIA_PY["K iso"]

# Anisotropic conductivity: rotation of diag(100, 20, 50) by π/4 in 1-2 plane
const K_aniso_mat = pyconvert(Matrix{Float64}, KE.K_ANISO_MAT)
const K_aniso = TensND.Tens(Array{Float64, 2}(K_aniso_mat))
const K_aniso_py = MEDIA_PY["K aniso"]

# =============================================================================
# Running storage for the summary tables
# =============================================================================

hill_rows = NamedTuple[]
crack_rows = NamedTuple[]
hill2_rows = NamedTuple[]
dhill_rows = NamedTuple[]

# =============================================================================
#  § 1  HILL TENSOR P  (elasticity, 4th order)
# =============================================================================

println("="^78)
println("  § 1  HILL TENSOR P  (elasticity, 4th order)")
println("="^78)

"""
    bench_hill(label, ell_jl, C_jl, C_py; jl_method = :auto,
               is_iso = false, echoes_algos = ("RESIDUES", "NUMINT3D"))

Compute P with MeanFieldHomogenization (one call), then with Echoes.  When
`is_iso = true`, a single "analytical" row is printed (Echoes uses the
closed-form path regardless of the declared algorithm).  Otherwise only
the algorithms listed in `echoes_algos` are tested.  Pass
`echoes_algos = ("NUMINT3D",)` for non-triclinic matrices where the
acoustic polynomial has quasi-multiple roots that break RESIDUES.
"""
function bench_hill(
        label, ell_jl, C_jl, C_py;
        jl_method = :auto, is_iso = false,
        echoes_algos = ("RESIDUES", "NUMINT3D"),
        euler_angles = (0.0, 0.0, 0.0)
    )
    a, b, c = ell_jl.semi_axes
    θ_ea, φ_ea, ψ_ea = euler_angles

    P_jl = hill_tensor(ell_jl, C_jl; method = jl_method)
    P_jl_KM = mandel(change_tens_canon(P_jl))
    tJ = @belapsed hill_tensor($ell_jl, $C_jl; method = $jl_method)

    println()
    println("─"^78)
    @printf "  Case: %s\n" label
    @printf "    semi-axes = (%.3g, %.3g, %.3g)   Julia method = :%s\n" a b c jl_method
    @printf "    t(Julia) = %s\n" fmt_time(tJ)

    if is_iso
        P_py = to_jlmat(
            py_hill(
                a, b, c, C_py, "DEFAULT";
                theta = θ_ea, phi = φ_ea, psi = ψ_ea
            )
        )
        if P_py === nothing
            @printf "    Echoes (analytical) : FAIL (C++ exception)\n"
            push!(
                hill_rows, (
                    label = label, algo = "analytical", status = "FAIL",
                    maxabs = NaN, maxrel = NaN,
                    tJ = tJ, tE = NaN, ratio = NaN,
                )
            )
        else
            maxabs, maxrel = compare(P_jl_KM, P_py)
            tE = @belapsed py_hill(
                $a, $b, $c, $C_py, "DEFAULT";
                theta = $θ_ea, phi = $φ_ea, psi = $ψ_ea
            )
            @printf "    Echoes (analytical) : max|Δ| = %.3e  ε_rel = %.3e  t = %s  ratio E/J = %.2f×\n" maxabs maxrel fmt_time(tE) (tE / tJ)
            push!(
                hill_rows, (
                    label = label, algo = "analytical", status = "OK",
                    maxabs = maxabs, maxrel = maxrel,
                    tJ = tJ, tE = tE, ratio = tE / tJ,
                )
            )
        end
        return nothing
    end

    for algo in echoes_algos
        P_py = to_jlmat(
            py_hill(
                a, b, c, C_py, algo;
                theta = θ_ea, phi = φ_ea, psi = ψ_ea
            )
        )
        if P_py === nothing
            @printf "    Echoes %s : FAIL (C++ exception)\n" algo
            push!(
                hill_rows, (
                    label = label, algo = algo, status = "FAIL",
                    maxabs = NaN, maxrel = NaN,
                    tJ = tJ, tE = NaN, ratio = NaN,
                )
            )
            continue
        end
        maxabs, maxrel = compare(P_jl_KM, P_py)
        tE = @belapsed py_hill(
            $a, $b, $c, $C_py, $algo;
            theta = $θ_ea, phi = $φ_ea, psi = $ψ_ea
        )
        @printf "    Echoes %-8s : max|Δ| = %.3e  ε_rel = %.3e  t = %s  ratio E/J = %.2f×\n" algo maxabs maxrel fmt_time(tE) (tE / tJ)
        push!(
            hill_rows, (
                label = label, algo = algo, status = "OK",
                maxabs = maxabs, maxrel = maxrel,
                tJ = tJ, tE = tE, ratio = tE / tJ,
            )
        )
    end
    return nothing
end

# The cases of `kernels_echoes.HILL_CASES`, the media by name on both sides.
const MEDIA_JL = Dict(
    "iso" => C_iso, "cubic" => C_cubic, "triclinic" => C_tric,
    "K iso" => K_iso, "K aniso" => K_aniso,
)
_ellipsoid(c) = Ellipsoid(Float64.(c["axes"])...; euler_angles = Tuple(Float64.(c["euler"])))
_is_closed_form(c) = c["algos"] == ["DEFAULT"]

for c in pyjl(KE.HILL_CASES)
    bench_hill(
        c["label"], _ellipsoid(c), MEDIA_JL[c["medium"]], MEDIA_PY[c["medium"]];
        jl_method = Symbol(c["jl_method"]), is_iso = _is_closed_form(c),
        echoes_algos = Tuple(c["algos"]), euler_angles = Tuple(Float64.(c["euler"]))
    )
end

# =============================================================================
#  § 2  CRACK COMPLIANCE CONTRIBUTION H  (size-independent, Echoes convention)
# =============================================================================

println()
println("="^78)
println("  § 2  CRACK COMPLIANCE CONTRIBUTION H — penny-crack")
println("="^78)

# CONVENTION — read before comparing any H across the two libraries.
#
# Both libraries build H from the SAME COD tensor B (normalized by the in-plane
# half-width b), but they normalize the flat limit differently:
#
#   MeanFieldHomogenization :  H = lim (c/b) Q⁻¹ = (3/4)  n̂ ⊗ˢ B ⊗ˢ n̂   (ellipse)
#                                     = (2/π)  n̂ ⊗ˢ B ⊗ˢ n̂   (ribbon)
#   Echoes       :  H = lim (c/a) Q⁻¹ = (3η/4) n̂ ⊗ˢ B ⊗ˢ n̂   (ellipse)
#                                     = (2/π)  n̂ ⊗ˢ B ⊗ˢ n̂   (ribbon)
#
# so for an elliptic crack of in-plane aspect ratio η = b/a:
#
#   H_echoes = η · H_MeanFieldHomogenization        (ellipse)
#   H_echoes =     H_MeanFieldHomogenization        (ribbon — normalized by b in both)
#
# Verified against Echoes on 2026-07-25 at η = 0.7, 0.5, 0.3, 0.1: the ratio is
# η to four decimals. The two coincide at η = 1, which is why the penny-crack
# comparison below needs NO rescaling — but any elliptic comparison does. The
# η ≠ 1 case is benchmarked separately in `bench_crack_elliptic` further down.
# See docs/src/theory/cod_tensors.md § Conventions.

function bench_crack(
        label, C_jl, C_py; jl_method = :auto, is_iso = false,
        echoes_algos = ("RESIDUES", "NUMINT3D")
    )
    crack = PennyCrack(1.0)

    H_jl = inverse_property_contribution(crack, C_jl; method = jl_method)
    H_jl_KM = mandel(change_tens_canon(H_jl))
    tJ = @belapsed inverse_property_contribution(
        $crack, $C_jl;
        method = $jl_method
    )

    println()
    println("─"^78)
    @printf "  Case: %s   Julia method = :%s\n" label jl_method
    @printf "    t(Julia) = %s\n" fmt_time(tJ)

    if is_iso
        # Penny iso analytical: B_nn = 16(1-ν²)/(3πE) and H = (3/4) B ⊗ˢ ...
        # The (3,3,3,3) slot: H[3,3,3,3] = (3/4) * B_nn = 4(1-ν²)/(πE).
        H3333_th = 4 * (1 - ν_iso^2) / (π * E_iso)
        @printf "    Analytical H[3,3,3,3] (Echoes convention) = %.4e\n" H3333_th
        @printf "      MeanFieldHomogenization H[3,3,3,3] = %.4e   err = %.2e\n" H_jl[3, 3, 3, 3] abs(H_jl[3, 3, 3, 3] - H3333_th)

        H_py = to_jlmat(py_crack(C_py, "DEFAULT"))
        if H_py === nothing
            @printf "    Echoes (analytical) : FAIL (C++ exception)\n"
            push!(
                crack_rows, (
                    label = label, algo = "analytical", status = "FAIL",
                    maxabs = NaN, maxrel = NaN,
                    tJ = tJ, tE = NaN, ratio = NaN,
                )
            )
        else
            @printf "      Echoes (analytical) H(KM)[3,3] = %.4e   err vs analytical = %.2e\n" H_py[3, 3] abs(H_py[3, 3] - H3333_th)
            maxabs, maxrel = compare(H_jl_KM, H_py)
            tE = @belapsed py_crack($C_py, "DEFAULT")
            @printf "    Echoes (analytical) : max|Δ| = %.3e  ε_rel = %.3e  t = %s  ratio E/J = %.2f×\n" maxabs maxrel fmt_time(tE) (tE / tJ)
            push!(
                crack_rows, (
                    label = label, algo = "analytical", status = "OK",
                    maxabs = maxabs, maxrel = maxrel,
                    tJ = tJ, tE = tE, ratio = tE / tJ,
                )
            )
        end
        return nothing
    end

    for algo in echoes_algos
        H_py = to_jlmat(py_crack(C_py, algo))
        if H_py === nothing
            @printf "    Echoes %-8s : FAIL (C++ exception)\n" algo
            push!(
                crack_rows, (
                    label = label, algo = algo, status = "FAIL",
                    maxabs = NaN, maxrel = NaN,
                    tJ = tJ, tE = NaN, ratio = NaN,
                )
            )
            continue
        end
        maxabs, maxrel = compare(H_jl_KM, H_py)
        tE = @belapsed py_crack($C_py, $algo)
        @printf "    Echoes %-8s : max|Δ| = %.3e  ε_rel = %.3e  t = %s  ratio E/J = %.2f×\n" algo maxabs maxrel fmt_time(tE) (tE / tJ)
        push!(
            crack_rows, (
                label = label, algo = algo, status = "OK",
                maxabs = maxabs, maxrel = maxrel,
                tJ = tJ, tE = tE, ratio = tE / tJ,
            )
        )
    end
    return nothing
end

for c in pyjl(KE.CRACK_CASES)
    bench_crack(
        c["label"], MEDIA_JL[c["medium"]], MEDIA_PY[c["medium"]];
        jl_method = Symbol(c["jl_method"]), is_iso = _is_closed_form(c),
        echoes_algos = Tuple(c["algos"])
    )
end

# -----------------------------------------------------------------------------
#  § 2b  ELLIPTIC crack, η ≠ 1 — the case that discriminates the two
#        normalization conventions (see the note at the top of § 2).
#
#  The penny comparisons above cannot detect a wrong η-dependence, because at
#  η = 1 the MeanFieldHomogenization (uniform-b) and Echoes (a-normalized) conventions
#  coincide. This sweep applies the documented factor explicitly and checks
#  that the two libraries then agree:
#
#        H_echoes  ==  η · H_MeanFieldHomogenization
# -----------------------------------------------------------------------------

function bench_crack_elliptic(C_jl, C_py; ωflat = pyconvert(Float64, KE.ELLIPTIC_CRACK_FLATNESS), ηs = Tuple(pyconvert(Vector{Float64}, KE.ELLIPTIC_CRACK_ETAS)))
    println()
    println("─"^78)
    println("  § 2b  Elliptic crack — convention check  H_echoes = η · H_MFH")
    println("─"^78)
    @printf "  %-7s %14s %14s %10s %10s %8s\n" "η" "H_echoes" "H_MFH" "ratio" "expected" "status"

    worst = 0.0
    for η in ηs
        H_py = to_jlmat(py_crack_ell(C_py, 1.0, η, ωflat))
        if H_py === nothing
            @printf "  %-7.3f %14s %14s %10s %10.4f %8s\n" η "FAIL" "-" "-" η "SKIP"
            continue
        end

        H_jl = inverse_property_contribution(EllipticCrack(1.0, η), C_jl)
        r = H_py[3, 3] / H_jl[3, 3, 3, 3]
        err = abs(r - η) / η
        worst = max(worst, err)
        @printf "  %-7.3f %14.6e %14.6e %10.4f %10.4f %8s\n" η H_py[3, 3] H_jl[3, 3, 3, 3] r η (
            err < 1.0e-3 ? "OK" : "MISMATCH"
        )
    end
    println()
    @printf "  worst relative deviation from the expected η factor: %.2e\n" worst
    println(
        worst < 1.0e-3 ?
            "  → conventions reconcile exactly; see docs/src/theory/cod_tensors.md" :
            "  → CONVENTION MISMATCH: do not trust elliptic H across the two libraries"
    )
    return worst
end

bench_crack_elliptic(C_iso, C_iso_py)

# =============================================================================
#  § 3  HILL TENSOR P  (conductivity, 2nd order)
# =============================================================================

println()
println("="^78)
println("  § 3  HILL TENSOR P  (conductivity, 2nd order)")
println("="^78)

order2_as_matrix(P) = Float64[P[i, j] for i in 1:3, j in 1:3]

function bench_hill_order2(
        label, ell_jl, K_jl, K_py; is_iso = false,
        echoes_algos = ("RESIDUES", "NUMINT3D"),
        euler_angles = (0.0, 0.0, 0.0)
    )
    a, b, c = ell_jl.semi_axes
    θ_ea, φ_ea, ψ_ea = euler_angles

    P_jl = hill_tensor(ell_jl, K_jl)
    P_jl_mat = order2_as_matrix(change_tens_canon(P_jl))
    tJ = @belapsed hill_tensor($ell_jl, $K_jl)

    println()
    println("─"^78)
    @printf "  Case: %s\n" label
    @printf "    semi-axes = (%.3g, %.3g, %.3g)\n" a b c
    @printf "    t(Julia) = %s\n" fmt_time(tJ)

    if is_iso
        # Sanity check: sphere → P = I/(3k)
        if a == b == c == 1.0
            expected = 1 / (3 * k_cond)
            err = maximum(abs(P_jl_mat[i, i] - expected) for i in 1:3)
            @printf "    Sphere/iso analytical P[i,i] = 1/(3k) = %.6e  err(J) = %.2e\n" expected err
        end

        P_py = to_jlmat(
            py_hill(
                a, b, c, K_py, "DEFAULT";
                theta = θ_ea, phi = φ_ea, psi = ψ_ea
            )
        )
        if P_py === nothing
            @printf "    Echoes (analytical) : FAIL (C++ exception)\n"
            push!(
                hill2_rows, (
                    label = label, algo = "analytical", status = "FAIL",
                    maxabs = NaN, maxrel = NaN,
                    tJ = tJ, tE = NaN, ratio = NaN,
                )
            )
        else
            maxabs, maxrel = compare(P_jl_mat, P_py)
            tE = @belapsed py_hill(
                $a, $b, $c, $K_py, "DEFAULT";
                theta = $θ_ea, phi = $φ_ea, psi = $ψ_ea
            )
            @printf "    Echoes (analytical) : max|Δ| = %.3e  ε_rel = %.3e  t = %s  ratio E/J = %.2f×\n" maxabs maxrel fmt_time(tE) (tE / tJ)
            push!(
                hill2_rows, (
                    label = label, algo = "analytical", status = "OK",
                    maxabs = maxabs, maxrel = maxrel,
                    tJ = tJ, tE = tE, ratio = tE / tJ,
                )
            )
        end
        return nothing
    end

    for algo in echoes_algos
        P_py = to_jlmat(
            py_hill(
                a, b, c, K_py, algo;
                theta = θ_ea, phi = φ_ea, psi = ψ_ea
            )
        )
        if P_py === nothing
            @printf "    Echoes %s : FAIL (C++ exception)\n" algo
            push!(
                hill2_rows, (
                    label = label, algo = algo, status = "FAIL",
                    maxabs = NaN, maxrel = NaN,
                    tJ = tJ, tE = NaN, ratio = NaN,
                )
            )
            continue
        end
        maxabs, maxrel = compare(P_jl_mat, P_py)
        tE = @belapsed py_hill(
            $a, $b, $c, $K_py, $algo;
            theta = $θ_ea, phi = $φ_ea, psi = $ψ_ea
        )
        @printf "    Echoes %-8s : max|Δ| = %.3e  ε_rel = %.3e  t = %s  ratio E/J = %.2f×\n" algo maxabs maxrel fmt_time(tE) (tE / tJ)
        push!(
            hill2_rows, (
                label = label, algo = algo, status = "OK",
                maxabs = maxabs, maxrel = maxrel,
                tJ = tJ, tE = tE, ratio = tE / tJ,
            )
        )
    end
    return nothing
end

for c in pyjl(KE.COND_CASES)
    bench_hill_order2(
        c["label"], _ellipsoid(c), MEDIA_JL[c["medium"]], MEDIA_PY[c["medium"]];
        is_iso = _is_closed_form(c), echoes_algos = Tuple(c["algos"]),
        euler_angles = Tuple(Float64.(c["euler"]))
    )
end

# =============================================================================
#  § 4  HILL DERIVATIVE ∂P/∂θ  (elasticity only)
#
#  Echoes' hill_derivative requires a concrete material-symmetry class
#  (ISO, TI, ORTHO, …) and segfaults on UNDEFSYM/triclinic parameterization.
#  We therefore compare against Echoes on the *natural* parameter set of
#  each symmetry class:
#
#   - ISO   : 2 params  (κ = 3k, η = 2μ)          — tested on 3 geometries
#   - ORTHO : 9 params  (C11, C22, C33, C12, C13, C23, μ₂₃, μ₁₃, μ₁₂)
#             Echoes order: (C11, C12, C13, C22, C23, C33, μ₂₃, μ₁₃, μ₁₂)
#
#  The triclinic case is tested on the Julia side only (no Echoes
#  counterpart).
# =============================================================================

println()
println("="^78)
println("  § 4  HILL DERIVATIVE ∂P/∂θ  —  ForwardDiff vs Echoes")
println("="^78)

# Build an iso stiffness from its 2 Walpole parameters (α=3k, β=2μ)
# in a way that supports ForwardDiff.Dual for α and β.
iso_from_walpole(α, β) = TensND.TensISO{3}(α, β)

# Indices map between `TensOrtho(C11, C22, C33, C12, C13, C23, μ23, μ13, μ12)` and
# Echoes' ORTHO parameter order `(C11, C12, C13, C22, C23, C33, μ23, μ13, μ12)`:
# `JL_TO_ECHOES_ORTHO[k]` is the Echoes index of the k-th Julia argument.
const JL_TO_ECHOES_ORTHO = Tuple(pyconvert(Vector{Int}, KE.JL_TO_ECHOES_ORTHO))
const ORTHO_LABELS = ("C11", "C22", "C33", "C12", "C13", "C23", "μ₂₃", "μ₁₃", "μ₁₂")

# ─── 4.1 — ISO cases (derive w.r.t. κ = 3k, η = 2μ) ───────────────────────

function bench_hill_derivative_iso(label, ell_jl)
    a, b, c = ell_jl.semi_axes
    α₀, β₀ = 3k_iso, 2μ_iso

    println()
    println("─"^78)
    @printf "  ISO Case: %s\n" label

    # Julia ForwardDiff wrt κ (= α)
    f_α = α -> mandel(change_tens_canon(hill_tensor(ell_jl, iso_from_walpole(α, β₀))))
    f_β = β -> mandel(change_tens_canon(hill_tensor(ell_jl, iso_from_walpole(α₀, β))))
    dP_jl_κ = ForwardDiff.derivative(f_α, α₀)
    dP_jl_η = ForwardDiff.derivative(f_β, β₀)
    tJ_κ = @belapsed ForwardDiff.derivative($f_α, $α₀)

    for (idx, label_idx, dP_jl, tJ) in (
            (0, "κ=3k", dP_jl_κ, tJ_κ),
            (1, "η=2μ", dP_jl_η, tJ_κ),
        )
        dP_py = to_jlmat(py_hill_derivative_iso(a, b, c, k_iso, μ_iso, idx))
        if dP_py === nothing
            @printf "    Echoes (analytical, %-5s) : FAIL\n" label_idx
            push!(
                dhill_rows, (
                    label = string(label, " | ", label_idx),
                    algo = "ISO", status = "FAIL",
                    maxabs = NaN, maxrel = NaN,
                    tJ = tJ, tE = NaN, ratio = NaN,
                )
            )
            continue
        end
        maxabs, maxrel = compare(dP_jl, dP_py)
        tE = @belapsed py_hill_derivative_iso($a, $b, $c, $k_iso, $μ_iso, $idx)
        @printf "    Echoes (analytical, %-5s) : max|Δ| = %.3e  ε_rel = %.3e  t = %s  ratio E/J = %.2f×\n" label_idx maxabs maxrel fmt_time(tE) (tE / tJ)
        push!(
            dhill_rows, (
                label = string(label, " | ", label_idx),
                algo = "ISO", status = "OK",
                maxabs = maxabs, maxrel = maxrel,
                tJ = tJ, tE = tE, ratio = tE / tJ,
            )
        )
    end
    return nothing
end

for c in pyjl(KE.DERIV_ISO_CASES)
    bench_hill_derivative_iso(c["label"], Ellipsoid(Float64.(c["axes"])...))
end

# ─── 4.2 — ORTHO case (cubic crystal, prolate, 9 params) ──────────────────

# Cubic-as-ORTHO Julia params (Julia order)
const JL_ORTHO_BASE = (
    C11c, C11c, C11c,       # C11, C22, C33
    C12c, C12c, C12c,       # C12, C13, C23
    C44c, C44c, C44c,
)       # μ23, μ13, μ12

# Corresponding echoes-order params [C11, C12, C13, C22, C23, C33, μ23, μ13, μ12]
const ECHOES_ORTHO_BASE = pyconvert(Vector{Float64}, KE.ECHOES_ORTHO_BASE)
# The same parameters as a Python list, built once so that no conversion enters the timing
const ECHOES_ORTHO_BASE_PY = pylist(ECHOES_ORTHO_BASE)

function ortho_from_params(p::NTuple{9, T}) where {T <: Number}
    return TensND.TensOrtho(p..., TensND.CanonicalBasis{3, T}())
end

function dP_dortho_forwarddiff(
        ell, base::NTuple{9, Float64}, jl_idx::Int;
        method::Symbol
    )
    f = δ -> begin
        z = zero(δ)
        p = ntuple(k -> k == jl_idx ? base[k] + δ : base[k] + z, 9)
        C = ortho_from_params(p)
        return mandel(change_tens_canon(hill_tensor(ell, C; method = method)))
    end
    return ForwardDiff.derivative(f, 0.0)
end

function bench_hill_derivative_ortho(
        label, ell_jl;
        echoes_algos = ("RESIDUES", "NUMINT3D")
    )
    a, b, c = ell_jl.semi_axes

    println()
    println("─"^78)
    @printf "  ORTHO Case: %s\n" label

    for jl_idx in 1:9
        echoes_idx = JL_TO_ECHOES_ORTHO[jl_idx]
        comp_label = ORTHO_LABELS[jl_idx]

        dP_jl = dP_dortho_forwarddiff(
            ell_jl, JL_ORTHO_BASE, jl_idx;
            method = :decuhr
        )
        tJ = @belapsed dP_dortho_forwarddiff(
            $ell_jl, $JL_ORTHO_BASE, $jl_idx;
            method = :decuhr
        )

        for algo in echoes_algos
            dP_py = to_jlmat(
                py_hill_derivative_ortho(
                    a, b, c,
                    ECHOES_ORTHO_BASE_PY,
                    echoes_idx, algo
                )
            )
            if dP_py === nothing
                @printf "    %-4s Echoes %-8s : FAIL\n" comp_label algo
                push!(
                    dhill_rows, (
                        label = string(label, " | ", comp_label),
                        algo = algo, status = "FAIL",
                        maxabs = NaN, maxrel = NaN,
                        tJ = tJ, tE = NaN, ratio = NaN,
                    )
                )
                continue
            end
            maxabs, maxrel = compare(dP_jl, dP_py)
            tE = @belapsed py_hill_derivative_ortho(
                $a, $b, $c,
                $ECHOES_ORTHO_BASE_PY,
                $echoes_idx, $algo
            )
            @printf "    %-4s Echoes %-8s : max|Δ| = %.3e  ε_rel = %.3e  t = %s  ratio E/J = %.2f×\n" comp_label algo maxabs maxrel fmt_time(tE) (tE / tJ)
            push!(
                dhill_rows, (
                    label = string(label, " | ", comp_label),
                    algo = algo, status = "OK",
                    maxabs = maxabs, maxrel = maxrel,
                    tJ = tJ, tE = tE, ratio = tE / tJ,
                )
            )
        end
    end
    return nothing
end

for c in pyjl(KE.DERIV_ORTHO_CASES)
    bench_hill_derivative_ortho(c["label"], Ellipsoid(Float64.(c["axes"])...); echoes_algos = Tuple(c["algos"]))
end

# ─── 4.3 — Triclinic Julia-only (no Echoes counterpart) ───────────────────

println()
println("─"^78)
println("  Triclinic case — Julia ForwardDiff only (Echoes hill_derivative")
println("                    segfaults on UNDEFSYM / 21-param stiffness)")
println("─"^78)

let ell = Ellipsoid(2.0, 1.0, 0.5), basis = CanonicalBasis{3, Float64}()
    # Perturb C_tric Kelvin-Mandel entry (1,1) and derive via :decuhr
    function dP_dC_ij(i, j)
        f = δ -> begin
            T = promote_type(Float64, typeof(δ))
            C = Matrix{T}(undef, 6, 6)
            for p in 1:6, q in 1:6
                C[p, q] = C_tric_KM[p, q]
            end
            C[i, j] += δ
            if i != j
                C[j, i] += δ
            end
            return mandel(change_tens_canon(hill_tensor(ell, from_mandel(C, basis); method = :decuhr)))
        end
        return ForwardDiff.derivative(f, 0.0)
    end
    dP_11 = dP_dC_ij(1, 1)
    t_tric = @belapsed $dP_dC_ij(1, 1)
    @printf "  ForwardDiff ∂P/∂C(1,1) on triclinic triaxial (:decuhr):\n"
    @printf "    max entry = %.3e   t(1 index) = %s\n" maximum(abs, dP_11) fmt_time(t_tric)
end

# =============================================================================
#  § 5  SUMMARY TABLES
# =============================================================================

println()
println("="^78)
println("  § 5  SUMMARY")
println("="^78)

function print_summary(title, rows)
    println()
    println(title)
    println("─"^96)
    @printf "  %-34s  %-11s  %6s  %10s  %10s  %11s  %11s  %9s\n" "case" "algo(E)" "status" "max |Δ|" "max ε_rel" "t_J" "t_E" "E/J"
    println("─"^96)
    for r in rows
        if r.status == "OK"
            @printf "  %-34s  %-11s  %6s  %10.2e  %10.2e  %11s  %11s  %9.1e\n" r.label r.algo r.status r.maxabs r.maxrel fmt_time(r.tJ) fmt_time(r.tE) r.ratio
        else
            @printf "  %-34s  %-11s  %6s  %10s  %10s  %11s  %11s  %9s\n" r.label r.algo r.status "—" "—" fmt_time(r.tJ) "—" "—"
        end
    end
    return
end

print_summary("Hill tensor P (elasticity, 4th order)", hill_rows)
print_summary("Crack compliance ΔS (ε=1)", crack_rows)
print_summary("Hill tensor P (conductivity, 2nd order)", hill2_rows)
print_summary("Hill derivative ∂P/∂C (elasticity, 4th order)", dhill_rows)

println()
println("="^78)

# ─── Results file, read by the "Performance vs Echoes" page ──────────────────
_jsonrow(r) = Dict(string(k) => (v isa AbstractFloat && !isfinite(v) ? nothing : v) for (k, v) in pairs(r))
let path = joinpath(@__DIR__, "figures", "benchmark_kernels.json")
    mkpath(dirname(path))
    out = Dict(
        "generator" => "scripts/bench_echoes/benchmark.jl (Echoes side: kernels_echoes.py)",
        "julia" => string(VERSION), "mfh" => string(pkgversion(MeanFieldHomogenization)),
        "tensnd" => string(pkgversion(TensND)), "cpu" => Sys.cpu_info()[1].model,
        "threads" => Threads.nthreads(),
        "protocol" => "BenchmarkTools @belapsed minimum of 20 samples, one thread each; Echoes through PythonCall",
        "hill" => map(_jsonrow, hill_rows), "crack" => map(_jsonrow, crack_rows),
        "conduction" => map(_jsonrow, hill2_rows), "derivative" => map(_jsonrow, dhill_rows),
    )
    write(path, JSON.json(out; pretty = true))
    println("wrote ", path)
end
println("=== bench done benchmark.jl")
