# =============================================================================
#  benchmark_hill_derivative.jl — cross-validation of Hill-tensor derivatives
#  ∂P/∂C between an analytical `hill_derivative` reference and MeanFieldHomogenization
#  (ForwardDiff through `hill_tensor`).
#
#  The reference hand-codes `hill_derivative(ell, C, index, sym[, algo])` per
#  material-symmetry class ; MeanFieldHomogenization obtains the SAME derivative by
#  automatic differentiation of the Hill kernel — for any reference type,
#  including the fully triclinic case a symmetry-typed routine cannot
#  represent.
#
#  Cases (per ellipsoid shape) :
#    * ISO   — ∂P/∂κ (index 0, κ = 3k = 𝕁-coeff) and ∂P/∂η (index 1,
#              η = 2μ = 𝕂-coeff) ; echoes analytical vs MFH ForwardDiff.
#    * ORTHO — the 9 orthotropic parameters ; echoes NUMINT3D vs MFH
#              ForwardDiff (recombined along an example direction).
#    * ANISO — a fully triclinic reference ; MFH ForwardDiff only (echoes'
#              hill_derivative has no triclinic parameterization).
#
#  Elastic 4th order only.  Requires PyCall + an importable `echoes`.
# =============================================================================

import Pkg
Pkg.activate(@__DIR__; io = devnull)

using PyCall
using MeanFieldHomogenization
using TensND
using ForwardDiff
using Printf
using LinearAlgebra

const np = pyimport("numpy")
const echoes = pyimport("echoes")

# ── Conventions shared with benchmark.jl ────────────────────────────────────
# MFH Hill tensor as a 6×6 Kelvin-Mandel matrix, minor-symmetrized (so the
# Dual and Float64 paths agree — `tomandel` on a general Tensor would give
# 9×9). `Core.mandel66_minor` forces the 6×6 form from the component array.
P_KM(ell, C) = MeanFieldHomogenization.Core.mandel66_minor(
    TensND.get_array(change_tens_canon(hill_tensor(ell, C)))
)
to_jlmat(x) = x === nothing ? nothing : convert(Array{Float64, 2}, x)
function relerr(A, B)
    scale = max(maximum(abs, A), maximum(abs, B), 1.0e-300)
    return maximum(abs, A .- B) / scale
end

# echoes ellipsoid (semi-axes a,b,c + Euler angles) — matches MFH
# `Ellipsoid(a, b, c; euler_angles = (θ, φ, ψ))`.
py_ell(a, b, c, θ, φ, ψ) = echoes.ellipsoidal(np.array([a, b, c, θ, φ, ψ]))

# ── Python reference wrappers ───────────────────────────────────────────────
py"""
import numpy as np
import echoes

def hill_deriv_iso(ell, k, mu, index):
    C = echoes.stiff_kmu(float(k), float(mu))
    dP = echoes.hill_derivative(ell, C, index, echoes.ISO,
                                epsrel=1e-8, epsabs=1e-8, maxnb=400000, epsroots=1e-8)
    return np.asarray(dP)

def hill_deriv_ortho(ell, params9_echoes_order, index):
    C = echoes.tensor(list(params9_echoes_order), [0.0, 0.0, 0.0])
    dP = echoes.hill_derivative(ell, C, index, echoes.ORTHO, algo=echoes.NUMINT3D,
                                epsrel=1e-8, epsabs=1e-8, maxnb=400000, epsroots=1e-8)
    return np.asarray(dP)
"""
const py_hill_deriv_iso = py"hill_deriv_iso"
const py_hill_deriv_ortho = py"hill_deriv_ortho"

# =============================================================================
#  ISO reference
# =============================================================================
println("="^80)
println("  Hill derivative ∂P/∂C — echoes (analytical) vs MeanFieldHomogenization (ForwardDiff)")
println("="^80)

const k_iso, μ_iso = 10.0, 10.0
const α0, β0 = 3k_iso, 2μ_iso

iso_shapes = (
    ("sphere (1,1,1)", (1.0, 1.0, 1.0), (0.0, 0.0, 0.0)),
    ("prolate (5,1,1)", (5.0, 1.0, 1.0), (0.0, 0.0, 0.0)),
    ("oblate (5,5,1)", (5.0, 5.0, 1.0), (0.0, 0.0, 0.0)),
    ("triaxial+Euler (3,2.5,1.6)", (3.0, 2.5, 1.6), (0.1, 0.2, 0.3)),
)

n_pass = 0
n_tot = 0
println("\n  ISO reference  k = $k_iso, μ = $μ_iso  (κ = 3k, η = 2μ)")
@printf "  %-28s %-8s %12s %12s   %s\n" "shape" "index" "‖dP‖∞" "rel.err" "pass"
println("  " * "─"^74)
for (label, (a, b, c), (θ, φ, ψ)) in iso_shapes
    ell_jl = Ellipsoid(a, b, c; euler_angles = (θ, φ, ψ))
    ell_py = py_ell(a, b, c, θ, φ, ψ)
    f_κ = κ -> P_KM(ell_jl, TensISO{3}(κ, β0))
    f_η = η -> P_KM(ell_jl, TensISO{3}(α0, η))
    dκ = ForwardDiff.derivative(f_κ, α0)
    dη = ForwardDiff.derivative(f_η, β0)
    for (idx, dP_jl, name) in ((0, dκ, "κ=3k"), (1, dη, "η=2μ"))
        dP_py = to_jlmat(py_hill_deriv_iso(ell_py, k_iso, μ_iso, idx))
        e = relerr(dP_jl, dP_py)
        ok = e < 1.0e-5
        global n_tot += 1; ok && (global n_pass += 1)
        @printf "  %-28s %-8s %12.4e %12.2e   %s\n" label name maximum(abs, dP_jl) e (ok ? "✓" : "✗")
    end
end

# =============================================================================
#  ORTHO reference (cubic-as-orthotropic, 9 parameters)
# =============================================================================
# Julia TensOrtho(C11,C22,C33,C12,C13,C23,μ23,μ13,μ12)
# echoes ORTHO order:  (C11, C12, C13, C22, C23, C33, μ23, μ13, μ12)
const C11c, C12c, C44c = 250.0e3, 100.0e3, 80.0e3
const JL_ORTHO = (C11c, C11c, C11c, C12c, C12c, C12c, C44c, C44c, C44c)
const ECHOES_ORTHO = [C11c, C12c, C12c, C11c, C12c, C11c, C44c, C44c, C44c]
# map: for echoes index e, which Julia argument perturbs it
#   echoes 0→C11(jl1) 1→C12(jl4) 2→C13(jl5) 3→C22(jl2) 4→C23(jl6) 5→C33(jl3)
#         6→μ23(jl7) 7→μ13(jl8) 8→μ12(jl9)
const ECHOES_TO_JL = (1, 4, 5, 2, 6, 3, 7, 8, 9)
const frame = TensND.CanonicalBasis{3, Float64}()
ortho_perturb(i, t) = TensND.TensOrtho(
    ntuple(j -> j == i ? JL_ORTHO[j] + t : Float64(JL_ORTHO[j]), 9)..., frame
)

println("\n  ORTHO reference (cubic C11=$(C11c/1e3),C12=$(C12c/1e3),C44=$(C44c/1e3) GPa), prolate (5,1,1)")
@printf "  %-10s %12s %12s   %s\n" "echoes idx" "‖dP‖∞" "rel.err" "pass"
println("  " * "─"^50)
let ell_jl = Ellipsoid(5.0, 1.0, 1.0), ell_py = py_ell(5.0, 1.0, 1.0, 0.0, 0.0, 0.0)
    for e_idx in 0:8
        jl_i = ECHOES_TO_JL[e_idx + 1]
        dP_jl = try
            ForwardDiff.derivative(t -> P_KM(ell_jl, ortho_perturb(jl_i, t)), 0.0)
        catch err
            @printf "  %-10d  (MFH ForwardDiff through TensOrtho failed: %s)\n" e_idx typeof(err)
            continue
        end
        dP_py = to_jlmat(py_hill_deriv_ortho(ell_py, ECHOES_ORTHO, e_idx))
        dP_py === nothing && (@printf "  %-10d  (echoes FAIL)\n" e_idx; continue)
        e = relerr(dP_jl, dP_py)
        ok = e < 1.0e-4
        global n_tot += 1; ok && (global n_pass += 1)
        @printf "  %-10d %12.4e %12.2e   %s\n" e_idx maximum(abs, dP_jl) e (ok ? "✓" : "✗")
    end
end

# =============================================================================
#  Triclinic reference — MFH ForwardDiff only (no echoes counterpart)
# =============================================================================
println("\n  Fully triclinic reference — MeanFieldHomogenization ForwardDiff only")
println("  (echoes hill_derivative has no triclinic parameterization)")
let ell_jl = Ellipsoid(3.0, 2.0, 1.0; euler_angles = (0.2, 0.3, 0.1))
    C_KM = collect(KM(TensISO{3}(3 * 60.0e3, 2 * 40.0e3)))
    C_KM[1, 1] += 8.0e3; C_KM[1, 4] += 3.0e3; C_KM[4, 1] += 3.0e3   # break symmetry class
    build(t) = begin
        M = C_KM .+ zero(t); M[1, 1] += t     # eltype promotes to typeof(t) (Dual-safe)
        MeanFieldHomogenization.Core.array_from_mandel66(M) |> a -> TensND.Tens(a)
    end
    dP = ForwardDiff.derivative(t -> P_KM(ell_jl, build(t)), 0.0)
    h = 1.0e-2
    fd = (P_KM(ell_jl, build(h)) - P_KM(ell_jl, build(-h))) / (2h)
    @printf "  ∂P/∂C₁₁₁₁ : ‖dP‖∞ = %.4e   AD vs central-FD rel.err = %.2e\n" maximum(abs, dP) relerr(dP, fd)
end

@printf "\n%d / %d echoes-comparable derivatives within tolerance.\n" n_pass n_tot
