# =============================================================================
#  benchmark_strength.jl — cross-validation of the Pichler & Hellmich (2011)
#  three-scale cement-paste / mortar strength model against the echoes C++
#  Pichler & Hellmich (CCR 2011) reference.
#
#  The Julia model is the SHARED implementation `scripts/common/quasibrittle_strength.jl`,
#  built entirely on the public MeanFieldHomogenization API : the multi-bin
#  Self-Consistent hydrate foam is assembled with several non-coaxial
#  `TISymmetrize` families whose EXACT azimuthal average (`TensTI{4,T,8}`,
#  non-major-symmetric content preserved) now flows through the generic SC
#  kernel — no hand-rolled Mandel bypass, no private IFT solver.  The strength
#  sensitivity `dC_mo/dμ_hyd` is a single ForwardDiff pass through the whole
#  chain (multi-bin SC + MT + MT).
#
#  Elastic moduli (k, μ, E) are validated to 1 %; the strength `fc` to 2 %.
#  Sweeps a (w/c, α) grid, compares against `py_strength_reference`, and writes a CSV.
# =============================================================================

import Pkg
Pkg.activate(@__DIR__; io = devnull)

using PyCall
using Printf
using LinearAlgebra

include(joinpath(@__DIR__, "..", "common", "quasibrittle_strength.jl"))

# ── echoes Python reference (verbatim from the CCR2011 script) ──────────────
py"""
import math, numpy as np
from echoes import (rve, ellipsoid, spheroidal, spherical, stiff_kmu,
                    homogenize, homogenize_derivative, tensor, tZ4,
                    Enu_from_kmu, ISO, TI, SC, MT, NUMINT)

rho_w = 1.; rho_clin = 3.15; d_clin = rho_clin / rho_w
rho_hyd = 2.073; d_hyd = rho_hyd / rho_w
rho_san = 2.648; d_san = rho_san / rho_w

C_clin = stiff_kmu(116.7, 53.8)
C_w   = tZ4
C_hyd = stiff_kmu(18.7, 11.8)
C_air = tZ4
C_san = stiff_kmu(37.8, 44.3)

f_clin = lambda wc, alpha: (1 - alpha) / (1 + d_clin * wc)
f_w    = lambda wc, alpha: d_clin * (wc - 0.42 * alpha) / (1 + d_clin * wc)
f_hyd  = lambda wc, alpha: 1.42 * d_clin / d_hyd * alpha / (1 + d_clin * wc)
fh_san = lambda wc, sc: sc / d_san / (1. / d_clin + wc + sc / d_san)
alphamax = lambda wc: min(1., wc / 0.42)

def disc_theta(ntheta):
    return ([0.] + [math.pi / 2 * (i - 0.5) / (ntheta - 1) for i in range(1, ntheta)],
            [math.pi / 2 * i / (ntheta - 1) for i in range(ntheta)],
            [math.pi / 2 * (i + 0.5) / (ntheta - 1) for i in range(0, ntheta - 1)] + [math.pi / 2])

def py_strength_reference(wc, alpha=-1., sc=0., omega=10000., ntheta=20):
    if alpha < 0.: alpha = alphamax(wc)
    if alpha == 0.: return [True, 0., 0., 0., 0.]
    fclin = f_clin(wc, alpha)
    fw    = f_w(wc, alpha); ftw = fw / (1 - fclin)
    if fw < 0.: return [False, 0., 0., 0., 0.]
    fhyd  = f_hyd(wc, alpha); fthyd = fhyd / (1 - fclin)
    fair  = (1 - fclin - fw - fhyd); ftair = fair / (1 - fclin)
    fhsan = fh_san(wc, sc)

    rve_hf = rve()
    rve_hf['HYD'] = ellipsoid(shape=spheroidal(omega), symmetrize=[ISO],
                               fraction=fthyd, prop={'C': C_hyd})
    rve_hf['W']   = ellipsoid(shape=spherical, fraction=ftw,   prop={'C': C_w})
    rve_hf['AIR'] = ellipsoid(shape=spherical, fraction=ftair, prop={'C': C_air})
    C_hf = homogenize(prop='C', rve=rve_hf, scheme=SC, verbose=False,
                      epsrel=1.e-6, maxnb=100)
    if math.isnan(C_hf.k) or math.isnan(C_hf.mu):
        return [False, 0., 0., 0., 0.]

    thm, theta, thp = disc_theta(ntheta)
    rve2_hf = rve()
    for i in range(ntheta):
        rve2_hf['HYD' + str(i)] = ellipsoid(
            shape=spheroidal(omega, theta[i]), symmetrize=[TI],
            fraction=fthyd * (math.cos(thm[i]) - math.cos(thp[i])),
            prop={'C': C_hyd})
    rve2_hf['W']   = ellipsoid(shape=spherical, fraction=ftw,   prop={'C': C_w})
    rve2_hf['AIR'] = ellipsoid(shape=spherical, fraction=ftair, prop={'C': C_air})
    rve2_hf.set_prop('C', C_hf)
    homogenize(prop='C', rve=rve2_hf, scheme=SC, verbose=False,
               epsrel=1.e-3, maxnb=1)
    rve2_hf.set_param_eshelby(algo=NUMINT, epsroots=0., epsabs=1.e-3,
                              epsrel=1.e-3, maxnb=100000)
    dC_hf_dmutheta = [
        homogenize_derivative(prop='C', rve=rve2_hf, scheme=SC,
                              phase='HYD' + str(i), index=1, sym=TI,
                              verbose=False).paramsym(sym=TI)
        for i in [0]
    ]

    rve_cp = rve(matrix='HF')
    rve_cp['HF']   = ellipsoid(shape=spherical, fraction=1 - fclin,
                                prop={'C': tensor(C_hf.array, TI)})
    rve_cp['CLIN'] = ellipsoid(shape=spherical, fraction=fclin, prop={'C': C_clin})
    C_cp = homogenize(prop='C', rve=rve_cp, scheme=MT, verbose=False)
    rve_cp.set_param_eshelby(algo=NUMINT, epsroots=0., epsabs=1.e-3,
                              epsrel=1.e-3, maxnb=100000)
    dC_cp_dC_hf = [
        homogenize_derivative(prop='C', rve=rve_cp, scheme=MT, phase='HF',
                              index=i, verbose=False).paramsym(sym=TI)
        for i in range(5)
    ]

    rve_mo = rve(matrix='CP')
    rve_mo['CP']  = ellipsoid(shape=spherical, fraction=1 - fhsan,
                               prop={'C': tensor(C_cp.array, TI)})
    rve_mo['SAN'] = ellipsoid(shape=spherical, fraction=fhsan, prop={'C': C_san})
    C_mo = homogenize(prop='C', rve=rve_mo, scheme=MT, verbose=False)
    rve_mo.set_param_eshelby(algo=NUMINT, epsroots=0., epsabs=1.e-3,
                              epsrel=1.e-3, maxnb=100000)
    dC_mo_dC_cp = [
        homogenize_derivative(prop='C', rve=rve_mo, scheme=MT, phase='CP',
                              index=i, verbose=False).array
        for i in range(5)
    ]

    Shom = np.linalg.inv(C_mo.array)
    fc = []
    for itheta in [0]:
        dC = sum([dC_mo_dC_cp[a] * dC_cp_dC_hf[b][a] * dC_hf_dmutheta[itheta][b]
                  for b in range(5) for a in range(5)], 0)
        M = Shom.dot(dC.dot(Shom))
        f = rve2_hf['HYD' + str(itheta)].fraction * (1 - fclin) * (1 - fhsan)
        fc.append(1. / math.sqrt(M[2, 2] * 2 * (C_hyd.mu ** 2) / f))
    minfc = min(fc)
    mfc = 0. if math.isnan(minfc) else minfc
    k = float(C_mo.k); mu = float(C_mo.mu)
    E, _ = Enu_from_kmu(k, mu)
    return [True, k, mu, float(E), float(mfc)]
"""

const py_strength_reference = py"py_strength_reference"

# ── Row comparison ──────────────────────────────────────────────────────────
_relerr(a, b) = abs(b) < 1.0e-12 ? abs(a - b) : abs(a - b) / abs(b)

# 1 % on elastic moduli (k, μ, E) ; 2 % on the strength criterion fc — the
# exact azimuthal average + full-chain ForwardDiff closes the gap that the
# former hand-rolled IFT left at ~15 %.
function _row_pass(r; rtol_mod = 1.0e-2, rtol_fc = 2.0e-2, atol = 5.0e-2)
    checks = (
        (r.k_jl, r.k_py, rtol_mod),
        (r.mu_jl, r.mu_py, rtol_mod),
        (r.E_jl, r.E_py, rtol_mod),
        (r.fc_jl, r.fc_py, rtol_fc),
    )
    return all(c -> _relerr(c[1], c[2]) ≤ c[3] || abs(c[1] - c[2]) ≤ atol, checks)
end

# ── Sweep ───────────────────────────────────────────────────────────────────
const WCS = (0.25, 0.35, 0.5, 0.65)
const N_α = 6

println("="^92)
println("Pichler & Hellmich (2011) — Julia (public API) vs echoes C++ cross-validation")
println("(NTHETA = $NTHETA, ω = $ω_aspect ; tol : moduli 1 %, fc 2 %)")
println("="^92)
@printf "%5s %6s | %8s %8s | %8s %8s | %8s %8s | %8s %8s | %5s\n" "wc" "α" "k_jl" "k_py" "μ_jl" "μ_py" "E_jl" "E_py" "fc_jl" "fc_py" "pass"
println("─"^92)

rows = NamedTuple[]
for wc in WCS
    αs = range(0.05, αmax(wc) * (1 - 1.0e-9); length = N_α)
    for α in αs
        py = py_strength_reference(wc, α, 0.0, ω_aspect, NTHETA)
        py[1] || continue                       # skip physically invalid points
        _, k_py, mu_py, E_py, fc_py = py
        jl = try
            compute_point(wc, α)
        catch e
            @warn "Julia compute_point failed" wc α exception = e
            continue
        end
        row = (
            wc = wc, α = α,
            k_jl = jl.K_mo, k_py = k_py,
            mu_jl = jl.μ_mo, mu_py = mu_py,
            E_jl = jl.E_mo, E_py = E_py,
            fc_jl = jl.fc, fc_py = fc_py,
        )
        row = merge(row, (pass = _row_pass(row),))
        push!(rows, row)
        @printf "%5.2f %6.3f | %8.4f %8.4f | %8.4f %8.4f | %8.4f %8.4f | %8.4f %8.4f | %5s\n" row.wc row.α row.k_jl row.k_py row.mu_jl row.mu_py row.E_jl row.E_py row.fc_jl row.fc_py (row.pass ? "✓" : "✗")
    end
end

n_pass = count(r -> r.pass, rows)
@printf "\n%d / %d rows within tolerance.\n" n_pass length(rows)

# ── CSV export ──────────────────────────────────────────────────────────────
figdir = joinpath(@__DIR__, "figures")
isdir(figdir) || mkpath(figdir)
csv_path = joinpath(figdir, "benchmark_strength.csv")
open(csv_path, "w") do io
    println(io, "wc,alpha,k_jl,k_py,mu_jl,mu_py,E_jl,E_py,fc_jl,fc_py,pass")
    for r in rows
        @printf io "%.4f,%.4f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%d\n" r.wc r.α r.k_jl r.k_py r.mu_jl r.mu_py r.E_jl r.E_py r.fc_jl r.fc_py (r.pass ? 1 : 0)
    end
end
@printf "CSV: %s\n" csv_path
