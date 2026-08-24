# # Symbolic homogenization schemes: sphere in an isotropic matrix
#
# Symbolic (closed-form) homogenization schemes for a sphere embedded in an
# isotropic matrix, using SymPy.jl through TensND's generic tensor algebra.
#
# This is the "schemes" companion of `05_symbolic.jl` (which stops at the
# Hill/Eshelby tensors): here the same symbolic genericity is pushed through
# the Dilute and Mori–Tanaka estimates, and — since the self-consistent
# scheme is an intrinsically numerical fixed-point iteration — through a
# hand-derived self-consistent equation solved with SymPy's `solve`.
#
# Three physically important limits are illustrated throughout:
# - general two-phase sphere-in-matrix (``k_i``, ``\mu_i`` free)
# - porous limit (``k_i, \mu_i \to 0``) — "pore", regularized to exactly 0
# - rigid limit (``k_i, \mu_i \to \infty``) — "rigid inclusion"
#
# Prerequisites: `Pkg.add("SymPy")`, `Pkg.add("Symbolics")`.
#
# **§0** Setup · **§1** Hill/Eshelby tensors · **§2** Dilute · **§3**
# Mori–Tanaka · **§4** Porous limit · **§5** Rigid limit · **§6**
# Self-consistent (hand-derived) · **§7** Symbolics.jl bonus.

import Pkg                                                          #jl
Pkg.activate(joinpath(@__DIR__, "..", "docs"); io = devnull)                  #jl

using MeanFieldHomogenization
using TensND
using SymPy
using Printf

println("=== Symbolic homogenization schemes — sphere in an isotropic matrix ===")
println()

# ## §0 Setup — symbolic matrix (k₀,μ₀), inclusion (kᵢ,μᵢ), fraction f

println("="^78)                                                                    #jl
println("  § 0  SETUP — symbolic matrix (k₀,μ₀), inclusion (kᵢ,μᵢ), fraction f")    #jl
println("="^78)                                                                    #jl

@syms k0::positive μ0::positive ki::positive μi::positive f::positive ν0::real

C0 = iso_stiffness(k0, μ0)     # matrix stiffness   = TensISO{3}(3k₀, 2μ₀)
Ci = iso_stiffness(ki, μi)     # inclusion stiffness = TensISO{3}(3kᵢ, 2μᵢ)

# The Hill tensor of a sphere does not depend on its radius, so a plain
# Float64 unit sphere already gives a fully symbolic P once C0 is symbolic —
# no need for a symbolic radius (see 05_symbolic.jl §3b for the caveats of
# symbolic geometry).
sphere = Ellipsoid(1.0)

println("  Matrix moduli   : k₀, μ₀  (symbolic, positive)")
println("  Inclusion moduli: kᵢ, μᵢ  (symbolic, positive)")
println("  Volume fraction : f       (symbolic, positive)")
println()

# ## §1 Hill tensor P and Eshelby tensor S (sphere, isotropic matrix)

println("="^78)                                                                          #jl
println("  § 1  HILL TENSOR P AND ESHELBY TENSOR S  (sphere, isotropic matrix)")          #jl
println("="^78)                                                                          #jl

P = hill_tensor(sphere, C0)
S = P ⊡ C0

αP, βP = get_data(P)
αS, βS = get_data(S)

println("\n  P = TensISO{3}(αP, βP) with:")
println("    αP = ", simplify(αP))
println("    βP = ", simplify(βP))
println("\n  S = P ⊡ C₀ = TensISO{3}(αS, βS) with:")
println("    αS = ", simplify(αS))
println("    βS = ", simplify(βS))

# Rewrite in terms of the matrix Poisson ratio ν₀ via k₀ = 2μ₀(1+ν₀)/(3(1-2ν₀)),
# to recover the classical Eshelby (1957) sphere eigenvalues S_J and S_K.
# Any isotropic 4th-order tensor decomposes as A = α·𝕁 + β·𝕂 (tens_J4/tens_K4
# convention, see TensND docstrings) — so get_data(S) = (αS, βS) ARE directly
# the two eigenvalues S_𝕁, S_𝕂, no extra rescaling needed.
k0_of_ν0 = 2 * μ0 * (1 + ν0) / (3 * (1 - 2 * ν0))
SJ = simplify(subs(αS, k0 => k0_of_ν0))   # S_𝕁 = spherical (𝕁) eigenvalue of S
SK = simplify(subs(βS, k0 => k0_of_ν0))   # S_𝕂 = deviatoric (𝕂) eigenvalue of S

println("\n  In terms of the matrix Poisson ratio ν₀ (classical Eshelby form):")
println("    S_𝕁 = ", SJ, "   (expected (1+ν₀)/(3(1-ν₀)))")
println("    S_𝕂 = ", SK, "   (expected 2(4-5ν₀)/(15(1-ν₀)))")
println()

# ## §2 Dilute estimate — explicit concentration tensor, then API check

println("="^78)                                                                         #jl
println("  § 2  DILUTE ESTIMATE — explicit concentration tensor, then API check")        #jl
println("="^78)                                                                         #jl

# Master formula, written out so the algebra is visible:
#   A_dil = (𝕀 + P:(Cᵢ-C₀))⁻¹          (strain concentration tensor)
#   C_dil = C₀ + f·(Cᵢ-C₀):A_dil
𝕀4 = tens_Id4(Val(3), Val(Sym))
Adil = inv(𝕀4 + P ⊡ (Ci - C0))
Cdil = C0 + f * ((Ci - C0) ⊡ Adil)

k_dil, μ_dil = k_mu(Cdil)
k_dil = simplify(k_dil)
μ_dil = simplify(μ_dil)

println("\n  Dilute estimate (explicit tensor algebra):")
println("    k_dil = ", k_dil)
println("    μ_dil = ", μ_dil)

# Cross-check against the scheme API. The ordinary `RVE(:M)` takes a symbolic
# fraction as it comes — the declared element type is only a promotion floor.
rve = RVE(:M)
add_matrix!(rve, sphere, Dict(:C => C0))
add_phase!(rve, :I, sphere, Dict(:C => Ci); fraction = f)

kD, μD = k_mu(homogenize(rve, Dilute(), :C))
println("\n  Cross-check vs. homogenize(rve, Dilute(), :C):")
println("    simplify(k_dil - kD) = ", simplify(k_dil - kD), "   (expected 0)")
println("    simplify(μ_dil - μD) = ", simplify(μ_dil - μD), "   (expected 0)")
println()

# ## §3 Mori–Tanaka estimate

println("="^78)                       #jl
println("  § 3  MORI–TANAKA ESTIMATE")  #jl
println("="^78)                       #jl

kMT, μMT = k_mu(homogenize(rve, MoriTanaka(), :C))
kMT = simplify(kMT)
μMT = simplify(μMT)

println("\n  Mori–Tanaka estimate (via homogenize API):")
println("    k_MT = ", kMT)
println("    μ_MT = ", μMT)
println("\n  (closed form: k_MT = k₀ + f(kᵢ-k₀)Aₖ / ((1-f)+fAₖ), Aₖ = A_dil's 𝕁-part)")
println()

# ## §4 Porous limit (kᵢ, μᵢ → 0)

println("="^78)                                #jl
println("  § 4  POROUS LIMIT  (kᵢ, μᵢ → 0)")     #jl
println("="^78)                                #jl

k_dil_por = simplify(subs(k_dil, ki => 0, μi => 0))
μ_dil_por = simplify(subs(μ_dil, ki => 0, μi => 0))
kMT_por   = simplify(subs(kMT, ki => 0, μi => 0))
μMT_por   = simplify(subs(μMT, ki => 0, μi => 0))

println("\n  Dilute, porous:")
println("    k_dil(pore) = ", k_dil_por)
println("    μ_dil(pore) = ", μ_dil_por)
println("\n  Mori–Tanaka, porous (Hashin–Shtrikman upper bound for a porous solid):")
println("    k_MT(pore)  = ", kMT_por)
println("    μ_MT(pore)  = ", μMT_por)
println()

# ## §5 Rigid limit (kᵢ, μᵢ → ∞)

println("="^78)                                #jl
println("  § 5  RIGID LIMIT  (kᵢ, μᵢ → ∞)")      #jl
println("="^78)                                #jl

k_dil_rig = simplify(limit(limit(k_dil, ki => oo), μi => oo))
kMT_rig   = simplify(limit(limit(kMT, ki => oo), μi => oo))

println("\n  Dilute, rigid inclusion:")
println("    k_dil(rigid) = ", k_dil_rig)
println("\n  Mori–Tanaka, rigid inclusion:")
println("    k_MT(rigid)  = ", kMT_rig)
println()

# ## §6 Self-consistent — hand-derived equation (never symbolic via API)

println("="^78)                                                                    #jl
println("  § 6  SELF-CONSISTENT — hand-derived equation (never symbolic via API)")  #jl
println("="^78)                                                                    #jl
println("""
  The self-consistent scheme is intrinsically a numerical fixed-point
  iteration in MeanFieldHomogenization (Anderson/Picard, convergence tests, positive-
  definiteness guards) — none of that can run on a Sym scalar. Instead, the
  self-consistent condition is written by hand and solved with SymPy's
  `solve`.  For two isotropic spherical phases it separates into two SCALAR
  equations (Kᵢ = 4μsc/3, the sphere's Hill "auxiliary" bulk modulus, and μ*
  its Mori–Tanaka-type shear reference):
""")

@syms ksc::positive μsc::positive

κstar  = 4 * μsc / 3
μstar  = μsc * (9 * ksc + 8 * μsc) / (6 * (ksc + 2 * μsc))

eqk = (1 - f) * (k0 - ksc) / (k0 + κstar) + f * (ki - ksc) / (ki + κstar)
eqμ = (1 - f) * (μ0 - μsc) / (μ0 + μstar) + f * (μi - μsc) / (μi + μstar)

println("  eq_k(ksc,μsc) = (1-f)(k₀-ksc)/(k₀+4μsc/3) + f(kᵢ-ksc)/(kᵢ+4μsc/3) = 0")
println("  eq_μ(ksc,μsc) = (1-f)(μ₀-μsc)/(μ₀+μ*)     + f(μᵢ-μsc)/(μᵢ+μ*)     = 0")
println("""
  The general two-phase case couples eq_k and eq_μ through μ* — a coupled
  polynomial system with no compact closed form. The clean limits do have
  one:
""")

# ── Porous SC: kᵢ = μᵢ = 0 ───────────────────────────────────────────────
eqk_por = subs(eqk, ki => 0)
eqμ_por = subs(eqμ, μi => 0)
sol_por = solve([eqk_por, eqμ_por], [ksc, μsc])

println("  Porous self-consistent (kᵢ=μᵢ=0) — solving the coupled pair:")
for s in sol_por
    println("    candidate: ksc = ", s[1], ",  μsc = ", s[2])
end

# Physically meaningful sanity check: the load-bearing (percolating) branch
# should vanish exactly at the percolation threshold f = 1/2 for spheres,
# for ANY matrix moduli (k₀,μ₀). SymPy's simplify does not, by itself,
# collapse the nested sqrt(perfect square) left by `solve` at a substituted
# f = 1/2 (a known limitation), so the check is done numerically instead —
# itself a very "symbolic → substitute numbers" pedagogical moment.
if !isempty(sol_por)
    ksc_por, μsc_por = sol_por[1]
    println("\n  Percolation check — ksc(f=1/2), μsc(f=1/2) for a few matrix moduli:")
    for (k0v, μ0v) in ((30.0, 15.0), (90.0, 30.0), (5.0, 50.0))
        kval = N(subs(ksc_por, k0 => k0v, μ0 => μ0v, f => Sym(1) // 2))
        μval = N(subs(μsc_por, k0 => k0v, μ0 => μ0v, f => Sym(1) // 2))
        @printf("    k₀=%5.1f  μ₀=%5.1f  ->  ksc(1/2) = %.3e   μsc(1/2) = %.3e\n", k0v, μ0v, kval, μval)
    end
    println("  (both vanish identically at f=1/2, independent of k₀,μ₀ — the classic")
    println("   sphere percolation threshold, see tutorials/porous_materials.md)")
end

# ── Rigid SC: kᵢ, μᵢ → ∞ ─────────────────────────────────────────────────
eqk_rig = simplify(limit(eqk, ki => oo))
eqμ_rig = simplify(limit(eqμ, μi => oo))
sol_rig = solve([eqk_rig, eqμ_rig], [ksc, μsc])

println("\n  Rigid self-consistent (kᵢ,μᵢ→∞) — solving the coupled pair:")
for s in sol_rig
    println("    candidate: ksc = ", s[1], ",  μsc = ", s[2])
end

# ── Numerical cross-check vs. the API on a genuinely numeric RVE ─────────
println("\n  Numerical cross-check vs. homogenize(..., SelfConsistent(), :C):")
k0n, μ0n, fn = 30.0, 15.0, 0.2
rve_por_num = RVE(:M)
add_matrix!(rve_por_num, Ellipsoid(1.0), Dict(:C => iso_stiffness(k0n, μ0n)))
add_phase!(rve_por_num, :V, Ellipsoid(1.0), Dict(:C => iso_stiffness(1.0e-6, 1.0e-6)); fraction = fn)
k_sc_num, μ_sc_num = k_mu(homogenize(rve_por_num, AsymmetricSelfConsistent(; abstol = 1.0e-12, maxiters = 200, select_best = true), :C))

if !isempty(sol_por)
    ksc_por, μsc_por = sol_por[1]
    ksc_por_val = N(subs(ksc_por, k0 => k0n, μ0 => μ0n, f => fn))
    μsc_por_val = N(subs(μsc_por, k0 => k0n, μ0 => μ0n, f => fn))
    @printf("    symbolic (numeric subs.): k = %.6f   μ = %.6f\n", ksc_por_val, μsc_por_val)
    @printf("    numeric AsymmetricSelfConsistent : k = %.6f   μ = %.6f\n", k_sc_num, μ_sc_num)
end
println()

# ## §7 Symbolics.jl bonus — same computation, the Num backend

println("="^78)                                                    #jl
println("  § 7  SYMBOLICS.JL BONUS — same computation, the Num backend")  #jl
println("="^78)                                                    #jl
println("""
  TensND does not care which symbolic engine produced the scalars: the same
  tensor algebra (⊡, inv, tens_Id4, ...) dispatches identically on SymPy's
  Sym and on Symbolics.jl's Num. Below, P and S are recomputed with @variables
  instead of @syms.
""")

using Symbolics

Symbolics.@variables k0s μ0s
C0s = iso_stiffness(k0s, μ0s)
Ps = hill_tensor(sphere, C0s)
Ss = Ps ⊡ C0s

αPs, βPs = get_data(Ps)
αSs, βSs = get_data(Ss)

println("  P (Symbolics.Num):")
println("    αP = ", Symbolics.simplify(αPs))
println("    βP = ", Symbolics.simplify(βPs))
println("  S = P ⊡ C₀ (Symbolics.Num):")
println("    αS = ", Symbolics.simplify(αSs))
println("    βS = ", Symbolics.simplify(βSs))
println()

println("="^78)   #jl
println("  Done.") #jl
println("="^78)   #jl
