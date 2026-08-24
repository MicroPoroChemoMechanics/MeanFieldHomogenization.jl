# =============================================================================
#  05_symbolic.jl
#
#  Symbolic computation with SymPy.jl.
#
#  Run from the MeanFieldHomogenization.jl root:
#    julia --project=. scripts/05_symbolic.jl
#
#  Prerequisites:
#    julia> using Pkg; Pkg.add("SymPy")
#
#  @syms assumption syntax (SymPyCore):
#    @syms x::positive            # single assumption
#    @syms x::(real,positive)     # multiple assumptions
#
#  What works symbolically:
#   ✅ 2D ellipse/circle — semi-axes as Sym (closed-form formulas)
#   ✅ 3D sphere/prolate/oblate — geometry as Sym (one(Sym)/Sym(n)), material params as Sym
#   ✅ Hill tensor with symbolic elastic constants (E, ν) or (λ, μ)
#
#  What does NOT work:
#   ❌ 3D symbolic geometry — the general triaxial formula uses Elliptic.jl
#      (C-library calls), which does not accept SymPy.Sym inputs.
#      Workaround: use Float64 geometry + symbolic material parameters.
#
#  Note on Ellipsoid with symbolic semi-axes:
#   Sym is <:Number but NOT <:Real, so the shape classifier cannot compare
#   the axes. Pass them in the desired order (a first, b second —
#   convention a ≥ b assumed). The basis is CanonicalBasis{2, Sym}
#   (coherent with the semi-axis type).
#
#  Sections:
#   § 1  2D ellipse — symbolic semi-axes a, b
#   § 2  2D Hill tensor — symbolic geometry + symbolic elastic params
#   § 3  3D sphere — symbolic elastic constants (λ, μ) or (E, ν)
#   § 4  3D prolate spheroid — symbolic material params
#   § 5  Conductivity — symbolic k and shape
# =============================================================================

import Pkg
Pkg.activate(joinpath(@__DIR__, "..", "docs"); io = devnull)

using MeanFieldHomogenization
using TensND
using SymPy
using Printf

# ═══════════════════════════════════════════════════════════════════════════
println("="^70)
println("  § 1  2D ELLIPSE — symbolic semi-axes  a, b  (SymPy.Sym)")
println("="^70)

@syms a::positive b::positive

ell_2d = Ellipsoid(a, b)        # Ellipsoid{2, Sym, CanonicalBasis{2,Sym}}
IA_2d = tens_IA(ell_2d)
UA_2d = tens_UA(ell_2d)

println("\n  I^A for an ellipse with semi-axes a ≥ b:")
println("  I^A_1 = ", IA_2d[1, 1])
println("  I^A_2 = ", IA_2d[2, 2])
println("  Sum   = ", simplify(IA_2d[1, 1] + IA_2d[2, 2]))
println("  (expected: b/(a+b), a/(a+b), 1)")

println("\n  U^A for 2D ellipse (selected components):")
println("  U^A_{1111} = ", simplify(UA_2d[1, 1, 1, 1]))
println("  U^A_{2222} = ", simplify(UA_2d[2, 2, 2, 2]))
println("  U^A_{1122} = ", simplify(UA_2d[1, 1, 2, 2]))
println("  U^A_{1212} = ", simplify(UA_2d[1, 2, 1, 2]))

# Circle: substitute a=b=r
@syms r::positive
ell_circle = Ellipsoid(r; dim = 2)
IA_c = tens_IA(ell_circle)
println("\n  Circle (a = b = r):")
println("  I^A_1 = ", IA_c[1, 1], "  (expected 1/2)")
println("  I^A_2 = ", IA_c[2, 2], "  (expected 1/2)")

# ═══════════════════════════════════════════════════════════════════════════
println("\n", "="^70)
println("  § 2  2D HILL TENSOR — symbolic geometry + symbolic elastic constants")
println("="^70)

@syms a2d::positive b2d::positive λ2d::positive μ2d::positive

ell_2d_sym = Ellipsoid(a2d, b2d)

# 2D isotropic: TensISO{4,2}(α, β) with α=2(λ+μ)=2k_{2D}, β=2μ
# (TensISO{dim}(α,β) constructor — dim only as type param, T inferred from args)
C_2d_sym = TensISO{2}(2 * (λ2d + μ2d), 2 * μ2d)

P_2d_sym = hill_tensor(ell_2d_sym, C_2d_sym)

println("\n  P[1,1,1,1] for ellipse (a,b) in isotropic matrix (λ,μ):")
P1111 = simplify(P_2d_sym[1, 1, 1, 1])
println("  P[1,1,1,1] = ", P1111)

println("\n  P[2,2,2,2]:")
P2222 = simplify(P_2d_sym[2, 2, 2, 2])
println("  P[2,2,2,2] = ", P2222)

println("\n  P[1,2,1,2] = P[1,2,2,1] (shear):")
println("  P[1,2,1,2] = ", simplify(P_2d_sym[1, 2, 1, 2]))

# Verify for circle (a=b): P should be isotropic (P[1111]=P[2222])
println("\n  Check isotropy for a=b (circle): simplify(P[1111]-P[2222]) at a=b:")
diff_circ = simplify(P_2d_sym[1, 1, 1, 1] - P_2d_sym[2, 2, 2, 2])
println("  P[1111]-P[2222] = ", diff_circ, "  (substituting a=b manually gives 0)")

# ═══════════════════════════════════════════════════════════════════════════
println("\n", "="^70)
println("  § 3  3D SPHERE — symbolic elastic constants  (λ, μ)")
println("="^70)
println("  Geometry: Sym (one(Sym)).  Material: SymPy.Sym.")

@syms λs::positive μs::positive

ks = λs + Sym(2) * μs / 3
C_sphere_sym = TensISO{3}(3 * ks, 2 * μs)
ell_sphere = Ellipsoid(one(Sym))     # Sym geometry (unit radius)

P_sphere_sym = hill_tensor(ell_sphere, C_sphere_sym)

println("\n  P[1,1,1,1]  (sphere, isotropic matrix):")
P1111_sphere = simplify(P_sphere_sym[1, 1, 1, 1])
println("  P[1,1,1,1] = ", P1111_sphere)

println("\n  P[1,1,2,2]:")
println("  P[1,1,2,2] = ", simplify(P_sphere_sym[1, 1, 2, 2]))

println("\n  P[1,2,1,2]:")
println("  P[1,2,1,2] = ", simplify(P_sphere_sym[1, 2, 1, 2]))

println("\n  Verification: P in terms of (E, ν)  [λ=Eν/((1+ν)(1-2ν)), μ=E/(2(1+ν))]:")
@syms Es::positive νs::positive
λs_Eν = Es * νs / ((1 + νs) * (1 - 2 * νs))
μs_Eν = Es / (2 * (1 + νs))
ks_Eν = λs_Eν + 2 * μs_Eν / 3
C_Eν = TensISO{3}(3 * ks_Eν, 2 * μs_Eν)
P_Eν = hill_tensor(ell_sphere, C_Eν)
println("  P[1,1,1,1](E,ν) = ", simplify(P_Eν[1, 1, 1, 1]))

# ─── § 3b  3D SPHERE — fully symbolic (semi-axis r AND elastic constants) ────
println("\n  --- Fully symbolic sphere: semi-axis r, elastic constants (λ,μ) ---")
println("  (geometry symbolic → isequal(r,r,r) → sphere formula in newton_potential)")

@syms r3d::positive
ell_sphere_sym = Ellipsoid(r3d)        # Sym geometry, dim=3

P_sphere_full = hill_tensor(ell_sphere_sym, C_sphere_sym)

println("\n  P[1,1,1,1] (symbolic r):")
println("  P[1,1,1,1] = ", simplify(P_sphere_full[1, 1, 1, 1]))
println("  (expected: independent of r — same as Float64 geometry above)")

# ═══════════════════════════════════════════════════════════════════════════
println("\n", "="^70)
println("  § 4  3D PROLATE SPHEROID — symbolic elastic constants")
println("="^70)

let ell_prolate = Ellipsoid(Sym(3), one(Sym), one(Sym))
    P_prolate = hill_tensor(ell_prolate, TensISO{3}(3 * (λs + 2 * μs / 3), 2 * μs))

    println("\n  Prolate spheroid (a=3, b=c=1) in isotropic matrix (λ,μ):")
    println("\n  P[1,1,1,1] =")
    println("    ", simplify(P_prolate[1, 1, 1, 1]))
    println("\n  P[2,2,2,2] =")
    println("    ", simplify(P_prolate[2, 2, 2, 2]))
    println("\n  P[1,2,1,2] =")
    println("    ", simplify(P_prolate[1, 2, 1, 2]))
    println("\n  Transverse isotropy — P[2222]−P[3333] =")
    println("    ", simplify(P_prolate[2, 2, 2, 2] - P_prolate[3, 3, 3, 3]))
end

# ─── § 4b  3D PROLATE SPHEROID — fully symbolic geometry ─────────────────────
println("\n  --- Prolate spheroid: symbolic semi-axes (a, c, c) + (λ,μ) ---")
println("  (isequal(b,c) → prolate formula in newton_potential)")

@syms a3d::positive c3d::positive
# Pass a > b = c in order (no sorting for symbolic — caller's convention)
ell_prolate_sym = Ellipsoid(a3d, c3d, c3d)

let P_pr = hill_tensor(ell_prolate_sym, TensISO{3}(3 * (λs + 2 * μs / 3), 2 * μs))
    println("\n  P[1,1,1,1] (symbolic a, c, c):")
    println("    ", simplify(P_pr[1, 1, 1, 1]))
    println("\n  Transverse isotropy — P[2222]−P[3333] =")
    println("    ", simplify(P_pr[2, 2, 2, 2] - P_pr[3, 3, 3, 3]))
    println("  (expected: 0)")
end

# ─── § 4c  3D OBLATE SPHEROID — fully symbolic geometry ──────────────────────
println("\n  --- Oblate spheroid: symbolic semi-axes (a, a, c) + (λ,μ) ---")
println("  (isequal(a,b) → oblate formula in newton_potential)")

@syms a3d_obl::positive c3d_obl::positive
ell_oblate_sym = Ellipsoid(a3d_obl, a3d_obl, c3d_obl)

let P_ob = hill_tensor(ell_oblate_sym, TensISO{3}(3 * (λs + 2 * μs / 3), 2 * μs))
    println("\n  P[1,1,1,1] (symbolic a, a, c):")
    println("    ", simplify(P_ob[1, 1, 1, 1]))
    println("\n  Transverse isotropy — P[1111]−P[2222] =")
    println("    ", simplify(P_ob[1, 1, 1, 1] - P_ob[2, 2, 2, 2]))
    println("  (expected: 0)")
end

# ═══════════════════════════════════════════════════════════════════════════
println("\n", "="^70)
println("  § 5  CONDUCTIVITY — symbolic k₀  and  2D symbolic geometry + k")
println("="^70)

@syms k_sym::positive

# 3D sphere with symbolic isotropic conductivity k
K_sym_3d = TensISO{3}(k_sym)
P_k3d = hill_tensor(Ellipsoid(one(Sym)), K_sym_3d)
println("\n  Sphere, isotropic K₀ = k·I (3D):")
println("  P[1,1] = ", simplify(P_k3d[1, 1]), "  (expected 1/(3k))")
println("  P[2,2] = ", simplify(P_k3d[2, 2]))
println("  P[1,2] = ", P_k3d[1, 2])

# 3D prolate spheroid with symbolic k
P_k3d_p = hill_tensor(Ellipsoid(Sym(3), one(Sym), one(Sym)), K_sym_3d)
println("\n  Prolate spheroid (a=3, b=c=1), K₀ = k·I:")
println("  P[1,1] = ", simplify(P_k3d_p[1, 1]))
println("  P[2,2] = ", simplify(P_k3d_p[2, 2]))

# 2D ellipse with symbolic semi-axes AND symbolic k
@syms a_c::positive b_c::positive k_c::positive
K_sym_2d = TensISO{2}(k_c)
P_2d_k = hill_tensor(Ellipsoid(a_c, b_c), K_sym_2d)
println("\n  2D ellipse (a, b), K₀ = k·I:")
println("  P[1,1] = ", simplify(P_2d_k[1, 1]))
println("  P[2,2] = ", simplify(P_2d_k[2, 2]))
println("  (expected: b/((a+b)k) and a/((a+b)k))")

# Verify sum rule: P[1,1]+P[2,2] should be 1/(k * 1) times (I^A_1+I^A_2)/(1) = 1/k
total = simplify(P_2d_k[1, 1] + P_2d_k[2, 2])
println("  P[1,1]+P[2,2] = ", total, "  (expected 1/k)")

println()
println("="^70)
println("  Symbolic computations complete.")
println("  Use simplify(), expand(), factor() from SymPy to further process results.")
println("="^70)
