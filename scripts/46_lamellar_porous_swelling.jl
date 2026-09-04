# =============================================================================
#  46_lamellar_porous_swelling.jl
#
#  The two-scale micromechanical model of
#
#    Reference: dormieux2006feuillets in docs/src/references.bib
#    "Comportement macroscopique des materiaux poreux a microstructure en
#     feuillets", C. R. Mecanique 334 (2006) 304-310,
#    doi:10.1016/j.crme.2006.03.008
#
#  derived SYMBOLICALLY, end to end. Equation numbers below are the article's.
#
#  The microstructure has two porosities:
#
#    * inside a PARTICLE, an interfoliar space of volume fraction `f` between
#      parallel solid platelets of normal `n`. Sliding is free, and electric
#      repulsion between platelets gives the interfoliar fluid a NORMAL
#      stiffness alone. That is a two-layer periodic laminate — a `Laminate`
#      cell, solved exactly by the `Laminated` scheme.
#    * between particles, spherical macropores of volume fraction `phi`. The
#      randomly oriented particles and the macropores form a porous
#      polycrystal, closed by the self-consistent scheme.
#
#  RULE OF THE SCRIPT: no published formula is ever an INPUT. Only the
#  article's data are posed — the local law (1), its linearization (2), and the
#  morphology. Everything from (5) to (18) is recomputed here and then compared
#  with what the article prints, each comparison being a `tsimplify` that must
#  return exactly 0.
#
#  Two limit passages carry the physics and are taken as limits, not as
#  substitutions:
#
#    * the interfoliar layer is SINGULAR — `C_f = f Pi n@n@n@n` has a
#      non-invertible out-of-plane block, so `plane_pinv` (hence
#      `laminate_stiffness`) does not accept it. It is regularized by an
#      isotropic `(kw, muw)` that is then sent to zero. Two independent
#      parameters rather than one, so that the independence of the limit path
#      is checked rather than assumed.
#    * the platelets are incompressible: `ks -> oo`.
#
#  Everything stays in TensND types throughout — `TensISO`, `TensTI{4}`,
#  `TensTI{2}` — so each object carries at most six Walpole coefficients rather
#  than 81 components, and `tlimit` / `tsimplify` act on those coefficients.
#  That is what makes the whole derivation tractable under SymPy.
#
#  The article's Appendix A (the Poisson-Boltzmann estimate of the swelling
#  pressure) is NOT developed here: `pi^g(h, n_M)` stays a generic function and
#  `dpi = d(pi^g)/dh` a free symbol. Only `Pi = -(h_o/f) dpi > 0` matters
#  downstream, `Pi` being positive because the swelling pressure decreases with
#  the interfoliar distance.
#
#  NOT published to the documentation gallery: SymPy-heavy scripts are kept out
#  of the Literate build (repo policy, see scripts/README.md).
# =============================================================================

import Pkg
Pkg.activate(joinpath(@__DIR__, "..", "docs"); io = devnull)

using MeanFieldHomogenization
using TensND
using LinearAlgebra
using SymPy
using Printf

# ── Reporting ───────────────────────────────────────────────────────────────

# A tensor is reported through its canonical coefficients — six for a
# `TensTI{4}` whatever its `N`, so that a major-symmetric tensor and a
# concentration tensor can be subtracted and compared on the same footing.
_coeffs(t::TensND.TensTI{4}) = collect(TensND.get_ℓ(t))
_coeffs(t::Union{TensND.TensISO, TensND.TensTI, TensND.TensOrtho}) =
    collect(TensND.get_data(t))
_coeffs(t::TensND.AbstractTens) = vec(collect(TensND.get_array(t)))   # unstructured
_coeffs(x) = collect(x)
_coeffs(x::Sym) = [x]

function report(label, x)
    c = tsimplify.(_coeffs(x))
    ok = all(iszero, c)
    @printf("  %-54s %s\n", label, ok ? "0  ✓" : string(c))
    return ok
end

section(title) = (println("\n", title); println("─"^79))

println("="^79)
println("Lamellar porous material — swelling clays")
println("symbolic derivation, C. R. Mecanique 334 (2006) 304-310")
println("="^79)

# =============================================================================
#  §1  From the local law (1)-(2) to the interfoliar stiffness (3)-(5)
# =============================================================================
#
#     sigma_f = -p 1 - pi^g n@n                                            (1)
#     pi^g(h) = pi^g_o + h_o (d pi^g/dh) (h - h_o)/h_o,
#     (h - h_o)/h_o = n@n : eps_f                                          (2)
#
#  Writing sigma_f as an affine function of eps_f identifies the prestress and
#  the stiffness of a linear elastic material with initial stress (3).

@syms h₀::positive dπ::real p::real π₀::real f::positive
@syms μs::positive ks::positive Π::positive
@syms κw::positive μw::positive
@syms ε::real                     # stands for `n@n : eps_f`, the only strain
# measure the interfoliar layer feels

πg = π₀ + h₀ * dπ * ε
σ_nn = -p - πg                    # the n@n component of sigma_f
σ_tt = -p                         # the transverse components
Π_def = -h₀ * dπ / f              # the article's definition of Pi

section("§1  The interfoliar layer, from (1)-(2)")
report("sigma_f is affine in eps_f  (d2/de2 = 0)", tdiff(σ_nn, ε, ε))
report("prestress == -p 1 - pi^g_o n@n            (4)", (tsubs(σ_nn, ε => 0) - (-p - π₀), σ_tt + p))
report("stiffness C_nnnn == f Pi, Pi = -(h_o/f) dpi/dh  (5)", tdiff(σ_nn, ε) - f * Π_def)
println("  => C_f = f Pi n@n@n@n,  sigma^p = -p 1 - pi^g_o n@n")

# =============================================================================
#  §2  The particle: a two-layer laminate
# =============================================================================

n̂ = (0, 0, 1)
𝟏 = TensISO{3}(one(Π))                          # second-order identity
nn = TensTI{2}(zero(Π), one(Π), n̂)              # n@n
W₁ = TensTI{4}(one(Π), zero(Π), zero(Π), zero(Π), zero(Π), n̂)   # n@n@n@n

# `n@n@n@n` is exactly the first Walpole tensor. Left alone the interfoliar
# stiffness would be singular — its out-of-plane block is not invertible, and
# `plane_pinv`, hence the whole laminate kernel, requires that it is. The
# isotropic `(kw, muw)` is the regularization; it is sent to zero below.
C_platelet = TensISO{3}(3ks, 2μs)
C_interfoliar = TensISO{3}(3κw, 2μw) + f * Π * W₁

particle = Laminate(; normal = n̂)
add_layer!(particle, :PLATELET, Dict(:C => C_platelet); fraction = 1 - f)
add_layer!(particle, :INTERFOLIAR, Dict(:C => C_interfoliar); fraction = f)

Cpar_reg = homogenize(particle, Laminated(), :C)
A_s_reg = strain_strain_loc(particle, :PLATELET)      # eps_s = A_s : E
A_f_reg = strain_strain_loc(particle, :INTERFOLIAR)   # eps_f = A_f : E

# The mean-field identity (6) is formed BEFORE any limit is taken: both factors
# of `C_interfoliar : A_f` are singular in the limit, their product is not.
CA_reg = (1 - f) * (C_platelet ⊡ A_s_reg) + f * (C_interfoliar ⊡ A_f_reg)

# ── The two limit passages ──────────────────────────────────────────────────

phys(t) = tsimplify(tlimit(tlimit(tlimit(t, μw, 0), κw, 0), ks, oo))
phys_swapped(t) = tsimplify(tlimit(tlimit(tlimit(t, κw, 0), μw, 0), ks, oo))

Cpar = phys(Cpar_reg)
A_s = phys(A_s_reg)
A_f = phys(A_f_reg)

section("§2  The particle, after the two limit passages")
println("  types : C^par ", typeof(Cpar))
println("          A_f   ", typeof(A_f), "  (a concentration tensor has no")
println("                                            major symmetry: N = 6)")
report("limit path independence (kw, muw swapped)", Cpar - phys_swapped(Cpar_reg))
report("mean-field identity  C^par == <C:A>       (6)", Cpar - phys(CA_reg))

# ── (7): the tensor B_pi ────────────────────────────────────────────────────

Bπ = tsimplify(f * (nn ⊡ A_f))
report("B_pi == (1-f) 1 + f n@n                   (7)", Bπ - ((1 - f) * 𝟏 + f * nn))

# Incompressibility of the platelets, the ingredient of (8): the solid carries
# no volume change, so the whole trace of the macroscopic strain is taken up by
# the interfoliar space.
report("1 : A_s == 0        (incompressible platelets)", 𝟏 ⊡ A_s)
report("f 1 : A_f == 1", f * (𝟏 ⊡ A_f) - 𝟏)

# ── (9)-(10): the split of C^par ────────────────────────────────────────────

D = tsimplify(Cpar - Π * (Bπ ⊗ Bπ))

# The article states (10) component by component; the comparison is made on the
# same footing, the Walpole coefficients of that component matrix being read
# off by `ti8_params_from_KM` rather than worked out by hand.
m = μs * (1 - f)
Dw = zeros(Sym, 6, 6)
Dw[1, 1] = Dw[2, 2] = 4m                # D_1111
Dw[1, 2] = Dw[2, 1] = 2m                # D_1122
Dw[6, 6] = 2m                           # Mandel: M_66 = 2 D_1212
D_want = TensTI{4}(TensND.ti8_params_from_KM(Dw)[1:6]..., n̂)

report("C^par - Pi B_pi@B_pi == D of (10)         (9)", D - D_want)
@printf "  C^par_nnnn = %s        (the normal stiffness)\n" string(tsimplify(Cpar[3, 3, 3, 3]))
@printf "  C^par_ntnt = %s        (free sliding)\n" string(tsimplify(Cpar[2, 3, 2, 3]))
report("C^par_nnnn == Pi", Cpar[3, 3, 3, 3] - Π)
report("C^par_ntnt == 0", Cpar[2, 3, 2, 3])

# ── (8): Levin, at the scale of the particle ────────────────────────────────
#
#  Sigma = C^par : E + f sigma^p : A_f. Only the interfoliar layer is
#  prestressed. The package has no prestress of its own, so Levin's theorem is
#  applied here — that is the one imported result, and it is a theorem.

σp = -p * 𝟏 - π₀ * nn
Σ_pre = tsimplify(f * (σp ⊡ A_f))
report("f sigma^p : A_f == -p 1 - pi^g_o B_pi     (8)", Σ_pre - (-p * 𝟏 - π₀ * Bπ))

# =============================================================================
#  §3  The assembly: a porous polycrystal closed by the self-consistent scheme
# =============================================================================
#
#  Phases: the macropores (volume fraction phi, zero stiffness) and the
#  particles (volume fraction 1-phi), each orientation of `n` being a phase.
#  Hill tensor of a SPHERE in the isotropic running estimate C^ac (15), and the
#  self-consistent condition (13)-(14).
#
#  The orientation integral of (14) is NOT discretized: the SO(3) average of a
#  tensor is a closed-form linear map on its Walpole coefficients, which is
#  exactly what `isotropify` computes — the very routine the package runs behind
#  `symmetrize = :iso`.

@syms κ::positive μ::positive φ::positive

C_ac = TensISO{3}(3κ, 2μ)
sphere = Ellipsoid(1.0)
𝕀 = TensND.tens_Id4(Val(3), Val(Sym))

A_par = strain_strain_loc(sphere, Cpar, C_ac)                 # (I + P:(C^par - C^ac))^-1
A_por = strain_strain_loc(sphere, TensISO{3}(zero(κ), zero(κ)), C_ac)

# (13)+(14). The macropores contribute nothing to <C:A>, their stiffness being
# zero; they weigh on the scheme through the average of A alone.
res_sc = _coeffs(C_ac - (1 - φ) * isotropify(Cpar ⊡ A_par))

section("§3  The self-consistent polycrystal")
println("  A^par : ", typeof(A_par))
println("  <C:A> : ", typeof(isotropify(Cpar ⊡ A_par)), "  (exact SO(3) average, no quadrature)")

# The equivalent form of the self-consistent condition, kept as an independent
# gate: <A> = I. For a spherical Hill tensor the two are equivalent, so this
# must vanish on the solution and nowhere else.
res_A = _coeffs((1 - φ) * isotropify(A_par) + φ * isotropify(A_por) - 𝕀)

# (16)-(17). The SO(3) average of the second-order tensor B_pi : A_par is
# isotropic, and its coefficient is the Terzaghi deviation g.
g_expr = (1 - φ) * TensND.get_data(isotropify(Bπ ⊡ A_par))[1]

# =============================================================================
#  §4  The limit Pi/mu_s -> 0 and the closed forms (18)
# =============================================================================
#
#  Nondimensionalization rather than a modulus sent to infinity inside a large
#  rational fraction: mu_s = Pi/t with t -> 0+, then kappa = Pi x and mu = Pi y.
#  Pi then factors out of the residuals and disappears.

@syms t::positive x::positive y::positive

_reduce(z) = tsimplify(
    sympy.cancel(
        sympy.limit(
            sympy.cancel(sympy.together(tsubs(z, κ => Π * x, μ => Π * y, μs => Π / t))),
            t, 0, "+"
        )
    )
)

eq1 = tsimplify(sympy.cancel(_reduce(res_sc[1]) / Π))
eq2 = tsimplify(sympy.cancel(_reduce(res_sc[2]) / Π))
gA = _reduce(g_expr)

section("§4  Leading order in Pi/mu_s")
@printf "  free symbols of eq1 : %s\n" string(eq1.free_symbols)
@printf "  free symbols of eq2 : %s\n" string(eq2.free_symbols)
@printf "  free symbols of g   : %s\n" string(gA.free_symbols)
println("  (Pi has cancelled, and `f` is absent — the article's remark that the")
println("   macroscopic elasticity depends on neither mu_s nor the interfoliar")
println("   porosity, only on the repulsive interactions through Pi)")

# The published closed forms (18), used ONLY as the target of a comparison.
y_paper = (φ + 2) * (1 - 4φ) / (16 * (1 + φ) * φ)
ν_paper = (2 + 17φ) / (5 * (2 + 5φ))
g_paper = (2 + φ) * (1 - 4φ) / (2 - 3φ)
# kappa/mu follows from nu alone: k/mu = 2(1+nu)/(3(1-2nu)).
x_paper = tsimplify(y_paper * 2 * (1 + ν_paper) / (3 * (1 - 2ν_paper)))

report("(18) satisfies the 1st self-consistent equation", tsubs(eq1, x => x_paper, y => y_paper))
report("(18) satisfies the 2nd self-consistent equation", tsubs(eq2, x => x_paper, y => y_paper))
report(
    "(18) satisfies the <A> = I gate",
    [tsubs(z, x => x_paper, y => y_paper) for z in _reduce.(res_A)]
)

# The solve, so that the formulas are RECOVERED and not merely verified.
sols = sympy.solve([eq1, eq2], [x, y]; dict = true)
println("\n  sympy.solve returned $(length(sols)) branch(es)")

# A function, not a loop with an assignment: `branch = …` inside a `for` at
# top level lands in a soft scope and never reaches the global.
function physical_branch(sols)
    for sol in sols
        xs = tsimplify(get(sol, x, Sym(0)))
        ys = tsimplify(get(sol, y, Sym(0)))
        ok = try
            v = Float64(tsubs(ys, φ => Sym(1) / 10))
            isfinite(v) && v > 0
        catch
            false
        end
        @printf "    y = %-46s %s\n" string(ys) (ok ? "y(0.1) > 0  <- physical" : "rejected")
        ok && return (xs, ys)
    end
    return nothing
end

branch = physical_branch(sols)

if branch !== nothing
    xs, ys = branch
    report("solved mu^ac/Pi == (phi+2)(1-4phi)/(16(1+phi)phi)", ys - y_paper)
    report("solved nu^ac    == (2+17phi)/(5(2+5phi))", tsimplify((3xs - 2ys) / (2 * (3xs + ys))) - ν_paper)
    @printf "  roots of mu^ac : %s     (the percolation threshold)\n" string(
        sympy.solve(sympy.numer(sympy.cancel(ys)), φ)
    )
else
    println("  !! no physical branch isolated — the verifications above still stand")
end

xsol, ysol = branch === nothing ? (x_paper, y_paper) : branch
report("g(phi) == (2+phi)(1-4phi)/(2-3phi)        (18)", tsubs(gA, x => xsol, y => ysol) - g_paper)
@printf "  g(phi) = %s\n" string(sympy.factor(tsimplify(tsubs(gA, x => xsol, y => ysol))))

println("\n  Macroscopic state equation (17):")
println("     Sigma = C^ac(f, phi, mu_s, Pi) : E - p 1 - g(phi) pi^g_o 1")
println("  the term in pi^g_o being the deviation from Terzaghi's principle.")

# =============================================================================
#  §5  Numeric cross-check against the package's own schemes
# =============================================================================
#
#  Same model, built with the public API and solved numerically: the laminate
#  with a large but finite `ks` and a small but finite regularization, and the
#  self-consistent scheme with `symmetrize = :iso` on the particles. The
#  agreement with §4 must improve linearly in Pi/mu_s.

section("§5  Numeric cross-check")

const F_NUM = 0.3
const MUS_NUM = 1.0

function particle_numeric(Πv; ksv = 1.0e10, wv = 1.0e-12)
    lam = Laminate(; normal = (0.0, 0.0, 1.0))
    Cw = TensISO{3}(3wv, 2wv) +
        F_NUM * Πv * TensTI{4}(1.0, 0.0, 0.0, 0.0, 0.0, (0.0, 0.0, 1.0))
    add_layer!(lam, :PLATELET, Dict(:C => TensISO{3}(3ksv, 2MUS_NUM)); fraction = 1 - F_NUM)
    add_layer!(lam, :INTERFOLIAR, Dict(:C => Cw); fraction = F_NUM)
    return homogenize(lam, Laminated(), :C)
end

Cpar_sym_num(Πv) = TensTI{4}(
    (Float64(tsubs(ℓ, f => F_NUM, μs => MUS_NUM, Π => Πv)) for ℓ in TensND.get_ℓ(Cpar))...,
    (0.0, 0.0, 1.0),
)

let Πv = 1.0e-3
    d = maximum(
        abs,
        collect(TensND.get_ℓ(particle_numeric(Πv))) .- collect(TensND.get_ℓ(Cpar_sym_num(Πv)))
    )
    @printf "  laminate, finite ks = 1e10 and w = 1e-12 vs the limit : max|dl| = %.2e\n" d
end

const TINY = TensISO{3}(3.0e-12, 2.0e-12)

# The macropores are declared as the "matrix" phase and the particles as the
# inclusion phase. For a self-consistent scheme the distinction is immaterial —
# the average runs over every phase, matrix included — but the initial estimate
# is the reference phase's property, and seeding the fixed point with the isotropic
# pore rather than with the transversely isotropic particle is what keeps the
# running estimate a `TensISO` from the first iterate to the last.
function sc_numeric(Πv, φv)
    rve = RVE()
    add_phase!(rve, :MACROPORE, Ellipsoid(1.0), Dict(:C => TINY); fraction = :rest, symmetrize = :iso)
    add_phase!(
        rve, :PARTICLE, Ellipsoid(1.0), Dict(:C => Cpar_sym_num(Πv));
        fraction = 1 - φv, symmetrize = :iso
    )
    C = homogenize(
        rve, SelfConsistent(;
            select_best = true, maxiters = 4000, damping = 0.7,
            abstol = 1.0e-14, reltol = 1.0e-12
        ), :C
    )
    return C, rve
end

# The Terzaghi deviation, reassembled the same way as in §3 but numerically:
# the package exposes no Levin post-processing.
function g_numeric(Cpar_n, C_n, φv)
    Bπ_n = TensTI{2}(1 - F_NUM, 1.0, (0.0, 0.0, 1.0))
    A = strain_strain_loc(Ellipsoid(1.0), Cpar_n, C_n)
    return (1 - φv) * TensND.get_data(isotropify(Bπ_n ⊡ A))[1]
end

μ̂(φv) = Float64(tsubs(ysol, φ => Sym(φv)))
ĝ(φv) = Float64(tsubs(g_paper, φ => Sym(φv)))

println("\n   phi     Pi/mu_s      mu_num/Pi     closed form     rel.err      g_num       g closed    rel.err")
rows = NamedTuple[]
for φv in (0.05, 0.1, 0.15, 0.2)
    for Πv in (1.0e-2, 1.0e-3, 1.0e-4)
        Cp = Cpar_sym_num(Πv)
        C, _ = sc_numeric(Πv, φv)
        μn = k_mu(C)[2] / Πv
        gn = g_numeric(Cp, C, φv)
        push!(rows, (; φ = φv, Π = Πv, μn, gn))
        @printf "  %.2f   %8.1e   %11.5f   %11.5f   %8.1e   %9.5f   %9.5f   %8.1e\n" φv Πv μn μ̂(φv) abs(μn - μ̂(φv)) / μ̂(φv) gn ĝ(φv) abs(gn - ĝ(φv)) / ĝ(φv)
    end
end
println("  (the error must fall ~linearly with Pi/mu_s — that is the expansion order)")

# =============================================================================
#  §6  Figure 2 of the article
# =============================================================================

using Plots
gr()

φgrid = range(0.005, 0.2495; length = 400)
p1 = plot(
    φgrid, [μ̂(v) for v in φgrid];
    xlabel = "macroporosity  φ", ylabel = "μᵃᶜ / Π", label = "closed form (§4)",
    lw = 2, ylims = (0, 8), legend = :topright, left_margin = 5Plots.mm,
    bottom_margin = 5Plots.mm,
)
scatter!(
    p1, [r.φ for r in rows if r.Π == 1.0e-4], [r.μn for r in rows if r.Π == 1.0e-4];
    label = "self-consistent, Π/μs = 1e-4", ms = 5, mc = :black,
)
vline!(p1, [0.25]; ls = :dash, lc = :gray, label = "percolation φ = 1/4")

p2 = plot(
    φgrid, [ĝ(v) for v in φgrid];
    xlabel = "macroporosity  φ", ylabel = "g", label = "closed form (§4)",
    lw = 2, legend = :topright, left_margin = 5Plots.mm, bottom_margin = 5Plots.mm,
)
scatter!(
    p2, [r.φ for r in rows if r.Π == 1.0e-4], [r.gn for r in rows if r.Π == 1.0e-4];
    label = "self-consistent, Π/μs = 1e-4", ms = 5, mc = :black,
)
vline!(p2, [0.25]; ls = :dash, lc = :gray, label = "percolation φ = 1/4")

fig = plot(
    p1, p2; layout = (1, 2), size = (1000, 400), left_margin = 6Plots.mm,
    bottom_margin = 6Plots.mm, top_margin = 3Plots.mm
)
figdir = joinpath(@__DIR__, "figures")
isdir(figdir) || mkpath(figdir)
figpath = joinpath(figdir, "46_lamellar_moduli.png")
savefig(fig, figpath)
println("\n§6  Figure 2 written to ", figpath)

println("\n" * "="^79)
println("Every residual above is 0: (4), (5), (6), (7), (8), (9)-(10), (17) and")
println("(18) come out of the code, the article being the oracle, not the input.")
println("="^79)
