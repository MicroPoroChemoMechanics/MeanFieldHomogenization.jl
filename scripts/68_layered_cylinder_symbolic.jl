# # [Layered cylinder: the transfer relations, symbolically](@id tut-layered-cylinder-symbolic)
#
# !!! info "Before this page"
#     The theory page [Layered cylinder](@ref th-layered-cylinder) states the four
#     radial problems of a coated fiber and the relations that carry their
#     solution from one layer to the next. This page derives each of them with
#     TensND's symbolic differential operators, then checks the package against
#     the result.
#
# A [`LayeredCylinder`](@ref) is an infinite fiber of axis ``\underline{e}_z`` made
# of coaxial layers, each transversely isotropic about ``\underline{e}_z``. Under a
# remote uniform strain the displacement of each layer is a combination of a few
# elementary fields, the **modes**; an interface relates the modes of the two
# layers it separates. Three things are derived here: the modes, the conditions
# at an interface, and the order in which the moduli must be multiplied when
# they are operators that do not commute (aging viscoelasticity).

import Pkg                                                          #jl
Pkg.activate(joinpath(@__DIR__, "..", "docs"); io = devnull)                 #jl

using MeanFieldHomogenization
using TensND
using SymPyPythonCall
using LinearAlgebra

# ## 1. A transversely isotropic layer in cylindrical coordinates
#
# The stiffness of a layer is written with Hill's moduli: ``k`` (plane-strain bulk
# modulus), ``m`` (transverse shear), ``p`` (axial shear), ``n = C_{3333}`` and
# ``l = C_{1133}``. The stress of a displacement field ``\underline{u}`` is
# ``\boldsymbol\sigma = \mathbb{C}:\boldsymbol\varepsilon(\underline{u})``.

Cyl = coorsys_cylindrical()
r, θ, z = get_coords(Cyl)
𝐞ʳ, 𝐞ᶿ, 𝐞ᶻ = unitvec(Cyl)
k, m, n, p = symbols("k m n p", positive = true)
l = symbols("l", real = true)
ℂ = ti_from_constants(k + m, k - m, l, n, p, (Sym(0), Sym(0), Sym(1)))
stress(u) = ℂ ⊡ symgrad(u, Cyl)
equilibrium(u) = divergence(stress(u), Cyl)
nothing #hide

# ## 2. The four problems and their modes
#
# A remote strain ``\boldsymbol{E}`` splits into an axial strain ``E_{zz}``, an
# in-plane equibiaxial strain, an in-plane shear and an antiplane shear. Each
# one drives a displacement of a fixed angular shape, whose radial part remains
# to be found.
#
# ### Axisymmetric problem
#
# With ``\underline{u} = f(r)\,\underline{e}_r + \varepsilon z\,\underline{e}_z``,
# the only equation left is radial:

f = SymFunction("f", real = true)
ε = symbols("varepsilon", real = true)
u_axi(fr) = fr * 𝐞ʳ + ε * z * 𝐞ᶻ
simplify(equilibrium(u_axi(f(r))) ⋅ 𝐞ʳ)

# Up to the factor ``k + m`` it is Euler's equation ``f'' + f'/r - f/r^2 = 0``, of
# solutions ``r`` and ``1/r``: ``f = A r + B/r``. The radial stress is the quantity an interface
# transmits:

A, B = symbols("A B", real = true)
σrr_axi = simplify((stress(u_axi(A * r + B / r)) ⋅ 𝐞ʳ) ⋅ 𝐞ʳ)

# The axial strain enters it through ``l\,\varepsilon``: the in-plane and axial
# loadings are coupled, which is why the concentration tensor of a cylinder has
# no major symmetry.
#
# ### Antiplane shear
#
# With ``\underline{u} = w(r)\cos\theta\,\underline{e}_z`` the equation is again
# Euler's, so ``w = A r + B/r``, and the stress an interface transmits is
# ``\sigma_{rz}``:

w = SymFunction("w", real = true)
simplify(equilibrium(w(r) * cos(θ) * 𝐞ᶻ) ⋅ 𝐞ᶻ / cos(θ))

#-

σrz = simplify((stress((A * r + B / r) * cos(θ) * 𝐞ᶻ) ⋅ 𝐞ʳ) ⋅ 𝐞ᶻ / cos(θ))

# Transverse conduction is the same problem: ``T = (A r + B/r)\cos\theta``, with
# the conductivity ``k_T`` in place of ``p`` and ``-q_r`` in place of
# ``\sigma_{rz}``.
#
# ### Transverse shear
#
# With ``\underline{u} = U(r)\cos 2\theta\,\underline{e}_r + V(r)\sin
# 2\theta\,\underline{e}_\theta``, power laws ``U = a\,r^s``, ``V = b\,r^s`` leave
# two linear equations for ``(a, b)``:

s, a, b = symbols("s a b", real = true)
u_shear(U, V) = U * cos(2θ) * 𝐞ʳ + V * sin(2θ) * 𝐞ᶿ
eq = equilibrium(u_shear(a * r^s, b * r^s))
E₁ = simplify((eq ⋅ 𝐞ʳ) / (r^(s - 2) * cos(2θ)))
E₂ = simplify((eq ⋅ 𝐞ᶿ) / (r^(s - 2) * sin(2θ)))
system = [diff(E₁, a) diff(E₁, b); diff(E₂, a) diff(E₂, b)]

# They have a nonzero solution when the determinant vanishes,

factor(det(system))

# that is for ``s \in \{1, 3, -3, -1\}``: four modes. Their shapes ``(a, b)`` are
# the null vectors of the system:

[simplify.(subs.(system, s => q).nullspace()[1]') for q in (1, 3, -3, -1)]

# Only the ratio ``\chi = m/k`` appears. Scaled so that no modulus divides
# another one than through ``\chi``, the four modes of amplitudes ``(a, b, c, d)``
# read

χ = m / k
c, d = symbols("c d", real = true)
U = r * a + r^3 * (χ - 1) * b + c / r^3 + (1 + χ) * d / r
V = -r * a + r^3 * (χ + 2) * b + c / r^3 - χ * d / r
@assert all(iszero, simplify.(TensND.canonical_components(equilibrium(u_shear(U, V)))))
σ = stress(u_shear(U, V))
σrr = simplify((σ ⋅ 𝐞ʳ) ⋅ 𝐞ʳ / cos(2θ))

#-

σrθ = simplify((σ ⋅ 𝐞ʳ) ⋅ 𝐞ᶿ / sin(2θ))

# The tractions carry the single factor ``m``. These are the expressions of the
# package, entry for entry:

const LC = MeanFieldHomogenization.LayeredCylinders
state = LC._shear_state(r, k, m, a, b, c, d)
@assert all(iszero, simplify.(collect(state) .- [U, V, σrr, σrθ]))

# ## 3. What an interface transmits
#
# Between two layers the **state** — the displacement and the traction on the
# cylinder ``r = R`` — is continuous when the interface is perfect. Two
# imperfect interfaces are available in elasticity.
#
# A **spring** lets the displacement jump in proportion to the traction, which
# stays continuous: ``[\![u_r]\!] = s_n\,\sigma_{rr}``,
# ``[\![u_\theta]\!] = s_t\,\sigma_{r\theta}``, ``[\![u_z]\!] = s_t\,\sigma_{rz}``.
#
# A **membrane** keeps the displacement continuous and carries a surface stress
# ``\boldsymbol\sigma^{\mathrm s} = \lambda^{\mathrm s}\,(\operatorname{tr}
# \boldsymbol\varepsilon^{\mathrm s})\,\boldsymbol{1}^{\mathrm s} +
# 2\mu^{\mathrm s}\,\boldsymbol\varepsilon^{\mathrm s}``, where
# ``\boldsymbol\varepsilon^{\mathrm s}`` is the strain in the tangent plane. Its
# equilibrium makes the traction jump:
# ``[\![\boldsymbol\sigma]\!]\cdot\underline{e}_r = -\operatorname{div}^{\mathrm
# s}\boldsymbol\sigma^{\mathrm s}``. The surface stress of a displacement read
# on the cylinder, extended without dependence on ``r``, has the surface
# divergence as its divergence:

λs, μs, R = symbols("lambda_s mu_s R", positive = true)
𝟏 = TensISO{3}(Sym(1))
Pᵗ = 𝟏 - 𝐞ʳ ⊗ 𝐞ʳ                           # projector on the tangent plane
function traction_jump(u)
    εs = Pᵗ ⋅ symgrad(u, Cyl) ⋅ Pᵗ
    σs = tsubs(λs * tr(εs) * Pᵗ + 2μs * εs, r => R)      # read on r = R …
    return -tsubs(divergence(σs, Cyl), r => R)           # … and extended radially
end
nothing #hide

# For the three displacement shapes, with ``\kappa^{\mathrm s} = \lambda^{\mathrm
# s} + \mu^{\mathrm s}``:

u₀, U₀, V₀, w₀ = symbols("u_0 U_0 V_0 w_0", real = true)
jump_axi = simplify(traction_jump(u₀ * 𝐞ʳ + ε * z * 𝐞ᶻ) ⋅ 𝐞ʳ)

#-

t = traction_jump(u_shear(U₀, V₀))
jump_shear = (simplify(t ⋅ 𝐞ʳ / cos(2θ)), simplify(t ⋅ 𝐞ᶿ / sin(2θ)))

#-

jump_anti = simplify(traction_jump(w₀ * cos(θ) * 𝐞ᶻ) ⋅ 𝐞ᶻ / cos(θ))

# that is
#
# ```math
# \begin{aligned}
# [\![\sigma_{rr}]\!] &= \frac{(\kappa^{\mathrm s} + \mu^{\mathrm s})\,u_r/R +
#   (\kappa^{\mathrm s} - \mu^{\mathrm s})\,\varepsilon}{R}
#   &&\text{(axisymmetric)},\\
# [\![\sigma_{rr}]\!] &= (\kappa^{\mathrm s} + \mu^{\mathrm s})\,\frac{U + 2V}{R^2},
#   \quad [\![\sigma_{r\theta}]\!] = 2(\kappa^{\mathrm s} + \mu^{\mathrm s})\,\frac{U + 2V}{R^2}
#   &&\text{(transverse shear)},\\
# [\![\sigma_{rz}]\!] &= \mu^{\mathrm s}\,\frac{u_z}{R^2}
#   &&\text{(antiplane shear)},
# \end{aligned}
# ```
#
# which the package applies ([`MembraneInterface`](@ref) stores
# ``\kappa^{\mathrm s}`` and ``\mu^{\mathrm s}``):

κs = λs + μs
itf = MembraneInterface(κs, μs)
@assert iszero(simplify(LC._jump(itf, LC.Axisymmetric(), R, (u₀, Sym(0)), ε)[2] - jump_axi))
J = LC._jump(itf, LC.TransverseShear(), R, (U₀, V₀, Sym(0), Sym(0)))
@assert all(iszero, simplify.([J[3] - jump_shear[1], J[4] - jump_shear[2]]))
@assert iszero(simplify(LC._jump(itf, LC.Antiplane(), R, (w₀, Sym(0)))[2] - jump_anti))

# In conduction the two interfaces are a **Kapitza resistance**,
# ``[\![T]\!] = -\rho\,q_r`` with a continuous flux, and a **conducting surface**,
# which carries the flux ``-k^{\mathrm s}\nabla^{\mathrm s}T`` and makes the normal
# flux jump by its surface divergence:

ks = symbols("k_s", positive = true)
T₀ = symbols("T_0", real = true)
Tsurf = T₀ * cos(θ)
simplify(subs(divergence(-ks * (Pᵗ ⋅ grad(Tsurf, Cyl)), Cyl), r => R) / cos(θ))

# The balance ``[\![q_r]\!] + \operatorname{div}^{\mathrm s}\underline{q}^{\mathrm s} = 0`` then gives
# ``[\![q_r]\!] = -k^{\mathrm s}\,T/R^2``.

# ## 4. From one layer to the next
#
# In a layer the amplitudes give the state at any radius (section 2), and the
# state at one radius gives the amplitudes back. Carrying a state across a layer
# is therefore a product of two matrices. For the antiplane problem, with the
# state ``(w, \sigma_{rz})``:

M(ρ) = [ρ 1 / ρ; p -p / ρ^2]                 # (A, B) ↦ (w, σ_rz) at radius ρ
r₁, r₂ = symbols("r_1 r_2", positive = true)
transfer = simplify.(M(r₂) * inv(M(r₁)))

# This is the transfer matrix of a layer, in the sense of [herve1995](@citet):
# with the jump of the interface in between, the product of such matrices from
# the core to the matrix gives the state everywhere per unit core amplitude,
# and the remote condition fixes that amplitude. The package composes the same
# two maps, without forming the matrix:

A₀, B₀ = LC._pot_amplitudes(r₁, p, w₀, U₀)
@assert all(iszero, simplify.(collect(LC._pot_state(r₂, p, A₀, B₀)) .- transfer * [w₀, U₀]))

# The closed forms of the composite-cylinder model follow. A fiber of radius
# ``a`` and moduli ``(k_1, p_1)`` coated up to the radius 1 by a matrix
# ``(k_2, m_2, p_2)`` answers an in-plane dilatation and an antiplane shear as a
# homogeneous fiber of moduli [hashinRosen1964](@cite)

k₁, p₁, k₂, m₂, p₂, a₁ = symbols("k_1 p_1 k_2 m_2 p_2 a", positive = true)
coating = ti_from_constants(k₂ + m₂, k₂ - m₂, l, n, p₂, (Sym(0), Sym(0), Sym(1)))
fiber = ti_from_constants(k₁ + m, k₁ - m, l, n, p₁, (Sym(0), Sym(0), Sym(1)))
sol = LC._elastic_solution(LayeredCylinder((a₁, Sym(1)), (fiber, coating)), coating)
u_R, σ_R = sol.dilatation.outside[2]
k_eff = factor(simplify(σ_R / (2u_R)))

#-

w_R, τ_R = sol.antiplane.outside[2]
p_eff = factor(simplify(τ_R / w_R))

#-

cf = a₁^2                                   # fiber fraction
@assert iszero(simplify(k_eff - (k₂ + cf / (1 / (k₁ - k₂) + (1 - cf) / (k₂ + m₂)))))
@assert iszero(simplify(p_eff - p₂ * (p₁ * (1 + cf) + p₂ * (1 - cf)) / (p₁ * (1 - cf) + p₂ * (1 + cf))))

# ## 5. Moduli that do not commute
#
# In aging linear viscoelasticity a modulus is a Volterra operator: the stress
# history is ``\sigma = k\circ\varepsilon``, and two operators commute only in
# special cases (non-aging laws on a uniform time grid). The amplitudes become
# histories, and every product above must be read in a definite order. The rule
# is the one the mechanics gives: a modulus acts on the left of the strain it
# multiplies.
#
# The transverse shear is the delicate case, because the shapes of its modes
# depend on the moduli. With non-commuting symbols, the strain of a mode
# ``U = \alpha\,r^s``, ``V = \beta\,r^s`` and its equilibrium read

K, Mo, α, β = symbols("K M alpha beta", commutative = false)
function shear_equilibrium(αe, βe, q)
    err = q * αe                              # ε_rr / r^(s−1)
    ett = αe + 2βe                            # ε_θθ / r^(s−1)
    g = -2αe + (q - 1) * βe                   # 2 ε_rθ / r^(s−1)
    srr = (K + Mo) * err + (K - Mo) * ett     # the moduli on the left
    stt = (K - Mo) * err + (K + Mo) * ett
    srt = Mo * g
    return expand((q - 1) * srr + 2srt + srr - stt), expand((q - 1) * srt - 2stt + 2srt)
end
shear_equilibrium(α, β, 3)

# The two equations of the mode ``s = 3`` are the same one,
# ``(2K + M)\circ\alpha + (K - M)\circ\beta = 0``: the mode exists, with
# ``\alpha = (2K + M)^{-1}\circ(M - K)\circ\beta``. Writing
# ``\chi = K^{-1}\circ M`` — the inverse on the **left** — it is
# ``\alpha = (\chi - 1)\circ b``, ``\beta = (\chi + 2)\circ b`` for any history
# ``b``, and likewise ``\alpha = (1 + \chi)\circ d``, ``\beta = -\chi\circ d`` for
# ``s = -1``:

Χ = K^-1 * Mo
bh, dh = symbols("b d", commutative = false)
residual(αe, βe, q) = simplify.(expand.(K^-1 .* shear_equilibrium(αe, βe, q)))
(residual((Χ - 1) * bh, (Χ + 2) * bh, 3), residual((1 + Χ) * dh, -Χ * dh, -1))

# Both vanish identically, with no assumption of commutation: the modes of
# section 2 hold for operators provided ``\chi`` is read as
# ``K^{-1}\circ M``. With the other order, ``M\circ K^{-1}``, they do not:

Χʳ = Mo * K^-1
residual((Χʳ - 1) * bh, (Χʳ + 2) * bh, 3)

# The package writes the three problems once, for numbers and for operators,
# each product in this order; the aging cylinder is the elastic one evaluated on
# Volterra matrices. See [Aging viscoelasticity](@ref man-viscoelasticity) for its use.
