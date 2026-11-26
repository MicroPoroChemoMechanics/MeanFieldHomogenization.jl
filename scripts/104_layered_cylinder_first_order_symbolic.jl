# # [Layered cylinder: the systems of the first order, symbolically](@id tut-layered-cylinder-first-order-symbolic)
#
# !!! info "Before this page"
#     [Layered cylinder: modes and transfer relations, symbolically](@ref tut-layered-cylinder-symbolic),
#     which solves the layers by their radial modes,
#     [Layered sphere: the systems of the first order, symbolically](@ref tut-layered-sphere-first-order-symbolic),
#     whose method is followed step by step, and the theory pages
#     [Layered spheres and cylinders: the radial problem](@ref th-radial-problems)
#     and [Layered cylinder](@ref th-layered-cylinder), which state the results
#     derived here.
#
# A layer of a [`LayeredCylinder`](@ref) is isotropic or transversely isotropic
# about the axis of the fiber. Each loading mode of the cylinder is a system of
# the first order ``r\,\mathrm{d}Y/\mathrm{d}r = \boldsymbol{M}\,Y`` on the state
# that an interface transmits. This page builds the matrices
# ``\boldsymbol{M}`` of such a layer from the field equations, with TensND's
# differential operators in cylindrical coordinates and moduli that may depend
# on the radius, crosses a uniform layer by the power of ``\boldsymbol{M}``, and
# finds the transfer relations of the radial modes again.

import Pkg                                                          #jl
Pkg.activate(joinpath(@__DIR__, "..", "docs"); io = devnull)                 #jl

using MeanFieldHomogenization
using TensND
using SymPyPythonCall
using LinearAlgebra

# ## 1. The layer in cylindrical coordinates
#
# TensND orders the cylindrical coordinates ``(r, \theta, z)``. The stiffness
# is written with Hill's moduli about the axis ``\underline{e}_z``,
# ``k = \tfrac12(C_{rrrr} + C_{rr\theta\theta})``, ``l = C_{rrzz}``,
# ``m = \tfrac12(C_{rrrr} - C_{rr\theta\theta})``, ``n = C_{zzzz}`` and
# ``p = C_{rzrz}``, taken as functions of ``r`` to see whether their
# derivatives enter the systems. `ti_from_constants` builds the tensor from
# them (`form = :hill`); its components are the same in every frame turned
# about the axis, so that it is a field of constant components in the frame of
# the cylindrical chart, which `tensor_at` gives in the coordinate system:

Cyl = coorsys_cylindrical()
r, θ, z = get_coords(Cyl)
𝐞ʳ, 𝐞ᶿ, 𝐞ᶻ = unitvec(Cyl)
fn(name) = SymFunction(name, positive = true)(r)
k, l, m, n, p = fn("k"), fn("l"), fn("m"), fn("n"), fn("p")
ℂ = tensor_at(ChartTens(ti_from_constants(k, l, m, n, p, (0, 0, 1); form = :hill), CylindricalChart()), Cyl)
stress(u) = ℂ ⊡ symgrad(u, Cyl)
component(σ, a, b) = (σ ⋅ a) ⋅ b
nothing #hide

# A uniform layer has constant moduli, here `L`, in which the package writes
# its radial modes:

const LC = MeanFieldHomogenization.LayeredCylinders
const RS = MeanFieldHomogenization.Core
k₀, l₀, m₀, n₀, p₀ = symbols("k l m n p", positive = true)
uniform = (k => k₀, l => l₀, m => m₀, n => n₀, p => p₀)
L = (k = k₀, l = l₀, m = m₀, n = n₀, p = p₀)
constant(A) = simplify.(subs.(A, uniform...))
nothing #hide

# The function `first_order` builds ``\boldsymbol{M}`` from the expression of a
# state and from the equilibrium equations, `first_order_forcing` the second
# member that a loading acting inside the layer leaves, and `mode_states` the
# eigenvectors of a constant matrix.

include(joinpath(pkgdir(MeanFieldHomogenization), "scripts", "common", "radial_first_order.jl"))
#md # ```@raw html
#md # <details class="source-file"><summary>Show the code of <code>scripts/common/radial_first_order.jl</code>, which defines <code>first_order</code>, <code>first_order_forcing</code> and <code>mode_states</code></summary>
#md # ```
#md #
#md # ```@eval
#md # Main.show_source("scripts/common/radial_first_order.jl")
#md # ```
#md #
#md # ```@raw html
#md # </details>
#md # ```

# ## 2. Antiplane shear, and conduction across the axis
#
# With ``\underline{u} = \varphi(r)\cos\theta\,\underline{e}_z``, the state is
# ``Y = (\varphi,\; r\,\sigma_{rz})``, the stress being its radial function on
# ``\cos\theta``, and the only equation is the axial one:

ϕ = SymFunction("varphi", real = true)(r)
σ = stress(ϕ * cos(θ) * 𝐞ᶻ)
Y_anti = [ϕ, simplify(r * component(σ, 𝐞ʳ, 𝐞ᶻ) / cos(θ))]
M_anti = first_order(Y_anti, [ϕ], [simplify(divergence(σ, Cyl) ⋅ 𝐞ᶻ / cos(θ))], r)

# Only the axial shear modulus enters, at the radius ``r`` and without its
# derivative. A solution ``Y = r^s\,\underline{y}`` has
# ``\boldsymbol{M}\,\underline{y} = s\,\underline{y}``: the exponents of the
# radial modes of a uniform layer are the eigenvalues of ``\boldsymbol{M}``,

s = symbols("s", real = true)
𝟙₂ = Sym[1 0; 0 1]
M₁ = constant(M_anti)
factor(det(M₁ - s * 𝟙₂))

# the modes ``r`` and ``r^{-1}``, and their states are its eigenvectors:

V₁ = mode_states(M₁, (1, -1))

# With ``\tau = \ln r`` the system has a constant matrix, and the state is
# carried from the radius ``r_1`` to the radius ``r_2`` by
# ``\xi^{\boldsymbol{M}} = \exp(\boldsymbol{M}\ln\xi)``, ``\xi = r_2/r_1``:

ξ = symbols("xi", positive = true)
P₁ = simplify.(V₁ * Diagonal([ξ, 1 / ξ]) * inv(V₁))

# It is the solution of
# ``\xi\,\mathrm{d}\boldsymbol{P}/\mathrm{d}\xi = \boldsymbol{M}\,\boldsymbol{P}``
# equal to the identity at ``\xi = 1``, as is the closed form of the package
# (`radial_power`), which does not use the eigenvectors:

power_of(P) = all(iszero, simplify.(ξ .* diff.(P, ξ) .- M₁ * P)) && all(iszero, simplify.(subs.(P, ξ => 1) .- 𝟙₂))
@assert power_of(P₁) && power_of(Matrix(RS.radial_power(LC._pot_system(LC.RadialPair(p₀, p₀)), ξ)))

# The package crosses such a layer by the amplitudes of its modes, on the
# state ``(\varphi, \sigma_{rz})`` without the factor ``r``. With
# ``\boldsymbol{D}(r) = \operatorname{diag}(1, r)``, which turns one state into
# the other, the two strategies give the same state at ``r_2``:

r₁, r₂ = symbols("r_1 r_2", positive = true)
D₂(x) = Diagonal([1, x])
ϕ₀, τ₀ = symbols("varphi_0 tau_0", real = true)
by_power = inv(D₂(r₂)) * subs.(P₁, ξ => r₂ / r₁) * D₂(r₁) * [ϕ₀, τ₀]
by_modes = collect(LC._pot_cross(r₁, r₂, p₀, (ϕ₀, τ₀))[2])
@assert all(iszero, simplify.(by_power .- by_modes))

# Conduction across the axis is the same problem. With a transverse
# conductivity ``k_T(r)``, an axial one ``k_A(r)``, and
# ``T = \hat T(r)\cos\theta``, the state is ``Y = (\hat T,\; -r\,\hat q)``,
# ``\hat q`` being the radial function of the outward flux:

k_T, k_A = fn("k_T"), fn("k_A")
𝐊 = tensor_at(ChartTens(TensTI{2}(k_T, k_A, (0, 0, 1)), CylindricalChart()), Cyl)
T̂ = SymFunction("T", real = true)(r)
T = T̂ * cos(θ)
q = -(𝐊 ⋅ (diff(T, r) * 𝐞ʳ + diff(T, θ) / r * 𝐞ᶿ))
Y_cond = [T̂, simplify(-r * (q ⋅ 𝐞ʳ) / cos(θ))]
M_cond = first_order(Y_cond, [T̂], [simplify(divergence(q, Cyl) / cos(θ))], r)

# ## 3. In-plane dilatation and axial strain
#
# With ``\underline{u} = f(r)\,\underline{e}_r + \varepsilon z\,\underline{e}_z``
# and ``Y = (f,\; r\,\sigma_{rr})``:

f = SymFunction("f", real = true)(r)
ε = symbols("varepsilon", real = true)
σ = stress(f * 𝐞ʳ + ε * z * 𝐞ᶻ)
Y_axi = [f, simplify(r * component(σ, 𝐞ʳ, 𝐞ʳ))]
equilibrium_axi = [simplify(divergence(σ, Cyl) ⋅ 𝐞ʳ)]
M_axi = first_order(Y_axi, [f], equilibrium_axi, r)

# The right-hand side is not zero when ``Y`` is: the axial strain acts inside
# the layer, through the modulus ``l``, as a second member proportional to
# ``\varepsilon\,r``,

F_axi = first_order_forcing(Y_axi, [f], equilibrium_axi, r)

# Since ``g = \varepsilon\,r`` obeys ``r\,g' = g``, it is appended to the state:
# on ``(f,\; r\,\sigma_{rr},\; \varepsilon r)`` the system is homogeneous again,
# of constant matrix in a uniform layer,

M₂ = constant([M_axi F_axi ./ (ε * r); 0 0 1])

#-

𝟙₃ = Sym.(Matrix(1I, 3, 3))
factor(det(M₂ - s * 𝟙₃))

# The exponent ``1`` is double, and has two eigenvectors: the in-plane
# dilatation ``u_r = r`` without axial strain, and the axial strain with the
# radial displacement that leaves the layer free of radial stress. The third
# mode is ``u_r = r^{-1}``:

V₂ = mode_states(M₂, (1, -1))

#-

P₂ = simplify.(V₂ * Diagonal([ξ, ξ, 1 / ξ]) * inv(V₂))

# The state ``(u_r, \sigma_{rr})`` that the package finds at ``r_2`` from the
# amplitudes ``u_r = A r + B/r`` is the same:

u₀, σ₀ = symbols("u_0 sigma_0", real = true)
by_power = (subs.(P₂, ξ => r₂ / r₁) * [u₀, r₁ * σ₀, ε * r₁])[1:2] ./ [1, r₂]
by_modes = collect(LC._axi_cross(r₁, r₂, L, (u₀, σ₀), ε)[2])
@assert all(iszero, simplify.(by_power .- by_modes))

# For an isotropic layer ``(\kappa, \mu)``, the matrix is that of the systems
# of the package, which are written for a layer transversely isotropic about
# the radius: the isotropic layer belongs to both classes.

κ, μ = symbols("kappa mu", positive = true)
iso = hill_moduli(iso_stiffness(κ, μ))
isotropic = (k₀ => iso.k, l₀ => iso.l, m₀ => iso.m, n₀ => iso.n, p₀ => iso.p)
radial = LC._radial_moduli(iso_stiffness(κ, μ))
@assert all(iszero, simplify.(subs.(M₂, isotropic...) .- Matrix(LC._axi_system(radial))[1:3, 1:3]))

# ## 4. In-plane shear
#
# With
# ``\underline{u} = U(r)\cos 2\theta\,\underline{e}_r + V(r)\sin 2\theta\,\underline{e}_\theta``
# and ``Y = (U,\; V,\; r\,\sigma_{rr},\; r\,\sigma_{r\theta})``, the stresses
# being their radial functions on ``\cos 2\theta`` and ``\sin 2\theta``:

U, V = SymFunction("U", real = true)(r), SymFunction("V", real = true)(r)
σ = stress(U * cos(2θ) * 𝐞ʳ + V * sin(2θ) * 𝐞ᶿ)
eq = divergence(σ, Cyl)
Y_shear = [U, V, simplify(r * component(σ, 𝐞ʳ, 𝐞ʳ) / cos(2θ)), simplify(r * component(σ, 𝐞ʳ, 𝐞ᶿ) / sin(2θ))]
M_shear = first_order(Y_shear, [U, V], [simplify(eq ⋅ 𝐞ʳ / cos(2θ)), simplify(eq ⋅ 𝐞ᶿ / sin(2θ))], r)

# Only ``k`` and ``m`` enter, without their derivatives. The exponents of a
# uniform layer are

M₃ = constant(M_shear)
𝟙₄ = Sym.(Matrix(1I, 4, 4))
factor(det(M₃ - s * 𝟙₄))

# the modes ``r``, ``r^3``, ``r^{-3}`` and ``r^{-1}``, of states

exponents = (1, 3, -3, -1)
V₃ = mode_states(M₃, exponents)

# They are the modes of the package, whose amplitudes ``(a, b, c, d)`` give the
# state ``(U, V, \sigma_{rr}, \sigma_{r\theta})``: each one, read at ``r = 1``,
# is an eigenvector of its exponent.

mode(j) = collect(LC._shear_state(Sym(1), k₀, m₀, (Sym(Int(i == j)) for i in 1:4)...))
@assert all(all(iszero, simplify.(M₃ * mode(j) .- q .* mode(j))) for (j, q) in enumerate(exponents))

# The power of the matrix follows, and the transfer of the package through
# ``\boldsymbol{D}(r) = \operatorname{diag}(1, 1, r, r)``:

P₃ = simplify.(V₃ * Diagonal([ξ^Sym(q) for q in exponents]) * inv(V₃))
D₄(x) = Diagonal([1, 1, x, x])
U₀, V₀, σ₁, σ₂ = symbols("U_0 V_0 sigma_1 sigma_2", real = true)
by_power = inv(D₄(r₂)) * subs.(P₃, ξ => r₂ / r₁) * D₄(r₁) * [U₀, V₀, σ₁, σ₂]
by_modes = collect(LC._shear_cross(r₁, r₂, L, (U₀, V₀, σ₁, σ₂))[2])
@assert all(iszero, simplify.(by_power .- by_modes))

# and, for an isotropic layer, the system of the package:

@assert all(iszero, simplify.(subs.(M₃, isotropic...) .- LC._shear_system(radial)))

# ## 5. One fiber in a matrix
#
# On the axis the solutions that stay bounded are those of the positive
# exponents, whose states are the corresponding eigenvectors. Take a fiber of
# axial shear modulus ``p_1`` and radius ``R`` in a matrix ``p_0``, under a
# remote antiplane shear. In the fiber the state is a multiple ``a`` of the
# regular one; in the matrix it is the uniform shear, of unit amplitude, plus a
# multiple ``b`` of the mode that decays. The continuity of the state at ``R``
# fixes both:

p₁, pₘ, a, b = symbols("p_1 p_0 a b", positive = true)
states(c) = mode_states(subs.(M₁, p₀ => c), (1, -1))
inside = states(p₁)[:, 1]
uniform_shear, decaying = eachcol(states(pₘ))
amplitudes = solve(a .* inside .- (uniform_shear .+ b .* decaying), [a, b])
factor(amplitudes[a])

# The strain of the fiber per unit remote strain, ``2p_0/(p_0 + p_1)``, which
# the package finds from the amplitudes of the modes:

ez = (Sym(0), Sym(0), Sym(1))
fiber = LayeredCylinder((Sym(1),), (ti_from_constants(k₀, l₀, m₀, n₀, p₁, ez; form = :hill),))
ϕ_R = LC._elastic_solution(fiber, ti_from_constants(k₀, l₀, m₀, n₀, pₘ, ez; form = :hill)).antiplane.outside[1][1]
@assert iszero(simplify(ϕ_R - amplitudes[a]))

# ## Where to go next
#
# - [Layered spheres and cylinders: the radial problem](@ref th-radial-problems):
#   the two strategies, and the layers each one reaches.
# - [Layered cylinder](@ref th-layered-cylinder): the statement of these
#   systems, the interfaces and the averages.
# - [Layered cylinder: layers transversely isotropic about the radius, symbolically](@ref tut-layered-cylinder-radial-symbolic):
#   the same systems for a wound layer, and a layer graded as a power of the
#   radius in closed form.
# - [Graded isotropic layers with interfaces at given radii](@ref tut-graded-layers-interfaces):
#   the systems integrated numerically when the moduli vary with the radius.
