# # [One derivative, several routes](@id tut-derivative-routes)
#
# !!! info "Before this page"
#     [Sensitivities](@ref man-sensitivities), for the parameters and the
#     symmetry classes their coefficients belong to.
#
# A stiffness tensor has as many parametrizations as there are symmetry classes
# that contain it: an isotropic tensor is also transversely isotropic about any
# axis, orthotropic in any frame, and it always has its 21 Kelvin-Mandel
# components. Each parametrization gives derivatives of an effective property,
# and they are tied by the chain rule. This page computes the derivative of one
# effective component through each class and checks them against one another;
# it ends with a derivative that takes the tensor out of its own class.

import Pkg                                                          #jl
Pkg.activate(joinpath(@__DIR__, "..", "docs"); io = devnull)        #jl

using MeanFieldHomogenization
using TensND
using ForwardDiff
using Printf

# ## The cell
#
# Spheres of an isotropic phase ``\mathbb{C}_I`` in an isotropic matrix, by the
# Mori-Tanaka scheme. The quantity differentiated is the component
# ``f = C^{\mathrm{hom}}_{3333}`` of the effective stiffness.

C_I = iso_stiffness(40.0, 30.0)                          # k, μ
function cell(C)
    rve = RVE()
    add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => iso_stiffness(10.0, 6.0)); fraction = :rest)
    add_phase!(rve, :I, Ellipsoid(1.0), Dict(:C => C); fraction = 0.3)
    return rve
end
f(C) = canonical_components(C)[3, 3, 3, 3]
∂(p) = derivative(cell(C_I), MoriTanaka(), p; indexer = f)
nothing #hide

# ## The chain rule
#
# Let ``p`` be a coefficient of any class that contains ``\mathbb{C}_I``, and
# ``M_{ij}(p)`` the Kelvin-Mandel matrix of the tensor in which ``p`` varies, the
# other coefficients of its class held fixed. Since ``f`` depends on
# ``\mathbb{C}_I`` through ``M``,
#
# ```math
# \frac{\partial f}{\partial p}=\sum_{i\le j}\frac{\partial f}{\partial M_{ij}}\,\frac{\partial M_{ij}}{\partial p},
# ```
#
# where ``\partial f/\partial M_{ij}``, for ``i<j``, moves ``M_{ij}`` and
# ``M_{ji}`` together, which is what the parameter `:Mij` does. The 21
# derivatives with respect to the Kelvin-Mandel components are computed once:

D = Dict((i, j) => ∂(property(:I, :C, Symbol("M$i$j"))) for i in 1:6 for j in i:6)
nothing #hide

# and ``\partial M/\partial p`` is the derivative of the tensor alone, without
# any homogenization, by ForwardDiff on the function that builds it:

km(C) = mandel(C, TensND.CanonicalBasis{3, eltype(C)}())
by_chain_rule(tensor, p₀) = (dM = ForwardDiff.derivative(x -> km(tensor(x)), p₀); sum(D[(i, j)] * dM[i, j] for i in 1:6 for j in i:6))
nothing #hide

# ## The isotropic class
#
# The moduli ``k`` and ``\mu`` of ``\mathbb{C}_I = 3k\,\mathbb{J} + 2\mu\,\mathbb{K}``,
# by their names and by the chain rule:

@printf "∂f/∂k :  by name %.12f   by the chain rule %.12f\n" ∂(property(:I, :C, :k)) by_chain_rule(k -> iso_stiffness(k, 30.0), 40.0)
@printf "∂f/∂μ :  by name %.12f   by the chain rule %.12f\n" ∂(property(:I, :C, :μ)) by_chain_rule(μ -> iso_stiffness(40.0, μ), 30.0)

# ## The same tensor, transversely isotropic
#
# About an axis ``\underline{n}``, the isotropic tensor has five Walpole
# coefficients ``(\ell_1, \ell_2, \ell_3 = \ell_4, \ell_5, \ell_6)``.
# `symmetry = (:TI, n)` writes it in that class first:

n = TensND.normalize([0.0, 1.0, 1.0])
T = first(TensND.project(C_I, :TI, n))
ℓ = TensND.get_data(T)
names = (:ℓ₁, :ℓ₂, :ℓ₃, :ℓ₅, :ℓ₆)
∂ℓ = [∂(property(:I, :C, s; symmetry = (:TI, n))) for s in names]
for (r, s) in enumerate(names)
    tensor(x) = TensND.rebuild(T, Base.setindex(ℓ, x, r))
    @printf "∂f/∂%s :  by name %.12f   by the chain rule %.12f\n" s ∂ℓ[r] by_chain_rule(tensor, ℓ[r])
end

# A modulus of the isotropic class is a function of the Walpole coefficients,
# and the chain rule between the two classes gives ``\partial f/\partial k``
# a third time:

dℓ_dk = ForwardDiff.derivative(k -> collect(TensND.get_data(first(TensND.project(iso_stiffness(k, 30.0), :TI, n)))), 40.0)
@printf "∂f/∂k = Σ ∂f/∂ℓᵣ ∂ℓᵣ/∂k = %.12f\n" sum(∂ℓ .* dℓ_dk)

# ## Orthotropic components and Kelvin-Mandel components
#
# In an orthotropic frame the coefficient ``C_{1313}`` is a component of the
# tensor, while ``M_{55} = 2C_{1313}`` is its Kelvin-Mandel component: the
# derivatives with respect to them differ by that factor. Here the effective
# component differentiated is ``C^{\mathrm{hom}}_{1313}``, which this shear moves.

frame = TensND.CanonicalBasis{3, Float64}()
g(C) = canonical_components(C)[1, 3, 1, 3]
∂g(p) = derivative(cell(C_I), MoriTanaka(), p; indexer = g)
@printf "∂g/∂C₁₃₁₃ = %.12f   2 ∂g/∂M₅₅ = %.12f\n" ∂g(property(:I, :C, :C1313; symmetry = (:ORTHO, frame))) 2 * ∂g(property(:I, :C, :M55))

# ## A Hill tensor, by three routes
#
# The same routes apply to any function of a tensor, here the Hill tensor
# ``\mathbb{P}`` of a prolate spheroid of axis ``\underline{e}_3`` in the isotropic
# matrix ``\mathbb{C}_0``, differentiated with respect to the bulk modulus ``k_0``
# of the matrix. [`get_coefficient`](@ref) and [`set_coefficient`](@ref) read and
# set a coefficient of a tensor, in a class that contains it, as the parameters
# do in a cell.

C₀ = iso_stiffness(10.0, 6.0)
spheroid = Ellipsoid(1.0, 1.0, 3.0)
P₃₃₃₃(C) = canonical_components(hill_tensor(spheroid, C))[3, 3, 3, 3]
nothing #hide

# **Isotropic.** ``k_0`` is a coefficient of the class of ``\mathbb{C}_0``:

by_k = ForwardDiff.derivative(k -> P₃₃₃₃(set_coefficient(C₀, :k, k)), 10.0)

# **Transversely isotropic.** About ``\underline{e}_3``, ``\mathbb{C}_0`` has
# five Walpole coefficients ``\ell_r``. The derivatives of ``P_{3333}`` with
# respect to them are taken with a matrix that is transversely isotropic, the
# other coefficients fixed; the chain rule through ``\partial\ell_r/\partial k_0``
# gives back the derivative with respect to ``k_0``:

e₃ = [0.0, 0.0, 1.0]
ℓ₀ = get_coefficient.(Ref(C₀), names; symmetry = (:TI, e₃))
∂P_∂ℓ = [ForwardDiff.derivative(x -> P₃₃₃₃(set_coefficient(C₀, s, x; symmetry = (:TI, e₃))), ℓ₀[r]) for (r, s) in enumerate(names)]
∂ℓ_∂k = ForwardDiff.derivative(k -> [get_coefficient(iso_stiffness(k, 6.0), s; symmetry = (:TI, e₃)) for s in names], 10.0)
by_TI = sum(∂P_∂ℓ .* ∂ℓ_∂k)

# **Anisotropic.** Seen as a general anisotropic tensor, ``\mathbb{C}_0`` has 21
# independent Kelvin-Mandel components. One call of `ForwardDiff.gradient` gives
# the 21 derivatives at once, the matrix being anisotropic as soon as one
# component moves; the chain rule through ``\partial M_{ij}/\partial k_0``
# closes the loop:

pairs = [(i, j) for i in 1:6 for j in i:6]
with_components(m) = foldl((C, ((i, j), x)) -> set_coefficient(C, Symbol("M$i$j"), x), zip(pairs, m); init = C₀)
∂P_∂M = ForwardDiff.gradient(m -> P₃₃₃₃(with_components(m)), [km(C₀)[i, j] for (i, j) in pairs])
∂M_∂k = ForwardDiff.derivative(k -> km(iso_stiffness(k, 6.0)), 10.0)
by_ANISO = sum(∂P_∂M[r] * ∂M_∂k[i, j] for (r, (i, j)) in enumerate(pairs))

(; by_k, by_TI, by_ANISO)

# The three numbers agree to rounding: each route is exact, and only the class
# in which the matrix is allowed to move differs.
#
# ## Out of the isotropic class
#
# ![](../../assets/theory/parameter_classes.svg)
#
# Moving ``\ell_5`` alone perturbs ``\mathbb{C}_I`` by a transversely isotropic
# tensor about ``\underline{n}``: the inclusion is no longer isotropic, and
# neither is the derivative of the effective stiffness. `jacobian` gives all its
# components; their projections show the class it belongs to, the relative
# distance being ``0`` for a class that contains it.

J = jacobian(cell(C_I), MoriTanaka(), [property(:I, :C, :ℓ₅; symmetry = (:TI, n))])
∂C = Tens(reshape(J[:, 1], 3, 3, 3, 3))
@printf "relative distance of ∂Cʰᵒᵐ/∂ℓ₅ to the isotropic class %.2e, to TI(n) %.2e\n" last(TensND.project(∂C, :ISO)) last(TensND.project(∂C, :TI, n))

# The derivative with respect to ``k``, by contrast, stays isotropic:

J = jacobian(cell(C_I), MoriTanaka(), [property(:I, :C, :k)])
@printf "relative distance of ∂Cʰᵒᵐ/∂k to the isotropic class %.2e\n" last(TensND.project(Tens(reshape(J[:, 1], 3, 3, 3, 3)), :ISO))

# This is the situation of the strength criterion of
# [Quasi-brittle strength of cement paste and mortar](@ref app-strength), where
# the modulus of the hydrate needles of a single orientation is perturbed.
