# # [Hill tensors by reduction to a sphere](@id tut-hill-reduction)
#
# !!! info "Before this page"
#     The tutorial [Hill polarization tensors in practice](@ref tut-hill-tensors),
#     where `hill_tensor` is called directly on each geometry and its numerical
#     back-ends are introduced, and the section
#     [Reduction to a sphere](@ref th-hill-reduction-sphere) of the theory page
#     on Hill tensors, where the relation used below is stated.
#
# The Hill tensor of an ellipsoid ``\boldsymbol{A}`` in a matrix ``\mathbb{C}``
# is that of a **sphere** in the stretched medium
# ``\tilde{\mathbb{C}} = \mathbb{B}:\mathbb{C}:\mathbb{B}``, carried back by the
# same tensor ``\mathbb{B} = \boldsymbol{A}^{-1}\stackrel{s}{\boxtimes}\boldsymbol{A}^{-1}``
# [barthelemyIJES2020_hilltrans](@cite):
#
# ```math
# \mathbb{P}(\boldsymbol{A},\mathbb{C})
# = \mathbb{B}:\mathbb{P}(\boldsymbol{1},\tilde{\mathbb{C}}):\mathbb{B}.
# ```
#
# When the stretched medium belongs to a class whose spherical Hill tensor has a
# closed form, the reduction is a closed form for the ellipsoid too. This page
# computes three Hill tensors that way and holds each to the three numerical
# back-ends of the package, applied to the ellipsoid itself: the residue
# algorithm (`:residues`), the adaptive cubature DECUHR (`:decuhr`) and nested
# Gauss-Kronrod quadratures (`:nestedquadgk`). Everything is written in tensor
# algebra: `⊠ˢ` for the symmetrized box product, `⊡` for the double
# contraction, `inv` of a tensor, and no component is read by index.

import Pkg                                                          #jl
Pkg.activate(joinpath(@__DIR__, "..", "docs"); io = devnull)                 #jl

using MeanFieldHomogenization
using TensND
using LinearAlgebra
using Printf
import DECUHR, Integrals          # the `:decuhr` back-end lives in an extension

# ## The reduction, and the numerical back-ends to hold it to
#
# [`shape_tensor`](@ref MeanFieldHomogenization.Core.shape_tensor) gives the
# symmetric ``\boldsymbol{A} = \boldsymbol{R}\cdot\mathrm{diag}(a,b,c)\cdot\boldsymbol{R}^{T}``
# of an ellipsoid, its semi-axes and their orientation, from which the stretch
# ``\mathbb{B}`` follows. How the Hill tensor of the unit sphere in
# ``\tilde{\mathbb{C}}`` is computed is passed as a function, since that is
# where each example takes its closed form.

stretch(ell) = (Ainv = inv(shape_tensor(ell)); Ainv ⊠ˢ Ainv)

function hill_by_reduction(ell, C, sphere_hill)
    B = stretch(ell)
    return B ⊡ sphere_hill(B ⊡ C ⊡ B) ⊡ B
end

relgap(P, Q) = norm(P - Q) / norm(Q)

const TOL = (; abstol = 1.0e-12, reltol = 1.0e-10)

function against_numerics(P_reduced, ell, C)
    for method in (:residues, :decuhr, :nestedquadgk)
        P = hill_tensor(ell, C; method, TOL...)
        @printf "  reduction against %-13s %.1e\n" method relgap(P_reduced, P)
    end
    return nothing
end
nothing #hide

# ## 1. A spheroid in a coaxial transversely isotropic matrix
#
# The matrix of [sevostianovIJSS2005; eq. 2.34](@citet), also that of Figure 1 of
# [barthelemyIJES2020_hilltrans](@citet), and an oblate spheroid of aspect ratio
# ``\omega = 0.2`` sharing its axis ``\underline{e}_3``.

const e₃ = (0.0, 0.0, 1.0)
C_ti = TensND.tens_ti(2.179, 0.579, 0.689, 10.345, 1.0, e₃)
ω = 0.2
spheroid = Spheroid(ω)
B = stretch(spheroid)
nothing #hide

# The stretch keeps the class: ``\tilde{\mathbb{C}}`` is transversely isotropic
# about the same axis, its Walpole coefficients being those of ``\mathbb{C}``
# divided by ``\omega`` to the number of indices equal to 3 in the components
# they gather, ``(\omega^{-4}, 1, \omega^{-2}, 1, \omega^{-2})``
# [barthelemyIJES2020_hilltrans; eq. 19](@cite). Its best transversely isotropic
# fit is exact, and the ratio of the coefficients shows the powers of
# ``\omega``:

C̃ = best_fit_ti(B ⊡ C_ti ⊡ B, e₃)
@printf "‖𝔹:ℂ:𝔹 − its TI fit‖ / ‖𝔹:ℂ:𝔹‖ = %.1e\n" relgap(C̃, B ⊡ C_ti ⊡ B)
collect(TensND.get_data(C̃) ./ TensND.get_data(C_ti)) .* [ω^4, 1, ω^2, 1, ω^2]

# The sphere in a transversely isotropic medium has a closed form, the case
# ``\omega = 1`` of [the coaxial spheroid](@ref th-hill-ti-coaxial); carried
# back, it is the Hill tensor of the spheroid, with no quadrature.

P_reduced = hill_by_reduction(spheroid, C_ti, C̃ -> hill_tensor(Ellipsoid(1.0), best_fit_ti(C̃, e₃)))
against_numerics(P_reduced, spheroid, C_ti)

# ## 2. An elliptically orthotropic matrix, from the isotropic closed form
#
# The elliptically orthotropic matrix
# ``\mathbb{C} = \lambda\,\boldsymbol{D}\otimes\boldsymbol{D}
# + 2\mu\,\boldsymbol{D}\stackrel{s}{\boxtimes}\boldsymbol{D}``, with
# ``\boldsymbol{D}`` symmetric positive definite, is mapped by the stretch
# ``\boldsymbol{D}^{-1/2}`` onto the isotropic
# ``\lambda\,\boldsymbol{1}\otimes\boldsymbol{1} + 2\mu\,\boldsymbol{1}\stackrel{s}{\boxtimes}\boldsymbol{1}``,
# and the ellipsoid ``\boldsymbol{A}`` onto
# ``\tilde{\boldsymbol{A}} = \boldsymbol{A}\cdot\boldsymbol{D}^{-1/2}``
# [barthelemyIJES2020_hilltrans; eq. 14](@cite):
#
# ```math
# \mathbb{P}(\boldsymbol{A},\mathbb{C})
# = \bigl(\boldsymbol{D}^{-1/2}\stackrel{s}{\boxtimes}\boldsymbol{D}^{-1/2}\bigr)
# :\mathbb{P}(\tilde{\boldsymbol{A}},\mathbb{C}^{\mathrm{iso}}):
# \bigl(\boldsymbol{D}^{-1/2}\stackrel{s}{\boxtimes}\boldsymbol{D}^{-1/2}\bigr).
# ```
#
# With ``\boldsymbol{D}`` coaxial with the ellipsoid, ``\tilde{\boldsymbol{A}}``
# is the ellipsoid of semi-axes ``a_i/\sqrt{d_i}``, whose Hill tensor in an
# isotropic matrix is a closed form in elliptic integrals: the matrix is
# orthotropic, and its Hill tensor is a closed form all the same.

λ, μ = 2.0, 1.5
d = (1.6, 1.0, 0.7)
D = TensND.Tens(Diagonal(collect(d)))
C_eo = λ * (D ⊗ D) + 2μ * (D ⊠ˢ D)
semi_axes = (2.0, 1.0, 0.5)
ellipsoid = Ellipsoid(semi_axes...)

Dmh = TensND.Tens(Diagonal(collect(1 ./ sqrt.(d))))
Bd = Dmh ⊠ˢ Dmh
C_iso = TensISO{3}(3λ + 2μ, 2μ)                 # λ 𝟏⊗𝟏 + 2μ 𝟏⊠ˢ𝟏
P_reduced = Bd ⊡ hill_tensor(Ellipsoid((semi_axes ./ sqrt.(d))...), C_iso) ⊡ Bd
against_numerics(P_reduced, ellipsoid, C_eo)

# ## 3. A rotated triaxial ellipsoid in a triclinic matrix
#
# With no class left to keep, a triclinic matrix and a rotated ellipsoid of
# three distinct semi-axes, the stretched medium is triclinic as well and the
# sphere in it needs a numerical method too. The reduction still pays: on the
# sphere the denominator ``\|\boldsymbol{A}\cdot\underline{\xi}\|^{3}`` of the
# ``\underline{\xi}`` form of the Hill tensor is one. Here the sphere is treated
# by the residue algorithm.

M = [
    210.0 80.0 75.0 5.0 4.0 3.0;
    80.0 195.0 90.0 -2.0 3.0 -1.0;
    75.0 90.0 220.0 1.0 -2.0 2.0;
    5.0 -2.0 1.0 60.0 2.5 1.5;
    4.0 3.0 -2.0 2.5 65.0 -1.0;
    3.0 -1.0 2.0 1.5 -1.0 55.0
]
C_tri = TensND.from_mandel(M, CanonicalBasis{3, Float64}())
rotated = Ellipsoid(2.0, 1.0, 0.5; euler_angles = (0.3, 0.8, -0.5))

P_reduced = hill_by_reduction(rotated, C_tri, C̃ -> hill_tensor(Ellipsoid(1.0), C̃; method = :residues, TOL...))
against_numerics(P_reduced, rotated, C_tri)

# The three back-ends agree with the reduction to the tolerance asked for, and
# in the first two examples the reduction needs no quadrature at all.

# ## Where to go next
#
# Every tensor above is the Hill tensor of one inclusion with a fixed
# orientation. [Symmetrization showcase: exact average vs best-fit projection](@ref tut-symmetrization)
# averages such tensors over the orientations of a population, and compares the
# exact average with a best-fit projection onto a symmetry class. The closed
# form of the first example is written out in
# [Transversely isotropic matrix coaxial with a spheroid](@ref th-hill-ti-coaxial),
# and the reduction itself in [Reduction to a sphere](@ref th-hill-reduction-sphere).
