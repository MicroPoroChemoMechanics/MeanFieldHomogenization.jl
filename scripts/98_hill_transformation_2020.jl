# # [Hill tensors by transformation: closed forms derived step by step](@id app-hill-transformation)
#
# !!! info "Before this page"
#     The section [Reduction to a sphere](@ref th-hill-reduction-sphere) and the
#     section [Transversely isotropic matrix coaxial with a spheroid](@ref th-hill-ti-coaxial)
#     of the theory page on Hill tensors, which state the results this page
#     derives, and the tutorial [Hill tensors by reduction to a sphere](@ref tut-hill-reduction),
#     which drives the reduction through the library.
#
# [barthelemyIJES2020_hilltrans](@citet) shows that the Hill polarization tensor
# of any ellipsoid is that of a sphere in a transformed medium, and draws two
# closed forms from it: an ellipsoid in an elliptically orthotropic matrix, and a
# spheroid in a coaxial transversely isotropic matrix, with the flat and needle
# limits of the latter. This page goes through the paper in its order, with the
# equation numbers of the article, and checks every step in one of three ways:
#
# - **by computation**, against the definition of the Hill tensor integrated
#   on the unit sphere and against the numerical back-ends of the package;
# - **by symbolic computation**, with SymPy, for the identities the derivation
#   rests on: they hold for free symbols, not for a sample of values;
# - **by a transcription in Julia of the Python script appended to the paper**,
#   whose Figure 1 is drawn again here, against the closed form the package
#   evaluates.
#
# Everything is written in the tensor algebra of TensND: `⊗` and `⊠ˢ` for the
# tensor products, `⋅` and `⊡` for the simple and double contractions, `⊙` for
# the full contraction of two fourth-order tensors, `inv` of a tensor. No
# component is read by index except to draw the figure.

import Pkg                                                          #jl
Pkg.activate(joinpath(@__DIR__, "..", "docs"); io = devnull)                 #jl

using MeanFieldHomogenization
using TensND
using LinearAlgebra
using Printf
using QuadGK: gauss
using ForwardDiff
using SymPyPythonCall
using Plots
gr()
default(; left_margin = 5Plots.mm, bottom_margin = 5Plots.mm)

relgap(P, Q) = norm(P - Q) / norm(Q)
nothing #hide

# ## [1. The transformation (Section 2 of the paper)](@id app-hill-transformation-reduction)
#
# ### The Hill tensor as an integral over the unit sphere
#
# The paper starts from the expression of the Hill tensor of the ellipsoid
# ``\boldsymbol{x}\cdot(\boldsymbol{A}^{T}\cdot\boldsymbol{A})^{-1}\cdot\boldsymbol{x}\le1``
# in a matrix ``\mathbb{C}`` as an integral over the unit sphere (eq. 1):
#
# ```math
# \mathbb{P}(\boldsymbol{A},\mathbb{C}) = \frac{1}{4\pi}\int_{\|\underline{\zeta}\|=1}
# \underline{y}\stackrel{s}{\otimes}\bigl(\underline{y}\cdot\mathbb{C}\cdot\underline{y}\bigr)^{-1}
# \stackrel{s}{\otimes}\underline{y}\,\mathrm{d}S_{\zeta},
# \qquad \underline{y} = \boldsymbol{A}^{-1}\cdot\underline{\zeta},
# ```
#
# where ``\boldsymbol{A}`` is any invertible tensor, symmetric or not. The
# product ``\underline{a}\stackrel{s}{\otimes}\boldsymbol{N}\stackrel{s}{\otimes}\underline{a}``
# symmetrizes ``a_i N_{jk} a_l`` over the first two and the last two indices.
# TensND's `⊠ˢ` symmetrizes over the last two only, so that the four terms are
# the half sum of ``(\underline{a}\otimes\underline{a})\stackrel{s}{\boxtimes}\boldsymbol{N}``
# and ``\boldsymbol{N}\stackrel{s}{\boxtimes}(\underline{a}\otimes\underline{a})``:

sandwich(a, N) = ((a ⊗ a) ⊠ˢ N + N ⊠ˢ (a ⊗ a)) / 2
hill_integrand(y, C) = sandwich(y, inv(y ⋅ C ⋅ y))
nothing #hide

# The integrand is a smooth function on the sphere. A Gauss-Legendre rule in
# ``z = \cos\theta`` and the trapezoidal rule in the longitude ``\varphi``, exact
# for the trigonometric polynomials the odd powers of ``\sqrt{1-z^2}`` come
# with, converge exponentially; `f` receives the unit vector and returns the
# tensor to average:

function sphere_average(f; nz = 96, nφ = 192)
    zs, ws = gauss(nz)
    return sum(
        w * f(TensND.Tens([sqrt(1 - z^2) * cos(φ), sqrt(1 - z^2) * sin(φ), z]))
            for (z, w) in zip(zs, ws) for φ in range(0, 2π; length = nφ + 1)[1:(end - 1)]
    ) * (2π / nφ) / (4π)
end

hill_by_definition(A, C) = (Ainv = inv(A); sphere_average(ζ -> hill_integrand(Ainv ⋅ ζ, C)))
nothing #hide

# The matrix and the shape of this section are deliberately generic: a
# triclinic stiffness, and a non-symmetric ``\boldsymbol{A}``, a rotation
# ``\boldsymbol{Q}`` times a symmetric tensor ``\boldsymbol{S}`` of semi-axes
# ``(2, 1, 1/2)``. As ``\boldsymbol{A}^{T}\cdot\boldsymbol{A} = \boldsymbol{S}^2``,
# the ellipsoid is that of ``\boldsymbol{S}``, which the package builds from its
# semi-axes and principal directions:

C_tri = TensND.from_mandel(
    [
        210.0 80.0 75.0 5.0 4.0 3.0;
        80.0 195.0 90.0 -2.0 3.0 -1.0;
        75.0 90.0 220.0 1.0 -2.0 2.0;
        5.0 -2.0 1.0 60.0 2.5 1.5;
        4.0 3.0 -2.0 2.5 65.0 -1.0;
        3.0 -1.0 2.0 1.5 -1.0 55.0
    ], CanonicalBasis{3, Float64}()
)
R = Matrix(rotation_matrix(0.3, 0.8, -0.5))
S = TensND.Tens(Symmetric(R * Diagonal([2.0, 1.0, 0.5]) * transpose(R)))
Q = TensND.Tens(Matrix(rotation_matrix(1.1, -0.4, 0.7)))
A = Q ⋅ S

"Ellipsoid of `A`: semi-axes and axes from the eigenpairs of `Aᵀ⋅A`."
function ellipsoid_of(A)
    F = eigen(Symmetric(Array(canonical_components(transpose(A) ⋅ A))))
    V = F.vectors * Diagonal([1, 1, sign(det(F.vectors))])      # a rotation
    return Ellipsoid(sqrt.(F.values)..., V)
end

P_ref = hill_tensor(ellipsoid_of(A), C_tri; method = :residues, abstol = 1.0e-13, reltol = 1.0e-12)
@printf "eq. 1 on the unit sphere against the residues:   %.1e\n" relgap(hill_by_definition(A, C_tri), P_ref)

# The rotation ``\boldsymbol{Q}`` only rotates the integration variable, so it
# leaves the Hill tensor unchanged, as the definition says:

@printf "P(Q⋅S, C) against P(S, C):                       %.1e\n" relgap(hill_by_definition(A, C_tri), hill_by_definition(S, C_tri))

# The change of variable ``\underline{\xi} = \boldsymbol{A}\cdot\underline{y}/\|\cdot\|``
# gives the other classical form (eq. 9), in which the shape appears through a
# denominator:
#
# ```math
# \mathbb{P}(\boldsymbol{A},\mathbb{C}) = \frac{\det\boldsymbol{A}}{4\pi}\int_{\|\underline{\xi}\|=1}
# \frac{\underline{\xi}\stackrel{s}{\otimes}(\underline{\xi}\cdot\mathbb{C}\cdot\underline{\xi})^{-1}
# \stackrel{s}{\otimes}\underline{\xi}}{\|\boldsymbol{A}\cdot\underline{\xi}\|^{3}}\,\mathrm{d}S_{\xi}.
# ```

hill_by_xi(A, C) = det(A) * sphere_average(ξ -> hill_integrand(ξ, C) / norm(A ⋅ ξ)^3; nz = 160, nφ = 320)
@printf "eq. 9 against the residues:                      %.1e\n" relgap(hill_by_xi(S, C_tri), P_ref)

# The denominator ``\|\boldsymbol{A}\cdot\underline{\xi}\|^{3}`` varies by a
# factor 64 over the sphere for these semi-axes, and the same rule needs more
# points for the same accuracy: this is the difficulty the transformation
# removes.
#
# ### Moving the shape onto the stiffness
#
# In eq. 1 the shape enters through ``\boldsymbol{A}^{-1}\cdot\underline{\zeta}``
# only. Writing ``\mathbb{B} = \boldsymbol{A}^{-1}\stackrel{s}{\boxtimes}\boldsymbol{A}^{-1}``,
# the acoustic tensor and the sandwich product become
# ``\underline{y}\cdot\mathbb{C}\cdot\underline{y} = \underline{\zeta}\cdot(\mathbb{B}^{T}:\mathbb{C}:\mathbb{B})\cdot\underline{\zeta}``
# and ``\underline{y}\stackrel{s}{\otimes}\boldsymbol{N}\stackrel{s}{\otimes}\underline{y}
# = \mathbb{B}:(\underline{\zeta}\stackrel{s}{\otimes}\boldsymbol{N}\stackrel{s}{\otimes}\underline{\zeta}):\mathbb{B}^{T}``
# (eq. 3). Any two pairs ``(\boldsymbol{A},\mathbb{C})`` and
# ``(\tilde{\boldsymbol{A}},\tilde{\mathbb{C}})`` that give the same
# ``\mathbb{B}^{T}:\mathbb{C}:\mathbb{B}`` (eq. 4) therefore have Hill tensors
# related by (eq. 7)
#
# ```math
# \begin{gathered}
# \mathbb{P}(\boldsymbol{A},\mathbb{C}) = \mathbb{M}:\mathbb{P}(\tilde{\boldsymbol{A}},\tilde{\mathbb{C}}):\mathbb{M}^{T},
# \\
# \mathbb{M} = (\boldsymbol{A}^{-1}\cdot\tilde{\boldsymbol{A}})\stackrel{s}{\boxtimes}(\boldsymbol{A}^{-1}\cdot\tilde{\boldsymbol{A}}),
# \qquad
# \tilde{\mathbb{C}} = \mathbb{M}^{T}:\mathbb{C}:\mathbb{M}.
# \end{gathered}
# ```
#
# The transposition is the major one, ``(\mathbb{M}^{T})_{ijkl} = \mathbb{M}_{klij}``,
# and ``\tilde{\mathbb{C}}`` is a stiffness whenever ``\mathbb{C}`` is one. The
# relation holds for any invertible ``\tilde{\boldsymbol{A}}``; here a second
# rotated triaxial shape:

function hill_by_transformation(A, C, Ã, hill_of)
    M = (inv(A) ⋅ Ã) ⊠ˢ (inv(A) ⋅ Ã)
    Mᵀ = TensND.Tensors.majortranspose(M)
    return M ⊡ hill_of(Ã, Mᵀ ⊡ C ⊡ M) ⊡ Mᵀ
end

Ã = TensND.Tens(Matrix(rotation_matrix(-0.2, 1.3, 0.4))) ⋅
    TensND.Tens(Diagonal([1.0, 3.0, 1.5]))
@printf "eq. 7 with an arbitrary Ã against the residues:  %.1e\n" relgap(hill_by_transformation(A, C_tri, Ã, hill_by_definition), P_ref)

# With ``\tilde{\boldsymbol{A}} = \boldsymbol{1}`` the transformed shape is the
# unit sphere (eq. 8), the [reduction to a sphere](@ref th-hill-reduction-sphere):
# **every Hill tensor is that of a sphere in the medium**
# ``\tilde{\mathbb{C}} = \mathbb{B}^{T}:\mathbb{C}:\mathbb{B}``. A numerical
# method may treat the sphere only, where ``\|\boldsymbol{A}\cdot\underline{\xi}\|=1``;
# here the residue algorithm of the package takes the sphere:

sphere_by_residues(_, C̃) = hill_tensor(Ellipsoid(1.0), C̃; method = :residues, abstol = 1.0e-13, reltol = 1.0e-12)
P_sph = hill_by_transformation(A, C_tri, TensISO{3}(1.0), sphere_by_residues)
@printf "eq. 8, the sphere by residues:                   %.1e\n" relgap(P_sph, P_ref)

# ## [2. An ellipsoid in an elliptically orthotropic matrix (Section 3)](@id app-hill-transformation-eo)
#
# The elliptically orthotropic matrix (eq. 10)
# ``\mathbb{C} = \lambda\,\boldsymbol{D}\otimes\boldsymbol{D} + 2\mu\,\boldsymbol{D}\stackrel{s}{\boxtimes}\boldsymbol{D}``,
# with ``\boldsymbol{D}`` symmetric positive definite, is isotropic for
# ``\boldsymbol{D} = \boldsymbol{1}``. The algebra of the tensor products
# carries it through the stretch (eq. 11):
#
# ```math
# \mathbb{B}^{T}:\mathbb{C}:\mathbb{B}
# = \lambda\,\boldsymbol{D}'\otimes\boldsymbol{D}' + 2\mu\,\boldsymbol{D}'\stackrel{s}{\boxtimes}\boldsymbol{D}',
# \qquad \boldsymbol{D}' = \boldsymbol{A}^{-T}\cdot\boldsymbol{D}\cdot\boldsymbol{A}^{-1}.
# ```
#
# Take ``\boldsymbol{D}`` with principal directions of its own, unrelated to
# those of the ellipsoid:

λ, μ = 2.0, 1.5
RD = Matrix(rotation_matrix(0.7, 0.5, 1.9))
D = TensND.Tens(Symmetric(RD * Diagonal([1.6, 1.0, 0.7]) * transpose(RD)))
C_eo = λ * (D ⊗ D) + 2μ * (D ⊠ˢ D)

B = inv(A) ⊠ˢ inv(A)
Bᵀ = TensND.Tensors.majortranspose(B)
D′ = transpose(inv(A)) ⋅ D ⋅ inv(A)
@printf "eq. 11:  %.1e\n" relgap(Bᵀ ⊡ C_eo ⊡ B, λ * (D′ ⊗ D′) + 2μ * (D′ ⊠ˢ D′))

# With the square roots ``\boldsymbol{D}^{\pm1/2}`` taken on the principal
# directions of ``\boldsymbol{D}`` and ``\tilde{\boldsymbol{A}} = \boldsymbol{A}\cdot\boldsymbol{D}^{-1/2}``
# (eq. 12), ``\boldsymbol{D}' = \tilde{\boldsymbol{A}}^{-T}\cdot\tilde{\boldsymbol{A}}^{-1}``:
# eq. 4 holds with the isotropic
# ``\tilde{\mathbb{C}} = \lambda\,\boldsymbol{1}\otimes\boldsymbol{1} + 2\mu\,\boldsymbol{1}\stackrel{s}{\boxtimes}\boldsymbol{1}``
# (eq. 13), and eq. 7 gives (eq. 14)
#
# ```math
# \mathbb{P}(\boldsymbol{A},\mathbb{C}) =
# (\boldsymbol{D}^{-1/2}\stackrel{s}{\boxtimes}\boldsymbol{D}^{-1/2}):
# \mathbb{P}(\tilde{\boldsymbol{A}},\tilde{\mathbb{C}}):
# (\boldsymbol{D}^{-1/2}\stackrel{s}{\boxtimes}\boldsymbol{D}^{-1/2}).
# ```
#
# The isotropic Hill tensor of the ellipsoid ``\tilde{\boldsymbol{A}}`` has a
# closed form in elliptic integrals, which the package evaluates: the Hill
# tensor of an arbitrary ellipsoid in an elliptically orthotropic matrix is a
# closed form too.

D_mh = TensND.Tens(Symmetric(inv(sqrt(Symmetric(Array(canonical_components(D)))))))
Ã_eo = A ⋅ D_mh
@printf "D′ = Ã⁻ᵀ⋅Ã⁻¹:  %.1e\n" relgap(D′, transpose(inv(Ã_eo)) ⋅ inv(Ã_eo))

C_iso = TensISO{3}(3λ + 2μ, 2μ)              # λ 𝟏⊗𝟏 + 2μ 𝟏⊠ˢ𝟏
B_D = D_mh ⊠ˢ D_mh
P_eo = B_D ⊡ hill_tensor(ellipsoid_of(Ã_eo), C_iso) ⊡ B_D
P_eo_num = hill_tensor(ellipsoid_of(A), C_eo; method = :residues, abstol = 1.0e-13, reltol = 1.0e-12)
@printf "eq. 14 against the residues:  %.1e\n" relgap(P_eo, P_eo_num)

# Conversely, eq. 5 maps an isotropic ``\tilde{\mathbb{C}}`` onto elliptically
# orthotropic matrices only: the method reaches no other symmetry from the
# isotropic case, as [pouya2000, pouya2006](@citet) had observed.
#
# ## [3. A spheroid in a coaxial transversely isotropic matrix (Section 4)](@id app-hill-transformation-ti)
#
# The spheroid has the axis ``\underline{e}_3`` and the aspect ratio
# ``\omega = c/a``; since the Hill tensor ignores a homothety of the shape,
# ``\boldsymbol{A} = \underline{e}_1\otimes\underline{e}_1 + \underline{e}_2\otimes\underline{e}_2
# + \omega\,\underline{e}_3\otimes\underline{e}_3`` (eq. 16). The matrix is
# transversely isotropic about the same axis, with the five constants
# ``C_{1111}, C_{1122}, C_{1133}, C_{3333}, C_{2323}``, and its Walpole
# decomposition (eq. 17) is the one TensND stores
# ([Walpole basis](@ref th-notation-bases)):
#
# ```math
# \begin{aligned}
# \mathbb{C} &= C_{3333}\,\mathbb{W}_1 + (C_{1111}+C_{1122})\,\mathbb{W}_2
# + \sqrt2\,C_{1133}\,(\mathbb{W}_3+\mathbb{W}_4)\\
# &\quad + (C_{1111}-C_{1122})\,\mathbb{W}_5 + 2\,C_{2323}\,\mathbb{W}_6 .
# \end{aligned}
# ```
#
# The derivation is carried out on free symbols. The Walpole tensors are
# orthogonal for ``\odot``, of squared norms ``1, 1, 1, 1, 2, 2``, which gives
# the coefficients of any transversely isotropic tensor without reading a
# component:

e₃ = (Sym(0), Sym(0), Sym(1))
W₁, W₂, W₃, W₄, W₅, W₆ = walpole_basis(e₃)
walpole(X) = (W₁ ⊙ X, W₂ ⊙ X, (W₃ + W₄) ⊙ X / 2, W₅ ⊙ X / 2, W₆ ⊙ X / 2)
(W₁ ⊙ W₁, W₂ ⊙ W₂, W₃ ⊙ W₃, W₄ ⊙ W₄, W₅ ⊙ W₅, W₆ ⊙ W₆, W₃ ⊙ W₄, W₁ ⊙ W₂)

# ### The transformed medium (eqs. 18–20)
#
# The stretch ``\mathbb{B}`` of the spheroid is diagonal, and
# ``\tilde{\mathbb{C}} = \mathbb{B}:\mathbb{C}:\mathbb{B}`` divides each
# component ``C_{ijkl}`` by ``\omega`` to the number of indices equal to 3
# (eq. 18). In the Walpole basis (eq. 19):

ω = symbols("omega", positive = true)
C₁₁, C₁₂, C₁₃, C₃₃, C₄₄ = symbols("C_1111 C_1122 C_1133 C_3333 C_2323", positive = true)
C_TI = ti_from_constants(C₁₁, C₁₂, C₁₃, C₃₃, C₄₄, e₃)
A_sph = TensND.Tens(Diagonal(Sym[1, 1, ω]))
B_sph = inv(A_sph) ⊠ˢ inv(A_sph)
ℓ̃ = walpole(B_sph ⊡ C_TI ⊡ B_sph)
ℓ̃ .- (C₃₃ / ω^4, C₁₁ + C₁₂, sqrt(Sym(2)) * C₁₃ / ω^2, C₁₁ - C₁₂, 2 * C₄₄ / ω^2) .|> simplify

# ``\tilde{\mathbb{C}}`` is thus transversely isotropic about ``\underline{e}_3``,
# of constants
# ``\tilde{C}_{1111} = C_{1111}``, ``\tilde{C}_{1122} = C_{1122}``,
# ``\tilde{C}_{1133} = C_{1133}/\omega^2``, ``\tilde{C}_{3333} = C_{3333}/\omega^4``,
# ``\tilde{C}_{2323} = C_{2323}/\omega^2``. So is the Hill tensor
# ``\tilde{\mathbb{P}} = \mathbb{P}(\boldsymbol{1},\tilde{\mathbb{C}})`` of the
# sphere, and carried back by eq. 8 the Hill tensor of the spheroid has the
# Walpole coefficients of ``\tilde{\mathbb{P}}`` divided by the same powers of
# ``\omega`` (eq. 20):
#
# ```math
# \mathbb{P} = \frac{\tilde p_1}{\omega^4}\,\mathbb{W}_1 + \tilde p_2\,\mathbb{W}_2
# + \frac{\tilde p_3}{\omega^2}\,(\mathbb{W}_3+\mathbb{W}_4) + \tilde p_5\,\mathbb{W}_5
# + \frac{\tilde p_6}{\omega^2}\,\mathbb{W}_6 .
# ```

p̃ = symbols("pt_1 pt_2 pt_3 pt_5 pt_6")
P̃_sym = TensND.TensTI{4}(p̃..., e₃)
walpole(B_sph ⊡ P̃_sym ⊡ B_sph) .- (p̃[1] / ω^4, p̃[2], p̃[3] / ω^2, p̃[4], p̃[5] / ω^2) .|> simplify

# ### The sphere in the transformed medium (eqs. 21–29)
#
# On the sphere, ``\underline{y} = \underline{\zeta}``. In the spherical
# coordinates of colatitude ``\theta`` and longitude ``\varphi``, with
# ``z = \cos\theta`` (eqs. 21–22),
#
# ```math
# \begin{gathered}
# \tilde{\mathbb{P}} = \frac{1}{4\pi}\int_{-1}^{1}\!\!\int_0^{2\pi}
# \underline{\zeta}\stackrel{s}{\otimes}\tilde{\boldsymbol{N}}\stackrel{s}{\otimes}\underline{\zeta}\,
# \mathrm{d}\varphi\,\mathrm{d}z,
# \qquad
# \tilde{\boldsymbol{N}} = (\underline{\zeta}\cdot\tilde{\mathbb{C}}\cdot\underline{\zeta})^{-1},
# \\
# \underline{\zeta} = z\,\underline{e}_3 + \sqrt{1-z^2}\,(\cos\varphi\,\underline{e}_1+\sin\varphi\,\underline{e}_2).
# \end{gathered}
# ```
#
# **The integral over the longitude.** Both the medium and the direction field
# are invariant by the rotations about ``\underline{e}_3``: the integrand at
# ``\varphi`` is the integrand at ``\varphi = 0`` rotated by ``\varphi``. Its
# average over ``\varphi`` is transversely isotropic, and its full contraction
# with each Walpole tensor, invariant by those rotations, is that of the
# integrand at ``\varphi = 0``. The integral over ``\varphi`` is therefore the
# projection of ``2\pi`` times the integrand at ``\varphi = 0`` on the Walpole
# basis (eq. 23), and the factor ``1/(4\pi)`` leaves a half:

z = symbols("z", real = true)
c̃₁₁, c̃₁₂, c̃₁₃, c̃₃₃, c̃₄₄ = (C₁₁, C₁₂, C₁₃ / ω^2, C₃₃ / ω^4, C₄₄ / ω^2)
C̃ = ti_from_constants(c̃₁₁, c̃₁₂, c̃₁₃, c̃₃₃, c̃₄₄, e₃)
ζ₀ = TensND.Tens(Sym[sqrt(1 - z^2), 0, z])
Ñ = inv(ζ₀ ⋅ C̃ ⋅ ζ₀)
p̃ᶻ = walpole(sandwich(ζ₀, Ñ)) ./ 2
nothing #hide

# The same average computed numerically, at a few values of ``z``, with the
# trapezoidal rule in ``\varphi`` on numerical moduli:

let vals = Dict(ω => 0.3, C₁₁ => 2.179, C₁₂ => 0.579, C₁₃ => 0.689, C₃₃ => 10.345, C₄₄ => 1.0)
    C̃_num = ti_from_constants((N(subs(c, vals...)) for c in (c̃₁₁, c̃₁₂, c̃₁₃, c̃₃₃, c̃₄₄))..., (0.0, 0.0, 1.0))
    W_num = walpole_basis((0.0, 0.0, 1.0))
    gaps = map((-0.9, -0.2, 0.4, 0.7)) do zv
        ring = sum(
            hill_integrand(TensND.Tens([sqrt(1 - zv^2) * cos(φ), sqrt(1 - zv^2) * sin(φ), zv]), C̃_num)
                for φ in range(0, 2π; length = 65)[1:(end - 1)]
        ) / 64 / 2
        sym = [N(subs(p, z => zv, vals...)) for p in p̃ᶻ]
        num = (W_num[1] ⊙ ring, W_num[2] ⊙ ring, (W_num[3] + W_num[4]) ⊙ ring / 2, W_num[5] ⊙ ring / 2, W_num[6] ⊙ ring / 2)
        maximum(abs.(sym .- num))
    end
    @printf "projection at φ = 0 against the average over φ:  %.1e\n" maximum(gaps)
end

# The projection gives the components of eq. 24 in terms of those of
# ``\tilde{\boldsymbol{N}}(z,0)``:

Ñc = canonical_components(Ñ)
s = sqrt(1 - z^2)
eq24 = (
    z^2 * Ñc[3, 3] / 2,
    (1 - z^2) * Ñc[1, 1] / 4,
    z * s * Ñc[1, 3] / (2 * sqrt(Sym(2))),
    ((1 - z^2) * Ñc[1, 1] + (1 - z^2) * Ñc[2, 2]) / 8,
    (z^2 * Ñc[1, 1] + z^2 * Ñc[2, 2] + (1 - z^2) * Ñc[3, 3] + 2 * z * s * Ñc[1, 3]) / 8,
)
p̃ᶻ .- eq24 .|> simplify

# **The acoustic tensor.** ``\tilde{\boldsymbol{N}}(z,0)`` has the components
# of eq. 25, over the two denominators ``q(z)`` and ``r(z)`` of eqs. 26 and 28.
# The roots of eq. 27 are not needed to see it: by Vieta's relations their sum
# and product are rational in the moduli, and ``q`` is a polynomial in
# ``z^2`` and ``1-z^2``:
#
# ```math
# \begin{aligned}
# q(z) &= \tilde{C}_{2323}\tilde{C}_{3333}\,\bigl(z^2+\eta_1^2(1-z^2)\bigr)\bigl(z^2+\eta_2^2(1-z^2)\bigr)\\
# &= a\,z^4 - b\,z^2(1-z^2) + c\,(1-z^2)^2,
# \end{aligned}
# ```
#
# with ``a = \tilde{C}_{2323}\tilde{C}_{3333}``, ``b = \tilde{C}_{1133}^2 + 2\tilde{C}_{1133}\tilde{C}_{2323} - \tilde{C}_{1111}\tilde{C}_{3333}``,
# ``c = \tilde{C}_{1111}\tilde{C}_{2323}``, and ``\eta_1^2+\eta_2^2 = -b/a``,
# ``\eta_1^2\eta_2^2 = c/a``.

a, b, c = c̃₄₄ * c̃₃₃, c̃₁₃^2 + 2 * c̃₁₃ * c̃₄₄ - c̃₁₁ * c̃₃₃, c̃₁₁ * c̃₄₄
q = a * z^4 - b * z^2 * (1 - z^2) + c * (1 - z^2)^2
r = c̃₄₄ * z^2 + (c̃₁₁ - c̃₁₂) / 2 * (1 - z^2)
eq25 = (
    (z^2 * (c̃₃₃ - c̃₄₄) + c̃₄₄) / q,
    (z^2 * (c̃₄₄ - c̃₁₁) + c̃₁₁) / q,
    -z * s * (c̃₁₃ + c̃₄₄) / q,
    1 / r,
)
(Ñc[1, 1], Ñc[3, 3], Ñc[1, 3], Ñc[2, 2]) .- eq25 .|> simplify

# ``r(z) = \tilde{C}_{2323}\bigl(z^2 + \eta_3^2(1-z^2)\bigr)`` with
# ``\eta_3^2 = (\tilde{C}_{1111}-\tilde{C}_{1122})/(2\tilde{C}_{2323})`` (eq. 28).
#
# **The integral over the colatitude.** Substituting eq. 25 into eq. 24, each
# ``\tilde p_i^z`` is an even polynomial in ``z`` over ``q(z)``, plus, for
# ``\tilde p_5^z`` and ``\tilde p_6^z``, one over ``r(z)``. Integrated over
# ``[-1,1]``, the monomials ``z^k/q`` and ``z^k/r`` give the integrals
#
# ```math
# \begin{gathered}
# \mathcal{J}_k(\eta_1,\eta_2) = \int_{-1}^{1}\frac{z^k\,\mathrm{d}z}{\bigl(z^2+\eta_1^2(1-z^2)\bigr)\bigl(z^2+\eta_2^2(1-z^2)\bigr)},
# \\
# \mathcal{I}_k(\eta) = \int_{-1}^{1}\frac{z^k\,\mathrm{d}z}{z^2+\eta^2(1-z^2)},
# \end{gathered}
# ```
#
# divided by ``a`` and by ``\tilde{C}_{2323}`` respectively. The map below
# performs that integration on a rational function of the form
# ``u(z)/q + v(z)/r``: it reads off ``u`` and ``v`` as the residues of the
# partial-fraction decomposition in the symbols ``1/q`` and ``1/r``, which
# stand for the two denominators:

𝒥 = symbols("J_0 J_2 J_4 J_6")
ℐ = symbols("I_0 I_2 I_4")
Qs, Rs = symbols("Q R", positive = true)

"Integral over ``[-1, 1]`` of `u(z)/q(z) + v(z)/r(z)`, written in the 𝒥ₖ and ℐₖ."
function integrate_z(expr)
    e = expand(subs(expr, q => Qs, r => Rs))
    u, v = expand(e.coeff(1 / Qs)), expand(e.coeff(1 / Rs))
    iszero(simplify(e - u / Qs - v / Rs)) || error("not of the form u/q + v/r")
    Ju = sum(u.coeff(z, 2k) * 𝒥[k + 1] for k in 0:3) / a
    Iv = sum(v.coeff(z, 2k) * ℐ[k + 1] for k in 0:2) / c̃₄₄
    return simplify(Ju + Iv)
end
nothing #hide

# Substituted in eq. 24 the components of eq. 25 give, once integrated, the
# Walpole coefficients of ``\tilde{\mathbb{P}}`` (eq. 29):

pz_explicit = (
    z^2 * eq25[2] / 2,
    (1 - z^2) * eq25[1] / 4,
    z * s * eq25[3] / (2 * sqrt(Sym(2))),
    ((1 - z^2) * eq25[1] + (1 - z^2) * eq25[4]) / 8,
    (z^2 * eq25[1] + z^2 * eq25[4] + (1 - z^2) * eq25[2] + 2 * z * s * eq25[3]) / 8,
)
p̃_int = integrate_z.(pz_explicit)
J₀, J₂, J₄ = 𝒥[1], 𝒥[2], 𝒥[3]
I₀, I₂ = ℐ[1], ℐ[2]
eq29 = (
    ((c̃₄₄ - c̃₁₁) * J₄ + c̃₁₁ * J₂) / (2 * c̃₄₄ * c̃₃₃),
    ((c̃₄₄ - c̃₃₃) * J₄ + (c̃₃₃ - 2c̃₄₄) * J₂ + c̃₄₄ * J₀) / (4 * c̃₄₄ * c̃₃₃),
    (c̃₄₄ + c̃₁₃) * (J₄ - J₂) / (2 * sqrt(Sym(2)) * c̃₄₄ * c̃₃₃),
    ((c̃₄₄ - c̃₃₃) * J₄ + (c̃₃₃ - 2c̃₄₄) * J₂ + c̃₄₄ * J₀) / (8 * c̃₄₄ * c̃₃₃) + (I₀ - I₂) / (8 * c̃₄₄),
    ((c̃₁₁ + c̃₃₃ + 2c̃₁₃) * J₄ - 2 * (c̃₁₁ + c̃₁₃) * J₂ + c̃₁₁ * J₀) / (8 * c̃₄₄ * c̃₃₃) + I₂ / (8 * c̃₄₄),
)
p̃_int .- eq29 .|> simplify

# The fourth line is eq. 29d, ``\tilde p_5 = \tilde p_2/2 + (\mathcal{I}_0-\mathcal{I}_2)/(8\tilde{C}_{2323})``,
# written out. No ``\mathcal{J}_6`` appears: the degree of the numerators stays
# within that of ``q``.
#
# ### Back to the spheroid (eqs. 30–36)
#
# The roots ``\eta_i^2`` of eq. 27, written with ``\tilde{\mathbb{C}}``, are
# ``\omega^2`` times the roots ``\gamma_i^2`` of the same polynomial written
# with ``\mathbb{C}`` (eq. 30): substituting ``Z = \omega^2 Y`` divides it by
# ``\omega^2``. Likewise ``\eta_3 = \omega\gamma_3`` (eq. 33):

Y = symbols("Y")
poly(Z, k₁₁, k₁₃, k₃₃, k₄₄) = Z^2 * k₄₄ * k₃₃ + Z * (k₁₃^2 + 2 * k₁₃ * k₄₄ - k₁₁ * k₃₃) + k₁₁ * k₄₄
(
    simplify(poly(ω^2 * Y, c̃₁₁, c̃₁₃, c̃₃₃, c̃₄₄) - poly(Y, C₁₁, C₁₃, C₃₃, C₄₄) / ω^2),
    simplify((c̃₁₁ - c̃₁₂) / (2c̃₄₄) - ω^2 * (C₁₁ - C₁₂) / (2C₄₄)),
)

# The roots are eq. 31, with the discriminant of eq. 32:

Δ = (C₁₁ * C₃₃ - C₁₃^2) * (C₁₁ * C₃₃ - C₁₃^2 - 4 * C₁₃ * C₄₄ - 4 * C₄₄^2)
let a = C₄₄ * C₃₃, b = C₁₃^2 + 2 * C₁₃ * C₄₄ - C₁₁ * C₃₃, c = C₁₁ * C₄₄
    simplify(b^2 - 4a * c - Δ)
end

# The integrals of the sphere in ``\tilde{\mathbb{C}}`` are therefore
# ``\mathcal{J}_k(\omega\gamma_1,\omega\gamma_2)`` and
# ``\mathcal{I}_k(\omega\gamma_3)`` (eq. 34), and the Walpole coefficients of the
# spheroid follow from eq. 20 with the constants of ``\tilde{\mathbb{C}}``
# written in those of ``\mathbb{C}`` (eqs. 35–36):

p_int = p̃_int ./ (ω^4, 1, ω^2, 1, ω^2)
eq36 = (
    ((C₄₄ - ω^2 * C₁₁) * J₄ + ω^2 * C₁₁ * J₂) / (2 * C₄₄ * C₃₃),
    ω^2 * ((ω^2 * C₄₄ - C₃₃) * J₄ + (C₃₃ - 2 * ω^2 * C₄₄) * J₂ + ω^2 * C₄₄ * J₀) / (4 * C₄₄ * C₃₃),
    ω^2 * (C₄₄ + C₁₃) * (J₄ - J₂) / (2 * sqrt(Sym(2)) * C₄₄ * C₃₃),
    ω^2 * ((ω^2 * C₄₄ - C₃₃) * J₄ + (C₃₃ - 2 * ω^2 * C₄₄) * J₂ + ω^2 * C₄₄ * J₀) / (8 * C₄₄ * C₃₃) +
        ω^2 * (I₀ - I₂) / (8 * C₄₄),
    ((ω^4 * C₁₁ + C₃₃ + 2 * ω^2 * C₁₃) * J₄ - 2 * ω^2 * (ω^2 * C₁₁ + C₁₃) * J₂ + ω^4 * C₁₁ * J₀) /
        (8 * C₄₄ * C₃₃) + I₂ / (8 * C₄₄),
)
p_int .- eq36 .|> simplify

# These are the coefficients written in the theory page, under
# [Transversely isotropic matrix coaxial with a spheroid](@ref th-hill-ti-coaxial).
#
# ### [The integrals (Appendix D)](@id app-hill-transformation-integrals)
#
# **Single root.** With ``D(z) = z^2 + \eta^2(1-z^2) = \eta^2 - (\eta^2-1)\,z^2``,
# the integral of ``1/D`` is an inverse hyperbolic tangent. For a real
# ``\eta > 1`` (the integrand is then positive and regular), its derivative
# checks the primitive, and its value at ``z = \pm1`` is that of Table D.1, by
# the identity ``\operatorname{artanh}\bigl(\sqrt{\eta^2-1}/\eta\bigr) = \operatorname{arcosh}\eta``:

η, η₁, η₂ = symbols("eta eta_1 eta_2", positive = true)
k² = η^2 - 1
F₀ = atanh(sqrt(k²) * z / η) / (η * sqrt(k²))
I₀_tab = 2 * acosh(η) / (η * sqrt(k²))
(
    simplify(diff(F₀, z) - 1 / (η^2 - k² * z^2)),
    simplify(diff(atanh(sqrt(k²) / η) - acosh(η), η)),
    subs(atanh(sqrt(k²) / η) - acosh(η), η => 1),
)

# ``\mathcal{I}_2`` and ``\mathcal{I}_4`` follow from ``\mathcal{I}_0`` without
# a new integral: ``z^2/D = (1 - \eta^2/D)/(1-\eta^2)``, so that
# ``\mathcal{I}_{k+2} = (2/(k+1) - \eta^2\,\mathcal{I}_k)/(1-\eta^2)``. The
# recurrence reproduces the second and third columns of Table D.1:

I₂_rec = (2 - η^2 * I₀_tab) / (1 - η^2)
I₄_rec = (Sym(2) / 3 - η^2 * I₂_rec) / (1 - η^2)
I₂_tab = 2 * (η * acosh(η) - sqrt(k²)) / k²^(Sym(3) / 2)
I₄_tab = Sym(2) / 3 * (3 * η^3 * acosh(η) + (1 - 4η^2) * sqrt(k²)) / k²^(Sym(5) / 2)
(
    simplify(z^2 / (η^2 - k² * z^2) - (1 - η^2 / (η^2 - k² * z^2)) / (1 - η^2)),
    simplify(I₂_rec - I₂_tab),
    simplify(I₄_rec - I₄_tab),
)

# **Two roots.** The partial fractions of the product of the two denominators,
# ``D_i(z) = z^2 + \eta_i^2(1-z^2)``, are
#
# ```math
# \begin{gathered}
# \frac{1}{D_1D_2} = \frac{1}{\eta_2^2-\eta_1^2}\left(\frac{1-\eta_1^2}{D_1} - \frac{1-\eta_2^2}{D_2}\right),
# \\
# \text{hence}\qquad
# \mathcal{J}_k = \frac{(1-\eta_1^2)\,\mathcal{I}_k(\eta_1) - (1-\eta_2^2)\,\mathcal{I}_k(\eta_2)}{\eta_2^2-\eta_1^2}:
# \end{gathered}
# ```

D₁, D₂ = z^2 + η₁^2 * (1 - z^2), z^2 + η₂^2 * (1 - z^2)
simplify(1 / (D₁ * D₂) - ((1 - η₁^2) / D₁ - (1 - η₂^2) / D₂) / (η₂^2 - η₁^2))

# This is the formula of the Python script appended to the paper. The third
# row of Table D.2 prints ``(\eta_1^2-1)`` and ``(\eta_2^2-1)`` in the
# numerator, which gives ``-\mathcal{J}_k``; the quadrature of the
# transcription below settles it on numbers as well.
#
# The second row, for coincident roots, is the limit of the third as
# ``\eta_2 \to \eta_1``; for ``\mathcal{J}_0``:

J₀_pf = ((1 - η₁^2) * subs(I₀_tab, η => η₁) - (1 - η₂^2) * subs(I₀_tab, η => η₂)) / (η₂^2 - η₁^2)
J₀_tab2 = (acosh(η₁) + η₁ * sqrt(η₁^2 - 1)) / (η₁^3 * sqrt(η₁^2 - 1))
simplify(limit(J₀_pf, η₂ => η₁) - J₀_tab2)

# ### [The same integrals in the package](@id app-hill-transformation-package-integrals)
#
# The page [Degenerate shapes](@ref th-degenerate-integrals) writes these
# integrals in the variable ``a = \eta^2``, as ``I_n(a)`` and
# ``J_{n,q}(a_1,a_2)``, with ``D(a,z) = a + (1-a)\,z^2``, which is the
# denominator ``z^2 + \eta^2(1-z^2)`` above. The correspondence is
#
# ```math
# \begin{gathered}
# \mathcal{I}_k(\eta) = I_k(\eta^2),\qquad
# \mathcal{J}_k(\eta_1,\eta_2) = J_{k,0}(\eta_1^2,\eta_2^2),
# \\
# R_C(a,1) = \frac{\operatorname{arcosh}\sqrt a}{\sqrt{a-1}} = \frac{\operatorname{arcosh}\eta}{\sqrt{\eta^2-1}},
# \end{gathered}
# ```
#
# and the closed forms of that page are those of Table D.1:

a_ = η^2
R_C = acosh(η) / sqrt(η^2 - 1)
I_pkg = (2R_C / sqrt(a_), 2 * (sqrt(a_) * R_C - 1) / (a_ - 1), 2 * (a_ * sqrt(a_) * R_C - 1 - Sym(4) / 3 * (a_ - 1)) / (a_ - 1)^2)
I_pkg .- (I₀_tab, I₂_tab, I₄_tab) .|> simplify

# The package does not evaluate ``\mathcal{J}_0, \mathcal{J}_2, \mathcal{J}_4``
# themselves but the combinations ``J_{2,1} = \mathcal{J}_2 - \mathcal{J}_4``
# and ``J_{0,2} = \mathcal{J}_0 - 2\mathcal{J}_2 + \mathcal{J}_4``, the moments
# of ``1-z^2`` and ``(1-z^2)^2``: written in them, the coefficients of eq. 36
# have no difference of terms ``\omega^2`` or ``\omega^4`` larger than the
# result, which those of eq. 36 have for a needle. The form the package
# evaluates, given in
# [Spheroid in a transversely isotropic matrix](@ref th-degenerate-ti-spheroid),
# is eq. 36 rearranged:

let k = C₄₄ * C₃₃
    j₄, l₂₁, l₀₂ = J₄ / k, (J₂ - J₄) / k, (J₀ - 2J₂ + J₄) / k
    i₂, i_w = I₂ / C₄₄, (I₀ - I₂) / C₄₄
    P₂ = ω^2 * (C₃₃ * l₂₁ + ω^2 * C₄₄ * l₀₂) / 4
    pkg = (
        (C₄₄ * j₄ + ω^2 * C₁₁ * l₂₁) / 2,
        P₂,
        -ω^2 * (C₄₄ + C₁₃) * l₂₁ / (2 * sqrt(Sym(2))),
        P₂ / 2 + ω^2 * i_w / 8,
        (C₃₃ * j₄ - 2 * ω^2 * C₁₃ * l₂₁ + ω^4 * C₁₁ * l₀₂ + i₂) / 8,
    )
    pkg .- eq36 .|> simplify
end

# ## [4. The script of the paper, in Julia](@id app-hill-transformation-script)
#
# The Python script appended to the paper evaluates the integrals of
# Tables D.1 and D.2 in complex arithmetic and the coefficients of eq. 36. Its
# transcription below accepts any number type: the moduli and the aspect ratio
# are promoted together, the arithmetic runs in complex numbers of that type,
# and the imaginary parts, which cancel, are dropped at the end. The square root
# and the inverse hyperbolic cosine of Julia have the principal branches the
# paper specifies (Appendix D), those of Python's `cmath`. The script switches to
# the special rows of the tables within ``10^{-6}`` of them; here they are taken
# on exact equality only, which is the point the formulas of the general rows
# cannot be evaluated at.

"Integrals ``(\\mathcal{I}_0, \\mathcal{I}_2, \\mathcal{I}_4)(\\eta)`` of Table D.1."
function 𝓘(η)
    k² = η^2 - 1
    iszero(k²) && return (2, 2 // 3, 2 // 5) .* one(η)
    sk, ac = sqrt(k²), acosh(η)
    return (2ac / (η * sk), 2 * (η * ac - sk) / (k² * sk), 2 * (3η^3 * ac + (1 - 4η^2) * sk) / (3 * k²^2 * sk))
end

"Integrals ``(\\mathcal{J}_0, \\mathcal{J}_2, \\mathcal{J}_4)(\\eta_1, \\eta_2)`` of Table D.2."
function 𝓙(η₁, η₂)
    η₁^2 == η₂^2 || return ((1 - η₁^2) .* 𝓘(η₁) .- (1 - η₂^2) .* 𝓘(η₂)) ./ (η₂^2 - η₁^2)
    k² = η₁^2 - 1
    iszero(k²) && return (2, 2 // 3, 2 // 5) .* one(η₁)
    sk, ac = sqrt(k²), acosh(η₁)
    return ((ac + η₁ * sk) / (η₁^3 * sk), (η₁ * sk - ac) / (η₁ * k² * sk), ((2 + η₁^2) * sk - 3η₁ * ac) / (k²^2 * sk))
end

"""
    hill_ti_paper(ω, C1111, C1122, C1133, C3333, C2323)

Hill tensor of a spheroid of aspect ratio `ω` coaxial with a transversely
isotropic matrix of axis ``\\underline{e}_3``, by eqs. 31–36 of the paper.
"""
function hill_ti_paper(ω, C1111, C1122, C1133, C3333, C2323)
    T = promote_type(typeof(ω), typeof(C1111), typeof(C1122), typeof(C1133), typeof(C3333), typeof(C2323))
    w, c11, c12, c13, c33, c44 = Complex{T}.((ω, C1111, C1122, C1133, C3333, C2323))
    Δ = (c11 * c33 - c13^2) * (c11 * c33 - c13^2 - 4c13 * c44 - 4c44^2)            # eq. 32
    γ₁ = sqrt((c11 * c33 - c13^2 - 2c13 * c44 + sqrt(Δ)) / (2c44 * c33))             # eq. 31
    γ₂ = sqrt((c11 * c33 - c13^2 - 2c13 * c44 - sqrt(Δ)) / (2c44 * c33))
    γ₃ = sqrt((c11 - c12) / (2c44))                                                  # eq. 33
    J₀, J₂, J₄ = 𝓙(w * γ₁, w * γ₂) ./ (c44 * c33)                                    # eq. 34
    I₀, I₂, _ = 𝓘(w * γ₃) ./ c44
    w² = w^2
    p₁ = ((c44 - w² * c11) * J₄ + w² * c11 * J₂) / 2                                 # eq. 36a
    p₂ = w² * ((w² * c44 - c33) * J₄ + (c33 - 2w² * c44) * J₂ + w² * c44 * J₀) / 4    # eq. 36b
    p₃ = w² * (c44 + c13) * (J₄ - J₂) / (2 * sqrt(T(2)))                             # eq. 36c
    p₅ = p₂ / 2 + w² * (I₀ - I₂) / 8                                                 # eq. 36d
    p₆ = ((w²^2 * c11 + c33 + 2w² * c13) * J₄ - 2w² * (w² * c11 + c13) * J₂ + w²^2 * c11 * J₀) / 8 + I₂ / 8
    return TensTI{4}(real.((p₁, p₂, p₃, p₅, p₆))..., (zero(T), zero(T), one(T)))     # eq. 35
end
nothing #hide

# **The integrals against a quadrature.** For two real roots, then a pair of
# complex conjugate ones, which a positive definite matrix may have, the
# transcription agrees with a Gauss-Kronrod quadrature of the definitions. The
# sign printed in the third row of Table D.2 gives the opposite:

using QuadGK: quadgk
for (e₁, e₂) in ((1.7 + 0im, 0.6 + 0im), (1.2 + 0.5im, 1.2 - 0.5im))
    D(e, x) = x^2 + e^2 * (1 - x^2)
    quad = [quadgk(x -> x^k / (D(e₁, x) * D(e₂, x)), -1, 1; rtol = 1.0e-14)[1] for k in (0, 2, 4)]
    as_printed = ((e₁^2 - 1) .* 𝓘(e₁) .- (e₂^2 - 1) .* 𝓘(e₂)) ./ (e₂^2 - e₁^2)
    @printf "η = %s, %s:  script %.1e   Table D.2 as printed %.1e\n" e₁ e₂ maximum(abs.(collect(𝓙(e₁, e₂)) .- quad)) maximum(abs.(collect(as_printed) .- quad))
end

# ### Figure 1 of the paper
#
# The matrix of [sevostianovIJSS2005; eq. 2.34](@citet), with
# ``C_{1111} = 2.179``, ``C_{1122} = 0.579``, ``C_{1133} = 0.689``,
# ``C_{3333} = 10.345`` and ``C_{2323} = 1``, and the Eshelby tensor
# ``\mathbb{S} = \mathbb{P}:\mathbb{C}`` against ``\xi = 1/\omega``, the
# transcription as lines and [`hill_tensor`](@ref) of the package as markers:

const C_sev = (2.179, 0.579, 0.689, 10.345, 1.0)
C_sev_t = ti_from_constants(C_sev..., (0.0, 0.0, 1.0))
ξs = 10 .^ range(-2, 2; length = 100)
S_of(P) = canonical_components(P ⊡ C_sev_t)
picks = ((1, 1, 1, 1), (3, 3, 3, 3), (1, 1, 2, 2), (1, 1, 3, 3), (3, 3, 1, 1), (1, 3, 1, 3))
labels = ["S₁₁₁₁" "S₃₃₃₃" "S₁₁₂₂" "S₁₁₃₃" "S₃₃₁₁" "S₁₃₁₃"]
S_paper = [S_of(hill_ti_paper(1 / ξ, C_sev...))[i...] for ξ in ξs, i in picks]
ξm = ξs[1:9:end]
S_mfh = [S_of(hill_tensor(Ellipsoid(1.0, 1.0, 1 / ξ), C_sev_t))[i...] for ξ in ξm, i in picks]
plt = plot(ξs, S_paper; xscale = :log10, label = labels, xlabel = "ξ = 1/ω", ylabel = "Sᵢⱼₖₗ", lw = 2, legend = :right, size = (720, 440))
scatter!(plt, ξm, S_mfh; label = "", color = (1:6)', markersize = 4, markerstrokewidth = 0)
plt

# The two agree along the whole sweep, and both with nested Gauss-Kronrod
# quadratures of the general anisotropic formula, which share no code with
# either:

@printf "transcription against hill_tensor, whole sweep:  %.1e\n" maximum(
    relgap(hill_ti_paper(1 / ξ, C_sev...), hill_tensor(Ellipsoid(1.0, 1.0, 1 / ξ), C_sev_t)) for ξ in ξs
)
for ω in (0.05, 0.5, 2.0, 20.0)
    P_quad = hill_tensor(Ellipsoid(1.0, 1.0, ω), C_sev_t; method = :nestedquadgk, abstol = 1.0e-14, reltol = 1.0e-13)
    @printf "ω = %5.2f   against the quadrature:  %.1e\n" ω relgap(hill_ti_paper(ω, C_sev...), P_quad)
end

# Since the transcription is generic in its number type, `ForwardDiff`
# differentiates it through the complex arithmetic; with respect to
# ``C_{1133}`` at ``\omega = 0.2``, against the derivative of the package's
# closed form:

dP_paper = ForwardDiff.derivative(x -> hill_ti_paper(0.2, C_sev[1], C_sev[2], x, C_sev[4], C_sev[5]), C_sev[3])
dP_mfh = ForwardDiff.derivative(x -> hill_tensor(Ellipsoid(1.0, 1.0, 0.2), ti_from_constants(C_sev[1], C_sev[2], x, C_sev[4], C_sev[5], (0.0, 0.0, 1.0))), C_sev[3])
@printf "∂P/∂C₁₁₃₃:  %.1e\n" relgap(dP_paper, dP_mfh)

# ### Where the formulas as written lose digits
#
# The partial fractions of Table D.2 divide by ``\eta_2^2 - \eta_1^2``. A
# matrix close to isotropy has close acoustic roots, and the general row then
# subtracts nearly equal terms; the exact isotropic matrix lands, after the
# rounding of ``\sqrt{\Delta}``, on it as well. Take an isotropic matrix of
# moduli ``\lambda = 1``, ``\mu = 1`` and stiffen ``C_{3333}`` by a relative
# ``\delta``:

for δ in (1.0e-2, 1.0e-4, 1.0e-6, 1.0e-8, 0.0)
    moduli = (3.0, 1.0, 1.0, 3.0 * (1 + δ), 1.0)
    P_pkg = hill_tensor(Ellipsoid(1.0, 1.0, 0.5), ti_from_constants(moduli..., (0.0, 0.0, 1.0)))
    @printf "δ = %.0e   transcription against the package:  %.1e\n" δ relgap(hill_ti_paper(0.5, moduli...), P_pkg)
end

# The error grows as ``\delta`` shrinks: the formulas are exact, their
# evaluation is not. The package evaluates the same integrals in three regimes
# that never subtract them, described in
# [Spheroid in a transversely isotropic matrix](@ref th-degenerate-ti-spheroid),
# and stays at rounding level there, derivatives included.

# ## [5. The flat spheroid and the crack (Section 4.3)](@id app-hill-transformation-flat)
#
# ### The expansions of the integrals (Appendix D)
#
# For ``0 < \eta < 1``, the case of a flat spheroid, the primitive of ``1/D``
# is an inverse tangent and Table D.1 takes its real form,
# ``\operatorname{arcosh}\eta/\sqrt{\eta^2-1} = \arccos\eta/\sqrt{1-\eta^2}``:

F₀_flat = atan(sqrt(1 - η^2) * z / η) / (η * sqrt(1 - η^2))
(
    simplify(diff(F₀_flat, z) - 1 / (η^2 + (1 - η^2) * z^2)),
    simplify(diff(atan(sqrt(1 - η^2) / η) - acos(η), η)),
    subs(atan(sqrt(1 - η^2) / η) - acos(η), η => 1),
)

# The same recurrence as above gives ``\mathcal{I}_2`` and ``\mathcal{I}_4``.
# Their expansions are those of a function of one variable, which SymPy takes
# at ``\eta\to0`` once; with ``\eta = \omega\gamma`` they are eqs. D.2:

γ, γ₁, γ₂, γ₃ = symbols("gamma gamma_1 gamma_2 gamma_3", positive = true)
I₀_flat(e) = 2 * acos(e) / (e * sqrt(1 - e^2))
I_series = let I₀s = series(I₀_flat(η), η, 0, 6).removeO()
    I₂s = series((2 - η^2 * I₀s) / (1 - η^2), η, 0, 6).removeO()
    I₄s = series((Sym(2) / 3 - η^2 * I₂s) / (1 - η^2), η, 0, 6).removeO()
    (I₀s, I₂s, I₄s)
end
nothing #hide
# The coefficient of ωᵏ in a Laurent polynomial in ω whose lowest power is
# ω⁻³: brought over one denominator free of ω, its numerator is a polynomial.
function laurent(e, k)
    num, den = sympy.fraction(cancel(e * ω^3))
    return simplify(sympy.Poly(num, ω).coeff_monomial(ω^(k + 3)) / den)
end
I_at(γx) = [subs(Ik, η => ω * γx) for Ik in I_series]
eqD2 = (PI / (γ * ω) - 2 + PI * γ * ω / 2, 2 - PI * γ * ω, Sym(2) / 3)
[laurent(I_at(γ)[n] - eqD2[n], k) for n in 1:3, k in -1:1]

# The partial fractions then give those of
# ``\mathcal{J}_k(\omega\gamma_1,\omega\gamma_2)`` (eqs. D.3), with
# ``\Sigma = \gamma_1^2+\gamma_1\gamma_2+\gamma_2^2``. The division by
# ``\eta_2^2-\eta_1^2 = \omega^2(\gamma_2^2-\gamma_1^2)`` is exact, coefficient
# by coefficient in ``\omega``:

Σ = γ₁^2 + γ₁ * γ₂ + γ₂^2
J_flat = map(I_at(γ₁), I_at(γ₂)) do I₁, I₂
    cancel(((1 - ω^2 * γ₁^2) * I₁ - (1 - ω^2 * γ₂^2) * I₂) / (ω^2 * (γ₂^2 - γ₁^2)))
end
eqD3 = (
    PI / (γ₁ * γ₂ * (γ₁ + γ₂) * ω^3) + PI / (2 * (γ₁ + γ₂) * ω) - Sym(2) / 3 + PI * Σ * ω / (8 * (γ₁ + γ₂)),
    PI / ((γ₁ + γ₂) * ω) - 2 + PI * Σ * ω / (2 * (γ₁ + γ₂)),
    2 - PI * Σ * ω / (γ₁ + γ₂),
)
[laurent(J_flat[n] - eqD3[n], k) for n in 1:3, k in -3:1]

# The expansions are derived for real roots; both sides are rational in
# ``\gamma_1`` and ``\gamma_2`` and symmetric in them, so that they hold for a
# pair of complex conjugate roots as well, by analytic continuation.
#
# ### The Hill tensor of a flat spheroid (eqs. 37–39)
#
# Substituted into eq. 36, the expansions give the first two terms of
# ``\mathbb{P} = \mathbb{P}^0 + \omega\,\mathbb{P}^1 + O(\omega^2)``. The
# leading term (eq. 37) is
# ``\mathbb{P}^0 = \underline{e}_3\stackrel{s}{\otimes}(\underline{e}_3\cdot\mathbb{C}\cdot\underline{e}_3)^{-1}\stackrel{s}{\otimes}\underline{e}_3``,
# that of a crack, and the next one has the Walpole coefficients of eq. 38:

flat = Dict(J₀ => eqD3[1], J₂ => eqD3[2], J₄ => eqD3[3], I₀ => subs(eqD2[1], γ => γ₃), I₂ => subs(eqD2[2], γ => γ₃))
p_flat = [subs(p, flat...) for p in eq36]
P⁰ = [laurent(p, 0) for p in p_flat]
P¹ = [laurent(p, 1) for p in p_flat]
eq37 = walpole(sandwich(TensND.Tens(Sym[0, 0, 1]), inv(TensND.Tens(Sym[0, 0, 1]) ⋅ C_TI ⋅ TensND.Tens(Sym[0, 0, 1]))))
eq38 = (
    PI / (2 * (γ₁ + γ₂)) * (C₁₁ - Σ * C₄₄) / (C₄₄ * C₃₃),
    PI / (4 * (γ₁ + γ₂)) * (1 / (γ₁ * γ₂ * C₃₃) + 1 / C₄₄),
    -sqrt(Sym(2)) * PI / (4 * (γ₁ + γ₂)) * (C₁₃ + C₄₄) / (C₄₄ * C₃₃),
    PI / 8 * (1 / (γ₁ * γ₂ * (γ₁ + γ₂) * C₃₃) + (1 / (γ₁ + γ₂) + 1 / γ₃) / C₄₄),
    PI / 8 * ((C₁₁ - 2γ₁ * γ₂ * C₁₃ - γ₁ * γ₂ * Σ * C₃₃) / (γ₁ * γ₂ * (γ₁ + γ₂) * C₄₄ * C₃₃) - γ₃ / C₄₄),
)
(simplify.(P⁰ .- eq37), simplify.(P¹ .- eq38))

# The terms in ``\omega^{-3}``, ``\omega^{-2}`` and ``\omega^{-1}``, which
# the integrals bring one by one, cancel in every coefficient:

[laurent(p, k) for p in p_flat, k in -3:-1]

# Eq. 38 involves the constants of ``\mathbb{C}`` only through the symmetric
# functions of the roots, which Vieta's relations write in the moduli
# (eqs. 39): with the roots of eq. 31,

let X = C₁₁ * C₃₃ - C₁₃^2 - 2 * C₁₃ * C₄₄
    g₁², g₂² = (X + sqrt(Δ)) / (2 * C₄₄ * C₃₃), (X - sqrt(Δ)) / (2 * C₄₄ * C₃₃)
    (simplify(g₁² + g₂² - X / (C₄₄ * C₃₃)), simplify(g₁² * g₂² - C₁₁ / C₃₃))
end

# so that ``\gamma_1\gamma_2 = \sqrt{C_{1111}/C_{3333}}``, the product of two
# positive reals or of two conjugates being positive, and
# ``\gamma_1+\gamma_2 = \sqrt{\gamma_1^2+\gamma_2^2+2\gamma_1\gamma_2}``.
#
# ### The crack compliance (eqs. 40–41)
#
# The second Hill tensor ``\mathbb{Q} = \mathbb{C} - \mathbb{C}:\mathbb{P}:\mathbb{C}``
# becomes singular as the spheroid flattens, and its inverse grows as
# ``1/\omega``. The crack compliance is the limit (eq. 40)
#
# ```math
# \mathbb{H} = \lim_{\omega\to0}\omega\,\mathbb{Q}^{-1}.
# ```
#
# With ``\mathbb{P} = \mathbb{P}^0 + \omega\,\mathbb{P}^1`` on free symbols for
# the five coefficients of ``\mathbb{P}^1``, the inverse of a transversely
# isotropic tensor stays one, and the limit has the two Walpole coefficients
# of eq. 41, which depend on ``\mathbb{P}^1`` only:

p¹ = symbols("p1_1 p1_2 p1_3 p1_5 p1_6")
P_lin = TensND.TensTI{4}((P⁰ .+ ω .* p¹)..., e₃)
Q_lin = C_TI - C_TI ⊡ P_lin ⊡ C_TI
H_sym = [subs(cancel(ω * h), ω => 0) for h in walpole(inv(Q_lin))]
eq41 = (
    -1 / (C₃₃^2 * p¹[1] + 2 * C₁₃^2 * p¹[2] + 2 * sqrt(Sym(2)) * C₃₃ * C₁₃ * p¹[3]),
    0, 0, 0,
    -1 / (4 * C₄₄^2 * p¹[5]),
)
simplify.(H_sym .- eq41)

# On the matrix of Figure 1, the crack compliance of eq. 41, with the
# coefficients of eq. 38, is the limit of ``\omega\,\mathbb{Q}^{-1}`` computed
# with [`second_hill_tensor`](@ref), and the tensor of a penny-shaped crack
# that the package builds from its crack opening displacement tensor,
# ``\mathbb{H} = \tfrac34\,\underline{n}\stackrel{s}{\otimes}\boldsymbol{B}\stackrel{s}{\otimes}\underline{n}``
# (see [`cod_tensor`](@ref)):

γ_num = let c11 = C_sev[1], c12 = C_sev[2], c13 = C_sev[3], c33 = C_sev[4], c44 = C_sev[5]
    Δn = complex((c11 * c33 - c13^2) * (c11 * c33 - c13^2 - 4c13 * c44 - 4c44^2))
    X = c11 * c33 - c13^2 - 2c13 * c44
    (sqrt((X + sqrt(Δn)) / (2c44 * c33)), sqrt((X - sqrt(Δn)) / (2c44 * c33)), sqrt(complex((c11 - c12) / (2c44))))
end
nothing #hide
# eqs. 38 and 41 as Julia functions, evaluated on the complex roots.
p¹_num = [lambdify(e, (C₁₁, C₁₂, C₁₃, C₃₃, C₄₄, γ₁, γ₂, γ₃))(complex.(C_sev)..., γ_num...) for e in eq38]
H41 = [lambdify(Sym(h), (C₁₁, C₁₂, C₁₃, C₃₃, C₄₄, p¹...))(complex.(C_sev)..., p¹_num...) for h in eq41]
H_eq41 = TensTI{4}(real.(H41)..., (0.0, 0.0, 1.0))
H_lim(ω) = ω * inv(second_hill_tensor(Ellipsoid(1.0, 1.0, ω), C_sev_t))
n₃ = TensND.Tens([0.0, 0.0, 1.0])
H_cod = 3 / 4 * sandwich(n₃, cod_tensor(PennyCrack(1.0), C_sev_t))
for ω in (1.0e-2, 1.0e-4, 1.0e-6)
    @printf "ω = %.0e   ω Q⁻¹ against eq. 41:  %.1e\n" ω relgap(H_lim(ω), H_eq41)
end
@printf "penny crack of the package against eq. 41:  %.1e\n" relgap(H_cod, H_eq41)

# The gap of ``\omega\,\mathbb{Q}^{-1}`` falls as ``\omega``, the order of the
# next term of the expansion.
#
# ## [6. The needle (Section 4.4)](@id app-hill-transformation-needle)
#
# When ``\omega\to\infty`` the spheroid tends to a circular cylinder of axis
# ``\underline{e}_3``, and the expansions D.4–D.5 of the integrals turn eq. 36
# into eq. 42: ``p_1``, ``p_3`` vanish and
#
# ```math
# p_2 \to \frac{1}{2C_{1111}},\qquad
# p_5 \to \frac{1}{4C_{1111}} + \frac{1}{2(C_{1111}-C_{1122})},\qquad
# p_6 \to \frac{1}{4C_{2323}},
# ```
#
# with corrections in ``\ln\omega/\omega^2``. On the matrix of Figure 1:

P_needle = TensTI{4}(0.0, 1 / (2C_sev[1]), 0.0, 1 / (4C_sev[1]) + 1 / (2 * (C_sev[1] - C_sev[2])), 1 / (4C_sev[5]), (0.0, 0.0, 1.0))
for ω in (1.0e2, 1.0e4, 1.0e6)
    @printf "ω = %.0e   P against eq. 42:  %.1e\n" ω relgap(hill_tensor(Ellipsoid(1.0, 1.0, ω), C_sev_t), P_needle)
end

# ## Where to go next
#
# The coefficients derived here are those the theory page writes for the
# [coaxial spheroid](@ref th-hill-ti-coaxial), and
# [Spheroid in a transversely isotropic matrix](@ref th-degenerate-ti-spheroid)
# shows how the package evaluates them without the losses met above. The flat
# limit is the starting point of the crack chapter:
# [Crack opening displacement tensors](@ref th-cod-tensors) derives the crack
# compliance for any anisotropy, of which eq. 41 is the transversely isotropic
# case. The tutorial [Hill tensors by reduction to a sphere](@ref tut-hill-reduction)
# applies the reduction through the library on other shapes and matrices.
