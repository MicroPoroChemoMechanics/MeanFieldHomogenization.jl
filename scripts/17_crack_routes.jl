# # [Two routes to a crack: the COD tensor and the flattened ellipsoid](@id tut-crack-routes)
#
# !!! info "Before this page"
#     The theory page [Crack opening displacement and compliance](@ref th-cod-tensors):
#     §2 defines the COD tensor ``\boldsymbol{B}``, normalized by the half-width
#     ``b`` of the crack, and [§3](@ref th-cod-compliance) proves that the
#     compliance contribution of an elliptical crack is
#     ``\mathbb{H} = \lim_{\rho\to 0}\rho\,\mathbb{Q}^{-1}
#     = \tfrac{3}{4}\,\underline{n}\stackrel{s}{\otimes}\boldsymbol{B}\stackrel{s}{\otimes}\underline{n}``,
#     with ``\rho = c/b`` the flatness of the ellipsoid it is the limit of. Every
#     tensor on this page is in that normalization; the
#     [Conventions](@ref th-cod-conventions) give the factor ``\eta = b/a`` that
#     separates it from papers normalizing by ``a``.
#
# The identity of §3 joins two independent computations of a crack, and this
# page runs both, first in an isotropic matrix where everything is a closed form,
# then in matrices drawn at random, to show that they meet.
#
# - **From the crack plane.** `cod_tensor` integrates the reduced kernel of the
#   Green operator along the crack normal and around the crack front
#   [kanaunlevin2009, barthelemyIJSS2026](@cite): it returns ``\boldsymbol{B}``,
#   and ``\mathbb{H}`` follows from the identity (`compliance_from_cod`).
# - **From a flattened ellipsoid.** The crack is the limit of the ellipsoid of
#   semi-axes ``(a, b, \rho b)`` as ``\rho \to 0`` [barthelemyIJSS2009](@cite).
#   The first-order term of its Hill tensor in ``\rho`` gives ``\mathbb{H}`` by
#   a computation on blocks of a matrix, and ``\boldsymbol{B}`` follows from the
#   identity read the other way (`cod_from_compliance`).

import Pkg                                                          #jl
Pkg.activate(joinpath(@__DIR__, "..", "docs"); io = devnull)                 #jl

using MeanFieldHomogenization
using TensND
using LinearAlgebra
using ForwardDiff
using Random
using Printf
import SymPyPythonCall
import DECUHR, Integrals

const canonical = TensND.CanonicalBasis{3, Float64}()
const tol = (; abstol = 1.0e-12, reltol = 1.0e-10)
relerr(A, B) = norm(components_canon(A) - components_canon(B)) / norm(components_canon(B))

# ## The flattened ellipsoid to first order
#
# The Hill tensor of the ellipsoid of semi-axes ``(a, b, \rho b)``, whose
# third axis is the crack normal ``\underline{n}``, expands as
#
# ```math
# \mathbb{P}(\rho) = \mathbb{P}_0 + \rho\,\mathbb{P}_1 + O(\rho^2),
# \qquad
# \mathbb{P}_0 = \underline{n}\stackrel{s}{\otimes}\bigl(\underline{n}\cdot\mathbb{C}\cdot\underline{n}\bigr)^{-1}\stackrel{s}{\otimes}\underline{n},
# ```
#
# so that the second Hill tensor ``\mathbb{Q} = \mathbb{C} - \mathbb{C}:\mathbb{P}:\mathbb{C}``
# expands as ``\mathbb{Q}_0 + \rho\,\mathbb{Q}_1`` with
# ``\mathbb{Q}_1 = -\mathbb{C}:\mathbb{P}_1:\mathbb{C}``. The leading term is
# singular: every strain ``\underline{n}\stackrel{s}{\otimes}\underline{v}`` lies in
# its kernel, three dimensions which, in the Kelvin–Mandel components of the
# crack frame ``(\underline{\ell}, \underline{m}, \underline{n})``, are the
# components ``33``, ``23`` and ``13``. In these blocks,
#
# ```math
# \rho\,\mathbb{Q}^{-1}
# = \rho\begin{pmatrix} X + \rho Y_{11} & \rho Y_{12} \\ \rho Y_{12}^{\!T} & \rho Y_{22}\end{pmatrix}^{-1}
# \;\xrightarrow[\rho\to 0]{}\;
# \begin{pmatrix} 0 & 0 \\ 0 & Y_{22}^{-1}\end{pmatrix} = \mathbb{H},
# ```
#
# with ``Y_{22}`` the block of ``\mathbb{Q}_1`` on the kernel of ``\mathbb{Q}_0``
# [barthelemyIJSS2009](@cite). Only ``\mathbb{P}_1 = \mathrm{d}\mathbb{P}/\mathrm{d}\rho\,(0)``
# is needed. The function below takes it as a Kelvin–Mandel matrix, in whatever
# arithmetic produced it, and returns ``\mathbb{H}`` in the crack frame ``\mathcal{B}``.

function H_from_first_order(P₁, C, ℬ)
    Q₁ = -mandel(C) * P₁ * mandel(C)
    Y = mandel(from_mandel(Q₁, canonical), ℬ)
    H = zeros(eltype(Y), 6, 6)
    H[3:5, 3:5] = inv(Y[3:5, 3:5])
    return from_mandel(H, ℬ)
end

# Evaluating ``\rho\,\mathbb{Q}^{-1}`` at a small but finite flatness is the
# obvious alternative, and the theory page does it: its error decreases only
# like ``\rho`` ([the check of ``\mathbb{H}``](@ref th-cod-H-check)), and two
# flatnesses are combined to cancel that first-order error
# ([every route in a triclinic matrix](@ref th-cod-every-route)). The block
# computation removes it at the source, with no flatness to choose.
#
# ## In an isotropic matrix, exactly
#
# With ``\mathbb{C}`` of Young's modulus ``E`` and Poisson's ratio ``\nu``, both
# routes are closed forms, and SymPy carries them exactly. The COD tensor of a
# penny-shaped crack of radius ``b`` is Sneddon's:
#
# ```math
# \boldsymbol{B} = \frac{16\,(1-\nu^2)}{3\pi E}\Bigl(\underline{n}\otimes\underline{n}
# + \frac{2}{2-\nu}\,\bigl(\boldsymbol{1}-\underline{n}\otimes\underline{n}\bigr)\Bigr),
# ```
#
# and `cod_tensor` returns it exactly (the difference simplifies to zero):

E, ν, ρ = SymPyPythonCall.symbols("E ν ρ", positive = true)
C_sym = TensISO{3}(E / (1 - 2ν), E / (1 + ν))
B_sym = get_array(cod_tensor(PennyCrack(1), C_sym))
B_sneddon = 16 * (1 - ν^2) / (3 * SymPyPythonCall.PI * E) * Diagonal([2 / (2 - ν), 2 / (2 - ν), 1])
SymPyPythonCall.simplify.(B_sym - B_sneddon)

# The flattened route takes the Hill tensor of the oblate spheroid
# ``(b, b, \rho b)``, a closed form in ``\arccos\rho``, differentiates it with
# respect to ``\rho`` and lets ``\rho \to 0``. The block computation then gives
#
# ```math
# H_{3333} = \frac{4\,(1-\nu^2)}{\pi E}, \qquad
# 4H_{2323} = 4H_{1313} = \frac{8\,(1-\nu^2)}{\pi E\,(2-\nu)},
# ```
#
# which is ``\tfrac{3}{4}\,\underline{n}\stackrel{s}{\otimes}\boldsymbol{B}\stackrel{s}{\otimes}\underline{n}``
# for the ``\boldsymbol{B}`` above, exactly, as §3 says:

P_sym = mandel(hill_tensor(Ellipsoid(1, 1, ρ), C_sym))
P₁_sym = [SymPyPythonCall.limit(SymPyPythonCall.diff(p, ρ), ρ => 0) for p in P_sym]
H_sym = H_from_first_order(P₁_sym, C_sym, canonical)
SymPyPythonCall.simplify.(mandel(H_sym) - mandel(compliance_from_cod(Tens(B_sym), PennyCrack(1))))

# ## In random anisotropic matrices
#
# Each matrix below is a random symmetric positive definite Kelvin–Mandel matrix,
# fully anisotropic (21 independent constants), and each crack has a random
# orientation. ``\mathbb{P}_1`` is now the derivative, by ForwardDiff, of the
# Hill tensor computed by the nested cubature, taken at two small flatnesses
# ``\rho_0`` and ``\rho_0/2`` and combined so that its own first-order term
# cancels: ``\mathbb{P}_1 \approx 2\,\mathbb{P}'(\rho_0/2) - \mathbb{P}'(\rho_0)``.

function P₁_by_forwarddiff(a, b, R, C; ρ₀ = 1.0e-5)
    P(ρ) = mandel(hill_tensor(Ellipsoid(a, b, ρ * b, R), C; method = :nestedquadgk, tol...), canonical)
    P′(ρ) = ForwardDiff.derivative(P, ρ)
    return 2 * P′(ρ₀ / 2) - P′(ρ₀)
end

function random_stiffness(rng)
    G = randn(rng, 6, 6)
    return from_mandel(100 * (G * transpose(G) / 6 + I), canonical)
end

function random_rotation(rng)
    F = qr(randn(rng, 3, 3))
    R = Matrix(F.Q) * Diagonal(sign.(diag(F.R)))
    return det(R) > 0 ? R : -R
end

# For each matrix and each aspect ratio ``\eta = b/a``, the table compares
# ``\mathbb{H}`` from the flattening with ``\mathbb{H}`` from ``\boldsymbol{B}`` by the
# identity, and the reverse; then ``\boldsymbol{B}`` from the crack plane by its
# three integrators, the nested cubature being the reference: the residues and
# DECUHR.

rng = Xoshiro(20261018)
@printf "%-6s %-5s  %-13s %-13s %-13s %-13s\n" "matrix" "η" "ℍ: flat/𝐁" "𝐁: flat/𝐁" "𝐁 residues" "𝐁 DECUHR"
for k in 1:4
    C = random_stiffness(rng)
    R = random_rotation(rng)
    for η in (1.0, 0.5, 0.2)
        crack = EllipticCrack(1.0, η, R)
        B = cod_tensor(crack, C; method = :nestedquadgk, tol...)
        H = H_from_first_order(P₁_by_forwarddiff(1.0, η, R, C), C, crack_basis(crack))
        @printf "%-6d %-5.2f  %-13.1e %-13.1e %-13.1e %-13.1e\n" k η relerr(
            H, compliance_from_cod(B, crack)
        ) relerr(cod_from_compliance(H, crack), B) relerr(
            cod_tensor(crack, C; method = :residues, tol...), B
        ) relerr(cod_tensor(crack, C; method = :decuhr, tol...), B)
    end
end

# The two routes agree to between ``10^{-10}`` and ``3\times 10^{-8}``, whatever the
# anisotropy: the residual of the ``O(\rho_0^2)`` left in ``\mathbb{P}_1``,
# larger for the narrower cracks. On the reduced kernel, the residues meet the
# nested cubature to rounding, and DECUHR, at the same tolerances, to
# ``3\times 10^{-12}`` at worst: the nested cubature is the default, DECUHR a
# second opinion.
#
# ## An analytic anchor in an anisotropic matrix
#
# Neither route above is a closed form in an anisotropic matrix. One exists when
# the matrix is the image of an isotropic one by a linear map
# ``\underline{x} = \boldsymbol{P}\cdot\tilde{\underline{x}}``,
# ``\mathbb{C} = (\boldsymbol{P}\stackrel{s}{\boxtimes}\boldsymbol{P}):\tilde{\mathbb{C}}:(\boldsymbol{P}\stackrel{s}{\boxtimes}\boldsymbol{P})^{\!T}``
# (elliptic orthotropy). The crack of radii ``(a, b)`` maps to a crack of radii
# ``(\tilde{a}, \tilde{b})`` and normal ``\tilde{\underline{n}}`` in the isotropic
# matrix, whose ``\tilde{\boldsymbol{B}}`` is a closed form, and
# [barthelemyIJSS2026](@cite) gives, in the normalization of the package,
#
# ```math
# \boldsymbol{B} = \frac{\tilde{b}/b}{\lVert\boldsymbol{P}^{\!T}\cdot\underline{n}\rVert}\;
# \boldsymbol{P}^{-T}\cdot\tilde{\boldsymbol{B}}\cdot\boldsymbol{P}^{-1}.
# ```

function transformed_crack(P, a, b, ℓ, m, n)
    g, h = P \ ℓ, P \ m
    area = norm(cross(g, h))
    s₊ = sqrt(a^2 * norm(g)^2 + b^2 * norm(h)^2 + 2a * b * area)
    s₋ = sqrt(a^2 * norm(g)^2 + b^2 * norm(h)^2 - 2a * b * area)
    ã, b̃ = (s₊ + s₋) / 2, (s₊ - s₋) / 2
    ñ = normalize(transpose(P) * n)
    w_ℓ = (a^2 * norm(g)^2 - b̃^2) * g + b^2 * dot(g, h) * h
    w_m = a^2 * dot(g, h) * g + (b^2 * norm(h)^2 - b̃^2) * h
    ℓ̃ = normalize(norm(w_ℓ) ≥ norm(w_m) ? w_ℓ : w_m)
    return ã, b̃, hcat(ℓ̃, cross(ñ, ℓ̃), ñ)
end

Pmap = [1.3 0.2 -0.1; 0.15 0.8 0.25; -0.05 0.3 1.1]
PP = mandel(Tens(Pmap) ⊠ˢ Tens(Pmap))
C̃ = TensISO{3}(3 * 20.0, 2 * 9.0)
C_EO = from_mandel(PP * mandel(C̃) * transpose(PP), canonical)
a, b = 1.0, 0.45
R = random_rotation(Xoshiro(2))
ℓ, m, n = eachcol(R)
ã, b̃, R̃ = transformed_crack(Pmap, a, b, ℓ, m, n)
B̃ = components_canon(cod_tensor(EllipticCrack(ã, b̃, R̃), C̃))
B_rule = (b̃ / b) / norm(transpose(Pmap) * n) * (transpose(Pmap) \ B̃ / Pmap)
crack = EllipticCrack(a, b, R)
H_EO = H_from_first_order(P₁_by_forwarddiff(a, b, R, C_EO), C_EO, crack_basis(crack))
for (label, B) in (
        ("crack plane, nested cubature", cod_tensor(crack, C_EO; method = :nestedquadgk, tol...)),
        ("crack plane, residues", cod_tensor(crack, C_EO; method = :residues, tol...)),
        ("crack plane, DECUHR", cod_tensor(crack, C_EO; method = :decuhr, tol...)),
        ("flattened ellipsoid", cod_from_compliance(H_EO, crack)),
    )
    @printf "%-30s %.1e\n" label norm(components_canon(B) - B_rule) / norm(B_rule)
end

# Every route meets the closed form: the reduced kernel to rounding, the
# flattened ellipsoid to the accuracy of its first-order term. The same
# comparison, for transformed transversely isotropic matrices and ribbon-like
# cracks, is part of the test suite.
#
# ## Where to go next
#
# A single crack is all this page needs. [Crack distributions: isotropic or
# parallel](@ref tut-crack-distributions) gives populations of them a density and
# an orientation rule, and follows the self-consistent estimates to percolation;
# [Validating a finite-element crack](@ref tut-fe-crack) computes the opening of
# one crack by finite elements and compares it with the closed-form
# ``\boldsymbol{B}``.
