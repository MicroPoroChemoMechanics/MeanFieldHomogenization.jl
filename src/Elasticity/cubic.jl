# =============================================================================
#  cubic.jl — the cubic symmetry class.
#
#  A fourth-order tensor with the minor symmetries and cubic symmetry has
#  **three** independent constants, not two.  The package could describe
#  isotropic, transversely isotropic and orthotropic tensors and not this one,
#  which is the class every cube-symmetric morphology lands in — a supersphere,
#  a cubic array of inclusions, a cubic crystal.
#
#  There is no TensND storage type for the class, so nothing here dispatches on
#  one and `material_symmetry` deliberately still answers
#  `GeneralAnisotropicSym` for a cubic tensor: inventing a trait with no type
#  behind it would let dispatch claim a structure the object does not carry.
#  What is provided instead is the algebra — the projectors, the projection, its
#  residual, and the constants — which is what a solver needs.
# =============================================================================

# The three orthogonal projectors of the class, in Kelvin-Mandel and in the cube
# frame.  Under the octahedral group the Kelvin-Mandel space splits as
# A1g + Eg + T2g, of dimensions 1 + 2 + 3, three inequivalent irreducible
# representations each of multiplicity one; the commutant is therefore
# three-dimensional and spanned by these.
#
# One consequence is worth stating because it is not obvious and it is used as a
# test: a tensor with the minor symmetries and cubic symmetry is **automatically
# major-symmetric**.  Each projector is symmetric, so any combination of them
# is.  A strain localization tensor, which in general has no major symmetry,
# recovers it here.
function _cubic_projectors(::Type{T}) where {T}
    v = T[1, 1, 1, 0, 0, 0]
    J = (v * v') ./ 3
    E = zeros(T, 6, 6)
    for i in 1:3, j in 1:3
        E[i, j] = (i == j ? one(T) : zero(T)) - one(T) / 3
    end
    F = zeros(T, 6, 6)
    for i in 4:6
        F[i, i] = one(T)
    end
    return (J, E, F)
end

# Components in the CANONICAL basis. `get_array` — and therefore a bare `KM` —
# returns the components in the tensor's *own* basis, which for a rotated tensor
# is not the same thing at all; reading one for the other has produced false bug
# reports before now.
_canonical_km(t::TensND.AbstractTens{4, 3}) = TensND.KM(TensND.change_tens_canon(t))
_canonical_km(t::TensND.AbstractTens{2, 3}) = TensND.KM(TensND.change_tens_canon(t))

"""
    cubic_parameters(t) -> (; C11, C12, C44, alpha, beta, gamma)

The three constants of the cubic part of a fourth-order tensor, read in the
**canonical basis**, whose vectors are taken to be the cube axes.

`alpha`, `beta` and `gamma` are the coefficients on the three orthogonal
projectors — `alpha = C11 + 2C12 = 3K`, `beta = C11 - C12`, `gamma = 2C44` —
and are what the projection is actually built from; `C11`, `C12`, `C44` are the
same information in the usual engineering convention.

For an isotropic tensor `3K J + 2μ K` this returns `alpha = 3K`,
`beta = gamma = 2μ`, hence `C44 = μ` and `C11 - C12 = 2μ`: the cubic class
contains the isotropic one, as it must.

A tensor that is *not* cubic is not refused — its cubic part is simply what is
returned. [`cubic_residual`](@ref) is what says how much was discarded.
"""
function cubic_parameters(t::TensND.AbstractTens{4, 3})
    return _cubic_parameters_km(_canonical_km(t))
end

function _cubic_parameters_km(M::AbstractMatrix)
    T = eltype(M)
    J, E, F = _cubic_projectors(T)
    alpha = LinearAlgebra.tr(M * J)          # tr(J²) = 1
    beta = LinearAlgebra.tr(M * E) / 2       # tr(E²) = 2
    gamma = LinearAlgebra.tr(M * F) / 3      # tr(F²) = 3
    C11 = alpha / 3 + 2 * beta / 3
    C12 = alpha / 3 - beta / 3
    C44 = gamma / 2
    return (; C11, C12, C44, alpha, beta, gamma)
end

"""
    best_fit_cubic(t) -> Tens{4,3}

Orthogonal projection of a fourth-order tensor onto the cubic class, the cube
axes being the canonical basis vectors.

The companion of TensND's `best_fit_iso`, `best_fit_ti` and `best_fit_ortho`,
which do not cover this class. It returns a plain canonical `Tens{4,3}` rather
than a structured type because there is none for the cubic class — the
structure is real, the storage type is missing.

Where the answer is cubic **by group theory** — a cube-symmetric inclusion in an
isotropic matrix — the projection changes nothing that is not discretization
error, and [`cubic_residual`](@ref) measures exactly that. That is a free error
bar: no reference solution is involved, only the symmetry the problem was posed
with.
"""
function best_fit_cubic(t::TensND.AbstractTens{4, 3})
    M = _canonical_km(t)
    J, E, F = _cubic_projectors(eltype(M))
    c = _cubic_parameters_km(M)
    return TensND.Tens(TensND.inv_KM(c.alpha .* J .+ c.beta .* E .+ c.gamma .* F))
end

"""
    best_fit_cubic(t::AbstractTens{2,3}) -> TensISO{2}

At **order two the cubic class is the isotropic class**: the octahedral group
leaves no second-order tensor invariant other than a multiple of the identity.

This is not a shortcut, it is the reason a cube-symmetric pore has a single
scalar resistivity contribution while its compliance contribution needs three
constants — and the reason a conduction computation on such a shape carries no
anisotropy signal at all, whatever the shape does in elasticity.
"""
best_fit_cubic(t::TensND.AbstractTens{2, 3}) = TensND.best_fit_iso(t)

"""
    cubic_residual(t) -> Float64

Relative Frobenius distance from `t` to its cubic projection,
`‖t - P(t)‖ / ‖t‖`, in the canonical basis.

**This is the number to read.** For a tensor whose class is known by symmetry it
is bounded by the discretization error and by nothing else, so it is an error
estimate that costs nothing and assumes nothing. A directional artifact would
show up here; a real material anisotropy would not, since it lives *inside* the
class.
"""
function cubic_residual(t::TensND.AbstractTens{4, 3})
    M = _canonical_km(t)
    J, E, F = _cubic_projectors(eltype(M))
    c = _cubic_parameters_km(M)
    P = c.alpha .* J .+ c.beta .* E .+ c.gamma .* F
    return LinearAlgebra.norm(M .- P) / LinearAlgebra.norm(M)
end

"""
    cubic_anisotropy(t) -> Real

``(t_{1111} - t_{1122} - 2t_{1212})/t_{1111}`` from the **tensor** components: a
Zener-type measure of how far a cubic tensor departs from an isotropic one.

Exactly zero when the tensor is isotropic, and it is precisely the quantity a
two-constant approximation of a cubic tensor throws away. In terms of the
projector coefficients it is ``(\\beta - \\gamma)/t_{1111}``.

Read together with [`cubic_residual`](@ref), which is its error bar: an
anisotropy far above the residual is a property of the material or the
morphology, not of the mesh.
"""
function cubic_anisotropy(t::TensND.AbstractTens{4, 3})
    M = _canonical_km(t)
    # M[6,6] is the Kelvin-Mandel entry, hence 2·t₁₂₁₂.
    return (M[1, 1] - M[1, 2] - M[6, 6]) / M[1, 1]
end

"""
    cubic_stiffness(C11, C12, C44) -> Tens{4,3}

A cubic fourth-order tensor from its three constants, the cube axes being the
canonical basis vectors. The counterpart of [`iso_stiffness`](@ref) one class
out.

`C11 = C12 + 2C44` gives back an isotropic tensor, which is the identity worth
remembering: the departure from it is what [`cubic_anisotropy`](@ref) measures.
"""
function cubic_stiffness(C11::Number, C12::Number, C44::Number)
    T = Core._floatlike(promote_type(typeof(C11), typeof(C12), typeof(C44)))
    J, E, F = _cubic_projectors(T)
    alpha = T(C11 + 2 * C12)
    beta = T(C11 - C12)
    gamma = T(2 * C44)
    return TensND.Tens(TensND.inv_KM(alpha .* J .+ beta .* E .+ gamma .* F))
end
