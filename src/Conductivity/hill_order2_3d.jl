# =============================================================================
#  hill_order2_3d.jl — 2nd-order Hill tensor (3D, conductivity / diffusion).
# =============================================================================

"""
    _hill_order2_3d_iso(ell::Ellipsoid{3}, K₀) -> AbstractTens{2,3}

2nd-order Hill polarization tensor of an ellipsoid in an isotropic
conductor ``\\boldsymbol{K}_0 = k_0\\,\\boldsymbol{1}``:

```math
\\boldsymbol{P}(\\boldsymbol{A}, k_0\\,\\boldsymbol{1})
= \\frac{\\boldsymbol{I}^{\\boldsymbol{A}}}{k_0} ,
```

where ``\\boldsymbol{I}^{\\boldsymbol{A}}`` is the Newton-potential geometric
tensor ([`tens_IA`](@ref), [willis1977](@cite)).
"""
function _hill_order2_3d_iso(ell::Ellipsoid{3, Spherical}, K₀)
    T = promote_type(eltype(ell.semi_axes), eltype(K₀))
    k = K₀.data[1]
    # The geometric factor of a sphere is 1/3 whatever its radius. Formed in the
    # promoted type it stays exact on symbolic conductivities, where reading it
    # off `tens_IA` carried in the `Float64` of the radius; on `Float64` the
    # value is the same to the bit.
    return TensND.TensISO{3}((T(1) / 3) / k)
end

function _hill_order2_3d_iso(ell::Ellipsoid{3}, K₀)
    k = K₀.data[1]
    IA = tens_IA(ell)
    # `P = I^A / k₀` as a tensor operation, in the basis `IA` is written in (the
    # ellipsoid's): no component is copied or changed of basis.
    return IA / k
end

"""
    _hill_order2_3d_aniso(ell::Ellipsoid{3}, K₀) -> AbstractTens{2,3}

2nd-order Hill polarization tensor of an ellipsoid in an arbitrarily
anisotropic conductor, via the closed-form square-root
change-of-variable of [giraudMOM2019](@citet)
(equivalent derivation by Green's function in
[barthelemyTIPM2009](@cite)):

```math
\\boldsymbol{P}(\\boldsymbol{A},\\boldsymbol{K})
= \\boldsymbol{K}^{-1/2}\\cdot
  \\boldsymbol{I}^{\\boldsymbol{A}\\cdot\\boldsymbol{K}^{-1/2}}\\cdot
  \\boldsymbol{K}^{-1/2} ,
```

where ``\\boldsymbol{I}^{\\boldsymbol{A}\\cdot\\boldsymbol{K}^{-1/2}}`` is the Newton
potential of the fictitious ellipsoid whose shape tensor is
``\\boldsymbol{A}\\cdot\\boldsymbol{K}^{-1/2}`` (semi-axes obtained by
diagonalizing ``\\boldsymbol{K}^{-1/2}\\cdot\\boldsymbol{A}^{\\!T}\\!\\cdot
\\boldsymbol{A}\\cdot\\boldsymbol{K}^{-1/2}``).
"""
function _hill_order2_3d_aniso(ell::Ellipsoid{3}, K₀)
    # In the ellipsoid's principal frame, where its shape tensor is diagonal: only
    # the conductivity, a 3×3, changes basis, and the result is written in that
    # frame, as the elastic kernels write theirs. `K₀[i, j]`, read before as
    # canonical, gave the components of `K₀` in its own basis, wrong for a
    # conductivity written in a rotated one.
    K_arr = Array(TensND.components(K₀, ell.basis))
    A = Diagonal(collect(ell.semi_axes))
    return MFH_Core._tens_in(_hill_order2_aniso(A, K_arr), ell.basis)
end

# P = K^{-1/2} · I^{A·K^{-1/2}} · K^{-1/2} in closed form: the semi-axes and
# axes of the fictitious ellipsoid from LAPACK, so floating-point only.
function _hill_order2_aniso(A::AbstractMatrix{<:AbstractFloat}, K_arr::AbstractMatrix{<:AbstractFloat})
    F = eigen(Symmetric(K_arr))
    invsqrt_K = F.vectors * Diagonal(1 ./ sqrt.(F.values)) * F.vectors'

    F2 = svd(A * invsqrt_K)
    perm = sortperm(F2.S, rev = true)
    s = F2.S[perm]
    V = F2.V[:, perm]

    if size(A, 1) == 3
        Iv, _ = MFH_Core.newton_potential_3d(s[1], s[2], s[3])
        P₀_arr = V * Diagonal([Iv[1], Iv[2], Iv[3]] ./ (4π)) * V'
    else
        Iv2 = MFH_Core.newton_potential_2d(s[1], s[2])
        P₀_arr = V * Diagonal([Iv2[1], Iv2[2]] ./ (2π)) * V'
    end
    return invsqrt_K * P₀_arr * invsqrt_K
end

# Any other element type: comparable ones (ForwardDiff duals) by the integral
# below, symbolic ones by the closed form of a conductivity diagonal in the
# ellipsoid's frame.
_hill_order2_aniso(A::AbstractMatrix, K_arr::AbstractMatrix) =
    _hill_order2_aniso(Val(TensND.is_hard_numeric(real(promote_type(eltype(A), eltype(K_arr))))), A, K_arr)

# ForwardDiff duals: `eigen` and `svd` have no derivatives to give, and at a
# repeated eigenvalue no axes. The same tensor
# is, in any dimension d and with G = A·Aᵀ,
#
#     P = (det A / 2) ∫₀^∞ (G + u K)⁻¹ det(G + u K)^{-1/2} du ,
#
# (the Newton potential of the fictitious ellipsoid, pulled back by the change
# of variable x ↦ K^{-1/2} x), which needs neither. Its value is taken from the
# closed form at the values, so that the value of a differentiated evaluation
# is the evaluation itself, and its derivatives from the integral.
function _hill_order2_aniso(::Val{true}, A::AbstractMatrix, K_arr::AbstractMatrix)
    P₀ = _hill_order2_aniso(MFH_Core.primal.(A), MFH_Core.primal.(K_arr))
    P = _hill_order2_integral(A, K_arr)
    return P₀ .+ (P .- MFH_Core.primal.(P))
end

# Symbolic components: no eigendecomposition and no quadrature, but when the
# conductivity is diagonal in the ellipsoid's frame (structurally: its
# off-diagonal entries are the exact zero), K^{-1/2} is diagonal too, the
# fictitious ellipsoid has the semi-axes aᵢ / √Kᵢᵢ on the same axes, and
# Pᵢᵢ = Iᵢ / (2(d − 1)π Kᵢᵢ). Any other conductivity is refused.
function _hill_order2_aniso(::Val{false}, A::Diagonal, K_arr::AbstractMatrix)
    d = size(K_arr, 1)
    all(isequal(K_arr[i, j], 0) for i in 1:d, j in 1:d if i != j) || throw(
        ArgumentError(
            "the Hill tensor of an ellipsoid in a conductivity with symbolic components " *
                "is only written in closed form when the conductivity is diagonal in the " *
                "ellipsoid's frame; got the components $K_arr in that frame"
        )
    )
    k = diag(K_arr)
    s = A.diag ./ sqrt.(k)
    T = eltype(s)
    Iv = d == 3 ? MFH_Core.newton_potential_3d(s...)[1] ./ (4 * T(π)) :
        MFH_Core.newton_potential_2d(s...) ./ (2 * T(π))
    return Matrix(Diagonal(collect(Iv) ./ k))
end

_hill_order2_integral(A::AbstractMatrix, K_arr::AbstractMatrix) =
    _hill_order2_integral(SMatrix{size(A, 1), size(A, 1)}(A), K_arr)

function _hill_order2_integral(A::SMatrix{d, d}, K_arr::AbstractMatrix) where {d}
    G = A * A'
    K = SMatrix{d, d}(K_arr)
    # u = c (t / (1 − t))², c a scale of G over K: the integrand, which
    # decays as u^{-(d+2)/2}, is then smooth on [0, 1] at both ends.
    c = tr(MFH_Core.primal.(G)) / tr(MFH_Core.primal.(K))
    function integrand(t)
        τ = t / (1 - t)
        M = G + (c * τ^2) * K
        return inv(M) * (2c * τ / (1 - t)^2 / sqrt(det(M)))
    end
    I, _ = quadgk(integrand, 0.0, 1.0; rtol = 1.0e-13, norm = MFH_Core._quadgk_norm)
    return Matrix(det(A) / 2 * I)
end

# ── Infinite cylinder (axis = e₁) ────────────────────────────────────────────

"""
    _hill_order2_3d_iso(cyl::Cylinder, K₀) -> AbstractTens{2,3}

2nd-order Hill polarization tensor of an infinite cylinder (axis
``\\underline{e}_1``, transverse semi-axes ``b\\ge c>0``) in an
isotropic conductor ``\\boldsymbol{K}_0 = k_0\\,\\boldsymbol{1}``, obtained from the
cylinder Newton-potential coefficients
[mura1987; §11.22](@cite):

```math
\\boldsymbol{P} = \\frac{\\boldsymbol{I}^{\\mathrm{cyl}}}{k_0} ,
\\qquad
I_1^{\\mathrm{cyl}} = 0,
\\quad
I_2^{\\mathrm{cyl}} = \\frac{c}{b+c},
\\quad
I_3^{\\mathrm{cyl}} = \\frac{b}{b+c} .
```

``P_{11} = 0`` expresses that no polarization is transmitted along the
cylinder axis.
"""
function _hill_order2_3d_iso(cyl::Cylinder, K₀)
    k = K₀.data[1]
    IA = tens_IA(cyl)
    return IA / k             # as for the ellipsoid, in the cylinder's basis
end

"""
    _hill_order2_3d_aniso(cyl::Cylinder, K₀) -> AbstractTens{2,3}

2nd-order Hill polarization tensor of an infinite cylinder in an
arbitrarily anisotropic conductor.  The transverse plane
``(\\underline{e}_2,\\underline{e}_3)`` carries the full 2-D
Hill problem: the ``\\boldsymbol{K}^{-1/2}`` transformation of
[giraudMOM2019](@citet) is applied to the
transverse 2×2 sub-matrix of ``\\boldsymbol{K}_0`` in the cylinder frame;
the 2-D Newton potentials produce the transverse block, and the axial
row/column is re-embedded as zero (``P_{1j}=0``).
"""
function _hill_order2_3d_aniso(cyl::Cylinder, K₀)
    # The transverse block of `K₀` in the cylinder's frame carries the 2-D
    # problem; the axial row and column of `P` are zero.
    K2 = Array(TensND.components(K₀, cyl.basis))[2:3, 2:3]
    b, c = cyl.semi_axes
    P2 = _hill_order2_aniso(Diagonal([b, c]), K2)
    return MFH_Core._tens_in(cat(zero(eltype(P2)), P2; dims = (1, 2)), cyl.basis)
end
