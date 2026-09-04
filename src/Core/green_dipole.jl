# =============================================================================
#  green_dipole.jl
#
#  Real-space gradient of the elastic Green function of an *isotropic*
#  infinite matrix (Kelvin / Lord Kelvin 1848), and the displacement field it
#  generates when contracted with a polarization (force-dipole) tensor.
#
#  Unlike the Fourier-space Green operator used by the Hill-tensor kernels,
#  this is the point-source solution in physical space.  Its role is to
#  provide the far field of a localized heterogeneity, which is what makes a
#  *finite* numerical Eshelby cell behave like an infinite medium: imposing
#
#       u(x) = E·x + ∇G(x) : Π            on the outer boundary
#
#  instead of the plain Hashin condition `u = E·x` removes the leading
#  O((a/R)³) bias of the truncation.  See Adessina, Barthélémy, Lavergne &
#  Ben Fraj, *Int. J. Eng. Sci.* 119 (2017) 1-15.
#
#  The anisotropic counterpart (Pan-Chou closed form for transverse isotropy,
#  or the Barnett-Willis line integral in general) is not implemented.
# =============================================================================

"""
    green_gradient_iso(C₀::TensISO{4,3}, x) -> SArray{Tuple{3,3,3}}

Gradient ``\\partial G_{ij}/\\partial x_k`` of the Kelvin Green function of an
isotropic elastic matrix `C₀`, evaluated at `x ≠ 0`.

With ``r = \\|x\\|``, ``\\hat n = x/r`` and
``A = 1/\\bigl(16\\pi\\mu(1-\\nu)\\bigr)`` the Kelvin solution reads

```math
G_{ij}(x) = \\frac{A}{r}\\left[(3-4\\nu)\\,\\delta_{ij} + n_i n_j\\right],
```

so that

```math
\\frac{\\partial G_{ij}}{\\partial x_k}
  = \\frac{A}{r^{2}}\\left[-(3-4\\nu)\\,\\delta_{ij} n_k
      + \\delta_{ik} n_j + \\delta_{jk} n_i - 3\\,n_i n_j n_k\\right].
```

Returned as a static `3×3×3` array indexed `[i, j, k]`.

Throws a `DomainError` at the origin.  Type-generic (`Float64`,
`ForwardDiff.Dual`, symbolic scalars).

See also [`dipole_displacement_iso`](@ref).
"""
function green_gradient_iso(C₀::TensND.TensISO{4, 3}, x::AbstractVector)
    E, ν = extract_iso_moduli(C₀)
    μ = E / (2 * (1 + ν))
    return _green_gradient_iso(μ, ν, x)
end

function _green_gradient_iso(μ, ν, x::AbstractVector)
    r = sqrt(x[1]^2 + x[2]^2 + x[3]^2)
    iszero(r) && throw(
        DomainError(
            x, "the Kelvin Green function is singular at the origin"
        )
    )
    n = SVector{3}(x[1] / r, x[2] / r, x[3] / r)
    A = one(r) / (16 * π * μ * (1 - ν) * r^2)
    c = 3 - 4 * ν
    δ = (i, j) -> _δ(i, j, typeof(A))
    return SArray{Tuple{3, 3, 3}}(
        @inbounds [
            A * (
                    -c * δ(i, j) * n[k] + δ(i, k) * n[j] + δ(j, k) * n[i] -
                    3 * n[i] * n[j] * n[k]
                )
                for i in 1:3, j in 1:3, k in 1:3
        ]
    )
end

"""
    dipole_displacement_iso(C₀::TensISO{4,3}, x, Π) -> SVector{3}

Displacement field at `x` generated in an infinite isotropic matrix `C₀` by a
point **polarization** (force dipole) `Π`, i.e. the contraction

```math
u_i(x) = \\frac{\\partial G_{ij}}{\\partial x_k}(x)\\; \\Pi_{jk}.
```

`Π` has the dimension of a stress times a volume: for an inclusion of volume
``V_{\\mathcal I}`` carrying a uniform polarization ``\\pi`` (i.e.
``\\sigma = \\mathbb C_0 : \\varepsilon + \\pi`` inside it),
``\\Pi = V_{\\mathcal I}\\,\\pi``.

For a symmetric `Π` the closed form collapses to

```math
u(x) = \\frac{1}{16\\pi\\mu(1-\\nu)r^{2}}
   \\Bigl[-2(1-2\\nu)\\,\\Pi\\!\\cdot\\!\\hat n
          + \\mathrm{tr}(\\Pi)\\,\\hat n
          - 3(\\hat n\\!\\cdot\\!\\Pi\\!\\cdot\\!\\hat n)\\,\\hat n\\Bigr],
```

which is the form evaluated here — it costs one matrix-vector product instead
of building the full `3×3×3` gradient, and is the expression used for the
corrected boundary condition of a finite Eshelby cell.

`Π` may be given as a `3×3` matrix or as a `TensND` 2nd-order tensor; in the
latter case it is read in the canonical frame.

See also [`green_gradient_iso`](@ref).
"""
function dipole_displacement_iso(C₀::TensND.TensISO{4, 3}, x::AbstractVector, Π)
    E, ν = extract_iso_moduli(C₀)
    μ = E / (2 * (1 + ν))
    return _dipole_displacement_iso(μ, ν, x, _as_matrix3(Π))
end

function _dipole_displacement_iso(μ, ν, x::AbstractVector, P::AbstractMatrix)
    r = sqrt(x[1]^2 + x[2]^2 + x[3]^2)
    iszero(r) && throw(
        DomainError(
            x, "the Kelvin Green function is singular at the origin"
        )
    )
    n = SVector{3}(x[1] / r, x[2] / r, x[3] / r)
    A = one(r) / (16 * π * μ * (1 - ν) * r^2)
    Pn = SVector{3}(
        P[1, 1] * n[1] + P[1, 2] * n[2] + P[1, 3] * n[3],
        P[2, 1] * n[1] + P[2, 2] * n[2] + P[2, 3] * n[3],
        P[3, 1] * n[1] + P[3, 2] * n[2] + P[3, 3] * n[3],
    )
    trP = P[1, 1] + P[2, 2] + P[3, 3]
    nPn = Pn[1] * n[1] + Pn[2] * n[2] + Pn[3] * n[3]
    a = -2 * (1 - 2 * ν)
    return A * (a * Pn + (trP - 3 * nPn) * n)
end

_as_matrix3(P::AbstractMatrix) = P
_as_matrix3(P::TensND.AbstractTens{2, 3}) = TensND.components_canon(P)
