# =============================================================================
#  green_aniso.jl — real-space Green function and Green operator of an
#  *anisotropic* infinite medium.
#
#  This is what `green_operator.jl` leaves open.  Where the isotropic kernel is
#  a closed form, a general anisotropic medium has no elementary one: the
#  displacement Green function is instead a line integral over the unit circle
#  perpendicular to the observation direction (Barnett 1972; Willis 1975),
#
#       G(x) = 1/(8π²r) ∮_{ξ ⊥ n̂, ‖ξ‖=1} K(ξ)⁻¹ dφ ,
#       K_ij(ξ) = ξ_k C_kijl ξ_l   (the acoustic / Christoffel tensor),
#
#  with r = ‖x‖ and n̂ = x/r.  The integrand is smooth and periodic, and the
#  whole expression is homogeneous of degree −1 in x.
#
#  The Green OPERATOR needs two more derivatives.  Rather than differentiating
#  the line integral by hand — the classical route, and an error-prone one —
#  they are taken with `ForwardDiff` through the quadrature itself. That is
#  exact (not a finite difference), type-generic, and reuses a single verified
#  expression instead of a second hand-derived one.  The value of the integral
#  does not depend on which orthonormal pair spans the perpendicular plane, so
#  differentiating through that choice is legitimate.
#
#  Conduction is easier: the anisotropic scalar Green function is elementary in
#  every dimension, so its Hessian is written in closed form below.
#
#  Ported from the exploratory `Green.jl` of `echoes_cpp/tests/python/Green/`,
#  which also carries the Pan-Chou closed form for transverse isotropy and a
#  TraTI push-forward. Only the general-anisotropy branch is brought over: the
#  line integral covers the transversely isotropic case too, and one code path
#  is one code path to validate.
#
#  NOTE — a normalization slip in the source. `Green.jl`'s scalar `Green2` /
#  `GradGreen2` divide by `det(c)` where the anisotropic Laplace Green function
#  needs `sqrt(det(c))`; with `c = σ₀ 𝟏` they return `1/(4π σ₀^{5/2} r)` instead
#  of `1/(4π σ₀ r)`. The correct normalization is used here, and the isotropic
#  cross-check in the test suite is what pins it.
# =============================================================================

"""
    _christoffel(C, ξ) -> SMatrix{3,3}

Acoustic (Christoffel) tensor ``K_{ij}(\\xi) = \\xi_k\\,C_{kijl}\\,\\xi_l`` of a
stiffness given as a `3×3×3×3` array.
"""
function _christoffel(C, ξ)
    T = promote_type(eltype(C), eltype(ξ))
    return SMatrix{3, 3, T}(
        @inbounds [
            sum(ξ[k] * C[k, i, j, l] * ξ[l] for k in 1:3, l in 1:3)
                for i in 1:3, j in 1:3
        ]
    )
end

"""
    _perp_basis(n̂) -> (u₁, u₂)

Two orthonormal vectors spanning the plane perpendicular to the unit vector
`n̂`. The choice is deliberate but arbitrary: the line integral below is
invariant under a rotation of `(u₁, u₂)` about `n̂`, which is what makes it
safe to differentiate through them.
"""
function _perp_basis(n̂)
    T = eltype(n̂)
    a = abs(n̂[3]) < 0.9 ? SVector{3, T}(0, 0, 1) : SVector{3, T}(1, 0, 0)
    u₁ = a - (a ⋅ n̂) * n̂
    u₁ = u₁ / sqrt(u₁ ⋅ u₁)
    return u₁, SVector{3, T}(n̂ × u₁)
end

# Gauss-Legendre rules are a function of the node count alone, so they are
# built once and kept.  This matters far more than it looks: the anisotropic
# Green operator differentiates its quadrature with nested duals, so an
# uncached rule would re-solve a 64x64 symmetric eigenproblem on *every* dual
# evaluation — measured at ~1.2 s per interaction tensor before caching, ~1 ms
# after.
const _GL_CACHE = Dict{Int, Tuple{Vector{Float64}, Vector{Float64}}}()

"""
    gauss_legendre_nodes(n, lo = -1, hi = 1) -> (x, w)

Gauss-Legendre nodes and weights on `[lo, hi]`, from the Golub-Welsch
eigenvalue problem of the Jacobi matrix. A few lines, no extra dependency, and
plenty accurate at the orders used in the package.

The rule for a given `n` is computed once and memoized — see the note above the
definition; this is a load-bearing optimization, not a micro-tuning.
"""
function gauss_legendre_nodes(n::Int, lo::Real = -1.0, hi::Real = 1.0)
    x, w = get!(_GL_CACHE, n) do
        n == 1 && return ([0.0], [2.0])
        β = [k / sqrt(4k^2 - 1) for k in 1:(n - 1)]
        F = eigen(SymTridiagonal(zeros(n), β))
        return (F.values, 2 .* (F.vectors[1, :] .^ 2))
    end
    return (hi + lo) / 2 .+ (hi - lo) / 2 .* x, (hi - lo) / 2 .* w
end

"""
    green_function_aniso(C₀, x; nodes = 32) -> SMatrix{3,3}

Displacement Green function of an infinite medium of arbitrary anisotropic
stiffness `C₀`, evaluated at `x ≠ 0` by the Barnett line integral

```math
G_{ij}(\\underline{x}) = \\frac{1}{8\\pi^2 r}
  \\oint_{\\underline{\\xi}\\perp\\underline{n},\\,\\|\\underline{\\xi}\\|=1}
     \\big[\\boldsymbol{K}(\\underline{\\xi})\\big]^{-1}_{ij}\\, \\mathrm{d}\\varphi ,
\\qquad
K_{ij}(\\underline{\\xi}) = \\xi_k\\, C_{kijl}\\, \\xi_l ,
```

with ``r = \\|\\underline{x}\\|`` and ``\\underline{n} = \\underline{x}/r``. The
integrand is smooth and periodic, so a Gauss-Legendre rule converges fast: the
Green *function* itself is at machine accuracy by 16 nodes.

Reduces to the Kelvin solution for an isotropic `C₀`, which is how the
implementation is tested. Type-generic, including `ForwardDiff.Dual` — the
Green operator is obtained by differentiating this very function.
"""
function green_function_aniso(C₀::TensND.AbstractTens{4, 3}, x::AbstractVector; nodes::Int = 32)
    return _green_function_aniso(_C_array(C₀), x, nodes)
end

function _green_function_aniso(C, x::AbstractVector, nodes::Int)
    T = promote_type(eltype(C), eltype(x))
    r = sqrt(x[1]^2 + x[2]^2 + x[3]^2)
    iszero(r) && throw(
        DomainError(x, "the Green function of an infinite medium is singular at the origin")
    )
    n̂ = SVector{3}(x[1] / r, x[2] / r, x[3] / r)
    u₁, u₂ = _perp_basis(n̂)
    t, w = gauss_legendre_nodes(nodes, 0.0, 2π)
    G = zero(SMatrix{3, 3, T})
    @inbounds for m in eachindex(t)
        ξ = cos(t[m]) * u₁ + sin(t[m]) * u₂
        G = G + w[m] * inv(_christoffel(C, ξ))
    end
    return G / (8 * π^2 * r)
end

"""
    green_operator_aniso(C₀, x; nodes = 32) -> SArray{Tuple{3,3,3,3}}

Real-space Green operator of an arbitrary anisotropic elastic medium,

```math
\\mathbb{G}^0_{ijkl}(\\underline{x}) = \\Big[\\frac{\\partial^2 G_{ik}}
  {\\partial x_j\\, \\partial x_l}(\\underline{x})\\Big]_{(ij)(kl)} .
```

The second gradient is taken with `ForwardDiff` through the line integral of
[`green_function_aniso`](@ref) — exact, and reusing one verified expression
rather than a second hand-derived one.

Two derivatives cost accuracy, so the operator needs a finer rule than the
function — and how much finer depends on the anisotropy. For an *isotropic*
stiffness the Christoffel inverse is a low-order trigonometric polynomial and
the rule is exact by 16 nodes. For a genuinely anisotropic one (a cubic
stiffness with its shear constant detuned by a third) the measured relative
error against a 128-node reference is `1.4e-2` at 16 nodes, `8e-4` at 24,
`1.7e-5` at 32 and `1.5e-12` at 64.

The default of 32 sits well below the truncation error of the multipole
expansion that consumes it, at about a ninth of the cost of 64 (≈ 1.5 ms versus
13 ms per interaction tensor). Raise it for a strongly anisotropic reference.

Considerably more expensive than the isotropic closed form
([`green_operator_iso`](@ref)) — microseconds against milliseconds — which is
why the dispatcher [`green_operator`](@ref) prefers the latter whenever the
reference is isotropic.
"""
function green_operator_aniso(
        C₀::TensND.AbstractTens{4, 3}, x::AbstractVector; nodes::Int = 32
    )
    C = _C_array(C₀)
    # H[(i,j), k, l] = ∂²G_ij/∂x_k∂x_l, obtained as a jacobian of a jacobian.
    f = y -> vec(Matrix(_green_function_aniso(C, y, nodes)))
    J2 = ForwardDiff.jacobian(y -> vec(ForwardDiff.jacobian(f, y)), collect(x))
    T = eltype(J2)
    H = Array{T, 4}(undef, 3, 3, 3, 3)
    @inbounds for i in 1:3, j in 1:3, k in 1:3, l in 1:3
        # `f` is column-major over (i, j); the inner jacobian appends k, the
        # outer one appends l.
        H[i, j, k, l] = J2[(k - 1) * 9 + (j - 1) * 3 + i, l]
    end
    return SArray{Tuple{3, 3, 3, 3}}(
        @inbounds [
            (H[i, k, j, l] + H[j, k, i, l] + H[i, l, j, k] + H[j, l, i, k]) / 4
                for i in 1:3, j in 1:3, k in 1:3, l in 1:3
        ]
    )
end

"""
    green_operator_aniso(K₀::AbstractTens{2,3}, x) -> SArray{Tuple{3,3}}

Conduction counterpart in three dimensions. Here the Green function *is*
elementary,

```math
G(\\underline{x}) = \\frac{1}{4\\pi\\sqrt{\\det \\boldsymbol{K}_0}\\;
   \\sqrt{\\underline{x}\\cdot\\boldsymbol{K}_0^{-1}\\cdot\\underline{x}}} ,
```

so its Hessian is written in closed form: with
``\\underline{y} = \\boldsymbol{K}_0^{-1}\\cdot\\underline{x}`` and
``s = \\underline{x}\\cdot\\underline{y}``,

```math
\\boldsymbol{G}^0 = \\frac{1}{4\\pi\\sqrt{\\det\\boldsymbol{K}_0}}
  \\left[\\frac{3\\,\\underline{y}\\otimes\\underline{y}}{s^{5/2}}
       - \\frac{\\boldsymbol{K}_0^{-1}}{s^{3/2}}\\right].
```

It is `K₀`-traceless in the sense ``\\boldsymbol{K}_0 : \\boldsymbol{G}^0 = 0``,
the anisotropic form of the vanishing isotropic part.
"""
function green_operator_aniso(K₀::TensND.AbstractTens{2, 3}, x::AbstractVector)
    return _green_operator_aniso_order2(TensND.get_array(K₀), x, 3)
end

"""
    green_operator_aniso(K₀::AbstractTens{2,2}, x) -> SArray{Tuple{2,2}}

Two-dimensional conduction counterpart, from
``G = -\\log\\sqrt{\\underline{x}\\cdot\\boldsymbol{K}_0^{-1}\\cdot\\underline{x}}
 / (2\\pi\\sqrt{\\det\\boldsymbol{K}_0})``:

```math
\\boldsymbol{G}^0 = \\frac{1}{2\\pi\\sqrt{\\det\\boldsymbol{K}_0}}
  \\left[\\frac{2\\,\\underline{y}\\otimes\\underline{y}}{s^{2}}
       - \\frac{\\boldsymbol{K}_0^{-1}}{s}\\right].
```
"""
function green_operator_aniso(K₀::TensND.AbstractTens{2, 2}, x::AbstractVector)
    return _green_operator_aniso_order2(TensND.get_array(K₀), x, 2)
end

function _green_operator_aniso_order2(K, x::AbstractVector, d::Int)
    Kinv = inv(Matrix(K))
    y = Kinv * collect(x)
    s = sum(x[i] * y[i] for i in 1:d)
    s > 0 || throw(
        DomainError(x, "the Green function of an infinite medium is singular at the origin")
    )
    detK = det(Matrix(K))
    pre = 1 / ((d == 3 ? 4π : 2π) * sqrt(detK))
    p = d == 3 ? 5 // 2 : 2
    q = d == 3 ? 3 // 2 : 1
    return SArray{Tuple{d, d}}(
        @inbounds [
            pre * ((d == 3 ? 3 : 2) * y[i] * y[j] / s^p - Kinv[i, j] / s^q)
                for i in 1:d, j in 1:d
        ]
    )
end

# ─── The dispatcher ──────────────────────────────────────────────────────────

"""
    green_operator(P₀, x; kw...) -> SArray

Real-space Green operator of an infinite medium of reference property `P₀`,
evaluated at `x ≠ 0` — the regular kernel of the Lippmann-Schwinger equation.

Dispatches on the symmetry class: an isotropic reference goes to the closed
form [`green_operator_iso`](@ref), anything else to
[`green_operator_aniso`](@ref); `green_nodes` sets the quadrature order of the
latter (the name is distinct from the `nodes` of
[`interaction_tensor`](@ref MeanFieldHomogenization.Interactions.interaction_tensor)'s own quadrature back-end, so both can be given at
once). Elasticity and conduction, 2D and 3D, are all
covered; the one gap is **plane-strain elasticity with an anisotropic
reference**, whose Green function needs the Stroh formalism rather than the
Barnett line integral, and which raises an `ArgumentError`.

This is the entry point the two-inclusion interaction kernels call, so lifting
the isotropic restriction here lifts it for
[`interaction_tensor`](@ref MeanFieldHomogenization.Interactions.interaction_tensor) as well.
"""
green_operator(P₀::TensND.TensISO, x::AbstractVector; kw...) = green_operator_iso(P₀, x)

green_operator(
    P₀::TensND.AbstractTens{4, 3}, x::AbstractVector; green_nodes::Int = 32, kw...
) = green_operator_aniso(P₀, x; nodes = green_nodes)

green_operator(P₀::TensND.AbstractTens{2, 3}, x::AbstractVector; kw...) =
    green_operator_aniso(P₀, x)

green_operator(P₀::TensND.AbstractTens{2, 2}, x::AbstractVector; kw...) =
    green_operator_aniso(P₀, x)

green_operator(P₀::TensND.AbstractTens{4, 2}, x::AbstractVector; kw...) = throw(
    ArgumentError(
        "green_operator: the plane-strain Green function of an anisotropic " *
            "reference needs the Stroh formalism, which is not implemented; only an " *
            "isotropic reference is supported in 2D elasticity."
    )
)
