# =============================================================================
#  shapes.jl — superspherical and superspheroidal geometry.
#
#  Everything here is closed form and independent of any solver: the level set,
#  the radial map that turns a direction into a surface point, the outward
#  normal, the characteristic radii, and the exact volume and shadow area that
#  serve as mesh-quality oracles.  Nothing in this file meshes or solves.
# =============================================================================

"""
    AbstractSuperShape{T}

A body described by a **radial map**: [`radial_distance`](@ref) gives, for a unit
direction ``\\underline n``, the distance from the center to the surface.

Every supersphere and superspheroid is star-shaped about its center for *every*
``p > 0``, the concave ones included. That is what makes the radial map
single-valued, and it is the property the whole meshing strategy rests on — a
surface mesh is obtained by pushing a subdivided sphere radially onto the shape,
with no risk of the map folding over.
"""
abstract type AbstractSuperShape{T <: Number} end

"""
    Supersphere(a, p)

The body ``|x/a|^{2p} + |y/a|^{2p} + |z/a|^{2p} \\le 1``, of cubic symmetry
(``m\\bar 3m``).

The exponent is ``2p``, not ``p``. This is the convention of the
Sevostianov–Giraud–Chen–Grgic literature these shapes come from, so

| ``p`` | shape |
|:--|:--|
| ``1`` | the sphere |
| ``1/2`` | the regular octahedron |
| ``< 1/2`` | concave, with conical points on the axes |
| ``\\to \\infty`` | the cube |

!!! warning "Two conventions for `p`"
    `scripts/common/docviz.jl` calls `p` the exponent *itself*, so its sphere
    sits at `p = 2`. The two differ by a factor of two;
    [`shape_exponent`](@ref) is what converts.

A `Supersphere` is a *shape*, not yet an inclusion: it has no closed-form
Eshelby solution. It becomes a phase of an [`RVE`](@ref) through the
finite-element or surrogate-backed inclusion built on it.
"""
struct Supersphere{T <: Number} <: AbstractSuperShape{T}
    a::T
    p::T
    # Declared explicitly, and that is the point: without an inner constructor
    # Julia generates the outer `Supersphere(a::T, p::T) where T`, which is
    # *more specific* than the checked `Supersphere(::Number, ::Number)` below
    # for two arguments of equal type. `Supersphere(-1.0, 0.5)` would then build
    # a negative semi-axis without ever reaching a validation.
    Supersphere{T}(a::T, p::T) where {T <: Number} = new(a, p)
end

"""
    Superspheroid(a, c, p)

The axisymmetric body ``(\\rho/a)^{2p} + |z/c|^{2p} \\le 1`` with
``\\rho^2 = x^2 + y^2``, transversely isotropic about ``\\underline e_3``.

Same convention for ``p`` as [`Supersphere`](@ref): ``p = 1`` is the spheroid of
semi-axes ``(a, a, c)``, ``p = 1/2`` the double cone, ``p < 1/2`` the concave
range, ``p \\to \\infty`` the cylinder.
"""
struct Superspheroid{T <: Number} <: AbstractSuperShape{T}
    a::T
    c::T
    p::T
    # See [`Supersphere`](@ref) for why this is spelled out.
    Superspheroid{T}(a::T, c::T, p::T) where {T <: Number} = new(a, c, p)
end

# The promoting constructors. `_floatlike` rather than `float`: `float(Sym)` is
# `Float64`, which would silently drop a symbolic element type at the door.
function Supersphere(a::Number, p::Number)
    T = Core._floatlike(promote_type(typeof(a), typeof(p)))
    _check_shape_parameters(T, (:a => a,), p)
    return Supersphere{T}(convert(T, a), convert(T, p))
end

function Superspheroid(a::Number, c::Number, p::Number)
    T = Core._floatlike(promote_type(typeof(a), typeof(c), typeof(p)))
    _check_shape_parameters(T, (:a => a, :c => c), p)
    return Superspheroid{T}(convert(T, a), convert(T, c), convert(T, p))
end

Supersphere(a::Number; p::Number) = Supersphere(a, p)
Superspheroid(a::Number, c::Number; p::Number) = Superspheroid(a, c, p)

# Positivity is an ORDER comparison on user data, and on a symbolic element type
# those are undecidable: SymPy answers `false` to a comparison it cannot settle,
# so validating unconditionally would reject legitimate symbolic input while
# appearing to have checked it. Validate where the type admits comparison — which
# includes `ForwardDiff.Dual`, whose `<` does return a `Bool` — and leave the
# responsibility with the caller otherwise.
function _check_shape_parameters(::Type{T}, semiaxes, p) where {T}
    is_hard_numeric(T) || return nothing
    for (name, v) in semiaxes
        v > 0 || throw(ArgumentError("semi-axis `$name` must be strictly positive, got $v"))
    end
    p > 0 || throw(ArgumentError("the shape exponent `p` must be strictly positive, got $p"))
    return nothing
end

Base.eltype(::AbstractSuperShape{T}) where {T} = T

function Base.show(io::IO, s::Supersphere)
    return print(io, "Supersphere(a = ", s.a, ", p = ", s.p, ")")
end
function Base.show(io::IO, s::Superspheroid)
    return print(io, "Superspheroid(a = ", s.a, ", c = ", s.c, ", p = ", s.p, ")")
end

"""
    shape_exponent(s) -> 2p

The exponent that appears in the level set, as opposed to the concavity
parameter ``p`` that names the shape. See the warning on [`Supersphere`](@ref).
"""
shape_exponent(s::AbstractSuperShape) = 2 * s.p

"""
    is_concave(s) -> Bool
    is_convex(s)  -> Bool
    is_sphere(s)  -> Bool

Shape classification, from ``p`` alone: concave below ``p = 1/2``, convex at or
above it, and a sphere exactly at ``p = 1``.

These are comparisons, so they need an element type on which a comparison
returns an honest `Bool` — every floating-point type and `ForwardDiff.Dual`, but
not a symbolic one, where they throw rather than answer wrongly.
"""
function is_concave(s::AbstractSuperShape{T}) where {T}
    _require_comparable(T, "is_concave")
    return s.p < 1 // 2
end

function is_convex(s::AbstractSuperShape{T}) where {T}
    _require_comparable(T, "is_convex")
    return s.p ≥ 1 // 2
end

function is_sphere(s::Supersphere{T}) where {T}
    _require_comparable(T, "is_sphere")
    return s.p == 1
end

function _require_comparable(::Type{T}, what) where {T}
    is_hard_numeric(T) && return nothing
    throw(
        ArgumentError(
            "`$what` is a comparison and $T does not compare to a `Bool`. " *
                "Substitute numeric values first, or branch on the value you " *
                "already know `p` to have."
        )
    )
end

# ─── The level set and the radial map ────────────────────────────────────────

"""
    level_set(s, x) -> f

``f < 0`` strictly inside, ``f = 0`` on the surface, ``f > 0`` outside:
``f(\\underline x) = \\sum_i |x_i/a_i|^{2p} - 1``.
"""
function level_set(s::Supersphere, x)
    m = shape_exponent(s)
    return abs(x[1] / s.a)^m + abs(x[2] / s.a)^m + abs(x[3] / s.a)^m - 1
end

function level_set(s::Superspheroid, x)
    m = shape_exponent(s)
    ρ = sqrt(x[1]^2 + x[2]^2)
    return (ρ / s.a)^m + abs(x[3] / s.c)^m - 1
end

"""
    radial_distance(s, n) -> r

Distance from the center to the surface along the **unit** direction
``\\underline n``.

No root finding is involved. The level set is positively homogeneous of degree
``2p``, so ``f(r\\,\\underline n) = 0`` solves in closed form:

```math
r = \\Bigl[\\, \\sum_i |n_i/a_i|^{2p} \\Bigr]^{-1/(2p)} .
```

That closed form is why the radial map is cheap enough to apply at every node of
a refined surface mesh, and why it differentiates cleanly with respect to the
shape parameters.
"""
function radial_distance(s::Supersphere, n)
    m = shape_exponent(s)
    q = (abs(n[1])^m + abs(n[2])^m + abs(n[3])^m) / s.a^m
    return q^(-inv(m))
end

function radial_distance(s::Superspheroid, n)
    m = shape_exponent(s)
    ρ = sqrt(n[1]^2 + n[2]^2)
    q = (ρ / s.a)^m + abs(n[3] / s.c)^m
    return q^(-inv(m))
end

"""
    surface_point(s, n) -> x

``r\\,\\underline n`` with ``r`` from [`radial_distance`](@ref): the radial
projection of a unit direction onto the surface. This is the map that turns a
subdivided sphere into a supersphere.
"""
surface_point(s::AbstractSuperShape, n) = radial_distance(s, n) .* n

"""
    outward_normal(s, x) -> n

Unit outward normal at a **surface** point, from ``\\nabla f``.

```math
\\partial_i f = \\operatorname{sign}(x_i)\\,|x_i/a_i|^{2p-1}/a_i
```

which is finite and non-zero as long as no coordinate vanishes. Where one does,
the surface is generally not differentiable, and the two ways it fails are
opposite:

* for ``2p > 1`` the vanishing coordinate's derivative **vanishes** too, and it
  contributes exactly nothing — the normal is well defined and comes from the
  surviving coordinates;
* for ``2p < 1`` it **diverges**, because the surface creases on the coordinate
  planes and spikes at the conical points on the axes. There the normal is not
  unique and no formula can invent one, so the cases are named rather than
  computed:

  | vanishing coordinates | feature | returned |
  |:--|:--|:--|
  | one, say ``x_i = 0`` | a crease | ``+\\underline e_i`` |
  | two or three | a conical point on an axis | the radial direction ``\\underline x/\\|\\underline x\\|`` |

  The crease's one-sided limits are ``+\\underline e_i`` as ``x_i \\to 0^+`` and
  ``-\\underline e_i`` as ``x_i \\to 0^-``; with ``x_i`` exactly zero there is
  nothing to choose between them and ``+\\underline e_i`` is the convention. The
  conical point's normals span a cone, and the radial direction is its axis.

Writing the raw formula and normalizing afterwards does **not** work: at such a
point it forms `0 * Inf`, Julia returns `NaN`, and a `NaN` normal that reaches a
mesh generator produces a negative Jacobian somewhere far away from the cause.

On a symbolic element type the case analysis is skipped — it is a set of
comparisons — and the normalized gradient is returned as it stands.
"""
function outward_normal(s::Supersphere{T}, x) where {T}
    m = shape_exponent(s)
    u = ntuple(i -> x[i] / s.a, 3)
    is_hard_numeric(T) || return _normalize3(
        ntuple(i -> sign(u[i]) * abs(u[i])^(m - 1) / s.a, 3)
    )
    _require_surface_point(x)
    if m ≥ 1 || !any(_vanishes, u)
        # `m ≥ 1` covers `m == 1` too, where Julia's `0^0 = 1` is harmless
        # because `sign(0) = 0` multiplies it away.
        g = ntuple(
            i -> _vanishes(u[i]) ? zero(u[i]) / s.a :
                sign(u[i]) * abs(u[i])^(m - 1) / s.a, 3
        )
        return _normalize3(g)
    end
    return _nonsmooth_normal(u, x)
end

function outward_normal(s::Superspheroid{T}, x) where {T}
    m = shape_exponent(s)
    ρ = sqrt(x[1]^2 + x[2]^2)
    uρ, uz = ρ / s.a, x[3] / s.c
    if !is_hard_numeric(T)
        gρ = abs(uρ)^(m - 1) / s.a
        return _normalize3(
            (gρ * x[1] / ρ, gρ * x[2] / ρ, sign(uz) * abs(uz)^(m - 1) / s.c)
        )
    end
    _require_surface_point(x)
    onaxis = _vanishes(uρ)
    if m ≥ 1 || !(onaxis || _vanishes(uz))
        gρ = onaxis ? zero(uρ) / s.a : abs(uρ)^(m - 1) / s.a
        g = (
            onaxis ? zero(gρ) : gρ * x[1] / ρ,
            onaxis ? zero(gρ) : gρ * x[2] / ρ,
            _vanishes(uz) ? zero(gρ) : sign(uz) * abs(uz)^(m - 1) / s.c,
        )
        return _normalize3(g)
    end
    # `2p < 1`. The equatorial plane is the crease; the two poles are the
    # conical points, whose normal cone has ±ê₃ for axis either way.
    z = zero(uρ)
    return onaxis ? (z, z, oftype(z, sign(x[3]))) : (z, z, one(z))
end

# A comparison on the *value*: `x[i] == 0` rather than `iszero(x[i])`, because a
# `ForwardDiff.Dual` carrying a zero value and a non-zero partial is at the
# singular point and must take the singular branch. `==` on a `Dual` compares
# values; `iszero` would also demand that the partials vanish and so would send
# it down the smooth branch, straight into the `0 * Inf`.
_vanishes(v) = v == 0

function _normalize3(g)
    nrm = sqrt(g[1]^2 + g[2]^2 + g[3]^2)
    return (g[1] / nrm, g[2] / nrm, g[3] / nrm)
end

function _require_surface_point(x)
    (_vanishes(x[1]) && _vanishes(x[2]) && _vanishes(x[3])) && throw(
        ArgumentError("`outward_normal` needs a point on the surface, got the center")
    )
    return nothing
end

function _nonsmooth_normal(u, x)
    vanishing = ntuple(i -> _vanishes(u[i]), 3)
    z = zero(u[1])
    if count(vanishing) == 1
        i = findfirst(vanishing)
        return ntuple(k -> k == i ? one(z) : z, 3)
    end
    return _normalize3(ntuple(i -> z + x[i], 3))
end

# ─── Characteristic radii ────────────────────────────────────────────────────

"""
    diagonal_radius(s::Supersphere)

Radius along ``(1,1,1)/\\sqrt3``, ``a\\,3^{(p-1)/2p}``.

The cubic group has three critical directions — axis, edge midpoint, body
diagonal — and with ``m = 2p`` their radii are

```math
r_k = a\\, k^{1/2 - 1/m}, \\qquad k = 1, 2, 3,
```

which is **monotone in ``k``**. So the edge midpoint always lies between the
other two, and the extremal radii of a supersphere are reached on the axis and
on the diagonal, never in between — which is what makes
[`bounding_radius`](@ref) and [`inner_radius`](@ref) a two-term comparison
rather than an optimization.

It is also the shape's cheapest signature: ``p = 1`` gives ``a``, ``p = 1/2``
gives ``a/\\sqrt3``, and ``p \\to \\infty`` gives ``a\\sqrt3``, the half-diagonal
of the cube of half-side ``a``.
"""
diagonal_radius(s::Supersphere) = s.a * 3^((s.p - 1) / (2 * s.p))

"""
    edge_radius(s::Supersphere)

Radius along ``(1,1,0)/\\sqrt2``, ``a\\,2^{1/2 - 1/2p}``: the case ``k = 2`` of
[`diagonal_radius`](@ref), always between ``a`` and the diagonal radius.
"""
edge_radius(s::Supersphere) = s.a * 2^(1 // 2 - inv(shape_exponent(s)))

# For a superspheroid the extremum is *not* on an axis in general. With
# u = sin²θ the radius is r = q^(-1/m) where
#
#     q(u) = u^(m/2)/aᵐ + (1-u)^(m/2)/cᵐ ,
#
# whose interior critical point solves u/(1-u) = (a/c)^(2m/(m-2)). It is a
# minimum of q — hence a MAXIMUM of r — for m > 2 and the other way round for
# m < 2. At m = 2, the true spheroid, q is affine in u and the extrema sit at
# the poles and the equator. Missing this interior point is how a bounding
# radius ends up smaller than the body it is meant to bound.
function _superspheroid_critical_u(s::Superspheroid{T}) where {T}
    is_hard_numeric(T) || return nothing
    m = shape_exponent(s)
    m == 2 && return nothing
    lr = (2 * m / (m - 2)) * log(s.a / s.c)          # log(u / (1-u))
    isfinite(lr) || return nothing                   # merged with a pole
    u = lr > 0 ? inv(1 + exp(-lr)) : exp(lr) / (1 + exp(lr))
    return (u > 0 && u < 1) ? u : nothing
end

function _superspheroid_radius_at_u(s::Superspheroid, u)
    m = shape_exponent(s)
    q = u^(m / 2) / s.a^m + (1 - u)^(m / 2) / s.c^m
    return q^(-inv(m))
end

function _extremal_radii(s::Superspheroid{T}) where {T}
    _require_comparable(T, "bounding_radius / inner_radius")
    u = _superspheroid_critical_u(s)
    u === nothing && return (min(s.a, s.c), max(s.a, s.c))
    rc = _superspheroid_radius_at_u(s, u)
    return (min(s.a, s.c, rc), max(s.a, s.c, rc))
end

"""
    bounding_radius(s)

Smallest ``R`` such that the body fits inside the ball of radius ``R``.
"""
function bounding_radius(s::Supersphere{T}) where {T}
    _require_comparable(T, "bounding_radius")
    return max(s.a, diagonal_radius(s))
end
bounding_radius(s::Superspheroid) = _extremal_radii(s)[2]

"""
    inner_radius(s)

Largest ``r`` such that the ball of radius ``r`` fits inside the body.
"""
function inner_radius(s::Supersphere{T}) where {T}
    _require_comparable(T, "inner_radius")
    return min(s.a, diagonal_radius(s))
end
inner_radius(s::Superspheroid) = _extremal_radii(s)[1]

# ─── Exact volume and shadow area — the mesh-quality oracles ─────────────────
#
# All three closed forms follow from the Dirichlet integral, for xᵢ > 0 and
# exponent m:
#
#     ∫_{Σ xᵢᵐ ≤ 1} dx₁…dxₙ = Γ(1 + 1/m)ⁿ / Γ(1 + n/m) .
#
# Evaluated through `loggamma` so that a small p — that is, a large 1/m — does
# not overflow on the way to a perfectly finite answer.

_dirichlet(n::Integer, m) = exp(n * loggamma(1 + inv(m)) - loggamma(1 + n / m))

"""
    shape_volume(s) -> V

Exact volume, in closed form:

```math
V_{\\text{supersphere}} = 8a^3\\,
  \\frac{\\Gamma(1 + \\tfrac{1}{2p})^3}{\\Gamma(1 + \\tfrac{3}{2p})},
\\qquad
V_{\\text{superspheroid}} = \\frac{2\\pi a^2 c}{2p}\\,
  \\frac{\\Gamma(\\tfrac{1}{2p})\\,\\Gamma(1 + \\tfrac{1}{p})}
        {\\Gamma(1 + \\tfrac{1}{p} + \\tfrac{1}{2p})} .
```

Both give ``4\\pi a^3/3`` at ``p = 1``, and at ``p = 1/2`` they give ``4a^3/3``
(the octahedron) and ``2\\pi a^2c/3`` (the double cone). This is the oracle a
meshed volume is compared against — the mesh is what is approximate here, not
this.

Uses ``\\Gamma``, so it needs a floating-point or `ForwardDiff.Dual` element
type; a symbolic one has no `loggamma`.
"""
shape_volume(s::Supersphere) = 8 * s.a^3 * _dirichlet(3, shape_exponent(s))

function shape_volume(s::Superspheroid)
    m = shape_exponent(s)
    # 2π a² c · B(1/m, 1 + 2/m) / m, through loggamma.
    lb = loggamma(inv(m)) + loggamma(1 + 2 / m) - loggamma(1 + 2 / m + inv(m))
    return 2 * π * s.a^2 * s.c * exp(lb) / m
end

"""
    projected_area(s::Supersphere) -> S

Exact area of the shadow on a coordinate plane, that is, the area enclosed by
the superellipse ``|x/a|^{2p} + |y/a|^{2p} = 1``:

```math
S = 4a^2\\,\\frac{\\Gamma(1 + \\tfrac{1}{2p})^2}{\\Gamma(1 + \\tfrac1p)} .
```

``p = 1`` gives ``\\pi a^2``; ``p = 1/2`` gives ``2a^2``, the square of diagonal
``2a``.
"""
projected_area(s::Supersphere) = 4 * s.a^2 * _dirichlet(2, shape_exponent(s))

"""
    equivalent_sphere_radius(s)

Radius of the ball of the same volume, ``(3V/4\\pi)^{1/3}``.

Two uses: normalizing contribution tensors the way the literature does, and
setting a mesh size that means the same thing across ``p`` — an element size
tied to ``a`` would be far coarser, relative to the body, at ``p = 0.3`` than at
``p = 1``.
"""
equivalent_sphere_radius(s::AbstractSuperShape) = (3 * shape_volume(s) / (4 * π))^(1 // 3)
