# =============================================================================
#  geometry.jl — the confocal layered ellipsoid: type, constructors, accessors.
#
#  The boundaries are the ellipsoids of semi-axes √(aᵢ² + λₖ), i = 1, 2, 3, of
#  one confocal family: λ₁ = 0 for the core, and the shifts λₖ increase outward.
# =============================================================================

"""
    LayeredEllipsoid(semi_axes, conductivities; basis)

An inclusion of `N` isotropic layers bounded by confocal ellipsoids, for the
conduction problem. `semi_axes[k]` is the triplet of semi-axes of the outer
boundary of the layer `k`, along the three vectors of `basis` (the canonical
basis by default); `conductivities[k]` is the conductivity of the layer `k`, a
`TensISO{2,3}`. The first layer is the core.

The boundaries must be confocal: the squares of the semi-axes of two of them
differ by the same amount along the three axes, and increase outward. This is
checked on numbers and taken as given on symbols.
[`layered_ellipsoid_from_fractions`](@ref) builds the family from the core and
the volume fractions.

The interfaces are perfect, and the surrounding medium is isotropic. A sphere
or a spheroid is the particular case of equal semi-axes, also served, with
imperfect interfaces, by [`LayeredSphere`](@ref MeanFieldHomogenization.LayeredSpheres.LayeredSphere)
and [`LayeredSpheroid`](@ref MeanFieldHomogenization.LayeredSpheroids.LayeredSpheroid).
"""
struct LayeredEllipsoid{T <: Number, N, Cs, B <: TensND.AbstractBasis} <: MFH_Core.AbstractInclusion{T}
    semi_axes::NTuple{N, NTuple{3, T}}      # outer boundary of each layer, ascending
    moduli::Cs                              # NTuple{N, TensISO{2,3}}
    basis::B                                # the common axes of the ellipsoids
    # Only this constructor, with every parameter given: the default one would
    # leave `T` unbound for a tuple of no layer.
    LayeredEllipsoid{T, N, Cs, B}(semi_axes, moduli, basis) where {T <: Number, N, Cs, B <: TensND.AbstractBasis} =
        new{T, N, Cs, B}(semi_axes, moduli, basis)
end

# The relative tolerance on the confocality of two boundaries, on the squares
# of the semi-axes.
const _CONFOCAL_RTOL = 1.0e-10

function LayeredEllipsoid(
        semi_axes::Tuple{NTuple{3, Number}, Vararg{NTuple{3, Number}, M}}, moduli::Cs;
        basis::TensND.AbstractBasis = TensND.CanonicalBasis{3, Float64}(),
    ) where {M, Cs <: Tuple{TensND.TensISO{2, 3}, Vararg{TensND.TensISO{2, 3}, M}}}
    T = TensND.float_like(mapreduce(a -> promote_type(map(typeof, a)...), promote_type, semi_axes))
    axes = map(a -> map(T, a), semi_axes)
    _check_confocal(Val(is_hard_numeric(T)), axes)
    return LayeredEllipsoid{T, M + 1, Cs, typeof(basis)}(axes, moduli, basis)
end
LayeredEllipsoid(semi_axes::Tuple, moduli::Tuple; kw...) = throw(
    ArgumentError(
        "LayeredEllipsoid: give one triplet of semi-axes and one isotropic conductivity (`TensISO{3}(k)`) per layer; " *
            "got $(length(semi_axes)) boundaries and $(length(moduli)) conductivities of types $(map(typeof, moduli))"
    )
)

# Symbols do not compare: the family is taken as given.
_check_confocal(::Val{false}, axes) = nothing
function _check_confocal(::Val{true}, axes)
    all(a -> all(x -> isfinite(x) && x > 0, a), axes) ||
        throw(ArgumentError("LayeredEllipsoid: the semi-axes must be finite and positive, got $axes"))
    core = map(abs2, first(axes))
    previous = zero(first(core))
    for (k, a) in enumerate(axes)
        shifts = map(abs2, a) .- core
        scale = maximum(map(abs2, a))
        maximum(shifts) - minimum(shifts) ≤ _CONFOCAL_RTOL * scale || throw(
            ArgumentError(
                "LayeredEllipsoid: the boundary $k, of semi-axes $a, is not confocal with the core $(first(axes)): " *
                    "the squares of its semi-axes exceed those of the core by $shifts, which should be one number. " *
                    "`layered_ellipsoid_from_fractions` builds a confocal family from the volume fractions"
            )
        )
        k == 1 || first(shifts) > previous ||
            throw(ArgumentError("LayeredEllipsoid: the boundaries must grow from the core outward; the boundary $k does not"))
        previous = first(shifts)
    end
    return nothing
end

"""
    layered_ellipsoid_from_fractions(core_axes, fractions, conductivities; basis) -> LayeredEllipsoid

The confocal layered ellipsoid whose core has the semi-axes `core_axes` and
whose layers have the volume fractions `fractions` in the inclusion (their sum
is 1; `fractions[1]` is that of the core).

The boundary of the layers 1 to `k` is the confocal ellipsoid of semi-axes
``\\sqrt{a_i^2+\\lambda_k}`` whose volume is the sum of their fractions:
``\\lambda_k`` is the root of
``\\prod_i(a_i^2+\\lambda)=\\bigl(\\prod_ia_i\\bigr)^2\\,(\\omega_k/\\omega_1)^2``,
found by Newton's method. Numbers and duals; symbols are refused, the
iteration compares its steps.
"""
function layered_ellipsoid_from_fractions(
        core_axes::NTuple{3, Number}, fractions::NTuple{N, Number}, moduli::NTuple{N, TensND.TensISO{2, 3}};
        basis::TensND.AbstractBasis = TensND.CanonicalBasis{3, Float64}(),
    ) where {N}
    T = TensND.float_like(promote_type(map(typeof, core_axes)..., map(typeof, fractions)...))
    is_hard_numeric(T) || throw(
        ArgumentError(
            "layered_ellipsoid_from_fractions solves for the confocal boundaries by an iteration, which symbols do not " *
                "allow; give the semi-axes of each boundary to `LayeredEllipsoid`"
        )
    )
    all(f -> f > 0, fractions) && isapprox(sum(fractions), 1; rtol = 1.0e-10) ||
        throw(ArgumentError("layered_ellipsoid_from_fractions: the volume fractions must be positive and sum to 1, got $fractions"))
    squares = map(a -> abs2(T(a)), core_axes)
    ω = cumsum(collect(map(T, fractions)))
    axes = ntuple(k -> map(s -> sqrt(s + _confocal_shift(squares, abs2(ω[k] / ω[1]))), squares), Val(N))
    return LayeredEllipsoid(axes, moduli; basis)
end

# The shift λ ≥ 0 such that Π(sᵢ + λ) = ratio · Π sᵢ, for ratio ≥ 1. The
# product is increasing and convex in λ and at least (min sᵢ + λ)³: Newton's
# method started from the root of that bound decreases to the root.
function _confocal_shift(squares::NTuple{3, T}, ratio) where {T}
    target = ratio * prod(squares)
    product(λ) = prod(s -> s + λ, squares)
    slope(λ) = (s = squares; (s[1] + λ) * (s[2] + λ) + (s[2] + λ) * (s[3] + λ) + (s[1] + λ) * (s[3] + λ))
    λ = cbrt(target) - minimum(squares)
    for _ in 1:100
        step = (product(λ) - target) / slope(λ)
        λ -= step
        abs(step) ≤ 4 * eps(T) * maximum(squares) && break
    end
    # One more step carries the derivatives of a dual through the converged root.
    return λ - (product(λ) - target) / slope(λ)
end

# ── Accessors ────────────────────────────────────────────────────────────────

layer_count(::LayeredEllipsoid{T, N}) where {T, N} = N
layer_modulus(e::LayeredEllipsoid, k::Int) = e.moduli[k]
layer_semiaxes(e::LayeredEllipsoid, k::Int) = e.semi_axes[k]
outer_semiaxes(e::LayeredEllipsoid) = last(e.semi_axes)

# Volume of the boundary of the layers 1 to k over that of the whole inclusion.
_cumulated_fractions(e::LayeredEllipsoid) = map(a -> prod(a) / prod(outer_semiaxes(e)), e.semi_axes)

"""
    layer_volume_fraction(e::LayeredEllipsoid, k) -> Real

Volume fraction of the layer `k` in the inclusion.
"""
function layer_volume_fraction(e::LayeredEllipsoid, k::Int)
    ω = _cumulated_fractions(e)
    return k == 1 ? ω[1] : ω[k] - ω[k - 1]
end

# The ellipsoid of the layers 1 to k, in the axes of the inclusion.
_boundary(e::LayeredEllipsoid, k::Int) = Ellipsoid(e.semi_axes[k]..., e.basis)

# ── AbstractInclusion interface ──────────────────────────────────────────────

MFH_Core.dimension(::LayeredEllipsoid) = 3
MFH_Core.inclusion_basis(e::LayeredEllipsoid) = e.basis
is_homogeneous_inclusion(::LayeredEllipsoid) = false

struct EllipsoidalLayered end
MFH_Core.shape_trait(::LayeredEllipsoid) = EllipsoidalLayered

"""
    shape_tensor(e::LayeredEllipsoid) -> AbstractTens{2,3}

The shape tensor of the outer boundary, that of the ellipsoid of the same
semi-axes and axes.
"""
MFH_Core.shape_tensor(e::LayeredEllipsoid) = MFH_Core.shape_tensor(_boundary(e, layer_count(e)))

Base.:(==)(a::T, b::T) where {T <: LayeredEllipsoid} =
    a.semi_axes == b.semi_axes && a.moduli == b.moduli && a.basis == b.basis
Base.hash(e::LayeredEllipsoid, h::UInt) = hash(e.basis, hash(e.moduli, hash(e.semi_axes, hash(LayeredEllipsoid, h))))
Base.show(io::IO, e::LayeredEllipsoid{T, N}) where {T, N} =
    print(io, "LayeredEllipsoid{", T, "} (", N, " layer(s), outer semi-axes ", outer_semiaxes(e), ")")
