# =============================================================================
#  geometry.jl — concrete type `LayeredSpheroid` (n-layer confocal
#  spheroidal composite inclusion, isotropic conduction only).
#
#  Confocal layers share a single focal distance `c`: for prolate
#  spheroids (revolution/axis semi-axis `a` > disk/transverse semi-axis
#  `b`), `c² = a² - b²`; for oblate (`a < b`), `c² = b² - a²`. Each
#  layer is located by its own confocal parameter `q_ℓ`, carried as
#  `q_ℓ = ρ_a,ℓ / c` (real, `> 1`) for prolate, or `q_ℓ = i·τ_ℓ`
#  (`τ_ℓ = ρ_a,ℓ / c real`, `c` itself carried as `-i·focal`) for
#  oblate — the complex substitution of Barthélémy & Bignonnet (2020,
#  eq:arabob) that lets every prolate formula in `conductivity.jl` and
#  `coupling.jl` carry over to oblate unchanged. The element type of
#  `q` (`Q = T` prolate, `Q = Complex{T}` oblate) is a type parameter,
#  so no branch is ever taken downstream on prolate/oblate: ordinary
#  dispatch on `Q<:Number` does the job (`sqrt`, `atanh`, … already
#  know how to handle `Complex`).
#
#  Convention: `q` is ascending in modulus, layer `k` occupying
#  `q_{k-1} < q ≤ q_k` (`q₀ = 1`, the degenerate point-focus limit,
#  implicit). The composite spheroid is embedded in an infinite matrix
#  outside `q_N` (not stored — passed to every API call as `K₀`, as for
#  `LayeredSphere`).
# =============================================================================

"""
    LayeredSpheroid{T, N, Q, Cs, Is} <: AbstractLayeredInclusion{3, T}

Isotropic `N`-layer confocal spheroidal composite inclusion (core +
concentric confocal shells), conduction only (no elastic counterpart
*yet* — see the `LayeredSpheroids` module docstring). Type
parameters:

- `T` — element type of the geometric scalars (radii, focal distance).
- `N` — number of layers (≥ 1).
- `Q` — element type of the confocal parameter `q`: `T` for prolate,
  `Complex{T}` for oblate (the complex substitution `c → -i c̄`, `q → i τ`).
- `Cs`, `Is` — concrete types of the `moduli` / `interfaces` tuples.

Use the keyword constructor
[`LayeredSpheroid`](@ref MeanFieldHomogenization.LayeredSpheroids.LayeredSpheroid)`(axis_radii,
disk_radii, moduli; interfaces, Nseries, axis)` for the common case, or
[`layered_spheroid_from_fractions`](@ref) to specify layers by volume
fraction.
"""
struct LayeredSpheroid{T <: Number, N, Q <: Number, Cs, Is} <:
    MFH_Core.AbstractLayeredInclusion{3, T}
    prolate::Bool
    focal::T                # real focal distance |c| > 0
    c::Q                    # focal distance: T (prolate) or -i·focal (oblate)
    q::NTuple{N, Q}         # per-layer confocal parameter, ascending |q|
    moduli::Cs              # NTuple{N, TensISO{2,3,T}}
    interfaces::Is          # NTuple{N, AbstractInterface}
    Nseries::Int            # 𝒩: series truncation (odd degrees 1,…,2𝒩-1)
    axis::NTuple{3, T}      # unit vector: symmetry (revolution) axis
end

# At least one layer is required — the runtime check this replaces said so. In
# the type, it also binds `T`, which the empty tuple left unbound.
"""
    LayeredSpheroid(axis_radii, disk_radii, moduli; interfaces, Nseries = 5,
                     axis = (0., 0., 1.), prolate = nothing)

Build an `N`-layer confocal spheroid from the per-layer **axis**
(revolution) and **disk** (transverse) semi-axes — ascending, confocal
(`axisᵢ² - diskᵢ²` constant across layers, up to the type's tolerance).
`axis_radii[ℓ] > disk_radii[ℓ]` ⟹ prolate; `<` ⟹ oblate.

`moduli`, `interfaces`, `Nseries`, `axis` follow [`LayeredSphere`](@ref)'s
conventions: `moduli::NTuple{N}` of per-layer `TensISO{2,3}` isotropic
conductivities; `interfaces::NTuple{N}` of [`AbstractInterface`](@ref)
(default: all [`PerfectInterface`](@ref)); `Nseries` the harmonic-series
truncation 𝒩 (only odd degrees `1,…,2𝒩-1` are kept, by symmetry);
`axis` the unit vector giving the spheroid's revolution axis in the
global frame.

The semi-axes may carry any element type: `Float64`, `BigFloat`,
`ForwardDiff.Dual` — mixed across the two tuples, so a single layer boundary
can be the differentiation variable — or a symbolic scalar.

`prolate` declares the family explicitly. Leave it at `nothing` for a numeric
element type and the sign of `axis² - disk²` decides. It is **required** for a
symbolic one: that sign is undecidable on a symbol, and SymPy answers `false`
to a comparison it cannot settle, so guessing would silently pick a family.
On a numeric type it is still checked: a hint that contradicts the semi-axes
throws, since every layer is verified to belong to the declared family.

!!! note "Symbolic support is prolate only"
    An oblate spheroid carries `q = iτ`, so its confocal parameter is a
    `Complex{T}` — and Julia's `Complex{T}` requires `T <: Real`, which
    `SymPy.Sym` is not. `Symbolics.Num` is `<: Real` and builds, but Symbolics
    has no `sym_lu` for a `Matrix{Complex{Num}}`. Prolate works with both.
"""
function LayeredSpheroid(
        axis_radii::Tuple{T, Vararg{T, M}}, disk_radii::Tuple{T, Vararg{T, M}},
        moduli::Cs;
        interfaces::Is = ntuple(_ -> PerfectInterface{T}(), Val(M + 1)),
        Nseries::Int = 5,
        axis::Tuple = (0.0, 0.0, 1.0),
        prolate::Union{Bool, Nothing} = nothing,
    ) where {M, T <: Number, Cs, Is}
    prolate_hint = prolate
    N = M + 1
    # Every check below is an ORDER comparison on user data. On a symbolic
    # element type those are undecidable — SymPy answers `false` to a comparison
    # it cannot settle, which would turn a silent wrong answer into a thrown
    # error or, worse, the reverse. Validate when the type admits comparison,
    # and let the caller carry the responsibility otherwise (`prolate` becomes a
    # required keyword in that case; see below).
    checkable = is_hard_numeric(T)
    if checkable
        for ℓ in 1:N
            axis_radii[ℓ] > 0 && disk_radii[ℓ] > 0 ||
                throw(ArgumentError("LayeredSpheroid semi-axes must be strictly positive"))
            axis_radii[ℓ] ≈ disk_radii[ℓ] &&
                throw(ArgumentError("LayeredSpheroid: layer $ℓ has aspect ratio 1 (use LayeredSphere)"))
        end
        for ℓ in 1:(N - 1)
            (axis_radii[ℓ] < axis_radii[ℓ + 1] && disk_radii[ℓ] < disk_radii[ℓ + 1]) ||
                throw(ArgumentError("LayeredSpheroid radii must be strictly ascending"))
        end
    end
    focal2s = ntuple(ℓ -> axis_radii[ℓ]^2 - disk_radii[ℓ]^2, Val(N))
    prolate = if prolate_hint !== nothing
        prolate_hint
    elseif checkable
        focal2s[1] > 0
    else
        throw(
            ArgumentError(
                "LayeredSpheroid: with a symbolic element type the sign of " *
                    "`axis² - disk²` cannot be decided; pass `prolate = true/false`"
            )
        )
    end
    if checkable
        any(prolate ? (f -> f ≤ 0) : (f -> f ≥ 0), focal2s) &&
            throw(ArgumentError("LayeredSpheroid: layers must share the same prolate/oblate type"))
    end
    focal_ref = sqrt(abs(focal2s[N]))
    if checkable
        for ℓ in 1:N
            isapprox(sqrt(abs(focal2s[ℓ])), focal_ref; rtol = sqrt(eps(float(real(T)))) * 10) ||
                throw(ArgumentError("LayeredSpheroid: layers are not confocal (focal distance $ℓ ≠ focal distance $N)"))
        end
    end
    focal = focal_ref
    Q = prolate ? T : Complex{T}
    c = prolate ? Q(focal) : Q(zero(T), -focal)
    q = ntuple(Val(N)) do ℓ
        τ = axis_radii[ℓ] / focal
        prolate ? Q(τ) : Q(zero(T), τ)
    end
    n = axis ./ sqrt(sum(a -> a^2, axis))
    return LayeredSpheroid{T, N, Q, Cs, Is}(
        prolate, focal, c, q, moduli, interfaces, Nseries, (T(n[1]), T(n[2]), T(n[3]))
    )
end

LayeredSpheroid(::Tuple{}, ::Tuple{}, moduli; kwargs...) = throw(
    ArgumentError("LayeredSpheroid requires at least one layer")
)

# Mixed-eltype semi-axes — promote across BOTH tuples, then delegate to the
# homogeneous method above. That method binds a single `T` across every
# semi-axis of both tuples, so `LayeredSpheroid((a₁::Dual, 2.0), (b₁::Dual, b₂), …)`
# — differentiating with respect to ONE layer boundary, the most natural
# sensitivity to ask of a coated inclusion — used to be a `MethodError`, while
# differentiating with respect to all of them at once worked. The counterpart of
# `LayeredSphere`'s promoting constructor.
function LayeredSpheroid(
        axis_radii::Tuple{Number, Vararg{Number, M}},
        disk_radii::Tuple{Number, Vararg{Number, M}},
        moduli; kwargs...
    ) where {M}
    T = promote_type(map(typeof, axis_radii)..., map(typeof, disk_radii)...)
    return LayeredSpheroid(map(T, axis_radii), map(T, disk_radii), moduli; kwargs...)
end

"""
    layered_spheroid_from_fractions(ω, outer_axis_radius, layer_fractions, moduli;
                                     interfaces, Nseries = 5, axis = (0., 0., 1.))

Convenience constructor specifying layers by volume fraction: build an
`N`-layer confocal spheroid of given
outer aspect ratio `ω` (`> 1` prolate, `< 1` oblate) and outer axis
semi-axis `outer_axis_radius`, with each layer occupying the prescribed
fraction of the total volume (`layer_fractions`, normalized to sum 1,
core first). Inner confocal parameters are found by bisection on the
volume relation `V(q) ∝ |q(q²-1)|` (eq:xLeg / spheroid_volume).
"""
function layered_spheroid_from_fractions(
        ω::T, outer_axis_radius::T, layer_fractions::NTuple{N, T}, moduli::Cs;
        interfaces::Is = ntuple(_ -> PerfectInterface{T}(), Val(N)),
        Nseries::Int = 5,
        axis::Tuple = (0.0, 0.0, 1.0),
    ) where {N, T <: Real, Cs, Is}
    ω ≈ 1 && throw(ArgumentError("layered_spheroid_from_fractions: ω ≈ 1 (use LayeredSphere)"))
    ftot = sum(layer_fractions)
    f = layer_fractions ./ ftot

    prolate = ω > 1
    # `axis_radius = ρ_a,N` is fixed by the caller; the disk radius
    # follows from `ω = ρ_a/ρ_t` (prolate: `ρ_t = ρ_a/ω`; oblate:
    # `ρ_t = ρ_a/ω` too, but `ρ_t > ρ_a` since `ω < 1` — same relation,
    # only the sign in `focal² = |ρ_a² - ρ_t²|` flips).
    focal = prolate ? outer_axis_radius * sqrt(ω^2 - 1) / ω :
        (outer_axis_radius / ω) * sqrt(1 - ω^2)
    qN = outer_axis_radius / focal   # τ_N for oblate (real, = axis_radius/focal in both cases)

    # Volume shape function (real cubic in the confocal parameter):
    # `φ(q) = q(q²-1)` for prolate (`q` real, `> 1`); for oblate the
    # confocal parameter is carried as `q = iτ` (`τ` real), and the
    # SAME cubic acquires a sign flip when rewritten in terms of the
    # real `τ`: `φ(τ) = τ(τ²+1)` (matches `|q(q²-1)| = τ(τ²+1)` for
    # `q = iτ`). Bisection on `x³ + C·x + D = 0`
    # (`C = -1` prolate, `C = +1` oblate) with a dependency-free
    # bisection since the root is bracketed and the cubic is monotone
    # there.
    Cbis = prolate ? -one(T) : one(T)
    φ(x) = x * (x^2 + Cbis)
    qs = Vector{T}(undef, N)
    qs[N] = qN
    for ℓ in N:-1:2
        target = φ(qs[ℓ]) - f[ℓ] * φ(qN)
        qs[ℓ - 1] = _bisect_cubic_root(-target, zero(T), qs[ℓ]; C = Cbis)
    end
    axis_radii = ntuple(ℓ -> focal * qs[ℓ], Val(N))
    disk_radii = ntuple(
        ℓ -> prolate ? focal * sqrt(qs[ℓ]^2 - 1) : focal * sqrt(qs[ℓ]^2 + 1), Val(N)
    )
    return LayeredSpheroid(axis_radii, disk_radii, moduli; interfaces, Nseries, axis)
end

# Mixed-eltype arguments — promote, then delegate. Same reason as the promoting
# `LayeredSpheroid` constructor above: the method it delegates to binds one `T`
# across `ω`, the outer semi-axis AND every volume fraction, so asking for the
# sensitivity to a single fraction, or to the aspect ratio alone, was a
# `MethodError`.
function layered_spheroid_from_fractions(
        ω::Number, outer_axis_radius::Number,
        layer_fractions::Tuple{Number, Vararg{Number, M}}, moduli; kwargs...
    ) where {M}
    T = promote_type(
        typeof(ω), typeof(outer_axis_radius), map(typeof, layer_fractions)...
    )
    return layered_spheroid_from_fractions(
        T(ω), T(outer_axis_radius), map(T, layer_fractions), moduli; kwargs...
    )
end

"""
    _bisect_cubic_root(D, lo, hi; C = -1, tol = ...) -> x

Real root of `x³ + C·x + D = 0` in `(lo, hi)`, by bisection. Used by
[`layered_spheroid_from_fractions`](@ref) to locate the confocal
parameter of an inner layer from the target cumulative volume
(monotone increasing on the physical bracket).
"""
function _bisect_cubic_root(D::T, lo::T, hi::T; C::T = -one(T), tol = eps(T)^(3 / 4)) where {T}
    f(x) = x^3 + C * x + D
    flo, fhi = f(lo), f(hi)
    flo * fhi > 0 && throw(ArgumentError("_bisect_cubic_root: root not bracketed in ($lo, $hi)"))
    a, b = lo, hi
    fa = flo
    while (b - a) > tol * max(one(T), abs(b))
        m = (a + b) / 2
        fm = f(m)
        if fa * fm ≤ 0
            b = m
        else
            a, fa = m, fm
        end
    end
    x = (a + b) / 2
    # Bisection decides every step by a SIGN TEST, and a sign test carries no
    # derivative: on a `ForwardDiff.Dual` the returned midpoint inherits the
    # partials of the bracket — zeros — so the sensitivity of a layer boundary
    # to the volume fractions or to the aspect ratio came out as a silent zero.
    # Two Newton steps in the full element type repair it. The value is already
    # converged, so they move it by roundoff; the partials become
    # `∂x/∂D = -1/(3x² + C)`, which is exactly what the implicit function theorem
    # gives for `x³ + Cx + D = 0`. No knowledge of the AD type is needed.
    for _ in 1:2
        x = x - (x^3 + C * x + D) / (3 * x^2 + C)
    end
    return x
end

# ── Accessors ────────────────────────────────────────────────────────────────

"""
    layer_count(spheroid) -> Int

Number of layers (excluding the matrix).
"""
layer_count(::LayeredSpheroid{T, N}) where {T, N} = N

"""
    layer_q(spheroid, k) -> Q

Confocal parameter `q_k` of the outer boundary of layer `k`.
"""
layer_q(s::LayeredSpheroid, k::Int) = s.q[k]

"""
    layer_modulus(spheroid, k)

Isotropic conductivity of layer `k`.
"""
layer_modulus(s::LayeredSpheroid, k::Int) = s.moduli[k]

"""
    layer_interface(spheroid, k) -> AbstractInterface

Interface condition at `layer_q(spheroid, k)` (between layer `k` and
layer `k+1` if `k < N`, or with the matrix if `k = N`).
"""
layer_interface(s::LayeredSpheroid, k::Int) = s.interfaces[k]

"""
    layer_semiaxes(spheroid, k) -> (axis, disk)

Real (axis, disk) semi-axes of the outer boundary of layer `k`.
"""
function layer_semiaxes(s::LayeredSpheroid{T}, k::Int) where {T}
    q = s.q[k]
    axis = real(s.c * q)
    disk = real(s.c * sqrt(q^2 - 1))
    return axis, disk
end

"""
    layer_volume_fraction(spheroid, k) -> T

Volume fraction of layer `k` inside the outer spheroid `q_N`, based on
the confocal volume shape function `φ(q) = |q(q²-1)|`
(`V(q) = (4π/3)·focal³·φ(q)`, eq:xLeg).
"""
function layer_volume_fraction(s::LayeredSpheroid{T, N}, k::Int) where {T, N}
    φ(q) = abs(q * (q^2 - 1))
    φk = φ(s.q[k])
    φkm1 = k == 1 ? zero(T) : φ(s.q[k - 1])
    return (φk - φkm1) / φ(s.q[N])
end

"""
    outer_semiaxes(spheroid) -> (axis, disk)

Real (axis, disk) semi-axes of the outermost boundary `q_N`.
"""
outer_semiaxes(s::LayeredSpheroid) = layer_semiaxes(s, layer_count(s))

# ── AbstractInclusion interface ──────────────────────────────────────────────

MFH_Core.dimension(::LayeredSpheroid) = 3

MFH_Core.inclusion_basis(::LayeredSpheroid{T}) where {T} =
    TensND.CanonicalBasis{3, MFH_Core._basis_eltype(T)}()

struct SpheroidalLayered end
MFH_Core.shape_trait(::LayeredSpheroid) = SpheroidalLayered

"""
    shape_tensor(spheroid::LayeredSpheroid) -> AbstractTens{2,3}

Symmetric 2nd-order shape tensor of the outer confocal boundary,
`diag(disk, disk, axis)` (transversely isotropic about `axis`).
"""
function MFH_Core.shape_tensor(s::LayeredSpheroid{T}) where {T}
    a, b = outer_semiaxes(s)
    return TensND.TensTI{2}(b, a, s.axis)
end

# ── Equality / hashing ──────────────────────────────────────────────────────

Base.:(==)(a::T, b::T) where {T <: LayeredSpheroid} =
    a.q == b.q && a.moduli == b.moduli && a.interfaces == b.interfaces && a.axis == b.axis

function Base.hash(s::LayeredSpheroid, h::UInt)
    h = hash(typeof(s), h)
    h = hash(s.q, h)
    h = hash(s.moduli, h)
    h = hash(s.interfaces, h)
    return hash(s.axis, h)
end

function Base.show(io::IO, s::LayeredSpheroid{T, N}) where {T, N}
    kind = s.prolate ? "prolate" : "oblate"
    return print(io, "LayeredSpheroid{", T, "} (", N, " layer(s), ", kind, ", focal = ", s.focal, ")")
end
