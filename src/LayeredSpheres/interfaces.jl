# =============================================================================
#  interfaces.jl — imperfect interface models for `LayeredSphere`.
#
#  Four physically-motivated interface types are provided, organized as
#  a "primal / dual" pair per physics:
#
#   Elasticity
#   ----------
#   - `SpringInterface(kn, kt)` — displacement jump (primal); `kn`/`kt` are
#     STIFFNESSES on the interface, stored internally as the compliances
#     `sn = 1/kn`, `st = 1/kt`:
#     `[u_n] = kn · t_n`, `[u_t] = kt · t_t`.
#   - `MembraneInterface(κs, μs)` — traction jump (dual, surface
#     elasticity): surface stiffness introduces a jump in the normal
#     component of the traction proportional to the surface strain.
#
#   Conductivity (thermal / electric / Darcy)
#   -----------------------------------------
#   - `KapitzaInterface(ρ)` — temperature jump (primal, interfacial
#     thermal resistance): `[T] = ρ · q_n`.
#   - `SurfaceConductiveInterface(ks)` — flux jump (dual, highly
#     conductive 2D layer): `[q_n] = -divₛ(ks ∇ₛ T)`.
#
#  `PerfectInterface` is the trivial limit of any of them (k→0 for the
#  primal types, ks→0 / κs=μs=0 for the dual types).
# =============================================================================

"""
    AbstractInterface{T}

Root supertype for interface conditions in a `LayeredSphere`.  Concrete
subtypes determine the jump matrix applied to the state vector
`(u_r, σ_rr)` (bulk), `(U, V, σ_rr, σ_rθ)` (shear), or `(T, q_n)`
(conductivity).
"""
abstract type AbstractInterface{T <: Number} end

"""
    PerfectInterface{T}()

Perfect (continuous) interface: all state-vector components are
continuous.
"""
struct PerfectInterface{T <: Number} <: AbstractInterface{T} end

PerfectInterface() = PerfectInterface{Float64}()

"""
    SpringInterface(kn, kt)
    SpringInterface(kn)
    SpringInterface(; kn, kt)      # stiffnesses
    SpringInterface(; sn, st)      # compliances

Imperfect interface of linear-spring type: the traction stays continuous
while the displacement jumps in proportion to it,

```
σ·n continuous,   [u_n] = σ_rr / kn = sn σ_rr,   [u_t] = σ_rθ / kt = st σ_rθ.
```

`kn`, `kt` are the normal and tangential **stiffnesses** — traction per unit
opening, the usual meaning of those symbols — and `sn = 1/kn`, `st = 1/kt`
the matching **compliances**. Both spellings read and write the same
interface:

```julia
itf = SpringInterface(50.0, 20.0)        # stiffnesses
itf.kn, itf.kt                           # (50.0, 20.0)
itf.sn, itf.st                           # (0.02, 0.05)
SpringInterface(; sn = 0.02, st = 0.05)  # == itf
```

Limits: `kn, kt → ∞` (equivalently `sn = st = 0`) recovers
[`PerfectInterface`](@ref); `kn = kt = 0` is a free surface, the layer
boundary fully decoupled. The one-argument form `SpringInterface(kn)` is a
normal spring with the tangential direction **bonded** (`st = 0`).

!!! note "Compliances are what is stored"
    The perfect interface is `sn = st = 0`, an exact zero, whereas in
    stiffnesses it is an infinity. Storing the compliances therefore keeps
    the near-perfect regime representable in `ForwardDiff.Dual` and in the
    symbolic types, where an `Inf` would poison the derivative. The
    stiffness spelling is a conversion on read and on write; the transfer
    matrices consume [`spring_compliances`](@ref).

Echoes' `PRIMALDISC` takes the same stiffness convention, so the two accept
the same numbers.

For a full compliance tensor with normal/tangential coupling, see
[`AnisotropicSpringInterface`](@ref MeanFieldHomogenization.Laminates.AnisotropicSpringInterface).
"""
struct SpringInterface{T <: Number} <: AbstractInterface{T}
    sn::T
    st::T
    # Inner constructor takes COMPLIANCES; every outer constructor below is
    # explicit about which of the two it speaks.
    SpringInterface{T}(sn::T, st::T) where {T <: Number} = new{T}(sn, st)
end

_spring_from_compliances(sn::Number, st::Number) =
    (T = promote_type(typeof(sn), typeof(st)); SpringInterface{T}(T(sn), T(st)))

SpringInterface(kn::Number, kt::Number) = _spring_from_compliances(inv(kn), inv(kt))

# Normal spring only, tangentially bonded: `st = 0` exactly (no `Inf` needed).
SpringInterface(kn::Number) = (sn = inv(kn); _spring_from_compliances(sn, zero(sn)))

function SpringInterface(; kn = nothing, kt = nothing, sn = nothing, st = nothing)
    (kn === nothing) || (sn === nothing) ||
        throw(ArgumentError("SpringInterface: give either `kn` or `sn`, not both"))
    (kt === nothing) || (st === nothing) ||
        throw(ArgumentError("SpringInterface: give either `kt` or `st`, not both"))
    kn === nothing && sn === nothing &&
        throw(ArgumentError("SpringInterface: one of `kn` or `sn` is required"))
    sn_ = sn === nothing ? inv(kn) : sn
    st_ = st !== nothing ? st :
        kt !== nothing ? inv(kt) : zero(sn_)   # default: tangentially bonded
    return _spring_from_compliances(sn_, st_)
end

# `kn` / `kt` are conversions, not stored fields — see the note above.
@inline function Base.getproperty(intf::SpringInterface, name::Symbol)
    name === :kn && return inv(getfield(intf, :sn))
    name === :kt && return inv(getfield(intf, :st))
    return getfield(intf, name)
end

Base.propertynames(::SpringInterface, private::Bool = false) = (:kn, :kt, :sn, :st)

"""
    spring_compliances(intf::SpringInterface) -> (sn, st)

Normal and tangential compliances — the quantities the transfer matrices
multiply the traction by. This is the stored pair; a perfect interface gives
exact zeros.
"""
@inline spring_compliances(intf::SpringInterface) =
    (getfield(intf, :sn), getfield(intf, :st))

"""
    spring_stiffnesses(intf::SpringInterface) -> (kn, kt)

Normal and tangential stiffnesses, `(1/sn, 1/st)`. A bonded direction has a
zero compliance and therefore an infinite stiffness.
"""
@inline spring_stiffnesses(intf::SpringInterface) = (intf.kn, intf.kt)

# Show both spellings: the default would print the stored compliances with no
# hint that the positional constructor speaks stiffnesses.
function Base.show(io::IO, intf::SpringInterface{T}) where {T}
    sn, st = spring_compliances(intf)
    return print(
        io, "SpringInterface{", T, "}(kn = ", intf.kn, ", kt = ", intf.kt,
        "  |  sn = ", sn, ", st = ", st, ")"
    )
end

"""
    MembraneInterface{T}(κs::T, μs::T)

Imperfect interface of surface-elastic (Gurtin–Murdoch "membrane") type —
the dual analog of [`SpringInterface`](@ref) and the elastic counterpart of
Echoes' `DUALDISC`.  The interface behaves as a 2D elastic shell with
surface moduli `κs = λs + μs` (surface dilatation, matching Echoes' `ks`)
and surface shear `μs`.  Displacement is continuous across the interface
and the surface strain generates a traction jump (`[σ·n] = −divₛσˢ`).  On a
spherical interface of radius `r`, the bulk (`Y₀`) mode jump is

```
[σ_rr] = (4 κs / r²) · u_r,
```

and the shear (`Y₂`-harmonic) mode jump, with `u_r = U P₂`,
`u_θ = W dP₂/dθ`, is

```
[σ_rr] = ( 4κs U − 12κs W) / r²,
[σ_rθ] = (−2κs U + (6κs + 4μs) W) / r².
```

The `κs = μs = 0` limit recovers [`PerfectInterface`](@ref).  These jumps
reproduce Echoes' `DUALDISC` concentration tensors and effective moduli to
machine precision.
"""
struct MembraneInterface{T <: Number} <: AbstractInterface{T}
    κs::T
    μs::T
end

"""
    KapitzaInterface{T}(resistance::T)

Thermal imperfect interface with scalar thermal resistance:
`[T] = resistance · q_n`, with `q_n` continuous.  Primal analog of
[`SpringInterface`](@ref).
"""
struct KapitzaInterface{T <: Number} <: AbstractInterface{T}
    resistance::T
end

"""
    SurfaceConductiveInterface{T}(conductance::T)

Highly-conductive 2D surface layer (dual analog of
[`MembraneInterface`](@ref)).  Introduces a flux jump driven by the
surface Laplacian of the temperature; for the spherical harmonic `Y_n`
on a spherical interface of radius `r`,

```
[q_n] = -n(n+1) · conductance · T / r².
```

`conductance = 0` recovers [`PerfectInterface`](@ref).
"""
struct SurfaceConductiveInterface{T <: Number} <: AbstractInterface{T}
    conductance::T
end

Base.eltype(::AbstractInterface{T}) where {T} = T
Base.eltype(::Type{<:AbstractInterface{T}}) where {T} = T
