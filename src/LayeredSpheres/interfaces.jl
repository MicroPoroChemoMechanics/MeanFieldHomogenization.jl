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
#     `[u_n] = t_n / kn = sn · t_n`, `[u_t] = t_t / kt = st · t_t`.
#   - `MembraneInterface(κs, μs)` — traction jump (dual, surface
#     elasticity): surface stiffness introduces a jump in the normal
#     component of the traction proportional to the surface strain.
#
#   Conductivity (thermal / electric / Darcy)
#   -----------------------------------------
#   - `KapitzaInterface(ρ)` — temperature jump (primal, interfacial
#     thermal resistance): `[T] = -ρ q_n`, the temperature dropping in the
#     direction of the flux.
#   - `SurfaceConductiveInterface(ks)` — flux jump (dual, highly
#     conductive 2D layer): `[q_n] = divₛ(ks ∇ₛ T)`.
#
#   `q = -k ∇T` is the flux, `q_n = q·n` its component along the normal n
#   pointing from the inner side to the outer one, and `[x] = x⁺ - x⁻`.
#
#  `PerfectInterface` is the trivial limit of any of them (k→0 for the
#  primal types, ks→0 / κs=μs=0 for the dual types).
# =============================================================================

"""
    AbstractInterface{T}

Root supertype for interface conditions in a `LayeredSphere`.  Concrete
subtypes determine the jump matrix applied to the state vector
``(u_r, \\sigma_{rr})`` (bulk), ``(U, W, \\sigma_{rr}, \\sigma_{r\\theta})`` (shear), or ``(T, q_n)``
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

```math
[\\![\\boldsymbol{\\sigma}\\cdot\\underline{n}]\\!] = \\underline{0},
\\qquad
[\\![u_n]\\!] = \\sigma_{rr}/k_n = s_n\\,\\sigma_{rr},
\\qquad
[\\![u_t]\\!] = \\sigma_{r\\theta}/k_t = s_t\\,\\sigma_{r\\theta}.
```

`kn`, `kt` are the normal and tangential **stiffnesses** ``k_n``, ``k_t`` — traction per unit
opening, the usual meaning of those symbols — and `sn`, `st` the matching
**compliances** ``s_n = 1/k_n``, ``s_t = 1/k_t``. Both spellings read and write the same
interface:

```julia
itf = SpringInterface(50.0, 20.0)        # stiffnesses
itf.kn, itf.kt                           # (50.0, 20.0)
itf.sn, itf.st                           # (0.02, 0.05)
SpringInterface(; sn = 0.02, st = 0.05)  # == itf
```

Limits: ``k_n, k_t \\to \\infty`` (equivalently ``s_n = s_t = 0``) recovers
[`PerfectInterface`](@ref); ``k_n = k_t = 0`` is a free surface, the layer
boundary fully decoupled. The one-argument form `SpringInterface(kn)` is a
normal spring with the tangential direction **bonded** (``s_t = 0``).

!!! note "Compliances are what is stored"
    The perfect interface is ``s_n = s_t = 0``, an exact zero, whereas in
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

Normal and tangential stiffnesses, ``(1/s_n, 1/s_t)``. A bonded direction has a
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
surface moduli ``\\kappa^{\\mathrm s} = \\lambda^{\\mathrm s} + \\mu^{\\mathrm s}`` (surface dilatation, matching Echoes' `ks`)
and surface shear ``\\mu^{\\mathrm s}``.  Displacement is continuous across the interface
and the surface strain generates a traction jump (``[\\![\\boldsymbol{\\sigma}\\cdot\\underline{n}]\\!] = -\\mathrm{div}_{\\mathrm s}\\,\\boldsymbol{\\sigma}^{\\mathrm s}``).  On a
spherical interface of radius ``r``, the bulk (``Y_0``) mode jump is

```math
[\\![\\sigma_{rr}]\\!] = \\frac{4\\kappa^{\\mathrm s}}{r^2}\\,u_r,
```

and the shear (``Y_2``-harmonic) mode jump, with ``u_r = U\\,P_2``,
``u_\\theta = W\\,\\mathrm{d}P_2/\\mathrm{d}\\theta``, is

```math
\\begin{aligned}
[\\![\\sigma_{rr}]\\!] &= (4\\kappa^{\\mathrm s} U - 12\\kappa^{\\mathrm s} W)/r^2,\\\\
[\\![\\sigma_{r\\theta}]\\!] &= (-2\\kappa^{\\mathrm s} U + (6\\kappa^{\\mathrm s} + 4\\mu^{\\mathrm s}) W)/r^2.
\\end{aligned}
```

The ``\\kappa^{\\mathrm s} = \\mu^{\\mathrm s} = 0`` limit recovers [`PerfectInterface`](@ref).  These jumps
reproduce Echoes' `DUALDISC` concentration tensors and effective moduli to
machine precision.
"""
struct MembraneInterface{T <: Number} <: AbstractInterface{T}
    κs::T
    μs::T
end

# The default inner constructor binds a single `T` across both surface moduli,
# so `MembraneInterface(κs::Dual, 0.5)` — differentiating with respect to the
# surface dilatation alone, the most natural sensitivity to ask of a membrane —
# was a `MethodError`. Promote instead, as `SpringInterface` already does.
MembraneInterface(κs::Number, μs::Number) =
    (T = promote_type(typeof(κs), typeof(μs)); MembraneInterface{T}(T(κs), T(μs)))

"""
    KapitzaInterface{T}(resistance::T)

Thermal imperfect interface with scalar thermal resistance ``\\rho`` (`resistance`):
the flux is continuous and the temperature drops across the interface in the
direction of the flux,

```math
[\\![q_n]\\!] = 0,
\\qquad
[\\![T]\\!] = T^+ - T^- = -\\rho\\,q_n,
```

with ``\\underline{q} = -k\\,\\nabla T`` the flux and ``q_n = \\underline{q}\\cdot\\underline{n}``, the normal pointing from the
inner side ``-`` to the outer side ``+``. Primal analog of [`SpringInterface`](@ref):
in the dictionary ``\\boldsymbol\\sigma \\equiv -\\underline{q}`` it reads ``[\\![T]\\!] = \\rho\\,\\sigma_n``, as
``[\\![u_n]\\!] = s_n\\,\\sigma_{nn}``.
"""
struct KapitzaInterface{T <: Number} <: AbstractInterface{T}
    resistance::T
end

"""
    SurfaceConductiveInterface{T}(conductance::T)

Highly-conductive 2D surface layer (dual analog of
[`MembraneInterface`](@ref)).  Introduces a flux jump driven by the
surface Laplacian of the temperature; for the spherical harmonic ``Y_n``
on a spherical interface of radius ``r`` and surface conductance ``k^{\\mathrm s}`` (`conductance`),

```math
[\\![q_n]\\!] = -n(n+1)\\,k^{\\mathrm s}\\,T/r^2.
```

``k^{\\mathrm s} = 0`` recovers [`PerfectInterface`](@ref).
"""
struct SurfaceConductiveInterface{T <: Number} <: AbstractInterface{T}
    conductance::T
end

Base.eltype(::AbstractInterface{T}) where {T} = T
Base.eltype(::Type{<:AbstractInterface{T}}) where {T} = T

"""
    interfaces_eltype(interfaces) -> Type

Element type common to a tuple of interface conditions, `Union{}` for an empty
tuple (the neutral element of `promote_type`).

Every solver in this package sizes its state buffers from a `promote_type` over
the geometry, the layer moduli and the matrix moduli. The interface parameters
belong in that promotion too: the transfer matrices widen locally to
`promote_type(eltype(intf), …)`, so a `ForwardDiff.Dual` spring stiffness or
Kapitza resistance produces a widened state vector that a narrower buffer then
refuses to store. Feeding this helper into the outer promotion is what makes
sensitivities with respect to interface parameters work.
"""
@inline interfaces_eltype(interfaces::Tuple) =
    promote_type(map(eltype, interfaces)...)
@inline interfaces_eltype(::Tuple{}) = Union{}
