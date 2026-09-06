# =============================================================================
#  interfaces_laminate.jl — imperfect interfaces of a periodic laminate.
#
#  The five interface types of `LayeredSpheres` are reused UNCHANGED, and
#  three **anisotropic** ones are added here. A planar interface is the
#  curvature-free case of a spherical one, and the algebra collapses to two
#  additive terms — which is what makes the laminate the cleanest possible
#  test of the package's interface conventions.
#
#  ── Why anisotropic interfaces exist here and not on the sphere ───────────
#  On an `n`-layer isotropic sphere the interface must be isotropic: the
#  spherical-harmonic recurrence only closes if the jump conditions share the
#  symmetry of the geometry, which is why `SpringInterface` carries the two
#  scalars `(kn, kt)` and `MembraneInterface` the two surface moduli. A plane
#  imposes no such restriction — the interface has a well-defined normal and
#  an arbitrary in-plane texture — so a laminate accepts a **full tensor**:
#
#    * `AnisotropicSpringInterface(𝕂)`   — any symmetric 3×3 compliance;
#    * `AnisotropicMembraneInterface(ℂˢ)` — any in-plane surface stiffness
#      (6 independent 2-D coefficients);
#    * `AnisotropicSurfaceConductiveInterface(𝐤ˢ)` — any in-plane surface
#      conductivity.
#
#  The primal *transport* interface needs no anisotropic form: `[T] = ρ q_n`
#  relates two scalars, so `KapitzaInterface(ρ)` is already fully general.
#
#  ── Primal (spring, Kapitza): a field jump ────────────────────────────────
#  `[u] = 𝕂·(σ·n)`, the traction staying continuous. This is the limit of a
#  layer of vanishing thickness whose out-of-plane compliance is `Ŝ/h` and
#  whose in-plane stiffness is zero. In the laminate formula,
#
#      f·ℙ            → ℙ_int / L        (finite)
#      f·ℂ_IO ℂ_OO⁻¹  → 0                (no in-plane/out-of-plane coupling)
#      f·Schur_IP(ℂ)  → 0                (no in-plane stiffness)
#
#  so a primal interface adds to `⟨ℙ⟩` and to nothing else.
#
#  ── Dual (membrane, surface-conductive): a surface stiffness ──────────────
#  The interfaces being PLANAR, `divₛ σˢ = 0`: there is no traction jump at
#  all. The surface stress is driven by the in-plane strain, which is
#  continuous and equal to the macroscopic `E`, so it adds straight to the
#  macroscopic stress:  `ℂ_hom ← ℂ_hom + Σ_j ℂˢ_j / L`, in the in-plane block.
#
#  Both terms carry the weight `1/L` — an interface *density*. Doubling every
#  thickness at fixed fractions therefore halves the interface correction, and
#  `L → ∞` recovers the perfect interface. That size effect is the physical
#  content of storing thicknesses rather than fractions.
#
#  ── Frames ────────────────────────────────────────────────────────────────
#  A tensor-valued interface field may be given either as a plain matrix, read
#  as **components in the layer frame `(ℓ, m, n)`**, or as a `TensND` tensor
#  carrying its own basis, converted on use. That is why the assembly helpers
#  take the laminate basis.
# =============================================================================

# ── The anisotropic interface types ─────────────────────────────────────────

"""
    AnisotropicSpringInterface(𝕂)

Imperfect interface of spring type with a **full compliance tensor**:

```math
[\\![\\underline{u}]\\!] = \\boldsymbol{\\mathcal{K}}\\cdot
  (\\boldsymbol{\\sigma}\\cdot\\underline{n}),
```

the traction staying continuous. Generalizes [`SpringInterface`](@ref), whose
two **stiffnesses** describe the isotropic case
`𝕂 = n⊗n/kn + (δ − n⊗n)/kt`; here `𝕂` is any symmetric second-order
compliance, so the normal and the two tangential directions may each have
their own compliance and be coupled.

`𝕂` is either a 3×3 matrix — read as components in the **layer frame**
`(ℓ, m, n)`, the third axis being the normal — or a `TensND` second-order
tensor carrying its own basis, converted on use.

`𝕂 = 0` recovers [`PerfectInterface`](@ref).

```julia
# a stiffer normal spring than tangential, plus an in-plane texture
𝕂 = [3.0e-3 5.0e-4 0.0; 5.0e-4 8.0e-3 0.0; 0.0 0.0 1.0e-3]
add_layer!(lam, :A, Dict(:C => C_A); thickness = 0.3,
           interface = AnisotropicSpringInterface(𝕂))
```
"""
struct AnisotropicSpringInterface{T <: Number, K} <: AbstractInterface{T}
    compliance::K
end

AnisotropicSpringInterface(K) =
    AnisotropicSpringInterface{eltype(K), typeof(K)}(K)

"""
    AnisotropicMembraneInterface(ℂˢ)

Surface-elastic (Gurtin-Murdoch) interface with a **full in-plane surface
stiffness**, generalizing [`MembraneInterface`](@ref) — whose two moduli
`(κs, μs)` describe the isotropic case. A 2-D elastic surface has six
independent coefficients, all of them available here.

`ℂˢ` is either a 3×3 matrix — the in-plane Kelvin-Mandel block on the basis
`(ℓ⊗ℓ, m⊗m, √2 ℓ⊗ˢm)`, so that its `[3,3]` entry is `2 C^s_{1212}` — or a
`TensND` fourth-order tensor, whose in-plane block in the layer frame is
taken.

The interfaces being planar there is no traction jump, so this adds directly
to the effective stiffness, in the in-plane block, with the weight `1/L`.

```julia
# an orthotropic membrane: stiffer along ℓ than along m
ℂˢ = [0.20 0.05 0.0; 0.05 0.09 0.0; 0.0 0.0 0.06]
add_layer!(lam, :A, Dict(:C => C_A); thickness = 0.3,
           interface = AnisotropicMembraneInterface(ℂˢ))
```
"""
struct AnisotropicMembraneInterface{T <: Number, S} <: AbstractInterface{T}
    stiffness::S
end

AnisotropicMembraneInterface(S) =
    AnisotropicMembraneInterface{eltype(S), typeof(S)}(S)

"""
    AnisotropicSurfaceConductiveInterface(𝐤ˢ)

Highly conductive 2-D surface layer with a **full in-plane surface
conductivity**, generalizing [`SurfaceConductiveInterface`](@ref) — whose
single scalar describes the isotropic case.

`𝐤ˢ` is either a 3×3 matrix in the layer frame or a `TensND` second-order
tensor; only its in-plane part is used, the surface flux being driven by the
in-plane gradient. Adds to the effective conductivity with the weight `1/L`.

There is deliberately **no** anisotropic counterpart of
[`KapitzaInterface`](@ref): the primal transport condition `[T] = ρ q_n`
relates two scalars, so the single resistance is already fully general.
"""
struct AnisotropicSurfaceConductiveInterface{T <: Number, K} <: AbstractInterface{T}
    conductance::K
end

AnisotropicSurfaceConductiveInterface(K) =
    AnisotropicSurfaceConductiveInterface{eltype(K), typeof(K)}(K)

# ── Reading a tensor-valued field in the layer frame ────────────────────────

# A plain matrix is already expressed in the layer frame; a TensND tensor
# carries its own basis and is converted.
@inline _in_layer_frame(M::AbstractMatrix, basis, ::Type{T}) where {T} =
    SMatrix{3, 3, T}(M)
@inline _in_layer_frame(t::TensND.AbstractTens{2, 3}, basis, ::Type{T}) where {T} =
    SMatrix{3, 3, T}(TensND.components(t, basis))

# Same, for the in-plane Kelvin-Mandel block of a surface stiffness.
@inline _ip_mandel_block(M::AbstractMatrix, basis, ::Type{T}) where {T} =
    SMatrix{3, 3, T}(M)
@inline _ip_mandel_block(t::TensND.AbstractTens{4, 3}, basis, ::Type{T}) where {T} =
    SMatrix{3, 3, T}(MFH_Core._ip_block(SMatrix{6, 6}(TensND.KM(t, basis))))

# Zero the out-of-plane row and column of a 3×3 written in the layer frame:
# the in-plane projection `p · X · p`.
@inline function _project_in_plane(X::AbstractMatrix{T}) where {T}
    z = zero(T)
    return @inbounds SMatrix{3, 3, T}(
        X[1, 1], X[2, 1], z, X[1, 2], X[2, 2], z, z, z, z
    )
end

# ── Elasticity, primal: out-of-plane compliance block ───────────────────────

"""
    _interface_P(itf, basis, ::Type{T}) -> SMatrix{6,6,T}

Contribution of one interface to `⟨ℙ⟩` (before the `1/L` weight), in the
layer frame and in Kelvin-Mandel form. Non-zero for the primal (field-jump)
types only.

For [`SpringInterface`](@ref)`(kn, kt)` — whose fields are *stiffnesses* —
the jump `[u] = 𝕂·(σ·n)` with the compliance
`𝕂 = n⊗n/kn + (δ − n⊗n)/kt` contributes the added strain
`(𝕂·(σ·n)) ⊗ˢ n`, i.e. the out-of-plane block
`diag(1/kn, 1/(2kt), 1/(2kt))` in Mandel slots. The halving of the tangential term is
the symmetrized product, and it is produced by
`Core.compliance_op_block` — the very helper that turns `𝐊⁻¹` into `ℙ`, so
the "interface = zero-thickness layer" statement is literal in the code.
[`AnisotropicSpringInterface`](@ref) goes through the same helper with a full
compliance tensor.
"""
_interface_P(::PerfectInterface, basis, ::Type{T}) where {T} = zero(SMatrix{6, 6, T})

function _interface_P(itf::SpringInterface, basis, ::Type{T}) where {T}
    sn, st = spring_compliances(itf)          # stored `kn`, `kt` are stiffnesses
    kn = convert(T, sn)
    kt = convert(T, st)
    z = zero(T)
    𝕂 = SMatrix{3, 3, T}(kt, z, z, z, kt, z, z, z, kn)   # (ℓ, m, n) frame
    return MFH_Core._op_embed(MFH_Core.compliance_op_block(𝕂))
end

function _interface_P(itf::AnisotropicSpringInterface, basis, ::Type{T}) where {T}
    𝕂 = _in_layer_frame(itf.compliance, basis, T)
    return MFH_Core._op_embed(MFH_Core.compliance_op_block(𝕂))
end

# Dual types contribute nothing to ⟨ℙ⟩ …
_interface_P(::MembraneInterface, basis, ::Type{T}) where {T} = zero(SMatrix{6, 6, T})
_interface_P(::AnisotropicMembraneInterface, basis, ::Type{T}) where {T} =
    zero(SMatrix{6, 6, T})
# … and the transport types are not elastic at all.
_interface_P(::KapitzaInterface, basis, ::Type{T}) where {T} = zero(SMatrix{6, 6, T})
_interface_P(::SurfaceConductiveInterface, basis, ::Type{T}) where {T} =
    zero(SMatrix{6, 6, T})
_interface_P(::AnisotropicSurfaceConductiveInterface, basis, ::Type{T}) where {T} =
    zero(SMatrix{6, 6, T})

# ── Elasticity, dual: in-plane surface stiffness block ──────────────────────

"""
    _interface_Cs(itf, basis, ::Type{T}) -> SMatrix{6,6,T}

Contribution of one interface to `ℂ_hom` (before the `1/L` weight), in the
layer frame and in Kelvin-Mandel form. Non-zero for the dual (surface
stiffness) types only.

For [`MembraneInterface`](@ref)`(κs, μs)` — Gurtin-Murdoch surface elasticity
with `κs = λs + μs` the surface dilatation modulus, matching the convention
of `LayeredSpheres` and of Echoes' `DUALDISC` — the 2-D surface law
`σˢ = λs tr(εˢ) p + 2μs εˢ` gives `C^s_1111 = κs + μs`, `C^s_1122 = κs − μs`,
`C^s_1212 = μs`, hence the in-plane Mandel block
`[κs+μs κs−μs 0; κs−μs κs+μs 0; 0 0 2μs]`.
[`AnisotropicMembraneInterface`](@ref) supplies that block directly.
"""
_interface_Cs(::PerfectInterface, basis, ::Type{T}) where {T} = zero(SMatrix{6, 6, T})
_interface_Cs(::SpringInterface, basis, ::Type{T}) where {T} = zero(SMatrix{6, 6, T})
_interface_Cs(::AnisotropicSpringInterface, basis, ::Type{T}) where {T} =
    zero(SMatrix{6, 6, T})

function _interface_Cs(itf::MembraneInterface, basis, ::Type{T}) where {T}
    κ = convert(T, itf.κs)
    μ = convert(T, itf.μs)
    z = zero(T)
    B = SMatrix{3, 3, T}(κ + μ, κ - μ, z, κ - μ, κ + μ, z, z, z, 2μ)
    return MFH_Core._ip_embed(B)
end

function _interface_Cs(itf::AnisotropicMembraneInterface, basis, ::Type{T}) where {T}
    return MFH_Core._ip_embed(_ip_mandel_block(itf.stiffness, basis, T))
end

_interface_Cs(::KapitzaInterface, basis, ::Type{T}) where {T} = zero(SMatrix{6, 6, T})
_interface_Cs(::SurfaceConductiveInterface, basis, ::Type{T}) where {T} =
    zero(SMatrix{6, 6, T})
_interface_Cs(::AnisotropicSurfaceConductiveInterface, basis, ::Type{T}) where {T} =
    zero(SMatrix{6, 6, T})

# ── Conduction, primal / dual ───────────────────────────────────────────────

"""
    _interface_P2(itf, basis, ::Type{T}) -> SMatrix{3,3,T}

Order-2 analogue of [`_interface_P`](@ref): the contribution of one interface
to the out-of-plane "compliance" average of a transport problem.
[`KapitzaInterface`](@ref)`(ρ)` imposes `[T] = ρ q_n`, hence `ρ · n⊗n`. Since
both sides of that condition are scalars, no anisotropic counterpart exists
or is needed.
"""
_interface_P2(::PerfectInterface, basis, ::Type{T}) where {T} = zero(SMatrix{3, 3, T})

function _interface_P2(itf::KapitzaInterface, basis, ::Type{T}) where {T}
    z = zero(T)
    ρ = convert(T, itf.resistance)
    return SMatrix{3, 3, T}(z, z, z, z, z, z, z, z, ρ)
end

_interface_P2(::SurfaceConductiveInterface, basis, ::Type{T}) where {T} =
    zero(SMatrix{3, 3, T})
_interface_P2(::AnisotropicSurfaceConductiveInterface, basis, ::Type{T}) where {T} =
    zero(SMatrix{3, 3, T})
_interface_P2(::SpringInterface, basis, ::Type{T}) where {T} = zero(SMatrix{3, 3, T})
_interface_P2(::AnisotropicSpringInterface, basis, ::Type{T}) where {T} =
    zero(SMatrix{3, 3, T})
_interface_P2(::MembraneInterface, basis, ::Type{T}) where {T} = zero(SMatrix{3, 3, T})
_interface_P2(::AnisotropicMembraneInterface, basis, ::Type{T}) where {T} =
    zero(SMatrix{3, 3, T})

"""
    _interface_Ks(itf, basis, ::Type{T}) -> SMatrix{3,3,T}

Order-2 analogue of [`_interface_Cs`](@ref): a highly conductive 2-D surface
layer carries a surface flux driven by the in-plane gradient, adding its
conductivity to the effective one. Isotropic for
[`SurfaceConductiveInterface`](@ref)`(ks)` (`ks (δ − n⊗n)`), arbitrary for
[`AnisotropicSurfaceConductiveInterface`](@ref).
"""
_interface_Ks(::PerfectInterface, basis, ::Type{T}) where {T} = zero(SMatrix{3, 3, T})

function _interface_Ks(itf::SurfaceConductiveInterface, basis, ::Type{T}) where {T}
    z = zero(T)
    ks = convert(T, itf.conductance)
    return SMatrix{3, 3, T}(ks, z, z, z, ks, z, z, z, z)
end

function _interface_Ks(
        itf::AnisotropicSurfaceConductiveInterface, basis, ::Type{T}
    ) where {T}
    return _project_in_plane(_in_layer_frame(itf.conductance, basis, T))
end

_interface_Ks(::KapitzaInterface, basis, ::Type{T}) where {T} = zero(SMatrix{3, 3, T})
_interface_Ks(::SpringInterface, basis, ::Type{T}) where {T} = zero(SMatrix{3, 3, T})
_interface_Ks(::AnisotropicSpringInterface, basis, ::Type{T}) where {T} =
    zero(SMatrix{3, 3, T})
_interface_Ks(::MembraneInterface, basis, ::Type{T}) where {T} = zero(SMatrix{3, 3, T})
_interface_Ks(::AnisotropicMembraneInterface, basis, ::Type{T}) where {T} =
    zero(SMatrix{3, 3, T})

# ── Assembly over the cell ──────────────────────────────────────────────────

"""
    _interface_terms(lam, ::Type{T}, ::Val{4}) -> (P_int, C_surf)
    _interface_terms(lam, ::Type{T}, ::Val{2}) -> (P_int, K_surf)

Sum the interface contributions of a laminate, each weighted by `1/L`.
Returns a pair `(primal, dual)` ready for
`Core.laminate_stiffness` / `Core.laminate_conductivity`.

Short-circuits to a pair of zeros when every interface is perfect, so the
common case pays nothing and is **bit-identical** to the no-interface path.
"""
function _interface_terms(lam::Laminate, ::Type{T}, ::Val{4}) where {T}
    all(itf -> itf isa PerfectInterface, lam.interfaces) &&
        return (zero(SMatrix{6, 6, T}), zero(SMatrix{6, 6, T}))
    # The interfaces carry their own element type: differentiating with respect
    # to an interface compliance makes `kn` a `Dual` while the moduli stay
    # `Float64`, so `T` alone would truncate the perturbation.
    Tp = _interface_eltype(lam, T)
    Z = zero(SMatrix{6, 6, Tp})
    L = laminate_period(lam)
    b = lam.basis
    P = Z
    Cs = Z
    for itf in lam.interfaces
        P = P + _interface_P(itf, b, Tp)
        Cs = Cs + _interface_Cs(itf, b, Tp)
    end
    return (P / L, Cs / L)
end

function _interface_terms(lam::Laminate, ::Type{T}, ::Val{2}) where {T}
    all(itf -> itf isa PerfectInterface, lam.interfaces) &&
        return (zero(SMatrix{3, 3, T}), zero(SMatrix{3, 3, T}))
    Tp = _interface_eltype(lam, T)
    Z = zero(SMatrix{3, 3, Tp})
    L = laminate_period(lam)
    b = lam.basis
    P = Z
    Ks = Z
    for itf in lam.interfaces
        P = P + _interface_P2(itf, b, Tp)
        Ks = Ks + _interface_Ks(itf, b, Tp)
    end
    return (P / L, Ks / L)
end

"""
    _interface_eltype(lam, ::Type{T}) -> Type

Promotion of `T` (the element type of the layers and thicknesses) with the
element type of every interface, so that a `ForwardDiff.Dual` reaching an
interface compliance alone still propagates.
"""
function _interface_eltype(lam::Laminate, ::Type{T}) where {T}
    Tp = T
    for itf in lam.interfaces
        Tp = promote_type(Tp, eltype(itf))
    end
    return Tp
end
