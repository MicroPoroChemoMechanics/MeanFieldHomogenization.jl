# =============================================================================
#  laminate_algebra.jl — the laminate kernels, in the Kelvin-Mandel
#  representation, on TensND's block algebra about the layer normal.
#
#  The partition of the Kelvin-Mandel slots into in-plane (`MANDEL_IP`) and
#  out-of-plane (`MANDEL_OP`) symmetric tensors, the blocks and their embeddings,
#  the Schur inverse, the plane pseudo-inverses, the acoustic tensor and the
#  spring-interface compliance block are tensor algebra and live in TensND
#  (`src/km_blocks.jl`); they are re-exported from `Core`. What stays here is the
#  laminate physics: the flat-inclusion Hill tensor under its own name, the
#  effective stiffness and conductivity of a periodic laminate, and the
#  concentration tensors of its layers.
#
#  Every matrix is an `SMatrix`, never an `MMatrix`: `MMatrix{6,6,T}(undef)` is
#  not constructible for a non-`isbits` `T` such as `SymPyPythonCall.Sym`.
# =============================================================================

"""
    flat_hill(C6) -> SMatrix{6,6}

Hill polarization tensor
``\\mathbb{P} = \\underline{n}\\stackrel{s}{\\otimes}\\boldsymbol{K}^{-1}\\stackrel{s}{\\otimes}\\underline{n}``
of a flat (layer) inclusion of Kelvin-Mandel stiffness `C6`, with
``\\boldsymbol{K} = \\underline{n}\\cdot\\mathbb{C}\\cdot\\underline{n}`` the acoustic tensor,
expressed in the layer frame — the limit of the
Hill tensor of a flat ellipsoid as its smallest aspect ratio tends to zero.

``\\mathbb{P}`` and TensND's `plane_pinv` of the stiffness are the **same object**: the
out-of-plane block of ``\\mathbb{P}`` is the matrix inverse of the out-of-plane block of
``\\mathbb{C}``, and every other block vanishes. Hence also
``\\mathbb{P}:\\mathbb{C}:\\mathbb{P} = \\mathbb{P}`` and
``\\mathbb{Q}:\\mathbb{P} = 0`` with ``\\mathbb{Q} = \\mathbb{C} - \\mathbb{C}:\\mathbb{P}:\\mathbb{C}`` the second Hill tensor.
"""
@inline flat_hill(C6::AbstractMatrix) = plane_pinv(C6)

# ── Frame handling ──────────────────────────────────────────────────────────

# The Kelvin-Mandel (Bond) rotation of a frame is TensND's `mandel_rotation(R)`.

# ── The laminate kernels ────────────────────────────────────────────────────

"""
    laminate_stiffness(C6s, f, P_int, C_surf; opinv = plane_pinv,
                       opinv_avg = plane_pinv) -> SMatrix{6,6}

Effective Kelvin-Mandel stiffness of a periodic laminate, in the layer frame:

```math
\\mathbb{C}^{\\mathrm{hom}} = \\langle\\mathbb{Q}\\rangle
 + \\langle\\mathbb{C}:\\mathbb{P}\\rangle : \\langle\\mathbb{P}\\rangle^{\\dagger}
   : \\langle\\mathbb{P}:\\mathbb{C}\\rangle
 + \\sum_j \\mathbb{C}^{\\mathrm s}_j / L ,
```

with ``\\langle\\cdot\\rangle = \\sum_i f_i\\,(\\cdot)_i``, ``\\mathbb{P}_i`` the flat-inclusion
Hill tensor of layer ``i`` (`flat_hill`) and
``\\mathbb{Q}_i = \\mathbb{C}_i - \\mathbb{C}_i:\\mathbb{P}_i:\\mathbb{C}_i`` (the in-plane Schur complement of ``\\mathbb{C}_i``).

Arguments:

- `C6s` — per-layer Kelvin-Mandel stiffnesses in the layer frame;
- `f` — per-layer volume fractions;
- `P_int` — ``\\sum_j \\mathbb{P}^{\\mathrm{int}}_j / L``, the primal (spring / Kapitza) interface term,
  which is out-of-plane-supported and adds to ``\\langle\\mathbb{P}\\rangle`` alone;
- `C_surf` — ``\\sum_j \\mathbb{C}^{\\mathrm s}_j / L``, the dual (membrane) surface term, which is
  in-plane-supported and adds to ``\\mathbb{C}^{\\mathrm{hom}}`` directly, the interfaces being
  planar (``\\mathrm{div}_s\\,\\boldsymbol{\\sigma}^{\\mathrm s} = \\underline{0}``, hence no traction jump).

`opinv` / `opinv_avg` are the out-of-plane inversion used for a layer
stiffness and for the average ``\\langle\\mathbb{P}\\rangle`` respectively. They are function arguments
so that the aging-viscoelastic laminate reuses this very kernel with the
Volterra block inversion substituted; Julia specializes on the function type,
so there is no abstraction cost and only one place where the physics lives.
"""
function laminate_stiffness(
        C6s, f, P_int, C_surf;
        opinv = plane_pinv, opinv_avg = plane_pinv
    )
    Z = zero(first(C6s))
    Pavg = Z
    CPavg = Z
    PCavg = Z
    Qavg = Z
    @inbounds for i in eachindex(C6s)
        C = C6s[i]
        P = opinv(C)                 # ℙ_i
        CP = C * P                   # in Kelvin-Mandel, ⊡ is the matrix product
        PC = P * C
        fi = f[i]
        Pavg = Pavg + fi * P
        CPavg = CPavg + fi * CP
        PCavg = PCavg + fi * PC
        Qavg = Qavg + fi * (C - CP * C)
    end
    Pavg = Pavg + P_int
    return Qavg + CPavg * opinv_avg(Pavg) * PCavg + C_surf
end

# ── Conduction (order 2) ────────────────────────────────────────────────────
#
# A second-order property needs no Mandel weights: in the frame (ℓ, m, n) the
# partition is simply `IP = (1, 2)` / `OP = (3)`, and the out-of-plane
# subspace is ONE-dimensional, so the pseudo-inverse degenerates to a scalar
# reciprocal.

"""
    laminate_conductivity(K3s, f, P_int, K_surf; opinv = plane_pinv2,
                          opinv_avg = plane_pinv2) -> SMatrix{3,3}

Effective conductivity (or diffusivity, permeability, …) of a periodic
laminate, in the layer frame — the order-2 transposition of
[`laminate_stiffness`](@ref), with the in-plane gradient continuous and the
normal flux continuous. `P_int` collects
``\\sum_j (\\rho_j/L)\\,\\underline{n}\\otimes\\underline{n}`` (Kapitza) and
`K_surf` collects ``\\sum_j (k^{\\mathrm s}_j/L)(\\boldsymbol{1} - \\underline{n}\\otimes\\underline{n})`` (highly conductive surface
layer).

Because the out-of-plane subspace is one-dimensional, the normal component
obeys the exact series law
``1/k^{\\mathrm{hom}}_{nn} = \\sum_i f_i/k_{i,nn} + \\sum_j \\rho_j/L``.
"""
function laminate_conductivity(
        K3s, f, P_int, K_surf;
        opinv = plane_pinv2, opinv_avg = plane_pinv2
    )
    Z = zero(first(K3s))
    Pavg = Z
    KPavg = Z
    PKavg = Z
    Qavg = Z
    @inbounds for i in eachindex(K3s)
        K = K3s[i]
        P = opinv(K)
        KP = K * P
        PK = P * K
        fi = f[i]
        Pavg = Pavg + fi * P
        KPavg = KPavg + fi * KP
        PKavg = PKavg + fi * PK
        Qavg = Qavg + fi * (K - KP * K)
    end
    Pavg = Pavg + P_int
    return Qavg + KPavg * opinv_avg(Pavg) * PKavg + K_surf
end

# ── Concentration of one layer, at either order ──────────────────────────────
#
# The four kernels serve both orders, told apart by the size of their
# Kelvin-Mandel matrices: 6×6 in elasticity, 3×3 in conduction, with the
# matching pseudo-inverse of the out-of-plane block and inverse of the effective
# tensor.

_plane_pinv_of(::StaticMatrix{6, 6}) = plane_pinv
_plane_pinv_of(::StaticMatrix{3, 3}) = plane_pinv2
_inv_km(M::StaticMatrix{6, 6}) = mandel_inv_schur(M)
_inv_km(M::StaticMatrix{3, 3}) = inv3(M)

"""
    laminate_gradient_concentration(P, Phom; opinv) -> SMatrix

Gradient concentration tensor of one layer of property `P` in a laminate of
effective property `Phom`, in Kelvin-Mandel form: the strain concentration
tensor ``\\mathbb{A}_i = \\mathbb{I} + \\mathbb{P}_i:(\\mathbb{C}^{\\mathrm{hom}} - \\mathbb{C}_i)``
for 6×6 stiffnesses, the gradient concentration tensor
``\\boldsymbol{A}_i = \\boldsymbol{1} + \\boldsymbol{P}_i\\cdot(\\boldsymbol{K}^{\\mathrm{hom}} - \\boldsymbol{K}_i)``
for 3×3 conductivities. Its in-plane block is the identity and its
in-plane/out-of-plane coupling block vanishes — the macroscopic in-plane
gradient is transmitted unchanged to every layer — and the fractions weight it
to the identity. Explicit name in elasticity:
[`laminate_strain_concentration`](@ref).
"""
@inline function laminate_gradient_concentration(
        P::StaticMatrix{N, N}, Phom::StaticMatrix{N, N}; opinv = _plane_pinv_of(P)
    ) where {N}
    return one(SMatrix{N, N, eltype(Phom)}) + opinv(P) * (Phom - P)
end

"""
    laminate_flux_concentration(P, Phom; opinv) -> SMatrix

Flux concentration tensor of one layer, the stress concentration tensor
``\\mathbb{B}_i = \\mathbb{C}_i:\\mathbb{A}_i:(\\mathbb{C}^{\\mathrm{hom}})^{-1}`` in elasticity, with
``\\sum_i f_i\\,\\mathbb{B}_i = \\mathbb{I}``. The inverse goes through a closed form,
so the result stays differentiable and symbolically evaluable. Explicit name in
elasticity: [`laminate_stress_concentration`](@ref).
"""
@inline function laminate_flux_concentration(
        P::StaticMatrix{N, N}, Phom::StaticMatrix{N, N}; opinv = _plane_pinv_of(P)
    ) where {N}
    return P * laminate_gradient_concentration(P, Phom; opinv = opinv) * _inv_km(Phom)
end

"""
    laminate_flux_gradient_concentration(P, Phom; opinv) -> SMatrix

Flux of one layer per unit macroscopic gradient,
``\\mathbb{A}_{\\sigma\\varepsilon,i} = \\mathbb{C}_i:\\mathbb{A}_i`` in elasticity. It weights to the
effective property, ``\\sum_i f_i\\,\\mathbb{A}_{\\sigma\\varepsilon,i} = \\mathbb{C}^{\\mathrm{hom}}``, and forms
no inverse of it, unlike [`laminate_flux_concentration`](@ref). Explicit name in
elasticity: [`laminate_stress_strain_concentration`](@ref).
"""
@inline function laminate_flux_gradient_concentration(
        P::StaticMatrix{N, N}, Phom::StaticMatrix{N, N}; opinv = _plane_pinv_of(P)
    ) where {N}
    return P * laminate_gradient_concentration(P, Phom; opinv = opinv)
end

"""
    laminate_gradient_flux_concentration(P, Phom; opinv) -> SMatrix

Gradient of one layer per unit macroscopic flux,
``\\mathbb{A}_{\\varepsilon\\sigma,i} = \\mathbb{A}_i:(\\mathbb{C}^{\\mathrm{hom}})^{-1}`` in elasticity, with
``\\sum_i f_i\\,\\mathbb{A}_{\\varepsilon\\sigma,i} = (\\mathbb{C}^{\\mathrm{hom}})^{-1}``. Explicit name in
elasticity: [`laminate_strain_stress_concentration`](@ref).
"""
@inline function laminate_gradient_flux_concentration(
        P::StaticMatrix{N, N}, Phom::StaticMatrix{N, N}; opinv = _plane_pinv_of(P)
    ) where {N}
    return laminate_gradient_concentration(P, Phom; opinv = opinv) * _inv_km(Phom)
end

# The names of elasticity, for 6×6 Kelvin-Mandel stiffnesses only.

"""
    laminate_strain_concentration(C6, Chom6; opinv = plane_pinv) -> SMatrix{6,6}

Name in elasticity of [`laminate_gradient_concentration`](@ref), for 6×6
Kelvin-Mandel stiffnesses.
"""
laminate_strain_concentration(C6::StaticMatrix{6, 6}, Chom6::StaticMatrix{6, 6}; kw...) =
    laminate_gradient_concentration(C6, Chom6; kw...)

"""
    laminate_stress_concentration(C6, Chom6; opinv = plane_pinv) -> SMatrix{6,6}

Name in elasticity of [`laminate_flux_concentration`](@ref), for 6×6
Kelvin-Mandel stiffnesses.
"""
laminate_stress_concentration(C6::StaticMatrix{6, 6}, Chom6::StaticMatrix{6, 6}; kw...) =
    laminate_flux_concentration(C6, Chom6; kw...)

"""
    laminate_stress_strain_concentration(C6, Chom6; opinv = plane_pinv) -> SMatrix{6,6}

Name in elasticity of [`laminate_flux_gradient_concentration`](@ref), for 6×6
Kelvin-Mandel stiffnesses.
"""
laminate_stress_strain_concentration(C6::StaticMatrix{6, 6}, Chom6::StaticMatrix{6, 6}; kw...) =
    laminate_flux_gradient_concentration(C6, Chom6; kw...)

"""
    laminate_strain_stress_concentration(C6, Chom6; opinv = plane_pinv) -> SMatrix{6,6}

Name in elasticity of [`laminate_gradient_flux_concentration`](@ref), for 6×6
Kelvin-Mandel stiffnesses.
"""
laminate_strain_stress_concentration(C6::StaticMatrix{6, 6}, Chom6::StaticMatrix{6, 6}; kw...) =
    laminate_gradient_flux_concentration(C6, Chom6; kw...)
