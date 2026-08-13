# =============================================================================
#  biot.jl — poroelastic upscaling of a saturated porous or fractured medium
#  whose solid phase is *homogeneous*.
#
#  Given the drained homogenized stiffness `C_hom` (whatever scheme produced
#  it) and the solid stiffness `C_s`, microporomechanics closes the poroelastic
#  constitutive law
#
#      Σ̇ = C_hom : Ė - ṗ B          φ̇ = B : Ė + ṗ / M
#
#  entirely: both `B` and `M` follow from `C_hom`, `C_s` and the Lagrangian
#  porosity `φ`, without any additional homogenization step.  These are the
#  relations of Dormieux-Kondo-Ulm (2006), quoted as eq. (2) of
#  Barthélémy & Daniel (ARMA 2011).
#
#  The whole file is pure tensor algebra on TensND objects — no Eshelby
#  problem, no scheme — so it is type-generic (Float64, BigFloat,
#  ForwardDiff.Dual, ComplexF64, SymPy.Sym) for free.
# =============================================================================

"""
    biot_tensor(C_hom, C_s) -> Tens{2,3}

**Biot tensor** ``\\boldsymbol{B}`` of a porous medium with a *homogeneous* solid
phase of stiffness `C_s`, from its drained homogenized stiffness `C_hom`:

```math
\\boldsymbol{B} = \\boldsymbol{1} : \\left(\\mathbb{I} - \\mathbb{S}_{\\rm s} :
\\mathbb{C}^{\\rm hom}\\right), \\qquad \\mathbb{S}_{\\rm s} = \\mathbb{C}_s^{-1}.
```

`\\boldsymbol{B}` is the tensor appearing in the poroelastic law
``\\dot{\\boldsymbol{\\Sigma}} = \\mathbb{C}^{\\rm hom} : \\dot{\\boldsymbol{E}} -
\\dot p\\,\\boldsymbol{B}``; it is generally **anisotropic** even when the solid is
isotropic, because the pore space is not.

Two limits are worth remembering as sanity checks: a homogeneous medium
(`C_hom == C_s`) gives ``\\boldsymbol{B} = \\boldsymbol{0}``, and a vanishing drained
stiffness gives ``\\boldsymbol{B} = \\boldsymbol{1}``. For an isotropic medium
the tensor collapses to ``b\\,\\boldsymbol{1}`` with the familiar
``b = 1 - k^{\\rm hom}/k_s``.

!!! note "Homogeneous solid phase only"
    These relations rest on the solid phase having *uniform* elastic
    properties. A medium with two distinct solid constituents needs the full
    Levin/eigenstrain route instead, and `C_s` is then not defined.

Pass the solid **stiffness**, not its compliance — the inverse is taken
internally. Computing several poroelastic parameters at once is cheaper through
[`poroelastic_parameters`](@ref), which inverts `C_s` only once.

See also [`inverse_biot_modulus`](@ref MeanFieldHomogenization.Poromechanics.inverse_biot_modulus), [`undrained_stiffness`](@ref),
[`terzaghi_stress`](@ref).
"""
biot_tensor(
    C_hom::TensND.AbstractTens{4, 3}, C_s::TensND.AbstractTens{4, 3}
) = _biot_tensor(C_hom, inv(C_s))

# Primitive taking the solid *compliance*, so that a caller holding `s_s`
# already (every function below, and the Gauss-point materials) never inverts
# `C_s` twice.
function _biot_tensor(
        C_hom::TensND.AbstractTens{4, 3}, s_s::TensND.AbstractTens{4, 3}
    )
    T = promote_type(eltype(C_hom), eltype(s_s))
    Id2 = TensND.tens_Id2(Val(3), Val(T))
    Id4 = TensND.tens_Id4(Val(3), Val(T))
    return Id2 ⊡ (Id4 - s_s ⊡ C_hom)
end

"""
    inverse_biot_modulus(C_s, B, φ) -> Number

Inverse **Biot modulus** ``1/M`` (also written ``1/N``) of a porous medium with
a homogeneous solid phase:

```math
\\frac{1}{M} = \\boldsymbol{1} : \\mathbb{S}_{\\rm s} : \\left(\\boldsymbol{B} -
\\varphi\\,\\boldsymbol{1}\\right).
```

`φ` is the **Lagrangian porosity** of the connected pore space in the reference
configuration — the volume fraction actually occupied by the fluid, which for a
fractured medium is the *crack* volume fraction ``\\sum_i (4\\pi/3) d_i
\\omega_i`` and therefore depends on the current apertures rather than on the
RVE alone (see [`pore_volume_fraction`](@ref)).

For an isotropic medium this reduces to the textbook ``1/M = (b -
\\varphi)/k_s``.

!!! warning "Incompressible saturating fluid"
    This expression assumes the pore fluid is **incompressible**, which is the
    setting of [barthelemyARMA2011](@cite). A fluid of finite bulk modulus
    ``k_f`` adds the storage term ``\\varphi/k_f``:

    ```math
    \\frac{1}{M} = \\boldsymbol{1} : \\mathbb{S}_{\\rm s} :
    (\\boldsymbol{B} - \\varphi\\,\\boldsymbol{1}) + \\frac{\\varphi}{k_f}.
    ```

    Add it yourself if the fluid compressibility matters — the function does
    not, since it has no way of knowing `k_f`. The distinction is not cosmetic:
    with an incompressible fluid and a *compressible* solid the fluid is
    effectively stiffer than the grains, and the Skempton coefficient then
    exceeds one (see [`skempton_tensor`](@ref)).

The *inverse* is the primitive rather than ``M`` itself because it is the
quantity that stays finite: an incompressible solid phase gives ``1/M = 0``,
i.e. ``M = \\infty``. [`biot_modulus`](@ref) returns `Inf` in that case, which
is correct but not something to feed to a linear solver.

See also [`biot_tensor`](@ref), [`biot_modulus`](@ref),
[`poroelastic_parameters`](@ref).
"""
inverse_biot_modulus(
    C_s::TensND.AbstractTens{4, 3}, B::TensND.AbstractTens{2, 3}, φ::Number
) = _inverse_biot_modulus(inv(C_s), B, φ)

function _inverse_biot_modulus(
        s_s::TensND.AbstractTens{4, 3}, B::TensND.AbstractTens{2, 3}, φ::Number
    )
    T = promote_type(eltype(s_s), eltype(B), typeof(φ))
    Id2 = TensND.tens_Id2(Val(3), Val(T))
    return Id2 ⊡ s_s ⊡ (B - φ * Id2)
end

"""
    biot_modulus(C_s, B, φ) -> Number

Biot modulus ``M = 1/(1/M)``, the reciprocal of
[`inverse_biot_modulus`](@ref MeanFieldHomogenization.Poromechanics.inverse_biot_modulus).

Returns `Inf` for an incompressible solid phase (``1/M = 0``). Prefer
[`inverse_biot_modulus`](@ref MeanFieldHomogenization.Poromechanics.inverse_biot_modulus) wherever the value is assembled into a system
matrix.
"""
biot_modulus(
    C_s::TensND.AbstractTens{4, 3}, B::TensND.AbstractTens{2, 3}, φ::Number
) = inv(inverse_biot_modulus(C_s, B, φ))

"""
    poroelastic_parameters(C_hom, C_s, φ) -> NamedTuple

Every poroelastic parameter of a saturated medium with a homogeneous solid
phase, in one pass:

```julia
(; B, inverse_modulus, modulus)
```

with `B` the [`biot_tensor`](@ref), `inverse_modulus` the
[`inverse_biot_modulus`](@ref MeanFieldHomogenization.Poromechanics.inverse_biot_modulus) ``1/M`` and `modulus` its reciprocal ``M``.

`C_s` is inverted **once**, which is why this is the entry point the
Gauss-point materials use rather than the three functions separately.

```jldoctest
julia> using MeanFieldHomogenization, TensND

julia> C_s = TensISO{3}(3 * 20.0, 2 * 12.0);   # k_s = 20, μ_s = 12

julia> rve = RVE(:M);

julia> add_matrix!(rve, Ellipsoid(1.0), Dict(:C => C_s));

julia> add_phase!(rve, :P, Ellipsoid(1.0), Dict(:C => TensISO{3}(1.0e-9, 1.0e-9));
                  fraction = 0.2);

julia> par = poroelastic_parameters(homogenize(rve, MoriTanaka()), C_s, 0.2);

julia> round(par.B[1, 1], digits = 4)   # b = 1 - k_MT/k_s = 1 - 12.8/20
0.36
```
"""
function poroelastic_parameters(
        C_hom::TensND.AbstractTens{4, 3}, C_s::TensND.AbstractTens{4, 3}, φ::Number
    )
    s_s = inv(C_s)
    B = _biot_tensor(C_hom, s_s)
    invM = _inverse_biot_modulus(s_s, B, φ)
    return (B = B, inverse_modulus = invM, modulus = inv(invM))
end

"""
    undrained_stiffness(C_hom, B, M) -> Tens{4,3}

**Undrained** stiffness ``\\mathbb{C}^{\\rm u} = \\mathbb{C}^{\\rm hom} +
M\\,\\boldsymbol{B} \\otimes \\boldsymbol{B}``.

It is the tangent stiffness of the closed system, obtained by eliminating `p`
from the poroelastic law under the undrained condition ``\\dot\\varphi = 0``.
`M = Inf` (incompressible solid *and* fluid) makes the result infinite, which
is the correct statement that no volume change is possible.

See also [`drained_stiffness`](@ref), [`skempton_tensor`](@ref).
"""
function undrained_stiffness(
        C_hom::TensND.AbstractTens{4, 3}, B::TensND.AbstractTens{2, 3}, M::Number
    )
    return C_hom + M * (B ⊗ B)
end

"""
    drained_stiffness(C_u, B, M) -> Tens{4,3}

Inverse of [`undrained_stiffness`](@ref): recover the drained stiffness
``\\mathbb{C}^{\\rm hom} = \\mathbb{C}^{\\rm u} - M\\,\\boldsymbol{B} \\otimes
\\boldsymbol{B}`` from an undrained measurement.
"""
function drained_stiffness(
        C_u::TensND.AbstractTens{4, 3}, B::TensND.AbstractTens{2, 3}, M::Number
    )
    return C_u - M * (B ⊗ B)
end

"""
    skempton_tensor(C_hom, B, M) -> Tens{2,3}

**Skempton tensor** ``\\boldsymbol{B}^{\\rm sk} = M\\,\\boldsymbol{B} : \\mathbb{S}^{\\rm u}`` (``\\mathbb{S}^{\\rm u} = (\\mathbb{C}^{\\rm u})^{-1}``), which gives
the pore pressure built up by an undrained stress increment:

```math
p = -\\,\\boldsymbol{B}^{\\rm sk} : \\boldsymbol{\\Sigma} .
```

Under isotropic compression ``\\boldsymbol{\\Sigma} = -p_0\\,\\boldsymbol{1}``
this yields ``p = p_0\\,{\\rm tr}\\,\\boldsymbol{B}^{\\rm sk}``, so
``{\\rm tr}\\,\\boldsymbol{B}^{\\rm sk}`` plays the role of the classical scalar
Skempton coefficient. In the isotropic case it reduces to ``M b / k^{\\rm u}``,
equivalently

```math
B = \\frac{1/k^{\\rm hom} - 1/k_s}
          {1/k^{\\rm hom} - 1/k_s + \\varphi\\,(1/k_f - 1/k_s)} .
```

!!! warning "The bound B ≤ 1 does not hold here"
    The familiar ``0 \\le B \\le 1`` assumes a fluid no stiffer than the solid
    grains, ``k_f \\le k_s``. [`inverse_biot_modulus`](@ref MeanFieldHomogenization.Poromechanics.inverse_biot_modulus) assumes an
    **incompressible** fluid (``k_f = \\infty``), so with a compressible solid
    the last term above is negative and ``B > 1``: the pore pressure exceeds the
    applied mean stress, because the pore volume is held fixed while the grains
    themselves compress. This is a genuine consequence of the assumption, not a
    numerical artifact. Strong anisotropy of the pore space, aligned cracks for
    instance, likewise removes any scalar bound.
"""
function skempton_tensor(
        C_hom::TensND.AbstractTens{4, 3}, B::TensND.AbstractTens{2, 3}, M::Number
    )
    return M * (B ⊡ inv(undrained_stiffness(C_hom, B, M)))
end
