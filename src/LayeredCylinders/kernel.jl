# =============================================================================
#  kernel.jl — the three radial problems of a layered cylinder, written once
#  for scalars and for operators.
#
#  A remote uniform strain on a cylinder with transversely isotropic coaxial
#  layers splits into three independent problems (Hill's moduli k, l, m, n, p):
#
#    * potential     φ(r) cos θ, flux f = c φ′           — antiplane shear
#                    (φ = u_z, c = p, f = σ_rz) and transverse conduction
#                    (φ = T, c = k_T, f = −q_r);
#    * axisymmetric  u_r(r) with a uniform axial strain ε — state (u_r, σ_rr);
#    * transverse shear  u_r = U cos 2θ, u_θ = V sin 2θ   — state
#                    (U, V, σ_rr, σ_rθ).
#
#  Each is propagated from the axis outward on a STATE (the quantities an
#  interface condition is written on), through the amplitudes of the layer.
#
#  The moduli and the states are "scalars" in a wide sense: numbers (floats,
#  duals, symbols) in elasticity and conduction, n×n Volterra matrices in aging
#  viscoelasticity, where a state entry maps the history of the core amplitude
#  to the history of that entry. Operators do not commute, so every product is
#  written in the order the mechanics gives it:
#
#    * a modulus acts on the LEFT of what it multiplies (σ = C ∘ ε);
#    * dividing by a modulus is a LEFT division, `_ldiv(c, x) = c⁻¹ ∘ x`;
#    * a normalization (changing the history the states are mapped from) acts
#      on the RIGHT, `x / A`.
#
#  No formula below uses a product of two moduli in an order that the
#  equilibrium equations do not impose.
# =============================================================================

# c⁻¹ ∘ x. For numbers, a plain division (it also gives m/k = 0 at k = ∞).
@inline _ldiv(c::Number, x) = x / c
@inline _ldiv(c::AbstractMatrix, x::AbstractMatrix) = c \ x

# χ + c·𝟙, the identity being that of the algebra χ lives in.
@inline _shift(χ, c::Real) = χ + c * one(χ)

# Exact zero of a modulus (an impermeable layer). Never true for a symbol,
# which does not compare.
@inline _is_zero_modulus(c::Number) = is_hard_numeric(c) && iszero(c)
@inline _is_zero_modulus(c::AbstractMatrix) = is_hard_numeric(eltype(c)) && iszero(c)

# ── Potential problem: φ = A r + B / r,  f = c (A − B / r²) ──────────────────

@inline _pot_state(r, c, A, B) = (r * A + B / r, c * (A - B / r^2))

@inline function _pot_amplitudes(r, c, φ, f)
    g = _ldiv(c, f)
    return (φ / r + g) / 2, r * (φ - r * g) / 2
end

# ── Axisymmetric problem: u = A r + B / r,  σ = 2k A − 2m B / r² + l ε ───────

@inline _axi_state(r, L, A, B, ε) =
    (r * A + B / r, 2 * (L.k * A) - 2 * (L.m * B) / r^2 + L.l * ε)

@inline function _axi_amplitudes(r, L, u, σ, ε)
    A = _ldiv(2 * (L.k + L.m), σ - L.l * ε + 2 * (L.m * u) / r)
    return A, r * u - r^2 * A
end

# ── Transverse shear: four modes r, r³, r⁻³, r⁻¹ with amplitudes (a, b, c, d) ─
#
#  With χ = k⁻¹ ∘ m, the displacement shapes that satisfy equilibrium are
#
#      U =  r a + r³ (χ − 1) b + c / r³ + (1 + χ) d / r,
#      V = −r a + r³ (χ + 2) b + c / r³ −      χ  d / r,
#
#  and the tractions, where no product of moduli is left,
#
#      σ_rr = m ( 2a            − 6 c / r⁴ − 4 d / r²),
#      σ_rθ = m (−2a + 6 r² b   − 6 c / r⁴ − 2 d / r²).
#
#  At k = ∞ (χ = 0) every entry stays finite: an incompressible layer needs no
#  special case here.

@inline function _shear_state(r, k, m, a, b, c, d)
    χ = _ldiv(k, m)
    χb = χ * b
    χd = χ * d
    U = r * a + r^3 * (χb - b) + c / r^3 + (d + χd) / r
    V = -r * a + r^3 * (χb + 2 * b) + c / r^3 - χd / r
    σrr = m * (2 * a - 6 * c / r^4 - 4 * d / r^2)
    σrθ = m * (-2 * a + 6 * r^2 * b - 6 * c / r^4 - 2 * d / r^2)
    return U, V, σrr, σrθ
end

# The inverse map, by elimination: only (1 + χ) is inverted.
@inline function _shear_amplitudes(r, k, m, U, V, σrr, σrθ)
    χ = _ldiv(k, m)
    τ₁ = _ldiv(m, σrr)
    τ₂ = _ldiv(m, σrθ)
    D = 2 * _shift(χ, 1)
    b = _ldiv(D, (U + V + r * (τ₁ + τ₂) / 6) / r^3)
    d = _ldiv(D, r * (U - V - r * (τ₁ - τ₂) / 2))
    a = (τ₁ - τ₂) / 4 + 3 * r^2 * b / 2 + d / (2 * r^2)
    c = r^6 * b / 2 - r^2 * d / 2 - r^4 * (τ₁ + τ₂) / 12
    return a, b, c, d
end

# ── Interface jumps ──────────────────────────────────────────────────────────
#
#  On a cylinder of radius r, with u_r = U cos mθ … and the Gurtin–Murdoch
#  surface law σˢ = λˢ tr(εˢ) 𝟏ˢ + 2 μˢ εˢ (κˢ = λˢ + μˢ), [σ·n] = −divˢ σˢ:
#
#      spring     [u_r] = sₙ σ_rr,  [u_θ] = sₜ σ_rθ,  [u_z] = sₜ σ_rz;
#      membrane   axisymmetric   [σ_rr] = ((κˢ + μˢ) u_r / r + (κˢ − μˢ) ε) / r,
#                 cos 2θ         [σ_rr] = (κˢ + μˢ)(U + 2V) / r²,
#                                [σ_rθ] = 2 (κˢ + μˢ)(U + 2V) / r²,
#                 antiplane      [σ_rz] = μˢ u_z / r²;
#      Kapitza    [T] = −ρ q_r = ρ f;
#      surface conduction  [q_r] = −kˢ T / r², that is [f] = kˢ T / r².

struct Antiplane end
struct TransverseConduction end
struct Axisymmetric end
struct TransverseShear end

_physics_name(::Antiplane) = "elasticity"
_physics_name(::Axisymmetric) = "elasticity"
_physics_name(::TransverseShear) = "elasticity"
_physics_name(::TransverseConduction) = "conduction"

_wrong_interface(itf, pb) = throw(
    ArgumentError(
        "a $(nameof(typeof(itf))) does not apply to a LayeredCylinder in $(_physics_name(pb)): " *
            "elasticity takes PerfectInterface, SpringInterface or MembraneInterface, " *
            "conduction takes PerfectInterface, KapitzaInterface or SurfaceConductiveInterface"
    )
)

@inline _jump(::PerfectInterface, _pb, r, s, ε = nothing) = s
@inline _jump(itf::AbstractInterface, pb, r, s, ε = nothing) = _wrong_interface(itf, pb)

@inline _jump(itf::SpringInterface, ::Antiplane, r, (φ, f), ε = nothing) =
    (φ + spring_compliances(itf)[2] * f, f)
@inline _jump(itf::MembraneInterface, ::Antiplane, r, (φ, f), ε = nothing) =
    (φ, f + itf.μs * φ / r^2)

@inline _jump(itf::KapitzaInterface, ::TransverseConduction, r, (φ, f), ε = nothing) =
    (φ + itf.resistance * f, f)
@inline _jump(itf::SurfaceConductiveInterface, ::TransverseConduction, r, (φ, f), ε = nothing) =
    (φ, f + itf.conductance * φ / r^2)

@inline _jump(itf::SpringInterface, ::Axisymmetric, r, (u, σ), ε) =
    (u + spring_compliances(itf)[1] * σ, σ)
@inline _jump(itf::MembraneInterface, ::Axisymmetric, r, (u, σ), ε) =
    (u, σ + ((itf.κs + itf.μs) * u / r + (itf.κs - itf.μs) * ε) / r)

@inline function _jump(itf::SpringInterface, ::TransverseShear, r, (U, V, σrr, σrθ), ε = nothing)
    sn, st = spring_compliances(itf)
    return (U + sn * σrr, V + st * σrθ, σrr, σrθ)
end
@inline function _jump(itf::MembraneInterface, ::TransverseShear, r, (U, V, σrr, σrθ), ε = nothing)
    h = (itf.κs + itf.μs) * (U + 2 * V) / r^2
    return (U, V, σrr + h, σrθ + 2 * h)
end

# ── Sequences of states ──────────────────────────────────────────────────────
#
#  Each sequence returns, for k = 1…N, the state on the inner side of r_k
#  (`inside[k]`), on its outer side (`outside[k]`), and the amplitudes of layer
#  k (`amps[k]`), all per unit core amplitude. `S` is the common type of the
#  entries (a number type or a matrix type), which the caller has promoted the
#  moduli and the seed to.

"""
    _pot_sequence(pb, radii, cs, interfaces, A₁) -> (inside, outside, amps)

States ``(\\varphi, f)`` of the potential problem `pb` at each interface and
amplitudes ``(A, B)`` of each layer, for the core amplitude `A₁`.

A layer of zero modulus other than the core is impermeable: no flux crosses
it, the layers inside it are shielded and carry no field, and its own field,
``A\\,(r - r_{k-1}^2/r)``, vanishes on its inner face (the limit of a vanishing
modulus). The sequence then restarts in that layer with the amplitude `A₁`.
"""
function _pot_sequence(pb, radii::NTuple{N}, cs::NTuple{N, S}, interfaces, A₁::S) where {N, S}
    inside = Vector{NTuple{2, S}}(undef, N)
    outside = Vector{NTuple{2, S}}(undef, N)
    amps = Vector{NTuple{2, S}}(undef, N)
    A, B = A₁, zero(A₁)
    for k in 1:N
        if k > 1
            if _is_zero_modulus(cs[k])
                _is_zero_modulus(cs[k - 1]) && throw(
                    ArgumentError(
                        "layers $(k - 1) and $k of the LayeredCylinder both have a zero modulus: " *
                            "the field of an impermeable layer is defined by a conducting " *
                            "neighbor inside it; merge the two layers"
                    )
                )
                for j in 1:(k - 1)
                    inside[j] = outside[j] = amps[j] = (zero(A₁), zero(A₁))
                end
                A, B = A₁, -radii[k - 1]^2 * A₁
            else
                A, B = _pot_amplitudes(radii[k - 1], cs[k], outside[k - 1]...)
            end
        end
        amps[k] = (A, B)
        inside[k] = _pot_state(radii[k], cs[k], A, B)
        outside[k] = _jump(interfaces[k], pb, radii[k], inside[k])
    end
    return inside, outside, amps
end

"""
    _axi_sequence(radii, Ls, interfaces, A₁, ε) -> (inside, outside, amps)

States ``(u_r, \\sigma_{rr})`` of the axisymmetric problem and amplitudes
``(A, B)`` of each layer, for the core amplitude `A₁` and the axial strain `ε`.
"""
function _axi_sequence(radii::NTuple{N}, Ls::NTuple{N}, interfaces, A₁::S, ε::S) where {N, S}
    inside = Vector{NTuple{2, S}}(undef, N)
    outside = Vector{NTuple{2, S}}(undef, N)
    amps = Vector{NTuple{2, S}}(undef, N)
    A, B = A₁, zero(A₁)
    for k in 1:N
        if k > 1
            A, B = _axi_amplitudes(radii[k - 1], Ls[k], outside[k - 1]..., ε)
        end
        amps[k] = (A, B)
        inside[k] = _axi_state(radii[k], Ls[k], A, B, ε)
        outside[k] = _jump(interfaces[k], Axisymmetric(), radii[k], inside[k], ε)
    end
    return inside, outside, amps
end

"""
    _shear_sequence(radii, Ls, interfaces, a₁, b₁) -> (inside, outside, amps)

States ``(U, V, \\sigma_{rr}, \\sigma_{r\\theta})`` of the transverse shear problem
and amplitudes ``(a, b, c, d)`` of each layer, for the core amplitudes
``(a_1, b_1)`` of its two regular modes.
"""
function _shear_sequence(radii::NTuple{N}, Ls::NTuple{N}, interfaces, a₁::S, b₁::S) where {N, S}
    inside = Vector{NTuple{4, S}}(undef, N)
    outside = Vector{NTuple{4, S}}(undef, N)
    amps = Vector{NTuple{4, S}}(undef, N)
    amp = (a₁, b₁, zero(a₁), zero(a₁))
    for k in 1:N
        if k > 1
            amp = _shear_amplitudes(radii[k - 1], Ls[k].k, Ls[k].m, outside[k - 1]...)
        end
        amps[k] = amp
        inside[k] = _shear_state(radii[k], Ls[k].k, Ls[k].m, amp...)
        outside[k] = _jump(interfaces[k], TransverseShear(), radii[k], inside[k])
    end
    return inside, outside, amps
end

# ── Normalization to a unit remote loading ───────────────────────────────────

# x ∘ X on every entry of a sequence (X on the right: it changes the history
# the states are mapped from).
_scale(seq::Vector{<:Tuple}, X) = map(s -> map(x -> x * X, s), seq)
_combine(s₁::Vector{<:Tuple}, X₁, s₂::Vector{<:Tuple}, X₂) =
    map((a, b) -> map((x, y) -> x * X₁ + y * X₂, a, b), s₁, s₂)
_combine(s₁::Vector{<:Tuple}, X₁, s₂::Vector{<:Tuple}) =
    map((a, b) -> map((x, y) -> x * X₁ + y, a, b), s₁, s₂)

"""
    _solve_potential(pb, radii, cs, interfaces, c₀) -> (; inside, outside, amps)

Potential problem under the unit remote amplitude ``A_\\infty = 1``
(``\\varphi \\to r\\cos\\theta``): the sequence of the unit core amplitude,
divided on the right by the amplitude ``A`` it produces in the matrix `c₀`.
"""
function _solve_potential(pb, radii::NTuple{N}, cs::NTuple{N, S}, interfaces, c₀::S) where {N, S}
    inside, outside, amps = _pot_sequence(pb, radii, cs, interfaces, one(c₀))
    A∞, _ = _pot_amplitudes(radii[N], c₀, outside[N]...)
    X = inv(A∞)
    return (; inside = _scale(inside, X), outside = _scale(outside, X), amps = _scale(amps, X))
end

"""
    _solve_axisymmetric(radii, Ls, interfaces, L₀) -> (dilatation, axial)

Axisymmetric problem under the two unit remote loadings: an in-plane
equibiaxial strain ``e = 1`` (``u_r \\to r``) at ``\\varepsilon = 0``, and an axial
strain ``\\varepsilon = 1`` at ``e = 0``. With ``\\alpha`` and ``\\beta`` the matrix
amplitudes produced by the unit core amplitude and by the unit axial strain,
the core amplitude under ``(e, \\varepsilon)`` is
``\\alpha^{-1}\\circ(e - \\beta\\circ\\varepsilon)``.
"""
function _solve_axisymmetric(radii::NTuple{N}, Ls::NTuple{N}, interfaces, L₀) where {N}
    𝟙 = one(L₀.k)
    𝟘 = zero(L₀.k)
    pa = _axi_sequence(radii, Ls, interfaces, 𝟙, 𝟘)
    pz = _axi_sequence(radii, Ls, interfaces, 𝟘, 𝟙)
    α, _ = _axi_amplitudes(radii[N], L₀, pa[2][N]..., 𝟘)
    β, _ = _axi_amplitudes(radii[N], L₀, pz[2][N]..., 𝟙)
    Xe = inv(α)
    Xz = -_ldiv(α, β)
    dilatation = (;
        inside = _scale(pa[1], Xe), outside = _scale(pa[2], Xe), amps = _scale(pa[3], Xe),
    )
    axial = (;
        inside = _combine(pa[1], Xz, pz[1]), outside = _combine(pa[2], Xz, pz[2]),
        amps = _combine(pa[3], Xz, pz[3]),
    )
    return dilatation, axial
end

"""
    _solve_shear(radii, Ls, interfaces, L₀) -> (; inside, outside, amps)

Transverse shear problem under the unit remote strain
``\\underline{e}_1\\otimes\\underline{e}_1 - \\underline{e}_2\\otimes\\underline{e}_2``
(``U \\to r``, ``V \\to -r``): the combination of the two core modes for which the
matrix amplitudes are ``(a, b) = (1, 0)``.
"""
function _solve_shear(radii::NTuple{N}, Ls::NTuple{N}, interfaces, L₀) where {N}
    𝟙 = one(L₀.k)
    𝟘 = zero(L₀.k)
    p₁ = _shear_sequence(radii, Ls, interfaces, 𝟙, 𝟘)
    p₂ = _shear_sequence(radii, Ls, interfaces, 𝟘, 𝟙)
    a¹, b¹, _, _ = _shear_amplitudes(radii[N], L₀.k, L₀.m, p₁[2][N]...)
    a², b², _, _ = _shear_amplitudes(radii[N], L₀.k, L₀.m, p₂[2][N]...)
    # [a¹ a²; b¹ b²] ∘ (X₁, X₂) = (𝟙, 0), by block elimination on a¹.
    X₂ = -_ldiv(b² - b¹ * _ldiv(a¹, a²), b¹ / a¹)
    X₁ = _ldiv(a¹, 𝟙 - a² * X₂)
    return (;
        inside = _combine(p₁[1], X₁, p₂[1], X₂), outside = _combine(p₁[2], X₁, p₂[2], X₂),
        amps = _combine(p₁[3], X₁, p₂[3], X₂),
    )
end

# ── Pointwise strain of each problem, from the amplitudes of a region ────────
#
#  In the cylindrical frame (e_r, e_θ, e_z) of the point, the remote strain has
#  the components e = (E_rr + E_θθ)/2, D = (E_rr − E_θθ)/2, E_rθ, E_rz, E_θz and
#  ε = E_zz, and the local strain does not depend on θ any more:
#
#      ε_rr = (A_e − B_e/r²) e + (A_z − B_z/r²) ε + U′ D,
#      ε_θθ = (A_e + B_e/r²) e + (A_z + B_z/r²) ε + ((U + 2V)/r) D,
#      ε_rθ = ((2U/r − V′ + V/r)/2) E_rθ,
#      ε_rz = φ′ E_rz,     ε_θz = (φ/r) E_θz,     ε_zz = ε.
#
#  The terms singular on the axis carry amplitudes that vanish identically in
#  the core; they are dropped there rather than evaluated as 0/0.

@inline _at_axis(r) = is_hard_numeric(typeof(r)) && iszero(r)
@inline _over(x, r, p::Int) = _at_axis(r) ? zero(x) : x / r^p

# (u_r/r − 2B/r², u_r/r) per unit loading: the pair (ε_rr, ε_θθ) of u_r = A r + B/r.
@inline _axi_strains(r, A, B) = (A - _over(B, r, 2), A + _over(B, r, 2))

# (φ′, φ/r) of φ = A r + B/r.
@inline _pot_gradients(r, A, B) = (A - _over(B, r, 2), A + _over(B, r, 2))

# (U′, (U + 2V)/r, (2U/r − V′ + V/r)/2) of the transverse shear modes.
@inline function _shear_strains(r, k, m, a, b, c, d)
    χ = _ldiv(k, m)
    χb = χ * b
    χd = χ * d
    c₄ = _over(c, r, 4)
    d₂ = _over(d, r, 2)
    dU = a + 3 * r^2 * (χb - b) - 3 * c₄ - _over(d + χd, r, 2)
    T = -a + 3 * r^2 * (χb + b) + 3 * c₄ + _over(d - χd, r, 2)
    H = a - 3 * r^2 * b + 3 * c₄ + d₂
    return dU, T, H
end

# (U, V) alone, for the displacement.
@inline function _shear_displacements(r, k, m, a, b, c, d)
    χ = _ldiv(k, m)
    χb = χ * b
    χd = χ * d
    c₃ = _over(c, r, 3)
    return r * a + r^3 * (χb - b) + c₃ + _over(d + χd, r, 1),
        -r * a + r^3 * (χb + 2 * b) + c₃ - _over(χd, r, 1)
end

"""
    _matrix_amplitudes(sol) -> (; dilatation, axial, shear, antiplane)

Amplitudes of the four problems in the matrix (region ``N+1``), read on the
outer side of ``r_N``.
"""
function _matrix_amplitudes(sol)
    N = length(sol.radii)
    R = sol.radii[N]
    𝟙 = one(sol.L₀.k)
    return (;
        dilatation = _axi_amplitudes(R, sol.L₀, sol.dilatation.outside[N]..., zero(𝟙)),
        axial = _axi_amplitudes(R, sol.L₀, sol.axial.outside[N]..., 𝟙),
        shear = _shear_amplitudes(R, sol.L₀.k, sol.L₀.m, sol.shear.outside[N]...),
        antiplane = _pot_amplitudes(R, sol.L₀.p, sol.antiplane.outside[N]...),
    )
end

"""
    _local_strain_entries(sol, mat, k, r) -> NamedTuple

Entries of the pointwise strain concentration of region `k` (the matrix for
``k = N+1``, of amplitudes `mat`) at radius `r`, in the cylindrical frame:
`rr_e`, `rr_z`, `rr_D` (``\\varepsilon_{rr}`` per ``e``, ``\\varepsilon`` and
``D``), `tt_e`, `tt_z`, `tt_D` (``\\varepsilon_{\\theta\\theta}``), and the shear
ratios `rt`, `rz`, `tz`.
"""
function _local_strain_entries(sol, mat, k::Int, r)
    N = length(sol.radii)
    L = k ≤ N ? sol.Ls[k] : sol.L₀
    amp(seq, name) = k ≤ N ? seq.amps[k] : getfield(mat, name)
    rr_e, tt_e = _axi_strains(r, amp(sol.dilatation, :dilatation)...)
    rr_z, tt_z = _axi_strains(r, amp(sol.axial, :axial)...)
    rr_D, tt_D, rt = _shear_strains(r, L.k, L.m, amp(sol.shear, :shear)...)
    rz, tz = _pot_gradients(r, amp(sol.antiplane, :antiplane)...)
    return (; rr_e, rr_z, rr_D, tt_e, tt_z, tt_D, rt, rz, tz)
end

"""
    _local_mandel(c) -> Matrix (6 × 6)

Mandel matrix, in the cylindrical frame ordered ``(rr, \\theta\\theta, zz,
\\theta z, rz, r\\theta)``, of the pointwise strain concentration of entries `c`.
"""
function _local_mandel(c)
    𝟘 = zero(c.rt)
    M = fill(𝟘, 6, 6)
    M[1, 1] = (c.rr_e + c.rr_D) / 2
    M[1, 2] = (c.rr_e - c.rr_D) / 2
    M[1, 3] = c.rr_z
    M[2, 1] = (c.tt_e + c.tt_D) / 2
    M[2, 2] = (c.tt_e - c.tt_D) / 2
    M[2, 3] = c.tt_z
    M[3, 3] = one(c.rt)
    M[4, 4] = c.tz
    M[5, 5] = c.rz
    M[6, 6] = c.rt
    return M
end
