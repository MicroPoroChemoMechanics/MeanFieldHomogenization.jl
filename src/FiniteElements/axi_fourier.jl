# =============================================================================
#  axi_fourier.jl — Fourier finite elements on the meridian half-plane.
#
#  A field on a solid of revolution is expanded in Fourier series in the
#  azimuth θ.  For elasticity, mode `m` reads
#
#      u_ρ = ū_ρ(ρ,z) cos mθ,   u_θ = ū_θ(ρ,z) sin mθ,   u_z = ū_z(ρ,z) cos mθ
#
#  and for transport  T = t̄(ρ,z) cos mθ.  Because the geometry and the
#  constituents are invariant by rotation about the axis, the modes **do not
#  couple**: each is a separate two-dimensional problem on the half-plane, and
#  a macroscopic strain (or gradient) excites exactly one of them.
#
#      | macroscopic loading             | mode |
#      |---------------------------------|------|
#      | ε = e₃⊗e₃  or  e₁⊗e₁ + e₂⊗e₂    |  0   |
#      | ε = e₁⊗ˢe₃                      |  1   |
#      | ε = e₁⊗e₁ − e₂⊗e₂               |  2   |
#      | ∇T = e₃                         |  0   |
#      | ∇T = e₁                         |  1   |
#
#  The strain in cylindrical components is
#
#      ε_ρρ = ∂ρ ū_ρ                  γ_ρz = ∂z ū_ρ + ∂ρ ū_z
#      ε_θθ = (ū_ρ + m ū_θ)/ρ         γ_ρθ = −m ū_ρ/ρ + ∂ρ ū_θ − ū_θ/ρ
#      ε_zz = ∂z ū_z                  γ_θz = ∂z ū_θ − m ū_z/ρ
#
#  (the θ-dependence factors out: the first group varies as `cos mθ`, the
#  second as `sin mθ`, and since a transversely isotropic material does not
#  couple the two, ∫dθ leaves a single constant — 2π for m = 0, π otherwise —
#  in front of the whole quadratic form.  The right-hand side being purely a
#  Dirichlet lift, that constant cancels and is not carried here.)
#
#  ── The axis ────────────────────────────────────────────────────────────────
#  Single-valuedness of the three-dimensional field on ρ = 0 forces
#
#      m = 0 : ū_ρ = 0                     m ≥ 2 : ū_ρ = ū_θ = ū_z = 0
#      m = 1 : ū_ρ + ū_θ = 0 and ū_z = 0
#
#  The m = 1 condition couples two components, which a plain Dirichlet
#  condition cannot express.  It is therefore solved by a change of unknowns,
#
#      ū_ρ = p + q,   ū_θ = −p + q     ⟺     p = (ū_ρ − ū_θ)/2,
#                                            q = (ū_ρ + ū_θ)/2,
#
#  after which the axis conditions read `q = 0` and `ū_z = 0` — two ordinary
#  Dirichlet conditions.  That substitution also removes every `1/ρ` term
#  attached to an unconstrained unknown, which is what keeps the quadrature
#  well behaved next to the axis.
# =============================================================================

const _SQ2 = sqrt(2.0)

# Change of unknowns per mode: columns of the returned matrix express
# (ū_ρ, ū_θ, ū_z) in terms of the *solved* components.
_axi_dof_map(m::Int) = m == 0 ? [1.0 0.0; 0.0 0.0; 0.0 1.0] :
    m == 1 ? [1.0 1.0 0.0; -1.0 1.0 0.0; 0.0 0.0 1.0] :
    [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0]

"Number of scalar unknowns per node in elastic mode `m`."
_axi_ncomp(m::Int) = m == 0 ? 2 : 3

"Components (in the solved unknowns) that must vanish on the axis."
_axi_axis_zeros(m::Int) = m == 0 ? (1,) : m == 1 ? (2, 3) : (1, 2, 3)

"""
    _axi_B_elast(m, N, dNρ, dNz, ρ) -> 6×ncomp matrix

Kelvin-Mandel strain ``(\\varepsilon_{\\rho\\rho}, \\varepsilon_{\\theta\\theta}, \\varepsilon_{zz}, \\gamma_{\\theta z}/\\sqrt{2}, \\gamma_{\\rho z}/\\sqrt{2}, \\gamma_{\\rho\\theta}/\\sqrt{2})`` produced
by one scalar shape function acting on each solved component, in Fourier mode
``m``. The index order matches `mandel66_minor` with ``1\\mapsto\\rho``, ``2\\mapsto\\theta``, ``3\\mapsto z``, so a
transversely isotropic material matrix can be used as is.
"""
function _axi_B_elast(m::Int, N::Float64, dNρ::Float64, dNz::Float64, ρ::Float64)
    iρ = N / ρ
    B = zeros(6, 3)                       # columns: ū_ρ, ū_θ, ū_z
    B[1, 1] = dNρ                                        # ε_ρρ
    B[2, 1] = iρ
    B[2, 2] = m * iρ                                     # ε_θθ
    B[3, 3] = dNz                                        # ε_zz
    B[4, 2] = dNz / _SQ2
    B[4, 3] = -m * iρ / _SQ2                             # γ_θz/√2
    B[5, 1] = dNz / _SQ2
    B[5, 3] = dNρ / _SQ2                                 # γ_ρz/√2
    B[6, 1] = -m * iρ / _SQ2
    B[6, 2] = (dNρ - iρ) / _SQ2                          # γ_ρθ/√2
    return B * _axi_dof_map(m)
end

"""
    _axi_B_cond(m, N, dNρ, dNz, ρ) -> 3×1 matrix

Cylindrical gradient ``(\\partial_\\rho\\bar t, -m\\bar t/\\rho, \\partial_z\\bar t)`` of one scalar shape function in
Fourier mode `m`.
"""
_axi_B_cond(m::Int, N::Float64, dNρ::Float64, dNz::Float64, ρ::Float64) =
    reshape([dNρ, -m * N / ρ, dNz], 3, 1)

# ─── Azimuthal projection ────────────────────────────────────────────────────
#
#  Averaging a modal field over θ picks exactly the Cartesian components the
#  mode is responsible for.  With `v` the Kelvin-Mandel vector of the modal
#  amplitude of a symmetric 2nd-order tensor `T`:
#
#      m = 0 :  ⟨T₁₁⟩ = ⟨T₂₂⟩ = (v₁+v₂)/2,  ⟨T₃₃⟩ = v₃
#      m = 1 :  ⟨T₁₃⟩ = (v₅ − v₄)/(2√2)
#      m = 2 :  ⟨T₁₁ − T₂₂⟩ = (v₁ − v₂)/2 − v₆/√2
#
#  What the modal blocks are written in are the coordinates on the *Kelvin
#  basis of the mode* — the normalized tensors `(e₁⊗e₁+e₂⊗e₂)/√2`, `e₃⊗e₃`,
#  `(e₁⊗e₃+e₃⊗e₁)/√2`, `(e₁⊗e₁−e₂⊗e₂)/√2` — hence the extra √2 factors below.

"Coordinates of a modal Kelvin-Mandel amplitude on the Kelvin basis of mode `m`."
function _axi_project(m::Int, v::AbstractVector{Float64})
    return if m == 0
        [(v[1] + v[2]) / _SQ2, v[3]]
    elseif m == 1
        [(v[5] - v[4]) / 2]
    else
        [((v[1] - v[2]) / 2 - v[6] / _SQ2) / _SQ2]
    end
end

"Coordinates of a modal cylindrical gradient on the Cartesian axis of mode `m`."
_axi_project_cond(m::Int, g::AbstractVector{Float64}) =
    m == 0 ? [g[3]] : [(g[1] - g[2]) / 2]

# ─── Global Kelvin basis ─────────────────────────────────────────────────────
#
#  Columns are the Kelvin-Mandel coordinates (order 11, 22, 33, 23, 13, 12,
#  √2 weights on the shears) of the six basis tensors, grouped by mode:
#  (m₁, m₂) span mode 0, (m₃, m₃′) mode 1, (m₄, m₄′) mode 2.

const _AXI_Q = let Q = zeros(6, 6)
    Q[1, 1] = Q[2, 1] = 1 / _SQ2          # m₁ = (e₁⊗e₁+e₂⊗e₂)/√2
    Q[3, 2] = 1.0                         # m₂ = e₃⊗e₃
    Q[5, 3] = 1.0                         # m₃ = (e₁⊗e₃+e₃⊗e₁)/√2
    Q[4, 4] = 1.0                         # m₃′ = (e₂⊗e₃+e₃⊗e₂)/√2
    Q[1, 5] = 1 / _SQ2
    Q[2, 5] = -1 / _SQ2                   # m₄ = (e₁⊗e₁−e₂⊗e₂)/√2
    Q[6, 6] = 1.0                         # m₄′ = (e₁⊗e₂+e₂⊗e₁)/√2
    Q
end

"""
    _axi_assemble_66(G0, G1, G2) -> 6×6

Kelvin-Mandel matrix of the transversely isotropic 4th-order tensor whose
mode-0 block is the ``2\\times 2`` `G0`, whose mode-1 eigenvalue is `G1` and whose mode-2
eigenvalue is `G2`. The two shear modes each appear twice, on the pairs
``(13, 23)`` and ``(11-22, 12)``.
"""
function _axi_assemble_66(G0::AbstractMatrix{Float64}, G1::Float64, G2::Float64)
    Ĝ = zeros(6, 6)
    Ĝ[1:2, 1:2] .= G0
    Ĝ[3, 3] = Ĝ[4, 4] = G1
    Ĝ[5, 5] = Ĝ[6, 6] = G2
    return _AXI_Q * Ĝ * _AXI_Q'
end

# ─── Boundary data ───────────────────────────────────────────────────────────

"Modal amplitude of `u = mⱼ·x`, `j` indexing the Kelvin basis of mode `m`."
function _axi_bc_affine(m::Int, j::Int, ρ::Float64, z::Float64)
    if m == 0
        # m₁ = (e₁⊗e₁+e₂⊗e₂)/√2 → u = (ρ/√2) e_ρ ;  m₂ = e₃⊗e₃ → u = z e_z
        return j == 1 ? (ρ / _SQ2, 0.0) : (0.0, z)
    elseif m == 1
        # m₃ = (e₁⊗e₃+e₃⊗e₁)/√2 → ū_ρ = z/√2, ū_θ = −z/√2, ū_z = ρ/√2
        return (z / _SQ2, 0.0, ρ / _SQ2)                 # (p, q, ū_z)
    else
        # m₄ = (e₁⊗e₁−e₂⊗e₂)/√2 → ū_ρ = ρ/√2, ū_θ = −ρ/√2, ū_z = 0
        return (ρ / _SQ2, -ρ / _SQ2, 0.0)
    end
end

"""
    _axi_bc_dipole(m, j, ρ, z, μ, ν, M) -> NTuple

Modal amplitude, in the solved unknowns, of the Kelvin dipole far field

```math
u_i(\\underline{x}) = +\\frac{\\partial G_{ij}}{\\partial x_k}(\\underline{x})\\,M_{jk}
```

for a polarization *moment* ``\\boldsymbol{M}`` times the Kelvin basis tensor ``\\boldsymbol{m}_j`` of mode ``m``
— the same field as `dipole_displacement_iso`, resolved on the cylindrical
basis.

The sign is the one the equivalent body force of a polarization imposes,
`f = +div p`, so that a *positive* dilatational polarization pushes material
outwards. Getting it backwards is not subtle in its effect but is invisible in
its symptom: the corrected result then carries exactly **twice** the
uncorrected truncation bias instead of none of it.

Every branch vanishes on the axis exactly where the modal axis conditions
require it, so the outer and the axis conditions never contradict each other
at the poles.
"""
function _axi_bc_dipole(
        m::Int, j::Int, ρ::Float64, z::Float64, μ::Float64, ν::Float64, M::Float64
    )
    r = sqrt(ρ^2 + z^2)
    A = M / (16 * π * μ * (1 - ν) * r^2)
    c = -2 * (1 - 2 * ν)
    nρ, nz = ρ / r, z / r
    if m == 0
        # m₁ = (e₁⊗e₁+e₂⊗e₂)/√2 : t = 1/√2, a = 0 ; m₂ = e₃⊗e₃ : t = 0, a = 1
        t, ax = j == 1 ? (1 / _SQ2, 0.0) : (0.0, 1.0)
        S = 2t + ax - 3 * (t * nρ^2 + ax * nz^2)
        return (A * (c * t + S) * nρ, A * (c * ax + S) * nz)
    elseif m == 1
        s = 1 / _SQ2                                     # m₃ amplitude
        ūρ = A * s * (c * nz - 6 * nρ^2 * nz)
        ūθ = -A * s * c * nz
        ūz = A * s * (c * nρ - 6 * nρ * nz^2)
        return ((ūρ - ūθ) / 2, (ūρ + ūθ) / 2, ūz)        # (p, q, ū_z)
    else
        d = 1 / _SQ2                                     # m₄ amplitude
        return (A * d * (c * nρ - 3 * nρ^3), -A * d * c * nρ, -A * d * 3 * nρ^2 * nz)
    end
end

"Modal amplitude of `T = H·x` on the outer sphere, `H = e₃` (m=0) or `e₁` (m=1)."
_axi_bc_affine_cond(m::Int, ρ::Float64, z::Float64) = m == 0 ? z : ρ

"""
    _axi_bc_dipole_cond(m, ρ, z, k₀, M) -> Float64

Modal amplitude of the scalar dipole far field ``T = \\underline{M}\\cdot\\underline{x}/(4\\pi k_0 r^3)`` generated
by a polarization moment ``\\underline{M}`` along the Cartesian axis of mode ``m``.
"""
function _axi_bc_dipole_cond(m::Int, ρ::Float64, z::Float64, k₀::Float64, M::Float64)
    r = sqrt(ρ^2 + z^2)
    return -M * (m == 0 ? z : ρ) / (4 * π * k₀ * r^3)
end
