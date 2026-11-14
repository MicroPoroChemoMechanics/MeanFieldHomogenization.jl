# =============================================================================
#  assembly.jl — from the states of the radial problems to the averages over
#  the cylinder and over its layers. Generic in the scalars, as `kernel.jl`.
#
#  Over a disc of radius r of the section, the average of a gradient is a
#  contour integral of the field, and that of an equilibrated flux a contour
#  integral of its normal component:
#
#      ⟨ε⟩ = (1/πr²) ∮ u ⊗ˢ n ds ,      ⟨σ⟩ = (1/πr²) ∮ x ⊗ˢ (σ·n) ds .
#
#  On the three problems they read
#
#      axisymmetric   in-plane strain  e = u_r / r,        mean stress  σ_rr,
#      cos 2θ         shear strain     (U − V) / (2r),     stress  (σ_rr − σ_rθ)/2,
#      potential      gradient         φ / r,              flux    f.
#
#  Read on the outer side of an interface, they include what it carries: the
#  opening of a spring (a strain without stress), the surface stress of a
#  membrane. Read on its inner side, they do not. An annulus is the difference
#  of two discs.
#
#  A transversely isotropic operator is handled through six entries
#  `(zz, ze, mz, me, T, L)`: for a stress per strain, σ_zz per ε_zz (`zz`) and
#  per in-plane equibiaxial strain e (`ze`), in-plane mean stress per ε_zz
#  (`mz`) and per e (`me`), transverse (`T`) and axial (`L`) shear ratios. A
#  stiffness (k, l, m, n, p) is `(n, 2l, l, 2k, 2m, 2p)` (`ze` reads `lt`, the
#  C₃₃₁₁ of an operator without major symmetry); in the Walpole basis
#  ℓ₁ = zz, ℓ₂ = me, ℓ₃ = ze/√2, ℓ₄ = √2 mz, ℓ₅ = T, ℓ₆ = L.
# =============================================================================

@inline _side(seq, k::Int, outer::Bool) = outer ? seq.outside[k] : seq.inside[k]

@inline _strain_moment(::Axisymmetric, s) = s[1]
@inline _strain_moment(::TransverseShear, s) = (s[1] - s[2]) / 2
@inline _strain_moment(::Antiplane, s) = s[1]
@inline _strain_moment(::TransverseConduction, s) = s[1]

# Average gradient over layer k: its material, plus the jump across its outer
# boundary when `external`, across its inner one when `internal`.
function _annulus_gradient(pb, seq, radii, k::Int, external::Bool, internal::Bool)
    r = radii[k]
    g = r * _strain_moment(pb, _side(seq, k, external))
    k == 1 && return g / r^2
    rᵢ = radii[k - 1]
    return (g - rᵢ * _strain_moment(pb, _side(seq, k - 1, !internal))) / (r^2 - rᵢ^2)
end

"""
    _whole_strain(sol, external) -> (; zz, ze, mz, me, T, L)

Average strain of the cylinder per unit remote strain, read on the matrix side
of ``r_N`` when `external` (the outer interface belongs to the cylinder) and on
its inner side otherwise.
"""
function _whole_strain(sol, external::Bool)
    N = length(sol.radii)
    R = sol.radii[N]
    𝟙 = one(sol.L₀.k)
    return (;
        zz = 𝟙, ze = zero(𝟙),
        mz = _side(sol.axial, N, external)[1] / R,
        me = _side(sol.dilatation, N, external)[1] / R,
        T = _strain_moment(TransverseShear(), _side(sol.shear, N, external)) / R,
        L = _side(sol.antiplane, N, external)[1] / R,
    )
end

"""
    _layer_strain(sol, k, external, internal) -> (; zz, ze, mz, me, T, L)

Average strain of layer `k` per unit remote strain.
"""
function _layer_strain(sol, k::Int, external::Bool, internal::Bool)
    𝟙 = one(sol.L₀.k)
    return (;
        zz = 𝟙, ze = zero(𝟙),
        mz = _annulus_gradient(Axisymmetric(), sol.axial, sol.radii, k, external, internal),
        me = _annulus_gradient(Axisymmetric(), sol.dilatation, sol.radii, k, external, internal),
        T = _annulus_gradient(TransverseShear(), sol.shear, sol.radii, k, external, internal),
        L = _annulus_gradient(Antiplane(), sol.antiplane, sol.radii, k, external, internal),
    )
end

# Axial stress carried by the membranes, per unit area of the section: the
# line force σˢ_zz = (κˢ − μˢ) u_r / r + (κˢ + μˢ) ε on each circle of radius r.
_axial_membrane(::AbstractInterface, r, u, ε) = zero(u)
_axial_membrane(itf::MembraneInterface, r, u, ε) =
    2 * ((itf.κs - itf.μs) * u + (itf.κs + itf.μs) * r * ε)

function _axial_stress(sol, seq, ε, external::Bool)
    N = length(sol.radii)
    R² = sol.radii[N]^2
    layers = sum(
        sol.fractions[k] * (2 * (sol.Ls[k].lt * seq.amps[k][1]) + sol.Ls[k].n * ε) for k in 1:N
    )
    surfaces = sum(
        _axial_membrane(sol.interfaces[k], sol.radii[k], seq.inside[k][1], ε) / R²
            for k in 1:(external ? N : N - 1); init = zero(layers)
    )
    return layers + surfaces
end

"""
    _whole_stress(sol, external) -> (; zz, ze, mz, me, T, L)

Average stress of the cylinder per unit remote strain: the stress of the
layers, plus the surface stress of the membranes, that of the outer one when
`external`. The in-plane components are read on the traction at ``r_N``; the
axial stress, which no traction on the lateral surface gives, is summed over
the layers and the membranes.
"""
function _whole_stress(sol, external::Bool)
    N = length(sol.radii)
    𝟙 = one(sol.L₀.k)
    s = _side(sol.shear, N, external)
    return (;
        zz = _axial_stress(sol, sol.axial, 𝟙, external),
        ze = _axial_stress(sol, sol.dilatation, zero(𝟙), external),
        mz = _side(sol.axial, N, external)[2],
        me = _side(sol.dilatation, N, external)[2],
        T = (s[3] - s[4]) / 2,
        L = 2 * _side(sol.antiplane, N, external)[2],
    )
end

# Entries of a stiffness, and the composition C ∘ A of two operators.
_stiffness_entries(L) = (; zz = L.n, ze = 2 * L.lt, mz = L.l, me = 2 * L.k, T = 2 * L.m, L = 2 * L.p)

_compose(C, A) = (;
    zz = C.zz * A.zz + C.ze * A.mz, ze = C.zz * A.ze + C.ze * A.me,
    mz = C.mz * A.zz + C.me * A.mz, me = C.mz * A.ze + C.me * A.me,
    T = C.T * A.T, L = C.L * A.L,
)

_minus(B, C) = map(-, B, C)

# The average stress of the material of layer k, its stiffness on the left of
# its average strain; in conduction, its average of k·∇T.
_layer_stress(sol, k::Int) =
    _compose(_stiffness_entries(sol.Ls[k]), _layer_strain(sol, k, false, false))

"""
    _contribution(sol, external) -> (; zz, ze, mz, me, T, L)

Contribution of the cylinder, ``\\mathbb N = \\langle\\boldsymbol\\sigma\\rangle - \\mathbb C_0\\circ\\langle\\boldsymbol\\varepsilon\\rangle``
per unit remote strain.
"""
_contribution(sol, external::Bool) = _minus(
    _whole_stress(sol, external),
    _compose(_stiffness_entries(sol.L₀), _whole_strain(sol, external)),
)

# ── Conduction ───────────────────────────────────────────────────────────────

_axial_surface_flux(::AbstractInterface, r) = zero(r)
_axial_surface_flux(itf::SurfaceConductiveInterface, r) = 2 * itf.conductance * r

"""
    _whole_gradient(sol, external) -> (; t, a)
    _whole_flux(sol, external) -> (; t, a)

Transverse (`t`) and axial (`a`) entries of the average gradient of the
cylinder and of its average ``\\boldsymbol k\\cdot\\nabla T`` per unit remote
gradient. The axial gradient is uniform; the axial flux adds the one carried
along each conducting surface, ``2 k^{\\mathrm s} r_k / r_N^2``.
"""
function _whole_gradient(sol, external::Bool)
    N = length(sol.radii)
    return (; t = _side(sol.transverse, N, external)[1] / sol.radii[N], a = one(sol.K₀.t))
end

_layer_gradient(sol, k::Int, external::Bool, internal::Bool) =
    (; t = _annulus_gradient(TransverseConduction(), sol.transverse, sol.radii, k, external, internal), a = one(sol.K₀.t))

function _whole_flux(sol, external::Bool)
    N = length(sol.radii)
    𝟙 = one(sol.K₀.t)
    layers = sum(sol.fractions[k] * sol.Ks[k].a for k in 1:N)
    surfaces = sum(
        _axial_surface_flux(sol.interfaces[k], sol.radii[k]) for k in 1:(external ? N : N - 1);
        init = zero(eltype(sol.radii))
    )
    return (; t = _side(sol.transverse, N, external)[2], a = layers + (surfaces / sol.radii[N]^2) * 𝟙)
end

function _layer_flux(sol, k::Int)
    G = _layer_gradient(sol, k, false, false)
    return (; t = sol.Ks[k].t * G.t, a = sol.Ks[k].a * G.a)
end

function _conduction_contribution(sol, external::Bool)
    G = _whole_gradient(sol, external)
    Q = _whole_flux(sol, external)
    return (; t = Q.t - sol.K₀.t * G.t, a = Q.a - sol.K₀.a * G.a)
end
