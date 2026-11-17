# =============================================================================
#  conductivity.jl — isotropic `n`-layer spherical inclusion, 2nd-order
#  transport problem (thermal / electric / Darcy).
#
#  Under a remote uniform gradient `∇T∞` (Y₁-harmonic direction field),
#  the temperature field inside a concentric-sphere laminate takes the
#  form
#
#      T(r, θ) = [A_k · r + B_k / r²] · (r̂ · ê∇),
#
#  where ê∇ is the direction of `∇T∞`.  The state vector at radius r is
#  `s(r) = (T̂(r), q̂_n(r))` with
#      T̂ = A r + B/r²                    (temperature amplitude),
#      q̂_n = -k (A - 2 B/r³)             (normal flux amplitude).
#
#  Intra-layer transfer  `s(r_out) = T_cond(r_out, r_in; k) · s(r_in)`
#  comes from `M(r_out) · M(r_in)⁻¹` with `M(r) = [r  1/r²; -k  2k/r³]`
#  and `det M(r) = 3k/r²`.
#
#  The same construction holds for the harmonic of any degree n, with the
#  radial functions rⁿ and r^-(n+1): every function below takes the degree as
#  an optional leading argument, the uniform gradient being n = 1.  The
#  response of the sphere to one harmonic (`harmonic_response`) is what a
#  scheme resolving the field around each inclusion needs.
#
#  Interface jumps (q̂_n is the outward flux, so heat flowing out of the inner
#  side across a resistance leaves it hotter than the outer side):
#     Perfect                       J = I
#     Kapitza(ρ)                    J = [1  -ρ ; 0  1]       [T] = -ρ q_n
#     SurfaceConductive(ks)         J = [1  0 ; -n(n+1) ks/r²  1]  with n=1 ⇒ 2ks/r².
#
#  Per-layer gradient concentration `α_k = A_k / A_∞`.
# =============================================================================

"""
    _cond_layer_transfer([n,] r_out, r_in, k_iso) -> Matrix(2×2)

Intra-layer field transfer of the degree-`n` harmonic (`n = 1`, the uniform
gradient, when omitted) between radii `r_in` and `r_out` in a layer of scalar
conductivity `k_iso`: ``\\mathbf{M}_n(r_{\\mathrm{out}})\\,\\mathbf{M}_n(r_{\\mathrm{in}})^{-1}``
with

```math
\\mathbf{M}_n(r) = \\begin{pmatrix} r^n & r^{-(n+1)} \\ -n\\,k\\,r^{n-1} & (n+1)\\,k\\,r^{-(n+2)} \\end{pmatrix},
\\qquad \\det\\mathbf{M}_n = \\frac{(2n+1)\\,k}{r^2} .
```

The inverse is written in closed form, so that a single division by `k_iso`
appears.
"""
@inline function _cond_layer_transfer(n::Integer, r_out, r_in, k_iso)
    T = promote_type(typeof(r_out), typeof(r_in), typeof(k_iso))
    ro = T(r_out); ri = T(r_in); k = T(k_iso)

    inv_k = 1 / k
    riⁿ = ri^n
    # M_n(ri)⁻¹ = [a b; c d]
    a = (n + 1) / ((2n + 1) * riⁿ)
    b = -ri * inv_k / ((2n + 1) * riⁿ)
    c = n * riⁿ * ri / (2n + 1)
    d = riⁿ * ri * ri * inv_k / (2n + 1)

    roⁿ = ro^n
    reg = roⁿ                         # r^n
    irr = 1 / (roⁿ * ro)              # r^-(n+1)
    dreg = -k * n * roⁿ / ro          # -n k r^(n-1)
    dirr = k * (n + 1) * irr / ro     # (n+1) k r^-(n+2)

    return T[
        reg * a + irr * c    reg * b + irr * d
        dreg * a + dirr * c  dreg * b + dirr * d
    ]
end
@inline _cond_layer_transfer(r_out, r_in, k_iso) = _cond_layer_transfer(1, r_out, r_in, k_iso)

"""
    _cond_seed_state([n,] r_1, k_1) -> Vector(2)

Seed state at ``r_1^-`` for a unit amplitude ``A_1 = 1`` in layer 1
(core).  ``B_1 = 0`` from finiteness at the origin, so ``\\hat T(r_1) = r_1^n``,
``\\hat q_n(r_1) = -n\\,k_1\\,r_1^{n-1}``.
"""
@inline function _cond_seed_state(n::Integer, r_1, k_1)
    T = promote_type(typeof(r_1), typeof(k_1))
    r = T(r_1)
    return T[r^n, -n * T(k_1) * r^(n - 1)]
end
@inline _cond_seed_state(r_1, k_1) = _cond_seed_state(1, r_1, k_1)

"""
    _cond_extract_AB([n,] r, k_iso, T̂, q̂n) -> (A, B)

Extract the local-expansion coefficients ``(A, B)`` of ``\\hat T = A\\,r^n + B\\,r^{-(n+1)}``
in a layer of conductivity `k_iso` given the state ``(\\hat T, \\hat q_n)`` at radius `r`.
"""
@inline function _cond_extract_AB(n::Integer, r, k_iso, T̂, q̂n)
    T = promote_type(typeof(r), typeof(k_iso), typeof(T̂), typeof(q̂n))
    Tr = T(r)
    Trⁿ = Tr^n
    Tk = T(k_iso)
    # Impermeable region (k = 0, e.g. a solid aggregate core): the normal flux
    # is identically zero and carries no information on (A, B); the `q̂n/k` terms
    # are the indeterminate `0/0`.  For the core, regularity at the origin fixes
    # `B = 0`, so `A = T̂ / rⁿ`.  (An impermeable shell: `_cond_layer_AB`.)
    if iszero(Tk)
        return T(T̂) / Trⁿ, zero(T)
    end
    A = ((n + 1) * T(T̂) - Tr * T(q̂n) / Tk) / ((2n + 1) * Trⁿ)
    B = (n * T(T̂) + Tr * T(q̂n) / Tk) * Trⁿ * Tr / (2n + 1)
    return A, B
end
@inline _cond_extract_AB(r, k_iso, T̂, q̂n) = _cond_extract_AB(1, r, k_iso, T̂, q̂n)

# An impermeable layer: a conductivity that is exactly zero. Never true for a
# symbol, which does not compare.
@inline _is_impermeable(k) = is_hard_numeric(typeof(k)) && iszero(k)
# The Volterra matrix of a conductivity: impermeable at every instant.
@inline _is_impermeable(k::AbstractMatrix) = all(_is_impermeable, k)

# Two impermeable layers side by side: the field of the inner one is defined by
# no conducting neighbor inside it.
_adjacent_impermeable(name::AbstractString, j::Int) = throw(
    ArgumentError(
        "layers $(j - 1) and $j of the $name both have a zero conductivity: the field of " *
            "an impermeable layer is defined by a conducting neighbor inside it; merge the " *
            "two layers"
    )
)

"""
    _cond_layer_AB([n,] radii, k_layers, j, T̂, q̂n) -> (A, B)

Coefficients ``(A, B)`` of layer `j` from its state at ``r_j^-``. In an
impermeable shell (``k_j = 0``, ``j > 1``) no flux crosses, the layers inside it
are shielded, and its own field is the limit of a vanishing conductivity: it
vanishes on its inner face, ``\\hat T = A\\,(r^n - r_{j-1}^{2n+1}/r^{n+1})``.
"""
@inline function _cond_layer_AB(n::Integer, radii, k_layers, j::Int, T̂, q̂n)
    j > 1 && _is_impermeable(k_layers[j]) ||
        return _cond_extract_AB(n, radii[j], k_layers[j], T̂, q̂n)
    ρ = radii[j - 1]^(2n + 1)
    A = T̂ * radii[j]^(n + 1) / (radii[j]^(2n + 1) - ρ)
    return A, -A * ρ
end
@inline _cond_layer_AB(radii, k_layers, j::Int, T̂, q̂n) = _cond_layer_AB(1, radii, k_layers, j, T̂, q̂n)

# ── Conductivity interface jump matrices ────────────────────────────────────

"""
    _cond_interface_T([n,] intf, k_iso, r) -> Matrix(2×2)

Jump matrix for the degree-`n` conductivity state vector
``(\\hat T, \\hat q_n)`` at an interface of type `intf` located at radius `r`
(`n = 1` when omitted).
"""
function _cond_interface_T end

_cond_interface_T(intf::AbstractInterface, k_iso, r) = _cond_interface_T(1, intf, k_iso, r)

function _cond_interface_T(::Integer, ::PerfectInterface, k_iso, r)
    T = promote_type(typeof(k_iso), typeof(r))
    return T[one(T) zero(T); zero(T) one(T)]
end

# The state carries the outward flux q̂_n = -k ∂T̂/∂r, so the temperature DROPS
# across a resistance in the direction of the flux: T̂⁺ = T̂⁻ - ρ q̂_n.  With
# `+ρ` the interface was a negative resistance, which raised the conductance of
# the sphere it was meant to lower (and diverged at ρ = r/k).
function _cond_interface_T(::Integer, intf::KapitzaInterface, k_iso, r)
    T = promote_type(eltype(intf), typeof(k_iso), typeof(r))
    return T[one(T) -T(intf.resistance); zero(T) one(T)]
end

# The surface current kˢ gradₛ T has the surface divergence -n(n+1) kˢ T̂ / r² on
# the degree-n harmonic, which the normal flux balances.
function _cond_interface_T(n::Integer, intf::SurfaceConductiveInterface, k_iso, r)
    T = promote_type(eltype(intf), typeof(k_iso), typeof(r))
    Tr² = T(r) * T(r)
    return T[one(T) zero(T); -n * (n + 1) * T(intf.conductance) / Tr² one(T)]
end

# ── Propagation and concentration ────────────────────────────────────────────

# Cached tuple of per-layer scalar conductivities, built once from the
# `sphere.moduli` field.
@inline function _cond_layer_moduli(sphere::LayeredSphere{T, N}) where {T, N}
    return ntuple(k -> _iso_scalar(layer_modulus(sphere, k)), Val(N))
end

"""
    _cond_state_seq([n,] sphere, k₀) -> NTuple{N, state⁻}, state⁺_N

Propagate the degree-`n` conductivity state vector from the core outward
(`n = 1` when omitted).
"""
function _cond_state_seq(n::Integer, sphere::LayeredSphere{T, N}, k₀) where {T, N}
    k_layers = _cond_layer_moduli(sphere)
    TP = promote_type(
        T, typeof(k₀),
        ntuple(k -> typeof(k_layers[k]), N)...,
        interfaces_eltype(sphere.interfaces)
    )
    radii = sphere.radii

    s = _cond_seed_state(n, TP(radii[1]), TP(k_layers[1]))

    inside_states = Vector{Vector{TP}}(undef, N)
    for k in 1:N
        inside_states[k] = copy(s)
        intf = layer_interface(sphere, k)
        Tint = _cond_interface_T(n, intf, TP(k_layers[k]), TP(radii[k]))
        s = Tint * s
        if k < N
            s = _cond_enter_layer!(n, inside_states, s, radii, k_layers, k + 1)
        end
    end
    return inside_states, s
end
_cond_state_seq(sphere::LayeredSphere, k₀) = _cond_state_seq(1, sphere, k₀)

# The state at the outer face of layer `j`, from the state `s` at its inner
# face. An impermeable shell restarts the sequence: the states inside it are
# zero (those layers are shielded) and its own field has a unit amplitude.
function _cond_enter_layer!(
        n::Integer, inside_states, s::Vector{TP}, radii, k_layers, j::Int
    ) where {TP}
    _is_impermeable(k_layers[j]) || return _cond_layer_transfer(
        n, TP(radii[j]), TP(radii[j - 1]), TP(k_layers[j])
    ) * s
    _is_impermeable(k_layers[j - 1]) && _adjacent_impermeable("LayeredSphere", j)
    for i in 1:(j - 1)
        inside_states[i] = zeros(TP, 2)
    end
    rj = TP(radii[j])
    return TP[rj^n - TP(radii[j - 1])^(2n + 1) / rj^(n + 1), zero(TP)]
end

"""
    harmonic_response(sphere::LayeredSphere, K₀, n) -> Number

Response of a layered sphere to the harmonic of degree `n` in a matrix of
isotropic conductivity `K₀`. Around the sphere the field of the matrix is a sum
of terms ``a_n^m\\,y_n^m + b_n^m\\,Y_n^m``, the regular and decaying solid
harmonics of [`regular_solid_harmonics`](@ref MeanFieldHomogenization.regular_solid_harmonics) and
[`irregular_solid_harmonics`](@ref MeanFieldHomogenization.irregular_solid_harmonics); the function returns
``\\zeta_n`` such that ``(n-m)!\\,(n+m)!\\;b_n^m = \\zeta_n\\,a_n^m``, that is, the
ratio of the radial functions ``r^{-(n+1)}`` and ``r^n``. The sphere is
isotropic, so ``\\zeta_n`` does not depend on the order ``m``. For a homogeneous sphere of radius
``R`` and conductivity ``k_1`` behind a perfect interface it is
``-n\\,(k_1 - k_0)\\,R^{2n+1} / (n\\,k_1 + (n+1)\\,k_0)``. The layers and the
interface models enter through the same transfer matrices as for the uniform
gradient, which is the case `n = 1`.
"""
function harmonic_response(sphere::LayeredSphere{T, N}, K₀::TensND.AbstractTens{2}, n::Integer) where {T, N}
    n ≥ 1 || throw(ArgumentError("harmonic_response: the degree must be at least 1, got $n"))
    k₀ = _iso_scalar(K₀)
    _, s = _cond_state_seq(n, sphere, k₀)
    A, B = _cond_extract_AB(n, sphere.radii[N], k₀, s[1], s[2])
    return B / A
end

"""
    _cond_concentration(sphere, k₀) -> NTuple{N, TP}

Per-layer gradient concentration ``\\alpha_k = A_k/A_\\infty`` under a remote
uniform gradient.  Reduces, for ``N = 1``, to the classical formula
``\\alpha_1 = 3k_0/(2k_0 + k_1)`` for a sphere inclusion.
"""
function _cond_concentration(sphere::LayeredSphere{T, N}, k₀) where {T, N}
    MFH_Core._bump!(MFH_Core.LAYER_RECURRENCES)
    k_layers = _cond_layer_moduli(sphere)
    TP = promote_type(
        T, typeof(k₀),
        ntuple(k -> typeof(k_layers[k]), N)...,
        interfaces_eltype(sphere.interfaces)
    )
    inside_states, s_matrix = _cond_state_seq(sphere, k₀)
    radii = sphere.radii

    A_inf, _ = _cond_extract_AB(TP(radii[N]), TP(k₀), s_matrix[1], s_matrix[2])
    inv_A_inf = one(TP) / A_inf

    return ntuple(N) do k
        (A_k, _) = _cond_layer_AB(radii, k_layers, k, inside_states[k][1], inside_states[k][2])
        A_k * inv_A_inf
    end
end

"""
    _cond_surface_flux(sphere, k₀) -> flux amplitude

Extra macroscopic flux carried **tangentially along the surface-conductive
interfaces** of the composite sphere, per unit remote gradient.  A
`SurfaceConductiveInterface(kₛ)` at radius ``r`` supports a surface current
``k^{\\mathrm s}\\,\\underline{\\operatorname{grad}}_{\\mathrm s}\\,T``; its contribution to the volume-averaged flux of the whole sphere
(radius ``R_N``) is, per unit ``A_\\infty``,

```math
\\frac{1}{V}\\oint k^{\\mathrm s}\\,\\underline{\\operatorname{grad}}_{\\mathrm s}\\,T\\,\\mathrm{d}S = \\frac{2k^{\\mathrm s}\\,\\hat T(r)\\,r}{R_N^3},
```

with ``\\hat T(r)`` the (continuous) temperature amplitude at the interface.  For a
single impermeable core coated by a surface-conductive shell this reduces to
``2k^{\\mathrm s}\\alpha/R``, i.e. the surface conductance is equivalent to adding ``2k^{\\mathrm s}/R`` to
the enclosed conductivity — reproducing Echoes' `DUALDISC` transmissivity.
The average **gradient** (concentration ``\\alpha_k``) is unaffected: only the flux
picks up the surface term. With `external = false` the outer interface is left
to the matrix (see [`gradient_gradient_concentration`](@ref)).
"""
function _cond_surface_flux(sphere::LayeredSphere{T, N}, k₀; external::Bool = true) where {T, N}
    k_layers = _cond_layer_moduli(sphere)
    TP = promote_type(
        T, typeof(k₀), ntuple(k -> typeof(k_layers[k]), N)...,
        interfaces_eltype(sphere.interfaces)
    )
    inside_states, s_matrix = _cond_state_seq(sphere, k₀)
    radii = sphere.radii
    A_inf, _ = _cond_extract_AB(TP(radii[N]), TP(k₀), s_matrix[1], s_matrix[2])
    RN³ = TP(radii[N])^3
    total = zero(TP)
    for k in 1:(external ? N : N - 1)
        intf = layer_interface(sphere, k)
        if intf isa SurfaceConductiveInterface
            ks = TP(intf.conductance)
            T̂ = inside_states[k][1]
            total += 2 * ks * T̂ * TP(radii[k]) / (RN³ * A_inf)
        end
    end
    return total
end

"""
    _effective_conductivity(sphere, k₀) -> k_eff

Average flux of the composite sphere per unit remote gradient,
``\\sum_k f_k\\,k_k\\,\\alpha_k`` plus the surface-conduction flux [`_cond_surface_flux`](@ref) of any
dual (surface-conductive) interface — the scalar of [`flux_gradient_concentration`](@ref),
not an effective conductivity: that one is its ratio to the average gradient,
[`gradient_gradient_concentration`](@ref).
"""
function _effective_conductivity(sphere::LayeredSphere{T, N}, k₀) where {T, N}
    α = _cond_concentration(sphere, k₀)
    k_layers = _cond_layer_moduli(sphere)
    f = ntuple(k -> layer_volume_fraction(sphere, k), Val(N))
    return sum(f[k] * k_layers[k] * α[k] for k in 1:N) + _cond_surface_flux(sphere, k₀)
end
