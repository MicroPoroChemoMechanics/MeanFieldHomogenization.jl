# =============================================================================
#  conductivity.jl — confocal spheroidal-harmonic transfer-matrix
#  recurrence for the N-layer spheroid conduction problem
#  (Barthélémy & Bignonnet, IJES 2020, §2).
#
#  Mirrors `LayeredSpheres/conductivity.jl`'s 2×2 state-vector transfer
#  for the sphere, but the spheroid's imperfect (LC/HC) interfaces
#  COUPLE DIFFERENT HARMONIC DEGREES (the sphere's do not), so the
#  "state" here is a `2𝒩`-vector of series coefficients
#  `Xℓ = [Aℓ; Bℓ]` (𝒩 = `Nseries` truncated odd degrees `1,3,…,2𝒩-1`,
#  eq:AlBlXl) rather than a 2-vector, and the transfer "matrix" `Rℓ` is
#  `2𝒩×2𝒩` (eq:aximat) built from the diagonal blocks `𝒥(k,q)`
#  (eq:matJ) plus, for LC/HC interfaces, the full coupling perturbation
#  `δ𝒥^{LC/HC}` (eq:axiLCdJ/axiHCdJ, eq:transLCdJ/transHCdJ) assembled
#  from the `I, J, K, L` matrices of `coupling.jl`.
#
#  Axial (`trans = false`, order `m = 0`, `H = H₃ê₃`) and transverse
#  (`trans = true`, order `m = 1`, `H = H₁ê₁`) problems share the exact
#  same machinery, differing only in which Legendre table (`P0/Q0` vs
#  `P1/Q1`) and which coupling matrix (`I` vs `J` for LC,
#  `J` vs `K+L/(q²-1)` for HC) enter — see `_transition`.
#
#  Convention: `KapitzaInterface(resistance)` ↔ paper's `αℓ` (LC,
#  temperature-jump resistance, entering the TEMPERATURE block);
#  `SurfaceConductiveInterface(conductance)` ↔ paper's `βℓ` (HC,
#  flux-jump surface conductance, entering the FLUX block) — the SAME
#  convention as `LayeredSphere`'s interfaces (not to be confused with
#  the inverse convention some `echoes` call sites use for the raw
#  `interf_prop` scalar; see the theory page).
# =============================================================================

@inline function _spheroid_layer_moduli(s::LayeredSpheroid{T, N}) where {T, N}
    return ntuple(k -> first(TensND.get_data(layer_modulus(s, k))), Val(N))
end

"""
    _as_scalar_k(k₀) -> Number

Normalize a matrix conductivity argument to its isotropic scalar,
accepting either a bare `Number` or a `TensISO{2,3}` (the natural way
to call the public API, matching `gradient_gradient_concentration`'s own `K₀`
convention).
"""
@inline _as_scalar_k(k₀::Number) = k₀
@inline _as_scalar_k(k₀) = first(TensND.get_data(k₀))

"""
    _spheroid_tables(s::LayeredSpheroid, m, degrees) -> NamedTuple

Per-layer scalar conductivities and the Legendre values and derivatives
``P_n^m``, ``Q_n^m`` at the `degrees` of the order `m`, each evaluated once at
layer ``k``'s own confocal parameter ``q_k``.
"""
function _spheroid_tables(s::LayeredSpheroid{T, N, Q}, m::Int, degrees) where {T, N, Q}
    k_layers = _spheroid_layer_moduli(s)
    idx = [n + 1 for n in degrees]
    tabs = [legendre_order_q(s.q[k], m, maximum(degrees)) for k in 1:N]
    P = [t[1][idx] for t in tabs]
    dP = [t[2][idx] for t in tabs]
    Qf = [t[3][idx] for t in tabs]
    dQ = [t[4][idx] for t in tabs]
    return (; k_layers, P, dP, Q = Qf, dQ, m, degrees)
end

"""
    _J_ext(k, P, Qf, dP, dQ) -> Matrix

The ``2\\mathcal{N}\\times 2\\mathcal{N}`` diagonal block matrix ``\\mathcal{J}(k,q)`` (eq:matJ),
with ``P``, ``Q``, ``P'``, ``Q'`` the arguments `P`, `Qf`, `dP`, `dQ`:

```math
\\mathcal{J}(k,q) = \\begin{pmatrix}
\\mathrm{diag}(P) & \\mathrm{diag}(Q)\\\\
k\\,\\mathrm{diag}(P') & k\\,\\mathrm{diag}(Q')
\\end{pmatrix}.
```
"""
function _J_ext(k, P::Vector{Qx}, Qf::Vector{Qx}, dP::Vector{Qx}, dQ::Vector{Qx}) where {Qx}
    𝒩 = length(P)
    Tm = promote_type(typeof(k), Qx)
    M = zeros(Tm, 2𝒩, 2𝒩)
    for r in 1:𝒩
        M[r, r] = P[r]
        M[r, 𝒩 + r] = Qf[r]
        M[𝒩 + r, r] = k * dP[r]
        M[𝒩 + r, 𝒩 + r] = k * dQ[r]
    end
    return M
end

"""
    _J_int(k, coef, P, Qf, dP, dQ, M, dual, degrees, m) -> Matrix

``\\mathcal{J}(k,q) + \\delta\\mathcal{J}^{\\mathrm{LC/HC}}``: the diagonal block `_J_ext` plus the full
coupling perturbation of generic row/column term
``[\\delta\\mathcal{J}]_{rs} = c\\,\\frac{2n_r+1}{2}\\,\\frac{(n_r-m)!}{(n_r+m)!}\\,M_{rs}\\,\\mathcal{R}_{n_s}``,
``c`` = `coef`, ``n_r`` the ``r``-th of the `degrees` (``\\mathcal{R} = P, Q`` for HC,
``\\mathcal{R} = P', Q'`` for LC), the factor of ``M_{rs}`` being the inverse of the
norm of ``P_{n_r}^m`` on ``[-1, 1]`` (eq:axiLCdJ/axiHCdJ/transLCdJ/transHCdJ at the
orders 0 and 1). `dual = true` (HC) perturbs the FLUX (lower) block using
unprimed ``\\mathcal{R}``; `dual = false` (LC) perturbs the TEMPERATURE (upper) block
using primed ``\\mathcal{R}'``.
"""
function _J_int(
        k, coef, P::Vector{Qx}, Qf::Vector{Qx}, dP::Vector{Qx}, dQ::Vector{Qx},
        M::AbstractMatrix, dual::Bool, degrees, m::Int,
    ) where {Qx}
    𝒩 = length(P)
    # `coef` carries the interface parameter (Kapitza resistance / surface
    # conductance), so it must enter the element type: `eltype(_J_ext(...))`
    # alone knows only about `k` and the Legendre tables, and a `Dual` `coef`
    # written into a `Float64` buffer is a `MethodError` — which is what made
    # every sensitivity with respect to an interface parameter unreachable.
    J0 = _J_ext(k, P, Qf, dP, dQ)
    Tm = promote_type(eltype(J0), typeof(coef), eltype(M))
    J = convert(Matrix{Tm}, J0)
    MP = zeros(Tm, 𝒩, 𝒩)
    MQ = zeros(Tm, 𝒩, 𝒩)
    Rp, Rq = dual ? (P, Qf) : (dP, dQ)
    for (r, n) in enumerate(degrees)
        ci = coef * (2n + 1) / 2 * _factorial_ratio(real(Tm), n, m)
        for c in 1:𝒩
            MP[r, c] = M[r, c] * ci * Rp[c]
            MQ[r, c] = M[r, c] * ci * Rq[c]
        end
    end
    if dual
        @views J[(𝒩 + 1):(2𝒩), 1:𝒩] .+= MP
        @views J[(𝒩 + 1):(2𝒩), (𝒩 + 1):(2𝒩)] .+= MQ
    else
        @views J[1:𝒩, 1:𝒩] .+= MP
        @views J[1:𝒩, (𝒩 + 1):(2𝒩)] .+= MQ
    end
    return J
end

# (n − m)!/(n + m)! in the element type `T`, exact for a symbolic one.
_factorial_ratio(::Type{T}, n::Int, m::Int) where {T} = prod((one(T) / j for j in (n - m + 1):(n + m)); init = one(T))

"""
    _transition(s, layer, k_next, tables) -> Matrix

Transfer matrix ``R_\\ell = \\mathcal{J}(k_{\\ell+1}, q_\\ell)^{-1}\\,\\bigl(\\mathcal{J}(k_\\ell, q_\\ell) + \\delta\\mathcal{J}\\bigr)``
(eq:aximat) across the interface at the outer boundary of `layer`, at the
order and degrees of `tables`. `k_next` is the conductivity on the other
side (next layer's, or the matrix ``k_0`` if `layer == N`).
"""
function _transition(s::LayeredSpheroid, layer::Int, k_next, tables)
    q = s.q[layer]
    kℓ = tables.k_layers[layer]
    intf = layer_interface(s, layer)
    P, Qf, dP, dQ = tables.P[layer], tables.Q[layer], tables.dP[layer], tables.dQ[layer]
    Jint = _J_interface(intf, s, q, kℓ, P, Qf, dP, dQ, tables.m, tables.degrees)
    Jext_next = _J_ext(k_next, P, Qf, dP, dQ)
    return Jext_next \ Jint
end

# The inner side of the transition across the interface `intf` of the layer of
# confocal coordinate `q`: the plain exterior operator for a perfect interface,
# the one carrying the jump of a Kapitza or a surface-conductive interface.
_J_interface(::PerfectInterface, s, q, kℓ, P, Qf, dP, dQ, m, degrees) = _J_ext(kℓ, P, Qf, dP, dQ)
function _J_interface(intf::KapitzaInterface, s, q, kℓ, P, Qf, dP, dQ, m, degrees)
    M = _coupling_quadrature(q, m, degrees, Val(:lc))
    coef = kℓ * intf.resistance / s.c * sqrt(q^2 - 1)
    return _J_int(kℓ, coef, P, Qf, dP, dQ, M, false, degrees, m)
end
function _J_interface(intf::SurfaceConductiveInterface, s, q, kℓ, P, Qf, dP, dQ, m, degrees)
    M = _coupling_quadrature(q, m, degrees, Val(:hc))
    coef = intf.conductance / (s.c * sqrt(q^2 - 1))
    return _J_int(kℓ, coef, P, Qf, dP, dQ, M, true, degrees, m)
end
_J_interface(intf, args...) =
    throw(ArgumentError("LayeredSpheroid: unsupported interface type $(typeof(intf))"))

# Product S_N = R_N ⋯ R_1 of the transfer matrices of the layers, from the core
# to the matrix, at the order and degrees of `tables`, in the element type
# that holds the layer conductivities, the interface parameters, `q` and `k₀`.
function _spheroid_transfer(s::LayeredSpheroid{T, N, Q}, k₀, tables) where {T, N, Q}
    # The layer conductivities and the interface parameters belong in this
    # promotion just as much as `q` and `k₀` do: `_J_ext`/`_J_int` widen locally
    # to whatever `kℓ` and `eltype(intf)` are, and a narrower buffer here then
    # refuses to store the widened transfer matrix. That is what made a single
    # `ForwardDiff.Dual` layer conductivity — or any sensitivity with respect to
    # a Kapitza resistance or a surface conductance — a `MethodError`.
    Qk = promote_type(
        Q, typeof(k₀),
        ntuple(ℓ -> typeof(tables.k_layers[ℓ]), Val(N))...,
        interfaces_eltype(s.interfaces)
    )
    S = Vector{Matrix{Qk}}(undef, N)
    for ℓ in 1:N
        k_next = ℓ < N ? tables.k_layers[ℓ + 1] : k₀
        R = _transition(s, ℓ, k_next, tables)
        S[ℓ] = ℓ == 1 ? convert(Matrix{Qk}, R) : R * S[ℓ - 1]
    end
    return S
end

"""
    spheroid_state_sequence(s, k₀, trans) -> Vector{Vector}

Series coefficient vectors ``X_\\ell = [A_\\ell; B_\\ell]`` (eq:AlBlXl), ``\\ell = 1, \\dots, N+1``
(the ``(N+1)``-th being the matrix), for the axial (`trans = false`) or
transverse (`trans = true`) problem in an isotropic matrix ``k_0``.
``B_1 = 0`` (core regularity, eq:axiBCbi) and
``A_{N+1} = (\\pm 1, 0, \\dots, 0)`` (unit remote field, ``+`` axial / ``-``
transverse, eq:axiBCai/eq:transBCai) are imposed exactly.
"""
function spheroid_state_sequence(s::LayeredSpheroid{T, N, Q}, k₀raw, trans::Bool) where {T, N, Q}
    k₀ = _as_scalar_k(k₀raw)
    𝒩 = s.Nseries
    tables = _spheroid_tables(s, trans ? 1 : 0, 1:2:(2𝒩 - 1))
    S = _spheroid_transfer(s, k₀, tables)
    Qk = eltype(S[N])

    Ainf = zeros(Qk, 𝒩)
    Ainf[1] = trans ? -one(Qk) : one(Qk)
    X1 = zeros(Qk, 2𝒩)
    @views X1[1:𝒩] = S[N][1:𝒩, 1:𝒩] \ Ainf

    X = Vector{Vector{Qk}}(undef, N + 1)
    X[1] = X1
    for ℓ in 1:N
        X[ℓ + 1] = S[ℓ] * X1
    end

    # The far-field condition IS `A_matrix = Ainf`: apart from the imposed
    # uniform remote field at degree 1, every GROWING amplitude in the matrix
    # vanishes identically, or the temperature would blow up at infinity.
    # `X1` was solved to make `S[N][1:𝒩,1:𝒩] * X1[1:𝒩]` equal `Ainf`, so
    # recomputing that product only reintroduces the linear solve's residual —
    # `O(1e-17)` instead of the exact zero.  Harmless in the near field, fatal
    # away from the particle: those residues multiply `P_{2r-1}(q) ~ q^{2r-1}`,
    # so at `Nseries = 12` and 500 particle radii a `1e-26` amplitude is
    # amplified past `1e40`, and raising `Nseries` made it worse rather than
    # better.  Write the boundary condition down instead of recovering it.
    @views X[N + 1][1:𝒩] .= Ainf
    return X
end

# ── Shape functions 𝒯ₐ, 𝒯ₜ, 𝒰ₐ, 𝒰ₜ (eq:TaTt, eq:VaVt) ─────────────────────

@inline _shape_Ta(q) = acoth(q) - 1 / q
@inline _shape_Tt(q) = acoth(q) - q / (q^2 - 1)
@inline _shape_Ua(q) = acoth(q) - q / (q^2 - 1)
@inline _shape_Ut(q) = acoth(q) + (2 - q^2) / (q * (q^2 - 1))

"""
    spheroid_ba_ratios(s, k₀) -> (ba_axial, ba_trans)

The two ratios ``b^0_{N+1,1}/a^0_{N+1,1}`` and ``b^1_{N+1,1}/a^1_{N+1,1}``
(eq:axiasb/eq:transasb) driving the volume-averaged concentration
tensors. Kept in their native (possibly complex, for the oblate
substitution ``q = \\mathrm{i}\\tau``) type — casting to real is only valid on the
FINAL shape-function product (``(b/a)\\,\\mathcal{T}(q_N)`` or ``(b/a)\\,\\mathcal{U}(q_N)``, done in
`spheroid_gradient_gradient` / `spheroid_flux_gradient` /
`_spheroid_concentration`), never on ``b/a`` alone.
"""
function spheroid_ba_ratios(s::LayeredSpheroid{T, N}, k₀) where {T, N}
    𝒩 = s.Nseries
    Xa = spheroid_state_sequence(s, k₀, false)
    Xt = spheroid_state_sequence(s, k₀, true)
    ba_a = Xa[end][𝒩 + 1] / Xa[end][1]
    ba_t = Xt[end][𝒩 + 1] / Xt[end][1]
    return ba_a, ba_t
end

"""
    spheroid_gradient_gradient(s, k₀) -> (αt, αa)

Real axial/transverse gradient-concentration scalars
``\\alpha_a = 1 + (b/a)|_a\\,\\mathcal{T}_a(q_N)``, ``\\alpha_t = 1 + (b/a)|_t\\,\\mathcal{T}_t(q_N)``
(eq:avgradTN/eq:transavgradTN, unit remote field), such that
``\\langle\\underline{\\operatorname{grad}}\\,T\\rangle_\\Omega = \\mathrm{diag}(\\alpha_t, \\alpha_t, \\alpha_a)\\cdot\\underline{\\operatorname{grad}}\\,T^{\\infty}`` in the spheroid's own (axis-aligned)
frame.
"""
function spheroid_gradient_gradient(s::LayeredSpheroid{T, N}, k₀) where {T, N}
    qN = s.q[N]
    ba_a, ba_t = spheroid_ba_ratios(s, k₀)
    αa = real(1 + ba_a * _shape_Ta(qN))
    αt = real(1 + ba_t * _shape_Tt(qN))
    return αt, αa
end

"""
    spheroid_flux_gradient(s, k₀) -> (βt, βa)

Real axial/transverse flux-concentration scalars
``\\beta_a = 1 + (b/a)|_a\\,\\mathcal{U}_a(q_N)``, ``\\beta_t = 1 + (b/a)|_t\\,\\mathcal{U}_t(q_N)``
(eq:avuN/eq:transavuN), such that
``\\langle\\boldsymbol{K}\\cdot\\underline{\\operatorname{grad}}\\,T\\rangle_\\Omega = k_0\\,\\mathrm{diag}(\\beta_t, \\beta_t, \\beta_a)\\cdot\\underline{\\operatorname{grad}}\\,T^{\\infty}`` in the spheroid's own frame
(before any surface-conductive-interface flux correction — none is
needed here: the HC surface-flux term is already folded into ``(b/a)``
through the transfer-matrix recurrence, unlike `LayeredSphere` where it
is a separate additive term).
"""
function spheroid_flux_gradient(s::LayeredSpheroid{T, N}, k₀) where {T, N}
    qN = s.q[N]
    ba_a, ba_t = spheroid_ba_ratios(s, k₀)
    βa = real(1 + ba_a * _shape_Ua(qN))
    βt = real(1 + ba_t * _shape_Ut(qN))
    return βt, βa
end
