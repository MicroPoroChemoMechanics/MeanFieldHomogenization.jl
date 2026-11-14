# =============================================================================
#  layered_alv_order2.jl — n-layer composite sphere in conduction (order 2),
#  aging linear Volterra (ALV) setting.
#
#  The conduction twin of `layered_alv.jl`, and the Volterra counterpart of the
#  elastic `LayeredSpheres/conductivity.jl`. Under a remote uniform gradient the
#  temperature of layer k is `T = (A_k r + B_k / r²) cos θ`, and with the stress
#  analog of the flux, `σ ≡ −q = K ∇T`, its radial component is
#  `σ_r = k_k ∘ (A_k − 2 B_k / r³) cos θ`. Every scalar of the elastic recurrence
#  becomes an n × n Volterra matrix: the conductivities `k_k(t, t′)` of the
#  layers and of the reference, and the amplitudes `A_k`, `B_k`, which map the
#  history of the seed (`A_1`) to theirs. The interface parameters are numbers.
#
#  Across the interface at `R`, between the inner layer `a` and the outer one `b`,
#  the temperature `u = R A + B/R²` and the radial "stress" `s = σ_r` jump as
#
#      Perfect                  u⁺ = u,          s⁺ = s
#      Kapitza(ρ)               u⁺ = u + ρ s,    s⁺ = s          [T] = ρ σ_n = −ρ q_n
#      SurfaceConductive(kˢ)    u⁺ = u,          s⁺ = s + 2kˢ u / R²
#
#  and the outer amplitudes follow in closed form,
#
#      A_b = (3 k_b)^{-∘} ∘ (s⁺ + (2/R) k_b ∘ u⁺),     B_b = R² (u⁺ − R A_b),
#
#  so the only Volterra inverse is that of `3 k_b`: an impermeable core
#  (`k_1 = 0`) needs none, as in the elastic recurrence. An impermeable shell
#  (`k_j = 0`, `j > 1`) lets no flux through: the layers inside it are shielded,
#  their field is zero, and its own field, the limit of a vanishing
#  conductivity, vanishes on its inner face, `A_j = 1`, `B_j = −r_{j−1}³`.
# =============================================================================

# The n × n conductivity matrix of a layer, of a reference law, or of a running
# 3n × 3n reference. A layer stored as an elastic tensor is a Heaviside law, and
# a law in `:creep` mode (a resistivity) is inverted.
_cond_volterra_alv(K, times::AbstractVector{<:Real}) =
    iso_order2_params_from_blocks(_trapezoidal_relaxation(_as_visco_law(K), times, 3))
_cond_volterra_alv(K::AbstractMatrix, ::AbstractVector{<:Real}) = iso_order2_params_from_blocks(K)
# An order-2 tensor of TensND is an `AbstractMatrix` too, but an elastic one.
_cond_volterra_alv(K::TensND.AbstractTens, times::AbstractVector{<:Real}) =
    _cond_volterra_alv(heaviside_law(K), times)

"""
    _cond_layer_moduli_alv(sphere, K0_ref, times) -> (layers, k₀)

The ``n\\times n`` conductivity matrices of the layers of `sphere` and of the
reference `K0_ref` (a `ViscoLaw`, or the ``3n\\times 3n`` matrix of a running
estimate). A non-finite conductivity has no Volterra matrix and is refused.
"""
function _cond_layer_moduli_alv(
        sphere::LayeredSphere{T, N}, K0_ref::_ALVReference,
        times::AbstractVector{<:Real}
    ) where {T, N}
    layers = ntuple(k -> _cond_volterra_alv(layer_modulus(sphere, k), times), N)
    for k in 1:N
        any(_nonfinite, layers[k]) && throw(
            ArgumentError(
                "layer $k of the LayeredSphere has an infinite conductivity, which has " *
                    "no Volterra matrix."
            )
        )
    end
    k₀ = _cond_volterra_alv(K0_ref, times)
    any(_nonfinite, k₀) &&
        throw(ArgumentError("the reference medium of the LayeredSphere has an infinite conductivity"))
    return layers, k₀
end

# The jump of the state (u, s) across an interface, its parameters constant in
# time. The elastic interfaces have no conduction law.
_cond_jump_alv(::PerfectInterface, u, s, R) = (u, s)
_cond_jump_alv(intf::KapitzaInterface, u, s, R) = (u .+ intf.resistance .* s, s)
_cond_jump_alv(intf::SurfaceConductiveInterface, u, s, R) =
    (u, s .+ (2 * intf.conductance / R^2) .* u)
_cond_jump_alv(intf::AbstractInterface, u, s, R) = throw(
    ArgumentError(
        "$(nameof(typeof(intf))) is an elastic interface; a conduction problem takes " *
            "PerfectInterface, KapitzaInterface or SurfaceConductiveInterface."
    )
)

# An impermeable layer: a conductivity matrix that is exactly zero, entry by
# entry as the elastic recurrence tests its scalar.

# The amplitudes (A, B) of the impermeable shell `j`, whose inner radius is `R`,
# and the zero field of the layers it shields: the Volterra form of
# `LayeredSpheres._cond_enter_layer!`.
function _cond_enter_impermeable_alv!(amps, states, layers, R, j::Int)
    LayeredSpheres._is_impermeable(layers[j - 1]) && LayeredSpheres._adjacent_impermeable("LayeredSphere", j)
    Z = zero(first(first(amps)))
    amps[1:(j - 1)] .= Ref((Z, Z))
    states[1:(j - 1)] .= Ref((Z, Z))
    return one(Z), -R^3 .* one(Z)
end

"""
    _cond_amplitude_seq_alv(sphere, K0_ref, times)
        -> (amps, states, (A_∞, B_∞), layers, k₀)

Forward recurrence of the conduction amplitudes. `amps[k] = (A_k, B_k)` and
`states[k] = (u, s)`, the temperature and the radial flux analog just inside
``r_k``, are ``n\\times n`` Volterra matrices acting on the seed ``A_1 = \\mathbb 1_n``,
``B_1 = 0``; ``A_\\infty`` is the amplitude of the remote gradient in the reference
and ``B_\\infty`` that of its decaying mode. An impermeable shell restarts the
sequence with a unit amplitude of its own (`_cond_enter_impermeable_alv!`).
"""
function _cond_amplitude_seq_alv(
        sphere::LayeredSphere{T, N}, K0_ref::_ALVReference,
        times::AbstractVector{<:Real}
    ) where {T, N}
    n = length(times)
    layers, k₀ = _cond_layer_moduli_alv(sphere, K0_ref, times)
    TP = promote_type(
        eltype(k₀), (eltype(m) for m in layers)...,
        interfaces_eltype(sphere.interfaces), T
    )
    radii = sphere.radii
    A = Matrix{TP}(I, n, n)
    B = zeros(TP, n, n)
    amps = Vector{NTuple{2, Matrix{TP}}}(undef, N)
    states = Vector{NTuple{2, Matrix{TP}}}(undef, N)
    for k in 1:N
        R = TP(radii[k])
        amps[k] = (A, B)
        u = R .* A .+ B ./ R^2
        s = layers[k] * (A .- (2 / R^3) .* B)
        states[k] = (u, s)
        if k < N && LayeredSpheres._is_impermeable(layers[k + 1])
            A, B = _cond_enter_impermeable_alv!(amps, states, layers, R, k + 1)
        else
            u, s = _cond_jump_alv(layer_interface(sphere, k), u, s, R)
            k_b = k < N ? layers[k + 1] : k₀
            A = volterra_left_divide(3 .* k_b, s .+ (2 / R) .* (k_b * u); block_size = 1)
            B = R^2 .* (u .- R .* A)
        end
    end
    return amps, states, (A, B), layers, k₀
end

"""
    _cond_unit_solution_alv(sphere, K0_ref, times) -> (; amps, states, layers, k₀)

The conduction solution per unit remote gradient history: the amplitudes
`amps[k] = (A_k, B_k)` of the layers, then of the reference (``k = N + 1``,
``A = \\mathbb 1_n``), and the states `(u, s)` just inside each ``r_k``, those of
[`_cond_amplitude_seq_alv`](@ref) with ``A_\\infty^{-1^{\\circ}}`` applied **on the
right**.
"""
function _cond_unit_solution_alv(
        sphere::LayeredSphere, K0_ref::_ALVReference, times::AbstractVector{<:Real}
    )
    amps, states, (A_∞, B_∞), layers, k₀ = _cond_amplitude_seq_alv(sphere, K0_ref, times)
    A_∞⁻¹ = volterra_inverse(A_∞; block_size = 1)
    unit(pair) = (pair[1] * A_∞⁻¹, pair[2] * A_∞⁻¹)
    return (;
        amps = push!(map(unit, amps), (one(A_∞), B_∞ * A_∞⁻¹)),
        states = map(unit, states), layers, k₀,
    )
end

"""
    gradient_concentration_alv(sphere, K0_law, times) -> NTuple{N, Matrix}

Per-layer gradient concentrations of a [`LayeredSphere`](@ref MeanFieldHomogenization.LayeredSpheres.LayeredSphere) in an isotropic
aging reference: the ``n\\times n`` Volterra matrices ``\\alpha_k`` with
``\\langle\\nabla T\\rangle_k = \\alpha_k\\circ\\nabla T^{\\infty}``, averaged over the material of
layer ``k`` (the conduction twin of [`bulk_concentration_alv`](@ref)). For a
non-aging problem they are the elastic ones of
[`gradient_gradient_concentration`](@ref) at every time step.
"""
function gradient_concentration_alv(
        sphere::LayeredSphere{T, N}, K0_law::_ALVReference,
        times::AbstractVector{<:Real}
    ) where {T, N}
    sol = _cond_unit_solution_alv(sphere, K0_law, times)
    return ntuple(k -> sol.amps[k][1], Val(N))
end

# The moments an interface at `r` adds to the averages, from the state (u, s)
# just inside it: that of its temperature jump (a Kapitza resistance) and that
# of its surface flux (a conducting surface).
_cond_interface_moments_alv(::AbstractInterface, u, s, r) = (zero(u), zero(u))
_cond_interface_moments_alv(intf::KapitzaInterface, u, s, r) =
    ((r^2 * intf.resistance) .* s, zero(u))
_cond_interface_moments_alv(intf::SurfaceConductiveInterface, u, s, r) =
    (zero(u), (2 * intf.conductance * r) .* u)

# The temperature jumps of the Kapitza interfaces and the surface flux of the
# surface-conductive ones among `interfaces`, as n × n blocks per unit remote
# gradient and per unit volume of the whole sphere.
function _cond_interface_terms_alv(sphere::LayeredSphere{T, N}, sol, interfaces) where {T, N}
    radii = sphere.radii
    R³ = radii[N]^3
    Z = zero(sol.k₀)
    moments = [
        _cond_interface_moments_alv(layer_interface(sphere, k), sol.states[k]..., radii[k])
            for k in interfaces
    ]
    return sum(first, moments; init = Z) ./ R³, sum(last, moments; init = Z) ./ R³
end

# The scalar (on 𝟏) of the average gradient of the whole sphere,
# `layer = nothing`, or of one layer, per unit remote gradient history.
function _gradient_params_alv(
        sphere::LayeredSphere{T, N}, sol, ::Nothing, external::Bool, internal::Bool
    ) where {T, N}
    α = sum(layer_volume_fraction(sphere, k) .* sol.amps[k][1] for k in 1:N)
    Δα, _ = _cond_interface_terms_alv(sphere, sol, _sphere_interfaces(sphere, external))
    return α .+ Δα
end

function _gradient_params_alv(sphere::LayeredSphere, sol, layer::Int, external::Bool, internal::Bool)
    _check_alv_layer(sphere, layer)
    Δα, _ = _cond_interface_terms_alv(sphere, sol, _layer_interfaces(layer, external, internal))
    return sol.amps[layer][1] .+ Δα ./ layer_volume_fraction(sphere, layer)
end

# The gradient concentration of the whole sphere in conduction, or of one
# layer: see the generic.
function _gradient_gradient_concentration_alv(
        ::Val{2}, sphere::LayeredSphere, K0_law::_ALVReference,
        times::AbstractVector{<:Real};
        layer::Union{Nothing, Int} = nothing, external::Bool = layer === nothing,
        internal::Bool = false,
    )
    sol = _cond_unit_solution_alv(sphere, K0_law, times)
    return iso_order2_blocks_from_params(_gradient_params_alv(sphere, sol, layer, external, internal))
end

# Σ_k f_k (k_k − k_ref)∘α_k, each conductivity on the left of its concentration.
_cond_layer_flux_sum_alv(sphere::LayeredSphere{T, N}, sol, k_ref) where {T, N} = sum(
    layer_volume_fraction(sphere, k) .* ((sol.layers[k] .- k_ref) * sol.amps[k][1]) for k in 1:N
)

# The conductivity contribution of the whole sphere: see the generic.
function _property_contribution_alv(
        ::Val{2}, sphere::LayeredSphere, K0_law::_ALVReference,
        times::AbstractVector{<:Real}; external::Bool = true
    )
    sol = _cond_unit_solution_alv(sphere, K0_law, times)
    Δα, surface = _cond_interface_terms_alv(sphere, sol, _sphere_interfaces(sphere, external))
    return iso_order2_blocks_from_params(
        _cond_layer_flux_sum_alv(sphere, sol, sol.k₀) .+ surface .- sol.k₀ * Δα
    )
end

# The average of k·∇T over the whole sphere, with the flux of its conducting
# surfaces, or over the material of one layer: see the generic.
function _flux_gradient_concentration_alv(
        ::Val{2}, sphere::LayeredSphere, K0_law::_ALVReference,
        times::AbstractVector{<:Real}, ::Nothing, external::Bool
    )
    sol = _cond_unit_solution_alv(sphere, K0_law, times)
    _, surface = _cond_interface_terms_alv(sphere, sol, _sphere_interfaces(sphere, external))
    return iso_order2_blocks_from_params(_cond_layer_flux_sum_alv(sphere, sol, zero(sol.k₀)) .+ surface)
end

function _flux_gradient_concentration_alv(
        ::Val{2}, sphere::LayeredSphere, K0_law::_ALVReference,
        times::AbstractVector{<:Real}, layer::Int, external::Bool
    )
    _check_alv_layer(sphere, layer)
    sol = _cond_unit_solution_alv(sphere, K0_law, times)
    return iso_order2_blocks_from_params(sol.layers[layer] * sol.amps[layer][1])
end

# Voigt and Reuss averages of the layer conductivities: what the two bounds
# assign to a layered sphere, as in elasticity.
function _layer_cond_kernels_alv(sphere::_LayeredALV, times)
    N = layer_count(sphere)
    Ks = map(1:N) do k
        K = layer_modulus(sphere, k)
        return _trapezoidal_relaxation(_as_visco_law(K), times, 3)
    end
    return Ks, [layer_volume_fraction(sphere, k) for k in 1:N]
end
_layer_voigt_alv2(sphere::_LayeredALV, times) = voigt_alv_order2(_layer_cond_kernels_alv(sphere, times)...)
_layer_reuss_alv2(sphere::_LayeredALV, times) = reuss_alv_order2(_layer_cond_kernels_alv(sphere, times)...)

# The average a scheme reads as the conductivity of a layered sphere, as
# `_layer_bound_alv` at order 4.
_layer_bound_alv2(::Reuss, sphere::_LayeredALV, times) = _layer_reuss_alv2(sphere, times)
_layer_bound_alv2(::HomogenizationScheme, sphere::_LayeredALV, times) = _layer_voigt_alv2(sphere, times)
