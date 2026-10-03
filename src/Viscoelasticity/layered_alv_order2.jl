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
#  (`k_1 = 0`) needs none, as in the elastic recurrence.
# =============================================================================

# The n × n conductivity matrix of a layer, of a reference law, or of a running
# 3n × 3n reference. A layer stored as an elastic tensor is a Heaviside law, and
# a law in `:creep` mode (a resistivity) is inverted.
function _cond_volterra_alv(K, times::AbstractVector{<:Real})
    law = K isa ViscoLaw ? K : heaviside_law(K)
    return iso_order2_params_from_blocks(_trapezoidal_relaxation(law, times, 3))
end
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

"""
    _cond_amplitude_seq_alv(sphere, K0_ref, times)
        -> (amps, states, A_∞, layers, k₀)

Forward recurrence of the conduction amplitudes. `amps[k] = (A_k, B_k)` and
`states[k] = (u, s)`, the temperature and the radial flux analog just inside
``r_k``, are ``n\\times n`` Volterra matrices acting on the seed ``A_1 = \\mathbb 1_n``,
``B_1 = 0``; `A_∞` is the amplitude of the remote gradient in the reference.
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
        u, s = _cond_jump_alv(layer_interface(sphere, k), u, s, R)
        k_b = k < N ? layers[k + 1] : k₀
        A = volterra_left_divide(3 .* k_b, s .+ (2 / R) .* (k_b * u); block_size = 1)
        B = R^2 .* (u .- R .* A)
    end
    return amps, states, A, layers, k₀
end

"""
    gradient_localization_alv(sphere, K0_law, times) -> NTuple{N, Matrix}

Per-layer gradient concentrations of a [`LayeredSphere`](@ref) in an isotropic
aging reference: the ``n\\times n`` Volterra matrices ``\\alpha_k`` with
``\\langle\\nabla T\\rangle_k = \\alpha_k\\circ\\nabla T^{\\infty}``, averaged over the material of
layer ``k`` (the conduction twin of [`bulk_localization_alv`](@ref)). For a
non-aging problem they are the elastic ones of
[`gradient_gradient_loc`](@ref) at every time step.
"""
function gradient_localization_alv(
        sphere::LayeredSphere{T, N}, K0_law::_ALVReference,
        times::AbstractVector{<:Real}
    ) where {T, N}
    amps, _, A_∞, _, _ = _cond_amplitude_seq_alv(sphere, K0_law, times)
    A_∞⁻¹ = volterra_inverse(A_∞; block_size = 1)
    return ntuple(k -> amps[k][1] * A_∞⁻¹, Val(N))
end

# The temperature jumps of the Kapitza interfaces and the surface flux of the
# surface-conductive ones, as n × n blocks per unit remote gradient, over the
# inner interfaces and the outer one when `external` is true.
function _cond_interface_terms_alv(
        sphere::LayeredSphere{T, N}, states, A_∞, external::Bool
    ) where {T, N}
    radii = sphere.radii
    R³ = radii[N]^3
    A_∞⁻¹ = volterra_inverse(A_∞; block_size = 1)
    Δα = zero(A_∞)
    surface = zero(A_∞)
    for k in 1:(external ? N : N - 1)
        intf = layer_interface(sphere, k)
        r = radii[k]
        u, s = states[k]
        if intf isa KapitzaInterface
            Δα .+= (r^2 * intf.resistance / R³) .* (s * A_∞⁻¹)
        elseif intf isa SurfaceConductiveInterface
            surface .+= (2 * intf.conductance * r / R³) .* (u * A_∞⁻¹)
        end
    end
    return Δα, surface
end

"""
    gradient_gradient_loc_alv(sphere, K0_law, times; external = true) -> Matrix

``3n\\times 3n`` gradient concentration of the whole [`LayeredSphere`](@ref) in an
isotropic aging reference, ``\\sum_k f_k\\,\\alpha_k + \\Delta\\alpha``, the conduction twin
of [`strain_strain_loc_alv`](@ref): ``\\Delta\\alpha`` adds the temperature jumps
``[\\![T]\\!] = \\rho\\,\\sigma_n = -\\rho\\,q_n`` of the Kapitza interfaces, the outer one
when `external` is `true`, as in the elastic [`gradient_gradient_loc`](@ref).
"""
function gradient_gradient_loc_alv(
        sphere::LayeredSphere{T, N}, K0_law::_ALVReference,
        times::AbstractVector{<:Real}; external::Bool = true
    ) where {T, N}
    amps, states, A_∞, _, _ = _cond_amplitude_seq_alv(sphere, K0_law, times)
    A_∞⁻¹ = volterra_inverse(A_∞; block_size = 1)
    α = sum(layer_volume_fraction(sphere, k) .* (amps[k][1] * A_∞⁻¹) for k in 1:N)
    Δα, _ = _cond_interface_terms_alv(sphere, states, A_∞, external)
    return iso_order2_blocks_from_params(α .+ Δα)
end

"""
    conductivity_contribution_alv(sphere, K0_law, times; external = true) -> Matrix

``3n\\times 3n`` conductivity contribution of the whole [`LayeredSphere`](@ref),
``\\sum_k f_k\\,(k_k - k_0)\\circ\\alpha_k - k_0\\circ\\Delta\\alpha`` plus the surface flux of the
surface-conductive interfaces: the conduction twin of
[`stiffness_contribution_alv`](@ref) and the Volterra counterpart of the elastic
`conductivity_contribution`. `K0_law` may also be the ``3n\\times 3n`` matrix of a
running estimate, which is how the self-consistent and differential schemes
call it.
"""
function conductivity_contribution_alv(
        sphere::LayeredSphere{T, N}, K0_law::_ALVReference,
        times::AbstractVector{<:Real}; external::Bool = true
    ) where {T, N}
    amps, states, A_∞, layers, k₀ = _cond_amplitude_seq_alv(sphere, K0_law, times)
    A_∞⁻¹ = volterra_inverse(A_∞; block_size = 1)
    N_K = sum(
        layer_volume_fraction(sphere, k) .* ((layers[k] .- k₀) * (amps[k][1] * A_∞⁻¹))
            for k in 1:N
    )
    Δα, surface = _cond_interface_terms_alv(sphere, states, A_∞, external)
    return iso_order2_blocks_from_params(N_K .+ surface .- k₀ * Δα)
end

# Voigt and Reuss averages of the layer conductivities: what the two bounds
# assign to a layered sphere, as in elasticity.
function _layer_cond_kernels_alv(sphere::LayeredSphere{T, N}, times) where {T, N}
    Ks = map(1:N) do k
        K = layer_modulus(sphere, k)
        return _trapezoidal_relaxation(K isa ViscoLaw ? K : heaviside_law(K), times, 3)
    end
    return Ks, [layer_volume_fraction(sphere, k) for k in 1:N]
end
_layer_voigt_alv2(sphere::LayeredSphere, times) = voigt_alv_order2(_layer_cond_kernels_alv(sphere, times)...)
_layer_reuss_alv2(sphere::LayeredSphere, times) = reuss_alv_order2(_layer_cond_kernels_alv(sphere, times)...)
