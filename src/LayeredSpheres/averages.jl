# =============================================================================
#  averages.jl — layer / sphere / cumulative averages of the strain (or
#  gradient) field inside an isotropic `LayeredSphere`.
#
#  For the bulk part (fully supported for any `N`), the volume
#  average in layer `k` of the strain tensor reduces to
#
#      <ε>_k = α_k · ε∞   for purely hydrostatic ε∞,
#
#  where `α_k` is the per-layer bulk concentration.  For a general remote
#  strain, the decomposition splits bulk + deviatoric and the shear
#  contribution is delegated to the multi-layer shear solver (single
#  layer only for now).
# =============================================================================

"""
    layer_gradient_average(sphere, C₀, ε∞, layer) -> Tens{2,3}

Volume-averaged strain tensor ``\\langle\\boldsymbol{\\varepsilon}\\rangle_k`` inside the `layer`-th layer
of a `LayeredSphere` embedded in an isotropic matrix `C₀`, under a
remote strain `ε∞`.  Returns a symmetric 2-tensor in the canonical
frame.  Combines the bulk concentration ``\\alpha_k`` (hydrostatic part) and
the shear concentration ``\\beta_k`` (deviatoric part).

This is the strain of the layer's **material**: the opening of a
[`SpringInterface`](@ref) on either side of it is not included.
[`gradient_gradient_concentration`](@ref)`(sphere, C₀; layer, external, internal)` adds it.
"""
function layer_gradient_average(
        sphere::LayeredSphere{T, N},
        C₀::TensND.TensISO{4, 3},
        ε∞::TensND.AbstractTens{2, 3},
        layer::Int,
    ) where {T, N}
    1 ≤ layer ≤ N || throw(BoundsError(sphere, layer))
    κ₀, μ₀ = TensND.k_mu(C₀)
    α = _bulk_concentration(sphere, κ₀, μ₀)[layer]
    β = _shear_concentration(sphere, C₀)[layer]

    Tres = promote_type(T, eltype(C₀), eltype(ε∞))
    I2 = TensISO{3}(one(Tres))
    tr_ε∞ = sum(ε∞[i, i] for i in 1:3)
    ε_sph = (tr_ε∞ / 3) * I2
    ε_dev = ε∞ - ε_sph
    return α * ε_sph + β * ε_dev
end

"""
    sphere_gradient_average(sphere, C₀, ε∞) -> Tens{2,3}

Volume-averaged strain over the whole composite sphere (all layers
combined): ``\\langle\\boldsymbol{\\varepsilon}\\rangle_\\Omega = \\sum_k f_k\\,\\langle\\boldsymbol{\\varepsilon}\\rangle_k`` where ``f_k`` is the volume fraction
of layer ``k`` inside the composite sphere.

This is the strain of the **material**, which leaves out the opening of the
spring interfaces. The strain of the sphere seen from the matrix, which the
schemes use, includes it: `gradient_gradient_concentration(sphere, C₀, C₀) ⊡ ε∞`. The two
agree when every interface is perfect or a membrane.
"""
function sphere_gradient_average(
        sphere::LayeredSphere{T, N},
        C₀::TensND.TensISO{4, 3},
        ε∞::TensND.AbstractTens{2, 3},
    ) where {T, N}
    f = ntuple(k -> layer_volume_fraction(sphere, k), N)
    avgs = ntuple(k -> layer_gradient_average(sphere, C₀, ε∞, k), N)
    return sum(f[k] * avgs[k] for k in 1:N)
end

"""
    _layer_dev_concentration_upto(sphere, amps, layer, r_upper) -> β

Deviatoric concentration averaged over the **truncated** shell
`(r_{layer-1}, r_upper)` rather than over the whole layer.

``\\alpha`` is constant inside a layer, but ``\\beta`` is not: the mode-2 amplitude carries
an ``r^3`` displacement profile whose contribution to the mean deviatoric strain
grows with the shell's outer radius, as
`_layer_avg_dev_shear_factor(r_a, r_b, κ, μ)` records.  Truncating therefore
means re-evaluating that factor at `r_upper`, not rescaling the full-layer
average.  With `r_upper = r_layer` this returns the full-layer ``\\beta`` exactly.
"""
function _layer_dev_concentration_upto(
        sphere::LayeredSphere{T, N}, amps, layer::Int, r_upper
    ) where {T, N}
    κ_k, μ_k = TensND.k_mu(layer_modulus(sphere, layer))
    a_k, b_k, _, _ = amps[layer]
    r_a = layer == 1 ? zero(eltype(sphere.radii)) : sphere.radii[layer - 1]
    return a_k + b_k * _layer_avg_dev_shear_factor(r_a, r_upper, κ_k, μ_k)
end

"""
    cumulative_strain_average(sphere, C₀, ε∞, r) -> Tens{2,3}

Volume-averaged strain over the ball of radius ``r \\in (0, r_N]`` centered on the
composite sphere center.  The ball may cross several layers; the outermost one
it reaches is generally **cut part way through**, and the average accounts for
that exactly.

!!! note "Why a truncated layer is not a scaled one"
    Weighting the full-layer average by the truncated volume — the obvious
    shortcut — is correct only where the field is uniform inside a layer, and
    it is not: the mode-2 term of the deviatoric solution varies as ``r^2``.  The
    truncated average is obtained instead by evaluating
    `_layer_avg_dev_shear_factor` at the cut radius.  The two agree at
    ``r = r_k``, which is why interface radii alone cannot expose the
    difference.
"""
function cumulative_strain_average(
        sphere::LayeredSphere{T, N},
        C₀::TensND.TensISO{4, 3},
        ε∞::TensND.AbstractTens{2, 3},
        r,
    ) where {T, N}
    r > 0 || throw(ArgumentError("cumulative_strain_average radius must be > 0"))
    radii = sphere.radii
    # Beyond r_N the ball takes in the surrounding medium, which is not this
    # sphere's: the average over the sphere alone used to be returned.
    r ≤ radii[N] || throw(
        ArgumentError("cumulative_strain_average radius must not exceed the outer radius $(radii[N]); got $r")
    )

    κ₀, μ₀ = TensND.k_mu(C₀)
    α = _bulk_concentration(sphere, κ₀, μ₀)
    amps = _shear_amplitude_seq(sphere, C₀)

    Tres = promote_type(T, typeof(r), eltype(C₀), eltype(ε∞))
    I2 = TensISO{3}(one(Tres))
    tr_ε∞ = sum(ε∞[i, i] for i in 1:3)
    ε_sph = (tr_ε∞ / 3) * I2
    ε_dev = ε∞ - ε_sph

    # The layers the ball reaches — all those inside it, and the one it cuts —
    # each weighted by the volume of its part inside the ball. Those volumes add
    # up to the ball's.
    K = count(<(r), radii) + 1
    acc = sum(1:K) do k
        r_prev = k == 1 ? zero(Tres) : Tres(radii[k - 1])
        r_upper = min(Tres(r), Tres(radii[k]))
        β_k = _layer_dev_concentration_upto(sphere, amps, k, r_upper)
        (r_upper^3 - r_prev^3) * (α[k] * ε_sph + β_k * ε_dev)
    end
    return (1 / Tres(r)^3) * acc
end

# ── Stress and transport averages ────────────────────────────────────────────
#
# The strain averages above had no counterpart on the stress side, nor in
# transport, although both follow from them in one contraction.

"""
    layer_flux_average(sphere, C₀, ε∞, layer) -> Tens{2,3}

Volume-averaged stress ``\\langle\\boldsymbol{\\sigma}\\rangle_k = \\mathbb{C}_k:\\langle\\boldsymbol{\\varepsilon}\\rangle_k`` inside the
`layer`-th layer, under a remote strain `ε∞`.

Note this is the average of the stress **inside the layer's material**.  A
[`MembraneInterface`](@ref) additionally carries a surface stress on the layer
boundary, which belongs to the interface rather than to either adjacent bulk,
and is therefore *not* included here — `property_contribution` and
`flux_gradient_concentration` account for it separately.  The gap is not small: on a
two-layer sphere with `MembraneInterface(1.3, 0.7)`,
`sphere_flux_average` and `flux_gradient_concentration ⊡ ε∞` differ by order one,
while with perfect interfaces they agree to machine precision.
"""
function layer_flux_average(
        sphere::LayeredSphere{T, N},
        C₀::TensND.TensISO{4, 3},
        ε∞::TensND.AbstractTens{2, 3},
        layer::Int,
    ) where {T, N}
    1 ≤ layer ≤ N || throw(BoundsError(sphere, layer))
    C_k = layer_modulus(sphere, layer)
    κk, μk = TensND.k_mu(C_k)
    _is_incompressible(κk) ||
        return C_k ⊡ layer_gradient_average(sphere, C₀, ε∞, layer)
    # k = ∞: the volumetric strain vanishes and the mean pressure is read from
    # the traction (`_layer_pressures`), not from ∞·0.
    κ₀, μ₀ = TensND.k_mu(C₀)
    α = _bulk_concentration(sphere, κ₀, μ₀)
    P = _layer_pressures(sphere, C₀, α)[layer]
    β = _shear_concentration(sphere, C₀)[layer]
    return TensISO{3}(P, 2 * μk * β) ⊡ ε∞
end

"""
    sphere_flux_average(sphere, C₀, ε∞) -> Tens{2,3}

Volume-averaged stress over the whole composite sphere,
``\\langle\\boldsymbol{\\sigma}\\rangle_\\Omega = \\sum_k f_k\\,\\mathbb{C}_k:\\langle\\boldsymbol{\\varepsilon}\\rangle_k``.
"""
function sphere_flux_average(
        sphere::LayeredSphere{T, N},
        C₀::TensND.TensISO{4, 3},
        ε∞::TensND.AbstractTens{2, 3},
    ) where {T, N}
    f = ntuple(k -> layer_volume_fraction(sphere, k), Val(N))
    avgs = ntuple(k -> layer_flux_average(sphere, C₀, ε∞, k), Val(N))
    return sum(f[k] * avgs[k] for k in 1:N)
end

"""
    layer_gradient_average(sphere, K₀, ∇T∞, layer) -> Vec{3}

Volume-averaged temperature gradient ``\\langle\\nabla T\\rangle_k = \\alpha_k\\,\\nabla T^{\\infty}`` inside the
`layer`-th layer of a `LayeredSphere` in an isotropic matrix `K₀`.

The conduction method.  The average is a scalar
multiple of the remote gradient, and that scalar is **constant inside a
layer**: with ``\\nabla T = [f'\\,\\underline{n}\\otimes\\underline{n} + (f/r)(\\boldsymbol{1} - \\underline{n}\\otimes\\underline{n})]\\cdot\\nabla T^{\\infty}`` and ``f = \\tilde A r + \\tilde B/r^2``, the
directional averages ``\\langle\\underline{n}\\otimes\\underline{n}\\rangle = \\boldsymbol{1}/3`` give

```math
\\langle\\nabla T\\rangle_{\\mathrm{dir}} = \\Big[\\frac{f'}{3} + \\frac{2}{3}\\,\\frac{f}{r}\\Big]\\nabla T^{\\infty} = \\tilde A\\,\\nabla T^{\\infty},
```

the ``\\tilde B/r^3`` terms of ``f'`` and ``f/r`` canceling exactly.  The decaying ``1/r^2``
mode is *not* absent — it is generally large — it simply contributes nothing
to the mean gradient.
"""
function layer_gradient_average(
        sphere::LayeredSphere{T, N},
        K₀::TensND.TensISO{2, 3},
        ∇T∞,
        layer::Int,
    ) where {T, N}
    1 ≤ layer ≤ N || throw(BoundsError(sphere, layer))
    α = _cond_concentration(sphere, _iso_scalar(K₀))[layer]
    g = TensND._extract_vec(∇T∞)
    return TensND.Vec{3}(ntuple(i -> α * g[i], 3))
end

"""
    sphere_gradient_average(sphere, K₀, ∇T∞) -> Vec{3}

Volume-averaged temperature gradient over the whole composite sphere,
``\\langle\\nabla T\\rangle_\\Omega = \\big(\\sum_k f_k\\,\\alpha_k\\big)\\,\\nabla T^{\\infty}``.

As in elasticity, an average over the material: the
temperature jump across a [`KapitzaInterface`](@ref) is left out, and
[`gradient_gradient_concentration`](@ref)`(sphere, K₀, K₀)` includes it.
"""
function sphere_gradient_average(
        sphere::LayeredSphere{T, N},
        K₀::TensND.TensISO{2, 3},
        ∇T∞,
    ) where {T, N}
    α = _cond_concentration(sphere, _iso_scalar(K₀))
    f = ntuple(k -> layer_volume_fraction(sphere, k), Val(N))
    s = sum(f[k] * α[k] for k in 1:N)
    g = TensND._extract_vec(∇T∞)
    return TensND.Vec{3}(ntuple(i -> s * g[i], 3))
end

"""
    layer_flux_average(sphere, K₀, ∇T∞, layer) -> Vec{3}

Volume-averaged flux ``\\langle\\underline{q}\\rangle_k = -k_k\\,\\langle\\nabla T\\rangle_k`` inside the `layer`-th
layer, with the Fourier / Fick sign convention.
"""
function layer_flux_average(
        sphere::LayeredSphere{T, N},
        K₀::TensND.TensISO{2, 3},
        ∇T∞,
        layer::Int,
    ) where {T, N}
    k_here = _cond_layer_moduli(sphere)[layer]
    g = layer_gradient_average(sphere, K₀, ∇T∞, layer)
    return TensND.Vec{3}(ntuple(i -> -k_here * g[i], 3))
end

"""
    sphere_flux_average(sphere, K₀, ∇T∞) -> Vec{3}

Volume-averaged flux over the composite sphere,
``\\langle\\underline{q}\\rangle_\\Omega = \\sum_k f_k\\,\\langle\\underline{q}\\rangle_k``, with the
Fourier / Fick sign convention: the conduction method of the function whose
elasticity method averages the stress.
"""
function sphere_flux_average(
        sphere::LayeredSphere{T, N},
        K₀::TensND.TensISO{2, 3},
        ∇T∞,
    ) where {T, N}
    f = ntuple(k -> layer_volume_fraction(sphere, k), Val(N))
    return sum(f[k] * layer_flux_average(sphere, K₀, ∇T∞, k) for k in 1:N)
end
