# [API — Arbitrary shapes by Gaussian approximation](@id api-gaussian-approximation)

Cracks of arbitrary contour and heterogeneities of arbitrary shape, by the
Gaussian approximation of their integral equations: see
[Approximation by Gaussians](@ref th-gaussian-approximation),
[Planar cracks of any contour](@ref th-gaussian-cracks) and
[Heterogeneities of any shape](@ref th-gaussian-volume) for the theory, and the
manual pages ([cracks](@ref man-planar-cracks),
[heterogeneities](@ref man-gaussian-inclusions)) for the usage. The crack functions
(`cod_tensor`, `sif`, the contributions) are those of
[API — Cracks](@ref api-cracks).

```@docs
MeanFieldHomogenization.GaussianApproximation
```

## The approximation

```@docs
MeanFieldHomogenization.GaussianGrid
MeanFieldHomogenization.gaussian_generator
MeanFieldHomogenization.quasi_interpolate
MeanFieldHomogenization.QuasiInterpolant
MeanFieldHomogenization.saturation_error
```

## Regions

```@docs
MeanFieldHomogenization.AbstractRegion
MeanFieldHomogenization.Disk
MeanFieldHomogenization.Ellipse
MeanFieldHomogenization.Rectangle
MeanFieldHomogenization.Superellipse
MeanFieldHomogenization.Annulus
MeanFieldHomogenization.StarShape
MeanFieldHomogenization.Polygon
MeanFieldHomogenization.Region
MeanFieldHomogenization.PixelRegion
MeanFieldHomogenization.UnionRegion
MeanFieldHomogenization.IntersectionRegion
MeanFieldHomogenization.DifferenceRegion
MeanFieldHomogenization.bounding_box
MeanFieldHomogenization.measure
MeanFieldHomogenization.inside_nodes
MeanFieldHomogenization.coverage
```

## Planar cracks

```@docs
MeanFieldHomogenization.PlanarCrack
MeanFieldHomogenization.ContourShape
MeanFieldHomogenization.crack_area
MeanFieldHomogenization.crack_opening
MeanFieldHomogenization.opening_surface
MeanFieldHomogenization.edge_rule
MeanFieldHomogenization.sif_from_dif
```

## Heterogeneities

```@docs
MeanFieldHomogenization.GaussianInclusion
MeanFieldHomogenization.Torus
MeanFieldHomogenization.inclusion_volume
MeanFieldHomogenization.GaussianApproximation.GaussianShape
MeanFieldHomogenization.GaussianApproximation.smoothed_green_operator
MeanFieldHomogenization.GaussianApproximation.radial_hessian
MeanFieldHomogenization.GaussianApproximation.radial_fourth_gradient
```

## Kernel and solver

```@docs
MeanFieldHomogenization.GaussianApproximation.harmonic_profile
MeanFieldHomogenization.GaussianApproximation.SymbolHarmonics
MeanFieldHomogenization.GaussianApproximation.symbol_harmonics
MeanFieldHomogenization.GaussianApproximation.sample_kernel
MeanFieldHomogenization.GaussianApproximation.SampledKernel
MeanFieldHomogenization.GaussianApproximation.self_block
MeanFieldHomogenization.GaussianApproximation.GridConvolution
MeanFieldHomogenization.GaussianApproximation.convolve!
MeanFieldHomogenization.GaussianApproximation.convolve_direct
MeanFieldHomogenization.GaussianApproximation.convolution_bytes
MeanFieldHomogenization.GaussianApproximation.masked_product
MeanFieldHomogenization.GaussianApproximation.gaussian_solve
```
