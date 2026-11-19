# [API — Arbitrary shapes by Gaussian approximation](@id api-gaussian-approximation)

Cracks of arbitrary contour, by the Gaussian approximation of their integral
equation: see [Approximation by Gaussians](@ref th-gaussian-approximation) and
[Planar cracks of any contour](@ref th-gaussian-cracks) for the theory, and the
[manual page](@ref man-planar-cracks) for the usage. The crack functions
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
MeanFieldHomogenization.edge_rule
MeanFieldHomogenization.sif_from_dif
```

## Kernel and solver

```@docs
MeanFieldHomogenization.GaussianApproximation.harmonic_profile
MeanFieldHomogenization.GaussianApproximation.SymbolHarmonics
MeanFieldHomogenization.GaussianApproximation.symbol_harmonics
MeanFieldHomogenization.GaussianApproximation.sample_kernel
MeanFieldHomogenization.GaussianApproximation.SampledKernel
MeanFieldHomogenization.GaussianApproximation.GridConvolution
MeanFieldHomogenization.GaussianApproximation.convolve!
MeanFieldHomogenization.GaussianApproximation.convolve_direct
MeanFieldHomogenization.GaussianApproximation.convolution_bytes
MeanFieldHomogenization.GaussianApproximation.masked_product
MeanFieldHomogenization.GaussianApproximation.gaussian_solve
```
