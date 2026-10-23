# Changelog

## v0.19.0 — tensors read in the basis they are written in; generic tensor code in TensND

A minor release. Components of TensND tensors were read in their own basis as if
they were canonical in a dozen places, which gave wrong results as soon as a
tensor was written in a rotated basis: a user's property, and above all the
anisotropic Hill tensors, `tens_IA` and the COD tensors, which are written in
their inclusion's basis. Every such place now reads the tensor in the basis it
is written in, or brings the cheap object (an axis, a frame, a 3×3 matrix) into
that basis. The Hill tensors are checked against echoes on 102 cases drawn at
random (shapes, media, orientations, and the basis each medium is written in).
The generic tensor algebra MFH carried moves to TensND 0.7.

### Breaking changes

- The minor version, and `TensND = "0.7"`: `[compat]` bounds on 0.18 do not
  accept 0.19.
- Results change where they were wrong, for tensors written in a rotated basis
  (list under Fixed).
- `extract_iso_moduli`, `extract_iso_conductivity` and `extract_ti_moduli` are
  deprecated in favor of TensND's `E_nu` / `mu_nu`, `get_data` and
  `arg_TI_Hoenig`; the Kelvin-Mandel block algebra (`KM_IP`, `KM_OP`,
  `plane_pinv`, `plane_pinv2`, `acoustic_tensor`, `compliance_op_block`) is
  TensND's, re-exported unchanged from `Core`.
- The anisotropic conduction Hill tensor of an ellipsoid, the finite-element
  concentration tensors and the COD tensor with an interface stiffness are now
  returned in the inclusion's basis rather than relabeled canonical: the same
  tensors; `get_array` of them gives components in that basis
  (`components_canon` gives canonical ones).

### Fixed

- `ALVKernelTI` and `ALVKernelOrtho` refused an axis of mixed number types
  (`(0.0, 0, 1)`, a `TypeError`); they take any real components now, and a dual
  or symbolic axis, which the kernel cannot honor, is refused with a message.
- `fe_available_gb` and the format check of a neural surrogate file caught every
  exception; they now catch the failures they mean and let a bug surface.
- An aging-viscoelastic estimate (`homogenize_alv`) could not be differentiated
  with respect to a crack density: the crack contributions, and the crack term
  of Mori-Tanaka, were accumulated in place into buffers of the reference's
  element type. They are now sums; the derivative matches the one of the
  elastic scheme in the elastic limit.
- `eim_bound_type` compared the eigenvalues of each contrast with an absolute
  1e-12: with moduli in small units every contrast read as zero and the answer
  was `:upper` whatever the particles. The threshold is now relative to the
  reference; dual moduli are read by their value (they threw), and symbolic ones
  are refused with a message.
- The quadrature oracle of `interaction_tensor` (`method = :quadrature`) and the
  `density_factor` of a `CustomInclusion` forced `Float64`, so a sensitivity to
  a semi-axis or to the factor threw; both now take the element type of the
  geometry.
- `confocal_layer_radii` with symbolic arguments failed with an obscure
  non-boolean error; it is a bisection, and now refuses them with a message
  (dual numbers still differentiate through it).
- Carlson's elliptic integrals on exact `Rational{BigInt}` arguments stopped at
  the tolerance of `Float64` (sixteen digits) instead of the `BigFloat`
  precision they float to.
- **An oblate layered spheroid differentiated by dual numbers had `NaN`
  derivatives** (its value was right). The confocal volume was taken by `abs`,
  and the whole-inclusion average evaluates it at a complex zero, where the
  modulus has no derivative. The volume is real and nonnegative as it stands,
  and is now taken by `real`; the derivative matches an extrapolated centered
  difference to 1e-11. The `4π/3` factor also follows the element type now,
  instead of being a `Float64`.
- **The elastic confocal layered spheroid was solved on a singular system** as
  soon as it had an inner shell. Papkovich–Neuber potentials are not unique, and
  the shear cases (transverse and longitudinal) admit one field-free combination
  of their regular modes, in the core and in every inner shell: two singular
  values at 1e-16 for two layers. The concentration came out to about 1e-6 only,
  differently along the dual path (2.4e-6 apart at ω = 2.6), and its derivative
  was off by 1e-5. One column of that combination is now removed in each region
  with regular modes; the system is full-rank (condition number about 1e4),
  Float64 and duals agree to 1e-15, and the derivative matches an extrapolated
  centered difference to 1e-11. A single inclusion was not affected beyond
  rounding.
- **2-D Green operator and 2-D pair tensors** (EIM, cluster model): a 2-D
  isotropic stiffness was read with the 3-D formulas (k = α/3), giving the Poisson
  ratio (α − β)/(2α + β) where the 2-D Hill tensor uses the plane-strain
  (α − β)/(2α); both now use the latter, the Poisson ratio of the 3-D material
  whose plane-strain stiffness the 2-D tensor is.
- **Conduction Hill tensor of an ellipsoid in an anisotropic conductor written in
  a rotated basis**: the conductivity was read in its own basis next to a
  canonical shape tensor.
- **EIM and cluster assemblies** flattened blocks written in different bases into
  one matrix.
- **Order-2 ALV Hill kernel**: `tens_IA` was read in the ellipsoid's basis as
  canonical, so a rotated ellipsoid was silently aligned.
- **Differential scheme**: the fully anisotropic state was read in the tensor's
  own basis; a TI or orthotropic increment about another axis was copied as if
  about the same.
- **Self-consistent residual and convergence of the derivatives** compared
  iterates written in different bases.
- **Interface stiffness of an oriented crack**: the COD tensor came back in the
  wrong frame.
- **ALV crack guard**: it read the crack normal in the crack's own basis, where
  it is always e₃, so a rotated crack passed it.
- **Finite-element frames**: a property written in a rotated basis was rotated
  twice.
- **Layered-sphere local displacement** under a strain written in a rotated
  basis.
- **Julia 1.14**: six generated constructors had unbound type parameters (Aqua);
  rebuilding a `NeuralConcentrationInclusion` without shape parameters threw,
  so its sensitivity to a semi-axis did. On Julia 1.14 the Hill, auxiliary and
  concentration tensors of a spheroid or a cylinder were inferred as the
  abstract `TensTI`: their symmetry axis was read off the basis as a tuple of
  run-time length, which 1.14 no longer sees through. It is now built with a
  static length, and the type is concrete on every version. And
  `check_inclusion_interface` took the package's generic fallbacks for methods
  an inclusion supplies, so it reported a crack as a valid volume-fraction
  phase: 1.14 hands the declared type of a parametric argument back in another
  form, which the comparison missed.
- **`_major_symmetrize` of a 3-parameter `TensTI{2}`** kept its skew part.
- **`extract_ti_moduli`** went through an Euler-angle frame and 81 components to
  read five moduli that do not depend on the axis; they are read off the Walpole
  coefficients, exactly, symbolic moduli included.
- **TI coaxiality test of the Hill dispatcher** threw on a symbolic axis.
- **`tens_UA` of an ellipse** (2-D): its diagonal carried the 3/2 factor of the
  3-D formula, and it switched to another, also wrong, formula when the semi-axes
  came within 1e-6 of each other, so the value jumped there and the derivative
  along the gap was lost. It is now the closed form of the 2-D integral, valid at
  `a = b`, and rebuilds Mura's Hill tensor exactly.
- **Self-consistent scheme through NonlinearSolve under nested dual numbers**
  (a second derivative) threw; the implicit-function lift now gives every order.
- **Self-consistent Newton** took any error of its linear solve, a bug included,
  for a singular Jacobian and regularized it; only a singular system is now
  regularized, and an error in the seed step is no longer swallowed.
- **Choice of the pipeline for nested or viscoelastic properties** skipped a member
  on any error raised while reading it; a missing property is now tested for, and
  any other error surfaces.
- **Finite-element inclusions shared by several tasks** could assemble the same
  cell twice or corrupt their cache; the cache is now locked.
- **Implicit ODE solvers in the differential scheme** (`Rosenbrock23()`,
  `Rodas5P()`) failed at their first time gradient: the incorporation rates
  were built in the state's element type only, so the dual τ of the time
  gradient was forced into Float64. They now run with their default ForwardDiff,
  and the documentation no longer recommends `AutoFiniteDiff()`.

### Changed

- `homogenize_alv` picks, once per call, the cheapest algebra all the operators
  belong to (isotropic, TI, orthotropic, else dense) and runs the scheme there,
  instead of four written-out attempts per scheme.
- The closed-form ALV schemes (Voigt, Reuss, dilute, dilute dual, Mori-Tanaka,
  Maxwell, dilute concentration and contribution) are written once over the
  algebra of the operator, instead of five times (dense order 4, dense order 2,
  iso, TI, orthotropic). The exported functions keep their signatures. The TI
  and orthotropic dilute schemes now accept dual fractions, and a dense
  concentration or contribution with blocks of the wrong size raises
  `DimensionMismatch` (order 2 raised `ArgumentError`, order 4 an assertion).
- `k_mu`, `iso_stiffness`, `E_nu`, `iso_stiffness_E_nu` are TensND's, re-exported
  unchanged; `hoenig_params` / `hoenig_stiffness` are built on TensND's Hoenig
  forms. Internal duplicates of TensND (`_bond6`, `_primal`, `_no_derivative`,
  `_make_ortho`, the identities, three axis-parallel tests) are gone.
- Orthotropic auxiliary and Hill tensors keep their `TensOrtho` structure under
  dual numbers (they fell back to a dense tensor).
- Hill kernels: components by one change of basis, axis permutations by
  indexing, instead of nested copy loops.
- Tensor comparisons in the tests are made as tensors, in their common basis.

### Performance

- The Laplacian of the plane-strain Green operator used by the disk-pair
  interaction is built on the stack: 3 times faster and allocation-free.
- The elastic strain concentration of a layered confocal spheroid is 13 times
  faster for a prolate one (790 ms → 59 ms for two layers) and 4 times for an
  oblate one (1.22 s → 0.28 s), with 90 times fewer allocations: the mode groups
  are concrete, the Legendre tables of a quadrature point are built once and
  shared by every mode and every azimuth, and the surface moment works on static
  matrices.
- Isotropic Green operator (elastic, 3-D and 2-D) written in its symmetrized
  closed form, without the two intermediate 81-component arrays: 2.7 times
  faster (1.08 µs to 0.40 µs in 3-D).
- Anisotropic Green operator static throughout (stiffness, input point, nested
  Jacobians): 15 % faster and 9 times fewer allocations.
- Aging-viscoelastic self-consistent, asymmetric self-consistent and
  differential schemes compute their products and inverses in the cheapest
  algebra the operands share (isotropic, transversely isotropic, orthotropic)
  rather than on dense 6n×6n block matrices, build the Hill kernel of an
  ellipsoid as a Kronecker sum, and run the right-hand side of the differential
  ODE over concrete records: 1.2 to 1.4, 1.5 and 1.6 times faster on 25 times
  with a prolate and a spherical family, with up to 30 % fewer allocations,
  results identical to 3e-16.

### Tests

- `test/Core/test_random_bases.jl`: each fixed function with its operands in
  random bases.
- `test/Elasticity/test_hill_random_bases.jl`: 102 echoes references
  (`scripts/bench_echoes/make_hill_random_bases.py`), each medium given in the
  canonical basis, a rotated one, a non-orthonormal one and its structured type;
  derivatives on exact identities.

## v0.18.1 — MFH Studio: the plot of the run shown, and no other

A patch release of the studio; the homogenization is unchanged.

### Fixed

- **A run that draws no curve left the curve of the run before.** After a
  sweep, a run of one point or of a sensitivity, or a script that failed or was
  refused, kept the previous curves under *Results*, although the current
  script computes nothing of the kind: the page showed a result the script on
  the right does not produce. Such a run now clears the plot, with a note that
  its numbers are in the output.
- **The memory guard of the studio did not see the compute process grow on
  macOS.** It read the resident size of that process with
  `proc_pidinfo(…, PROC_PIDTASKINFO, …)`, which on the macOS runners of GitHub
  never showed the watchdog a process allocating 600 MB: it was never stopped,
  and its test failed there. It now reads its physical footprint with
  `proc_pid_rusage`, what macOS charges a process, compressed pages included. A
  test reads the memory of a child process on every system, and the test of the
  watchdog allocates random bytes, which no system compresses.
- **The script that takes the screenshots of the documentation** waited only
  for the *Run* button to come back, so a run the studio refused (not enough
  memory free, a calculation already running) or that failed would have given
  a picture of an empty plot without a word. It now waits until the studio is
  free (the warm-up calculation follows its first "ready"), and stops with what
  the run printed when no curve is drawn.

## v0.18.0 — MFH Studio in Julia, inside the package

A minor release. MFH Studio, until now a Python application shipped beside the
package in `tools/mfhstudio` with a Julia sidecar, is now Julia from end to
end: the submodule `MeanFieldHomogenization.Studio` (`src/studio/`), started by
`mfhstudio()` as before. Using it needs Julia and a browser, nothing else; only
the translation of an Echoes script still calls Python. The homogenization
itself is unchanged: no function outside the studio changed, and no result
moves.

### Breaking changes

- **The compat bound.** Below 1.0 a minor version is breaking for Julia's
  resolver: a bound `MeanFieldHomogenization = "0.17"` does not admit 0.18, so
  a package or an environment that bounds MeanFieldHomogenization must widen
  its bound to take this release.
- **New dependencies: HTTP.jl 2.11 and Malt.jl 1.4**, with the standard
  libraries TOML and InteractiveUtils, which the studio needs. An environment
  that also holds a package requiring HTTP 1.x can no longer resolve with this
  release until that package admits HTTP 2.
- **`mfhstudio` lost its `python` keyword**, there being no Python server any
  more. The `MFHSTUDIO_PYTHON` environment variable now only names the Python
  that translates Echoes scripts.
- **`tools/mfhstudio` is removed**: `python3 -m mfhstudio` no longer exists. The
  worked examples moved to `src/studio/library/`.
- **`mfhstudio(; project)`** names the environment the scripts run in, which
  is checked (it must hold the MeanFieldHomogenization that runs, TensND and
  Plots) and never modified. Without it, the studio uses an environment of its
  own, `@mfh-studio`, which its first launch prepares, instead of the
  package's environment.
- **`mfhstudio(wait = false)`** returns a handle that `close(h)` stops (and
  `wait(h)` waits on), instead of the handle of a Python process.

### Added

- **The studio in Julia.** The page talks to a server running in the user's
  session (HTTP.jl, on the loopback interface, with the session's random token
  required on every call), where MeanFieldHomogenization is loaded: the forms,
  the generated script, the 3-D view and the reading of a script back answer at
  once, with no process to wait for. A script runs in a second Julia process
  (Malt.jl) in `@mfh-studio`, which holds the very MeanFieldHomogenization
  that is loaded (that version, or the clone being developed), TensND and
  Plots; a script that loops or exhausts the memory takes that process down,
  never the session. Its first launch prepares the environment, with the
  progress shown in the page; **Stop** interrupts a run, and restarts the
  process when the interrupt does not take; **Restart** boots it anew; a run
  is refused when less than 1.5 GB of memory are free, and stopped when it
  outgrows what the machine can spare. The scripts it writes are the ones the
  Python studio wrote, byte for byte, on the 1,115 generations recorded from
  it (the ten examples, the demo scripts, the models of its tests and random
  models).
- **The page in English or in French**, from a button in the header; the
  script and its comments stay in English.
- **`MeanFieldHomogenization.Studio.regenerate_library!()`** writes the worked
  examples again from their own models after a change of the code
  generation; the tests check that they match what the studio writes today.

### Fixed

Defects of the Python studio, found while porting it:

- **Reading back a script lost its matrix.** The phase taking up the rest
  (`fraction = :rest`) was read as a phase of fraction `":rest"`: the cell had
  no matrix, and every save wrote that problem into the script's header.
- **No multiscale script was ever read back as a model.** A builder holding a
  seam (`Homogenized(build_x(), scheme)`) was recognized only under the
  `build_` prefix, and checked in a model holding it alone, where its inner
  scale was missing, so that it was always kept as written. The seams are now
  read with Julia's parser, under any builder name and with the options of
  their scheme, and the builders are checked together.
- **The 3-D view evaluated the text the page sent**, in a "sandbox" that let
  any expression through. The views are now built from the model by the
  package's own constructors, a size or an angle being a number or an
  arithmetic expression of numbers and parameters. The page's JavaScript copy
  of the code generation, which lost the orientation of a layered spheroid, is
  gone, and a laminate with a layer read from an inner scale can be drawn.
- **A hung sidecar could not be restarted** from the page: Stop and Restart
  now exist.
- **The "Matrix phase" field changed a name the script never reads**; it is
  gone, the role of each phase deciding the matrix.
- **Saving after opening a worked example wrote over the package's file**, or
  failed with a bare error where the package is installed read-only. An
  example now opens with no file bound to it, and Save asks where to keep the
  copy; a save that cannot write says why.
- **On the dark theme**, the legend and the axes of the 3-D view were dark gray
  on a dark background; the legend of the results sat on the title of the
  x axis.

## v0.17.4 — MFH Studio: a session token, a free port, a theme, and four defects

A patch release of the studio shipped in `tools/mfhstudio`; the package itself
is unchanged. The studio runs Julia on request, and any web page the user
visited could ask it to: it now answers only its own page. Its default port
gives way when taken instead of failing, and four defects found while reading
it are fixed.

### Fixed

- **Any web page could run code through the studio.** The server checked
  neither who called it nor from where. Each start now draws a random token,
  printed in the address (`http://127.0.0.1:8765/?token=…`); the page keeps it
  for the tab, removes it from the address bar and sends it with every
  request, and the server refuses a request without it, one from another site
  (`Origin`, `Host`), or a body that is not JSON.
- **A failed *Open* put an empty model on the file's path**, so that the next
  *Save* overwrote the script with nothing. A file that cannot be read back
  is now refused with its reason (422), and the model and the path are left as
  they were.
- **Two matrix phases dropped one from the script without a word.** The page
  read a flag the server never sent (`is_matrix`), so it showed no phase as the
  matrix and let two be declared. The matrix is now the phase whose amount is
  the rest, as the model has it; choosing another one turns the former matrix
  into an inclusion, and a model with two is refused with both names.
- **`mfhstudio()`'s message for a missing studio** said the studio was not part
  of the released package; it is, under `tools/mfhstudio`.

### Changed

- **The port.** Without `port` (in `mfhstudio`, or `--port` in the shell) the
  studio takes 8765, or the next free port when 8765 is taken, by a port
  forward or another studio; a port given is used or refused as before.
  `mfhstudio(; port)` defaults to `nothing` instead of `8765`.
- **A light or a dark theme**, from a button in the header: as the system has
  it by default; the plots follow the theme.

## v0.17.3 — Residues on multiple roots, and the bridge from B to H

A patch release. The residue algorithm, reached by an explicit
`method = :residues`, lost its accuracy, gave a wrong value without warning, or
refused the case, whenever roots of the acoustic polynomial coincided or nearly
did: an isotropic or transversely isotropic reference along the ellipsoid axes,
a reference close to one, and flat ellipsoids, whose roots all shrink with the
aspect ratio. The default `method = :auto`, which uses a cubature, was not
affected. Defects outside the residues are fixed too: `compliance_from_cod`
returned a wrong ℍ from 𝐁 for all but one component, the cubature of 𝐁 broke
down at tight tolerances, the self-consistent scheme converged to a wrong
tensor wherever the transversely isotropic or orthotropic estimate had a
negative or zero coupling, and symbolic spheroids, ellipsoids and cylinders
were refused. Errors below are relative, against the cubature or a closed form,
as measured on 0.17.2 and on this release.

### Fixed

- **`cod_tensor(...; method = :residues)` dropped multiple roots.** A root
  where the derivative of the acoustic polynomial vanished was skipped, and its
  contribution lost, where the residue at a multiple pole was due. For an
  isotropic matrix entered as a general tensor, which has a triple root, the
  COD tensor came out wrong by 1.0 for a penny-shaped crack, 0.73 for a ribbon,
  1.0 for an ellipse of aspect 0.5; a transversely isotropic matrix with a
  double root, by 1.1e-7. All are now within 1e-12.
- **`hill_tensor(...; method = :residues)` on multiple and close roots.** Roots
  closer than 1e-3 in absolute terms were declared coincident and moved to their
  mean, an error of the order of their distance. A transversely isotropic matrix
  around a sphere was wrong by 8.2e-3 with C66 = C44 (1 + 1e-3), a cubic matrix
  around a 3:1:1 prolate spheroid by 8.4e-6, a double root away from `i` by
  1.9e-5. An isotropic matrix entered as a general tensor, around a sphere or a
  spheroid, a transversely isotropic one with C11 = C33, and one with a double
  root at `i` were refused with an `ArgumentError`. All are now within 3e-12, and
  the isotropic ones within 1e-13 of Eshelby's tensor.
- **Flat ellipsoids.** The roots shrink with the aspect ratio, so that the same
  absolute threshold grouped distinct roots: a flat ellipsoid of aspect 1e-3 in a
  triclinic matrix was wrong by 1.3e-2, and flatter ones were refused. Down to
  1e-6 they are now within the tolerance of the outer quadrature, 1e-15 at
  tight tolerances.
- **The root finder.** On the polynomial of an isotropic reference around a
  sphere, with two triple roots, the iterative root finder returned, at some
  angles, five roots out of six far from any root. The roots are now the
  eigenvalues of the companion matrix, which the QR algorithm finds with a
  backward error at the rounding level.
- **`compliance_from_cod` was right on ℍ₃₃₃₃ only.** It placed the components
  ``H_{i3k3}`` of ``k\,\underline{n}\stackrel{s}{\otimes}\boldsymbol{B}\stackrel{s}{\otimes}\underline{n}``
  and then averaged over eight index permutations as if all were filled, so
  that, the normal being ``\underline{e}_3``, ``H_{1313}`` came out as
  ``k B_{11}/16`` instead of ``k B_{11}/4`` and ``H_{1333}`` as ``k B_{13}/4``
  instead of ``k B_{13}/2``, which its docstring presented as a property of the
  framework. It now forms the symmetrized product, as
  `inverse_property_contribution` always did, and is the exact inverse
  of `cod_from_compliance`, which was right. Both dispatch on the shape family
  of the crack, so that a user-defined crack declaring its `shape_trait` has
  them too.
- **The cubature of 𝐁 at tight tolerances.** Along the normal coordinate
  ``\xi_n = \rho\cot\alpha`` the kernel cancels down to ``O(\sin^2\alpha)``
  as ``\alpha \to 0`` and lost ``\epsilon/\sin^2\alpha`` of its accuracy: asked
  for 1e-13, the adaptive quadrature chased the rounding noise down to
  ``\alpha \approx 10^{-150}`` and stopped on a `DomainError`. Below
  ``\sin\alpha = 0.05`` the kernel is now taken in a form free of cancellation;
  at tight tolerances the cubature then agrees with the residues to 1.6e-15.
  This concerns `:nestedquadgk`, the default, and `:decuhr` alike.
- **A reference without a Green operator.** A reference whose acoustic tensor is
  singular along a direction of the integration, one without shear stiffness for
  instance, was integrated on a polynomial cut down to a lower degree, which
  turned an undefined integral into a number. It is refused with an
  `ArgumentError`, as is a reference whose acoustic tensor changes sign.
- **The self-consistent iteration clamped couplings.** Before each step, the
  guard that keeps the running estimate away from the trivial null fixed point
  raised every stored coefficient below a small threshold to that threshold.
  Those of an isotropic or a cubic tensor are its eigenvalues; those of a
  transversely isotropic (Walpole) or an orthotropic tensor include couplings,
  which are negative for an auxetic material and zero for an uncoupled one, and
  clamping them moved the fixed point. With phases of Poisson's ratios -0.4 and
  -0.3 around aligned 3:1 spheroids at 40 %, `SelfConsistent()` returned a
  tensor that missed its own condition by 1.3e-3, and
  `SelfConsistent(algorithm = NewtonDefault())` one that missed it by 1.0, the
  coupling at -2052 instead of -6.8. The guard now reads the spectrum block by
  block and raises only the eigenvalues below the threshold, keeping their
  eigenvectors; both algorithms meet the condition to 1e-11. Its isotropic
  method, declared on the wrong type parameter, had never applied.
- **Symbolic inclusions.** `Symbolics.Num` subtypes `Real`, and the Newton
  potentials, the classification of a cylinder and the stopping tests of
  Carlson's integrals were routed on `T <: Real`: the Hill tensor of a symbolic
  spheroid, triaxial ellipsoid or cylinder, and the construction of a symbolic
  `Cylinder` itself, threw `TypeError: non-boolean (Num) used in boolean
  context`, where the docstrings announce symbolic support. They are now routed
  on whether comparisons return a `Bool`. The spheroids take their closed forms
  in `acos` and `acosh`; the triaxial ellipsoid keeps its incomplete elliptic
  integrals as unexpanded calls, which the Symbolics extension now registers as
  it did the complete ones. SymPy, whose `Sym` is not a `Real`, was not
  affected.
- **A crack given in numbers under symbolic moduli lost its exact closed form.**
  The factors of the circle (π/4, π/2) were evaluated in the arithmetic of the
  crack geometry, Float64, so that the COD tensor of `PennyCrack(1.0)` in a
  matrix of symbolic `E`, `ν` carried `1.6977…` in place of `16/(3π)`. They are
  evaluated in the arithmetic of the moduli when it is symbolic, and a ratio given
  as `1.0` enters as the integer it is: the result is Sneddon's closed form,
  exactly.
- **Cylinders and ellipses close to circular.** The Newton potentials of an
  elliptic cylinder and of an ellipse switched to the circular forms within
  1e-6 (relative) of a circular section, although the elliptic forms hold down
  to it: up to 5e-7 in value, and the whole derivative along the difference of
  the semi-axes, were lost there. The circular forms are now taken only for
  semi-axes that are one, value and derivatives; the potentials agree with the
  limit of an elongated ellipsoid to 1e-12, in value and in derivative.
- **`cumulative_strain_average` beyond the outer radius** returned the average
  over the sphere alone, normalized by the sphere's volume, for a ball that
  takes in the surrounding medium too. A radius beyond the outer one, outside
  the documented domain, is now refused.
- `check_material_interface` reported a tangent check skipped at a zero tangent
  as "non-smooth".
- **The derivative of the Hill tensor of a very flat ellipsoid was NaN** at some
  flatnesses below 5e-6, by the cubature (`:nestedquadgk`, the default for an
  anisotropic matrix, and `:decuhr`), its value being right. The cubature
  variable was promoted to the dual type before `√(1 − z²)`, whose derivative at
  the node where `1 − z²` rounds to zero is `0·∞`. The cubature variables now stay
  plain numbers, and `1 − z²` is formed from the exact `1 − z`. The first-order
  term of the Hill tensor in the flatness, by ForwardDiff, then gives the crack
  compliance by the block computation of Barthélémy (2009) within 1e-8 of the
  one from the COD tensor.
- `method = :residues` with a dual, symbolic or complex stiffness or geometry
  failed with a `MethodError` deep in the kernel; it is refused with an
  `ArgumentError` that points to the cubature, which `:auto` selects and which
  differentiates.

The residues of roots that coincide or nearly do are now summed as a group, by
the expansion of the contour integral around it, which depends on the roots only
through symmetric functions of their offsets and is well conditioned; for
coincident roots it is the residue at a multiple pole. The inner integral of the
Hill tensor is taken in the variable of Masson's substitution `t = cot(θ/2)`, in
which `i` is a double pole like any other, with its numerators in factored form,
so that no digit is lost near `t = ±1` or `t = 0`, where the poles of flat and
long ellipsoids lie. The derivation is on the theory page of Hill tensors.

### Changed

- The residue algorithm runs twice as fast: 1.9 ms for the Hill tensor of a
  3:2:1 ellipsoid in a triclinic matrix, against 4.1 ms, with 2.7 MiB allocated
  against 7.0 MiB; 1.3 ms for the COD tensor of a penny-shaped crack, against
  2.7 ms. The acoustic polynomial, its adjugate and the numerators are products
  of fixed size rather than polynomial objects, and the 21 numerators of the
  Hill tensor are integrated through their coefficients, as a bilinear form.
- The `PolynomialRoots` dependency is dropped.
- `newton_potential_3d` and `newton_potential_2d` accept semi-axes of mixed
  types, as `newton_potential_3d_cylinder` did.
- `fe_available_gb` takes the file to read, `/proc/meminfo` by default.
- The differential scheme no longer intercepts a failure of the Hill tensor at
  its first steps to advise against `method = :residues`: the residues now take
  the isotropic values a fully anisotropic running medium starts from, and any
  other failure reports its own message.

### Tests

- The residue sums over groups of poles against partial fractions in 512-bit
  arithmetic; the inner integral of the Hill tensor against a quadrature of the
  same rational function; the Hill tensor on multiple and nearly multiple roots
  against the cubature and Eshelby's tensor; the COD tensor of an isotropic
  matrix entered as a general tensor against the closed forms. A test that
  pinned the failure of the residues for a crack whose normal is perpendicular
  to the axis of a transversely isotropic matrix now checks their agreement with
  the cubature there.
- A test file of the residue algorithm on aligned transversely isotropic and
  orthotropic matrices was never included in the test suite; it is now.
- Every route to ℍ and 𝐁 in a triclinic matrix: 𝐁 from the reduced kernel by
  the cubature, DECUHR and the residues, and ℍ from the flattening of an
  ellipsoid with the Hill tensor by the same three, extrapolated to zero
  flatness, compared both ways through the bridge. They agree to 1e-14 among
  the integrations of 𝐁, and to 1.5e-9 with the extrapolated flattening.
- The bridge between ℍ and 𝐁 on every component, against the symmetrized
  product written out from its definition.
- The guard of the self-consistent iteration against the full Kelvin–Mandel
  spectrum, for every class it knows; the self-consistent estimate of auxetic
  phases against its own condition, evaluated without the guard.
- Symbolic Hill tensors of spheroids, triaxial ellipsoids and cylinders, in
  elasticity and conduction, and the symbolic Newton potentials, against the
  numerical ones.
- The potentials of a cylinder near a circular section against those of an
  elongated ellipsoid, with their derivatives.
- 𝐁 of an elliptical and of a ribbon-like crack in an elliptically orthotropic
  and in a transformed transversely isotropic (monoclinic) matrix, by the three
  integrators, against the transformation rule of Barthélémy (2026) applied to
  the closed forms: within 2e-13. The derivative with respect to a modulus, by
  ForwardDiff through the cubature, against the same rule.
- Branches no test reached: the regularized and the Picard steps of
  `NewtonDefault`, aging viscoelastic cracks with an interface stiffness against
  their elastic limit, the unfinished self-consistent iterations of aging
  viscoelasticity, the Rabotnov model against its Mittag-Leffler form, the
  complete elliptic integrals of a complex parameter, and the refusals of
  assemblies, laminates, finite elements and neural surrogates.

### Documentation

- A tutorial, *Two routes to a crack: the COD tensor and the flattened
  ellipsoid*: 𝐁 from the crack plane and ℍ from the first-order term of the Hill
  tensor of a flattened ellipsoid, inverted on the kernel of the leading term,
  met through the bridge. Exact in SymPy in an isotropic matrix, where both give
  Sneddon's crack; with the first-order term by ForwardDiff in random anisotropic
  matrices, where they meet within 3e-8; with the residues and DECUHR beside the
  nested cubature; and against the transformation rule of an elliptically
  orthotropic matrix.
- The page on COD tensors shows the comparison of every route in a triclinic
  matrix, and proves the identity ℍ = k n ⊗ˢ 𝐁 ⊗ˢ n step by step in the
  normalization of the package (𝐁 and the flatness both by the half-width b, as
  in Barthélémy, 2026): the strain a cavity adds carried by its surface, Eshelby's
  uniform strain, the collapse of the surface onto the lips, and k = cS/V. The
  normalization by the semi-major axis of Barthélémy et al. (2021, 2023) and
  Echoes follows from it, with tensors η times those of the package and the same
  factor 3/4.
- The COD tensor of an arbitrarily anisotropic matrix is described as what the
  code computes: the inverse of an integral of the reduced kernel
  ``\hat{\boldsymbol{Q}}^{\star}_{nn}``, after Barthélémy (2026), a line integral
  along the normal (Kanaun and Levin, 2009). The docstrings attributed it to the
  flattening of an ellipsoid (Barthélémy, 2009), an independent route to the
  same tensor, and its residue evaluation to Masson (2008), whose paper does not
  treat cracks.

## v0.17.2 — Mori–Tanaka in aging viscoelasticity, on aligned inclusions

A patch release. The aging linear viscoelastic Mori–Tanaka scheme gave a wrong
stiffness whenever its fast transversely isotropic or orthotropic path was taken,
that is for inclusions aligned with the canonical frame and not all spherical.

### Fixed

- **`homogenize_alv` with `MoriTanaka()` composed its two operators in the wrong
  order on the transversely isotropic and orthotropic paths.** The scheme is
  ``\tilde{\mathbb{C}}_0 + \bigl(\sum_i f_i \tilde{\mathbb{N}}_i\bigr)\circ
  \bigl(f_0\,\mathbb{I} + \sum_j f_j \tilde{\mathbb{A}}_j\bigr)^{-\circ}``, the
  inverse on the right, and the general path computes it so; the two fast paths
  divided on the left. Neither the normal block of a transversely isotropic or
  orthotropic operator (a 2×2 or 3×3 matrix) nor the time axis of an aging one
  commutes, so the answer was wrong even in the elastic limit, where the
  shear terms alone came out right: by 0.4 % of the largest stiffness for oblate
  spheroids of aspect 0.3 at a contrast of 2 and 20 % inclusions, by 4.3 % for
  an aspect of 0.1 at a contrast of 10 and 30 %, by 4.6 % for 10:1 fibers at a
  contrast of 20 and 30 %. The isotropic path, spherical inclusions, and every
  other scheme were right. Both paths now agree with the general one to the
  rounding, in the elastic limit and on random aging operators.

### Tests

- Every ALV scheme on each path its dispatch takes by symmetry — transversely
  isotropic, orthotropic and general — is held to its elastic limit, and the
  structured Mori–Tanaka and Maxwell paths to the general algebra on random
  aging operators. A test of the orthotropic path claimed to check Mori–Tanaka
  and Maxwell and checked neither.
- The Ferrite material backend is tested: the plane-strain element on a thick
  cylinder against Lamé's solution, converging at second order, and the coupled
  ``(\underline{u}, p)`` element on the integrals it must assemble, which a field
  linear in the coordinates reproduces exactly.

### Tooling

- The Gridap backend is left out of the coverage report: CI does not install
  Gridap, whose Gmsh reader pulls MPI and Metis, and its tests run where the
  stack is installed.

### Documentation

- The Ferrite backend page says first what the finite-element code and the
  Gauss-point law each do.

## v0.17.1 — JSON.jl 1 in place of the deprecated JSON3.jl

A patch release. JSON3.jl is deprecated: its README now sends its users to
JSON.jl 1, which absorbed its features, and it receives no further releases.
The package used it in a single place, the files of the neural-inclusion
surrogates, but as a strong dependency, loaded by every
`using MeanFieldHomogenization`. It now uses JSON.jl instead, with no change to
the API, to the results or to the files.

### Changed

- **The JSON dependency is JSON.jl 1, no longer JSON3.jl.** `save_surrogate`
  writes with `JSON.json(io, payload; pretty = 4)` and `load_surrogate` reads
  with `JSON.parsefile`. The surrogate format is unchanged (1.2.0): the shipped
  models, and any model saved by an earlier version, load as before and give
  the same predictions bit for bit; a model saved now also loads in an earlier
  version. Keys are looked up by string, which every JSON.jl 1.x release
  supports, hence the bound `JSON = "1"`.

### Tooling

- The benchmark harness (`scripts/bench`), the comparisons against Echoes
  (`scripts/bench_echoes`) and the MFH Studio sidecar read and write JSON with
  JSON.jl as well. The environment of `scripts/bench_echoes` now declares it:
  its scripts loaded JSON3 without declaring it, so they only ran from an
  environment that happened to provide it.
- **The MFH Studio sidecar warned, on its first catalog, trace and parse
  requests, that its code would error in a future Julia.** It reached the handlers defined at run time through
  `Base.invokelatest(catalog)`, which calls the latest method but reads the
  name `catalog` in the caller's world, before the binding exists there.
  `Base.@invokelatest catalog()` also looks the name up in the latest world;
  the replies are unchanged.
- **The coverage is measured on the latest release of Julia, whatever its
  number.** The CI named its versions '1.12' and '1.13' and measured the
  coverage where the version was '1.12', a literal the matrix had to keep
  listing: TensND sent Codecov nothing for a month once its matrix and its
  conditions no longer agreed. The matrix now names the aliases 'min' (the
  oldest minor that Project.toml admits), '1' and 'pre', which no release of
  Julia makes stale, and the coverage runs on '1'. The depot caches of the
  tests, the documentation and the format check are keyed on the installed
  version as well, so that a new release does not carry the previous one's
  `compiled/` along, and the job that merges the coverage of the four parts
  fails when none of them produced any, rather than warning.

### Documentation

- **The norm of a deviator carries no factor.** The conventions page now
  defines σ_d = √(σᵈ:σᵈ), and ε_d likewise, as in Barthélémy (2005), and the
  nomenclature follows. The pages that reproduce an article — the sandstone of
  Dormieux et al. (2010), the granular medium of Maalej et al. (2009), the
  mortar of Pichler and Hellmich (2011) — keep the ½ under the root of the
  article and say so, with the conversion; the map of the local fields of a
  layered sphere shows σ_d rather than the von Mises stress.
- **The strength ellipse of the porous-strength tutorial was displayed
  inverted.** The page and `scripts/40_porous_strength_criterion.jl` printed
  x²/(2A) + y²/B = 1/(1 − φ), while the code draws 2A x² + B y² = 1 − φ. The
  formula now matches the code, which is unchanged and gives back eq. (5.33) of
  the thesis for spherical pores under Mori–Tanaka.
- **The quasi-brittle strength page showed a formula off by √2 and a wrong
  strength.** With M = S:∂ℂ/∂μ:S it wrote 2μ² where the article's eq. (22) gives
  μ²; the code, which differentiates with respect to 2μ, was right. And it said
  that Pichler and Hellmich calibrate σ_ult to 70–90 MPa, where they identify
  about 50 MPa (their eq. 31); the model gives 49.1 MPa from their
  f_c = 54.1 MPa.
- **Levin's theorem is proved on two drawn decompositions**, with prestresses
  under a macroscopic strain and with free strains under a macroscopic stress,
  and with averages: Σ* = ⟨σ*:𝔸⟩ and E* = ⟨ε*:𝔹⟩ for fields of any form, the
  sums over the phases coming last. The free strain is proved directly.
- The Hill tensor tutorial writes P₁₁₁₁ of a sphere as
  1/(5(λ + 2μ)) + 2/(15μ).

## v0.17.0 — concentration tensors of every scheme, Levin's theorem, one function whatever the order

A minor release, and a breaking one. Every scheme now returns, besides the
effective tensor, the concentration tensors of each of its phases, and with them
the contribution of each phase to the estimate and, by Levin's theorem, the
response of the composite to free strains and prestresses of its phases: a
thermal expansion, or the Biot tensor of a solid made of several phases or
holding pressurized cracks, which no single-solid formula covers.

The release also settles the names of these objects. The package now says
*concentration tensor*, the English term of the literature, where it used to
say *localization*, a gallicism. And each notion is now one function whatever
the order of the tensors, as `homogenize` already was: a concentration tensor, a
contribution, its dilute seam, Levin's theorem, the fields of a layered sphere
or of a laminate and their aging-viscoelastic counterparts are each computed by
a single generic function, named after the gradient and the flux, which serves
elasticity on fourth-order properties and conduction on second-order ones by
dispatch on what it receives. Until now the two physics had parallel families of
names, so that code written for one could not run the other, and a method added
for a new inclusion had to be written twice. The names of elasticity remain, and
those of conduction for the contributions, as aliases restricted to their order.

### Breaking changes

- **The compat bound must be widened.** Below 1.0, Julia's resolver treats a
  minor bump as breaking: a package declaring `MeanFieldHomogenization = "0.16"`
  does not accept 0.17.0 and must widen its bound to `"0.16, 0.17"`, or move to
  `"0.17"` once it uses the new names.
- **TensND 0.6.2 is required.** The concentration tensors of a scheme divide
  isotropic tensors by the weight of a phase, which may be a symbolic number,
  and that division was ambiguous between TensND and Symbolics before 0.6.2.
- **Every *localization* name is renamed, without alias**, to the generic
  function that now serves both orders. Where elasticity keeps a name of its
  own, it is given in the last column; it accepts fourth-order arguments only.

  | Former names | Generic function | Name in elasticity |
  | :-- | :-- | :-- |
  | `strain_strain_loc`, `gradient_gradient_loc` | `gradient_gradient_concentration` | `strain_strain_concentration` |
  | `stress_strain_loc`, `flux_gradient_loc` | `flux_gradient_concentration` | `stress_strain_concentration` |
  | `strain_stress_loc`, `gradient_flux_loc` | `gradient_flux_concentration` | `strain_stress_concentration` |
  | `stress_stress_loc`, `flux_flux_loc` | `flux_flux_concentration` | `stress_stress_concentration` |
  | `local_strain_strain_loc`, `local_gradient_gradient_loc`, and the six other pointwise fields | `local_gradient_gradient_concentration`, … | `local_strain_strain_concentration`, … |
  | `layer_strain_localization`, `layer_gradient_localization` | `layer_gradient_concentration` | `layer_strain_concentration` |
  | `layer_stress_localization`, `layer_flux_localization` | `layer_flux_concentration` | `layer_stress_concentration` |
  | `laminate_<…>_localization` (eight) | `laminate_gradient_concentration`, `laminate_flux_concentration`, `laminate_flux_gradient_concentration`, `laminate_gradient_flux_concentration` | `laminate_strain_concentration`, … |
  | `strain_strain_loc_alv`, `gradient_gradient_loc_alv` | `gradient_gradient_concentration_alv` | `strain_strain_concentration_alv` |
  | `loc_and_stiffness`, `loc_and_stress_average`, `compliance_and_stiffness_contribution` | `concentration_and_contribution`, `concentration_and_flux_average`, `inverse_and_property_contribution` | |
  | `shell_localization`, `cluster_localizations` | `shell_concentration`, `cluster_concentrations` | |
  | `fe_cell_localization`, `fe_axi_localization`, `fe_axi_pore_localization` | `fe_cell_concentration`, `fe_axi_concentration`, `fe_axi_pore_concentration` | |
  | `bulk_localization_alv`, `shear_localization_alv`, `gradient_localization_alv` | `bulk_concentration_alv`, `shear_concentration_alv`, `gradient_concentration_alv` | |
  | `NeuralLocalizationInclusion` | `NeuralConcentrationInclusion` | |
  | `StrainLocTI`, `StressLocTI`, `StrainLocCubic`, `GradLocISO2`, `GradLocTI2` | `GradientConcentrationTI`, `FluxConcentrationTI`, `GradientConcentrationCubic`, `GradientConcentrationISO2`, `GradientConcentrationTI2` | |

- **The names of one physics now forward to a generic function.** A call of the
  right order is unchanged; a call of the other order, which raised a
  `MethodError`, now raises an `ArgumentError` naming the generic function.
  What breaks is a method added to one of these names, for instance
  `stiffness_contribution` for a new inclusion type: the schemes call the
  generic function and no longer reach it, so it must be added to the generic
  function instead.

  | Names kept | Generic function |
  | :-- | :-- |
  | `stiffness_contribution`, `conductivity_contribution` | `property_contribution` |
  | `compliance_contribution`, `resistivity_contribution` | `inverse_property_contribution` |
  | `delta_stiffness`, `delta_conductivity` | `delta_property` |
  | `delta_compliance`, `delta_resistivity` | `delta_inverse_property` |
  | `layer_strain_average`, `layer_stress_average` | `layer_gradient_average`, `layer_flux_average` |
  | `sphere_strain_average`, `sphere_stress_average` | `sphere_gradient_average`, `sphere_flux_average` |
  | `local_strain`, `local_stress` | `local_gradient`, `local_flux` |
  | `layer_stiffness_average`, `layer_conductivity_average` | `layer_property_average` |
  | `layer_compliance_average`, `layer_resistivity_average` | `layer_inverse_property_average` |
  | `stiffness_contribution_alv`, `conductivity_contribution_alv` | `property_contribution_alv` |
  | `stiffness_contribution_alv_at`, `compliance_contribution_alv` | `property_contribution_alv_at`, `inverse_property_contribution_alv` |
  | `delta_stiffness_alv`, `delta_compliance_alv` | `delta_property_alv`, `delta_inverse_property_alv` |

- **A laminate layer reads `:C` by default in both physics.**
  `layer_gradient_concentration(lam, name)` and `layer_flux_concentration` read
  the stiffness, as every other function of the package does; the conduction
  call, which `layer_gradient_localization(lam, name)` made by default, now
  passes `property = :K`.
- **A `CustomInclusion` takes four callbacks instead of eight.** They are
  passed as `gradient_gradient_concentration`, `flux_gradient_concentration`,
  `property_contribution` and `inverse_property_contribution`, and each receives
  the properties of either order, as `hill_tensor` already did. An inclusion
  that had distinct elastic and conduction callbacks merges them into one that
  dispatches on the order of `P₀`; a callback typed on one order serves that
  physics only, which `check_inclusion_interface` now reads by dispatch.
- **The class keys of a neural surrogate follow the new names.** `:loc_ti`,
  `:stress_loc_ti`, `:loc_cubic`, `:grad_loc_iso2` and `:grad_loc_ti2` become
  `:gradient_concentration_ti`, `:flux_concentration_ti`,
  `:gradient_concentration_cubic`, `:gradient_concentration_iso2` and
  `:gradient_concentration_ti2`, the suffix `2` marking the classes of
  second-order tensors as in `HillTI2`. The model files shipped with the package are
  updated; a model file saved by an earlier version must have its `"class"`
  field renamed the same way before it loads.
- **Documentation addresses.** The theory page `th-concentration`
  (`theory/concentration`) replaces `th-localization`, and the API page
  `api/concentration` replaces `api/localization`; links to the former addresses
  no longer resolve.

### Added

- **`concentration_tensors(cell, scheme, property)`**, returning a
  `ConcentrationTensors`: the gradient and flux concentration tensors of every
  phase, ⟨ε⟩_r = 𝔸_r : E and ⟨σ⟩_r = 𝔸_σε,r : E in elasticity, with the
  effective tensor they are consistent with. Every scheme on an `RVE` is covered
  — the bounds, the dilute estimates, Mori–Tanaka, Maxwell, Ponte
  Castañeda–Willis, both self-consistent schemes and the differential scheme,
  whose tensors are integrated along the path together with the medium — as
  well as the exact laminate and the cluster model and equivalent inclusion
  method on a `ParticleAssembly`, in elasticity and in conduction, crack
  families included per unit density. Every result satisfies Σ w 𝔸 = 𝕀 and
  Σ w 𝔸_σε = ℂ^hom. Until now only the dilute tensors, and those of a few cells,
  were available, and the mean fields of the phases under the actual scheme had
  to be rebuilt by hand.
- **Readers and contributions.** The four concentration functions read a
  `ConcentrationTensors` by phase name, in the gradient-driven and flux-driven
  forms. `property_contribution` and `inverse_property_contribution` give the
  contribution of a phase to the estimate, ⟨σ − ℂ_ref : ε⟩_r = ℕ_r : E,
  relative to the matrix of the scheme or to any phase or tensor passed as
  `reference`. The names of elasticity and of conduction read it too.
- **Levin's theorem.** `levin_free_gradient` turns eigenstrains of the phases
  into the macroscopic free strain — a thermal expansion tensor, a swelling —
  and `levin_preflux` turns prestresses into the macroscopic prestress, a pore
  pressure giving the Biot tensor. The latter accepts crack families, so that
  the pressure in the cracks contributes, and holds for any number of solid
  phases, where `biot_tensor` assumes a single one. On a conduction estimate the
  same two functions take eigengradients and pre-fluxes. In elasticity they also
  answer to `levin_free_strain` and `levin_prestress`.
- **The mean flux of a layered sphere in conduction**,
  `sphere_flux_average(sphere, K₀, ∇T∞)`, which had only its elastic method.
- The closed-form schemes run on `ForwardDiff.Dual`, nested, and on SymPy and
  Symbolics numbers; the iterative ones are numeric, as their effective tensors
  are.

### Fixed

- **`DifferentialScheme` in conduction threw a `DimensionMismatch` as soon as
  its estimate became fully anisotropic**, for instance with two families of
  spheroids tilted differently. The state of a fully anisotropic conductivity
  was laid out on nine components and rebuilt from six; it now holds the six
  components of a symmetric tensor. Elasticity was not affected, and the
  transversely isotropic case, which never reached that state, is unchanged.
- **The thermal COD of a circular crack threw in an isotropic conductor stored
  as a dense tensor.** The discriminant of the eigenvalues it needs was written
  tr² − 4 det, which rounds below zero there, so that `sqrt` threw; the
  differential scheme in conduction met it as soon as its running conductivity
  was dense, with a crack family and an inclusion breaking the transverse
  isotropy. The discriminant is now a sum of squares, and near the circular
  point a series in the discriminant replaces its root, so that derivatives of
  every order exist there.
- **The conduction Hill tensor of a sphere was not exact on symbolic
  conductivities.** It read the geometric factor 1/3 off the Newton-potential
  tensor of the sphere, formed in the `Float64` of its radius, so that a SymPy
  estimate carried `0.333…` and left a residue of order 1e-16 in identities that
  should vanish. The factor is now formed in the element type of the
  conductivity; on `Float64` the value is unchanged to the bit.

### Documentation

- **Concentration tensors in the theory.** The overview of the schemes defines
  the concentration tensors of an estimate, the averages they satisfy, the
  contributions relative to any reference and the one-site construction
  𝔸_i = 𝔸_i^dil : 𝔼; every scheme page ends with its own tensors, and the
  page on the differential scheme derives the equations along the path that
  carry them.
- **A page on Levin's theorem**, at the end of the chapter on schemes: the
  statement, its proof by Hill's lemma, the exact two-phase formula, which holds
  for every scheme of the package, and the Biot tensor with pores and cracks.
- **A tutorial, Thermal expansion and Biot tensor**: the thermal expansion of a
  two-phase composite under five schemes, its anisotropy with aligned platelets,
  the Biot tensor of a solid with pressurized cracks, and both results in closed
  form with SymPy. The manual gains the sections *Concentration tensors of the
  phases* and *Several solid phases* (poromechanics).
- **The names of elasticity and of conduction** are tabulated against the
  generic functions in the theory of concentration tensors and on each API page
  concerned.
- Three references added, each resolved on Crossref: Dvorak & Benveniste
  (1992), Rosen & Hashin (1970) and Dormieux, Kondo & Ulm (2006). Levin's own
  article of 1967 is not entered: it has no DOI, and the reference lists
  deposited with the first two disagree on its year.
- Two formulas of the theory, in the Ponte Castañeda–Willis and thermal crack
  pages, were broken by a line starting with a minus sign, which Markdown reads
  as a list item; they render as formulas again.

### Changed

- The test suite runs in four parts, `core`, `schemes`, `patterns` and
  `visco-neural`, selected by `Pkg.test(test_args = [...])` and run as four
  parallel CI jobs; with no argument every part runs. A single job had grown
  past the time limit of the CI on Julia 1.12. The raw coverage of the four
  parts is merged before it is processed, so that the report is the one a
  single job would give, and the parts of a Julia version share one depot
  cache. The test summary gives the time of every group.

## v0.16.3 — a void has a compliance contribution

A patch release. The compliance contribution of an inclusion with no stiffness,
and the resistivity contribution of one with no conductivity, came out as NaN,
and so did `DiluteDual` and the compliance form of the differential scheme on
pores and insulators. They are now finite and exact. The theory chapter is
reorganized: the problems of a single heterogeneity come first, then one page per
family of schemes, each with its figure, its assumption and its equations, and
the Maxwell and Ponte Castañeda–Willis schemes are derived from the Green
operator. Writing them out turned up statements in the documentation and the
docstrings that were wrong, all corrected.

### Fixed

- **`compliance_contribution(incl, C₁, C₀)` returned NaN for a void
  (`C₁ = 0`).** It evaluated its definition (𝕊₁ − 𝕊₀) : 𝔸_σσ literally, an
  infinite compliance times a zero stress concentration. It now reads
  ℍ = −𝕊₀ : ℕ : 𝕊₀ from the stiffness contribution, an exact identity that
  never inverts `C₁` and was already the route of heterogeneous inclusions.
  `resistivity_contribution` does the same for an insulator (`K₁ = 0`). For a
  spherical void the result is the closed form, checked in value, under
  `ForwardDiff` and with `Symbolics`; for every other inclusion it agrees with
  the former value to rounding. `DiluteDual` and
  `DifferentialScheme(; formulation = :compliance)`, which call these functions,
  now accept pores: the first returns the closed form, and the second agrees
  with the stiffness form.

### Documentation

- **The theory chapter reads in a new order.** The Eshelby problem, with the
  interaction of two inclusions; the cracks, with the degenerate shapes; the
  layered patterns; and only then the homogenization schemes. The overview of
  the schemes states what every scheme estimates, the identities between
  contribution tensors and the way a crack family enters, and gathers the
  schemes in one table. Each family then has its own page, with its figure,
  its assumption and its equations: bounds, dilute estimates, Mori–Tanaka,
  Maxwell and Ponte Castañeda–Willis, self-consistent schemes, then the
  differential scheme, the laminate, the cluster model and the equivalent
  inclusion method, whose pages keep their content. The transitions between
  pages are rewritten to follow that order.
- **Maxwell and Ponte Castañeda–Willis are derived.** From the far field of an
  inhomogeneity for Maxwell, and from the field inside the distribution
  ellipsoid for Ponte Castañeda–Willis, in the notation of Barthélémy et al.
  (2019). Two proofs show that the two coincide for a single distribution shape,
  and Mori–Tanaka is shown to be the choice of a distribution shaped like each
  source phase, which is not symmetric. The aging viscoelastic page restates
  every scheme in Volterra form.
- **A spherical distribution reproduces Mori–Tanaka only for spherical
  inclusions.** The manual, two docstrings and the error message of an
  undeclared distribution said it did for any inclusion; with spheroids the two
  estimates differ.
- **Docstrings corrected.** Mori–Tanaka does add the crack term to its
  denominator; the self-consistent scheme does handle cracks, with the prefactor
  π for a ribbon family; it cites Hill (1965) and Budiansky (1965), not a paper
  on the differential scheme; Voigt and Reuss cite Hill (1952); the aging
  viscoelastic Maxwell and dual dilute kernels are stated as implemented, and the
  Maxwell table of the aging viscoelastic page no longer inverts the estimate.
- Nine references added, each resolved on Crossref; `hill1965` and `willis1982`
  carry their DOI.

## v0.16.2 — tensors compared as tensors

A patch release. The documentation, the tutorials and the tests compared
tensors through the arrays of their components, which `get_array` and
`t[i, j, k, l]` give in each tensor's own basis: the difference of two such
arrays means something only when the two bases coincide. They now compare the
tensors themselves, which holds in any basis. Two internal kernels are written
in static algebra, equal to the bit and faster. No result changes.

### Changed

- The isotropic conduction Green operator, (𝟏 − 3 n⊗n)/(4πσ₀r³) in 3D and
  (𝟏 − 2 n⊗n)/(2πσ₀r²) in 2D, was built entry by entry into a heap array then
  converted to a static one. It is now formed in static algebra, with the same
  products in the same order: equal to the bit, 51 → 38 ns in 3D and
  45 → 26 ns in 2D, one allocation instead of three.
- The second moment of an ellipsoid is Q diag(aᵢ²/(d + 2)) Qᵀ through
  `Diagonal`, 103 → 78 ns, equal to the bit. It takes `transpose(Q)`, not
  `Q'`, which would conjugate a symbolic frame.
- `TensND` is bounded to `"0.6.1"` (package, tests and documentation). The
  documentation uses `𝐞(3, 3, Float64)`, which works from that release.

### Tests and documentation

- Pages and published scripts use the tensor API instead of rewriting it:
  `norm`, `≈` and `==` on tensors, `⊡` for invariants (𝟏 ⊡ ℂ ⊡ 𝟏 / 9), `KM`,
  `tens_cubic`, `best_fit_iso`, `components_canon`, `𝐞₃ ⊗ 𝐞₃`. A 300 × 300
  mask that was the identity is `I`, and a standard error of the mean is
  `std`. The 15 tutorial pages are regenerated from their scripts; every
  modified script runs. `Statistics` is declared in the documentation
  environment.
- Tests compare tensors with `≈`, or through `components_canon` where a
  tolerance was set on the largest component. A loop of one `@test` per
  component is one test. Invariants are written as tensors. The independent
  references a test checks against (an explicit contraction by components,
  the 3⁸-term rotations, the closed forms of Mura and Eshelby by components)
  are kept as written, since they are what makes the check independent.

## v0.16.1 — the default method resolved at compile time, NestedQuadGK by default, and degenerate shapes exact

Every call of `hill_tensor` and `cod_tensor` with the default `method = :auto`
paid a dynamic dispatch and a dynamic keyword call, and handed its caller a value
of type `Any`: the default was not propagated as a constant down to the `Val`
that selects the algorithm. It now is, with results identical to the bit. And
the default cubature for an anisotropic reference is now the nested Gauss–Kronrod
one, measured both faster and more accurate than DECUHR on every case tried.
Comparing the two on very flat ellipsoids, and checking the closed forms against
a 256-bit quadrature, turned up defects in all three. The cubature stopped short
on flat shapes. The closed forms of a spheroid, in an isotropic matrix and in a
coaxial transversely isotropic one, were wrong near the sphere, and the latter
gave wrong derivatives at an isotropic point. Following them up to every shape
where two semi-axes meet, or a spheroid becomes a disc or a needle, turned up
more: wrong derivatives with respect to a semi-axis, a triaxial ellipsoid next
to a spheroid, a crack next to a circle, two traps of the cubatures under
`ForwardDiff`, and a self-consistent estimate that lost its symmetry class. All
are treated, and a new theory page, *Degenerate shapes*, says how.

### Changed

- `method = :auto` for an anisotropic three-dimensional elastic reference — Hill
  tensors and crack COD tensors — is `NestedQuadGK`, whatever is loaded. It was
  `DECUHR` as soon as its extension was loaded. Against closed forms, and
  against tight references checked by a second algorithm, `NestedQuadGK` is
  within 1e-15 to 2e-12 where `DECUHR` is within 1e-11 to 6e-8, at two to twelve
  times less cost. Results computed with DECUHR loaded move by at most that
  former error, and the default no longer depends on what the session has
  loaded. `method = :decuhr` remains available.

### Fixed

- The Newton potentials of a spheroid, behind every closed form in an
  isotropic matrix (Hill, Eshelby and concentration tensors, 𝕌ᴬ and 𝕍ᴬ,
  conduction), were wrong near the sphere. The potential I₃₁ was formed as the
  divided difference (I₃ − I₁)/(a² − c²) of two closed forms that cancel there.
  The Hill tensor of a spheroid was off by 1e-5 at ω = 0.9999 and 1.0001, by
  2e-3 at |ω − 1| = 1e-5, and by 0.7 to 2.8 times its own size at
  |ω − 1| = 2e-6; below 1e-6 the sphere values were returned. The potentials are
  now evaluated as the integrals they stand for, through the regimes described
  below. The Hill tensor of a spheroid now agrees with the tight cubature to
  5e-16 at every distance from the sphere tried, from 1e-9 to 1e-2.
- A triaxial ellipsoid whose two closer semi-axes nearly coincide went through
  elliptic integrals and their divided differences, which lose accuracy as
  1/δ² with the relative gap δ: up to 5e-5 at δ = 1e-6, where the two
  semi-axes were taken as equal instead. Below a gap of 2 % (in squares) the
  potentials are now expanded around the spheroid they are close to, with
  spheroidal coefficients: within 1e-15 of the cubature for gaps of 1e-9 to
  1e-2, next to a flat, an elongated or a near-spherical spheroid.
- The derivative with respect to a semi-axis where it meets another one was
  silently wrong, by 100 % to 550 %: the shape was classified as a sphere or a
  spheroid, whose closed form no longer depends on the gap. Two semi-axes
  carried by dual numbers are now merged only when their derivatives coincide
  too, for ellipsoids, ellipses, cylinders and cracks, and the expansion above
  carries the derivative. The derivatives agree with those taken through the
  cubature to 2e-14.
- The closed-form Hill tensor of a spheroid coaxial with a transversely
  isotropic matrix, the `:auto` path for that case, was wrong in three regimes.
  The errors below were measured against a 256-bit quadrature of the same
  integrals.
  - **Near-spherical spheroid, isotropic or nearly isotropic matrix.** The error
    was 1e-5 at ω = 0.9999 and 1.0001. At ω = 1 ± 1e-6 the relative error
    reached 6 to 23. The closed forms cancel as η → 1, and a limit value was
    used up to |η² − 1| = 1e-6.
  - **Flat spheroid, ω ≤ 1e-4.** The error was about 0.1 ω (3e-5 at
    ω = 1e-4). The test for coincident roots had an absolute floor that took any
    two small roots for equal ones.
  - **Matrix within one part in a million of isotropy.** It was evaluated as
    isotropic. Values were off by up to 7e-7. ForwardDiff derivatives along the
    directions that break isotropy were off by 24 % to a factor 14, at an
    isotropic point included. This concerns any TI reference medium that is
    isotropic in value. The same test, with its own absolute floor, also took
    moduli below about 1e-4, in whatever unit they are given, for isotropic
    ones.

  The integrals are now evaluated from the mean and the squared half-difference
  of the two roots, which are rational in the moduli. Three regimes apply: a
  power series near the sphere, an expansion in the squared half-difference near
  coincidence, and elsewhere the closed forms with a recurrence for flat shapes.
  Against the same reference the error is now below 1e-14 for 1e-8 ≤ ω ≤ 10, on
  every matrix tried and on isotropic values. Derivatives near an isotropic
  point agree with a 256-bit reference to 4e-14 for ω ≤ 3 and to 1e-10 at
  ω = 30. Needles, ω ≫ 1, lost accuracy as ω², the Walpole coefficients being
  differences of terms ω² or ω⁴ larger than the result: 5e-7 at ω = 10⁴, and
  derivatives 7e-7 off at ω = 30. They are now written with ∫z²(1−z²)/(D₁D₂)
  and ∫(1−z²)²/(D₁D₂), computed directly, and for a needle the part of these
  integrals rational in the roots is summed exactly: within 1e-14 up to
  ω = 10⁶, and derivatives within 2e-10 at ω = 10³.
- Every cubature differentiated by `ForwardDiff` could stop with a
  `DomainError`: QuadGK's error estimate goes through `LinearAlgebra.norm`,
  which rescales by the largest entry unless it `iszero`, and a dual number of
  value zero with non-zero partials is not. On a segment where the Kronrod and
  Gauss rules agree to the last bit the estimate came out `NaN` — the
  derivative of the anisotropic conduction tensor of a sphere, for one. The
  cubatures now hand QuadGK the norm of the values: the refinement makes the
  decisions of an evaluation without derivatives, whose value it reproduces.
- The derivative with respect to a crack semi-axis through the crack cubature
  was a `MethodError`: its integration bounds were taken in the number type of
  the crack, and QuadGK has no rule for a dual interval. They are plain numbers
  now.
- The COD tensor of an elliptical crack close to a circle formed
  𝒮 = (𝒦 − ℰ)/k² and 𝒞 = 𝒦 − 𝒮, which cancel as k² → 0 (an error near ε/k²,
  1e-6 at the threshold where the crack became a `PennyCrack`). Below k² = 0.1
  they are now the series of the complete integrals: within 1.1e-15 of the same
  closed forms in 256-bit arithmetic. 𝐁, normalized by the smaller semi-axis,
  has a kink at the circle; the derivative AD returns there is the one-sided
  one in the direction of the increase, which the documentation now states.
- The self-consistent scheme rebuilt a transversely isotropic estimate as a
  generic tensor at every iterate: products of TI tensors land in the six- or
  eight-parameter form, which its major symmetrization did not recognize. The
  Hill tensor of a coaxial spheroid then went through a cubature, as accurate
  as its tolerance. The projection is now made within the class, onto the
  five-parameter form; on a 3:1 spheroid the scheme takes 46 µs instead of
  4.2 ms, and its fixed point is the closed-form one.
- `NestedQuadGK` stopped with an error far above the tolerance asked for on
  very flat ellipsoids. At the default `reltol = 1e-6` the error was 2e-5 at
  ω = 1e-5 and 2e-6 at ω = 1e-6. The integrand also varies over a layer
  z ~ ω at the end of the interval that the change of variable does not
  stretch, and the first Gauss–Kronrod nodes missed it. Below ω = 1e-3,
  breakpoints at z = ω, 10ω and 100ω now give that layer intervals of its own.
  The error falls to 2e-8 to 5e-7, and results for ω ≥ 1e-3 are unchanged to
  the bit. DECUHR misses the tolerance on these shapes as well: its error is
  1e-6 to 2e-5 at ω = 1e-5. At tolerances 1e-13/1e-12 it is still 1.3e-6 to
  1.6e-6, where `NestedQuadGK` reaches 3e-16 at a hundredth of the cost.
- `hill_tensor(…; method = :residues)` on a reference whose acoustic polynomial
  has roots of a multiplicity the residue sums do not handle — a reference
  anisotropic in type but isotropic or transversely isotropic in value along the
  ellipsoid axes — threw QuadGK's `DomainError` ("integrand produced NaN"), and
  the fallback to `NestedQuadGK` written for that case could never be reached.
  It now throws an `ArgumentError` that says what failed and which method
  evaluates the same integral; the dead fallback is gone.
- A ribbon crack under `DECUHR` failed ("DECUHR failed: retcode = Failure"):
  DECUHR integrates over rectangles of dimension two or more, and the ribbon's
  reduced kernel is a single one-dimensional integral. With DECUHR loaded this
  was the default for a ribbon crack in an anisotropic reference, which the
  self-consistent and differential schemes produce. `:decuhr` on a ribbon now
  evaluates that integral by Gauss–Kronrod, as `:residues` on a cylinder falls
  back to the quadrature that applies.

### Performance

Same machine, 0.16.0 → 0.16.1, results identical to the bit except on the
spheroid, whose Newton potentials are now evaluated otherwise (see *Fixed*):

- `hill_tensor` of a sphere in an isotropic matrix 132 → 37 ns and
  `cod_tensor` of a penny crack 280 → 88 ns, both now inferred and allocation
  free;
- Mori–Tanaka on a sphere 1.35 → 1.04 µs, self-consistent 20.6 → 17.7 µs,
  differential 49.1 → 42.5 µs, Mori–Tanaka on a spheroid 1.67 → 1.4 µs;
- the first `cod_tensor` call of a session 276 → 50 ms.

The structural reasons: `_with_method` lifts `:auto` at compile time and
dispatches any other method once, on its `Val`, into a kernel call with
positional arguments; `Base.@constprop :aggressive` on `hill_tensor`,
`cod_tensor`, `sif`, `dif` and `interaction_tensor` carries the default down to
it; the closed-form COD tensors are built with their dimension in their type.
Julia 1.12 propagates such a constant for inference only. It compiled the entry
of `cod_tensor` for a run-time `Symbol`, whose other branch returns an abstract
type, and boxed the result: 64 bytes per call, caught by the allocation test on
1.12. The entry points of `hill_tensor`, `cod_tensor` and `interaction_tensor`
are now inlined down to `_with_method`, so the literal selects the branch at
compile time on 1.12 and 1.13 alike.

### Documentation

- *Degenerate shapes*, a new theory page: every situation above where the
  closed forms or the cubatures fail — a spheroid next to the sphere, a
  triaxial ellipsoid next to a spheroid, the derivative where two semi-axes
  meet, the transversely isotropic closed form at the sphere, at coincident
  roots, for a disc and a needle, a crack next to the circle, the cubatures
  under `ForwardDiff`, the self-consistent estimate of TI class — what goes
  wrong, how it is evaluated instead, and the accuracy measured.
- COD tensors: the calculation of 𝐁 in an isotropic matrix, which the
  literature quotes, carried out from the Fourier kernel to the closed form —
  the reduced kernel and its three coefficients, the contour integral, the
  inversion, the circle and the ribbon. The angles are those of
  barthelemyIJSS2026 throughout: its contour for the integral, its definitions
  of 𝒦, ℰ, 𝒞 and 𝒮, related by one change of variable.
- Thermal cracks: the contractions that were missing in a dozen tensor
  products are written out.
- Hill tensors: the proof of the surface-element identity, the Kelvin–Mandel
  matrices of 𝕌ᴬ and 𝕍ᴬ,
  the definitions of the elliptic integrals, why ℙ is positive
  definite, and an account of the three algorithms with their current costs,
  the long parts folded.
- COD tensors, after the appendices of the now published barthelemyIJSS2026
  (International Journal of Solids and Structures 341 (2026) 114293), cited
  under that key: the general formula of 𝐁 through the reduced
  kernel and the Fourier conventions behind it, the isotropic ribbon and the
  transversely isotropic closed forms in engineering parameters as implemented,
  K = 4 Q̂*ₙₙ(ν)·N and
  the energy release rate as a quadratic form, other parametrizations of the
  front. The page stated that every source normalizes 𝐁 by the
  half-width; the earlier papers do not, and the warning now says so.
- The performance notes and the dispatch rationale give the current costs of
  the anisotropic back-ends.

## v0.16.0 — derivatives everywhere, and the anisotropic Hill cubatures ×90

An audit of the package against automatic differentiation, its own invariants
and the Julia performance tips. Derivatives now go through the multipole
interaction tensor, the anisotropic Hill tensors of conduction and of the nested
cubature, the orientation of inclusions and laminates, the aging-viscoelastic
schemes whatever phase carries them, and they converge in the self-consistent
schemes. Four results were wrong without saying so and are corrected (see
Breaking changes). The anisotropic Hill cubatures run up to ninety times faster,
with results identical to the bit, and the package holds no boxed captured
variable any more.

### Breaking changes

- Below 1.0 a minor release is breaking for the resolver: packages with
  `MeanFieldHomogenization = "0.15"` must widen their bound to accept 0.16.
- TensND 0.6 and DECUHR 0.3 are required.
- Results that change because they were wrong:
  - `AsymmetricSelfConsistent` in conduction ignored the crack phases: their
    resistivity was never added, and a matrix with insulating cracks returned
    the matrix conductivity. They are now added, as in elasticity.
  - The aging-viscoelastic laminate (`homogenize_alv(lam, Laminated(), :C)`)
    with a normal other than e₃ returned its operator in the layer frame; it is
    now in the canonical frame, as every other ALV operator and as the elastic
    `Laminated` result.
  - A crack aligned with the axis of a TI medium took the closed form whatever
    `method` was asked; an explicit numerical method is now honored, as for an
    ellipsoid.
  - The derivatives of the self-consistent schemes were judged converged on the
    values alone. From a start at the solution (`init`), the derivative with
    respect to a volume fraction was 2 % off with `AndersonDefault` and zero
    with `NewtonDefault`; from the default start `NewtonDefault`'s was 6e-8 off.
    Each partial must now meet the tolerance of the value.
- The NonlinearSolve extension takes the exact ForwardDiff Jacobian instead of
  a central finite difference: the derivatives of SC solves through SciML
  algorithms move at the level of the former truncation error.
- The 2D anisotropic Hill tensor is exactly symmetric; its values move by 3e-17
  relative at most.

### Fixed

- Derivatives with respect to a semi-axis or an orientation through the
  multipole interaction tensor (the second moment was stripped to Float64
  before `eigen`, and a second derivative threw).
- A dual semi-axis in the nested-QuadGK Hill tensor (`Float64(ω)` threw).
- The anisotropic conduction Hill tensor under ForwardDiff: its value keeps the
  closed form, its derivatives come from an integral representation that needs
  neither `eigen` nor `svd`, at a repeated eigenvalue too.
- A dual Euler angle (`_normalize_euler` converted it to Float64), a dual
  rotation matrix given to a constructor, and a dual frame whose axes are
  sorted.
- A dual or symbolic laminate normal in the elastic `Laminated` scheme; the ALV
  laminate rotation in the type of its frame.
- The order-4 ALV schemes, and their iso/TI/orthotropic fast paths, with a dual
  on any phase but the first (MethodError).
- Work counters and the memoized Gauss–Legendre rules under threads (atomic
  counters, a lock).
- Docstrings detached from their definitions by a comment (`_sc_initial`,
  `_bulk_state_seq`, `_ti_identity`); the detached docstring of `add_phase!`
  described the matrix phase of an RVE that no longer has one.
- `differential_alv` took an inclusion declared `fraction = :rest` for a crack
  family and refused it as one; it now says that this phase has no amount to
  integrate up to, as the elastic and the order-2 differential schemes do. The
  asymmetric self-consistent ALV scheme checks up front, as the other ALV
  schemes do, that a phase carrying a crack density is a crack.

### Performance

Same machine, before → after, results identical to the bit:

- nested-QuadGK Hill tensor 75 → 0.85 ms (an ellipsoid 3:2:1), 102 → 1.06 ms
  (a flat one), 79 → 2.9 ms under a dual semi-axis; allocations 103 MB →
  0.26 MB. The stiffness array, assigned twice and captured by the integrand,
  was boxed, so every access in the innermost loop was a dynamic one;
- DECUHR Hill tensor 18.3 → 2.7 ms (22.6 → 8.3 ms under a dual); cylinder
  0.39 → 0.021 ms; 2D 14.9 → 11.5 µs: the integrand is an `SVector` on the
  stack, shared by the three cubatures;
- no `Core.Box` left in the package and its extensions (fourteen sites before:
  the residue Hill kernel, five ALV TI/orthotropic schemes, the ALV SC schemes,
  the cracked material, layered spheroids, the Prony fit, the gmsh crack weld).

### Internals

- Run-time type tests turned into dispatch wherever the type holds the
  answer: the kind of amount of a phase (`VolumeFraction`, `CrackDensity`,
  `Remainder`) selects its contribution, its Mori–Tanaka term, the keyword that
  declares it again and its display; the layout of the axis sort
  (`Val(:crack)`, `Val(:cylinder)`, `Val(:ellipsoid_3d)`); the interface of a
  layered spheroid in conduction; the average a bound assigns to a layered
  sphere in the ALV schemes, which no longer computes the Voigt one only to
  discard it under Reuss; the frame of an ALV laminate; the order of an ALV
  property. The ALV crack corrections call `crack_density_factor` instead of
  repeating it. What stays a test is a test on a value (a keyword, the mode of
  a law), a guard that throws, or a filter over a heterogeneous collection.

### Documentation

- The ChemistryLab bound of the documentation environment raised to 0.38.
- The benchmark against Echoes (`developer/benchmarks`) measured again on this
  version: the nested-QuadGK cubature now runs 95 to 745 times faster than
  Echoes' NUMINT3D, DECUHR at the speed of the residues. Its environment
  declares DECUHR and Integrals, which the script imports and which it lacked.

## v0.15.0 — the interfaces of the layered sphere, checked to the aging Volterra setting

### Breaking changes

- Below 1.0, a minor release is breaking for the resolver: packages with
  `MeanFieldHomogenization = "0.14"` must widen their bound to accept 0.15.
- `kelvin_iso(k₀, μ₀, …)` now builds the compliance its docstring states,
  `1/(3k₀)` and `1/(2μ₀)`; it built `1/k₀` and `1/μ₀`. Results computed with
  it change (see Fixed).
- The order-2 ALV `Maxwell` scheme takes the distribution shape the RVE
  declares, and raises when it declares none, as at order 4 and in
  elasticity; it used a sphere whatever the RVE said.
- Several ALV results change because they were wrong (see Fixed): layered
  spheres with springs between viscoelastic media, creep laws in layered
  spheres, the Voigt and Reuss bounds of layered spheres, penny cracks in an
  aging matrix, and the order-2 Mori–Tanaka estimate.


The concentration tensors of a `LayeredSphere` with imperfect interfaces were
wrong, and every scheme consumes them. The opening of a spring interface did not
strain the composite sphere, so any estimate on spring-bonded grains was off (a
Mori–Tanaka bulk modulus of 2.30 instead of 1.66 on a two-spring example), and a
Kapitza interface lowered the resistance it was meant to add. In the aging
viscoelastic recurrence, a spring between two viscoelastic media composed their
moduli in the wrong order. Results computed with `SpringInterface` or
`KapitzaInterface` on a `LayeredSphere` should be recomputed. All now agree with
Echoes to at least ten digits and with closed forms of the literature, which
the three new strength applications derive symbolically from the library.

Checking the aging viscoelastic side against Echoes on an aging law and a
non-uniform time grid, where Volterra matrices stop commuting, found four more
products composed in the wrong order and three paths giving silently wrong
results; all are fixed. The layered sphere now has its conduction counterpart
in that setting, and the self-consistent schemes take it.

### Fixed

- **The opening of a spring interface did not strain the composite sphere.**
  The concentration tensor of a `LayeredSphere`, which every scheme consumes,
  averaged the strain over the material of the layers and so left out the
  displacement jump across its `SpringInterface`s. The stress side was right,
  the traction being continuous, so only the strain-average rule broke, and
  with it every estimate: a two-layer sphere with two springs gave a
  Mori–Tanaka bulk modulus of 2.30 where the right value is 1.66. The jumps now
  count, those of the inner interfaces always and that of the outer one by
  default, which is the convention of Echoes, of `LayeredSpheroid` and of the
  laminates. The tensors reproduce Echoes to ten digits (inner and outer
  springs, Mori–Tanaka, self-consistent) and the self-consistent closed forms
  of Dormieux et al. (2010) for bonded grains. The ALV kernels
  (`strain_strain_loc_alv`, `stiffness_contribution_alv`) had the same gap and
  are fixed the same way. The averages of `averages.jl`
  (`sphere_strain_average`, …) are unchanged: they average over the material,
  and their docstrings now say that the opening is not part of it.
- **A Kapitza interface on a `LayeredSphere` was a negative resistance.** Its
  jump matrix carried `+ρ` on a state whose flux is the physical
  `q = −k ∇T`, so the temperature rose in the direction of the flux: a grain
  conducted better with the resistance than without, and diverged at
  `ρ = r/k`. It now drops, `[T] = −ρ q_n`, and the concentration tensors match
  Echoes, the Hasselman–Johnson equivalent grain and the spherical limit of the
  confocal spheroid, which had the right sign all along. The jump is, as
  everywhere, the outer value minus the inner one, and the minus sign is that
  of the dictionary `σ ≡ −q`: the Kapitza law is the analog of the spring law,
  `[T] = ρ σ_n = −ρ q_n`. The pages wrote `[T] = ρ q_n`, mixing the physical
  flux with the law written for its analog; they now say where the sign comes
  from. The surface-conductive law had its sign flipped the same way in two
  tutorials and on the spheroid page, while the code was right.
- **An incompressible layer (`k = ∞`) failed** with "matrix contains Infs or
  NaNs" as soon as the sphere had an imperfect interface or a second layer,
  although the theory page promised the limit. Modes 2 and 4 of the deviatoric
  recurrence are now scaled by `μ/(k + μ)`, which makes every entry affine in
  that ratio and exact at `k = ∞`; the mean pressure of such a layer is read
  from the traction instead of `∞·0`. The pointwise stress of an incompressible
  region, which the strain does not determine, now raises an `ArgumentError`
  instead of returning `NaN`.
- **A spring interface between two viscoelastic media composed their moduli
  in the wrong order** in the aging linear viscoelastic (ALV) recurrence of
  `LayeredSphere`. One of the four terms of the bulk transition across a
  `SpringInterface` multiplied the Volterra matrices of the inner and outer bulk
  moduli in reverse order, which is harmless only when they commute: on a
  uniform time grid with non-aging laws, or when one side is elastic. On a
  non-uniform grid, a logarithmic one included, or with an aging layer, the
  bulk concentration tensors and the effective relaxation were wrong, by 0.2 to
  0.4 % in the cases checked. The defect dates from v0.1.0. With springs or
  membranes, an aging core and a non-uniform grid, the ALV concentration
  tensors of the layers and of the whole sphere, with or without the outer
  interface, and the dilute and Mori–Tanaka estimates now match Echoes to
  machine precision.
- **A creep law in a `LayeredSphere` was read as a relaxation.** The ALV
  recurrences of the layered sphere discretized the laws of its layers and of
  the matrix without looking at their mode, so a law given in `:creep` mode, a
  compliance, entered as a stiffness: 5 to 63 % off on a one-layer sphere that
  the ellipsoid path, which inverts it, got right. Both paths now agree
  whichever mode each law is given in.
- **The ALV Voigt and Reuss bounds gave a `LayeredSphere` the stiffness of a
  dilute estimate** (`C₀ + N`), 34 % and 15 % off Echoes on an aged
  two-layer sphere. They now take the Voigt and Reuss averages of its layers,
  as the elastic bounds do, and match Echoes to machine precision.
- **`kelvin_iso(k₀, μ₀, …)` built a compliance three times too large on the
  spherical part and twice too large on the deviatoric one**: `1/k₀` and
  `1/μ₀` where its docstring, its argument names and `maxwell_iso` all mean
  `1/(3k₀)` and `1/(2μ₀)`, and the same for every Kelvin branch. Its test had
  pinned the wrong values. It now returns the inverse of
  `iso_stiffness(k₀, μ₀)` at `t = t′`. Results computed with `kelvin_iso`,
  MFH Studio's "Kelvin chain" included, change accordingly. The manual's
  example passed numbers where the branches are vectors.
- **The COD tensor of a penny crack in an aging matrix composed its Volterra
  factors in the wrong order**, so the ALV crack contributions were off by up
  to 1.6 % on an aging matrix (exact in the elastic limit and for a non-aging
  matrix on a uniform grid). The factors now come in the order of the flat
  limit of a void spheroid, whose Hill kernel holds no product; the crack
  estimates match that limit and Echoes to machine precision.
- **The order-2 (conduction, diffusion) ALV Mori–Tanaka estimate put the
  inverse of `f₀ 1 + Σ f A` on the left of `Σ f N`**, against its own
  docstring and the order-4 scheme: 0.5 % off Echoes with an aging inclusion.
  It is now on the right; Mori–Tanaka equals Maxwell for one family of
  spheres, as it must, and every order-2 scheme matches Echoes to machine
  precision, or to the tolerance of its fixed point. The differential scheme
  agrees with Echoes' explicit stepping, which converges to it as one over the
  number of steps. The asymmetric self-consistent scheme, which has no order-2
  implementation, is refused by name.
- An incompressible layer (`k = ∞`), or an infinite conductivity, in the ALV
  path now raises an `ArgumentError` naming the layer, instead of failing
  inside a factorization: it has no Volterra matrix, and the elastic functions
  treat it exactly. The docstring of `homogenize_alv` and the manual said the
  self-consistent scheme accepted a `LayeredSphere`, which failed with a
  `MethodError`; it now does accept one (see Added).
- MFH Studio generated `maxwell_iso(k, μ, τ)` and `kelvin_iso(k, μ, τ)`, which
  no signature accepts, for a property saved with the older `visco` field.
- **A layered sphere with an imperfect interface failed on SymPy moduli.**
  `Matrix{T}(I, 4, 4)` holds the logical `True` and `False` when `T` is `Sym`,
  and they refuse to multiply. Once past that, the pivoted solve of the
  deviatoric recurrence swelled for more than ten minutes on a single grain.
  The mode matrix is now inverted in closed form (its determinant,
  `350 μ² (s + 3)² / r⁴`, never vanishes): the symbolic concentration tensors of
  a spring-bonded grain take about a second, and the numeric recurrence no
  longer pivots.
- **A self-consistent iteration seeded with an infinite Voigt average** (a rigid
  or incompressible phase) now says so and how to seed it, instead of failing
  inside a linear solve.
- **echoes2mfh** translated a transport `PRIMALDISC` interface into a
  `SpringInterface`, and a transport `DUALDISC` into a `MembraneInterface`: they
  become `KapitzaInterface(1 / h)` (Echoes gives a conductance) and
  `SurfaceConductiveInterface`. A layer modulus containing a comma, such as
  `5.0 * one(TensISO{2, 3})`, was split into two fragments and lost its tuple.
- **MFH Studio** labeled the Kapitza field `h`, the symbol of a conductance,
  for the resistance `KapitzaInterface` takes.

### Added

- `external` on the whole-sphere concentration and contribution tensors of
  `LayeredSphere` (`strain_strain_loc`, `stress_strain_loc`,
  `stiffness_contribution`, `gradient_gradient_loc`, `flux_gradient_loc`,
  `conductivity_contribution`, and the ALV kernels): `false` leaves the outer
  interface to the matrix, both its opening and the surface stress of a
  membrane, as Echoes' `sphere_eE(n - 1, external = False)` does.
- `external` and `internal` on the per-layer forms
  `strain_strain_loc(sphere, C₀; layer)` and
  `gradient_gradient_loc(sphere, K₀; layer)`, with Echoes' meaning
  (`layer_eE`); the default stays the average over the material. With
  `external = true` on every layer the shells partition the sphere.

- **Aging linear Volterra conduction of a `LayeredSphere`**, with Kapitza and
  surface-conductive interfaces: `gradient_localization_alv`,
  `gradient_gradient_loc_alv` and `conductivity_contribution_alv`, the order-2
  twins of `bulk_localization_alv`, `strain_strain_loc_alv` and
  `stiffness_contribution_alv`, and a layered sphere in every order-2 scheme of
  `homogenize_alv`. A jump is the outer value minus the inner one, and the
  Kapitza law `[T] = ρ σ_n = −ρ q_n` takes its sign from the dictionary
  `σ ≡ −q`. Echoes' n-layer sphere reference being mechanical, the recurrence
  is checked against exact identities on an aging core: the elastic limit, the
  interface conditions solved as one system, and the equivalent sphere of a
  Kapitza or surface-conductive interface.
- **The order-2 self-consistent and Ponte Castañeda–Willis schemes**
  (`self_consistent_alv_order2`, through `homogenize_alv(rve,
  SelfConsistent(), :K; times)`), matching Echoes.
- **The self-consistent and asymmetric self-consistent ALV schemes take a
  `LayeredSphere`** at order 4, its concentration and stress coming from the
  recurrences against the running estimate; they match Echoes.

### Changed

- ALV now stands for **aging linear Volterra**, the setting of a linear,
  hereditary, possibly aging behavior written with Volterra operators:
  viscoelasticity at order 4 and conduction or diffusion at order 2. The API
  keeps its names; the theory page is *Aging linear Volterra behavior*.
- US spelling throughout: *aging*, in the prose, the docstrings, the scripts
  and the file names. The application page is now `applications/aging_creep`,
  the tutorial `tutorials/generated/aging_ages_aspect`, the scripts
  `53_aging_creep_solid.jl` to `57_aging_creep_cracks.jl` and
  `87_aging_ages_aspect.jl`, and the MFH Studio example `09_aging_creep.jl`;
  links to the former documentation URLs no longer resolve. The titles of the
  cited works keep their spelling, and so do the earlier sections of this file.
- `stiffness_contribution(sphere::LayeredSphere, C₀)` returns the
  three-argument tensor `stiffness_contribution(sphere, C₀, C₀)`, which it
  duplicated; the two may differ in the last digit from before.

### Documentation

- Three applications reproduce published strength models, each with its
  paper cited at the top and its published values checked on the page:
  - **The elastic limit of concrete**, after Königsberger, Pichler and
    Hellmich (2014): ITZ failure and ITZ–aggregate separation around the
    aggregates. The stress in the ITZ comes from the Hadamard jump conditions
    written as one tensor equation, with the flat Hill tensors of the laminate
    cell, rather than component by component; it is also the thin-shell limit
    of the coated `LayeredSphere`. Every value of the article is recovered,
    and the elastic limit surface is drawn in principal stress space.
  - **Friction of a granular medium**, after Maalej, Dormieux and Sanahuja
    (2009): rigid grains bonded by springs, the self-consistent moduli derived
    symbolically (their eqs. 39 and 42 come out of the library), and the
    Drucker–Prager slope by automatic differentiation through the
    self-consistent solve.
  - **The strength of a sandstone**, after Dormieux et al. (2010): von Mises
    grains and Mohr–Coulomb contacts, with eqs. (49) to (54) derived
    symbolically, and the strength envelopes of the modified secant method.
- The Applications chapter is organized by subject: general concepts first
  (interacting particles, morphologies computed by finite elements), then the
  cementitious materials, the geomaterials and the bituminous mixtures, each
  page coming after those it builds on, with the transitions rewritten to
  match.
- The data these pages take from articles live in `data/literature/*.json`,
  with their source, location and transcription, instead of in the pages.
- The title of `konigsberger2013` in the bibliography had the wrong subtitle.
- The docstrings of the ALV interface transfers said that an interface
  parameter could be a `ViscoLaw`. The interface types hold numbers, so an
  interface is constant in time; the docstrings say so, and the method that
  would have read an aging parameter, which nothing could reach, is removed.
- The `SelfConsistent` docstring states the floor its positivity guard sets:
  every component of the iterate is kept above `√eps` times the largest
  component of the seed, so a solution far below the seed is out of reach and
  wrong without notice; seed near the expected moduli in that case.
- A pull request no longer builds the whole site in one job: a draft build of
  the full tree checks the references, and the pages are built in parallel
  shards (`docs/shards.jl`); the full build, with a longer time limit, runs on
  `main` and deploys.
- The README and the module docstring described `LayeredSpheroids` as
  conduction only, which it has not been since v0.11.0: they now name the
  elastic confocal spheroid, prolate or oblate, with perfect interfaces.
- Citations are author-year, as in ChemistryLab: a `[42]` told the reader
  nothing without a trip to the References page. A citation that is the subject
  of a sentence or follows a preposition reads "Eshelby (1957) showed", the
  others "(Eshelby, 1957)", and adjacent ones are grouped in one parenthesis.
  Hovering a citation still shows the full entry.
- No citation is left inside a figure caption; the source of a drawing is
  given in the sentence before it.
- The Echoes entry cited a version DOI that Zenodo marks as deprecated, dated
  2022 although that version was issued in 2025. It now cites the concept DOI
  `10.5281/zenodo.7348758`, the one the Echoes repository gives, which resolves
  to the current version (v1.0.0, 2026).
- The key of Parrott and Killoh (1984) misspelled the first author's name.
- A **Getting started** page opens the site: the installation, and one
  estimate taken from the RVE to the effective moduli with each object defined
  on the way. It replaces the Manual's installation page; the instructions for
  running the test suite move to the developer's testing page.
- A **Nomenclature** page holds the typography of the formulas, the operators
  and the rules every page and docstring follows for decorations (`hom`, the
  indices of the reference medium and of a phase, upright labels, the Eshelby
  tensor). The theory page that held the typography keeps the conventions and
  is now called *Conventions*. The home page names three reading paths, and the
  API is visible again in the navigation bar.
- One symbol per quantity across the pages and the published scripts: an
  effective property is written `^{hom}` (it was written six ways, `eff`
  included), the reference medium carries the index 0 and a phase its index as
  a subscript, labels are upright, vectors are underlined without hats, order-2
  tensors are bold. Each formula of the site now typesets under the
  documentation's own MathJax configuration.
- Four formulas said something other than what the code does or than their
  neighbors on the same page:
  - the spring interface of the layered sphere wrote the displacement jump as
    `k_n σ_rr`, `k_n` being the stiffness; it is `σ_rr/k_n`, as the code has it;
  - `𝔹_i` meant stress per strain on the differential-scheme page and stress per
    stress on the laminate page; the first is now the `𝔸_{σε}` of the
    localization page;
  - `𝕁` was both the spherical projector and the creep kernel, on the same
    viscoelasticity page; the kernels are now the relaxation `ℂ(t, t')` and the
    creep `𝕃(t, t')`;
  - `𝕊` was both the compliance and the Eshelby tensor; the latter is `𝕊ᴱ`.
- MFH Studio labels the conductivities `k`, as the documentation writes them,
  instead of `κ`.
- The formulas of the docstrings are written in LaTeX under the same notation,
  so that the API pages render them and the hints below read them. Docstrings
  that disagreed with the code are aligned on it: the state vector of the shear
  recurrence, the type returned by `layer_gradient_average` and by
  `spheroid_strain_concentration`, and the kinds accepted by `legendre_table`.
- Hovering an equation shows the symbols it holds, with their meaning on the
  page, from the new Nomenclature page's source, `docs/nomenclature.toml`. A
  letter with several meanings (`p`, `φ`, `ω`, `𝔹`) is explained only on the
  pages where each holds. `test/Documentation` checks the nomenclature and the
  typography of the formulas of the pages and of the docstrings.
- Every page opens with a "Before this page" note and closes with a "Where to
  go next" section, and the H2 sections of the theory pages are numbered. The 188
  relative `.md` links became `@ref` links on anchors, and the 22 pages that had
  no title anchor have one. The manual says imperfect interfaces exist for the
  elastic layered sphere as well as in conduction, and the differential scheme
  is called `DifferentialScheme()` in its example.
- The isotropic stiffness is `3λ𝕁 + 2μ𝕀` on the Hill-tensor page; it read
  `3λ𝕀 + 2μ𝕂`.
- Documentation builds can be limited: `MFH_DOCS_PREFLIGHT_ONLY=1` runs the checks
  in under a minute, and `MFH_DOCS_ONLY=<patterns>` builds only the matching pages.

## v0.14.2 — where the sample count stops paying, and the mesh that produced it

0.14.1 shipped the layered-spheroid surrogates on 1200 finite-element solves and
recorded that the sample count was the lever that worked, each factor of 1.7
buying a factor of three in the held-out block error. This release finds where
that ends, by measuring two further steps instead of extrapolating from one.

### It ends, and three points say so rather than one

| training solves | block rms, `𝔸_εε` | gain | exponent `α` in `rms ∝ N^-α` |
|--:|--:|--:|--:|
| 400 | `7.3e-3` | — | — |
| 700 | `2.4e-3` | 2.98× | 1.95 |
| 1200 | `7.8e-4` | 3.11× | 2.10 |
| 2000 | `4.8e-4` | 1.63× | 0.95 |
| 2800 | `4.0e-4` | 1.22× | 0.59 |

The exponent falls monotonically, `2.10 → 0.95 → 0.59`. The stress side follows,
`5.5e-4 → 3.3e-4 → 2.8e-4`. The shipped pair is retrained on 2800 and the models
are better, but only marginally: **beyond about 2000 solves more samples no
longer buy anything reliable.**

### And the block error nearly told the opposite story

At 2000 solves the held-out block error had already flattened — `α` fell from 2.1
to 0.95 — which taken alone said the extra solves were wasted. Judged instead
against the closed form on the dense sweep the tutorial plots, that same step
bought between 3× and 13× *at the faces of the sampling box*, where a Halton set
is sparsest and a fit's slope suffers first. The bulk of the box was by then near
the **teacher's** own floor, the cell being exact only to about `1e-4` itself, so
the block error saturated while the tail still fell.

From 2000 to 2800 the two metrics finally agree: the worst derivative over the
sweep improved from 2.5 % to 1.0 %, the worst value went 0.05 % to 0.06 %, and at
the box face the derivative went 0.8 % to 1.2 %. Those are maxima over
twenty-one points — order statistics, and noisy — so the mixture is what
saturation looks like, not a further gain and not a regression.

The lesson generalizes and is the third instance of it in two releases: a block
error over a hundred Halton points measures the bulk, a dense sweep measures the
tail, and the tail is what limits what a surrogate is good for. Read both.

**Refining the mesh remains the wrong lever, arithmetically.** The teacher
reproduces the closed form to about `1e-4` while the fit sits at `4.0e-4`, so the
fit still dominates by a factor of four: refining would buy a floor that is not
being reached, while making each solve some 2.5× more expensive *and* discarding
every label, the checkpoint key including the mesh.

### The mesh the labels were computed on

The tutorial's mesh figure gained a fifth panel, and it is the only one that is
not an illustration: `nradial = 14`, `R/a = 5`, 11 030 cells at the middle of the
sampling box. The other four use `nradial = 16` and `R/a = 3` because that reads
better on a page, so the discretization the shipped surrogates actually learned
from appeared nowhere.

### Added

- 1600 further finite-element labels, and `layered_spheroid_strain` /
  `layered_spheroid_stress` retrained on 2800.
- The training-mesh panel, and the sentence in the tutorial saying which panel
  is the real one.

### Changed

- `docs/Project.toml` accepts `ChemistryLab` 0.16. The dependency is a
  documentation one only; 0.16 breaks nothing this documentation touches — the
  one behavior change is in `solve_certified`, which no page or script here
  calls, while `powers_alpha_max`, which the hydration scripts do call, is
  unchanged. Older bounds are kept alongside the new one so the file resolves on
  either side of a registration.
## v0.14.1 — the surrogate for the layered spheroid, and the frame it exposed

0.14.0 shipped the meshed layered spheroid and said its sensitivities would have
to come from a surrogate. This is that surrogate — and training it found a bug in
the machinery that had been waiting for the first *oriented* localization model.

### A localization surrogate on a prolate shape read the wrong axis

`NeuralLocalizationInclusion` sorted its semi-axes descending and permuted the
basis to match. That is right for a **Hill** surrogate, whose analytic teacher
`Ellipsoid` returns its components in the sorted frame. It is wrong for a
**localization** surrogate, whose teacher is a finite-element cell: the cell's
frame is unsorted, and its response axis is *column 3* by the package's
convention — the column `_class_frame` reads for every localization class, and
the one `FiniteElements._fe_frame` solves about.

Sorting moved that column. `(1, 1, ω)` with `ω > 1` sorts to `(ω, 1, 1)`, the
revolution axis lands in column 1, and `_class_frame` still reads column 3, so
the decoded tensor came back transversely isotropic about an **equatorial**
direction. Measured against the cell on the same body: `15 %` on `𝔸_εε` and
`32 %` on `𝔸_σε`, falling to `5.3e-4` and `3.8e-3` once the sort is removed.

Three reasons it stayed hidden, all three now covered by a test:

- every localization surrogate shipped before this one was trained on a
  **sphere** (`FEExcenteredSphere`), where the sort is the identity;
- the gate-B tests used `(1, 1, 0.4)` — **oblate**, hence already descending,
  hence the identity again;
- the projection residual that catches a wrong axis is measured when *encoding*
  a label, not when decoding a prediction, so nothing threw.

**No correct result changes.** The sort is the identity for a sphere and for any
oblate shape, so only prolate localization inclusions are affected, and for those
the previous answer was wrong.

### `layered_spheroid_strain` and `layered_spheroid_stress`

Trained **on the cell**, because half of gate B exists nowhere else: the analytic
`LayeredSpheroid` supplies no stress side. One solve fills a column of both label
matrices — `𝔸_σε ≠ ℂ₁ : 𝔸_εε` for a heterogeneous inclusion, so meshing twice to
learn two halves of one solve would have doubled an hour of finite elements.

Every feature is a **contrast ratio**, never an absolute modulus: a heterogeneous
morphology carries its constituents inside itself. The pair covers the confocal
**prolate** slice at `ν₀ = 0.2`, `c/a ∈ [1.25, 3]`, `w ∈ [0.2, 0.7]`, both
contrasts in `[0.5, 4]`; `guard = :error` refuses a query outside the box.

Read the **block error**, not the worst component: two of the six components of
`𝔸_εε` change sign over the held-out set, so their relative columns measure
nothing there. On 1200 training solves and 100 held out: rms `7.8e-4`, median
`5.2e-4`, p90 `1.3e-3`, worst `2.4e-3`; the stress side crosses no zero, worst
`2.0e-3`.

**The sample count was measured, not guessed**, and it is the lever that worked:

| training solves | block rms, `𝔸_εε` | block rms, `𝔸_σε` |
|--:|--:|--:|
| 400 | `7.3e-3` | `6.8e-3` |
| 700 | `2.4e-3` | `1.9e-3` |
| 1200 | `7.8e-4` | `5.5e-4` |

Every factor of 1.7 in samples buys a factor of three, so the gain is
**superlinear** — the signature of four features genuinely under-sampled, not of
an estimator at its statistical floor. It will stop: the teacher reproduces the
closed form to about `1e-4`, so that is where the sequence ends, and `7.8e-4` is
close to it. The checkpointing below is what made the experiment affordable at
all — each step reused every solve already paid for, so 1200 samples cost 1200
solves and not 2300.

### The derivative costs ten times the value, and that is arithmetic

On the confocal slice `𝔸_εε` has a closed form, so the derivative has an *exact*
reference rather than a discretized one. Against it, differenced at `h = 1e-4`
and converged to `1e-9`:

| route | value | derivative |
|:--|--:|--:|
| the Fourier cell, centrally differenced | `1e-4` | `5e-5` … `9e-4` |
| the surrogate, over the whole box | median `5.0e-4`, p90 `4.2e-3` | median `7.5e-3`, p90 `6.1e-2` |

At 700 solves those last two were `2.0e-2` and `2.3e-1`, so the derivative
improved by very nearly the same factor as the value — which is the arithmetic
below working in the useful direction.

A fit carries a smooth bias of amplitude `ε` varying over a scale `L`, and the
derivative inherits `ε/L`, which is why it trails the value by more than an order
of magnitude. Read the derivative as a **distribution**, not as one figure: under a percent
typically, and still near 55 % at worst, at the faces of the box and the
strongest contrast pairs.

Its normalization is a decision. Of the 175 grid points, 70 have layers of equal
modulus — the inclusion is then homogeneous, `𝔸_εε` does not depend on `w`, and
the exact derivative is identically **zero**. A ratio to it is meaningless, the
same trap `validate_surrogate` guards against for sign-crossing components, so
the scale is the rms of the exact derivative over the grid, `0.68`.

Two conclusions, both counter-intuitive enough to be worth recording so the
experiments are not repeated. **More capacity would not help**: that same sweep
shows no ripple, so the network is not over-fitting and is not capacity-limited;
what sets the derivative is the bias amplitude, hence the sample count and how
much of the answer the network must learn at all. And **the cell was innocent**:
the mesher caps element size per layer and that thickness moves with `w`, so a
staircase was a real suspicion — checked on 25 values, the cell count grows
smoothly from 10 066 to 12 336 and the differenced cell matches the exact
derivative to `1e-4`.

### The anchored baseline for this morphology, and it does not pay either

`AnchoredHill` gained a second baseline, `:layered_spheroid`: the **homogeneous**
spheroid at the layers' mean modulus, `𝔸_b = (𝕀 + ℙ:(ℂ̄ − ℂ₀))⁻¹` with
`ℂ̄ = r̄ ℂ₀`. It is exact on the whole face `r₁ = r₂` of the sample box — a
layered spheroid whose layers agree *is* a homogeneous one — which is a
three-dimensional face where `:spheroid_cavity` is exact at a single point. For
a stress class the baseline is `ℂ̄ : 𝔸_b`, not `𝔸_b`: `𝔸_σε` is of degree one in
the moduli where `𝔸_εε` is of degree zero.

0.14.0 recorded that the anchored specification did not pay on the
superspheroidal cavity, and hoped a layered spheroid's closer baseline would
change that. **Measured, it does not.** Both were trained on the same 700
labels — the checkpointing makes a change of specification a re-encoding rather
than a re-solve — and judged on the decoded tensor over a 175-point grid
covering all four features, against the closed form:

| strain side | value median | value p90 | derivative median | derivative p90 |
|:--|--:|--:|--:|--:|
| dimensionless | `1.47e-3` | `1.00e-2` | `2.04e-2` | `2.30e-1` |
| anchored | `1.35e-3` | `3.53e-2` | `2.17e-2` | `2.40e-1` |

Marginally better at the median, **3.5× worse at p90**, and worse on the
derivative at every quantile. The reason is the one 0.14.0 identified and is
structural rather than a matter of tuning: near the exact face `𝕄 ≈ 𝕀`, whose
off-diagonal Walpole components are zero, so the anchored target carries zeros
by construction exactly where the anchor is perfect, and the tail pays for it.

Worth recording separately: the anchored pair's held-out **block** error was
*better* on both sides — `2.23e-3` against `2.44e-3` on the strain side, and
`6.60e-4` against `1.89e-3` on the stress side. A block error over 100 Halton
points is not accuracy over a dense grid, and the tail is where they part.

The baseline ships anyway, tested and exact on its face, for the same reason the
machinery shipped in 0.14.0: so the measurement is reproducible and the
experiment is not repeated. The shipped models stay `DimensionlessHill`.

### Added

- `layered_spheroid_strain` and `layered_spheroid_stress`, trained on the cell:
  both sides of gate B, one solve per sample.
- The `:layered_spheroid` anchor baseline, and `component_labels` for an
  `AnchoredHill` — without which a shipped output specification could not be
  reported on at all.
- `scripts/94_fe_neural_layered_spheroid.jl`, the whole computation end to end —
  the two ways of building a nest, the mesh report, the cell against both closed
  forms, the near-sphere failure the comparison found, the surrogate, the
  schemes, and the derivative the cell refuses — and
  `scripts/nn/make_layered_spheroid_figures.jl` for its figures.
- The surrogate section of the layered-spheroid tutorial, with the closed form,
  the cell and the surrogate on one figure, on values and on the derivative.

### Changed

- `NeuralLocalizationInclusion` keeps the semi-axes in the order given instead of
  sorting them descending, for the reason above. `NeuralHillInclusion` still
  sorts, and the asymmetry is now asserted by a test rather than implied.
- `scripts/nn/train_layered_spheroid.jl` **checkpoints every label as soon as it
  exists** and recomputes only what is missing on a restart — a Halton point
  depends on its index alone, never on `n`, so the index is a stable name for a
  sample. `MFH_NN_MAX_NEW` bounds the new solves one process performs per data
  set, so an hour of finite elements becomes a sequence of short runs. An
  interrupted run used to lose every sample it had already paid for, which is how
  this was measured. The held-out offset is a fixed constant instead of `NTRAIN`,
  so enlarging the training set no longer invalidates the held-out set.
- **The documentation is reorganized.** A page whose subject is one particular
  paper is an *application*, not a tutorial, so `cluster_model` (Molinari & El
  Mouden 1996, Figs. 3, 5, 6 and 16) and `eim_assembly` (the equivalent inclusion
  method against a published table) moved out of Tutorials; `docs/literate.jl`
  now routes a generated page to either output directory instead of assuming one.
  Applications are grouped into four named sections — cementitious materials;
  aggregates, pores and layered media; time-dependent behavior; interacting
  particle assemblies — in place of one flat list of thirteen.
- The `neural_inclusions` manual table lists **all** the shipped models: it was
  missing `excentered_sphere_strain` and `excentered_sphere_stress`, while its
  prose spoke of "the last four rows" trained on a cell.

### Fixed

- The frame convention above.
- The note shipped inside `excentered_sphere_strain.json` carried a UK spelling
  of *center* while its generator already carried the US one: the artifact had
  been written before that fix.

## v0.14.0 — the layered spheroid, meshed; and an anchored surrogate

Two pieces, and the honest result of one of them is that it does not pay yet.

### `N` nested spheroids by Fourier axisymmetric finite elements

`FEAxiLayeredSpheroid` solves `N` coaxial, concentric spheroids in an isotropic
matrix on the meridian half-plane. It is the **solid** declination of the
corrected finite Eshelby cell, so unlike the cavity both outputs of the fixed
point are used: `𝔹 ≠ 0`, and that second output *is* gate B's stress side. The
cavity was the easy case.

**Layers are given by their semi-axes, freely.** The same
`(axis_radii, disk_radii)` pair `LayeredSpheroid` takes, in the same order, so
one description builds the finite-element object and the analytic one:

```julia
ar, dr = confocal_layer_radii(2.0, 1.0, (0.3, 0.7))
fe  = FEAxiLayeredSpheroid(ar, dr, K)
ana = LayeredSpheroid(ar, dr, K)          # the same body, in closed form
```

Nothing in the type knows what *confocal* means. Handed
`confocal_layer_radii` it is the body the closed form solves; handed anything
else — an oblate core inside a prolate shell, say — it is a nest no closed form
covers, which is the case it exists for. `confocal_layer_radii` was extracted
out of `layered_spheroid_from_fractions` for exactly this reason: a cross-check
is worth nothing if the two solvers are handed two transcriptions of a geometry
instead of one geometry.

### Two exact families, and they are orthogonal

The space of nested spheroids is two parameters per layer, and it is crossed by
**two** independent closed forms that meet only at the equal-radii sphere:

| slice | closed form | conduction | elasticity |
|:--|:--|--:|--:|
| confocal, prolate | `LayeredSpheroid` | `3e-4` | `3e-5` |
| confocal, oblate | `LayeredSpheroid` | `5e-4` | `2e-4` |
| concentric spheres, **free** radii | `LayeredSphere` | `1e-4` | `5e-4` |
| equal moduli | the homogeneous spheroid's Hill tensor | `3e-5` | — |

Both branches of the confocal family are exercised deliberately: the oblate one
runs the analytic computation in complex arithmetic, through `c → -ic̄`,
`q → iτ`, and relies on an exact cancellation of the imaginary part. A
prolate-only comparison validates the easy half. And the sphere slice covers an
arbitrary layer count and arbitrary radii without assuming anything confocal —
which is what makes the free-radii route trustworthy at all.

The stress side is checked too, and it is not derivable: the inclusion is
heterogeneous, so `𝔸_σε ≠ ℂ₁ : 𝔸_εε` for any single `ℂ₁`. Measured on the same
solve, it agrees with `LayeredSphere` to `4e-5`.

### What refuses, and why each refusal exists

- **Nesting.** Two coaxial concentric ellipses are nested if and only if both
  semi-axes grow outwards, so `check_nested_spheroids` is two comparisons per
  layer. A violation is not a wrong answer but a self-intersecting geometry,
  which gmsh rejects from deep inside its own pipeline naming neither the layer
  nor the semi-axis — measured, while writing the mesher.
- **Imperfect interfaces**, by name. The axisymmetric formulation has no
  displacement- or temperature-jump term at all; the only jump in the package is
  the three-dimensional crack's. Accepting a `KapitzaInterface` and ignoring it
  would be the silent kind of wrong.
- **Mixed tensor orders** across layers, and a bound of the other physics: one
  object per physics.

### The element size is bounded by the thinnest layer, not by `nradial`

With free semi-axes two boundaries can come arbitrarily close, and an element
spanning a whole layer leaves it unresolved while the mesh looks perfectly
reasonable. So each boundary's size is capped at a third of its distance to its
neighbors. That is the same failure the concave superspheroid's wedge showed in
0.13.0, where an unresolved gap left `R₃₃` 1.2 % wrong and converging in
appearance only.

### Two bugs the comparison found, both of them older than this release

Solving one body two ways is worth having, and here is what it bought.

**The generic stress side was answering for inclusions it cannot serve.**
`stress_strain_loc`'s generic evaluates `ℂ₁ : 𝔸_εε`, which needs a single
uniform stiffness, and its docstring has warned since the contract was written
that a type declaring `is_homogeneous_inclusion == false` **must** supply its
own. Nothing enforced it, and `LayeredSpheroid` — which declares itself
heterogeneous and implements no elastic stress side — was quietly getting a
value wrong by **110 %**, uniformly at every aspect ratio. It surfaced the
moment a finite-element cell that measures *both* sides was pointed at the same
body: only the stress side disagreed, and by a constant factor, which is the
signature of a wrong quantity rather than a coarse mesh. The generic now refuses
instead, in both physics. Nothing in the suite exercised that path, so nothing
was relying on the wrong answer.

**The analytic elastic confocal spheroid was losing precision as it approached
the sphere, and now does not.** A one-layer confocal spheroid *is* a homogeneous
spheroid, so the closed-form Eshelby result is its answer. Measured against it:

| `\|1 − ω\|` | before | after | transport, same chart |
|---:|---:|---:|---:|
| 0.3 | `3e-15` | `2e-15` | `2e-16` |
| 0.1 | `1.6e-3` | `3e-15` | `2e-15` |
| 0.05 | `5.4e-2` | `4e-14` | `3e-15` |
| 0.01 | `7.7e-1` | `6e-13` | `1e-14` |
| 0.001 | `1.0e+0` | `4e-9` | `4e-12` |

Not the geometry — transport on the identical chart was always exact — not the
layer coupling, since one layer already showed it, and not the oblate
substitution, since both families failed. The columns of the elastic system are
amplitudes of Papkovich–Neuber potentials at the interface, so a growing mode of
degree `n` scales like `qⁿ` and a decaying one like `q^{-n-1}`; as the focal
distance goes to zero the magnitudes span `q^{2n+1}`, about `10²⁵` at
`ω = 0.999` with degrees to nine.

The degeneracy is in the basis's **normalization**, not the problem: as `q → ∞`
the spheroidal harmonics tend to spherical ones, which are perfectly
independent. So the solve equilibrates its columns before factorizing — a change
of unknowns, exact in exact arithmetic, and the scaling that minimizes the
2-norm condition number over all diagonal choices to within `√n`. **Columns
only**: equilibrating rows would reweight an overdetermined least squares and
change which solution is returned. Skipped for symbolic element types, where the
arithmetic is exact and a symbolic column norm would blow up every expression
downstream.

The usable range moves from `|1 − ω| ≥ 0.3` to `10⁻³`. A residual limit remains
at `10⁻⁴`, where the spread reaches `q¹⁹ ≈ 10³⁵` and even transport is at
`8e-10` — past what a diagonal scaling can repair, and stated rather than
hidden. Nothing caught the defect before because it changes no strain-side
result away from the sphere, is invisible in transport, and no test covered the
elastic near-sphere limit; there is now a regression test on the one-layer
oracle at every aspect ratio, in both physics.

### `AnchoredHill`, and the measurement that says not yet

A new output specification: the network predicts `𝕄 = 𝔸_b⁻¹ : 𝔸` against a
closed-form baseline and the decoder returns `𝔸_b : 𝕄`, so the answer is exact
wherever the baseline is. The same idea as `AffineHill`, which is twelve times
more accurate than the generic form on a network of the same size because it
does not spend capacity on a dependence already known — applied to a shape
family rather than to the material dependence.

The algebra stays in the **Walpole basis**: `inv` and the double contraction on
transversely isotropic tensors, a closed-form 2×2 inverse plus two scalars. A
component-wise ratio in Kelvin-Mandel mixes the transverse block and
manufactures sign changes that are an artifact of the reading — measured, and
the reason this is a specification and not a division.

**And on the superspheroidal cavity it does not pay.** Judged on the decoded
tensor: rms `1.30e-2` against `5.31e-3`, median `9.16e-3` against `2.64e-3`,
for 14 % gained on the extreme maximum. The reason is structural — near the
exact face `𝕄 ≈ 𝕀`, whose off-diagonal Walpole components are zero, so the
anchored target has zeros by construction exactly where the anchor is perfect.
What decides is the baseline's **proximity**, not its existence: `p = 0.25` is
far from `p = 1`. It is recorded here so the experiment is not repeated, and the
machinery ships because a layered spheroid's homogeneous limit is reached as soon
as the layers' moduli agree, which is a far closer baseline.

### Added

- `FEAxiLayeredSpheroid`, `LayeredSpheroidShape`, `layer_volumes`,
  `layer_fractions`, `check_nested_spheroids`, `axi_layer_set`, and a
  `layer_count` method.
- `confocal_layer_radii`, extracted from `layered_spheroid_from_fractions` so
  the geometry has one source.
- `AnchoredHill`, `anchor_baselines`, `anchor_tensor`, `encode`,
  `spec_baseline`; surrogate file format 1.2.0 carries the baseline's name, and
  a file written without one is refused rather than guessed.
- A tutorial, `A layered spheroid, meshed`, with the meridian meshes and the
  comparison against both closed forms.

### Changed

- `_axi_solve_mode` takes the region list as an argument. It was
  `(AXI_SET_CORE, AXI_SET_SHELL)` in the body — right for one morphology and
  silently wrong for any other, since the average would run over regions the
  grid does not have.
- `fe_axi_grid_counts` builds its region list from the grid instead of naming
  three fixed sets.
- The dataset cache is keyed on the output specification too: an anchored label
  is a different label for the same solve.
- `check_docs_blocks.jl` checks both directions — every documented export must
  appear in a `@docs` block, which is what `checkdocs = :exports` demands and
  what failed two doc builds.

### Fixed

- The generic `stress_strain_loc` and `flux_gradient_loc` refuse an inclusion
  whose `is_homogeneous_inclusion` is `false`, instead of returning
  `ℂ₁ : 𝔸_εε` — wrong by order one for such an inclusion, and the docstring had
  said so all along. Affects `LayeredSpheroid` in elasticity, the one type that
  declared itself heterogeneous without supplying its own method.
- The elastic confocal spheroid's solve equilibrates its columns, which restores
  the closed-form limit near the sphere: from a total loss at `|1 − ω| = 10⁻³`
  to `4e-9`, and machine precision down to `10⁻²`.


## v0.13.0 — the concave pore, axisymmetric

The 0.12.0 release treated the **cubic** member of the supershape family
completely: a three-dimensional cell in an octant, trained models, a comparison
against Chen *et al.* (2015). This one treats the **axisymmetric** member, and
it is the paper with the tables:

> Sevostianov, Chen, Giraud & Grgic, *Compliance and resistivity contribution
> tensors of axisymmetric concave pores*, International Journal of Engineering
> Science **101** (2016) 14-28.

Its shape is `Superspheroid(a, aγ, p)` without conversion, and being a solid of
revolution it asks for a different tool: the fields separate into Fourier modes
in the azimuth, so each mode is a **two-dimensional** problem on the meridian
half-plane. Where the octant divides the three-dimensional cost by eight, this
divides it by orders of magnitude — three assemblies and eight solves on a
triangle mesh.

### The mode count is the answer, not a coincidence

Mode 0 gives a `2×2` block, modes 1 and 2 a scalar each. Three plus one plus one
is exactly the five constants of a transversely isotropic compliance
contribution, with modes 0 and 1 giving the two of the resistivity. Nothing had
to be written for the decoding: `_AXI_Q` and `_axi_assemble_66` already did it,
and the same representation theory sits on both sides of the equality.

### Two numbers that are statements rather than measurements

**Transverse isotropy holds to `1e-16`** — `A₁₂₁₂ = (A₁₁₁₁ − A₁₁₂₂)/2`, and the
normal-to-shear block exactly zero — at *any* refinement, because the modes
decode straight onto the Kelvin basis. It is therefore a free check that the
three modes, the azimuthal projections, the boundary integral and the reassembly
are simultaneously right, which no one of them could establish alone.

**Conduction on a sphere is exact to `5e-6` and does not improve with
refinement**, because the exterior perturbation of a spherical cavity *is* a pure
dipole: the corrected condition has no higher multipole left to truncate.

Validated against the exact spheroidal cavity, which the package already knows
in closed form — in **canonical** components throughout, `Spheroid` sorting its
semi-axes and permuting its frame:

| `c/a` | conduction | elasticity |
|:--|--:|--:|
| 0.5 oblate | 4 digits | `1.8e-4` |
| 1.0 sphere | 5 digits | `2.3e-5` |
| 2.0 prolate | 5 digits | `1.5e-5` |

What remains in elasticity is truncation and the radius sweep says so:
uncorrected falls as `R^-2.85`, the `(a/R)³` signature, corrected as `R^-5` —
a factor 10 better at `R/a = 2.5` and 113 at `R/a = 8`. That sweep is also the
only test that proves the dipole sign, a wrong one leaving twice the bias.

### The route that looked cleverer, and the reason it is not

A cavity has no interior, so `⟨ε⟩_D` cannot come from `fe_axi_average`. It comes
from the wall — a line integral on the meridian trace, `fe_axi_pore_boundary`,
the tenth method of the axisymmetric contract and the only one a solid inclusion
does not need.

That was avoided first. The divergence identity on the matrix gives `⟨ε⟩_D` from
the volume average plus an outer term that is analytic, reusing only tested code
and adding no generic. It is algebraically exact and numerically hopeless: it
obtains `V_D` as the difference of `V_Ω` and `V_M`, whose ratio is `(R/a)³`.
Measured before it was abandoned — the implied cavity volume 2.9 % wrong at
`V_Ω/V_D = 8`, 5.8 % at 64, 9.2 % at 216, with the localization error tracking
it, and convergence of `O(h)` where the core-shell cell gets `O(h²)`. The error
**grew with the cell radius**, which is the opposite of what a larger cell is
for. It is recorded in the driver's header because the trap is attractive.

One consequence of that detour survives it: **normalize by the volume the meshed
wall encloses**, not by the closed form. Integrating over one boundary and
dividing by another's volume leaves a systematic error of the geometry's own
size that refinement never removes, because refining shrinks both together.
Fixing it took the spherical gate in conduction from `9.1e-4` to `5.0e-6`.

### The mesh, and the two corners

The meridian profile has **two** corners for a concave shape, and the source
already said which: "the equatorial plane is the crease; the two poles are the
conical points". So it is cut into two splines meeting at the equator, with the
poles as curve endpoints — a corner interior to a spline is a corner rounded off
— and the sampling clusters at both ends, a uniform step in the parameter
spending its points where the profile is flat.

The parametrization `ρ = a t^{1/m}`, `z = c (1−t)^{1/m}` satisfies the level set
identically, so no root find and no quadrature enter the geometry, and the tests
check that `level_set` returns zero on every sampled point rather than assuming
it.

### A graded mesh at the crease, worth 1.2 % on the answer

Uniform elements along the meridian were not enough, and the symptom was
specific: `R₃₃` — mode 0, the field pushed along the axis — stopped converging
while `R₁₁` from mode 1 converged cleanly. A concave superspheroid closes at the
equator as a **wedge**: at `p = 0.3, c/a = 0.6` the half-gap is `0.0012a` where a
uniform element is `0.031a`, so the elements nearest the crease straddled a gap
twenty-five times thinner than themselves. Grading the element size linearly in
the distance to the two corners — gated on `is_concave`, a convex profile needing
none of it — took the refinement increment from `4.4e-3` / `3.7e-3` to `3.8e-4` /
`1.9e-4`, and the ungraded value was **1.2 % wrong**, not merely unconverged.

The hypothesis that a concave enough `p` has no admissible mesh at all, and would
need extrapolation, was **measured and refuted**: the minimum mesh angle stays
between 33.8° and 39.5° all the way down to `p = 0.20`, with refinement
increments of `1e-4` to `1e-3`. There is no floor, no degenerate element and no
Richardson step. What the measurement did find is the subject of the next
section.

### Where the surrogate error actually came from

The first axisymmetric models were mediocre — `2.3e-2` worst case, where the
analytic-teacher models reach `1e-4` — and none of the three obvious levers
helped. More samples bought a factor 1.7 and then stalled; the graded mesh above
bought nothing on the fit, because it corrects the teacher's *value* rather than
its *shape*; narrowing the sample box made the worst case **worse** while the rms
improved, which is what a few extreme samples carrying the maximum looks like.

The cause was in neither the mesh nor the network. `R₃₃` runs from `1.66` at
`p = 0.6` to `15.0` at `p = 0.20`, near-diverging as the body tends to a crack
pierced by a needle, and two defaults were calibrated for quantities that do not
do that:

- `fit_scaling`'s `log_threshold` of 30 declined to fit the output in `log`,
  because a factor of nine is not decades. With identity scaling the loss is
  dominated by the large values and the relative error elsewhere is whatever is
  left over. Hence `TrainingOptions(; log_threshold)`, set to 5 for these two
  models: `2.3e-2` → `7.1e-3`.
- A `SampleBox` is **linear**, so the feature *is* the sampling law. Uniform in
  `p`, most of the budget lands where the response is flat. Hence the `:log_p`
  feature beside `:p` (and `:log_aspect` read off a pore's own parameters):
  `7.1e-3` → `3.7e-3`, with `R₁₁` at `2.0e-4`.

A factor of six on the transport model, none of it from a bigger network. Both
knobs are now named and documented rather than implicit in a default, which is
the point: the defaults were not wrong, they were calibrated on a different kind
of quantity.

The elastic model was a different story, and the useful part of it is that three
measurements each ruled out a lever rather than fixing one.

Its box is three-dimensional and `ℓ₁` runs from 1.4 to 63 across it. The teacher
was excluded first: its own mesh convergence is `1.5e-3` between `nradial` 8 and
20, forty times below the fit. Raising the sample count from 780 to 1800 then
improved **every** rms by a factor 1.2 to 2.3 while the reported maximum *rose* —
because a held-out set 2.2 times larger reaches further into the tail. Widening
the network to `[96, 96, 96]` did the opposite: better maximum, worse rms
everywhere. Neither is the limit.

The limit is a **near-singular corner of the sample box**. The ten worst held-out
points all have `p ≤ 0.41`, `c/a ≤ 0.9` and `ν₀ ≥ 0.39`: a flattened, strongly
concave cavity in a nearly incompressible matrix, where the localization reaches
18 to 34 times the identity against 2 for a sphere. That is the near-crack limit,
and no amount of sampling repairs a divergence. It is documented with its
coordinates rather than removed by narrowing the box.

**So the metric was the thing to fix.** A maximum over a heavy-tailed
distribution is not a summary of it and is not comparable between runs of
different size — which is exactly how it sent this work after the wrong lever.
`validate_surrogate` now returns the whole distribution of block errors and
`report_surrogate` prints rms, median, p90 and p99 beside the maximum. For the
elastic model: max `6.3e-2`, but rms `6.4e-3`, median `3.0e-3`, p90 `6.6e-3` —
which is the accuracy over the domain where it will be used, and comparable to
the analytic-teacher models.

Two more results of that diagnosis. `ℓ₃` and `ℓ₄` pass within `2e-4` of **zero**
inside the box, so their per-component relative error is not a measure of the fit
near that crossing; `report_surrogate` flags such a row instead of printing a
number that reads as one. And a finite-element dataset is now **cached** between
runs, keyed on everything the labels depend on: the 1800-solve set took 7752
seconds to build and 0.1 second to reuse, which is what made three fit
experiments affordable at all.

### Against the paper's own tables, all thirty-five rows

Their Table B.1 (eighteen rows, five components) and Table B.4 (seventeen rows,
two) are transcribed in full and plotted against the cell, in `E₀ℍ` and `k₀𝑹` as
they report them. The material is nowhere in the paper and is **inferred** from
its own `p = 1` row, where the body is an exact sphere: `E₀ = 1`, `ν₀ = 1/3`,
`k₀ = 1`, residual 0.06 %.

**Four of the five compliance components and the axial resistivity agree across
the whole range.** `H₃₃₃₃` is the demanding one — it runs from 2.00 at the sphere
to 27.7 at `p = 0.20` — and agrees to 0.72 % at worst, 0.19 % at the most concave
row. That is two independent finite-element formulations, theirs
three-dimensional on a million nodes and this one two-dimensional on sixty
thousand, landing on the same five-constant tensor for a shape family with no
closed form.

Three things do not agree, and they are of three kinds.

`k₀R₁₁` disagrees in **trend**: theirs rises with concavity, ours falls, 37 % at
`p = 0.20`. Ours was checked before it was published — invariant to six figures
across `radius_ratio` from 4 to 10 and `nradial` from 20 to 28 — and then checked
*again* by a route with nothing in common, the three-dimensional octant cell on
the same shape, which lands 0.13 % away and converges towards it from below.
Their own control row is 0.08 % off at the sphere, where the answer is `3/2`
exactly.

Their transverse block **reverses direction** below `p = 0.30` (`H₁₁₁₁` 1.8878,
then 1.8080, then 1.8200) where ours is monotone; and their `H₁₃₁₃` at
`p = 0.65` is identical to six decimals to their `p = 0.60` row, on the very row
their own Table B.3 flags with its largest change in that column.

The same comparison carries a counter-check that runs the other way, and it is
in the docs for that reason: on `H₃₃₃₃` *our* octant at level 4 is 4 % below our
two-dimensional value and still climbing, while theirs is 0.16 % from it.
Three-dimensional meshes converge from below on that component; their mesh got
there and our level 4 did not.

### And the surrogate, because otherwise there is no sensitivity

`GradLocTI2` is the transport class an axisymmetric cavity needs: two components,
`R₁₁` and `R₃₃`, where the supersphere's is isotropic and needs one. Same
argument as `GradLocISO2` in 0.12.0 — `HillTI2` has the same two components and
a different dimension, so reusing it would divide every prediction by `k₀`.

Elasticity uses `StrainLocTI`, and that raised a real design question.
`_reference_medium` has **no** method for that class, deliberately, because it
also serves *heterogeneous* morphologies which carry their constituents inside
themselves — guessing there would train on corrupted labels. A cavity has no
constituent and is of degree 0 in the reference, so the class is perfectly
usable; what is missing is not a method but a *statement*. Hence a `reference`
keyword on `generate_dataset`: the caller declares what the class cannot decide,
which is the opposite of correcting silently.

### Added

- `FEAxiSupershapePore` and `AxiSupershapePoreShape`, with
  `fe_axi_pore_localization`, `fe_axi_pore_breakdown` and
  `fe_axi_pore_mesh_report`.
- `fe_axi_pore_boundary` — the tenth generic of the axisymmetric backend
  contract, with a Ferrite implementation.
- `GradLocTI2`, and a `reference` keyword on `generate_dataset`.
- `TrainingOptions(; log_threshold)`, forwarded to `fit_scaling`, and the
  `:log_p` feature (with `:log_aspect` extended to a meshed pore).
- `FEAxiMeshOptions(; nprofile, tip_refine)`, read by the axisymmetric pore
  alone.
- Two trained models, `axi_supershape_pore_conduction` (max `3.7e-3`) and
  `axi_supershape_pore_elastic` (rms `6.4e-3`, p90 `6.6e-3`, max `6.3e-2` at one
  near-crack point).
- `validate_surrogate` returns `block_errors`, `block_rms`, `block_median`,
  `block_p90` and `block_p99`; `report_surrogate` prints them and flags a
  component that changes sign over the held-out set.

### Changed

- `scripts/89_fe_supersphere_pore.jl` is now **`scripts/89_fe_concave_pores.jl`**
  and covers both families and both papers. One script and one application page
  for one subject: the 80-89 block is full, and a second page beside the first
  would have said the same things twice.
- `_axi_setup` and `_axi_mode_setup` are no longer typed on
  `FEExcenteredSphere`. They read `.cache.setup`, `.backend` and `.mesh.order`
  and nothing else, so every axisymmetric morphology is served by the same code;
  the rest of the driver stays typed per morphology, the constituents, the
  averaging sets and the fixed point genuinely differing.
- `fe_axi_grid_counts` reports the region sets the grid actually has. It asked
  for three unconditionally, and a cavity has one.


## v0.12.0 — superspheres, from the shape to a differentiable phase

A shape family with **no closed-form Eshelby solution** now reaches every
scheme, in both physics, either by solving one finite-element cell or by
evaluating a network trained on those solves. Along the way the package gained
the symmetry class that family lives in, a third finite-element cell family, and
two documentation pages the layered inclusions had been missing.

The family is `|x/a|^{2p} + |y/a|^{2p} + |z/a|^{2p} ≤ 1` and its axisymmetric
sibling: the sphere at `p = 1`, the octahedron or double cone at `p = 1/2`, a
**concave** body with conical points below that, the cube or cylinder as
`p → ∞`.

### The geometry, and why it is meshable at all

These bodies are star-shaped about their center for *every* `p > 0`, concave
ones included, and the level set is positively homogeneous of degree `2p`, so
the distance from the center to the surface in a given direction is a **closed
form** rather than a root find. That is what makes pushing a subdivided sphere
onto the surface a viable meshing strategy, and what makes the whole geometry
differentiate cleanly in the shape parameters.

The surface is discretized from a subdivided **octahedron**, for three reasons
at once: its face edges lie exactly in the coordinate planes, so an octant is
native; for a concave `p` those same planes are where the surface creases, so
mesh edges land on the creases rather than straddling them; and there is no
polar degeneracy. And the octahedron *is* the `p = 1/2` supersphere, which makes
that value a **bit-exact** oracle — volume to `2e-16`, every angle 60 degrees.

### One eighth of the cell, and the answer is the same

The cell around a supershape is now meshed as an **octant** when asked
(`FECellMeshOptions(; octant = true)`), and that is what turns the concave range
from unaffordable into routine: at `p = 0.3` level 4 costs about 70 000 degrees
of freedom and half a minute, where the whole cell needed some 360 000 and
brought a 16 GB machine down.

The licensing condition is **not** cubic symmetry. It is that the three
coordinate planes be mirror planes of the shape — the group
`{diag(±1,±1,±1)}` of order 8 — which a superspheroid satisfies without being
cubic at all. `has_coordinate_mirrors` records it as a trait defaulting to
`false`, so a new shape opts in deliberately; `check_coordinate_mirrors` audits
a candidate numerically, and is explicitly *not* the guard, because a sampling
that passes is a weak guarantee while a missing trait is a loud error.

Three things came for free and are worth naming, because they are why this was
cheap. The caps already existed — `octant_patch` was written for this, its face
edges lying exactly in the coordinate planes. Snapping preserves the planes
**exactly**: both passes scale a node radially, and `r * 0.0 == 0.0` in floating
point, mid-edge nodes included. And `fe_cell_curved_volume` already returns
`V/8`, integrating `⅓∮x·n` over the cap alone while the flat faces pass through
the origin.

What had to be built is the three flat faces. The obvious construction — a
structured grid between the two boundary traces — needs them to carry the same
number of nodes, which would force `level == outer_level` and throw the saving
away, `outer_level` being a floor set by resolving the dipole term. Meshing each
face as an **unstructured** planar region makes the question vanish instead of
answering it. The boundary is assembled in Julia as a single deduplicated node
list, so watertightness is a property of the data structure and is asserted by a
divergence identity **before** any mesher runs — measured at `1.4e-16`.

### The part that decides whether an octant is admissible at all

The corrected boundary condition adds a dipole field to the remote one. If the
two did not transform alike under a reflection, no single plane condition could
serve both. They do:

    R u(Rx; Π) = u(x; RΠR),    T(Rx; M) = T(x; RM),

which is exactly the law obeyed by `E·x` and `M·x`. The driver drives both
families with the same Kelvin tensor, so remote load and dipole share a parity
for every case — and the test that checks it finds the two sides **bit-equal**,
not equal to a tolerance.

Two consequences the implementation depends on. The six elastic load cases fall
into **four** parity classes and the three transport ones into three, so the
stiffness is assembled once and factorized four (resp. three) times on a matrix
eight times smaller. And an octant average is **not** one eighth of the whole:
reflecting the surface integral gives `8·P_χ(I_oct)`, so the components whose
parity differs from the load case cancel *between* octants rather than vanishing
in each. Multiplying by eight without projecting would keep them at full
amplitude. The upside is that the couplings symmetry forbids come out exactly
zero, where the full cell finds them at a few percent of the norm.

The trap to remember: `V` is the volume of the **whole** cavity in both modes,
because the dipole moment is that of the entire inclusion; only the surface
average sees the eighth. Dividing everywhere weakens the correction eightfold,
which reads as a mesh that will not converge.

### The cell, and the correction that makes it usable

The third cell family, after the flat crack and the axisymmetric core-shell
sphere: a shell of matrix between the shape and an outer sphere, meshed through
gmsh's *discrete* entities because no CAD kernel represents the surface. Eight
backend generics, `fe_cell_*`, with a Ferrite implementation; a curved
second-order boundary whose mid-edge nodes are snapped onto the exact shape.

Validated against the **exact spherical pore** in both physics, which a
supersphere at `p = 1` is: `1.15e-3` in conduction against `A = 3/2`, `1.48e-3`
in elasticity against `[I − S]⁻¹`, at level 3.

Four things the tests pin that matter more than the accuracy figure. The
**uncorrected** error does not converge — refining takes the corrected error
from `9.0e-3` to `1.2e-3` while the uncorrected one moves only from `4.0e-2` to
`3.3e-2`, it being a truncation bias and not a discretization error. The
corrected answer **no longer depends on where the cell is cut**: flat at about
`9e-3` across `R/a = 2.5…5` where the uncorrected one falls `6.3e-2 → 1.6e-2`.
The dipole diagnostic decays like `R^-2.96` against `-3` from theory. And the
**wrong sign roughly doubles** the bias instead of removing it, which is the
signature to recognize.

### A cavity is a *heterogeneous* inclusion, and that is not a trick

`FESupershapePore` declares `is_homogeneous_inclusion = false`, because that
flag asks whether a single `C₁` describes the interior and for a cavity
`inv(C₁)` is meaningless. The package's own exact identities then take over, and
with a cavity's stress-side localization identically zero they collapse to
`N = -C₀:A` and `H = A:S₀` — the right answer, with no `C₁` in it, and not one
contribution tensor overridden. Declaring a cavity *homogeneous* instead would
send every scheme down `(C₁ - C₀):A` with whatever soft-but-not-zero stiffness
the phase carries, for a percent-level error nothing reveals.

### And then it becomes cheap and differentiable

`StrainLocCubic` is the neural class for a cube-symmetric localization tensor:
**three** components against `StrainLocTI`'s six, because a tensor with the minor
symmetries and cubic symmetry is major-symmetric automatically and has no
antisymmetric content to drop. `FESupershapePore(shape; elastic = s)` hands a
trained network to the very type that would otherwise mesh and solve —
everything else unchanged, no backend loaded, no assembly performed.

**The pore then differentiates in its own morphology**, which no finite-element
inclusion can: their solve runs in `Float64` and memoizes on the reference alone,
so the derivative would be a silent zero and the package refuses the request.
Two new hooks make it work for a type that holds a shape *object* rather than
its scalars — `Schemes._geom_field` reads and `_replace_geom_field` rebuilds,
both through `pore_shape_params` so the name sets cannot drift.

**Two trained models ship**, one per physics, produced by
`scripts/nn/train_supershape.jl` — and they exist only because the octant makes
780 finite-element solves cost about three minutes (0.19 s and 0.26 s each). `GradLocISO2` is the transport
class they needed: a second-order tensor invariant under the octahedral group is
isotropic, so a cavity that takes three constants in elasticity takes **one**
here. `HillISO2` has the same single component and a different dimension, and
reusing it would have divided every prediction by `k₀`; the constructor now
refuses that, along with a wrong tensor order and a feature the shape cannot
supply, at construction rather than at the first solve.

Their accuracy is **teacher-limited, and quoted as such**: worst held-out error
`1.5e-3` in transport and `2.1e-2` in elasticity, against an rms an order of
magnitude smaller, because the cell's own departure from the symmetry class
reaches `1.5e-2` at `p = 0.35` and does not improve with refinement. Adding
samples makes the worst case *rise*, a denser sample reaching further into the
concave corner — so `generate_dataset` grew an `atol`, since its default of
`1e-8` belongs to an analytic teacher and rejects a perfectly good measured
label.

### Added

- `Supersphere`, `Superspheroid` and their closed-form geometry —
  `level_set`, `radial_distance`, `surface_point`, `outward_normal`,
  `shape_volume`, `projected_area`, `equivalent_sphere_radius`,
  `diagonal_radius`, `edge_radius`, `bounding_radius`, `inner_radius`,
  `shape_exponent`, `is_concave` / `is_convex` / `is_sphere`.
- `TriSurface` and the surface mesher: `octant_patch`, `unit_octahedron`,
  `project_to_shape!`, `shape_surface`, `relax_surface!`, `mesh_area`,
  `mesh_volume`, `edge_lengths`, `mesh_quality`.
- `FECellMeshOptions`, `fe_cell_size_estimate`, `fe_cell_curved_volume`,
  `fe_cell_meshed_volume`, `fe_available_gb`, and the eight `fe_cell_*` backend
  generics with their Ferrite implementation.
- `FESupershapePore`, `fe_cell_localization`, `fe_cell_mesh_report`,
  `has_surrogate`, `pore_shape_params`.
- `green_gradient_iso2` and `dipole_temperature_iso` — the transport
  counterpart of the Kelvin dipole, which the package had for elasticity only.
- `cubic_residual` and `CubicSym`; `StrainLocCubic` and `GradLocISO2`.
- `has_coordinate_mirrors` and `check_coordinate_mirrors` — the trait that
  licenses an octant, and its numerical audit.
- `boundary_chains` and `patch_corners` — the ordered boundary of an octant
  patch, found from the mesh rather than from the subdivision lattice, so it
  survives relaxation and projection.
- `FECellMeshOptions(; octant = true)`, and an `atol` on `generate_dataset`.
- Two trained models, `supershape_pore_conduction` and
  `supershape_pore_elastic`, with `scripts/nn/train_supershape.jl` that produces
  them and `scripts/89_fe_supersphere_pore.jl` that exercises the family against
  the literature.

### Fixed, and found by porting

The port doubled as an audit of the prototype it came from, and of this package.

- **A validation that could not run.** Without an inner constructor Julia
  generates `Supersphere(a::T, p::T)`, more specific than the checked
  `Supersphere(::Number, ::Number)`, so a negative semi-axis was accepted
  silently.
- **`0 × Inf = NaN` in the outward normal** wherever a coordinate vanishes and
  `2p < 1`: the fallback then returned the **zero vector**, on the axes and the
  coordinate planes — exactly where a subdivided octahedron puts its nodes. The
  non-smooth cases are named now rather than computed.
- **"The real fix is to refine" was false.** The snapping limitation *grows*
  with refinement — 6, 37, 73 nodes at levels 2, 3, 4 for `p = 0.35` — because
  the curvature at a conical point is unbounded. What is stable is the
  fraction. And what triggers it is unbounded curvature, **not** sharpness: the
  octahedron has edges and vertices everywhere and needs no snapping at all.
- **Relaxation is a trade, not an improvement**: it equalizes edge lengths and
  degrades the smallest angle. Both measured, both documented.
- **The meshed area converges from above for a concave shape** and from below
  for a convex one.
- `Core/counters.jl` referenced a `_maybe_count` that no longer exists — a
  dangling cross-reference that would have broken the documentation build the
  day that docstring was listed.

### Documentation

- A new application page, `applications/concave_pores.md`, reproducing what can
  be reproduced from [Chen, Sevostianov, Giraud & Grgic, IJES 97 (2015)] and
  stating plainly where this package disagrees: a supersphere is **cubic**, and
  their two-constant isotropic approximation — and their own data — carry no
  anisotropy at all. Their linear fit for `η` is also non-monotone inside its
  own range and negative below `p = 0.2`. The control that licenses saying so is
  the sphere, where the true anisotropy is zero and this chain's spurious one is
  below `5e-5`.
- `theory/corrected_cell.md` gained **the pore declination**, the section two
  places in the repository already pointed at without it existing, and the
  octant's parity argument with it. The declinations table now has three columns
  rather than two.

- `manual/fe_inclusions.md` and `manual/neural_inclusions.md` **restructured on
  one rule**: the principle, then the syntax common to every case, then the
  cases. The first was a page about the elliptical crack with the general
  principle reduced to a pointer; it now carries what a finite-element inclusion
  *is*, the three steps shared by all of them, the shared refusals, and two
  questions that come up for any new morphology — how to mesh a shape with no
  CAD representation, and why a cavity must declare itself heterogeneous.
- A new `manual/layered_inclusions.md`: the layered families had three theory
  pages and two API pages and **no usage page**, the only inclusion family for
  which the gallery sent a reader wanting to *use* one to a derivation.
- A new `manual/index.md`, the reading path the Manual lacked while Theory,
  Tutorials and FE coupling each had one.
- The gallery wrote the supersphere as `|x|^p + …`, called it "convex, close to
  a sphere", and drew it at a `p` that made it deeply concave. Corrected to the
  `2p` convention, with the factor of two between the two conventions stated.

### Dependencies

- **`TensND = "0.5"`**, for `TensCubic`. The cubic symmetry class — three
  constants and a cube frame — was developed upstream rather than here, so this
  package now carries only what a homogenization package adds on top: a name for
  the projection residual, and the symmetry trait, which is honest only now that
  a storage type exists behind it.


## v0.11.0 — the elastic confocal spheroid, from nothing to a mean-field phase

`LayeredSpheroid` solved conduction and nothing else. It now solves
**elasticity** as well, prolate and oblate, with any number of confocal layers,
and feeds the homogenization schemes: `spheroid_strain_concentration` assembles
the full transversely isotropic strain concentration tensor, and
`strain_strain_loc` / `stiffness_contribution` hand it to `homogenize`.

This is a derivation rather than a port, and
`theory/layered_spheroid_elasticity.md` carries it in full because it is not in
the literature. The pieces exist separately and none is the piece needed: Duan
et al. (2005) give the Papkovich–Neuber representation for a *uniform* spheroid
and refer to Love (1927) for the operators; Barthélémy & Bignonnet (IJES 2020)
build the confocal *multilayer* transfer matrices but for the scalar Laplace
equation. The displacement and stress operators in the spheroidal frame are in
neither.

### One evaluator, and the cases as data

The three elementary problems differ in which potentials are active, at which
order `m`, and what the remote field is — not in their operators. Written on
all four Papkovich–Neuber potentials, with `Φ = φ₀ + x⃗·φ⃗`,

    2μ u = ∇Φ - 4(1-ν) φ⃗,   σ = -2ν (div φ⃗) 𝟙 + ∇∇Φ - 4(1-ν) sym(∇φ⃗)

and both statements verified symbolically for four *arbitrary* harmonic
potentials. So there is one evaluator, and no operator had to be derived twice.

A mode is a product of three univariate functions, so all its derivatives are
products of tabulated pieces; only `Φ` needs Leibniz, and a second-order **jet**
carries value, gradient and Hessian through that product exactly, in any element
type, without nesting `ForwardDiff` inside a solve that may itself be
differentiated.

### Why one global system and not a chain of transfer matrices

Confocal surfaces are not homothetic, so the harmonic degrees couple — and
unlike conduction, where that happens only at an imperfect interface, here they
couple across a perfect one too. There is no `2𝒩 × 2𝒩` block to chain. What is
assembled instead is the conditions at every interface at once, projected on the
Legendre degrees by Gauss quadrature, which is *exact*: each condition is a
polynomial in `p` once the shared radicals are cleared. The azimuth needs no
quadrature at all — every field component of a case carries one azimuthal
harmonic, the same on both sides, so it divides out.

Whether the coupling is banded or triangular is decided by which multiplier acts
on `Pₙ(p)`. A bare `Pₙ′` reaches every lower degree; `(1-p²)Pₙ′` and `p Pₙ` reach
one, `p²Pₙ` two. The tangential conditions therefore carry the weight `1-p²`,
which is shared geometry across a confocal interface and the weight for which
the `Pₙ′` are orthogonal. That weighting is what makes truncation legitimate.

### Averaging without per-layer bookkeeping

The divergence theorem reduces the volume average to a surface integral on a
confocal boundary. With perfect interfaces the displacement is continuous, so
the interior boundaries cancel in pairs and only the outer surface survives —
whatever `N` is, with no volume integral anywhere. A single layer's average is
the difference of the two moments on its own boundaries, which is what
`stiffness_contribution` needs and what the total alone cannot give, the layers
having different moduli.

The two routes are independent, so `Σ f_k 𝔸_k = 𝔸` is a check rather than a
restatement. It holds to the truncation accuracy.

### Validated on all 81 components

| | discrepancy vs Eshelby | residual |
|:--|:--|:--|
| prolate, `ω = 1.5, 2, 4, 10` | `8e-15` | `≤ 2e-14` |
| oblate, `ω = 0.7, 0.4, 0.2` | `6e-15` | `≤ 2e-15` |

An oblate spheroid goes through the complex substitution `q = iτ`, `c = -i c̄`,
exactly as on the conduction side; the imaginary part of the answer comes out at
`8e-17` relative, and the code refuses rather than truncates if it does not.

Then, on the assembly rather than on one solve: a shell carrying the core's own
moduli changes nothing (`7e-16`); the two members of each shear pair agree,
which nothing enforces; transverse isotropy holds to the truncation accuracy;
and through `homogenize`, a single-layer spheroid degenerates onto the
equivalent `Ellipsoid` to `2e-15` under both Dilute and Mori–Tanaka.

### Three findings, two of them corrections

- **No rigid-body rotations are needed.** Duan adds two to case III and
  attributes to their omission the error in Riccardi & Montheillet (1999). With
  the *full* four-potential set the rotation is already spanned: it is the
  antisymmetric combination of the two potentials whose symmetric combination is
  the remote shear. Duan's extra unknowns are an artifact of a gauge that drops
  one of them. Pinned as a test — that combination carries no traction — and
  confirmed by the oracle.
- **Degree 0 of `φ₀` is not optional.** Only its *regular* part is an inert
  constant; the irregular one is `arccoth q`, essential. Dropping it makes the
  case-I system genuinely inconsistent — residual `1e-1` instead of `1e-16`,
  and 10% error — while cases II and III stay exact.
- **Parity is part of the answer.** Admitting both parities splices in the
  *other* problem of the same order, the one whose remote field is odd where
  this one's is even, and the truncated system leaks between them.

### Added

- `spheroid_strain_concentration`, `spheroid_layer_strain_concentration`,
  `spheroid_core_strain`, `spheroid_elastic_coefficients`, `AxisymmetricCase`,
  `TransverseShearCase`, `LongitudinalShearCase`, and `strain_strain_loc` /
  `stiffness_contribution` methods for `LayeredSpheroid`.
- `PNMode`, `ModeGroup`, `mode_fields`, `bb_letter` — the generic evaluator.
- `legendre_table`, `legendre_degrees`: the degree set is now part of the
  problem, elasticity splitting the potentials by parity where conduction needed
  only the odd degrees.
- Legendre orders `m = 2` (`:P2`, `:P2p`, `:Q2`). Two structural points the
  lower orders did not expose: the three-term recurrence is **singular at
  `n = m-1`**, so the seed must reach `n = m`; and the `Q₂²` seed cancels at
  large `q`, documented rather than worked around — it is a uniform
  normalization error, not an instability, and large `q` means a nearly
  spherical spheroid.

### Notation

The module follows Barthélémy & Bignonnet (2020) throughout: **regular**
harmonics `Pₙᵐ(p)Pₙᵐ(q)` against **irregular** ones `Pₙᵐ(p)Qₙᵐ(q)`, and the
amplitude letters `a, b, c, d` meaning regular-cos, irregular-cos, regular-sin,
irregular-sin. That exposed a clash with this module's earlier `:A :B :C :D`
labels, where `C` and `D` meant a different *potential* rather than the sine
families; `PNMode` carries `(regularity, trig)` explicitly now. Conduction has
one field, so elasticity adds a potential index: `a^{i,m}_{ℓ,n}`.

### Fixed

- **`Jet2` could not multiply across element types**, so a `ForwardDiff.Dual`
  modulus on a `Float64` geometry was a `MethodError`. Promotion added, and the
  cause fixed at source.
- **The promotion omitted the layer moduli** — the very defect v0.10.1 fixed on
  the conduction side, reproduced here and caught by the tests.
- **An exactly zero column** (the regular degree-0 constant) is now filtered
  rather than tolerated: `Float64` QR survived it, `Dual` raised
  `SingularException`.

### Documentation

- New theory page, with every derivation executed at build time and printing
  zeros: the four-potential representation and its gauge, the closed-form
  displacement operators, the coordinate-free stress, the Hessian in the
  spheroidal frame, the tractions, the four interface conditions, the banding
  argument with its counter-examples, and the solver's validation tables. An
  appendix recalls the two formulas from BB2020 the argument leans on — the
  Legendre orthogonality relations, and appendix C's precision rule, which turns
  out to *predict* the measured `Float64` ceiling.
- `theory/layered_sphere.md`: the two-column figure table rendered broken, and
  the three-phase model is a *scheme* built on the composite-sphere solution
  rather than a statement of it. Replaced by one vector figure of the pattern
  itself. Also fixes a combining circumflex inside math, which MathJax cannot
  compose, and two unicode superscripts.
- Corrected a claim this package made in three places (four, counting the
  roadmap): that the spheroidal harmonic decomposition "is specific to the
  scalar Laplace equation and does not carry over to the vector elastic
  problem". Barthélémy & Bignonnet say the opposite — the formalism was
  imported *from* elasticity.

### Still out of reach, and why

**Imperfect interfaces do not fit this formulation.** A uniform spring or
membrane law puts an *odd* power of the metric factor `w = √(q²-p²)` into the
matching condition, so it stops being a polynomial identity in `p` and takes
both the exactness of the projection and the banding with it. A perfect
interface escapes this only because every geometric factor there is shared
across a confocal surface and cancels; a compliance is a new length scale that
does not. The workable route is a thin confocal interphase — an extra layer,
already handled — but not an equivalent one: such a shell is exactly `ω` times
thicker at the equator than at the pole, so it models a compliance that varies
along the interface.

Also outstanding, and merely work: pointwise fields, an arbitrary axis, and a
compliance-side contribution tensor.

## v0.10.1 — sensitivities with respect to interface parameters

A composite inclusion is usually designed at its interface: the coating
stiffness, the interfacial thermal resistance, the surface conductance are the
knobs an optimizer wants to turn. None of them was differentiable. Asking
`ForwardDiff` for the sensitivity of an effective property to a spring
stiffness, a membrane surface modulus, a Kapitza resistance or a surface
conductance raised `MethodError: no method matching Float64(::Dual)` — in
`LayeredSphere` and `LayeredSpheroid` alike, in elasticity and in conduction.

The cause was a single omission repeated across the package. Every solver sizes
its state buffers from a `promote_type` over the geometry, the layer moduli and
the matrix moduli; the interface parameters were left out of that promotion.
The transfer matrices widen correctly on their own — they already promote
`eltype(intf)` locally — so the widened result then met a buffer too narrow to
hold it. Nothing in the suite asked for such a derivative, so nothing noticed.

### Fixed

- **Interface parameters are differentiable**, for `SpringInterface` (`kn`,
  `kt`, and the stored compliances `sn`, `st`), `MembraneInterface` (`κs`,
  `μs`), `KapitzaInterface` and `SurfaceConductiveInterface`, on both
  `LayeredSphere` and `LayeredSpheroid`. A new `interfaces_eltype` helper
  carries the interfaces into the outer promotion at every site.
- **`MembraneInterface(κs, μs)` promotes its two arguments.** The default inner
  constructor bound one type across both, so differentiating with respect to
  the surface dilatation alone failed at construction.
- **`LayeredSpheroid` state buffers include the layer conductivities.** They
  were promoted from `q` and the matrix conductivity only, so a single `Dual`
  layer among `Float64` ones threw. `LayeredSphere` had had this fixed; the
  spheroid had not.
- **`_J_int` types its coupling blocks from `coef`**, the factor that carries
  the interface parameter, instead of from the diagonal block alone.
- **Mixed-eltype semi-axes.** `LayeredSpheroid` and
  `layered_spheroid_from_fractions` bound a single element type across every
  semi-axis and every volume fraction, making one `Dual` boundary among
  `Float64` ones a `MethodError`. Both now promote, as `LayeredSphere` already
  did.
- **`layered_spheroid_from_fractions` no longer loses the derivative.**
  The confocal parameter of an inner layer is located by bisection, and a
  bisection decides by sign tests, which carry no derivative: the result came
  back with zero partials — silently, since the value was right. Two Newton
  steps in the full element type restore `∂x/∂D = -1/(3x²+C)`, exactly the
  implicit function theorem's answer for `x³ + Cx + D = 0`.

### Added

- **Symbolic `LayeredSpheroid`.** The element type was bound to `Real`, which
  excludes `SymPy.Sym`. It is now `Number`, the geometric validations are gated
  on `is_hard_numeric`, and `_q_recurrence_plan` returns the plain upward
  Legendre recurrence on an exact type — Miller's downward variant exists only
  to control floating-point cancellation, and an exact type has none. A
  symbolic solve now yields a closed form in the semi-axes and conductivities,
  and substituting numbers into it reproduces the `Float64` answer to `1e-16`.

  Because the sign of `axis² − disk²` is undecidable on a symbol, and SymPy
  answers `false` to a comparison it cannot settle, a symbolic spheroid
  requires an explicit `prolate = true | false`.

  Prolate only, and the limit is structural rather than an omission: an oblate
  spheroid carries `q = iτ`, so its confocal parameter is a `Complex{T}`, and
  Julia's `Complex{T}` requires `T <: Real` — which `SymPy.Sym` is not.
  `Symbolics.Num` is `<: Real` and gets further, but Symbolics has no `sym_lu`
  for a `Matrix{Complex{Num}}`. Both facts are pinned by tests.
- Tests: `test/LayeredSpheres/test_interface_sensitivity.jl` and
  `test/LayeredSpheroids/test_ad.jl`. Every derivative is checked against a
  central difference, and the interface cases additionally assert the
  derivative is non-zero — a silent zero being the failure mode the bisection
  fix guards against. `LayeredSpheroid` had no type-genericity test at all.

### Documentation

- Corrected a claim this package made in three places: that the spheroidal
  harmonic decomposition "is specific to the scalar Laplace equation and does
  not carry over to the vector elastic problem". Barthélémy & Bignonnet
  (IJES 2020, §2.1) say the opposite — the transfer-matrix formalism was
  imported *from* elasticity (Hervé & Zaoui 1993, Hervé-Luanco 2014). The
  obstruction to an elastic `LayeredSpheroid` is geometric: confocal surfaces
  are not homothetic, so harmonic degrees couple, and in elasticity they couple
  across perfect interfaces too.

## v0.10.0 — the field at a point, and `kn` finally meaning stiffness

The n-layer sphere was solved completely — Hervé–Zaoui recurrences for both
harmonics, perfect and imperfect interfaces, elasticity and transport — but
only the per-layer *averages* were reachable. `scripts/31_local_nlayers.jl`
said so in its own header: the deviatoric profile "needs the per-radius
evaluation of the 4×4 shear-recurrence state vector, which is not yet exposed
by MeanFieldHomogenization", and worked around it by importing private symbols
and restricting itself to hydrostatic loading. The material was all there; the
reconstruction was not.

It is now, and while cross-checking it against the C++ reference a second
thing surfaced: `SpringInterface(kn, kt)` stored *compliances* under names that
everywhere else in mechanics mean stiffnesses.

### Pointwise localization in an n-layer sphere

`local_strain_strain_loc(sphere, C₀, x)` returns `𝔸(x)` with
`ε(x) = 𝔸(x) : ε∞`, at any point — inside any layer **and** in the surrounding
matrix — with perfect, spring or Gurtin–Murdoch membrane interfaces. The three
other couplings follow the package's existing naming, so an arbitrary remote
strain **or** stress is a matter of picking the right one:
`local_stress_strain_loc`, `local_strain_stress_loc`, `local_stress_stress_loc`.
`local_strain`, `local_stress` and `local_displacement` give the fields for one
loading; the transport twins are `local_gradient_gradient_loc` and its three
siblings, plus `local_temperature`, `local_gradient`, `local_flux`.

The configuration is rotation-invariant about the center, so `𝔸(x)` is
transversely isotropic about `n = x/r` and carries no major symmetry: it comes
back as a `TensND.TensTI{4,T,6}` — six Walpole coefficients and an axis, in
closed form, never an 81-component array. Points are given either as a
Cartesian vector or as `(r, θ, φ)`.

Two details that are not decoration:

- **`side`.** Exactly on an interface the field has two limits, and with a
  spring or a membrane they genuinely differ — that difference *is* the
  interface law. Every entry point takes `side = :inner | :outer`, and
  `get_layer(sphere, r; side)` says which region that resolves to.
- **`LayeredSphereFields(sphere, C₀)`.** Solves the recurrence once. The
  `(sphere, C₀)` forms re-solve it on every call, which a field map over
  thousands of points pays for in full.

`shell_localization(fields, k)` recovers `(α_k, β_k)` from the same cached
amplitudes, so the pointwise and averaged routes cannot drift apart. That is
not a claim: averaging `𝔸(x)` over a shell reproduces
`_layer_avg_dev_shear_factor` exactly, mode 1 contributing `a_k`, modes 3 and 4
nothing pointwise, and mode 2 the `21/5 (3κ+μ)/μ` term. The whole field is
validated against Echoes' `loc_eE` / `loc_eS` / `loc_sE` / `loc_sS` to `1e-14`
on perfect, spring and membrane interfaces alike, and independently against
`div σ = 0`, `ε = sym ∇u`, and the closed-form exterior field of a single
sphere. `scripts/bench_echoes/benchmark_nlayers.jl` carries that cross-check as
a rerunnable benchmark (§5, worst discrepancy `8.4e-13` over the four couplings,
three interface families and ten points spanning every region), and no longer
reaches into the recurrence for its stress profile.

The `LayeredSpheroid` gained the matching vocabulary — `get_layer(…; side)`,
`LayeredSpheroidTransportFields`, a remote gradient given as a vector, and the
four `local_*_*_loc` couplings — so the two inclusion families are asked the
same questions the same way even though they will never share an
implementation. Its localization tensor is a general `Tens{2,3}`: a confocal
spheroid is not rotation-invariant about the field point.

`cumulative_strain_average` now integrates a truncated layer exactly, and the
stress and transport averages that had no counterpart are there:
`layer_stress_average`, `sphere_stress_average`, `layer_gradient_average`,
`sphere_gradient_average`, `layer_flux_average`.

The layered-spheroid theory page now **derives** its chart instead of quoting
it: `TensND`'s symbolic spheroidal coordinate system rebuilds the Lamé
coefficients, the frame and the volume element of Appendix A of
[barthelemyBignonnetIJES2020] at documentation-build time, and checks the
harmonicity of `Pₙᵐ(p) Pₙᵐ(q) cos mφ` — for `m = 0, 1`, which conduction uses,
and for `m = 2`, which the elastic counterpart will.

### Breaking changes

- **`SpringInterface(kn, kt)` now takes stiffnesses.** It used to store the
  numbers it was given as compliances, so `[u] = kn σ`, which inverts every
  limit: perfect bonding was `kn = 0` and a free surface `kn → ∞`. Both are now
  the other way round, matching the universal meaning of a spring stiffness and
  matching Echoes' `PRIMALDISC`, which accepts the same numbers. **Existing code
  passing compliances silently computes a different interface** — it does not
  error.

  The type still *stores* compliances, now named `sn`, `st`, because a perfect
  interface is then the exact zero `sn = st = 0` rather than an infinity, which
  keeps the near-perfect regime representable in `ForwardDiff.Dual` and in the
  symbolic types. Both spellings read and write:

  ```julia
  itf = SpringInterface(50.0, 20.0)        # stiffnesses
  itf.kn, itf.kt                           # (50.0, 20.0)
  itf.sn, itf.st                           # (0.02, 0.05)
  SpringInterface(; sn = 0.02, st = 0.05)  # the same interface
  ```

  To port: either invert your numbers, or switch the call to the keyword form
  `SpringInterface(; sn = …, st = …)` and keep them. The one-argument
  `SpringInterface(k)` still means "normal spring, tangentially bonded", now
  with `k` a normal stiffness and `st = 0`. Accessors `spring_stiffnesses` and
  `spring_compliances` are exported; `interface_param(i, :kn)` and
  `interface_param(i, :sn)` both differentiate, giving reciprocal
  sensitivities. `AnisotropicSpringInterface` is unchanged — its field was
  always honestly named `compliance`.

- **A minor bump is breaking for the resolver.** Downstream packages bounded on
  `"0.9"` must widen to `"0.10"`.

### Fixed

- **`ForwardDiff` through a single layer's modulus.** `_bulk_promote` was typed
  `NTuple{N, <:Any}`, which binds *one* element type, so a stack with one
  `Dual` layer among `Float64` ones raised a `MethodError` — while making every
  layer dual worked, which is why no test caught it. Differentiating
  `strain_strain_loc(sphere, C₀; layer = k)` with respect to one phase now
  works.
- **`ForwardDiff` through a single interface radius.** `LayeredSphere` required
  a homogeneous radii tuple, so `LayeredSphere((r₁, 2.0), …)` with `r₁::Dual`
  was a `MethodError`. Mixed-eltype radii are promoted.
- **`_shear_M_matrix`'s docstring** claimed rows 3–4 were `τ = σ/μ`; they are
  the physical tractions, as the file header always said.
- **The theory page** still stated `β_k = a_k` for the layer shear
  localization, dropping the mode-2 term that the code has carried since it was
  found to matter by 1–50 % on genuine multi-layer stacks.
- **`cumulative_strain_average` was wrong whenever the ball cut a layer part
  way through.** It weighted the *full-layer* average by the *truncated*
  volume, which is exact only where the field is uniform inside a layer — and
  the mode-2 term makes it vary as `r²`. On a three-layer sphere the error
  reached **6 %** at mid-layer radii and was **exactly zero at every interface
  radius**, which is precisely where the test suite evaluated it. The fix is
  closed-form, not a quadrature: the truncated shell average is
  `_layer_avg_dev_shear_factor` evaluated at the cut radius. Results at
  `r = r_k` are unchanged to machine precision, and the new test compares
  eleven radii against the pointwise field integrated by a 3-point
  Gauss–Legendre rule that is exact for this integrand.
- **A `LayeredSpheroid` diverged away from the particle, and raising `Nseries`
  made it worse.** At 100 particle radii the axial gradient read `-3.4e24`
  instead of `1` with `Nseries = 12`; `Nseries = 20` was already wrong at ten
  radii. Two independent defects, each verified separately.

  *The `Q` Legendre functions were run by an upward recurrence.* `Qₙ` is the
  **minimal** solution of the three-term recurrence — `|Qₙ/Pₙ| ~ ρ^{-(2n+1)}`
  with `ρ = |x + √(x²−1)|` — so going upward amplifies the seed's rounding
  error by `ρ^{2n}`. Against the same recurrence evaluated at 600 bits,
  `Q₁₅(5)` was wrong by a relative `1.3e13` and `Q₁₅(50)` by `7.9e26`, for
  **prolate as well as oblate**. `Q` is now obtained by Miller's downward
  recurrence, normalized on an exact low-degree closed form, with derivatives
  from the identity `(x²−1) dQₙᵐ/dx = n x Qₙᵐ − (n+m) Qₙ₋₁ᵐ` rather than from a
  second unstable recurrence.

  The direction is chosen from `ρ`, which governs both methods in opposite
  senses: upward is kept when `ρ ≈ 1` — a nearly degenerate spheroid, where
  `Q` barely decays and Miller would need thousands of steps. Getting that
  backwards is not benign: forcing Miller at `ρ = 1.017` (a 1:60 flat disc)
  silently returned a `Q` good to only `2e-7`, and with it a single-layer
  spheroid that no longer matched its closed-form Eshelby value.

  *The far-field condition was recovered instead of imposed.* In the matrix
  every growing amplitude above degree 1 vanishes identically, or the
  temperature would blow up at infinity. The core amplitudes are solved to make
  exactly that true, but recomputing the matrix block through the layer
  transfer reintroduced the linear solve's `O(1e-17)` residue — which
  `P_{2r-1}(q) ~ q^{2r-1}` then amplified past `1e40`. It is now written down.

  Both are pre-existing. The shipped documentation and tests never strayed into
  the affected region, which is why nothing caught them; the new
  `test/LayeredSpheroids/test_legendre_stability.jl` checks `Q` against a
  600-bit run of the **original** upward recurrence — an outside reference, not
  the new algorithm in wider precision — and pins the `Nseries`-independence of
  the far field.
- **An oblate `LayeredSpheroid` returned `NaN` for its local field on the
  revolution axis.** At `|p| = 1` the confocal chart degenerates and `h_p` is
  infinite; dividing a real number by `Inf` gives the correct `0`, but an
  oblate particle carries a **complex** `q`, and Julia's complex division
  `z / (Inf + 0im)` yields `NaN + NaN im`. The `NaN` then survived even a
  purely axial loading, through `H_trans * NaN` with `H_trans = 0`. The term is
  now written `∂T/∂p · p̄ / (c q̄ₚ)` — algebraically identical, regular on the
  axis — and the on-axis value matches the `p → 1` limit to `1e-9`.

  A *transverse* loading exactly on the axis carries two further `0/0`,
  `T_t / h_φ` and `∂T_t/∂p · p̄`. Both are **removable exactly**, with no
  asymptotic expansion: the `P¹` table is seeded `P₁¹(p) = -√(1-p²)`, i.e. the
  Condon-Shortley convention `P¹ₙ = -p̄ P′ₙ`, so the `p̄` of the numerator
  cancels the `p̄` of `h_φ` identically; and the Legendre equation removes the
  second derivative from the other term. Both then need only the ordinary
  Legendre table. The removal is checked three ways, because a wrong one would
  still look plausible: the on-axis value is the limit of the values around it,
  it is **independent of the azimuth** — which is undefined on the axis, and is
  what a wrong coefficient would break — and it points along the remote
  gradient, as the reflection symmetry requires.
- **`_shear_M_matrix_alv`'s docstring described a formulation the code does not
  use.** It announced a `τ = σ/μ` scaling whose rows 3–4 "no longer carry an
  explicit `μ` factor" and a perfect-interface jump `τ_rr⁺ = (μ⁻/μ⁺) τ_rr⁻`
  handled by `volterra_divide`. The code is in `σ` form (`blocks[3,1] = 4 M_μ`),
  its perfect interface is the identity, and no `volterra_divide` appears in any
  interface helper. The docstring now describes the actual state vector and
  says where the Volterra inverse of `μ` really occurs — which also means a
  layer with vanishing shear modulus is not admissible on that path.

## v0.9.0 — what a tolerance asks for, and what a distribution shape is

Two things that were quietly not what they looked like.

`abstol` was the only tolerance the manual named, and 54 call sites across the
documentation, the scripts and the test suite passed it alone — believing they
were tightening a self-consistent iteration. They were not. `reltol` has always
existed and has always been honored, the stopping test being the additive SciML
convention `‖Δx‖ ≤ abstol + reltol · ‖x‖`; and because a stiffness carries a
physical magnitude, `reltol · ‖C‖ ≈ 1e-7` at the default `reltol = 1e-8`
decides every one of those calls. `abstol = 1e-15` bought nothing.

And `PonteCastanedaWillis()` on an RVE that declared no distribution shape
returned Mori-Tanaka. Exactly Mori-Tanaka: a spherical distribution — the value
the field defaulted to — makes the formula degenerate onto the very estimate the
scheme exists to generalize. It said nothing about it.

### Breaking changes

- **Maxwell and PCW refuse an undeclared distribution shape.** `RVE()` now
  keeps `nothing` instead of inventing a unit sphere, and the two schemes that
  read the field raise an `ArgumentError` naming the collapse rather than
  performing it. Declaring the sphere restores the previous number exactly:

  ```julia
  rve = RVE(; distribution_shape = Ellipsoid(1.0))
  homogenize(rve, PonteCastanedaWillis()) ≈ homogenize(rve, MoriTanaka())   # true, and now on purpose
  ```

  The check is resolved in `validate_cell(rve, scheme)`, so it fires before any
  kernel runs, and `homogenize_alv` performs it too — that path does not share
  the validation. The new trait is `requires_distribution_shape(scheme)` and the
  resolver is `distribution_shape(rve, scheme)`, mirroring `requires_matrix` /
  `matrix_name` from v0.8.0.

  **The field stays on the `RVE`.** It was tempting to move it onto the schemes
  that read it, as the matrix role moved in v0.8.0, but the two are not alike.
  The matrix was never microstructure — naming a reference medium is a modeling
  decision, and the same phases are a matrix/inclusion composite under
  Mori-Tanaka and a matrix-free aggregate under the self-consistent scheme. The
  distribution ellipsoid *is* microstructure: the ellipsoidal symmetry of the
  medium's two-point statistics. That nine schemes out of eleven ignore it is
  the ordinary condition of microstructural data — `Voigt` ignores every phase
  geometry too — and not a sign that it sits in the wrong place. What it lacked
  was not a home but a default, and it no longer has one.

- **`abstol` now means the same thing for every solver.** It bounds the
  **Frobenius norm of the tensor**, whatever the `algorithm` and whatever the
  symmetry class. It used to bound that norm under `AndersonDefault` and an
  unweighted sum over canonical components under `NewtonDefault` and the
  `NonlinearSolve` algorithms — quantities that differ by a class-dependent
  factor, `√5` on the deviatoric component of an isotropic stiffness and `√3`
  on a conductivity. On the same first iterate of a porous self-consistent
  solve, Picard reported `‖Δ‖ = 59.354` and Newton `‖F‖ = 46.655`; they now
  both report `59.354`.

  `NewtonDefault` reaches it by working in the isometric parametrization of
  `_sc_param_weights` — Newton is invariant under a diagonal rescaling of its
  unknowns, so the iterates are untouched and only the stopping test and the
  Armijo condition change metric. The `NonlinearSolve` path instead receives
  the weights as the `internalnorm` of a `NormTerminationMode`, because a trust
  region is *not* invariant under such a rescaling: rescaling its unknowns moved
  where it stopped in the percolated regime, onto the plateau of the
  positive-definiteness guard where the residual is large and its Jacobian
  identically zero. Measuring differently is safe; searching differently is not.
  `NormTerminationMode` keeps both tolerances (`‖Δ‖ ≤ abstol` or
  `‖Δ‖ ≤ reltol · ‖Δ + u‖`), so `reltol` now bites on a SciML algorithm the way
  it always did on the built-in ones: asking for `reltol = 1e-2` against
  `1e-8` on a porous solve moves the answer by `0.29` under `TrustRegion` and
  `0.52` under `NewtonDefault`, where before the two could not be compared.

  Measured effect on results across porous and stiff RVEs at
  `f ∈ {0.1, 0.3, 0.5}`, three solvers, elasticity and conductivity: `2.9e-13`
  at worst, and nothing at all for `AndersonDefault`. Superlinear solvers
  overshoot their tolerance by orders of magnitude, so the guarantee matters
  where a solve stops early, not where it converges.

  The convergence comparator is `≤` everywhere; `AndersonDefault` used `<`.

### Fixed

- **The Echoes migration mapped `epsrel` onto `abstol`.** Echoes' fixed point
  tested `‖X - X_old‖ > epsrel · ‖X_old‖` — purely relative, with no absolute
  term (`epsabs` is a quadrature option there, never a scheme one). The
  translation therefore turned a relative requirement into an absolute one and
  left MFH's own `reltol` at its default, where it silently took over.
  `echoes2mfh` now emits `reltol`, together with `abstol = 0.0` so that a tight
  request such as `epsrel = 1e-15` is not overridden by the `1e-12` default.
  The `from_echoes` table is corrected with it.

- **MFH Studio advertised the wrong defaults.** Its `OPTION_DEFAULTS` table was
  flat, so it reported the ODE solver's `1e-8` / `1e-6` for `SelfConsistent`,
  whose solver actually uses `1e-12` / `1e-8` — a wrong number in the one place
  a user looks to find out what they are overriding. Defaults are now per
  scheme.

- `get_param` / `set_param` on a distribution-shape lens now say that the RVE
  declares no shape, instead of reporting a type mismatch against `Nothing`.

### Documentation

- A **Solver tolerances** section in the schemes manual: the stopping test, the
  defaults per scheme family, and why `abstol` alone will not tighten a
  stiffness iteration. `reltol` and `select_best` are listed wherever the kwarg
  lists were incomplete — the manual, `_evaluate(::SelfConsistent)`, and
  `AsymmetricSelfConsistent`, which named no tolerance at all.
- The percolation-threshold presets of the crack-distribution tutorial ask for
  what they meant (`abstol = 0.0, reltol = 1e-14`). The published thresholds are
  unchanged: `9/16` and `ε ≈ 1.158` were already converged far past the stopping
  test, and re-measuring at `reltol = 1e-15` moves them by `6e-9`. The `1.13e-6`
  residual against `9/16` is the linear extrapolation's, not the solver's.
- The distribution-shape section of the manual states which schemes read the
  field, why the others do not, and why it has no default.

## v0.8.1 — Aqua.jl, and the dispatch defects it found

MeanFieldHomogenization is now checked by
[Aqua.jl](https://github.com/JuliaTesting/Aqua.jl) on every run of the test
suite: method ambiguities, type parameters bound by nothing, undefined exports,
dependency hygiene, type piracy, and tasks left running at load. All eight
checks pass on the package as it stands, across the eighteen sub-modules and
nine extensions, with nothing exempted. Getting there fixed real defects.
Nothing in the exported API changes name or meaning.

### Bug fixes

- **`algo = :nestedquadgk` on a cylinder in an isotropic matrix raised an
  ambiguity `MethodError`** instead of taking the closed form. The four other
  algorithm names each had a rule saying the isotropic cylinder is analytical;
  this one had been left out, so the call fell between the generic `Val` rule
  and the cylinder rule with neither more specific than the other.

- **Dividing an ALV kernel by a `Symbolics.Num` was ambiguous.** An
  `AbstractALVKernel` is an `AbstractMatrix`, and Symbolics defines
  `/(::AbstractArray{<:Real}, ::Num)`; against the package's own
  `/(::ALVKernel*, ::Number)` neither signature wins. Symbolic viscoelastic
  parameters therefore failed on a plain division. The resolution lives in the
  Symbolics extension, where it belongs, and keeps the kernel's structure
  rather than degrading it to a bare matrix.

- **Five signatures had a type parameter bound by nothing.** `NTuple{N, T}` also
  matches the empty tuple, which leaves `T` undetermined. `LayeredSphere`,
  `LayeredSpheroid` and `_replace_geom_field` now require what they always
  meant — at least one layer, at least one semi-axis — and the zero-layer case
  keeps its `ArgumentError` message through an explicit method instead of
  reaching an undispatchable signature.

- **`add_phase!` with neither `fraction` nor `density`** hit an ambiguity in
  `_to_amount` rather than the intended message. The corner is now well defined
  for any caller, not only for the one path that checked it beforehand.

## v0.8.0 — the matrix role leaves the RVE

An `RVE` used to require a matrix. Every one of them, whatever scheme was going
to consume it — including the self-consistent scheme, whose whole premise is
that *no* phase surrounds the others. `RVE(:M)` fixed the matrix name before a
single phase existed, `add_matrix!` declared it differently from every other
phase, and a polycrystal could not be written down at all without inventing a
matrix for it.

Reading the code showed why the notion resisted: one name, `matrix_name`, was
carrying three independent things.

1. **The phase whose fraction is derived**, `1 - Σ f`. Pure bookkeeping, and
   legitimate for any scheme.
2. **The reference medium** localization tensors are computed in. A property of
   the *model*, not of the microstructure: the same phases are a
   matrix/inclusion composite under Mori-Tanaka and a matrix-free aggregate
   under the self-consistent scheme.
3. **The homogeneous solid** of microporomechanics — a constitutive reading
   again, and not necessarily the same phase.

They are now stated separately, and each where it belongs.

### Breaking changes

- **`RVE(:M)` and `add_matrix!(::RVE, …)` are removed.** An RVE is built empty
  and every phase is declared the same way:

  ```julia
  rve = RVE()
  add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => C_m); fraction = :rest)
  add_phase!(rve, :I, Ellipsoid(1.0), Dict(:C => C_i); fraction = 0.3)
  ```

  `fraction = :rest` marks the phase taking up the volume complement. It says
  nothing about morphology, and an RVE may have none.

- **The reference medium is named on the scheme**: `MoriTanaka(:M)`,
  `Dilute(matrix = :M)`, and likewise `DiluteDual`, `Maxwell`,
  `PonteCastanedaWillis`, `DifferentialScheme`, `AsymmetricSelfConsistent`.
  Named nothing, a scheme takes the `:rest` phase; with no such phase it raises
  an error listing the candidates rather than guessing, because a different
  matrix is a different composite. **Existing calls keep working**: on a
  migrated RVE, `MoriTanaka()` resolves to what used to be the matrix.

- **`SelfConsistent` has no matrix at all**, and neither do `Voigt` and `Reuss`.
  An RVE whose fractions all sum to one — a polycrystal, a granular aggregate —
  is now expressible, and `validate_rve` accepts it:

  ```julia
  poly = RVE(; closure = :strict)
  add_phase!(poly, :A, Ellipsoid(1.0), Dict(:C => C_A); fraction = 0.4)
  add_phase!(poly, :B, Ellipsoid(1.0), Dict(:C => C_B); fraction = 0.6)
  homogenize(poly, SelfConsistent(), :C)
  ```

- **Fraction closures.** How the declared fractions are made to sum to one is
  now explicit, through `closure =` on the constructor:
  `StrictFractions` (they must already sum to one), `ComplementFraction` (one
  phase absorbs `1 - Σ f`, the historical behavior), `RescaledFractions` (they
  are relative proportions: `2, 3, 5` reads as `0.2, 0.3, 0.5`). Inferred when
  not given. Crack densities take no part in any of them.

- **`SelfConsistent(; init = …)`.** With no phase distinguished the fixed point
  needs a seed of its own. Unset, it is the `:rest` phase when the RVE names one
  — the historical seed, so every existing number is unchanged — and the Voigt
  average otherwise, which is what makes a matrix-free RVE solvable. It also
  accepts `:voigt`, `:reuss`, a phase name or an explicit tensor.

  Making `:voigt` the blanket default was measured and rejected: across a porous
  RVE at `f ∈ {0.1, 0.3, 0.45, 0.6}` and a stiff one at `f ∈ {0.1, 0.3, 0.5,
  0.7}` the two seeds agree to `1e-9` or better, and on a deeply percolated
  oblate configuration the Voigt start makes `TrustRegion` stop where the
  finite-difference IFT Jacobian is exactly singular. No advantage, one
  drawback.

- **Removed for `RVE`**: `matrix_phase`, `matrix_property`,
  `matrix_volume_fraction`, and the one-argument `inclusion_phase_names(rve)`.
  Use `phase_property(rve, name, key)`, `volume_fraction(rve, name)`,
  `remainder_volume_fraction(rve)`, `phase_names(rve)` and
  `inclusion_phase_names(rve, matrix)`. The three names **survive for
  `ParticleAssembly`**, where a matrix is structural rather than a modeling
  choice: both N-body models resolve interactions between particles embedded in
  one reference medium.

- **`biot_tensor(rve, …)` and `poroelastic_parameters(rve, …)` gain `solid =`**,
  naming the phase that plays the homogeneous solid. Unset, it is the `:rest`
  phase, so existing calls are unaffected.

- **`TISymmetrize(; matrix_projection)` is renamed `reference_projection`.** It
  always described the projection of the reference medium, never of a matrix
  phase.

- **Voigt and Reuss now honor the `symmetrize` of every phase**, the former
  matrix included; it used to enter the bounds unprojected while the inclusions
  were projected. A no-op for an isotropic matrix — which every shipped example
  has — and a correction otherwise. The ageing-viscoelastic self-consistent
  path had the same asymmetry, hard-coding `NoSymmetrize()` for the matrix, and
  is corrected with it.

### Fixed

- `homogenize_alv` computed the Maxwell and PCW bodies against
  `matrix_property(rve, :C)` with the property **hard-coded**, ignoring the
  `prop` it had been given: both were wrong for `:K`.
- Three test files placed `using SymPy` inside a `@testset` body, where it runs
  *after* the `@syms` in the same block has already been macro-expanded. They
  only worked because an earlier file in the suite had leaked `SymPy` into
  `Main`, and could not be run on their own.
- `promote_rve` had no method for the new complement amount.
- The spelling checker scanned `docs/node_modules`, burying real hits under
  hundreds of vendored ones — and its documented `convert . --to us` would have
  rewritten third-party packages in place.

## v0.7.0 — a laminate answers the localization generics, exactly

### Breaking changes

- **`TensND` compat moves to `"0.4"`.** Three groups of functions have moved up
  into `TensND`, where they belong: the exact rotation-group averages
  (`isotropify`, `transverse_isotropify`, `mandel66_minor`,
  `array_from_mandel66`, `ti_average_mandel66`, `iso_average_mandel66`), the
  best-fit projections (`best_fit_iso`, `best_fit_ti`, `best_fit_ortho`) and the
  scalar-type predicate `is_hard_numeric`. All of them are re-exported from the
  same names as before — `MeanFieldHomogenization.isotropify`,
  `Core.isotropify`, `Schemes.best_fit_ti`, `Elliptic.is_hard_numeric` all still
  resolve — so no user code changes. Under 1.0 Julia's resolver nevertheless
  treats a minor bump as breaking, hence this one.
- Nothing else changed in meaning. The four localization generics gained
  `Laminate` methods; the pre-existing `layer_*_localization` names are
  untouched and remain exported.

### A `Laminate` answers `strain_strain_loc` & co.

A laminate is a *cell*, not an `AbstractInclusion`, so the
`(incl, ℂ₁, ℂ₀)` signature of the localization generics did not apply to it and
it exposed its own `layer_strain_localization` / `layer_stress_localization`
instead. Those two are the strain-driven and stress-driven tensors; the two
*mixed* ones existed nowhere, and they are exactly what a Levin-type
post-processing of a laminate needs — `Σ_i f_i σ⁰_i : 𝔸_i` is built from
`ℂ_i : 𝔸_i`, not from `𝔸_i`.

All eight generics now have a `Laminate` method, keyed on a layer **name** in
place of the reference-medium pair, the same shape as
`strain_strain_loc(::LayeredSphere, ℂ₀; layer)`:

```julia
strain_strain_loc(lam, :LAYER)      # 𝔸ᵢ      — same object as layer_strain_localization
stress_strain_loc(lam, :LAYER)      # ℂᵢ:𝔸ᵢ   — new, Σ fᵢ · = ℂʰᵒᵐ
strain_stress_loc(lam, :LAYER)      # 𝔸ᵢ:𝕊ʰᵒᵐ — new, Σ fᵢ · = 𝕊ʰᵒᵐ
stress_stress_loc(lam, :LAYER)      # 𝔹ᵢ
```

plus `gradient_gradient_loc`, `flux_gradient_loc`, `gradient_flux_loc` and
`flux_flux_loc` at order 2. Two kernels back them in
`Core/laminate_algebra.jl` at each order, inverting `ℂʰᵒᵐ` through the
cofactor `_inv_km6` and never through an LU, so all eight stay exact under
`ForwardDiff.Dual` and evaluable symbolically.

The sum rules `Σ fᵢ 𝔸ᵢ = 𝕀`, `Σ fᵢ 𝔹ᵢ = 𝕀`, `Σ fᵢ ℂᵢ:𝔸ᵢ = ℂʰᵒᵐ` and
`Σ fᵢ 𝔸ᵢ:𝕊ʰᵒᵐ = 𝕊ʰᵒᵐ` hold for perfect and dual interfaces and are asserted;
with a primal (spring, Kapitza) interface part of the macroscopic strain is
carried by the displacement jumps and the strain-side rules do not close — that
too is now asserted, so the caveat cannot rot.

### The localization tensors keep their symmetry class

They used to come back as a fully generic `Tens{4,3}`: 81 components and no
structure, although a laminate of layers all transversely isotropic about its
own normal has localization tensors that are transversely isotropic about that
same normal — evidently so, and for the same structural reason the effective
stiffness already exploited.

The reason was narrow: the effective-property path goes through `_wrap4`, which
reads the five Walpole coefficients off the layer-frame matrix, whereas the
localization path went straight to `inv_KM`. `_wrap4` could not simply be
reused, because `TensND.ti_params_from_KM` *projects* onto the
**major-symmetric** span — and `𝔸ᵢ = 𝕀 + ℙᵢ:(ℂʰᵒᵐ − ℂᵢ)` is a product of
major-symmetric tensors, which is not one. Fed a concentration tensor it would
have replaced `ℓ₃` and `ℓ₄` by their half-sum and reported the result as exact.

The new `TensND.ti8_params_from_KM` is the read-off on the whole
axially-invariant subspace, so `_wrap4_general` now returns an exact
`TensTI{4,T,6}` — `ℓ₇` and `ℓ₈` being structurally zero here — and the order-2
localizations go through the existing `_wrap2` and return `TensTI{2,T,2}`.

What that buys is not tidiness. A `TensTI` carries six coefficients where a
`Tens` carries 81; products and inverses become a 2×2 Walpole matrix and two
scalars; and a symbolic derivation that used to be written in Kelvin-Mandel
matrices because the tensors had lost their class can now be written in tensor
notation and runs in under a second where it took minutes.

### `scripts/46_lamellar_porous_swelling.jl`

The two-scale model of Dormieux, Lemarchand & Sanahuja, *Comportement
macroscopique des matériaux poreux à microstructure en feuillets*, C. R.
Mécanique **334** (2006) 304-310, derived symbolically end to end. The
interfoliar space of a clay particle is a laminate layer with a normal
stiffness alone — singular, hence regularized and taken to the limit — the
platelets are incompressible (`kₛ → ∞`), and the assembly of randomly oriented
particles and macropores is closed by the self-consistent scheme, its
orientation integral being the exact `isotropify` rather than a quadrature.
Equations (5) to (18) of the article come out of the code; none of them is an
input.

### Fixed

- **A `Symbolics.Num` volume fraction crashed the cell.** `add_layer!`,
  `validate_laminate` and `validate_rve` guarded a negative amount with
  `h isa Real && h < 0`. `Num <: Real`, so `h < 0` returned a `Num` and the
  `if` threw `TypeError: non-boolean (Num) used in boolean context`. `SymPy.Sym`
  is not `<: Real` and short-circuited, which is why every symbolic test passed:
  the suite only exercised `Num` through the bare kernel, never through the
  cell. The guards now go through `is_hard_numeric`, the predicate the rest of
  those files already used for exactly this distinction.

## v0.6.0 — the Laplace-Carson half of viscoelasticity

The viscoelasticity module could only go one way. It carried a complete
*ageing* time-domain pipeline — Volterra operators, structured kernels, every
scheme — and, for the non-ageing case, nothing at all: the correspondence
principle was available in the sense that all the schemes happen to be
complex-safe, so users evaluated a complex modulus at `p = iω` by hand and
called `homogenize`. That ad-hoc pattern was duplicated in two scripts and
three documentation pages, `Mod2S2P1D` was redefined inline in the bituminous
application with no source and no tests, and there was no way to get back into
the time domain at all. `scripts/61_freq_vs_time.jl` even carried a section
titled "Why no numerical Laplace inversion is needed".

This release supplies the missing half, and the section title has changed.

### Numerical inverse Laplace transforms, that autodiff survives

Four algorithms, all generic in the number type: `GaverStehfest`,
`FixedTalbot` (the default), `TalbotTrefethen` and `DeHoog`, behind
`inverse_laplace`, `inverse_carson` and `inverse_carson_rate`. They accept a
scalar, a `TensND` tensor of any symmetry class, or a `6×6` Mandel matrix, and
the symmetry class of the result is the class the transform returns — axis
included.

`ForwardDiff` goes through all four: with respect to model parameters, with
respect to `t`, nested, and over `Complex{Dual}` on the contour methods. That
requirement is the reason none of this delegates to `InverseLaplace.jl`, the
only registered package for the job: its entry points are annotated
`t::AbstractFloat`, and `ForwardDiff.Dual` is `<: Real` but not
`<: AbstractFloat`, so a `Dual` is refused before any arithmetic happens. Its
`Weeks` method is hard-coded to `Float64` and pulls FFTW, and its Stehfest
coefficients are `BigFloat`.

Three things the measurements contradicted, and which are documented as
measured rather than as expected:

- **Branch cuts are not the obstacle.** The Talbot contours are Hankel-shaped —
  they wrap *around* the negative real axis rather than crossing it, which is
  what they were designed for. Every fractional model inverts at `1e-12`.
- **Oscillation is.** A rotating phase is invisible to a method whose nodes are
  all real; `GaverStehfest` returns O(1) error on `sin(ωt)/ω` where
  `FixedTalbot` returns `1e-12`.
- **The error is absolute, not relative.** Every method controls its error
  against the scale of `f`, so once a relaxation function has decayed many
  decades below its initial value the relative error is unbounded while the
  absolute one still holds. This is intrinsic, and it is why a viscoelastic
  *solid* — with a plateau `E_∞ > 0` — is the comfortable case.

The Gaver-Stehfest weights are computed once in exact `Rational{BigInt}`
arithmetic and converted to the working type on demand, so `BigFloat` genuinely
buys precision: at `N = 34` the error is `1e-14` where `Float64` is at `1e5`.

`DeHoog`'s nodes do not depend on `t`, which allows one pass over the transform
to serve several times — but only over a window spanning a factor of about
three, because its accuracy depends on `t/T` alone. Rather than leave that as a
trap, the grid form sorts the times, splits them into scale blocks and gives
each its own node set. A 200-point grid over seven decades then costs ~500
transform evaluations instead of 6600, at uniform accuracy.

### The exact Kelvin ⇄ Maxwell conversion

`maxwell_to_kelvin` and `kelvin_to_maxwell` convert a generalized Maxwell chain
into the exactly equivalent generalized Kelvin chain and back, so that
`J*(p) R*(p) = 1` identically.

The reference implementation of this idea (`Kelvin2Maxwell.py` in ECHOES) does
it symbolically: expand `R*(p)∏(1+pτᵢ)` into a polynomial and root-find. That
is exact on paper and ill-conditioned in practice — real relaxation spectra
carry dozens of branches over six to ten decades, and such a polynomial cannot
be formed in floating point without losing its roots.

The implementation here uses the structure instead. In the retardation-time
variable `σ = -1/p` the transform becomes a Stieltjes function, strictly
monotone between consecutive poles, so its zeros **interlace** the input times:
one in each gap, plus one beyond the last pole exactly when the material is a
solid. Every unknown is isolated before any arithmetic happens, bisection in
`log σ` cannot fail whatever the spread, and the residues come out positive by
construction rather than by luck. The round trip stays at `1e-15` out to twenty
branches spread over six decades.

The bracketing tables are deliberately *not* symmetric between the two
directions — relaxation → creep has no root below the first pole, creep →
relaxation does — which is also why a fluid Kelvin chain of `n` branches
converts to `n+1` Maxwell branches. Both facts are asserted in the tests.

`ForwardDiff` traverses the conversion, by the implicit function theorem: the
bisection runs on values only, and a single Newton step taken in the full `Dual`
type transports the derivatives exactly.

Three independent oracles pin it down, chosen so that no single mistake could
satisfy all of them: the reciprocity identity at machine precision, the
analytic Zener relations printed by the Python reference, and the closed-form
`cosh`/`sinh` Burgers relaxation function — which exercises the whole fluid
path (degenerate-branch detection, the unbounded outermost bracket, both
residue signs) and matches to `1e-16`.

`prony_fit_relaxation` / `prony_fit_creep` fit a chain to an *arbitrary*
transform by collocation, with a non-negativity constraint by default. That is
not only a matter of passivity: on a fractional Zener model the constrained fit
is three times more accurate on the master curve than the unconstrained one,
which puts half its moduli below zero.

### A catalog of rheological models

Every model exposes all four functions — `relaxation`, `creep`,
`carson_relaxation`, `carson_creep` — whichever one it was defined by, plus
`complex_modulus`, `glassy_modulus`, `equilibrium_modulus` and `is_fluid`. A
new model is complete once `carson_relaxation` is written; the rest follows
from exact algebra or numerical inversion.

Shipped: `Spring`, `Dashpot`, `MaxwellUnit`, `KelvinUnit`, `PronyRelaxation`,
`PronyCreep`, `zener_maxwell`, `zener_kelvin`, `burgers`, `ScottBlair`,
`FractionalMaxwell`, `FractionalKelvin`, `FractionalZener`, `Rabotnov`,
`HuetSayegh`, `Model2S2P1D`, `LogarithmicCreep`.

Two of these turned out to need less machinery than expected:

- **Rabotnov's Carson transform is elementary.** The kernel is normally written
  with a Mittag-Leffler function; its transform is
  `μ₀(1 + λ₀/(p^{α+1} + β))`, with no special function in it. That is what let
  `scripts/52_rabotnov_mittag_leffler.jl` drop `PyCall` and its dependence on
  an external Python module — the script is now self-contained, and still
  reproduces the closed form to `1.3e-3`.
- **2S2P1D is exact in both domains.** Because `LC{tᵃ} = Γ(a+1)p⁻ᵃ`, the
  denominator of the 2S2P1D transform is the transform of an ordinary function
  of time, term by term. So one model object is *simultaneously* an exact
  Laplace-Carson model and an exact Volterra model — `creep_kernel`,
  `carson_creep_kernel` and `creep_kernel_law` expose the three views — and the
  two homogenization routes can be compared on it with no approximation on the
  input side.

`MittagLeffler.jl` is a new weak dependency supplying closed-form time values
for the two fractional kernels that have one. It is deliberately restricted to
`0 < a ≤ 1`: v1.0.0 of that package silently returns `1.0` for `1 < a < 2` at
small `|z|` — `mittleff(1.3, 1.0, -0.5)` gives `1.0` where the series gives
`0.633` — and the extension declines that band rather than propagate it.
Nothing depends on the extension: without it the numerical inversion of the
closed-form transform supplies the same numbers to `1e-10`.

### One object, both routes

`iso_rheology(k_model, μ_model)` lifts two scalar models to an isotropic
fourth-order model. That single object then yields `carson_relaxation(m, p)`
for the transform route **and** `ViscoLaw(m)` for the ageing time-domain one,
so a non-ageing material is described exactly once and the two pipelines are
guaranteed to be comparing the same thing.

`homogenize_lc(build_cell, scheme, prop; p)` or `; times` is the driver:
it evaluates the cell at whatever Carson variables the inversion asks for, runs
the ordinary elastic `homogenize`, and — in the `times` form — inverts the
answer back to the time domain. Any cell works, nested chains included.

Choosing `GaverStehfest` there has a consequence beyond accuracy: its Carson
variables are all **real**, so the whole sweep runs in real arithmetic. That
costs a third fewer scheme evaluations, and it is the one way to use
`SelfConsistent(algorithm = NewtonDefault())` on this route, whose `ForwardDiff`
Jacobian cannot carry a `Dual` over a complex scalar.

### Symbolic parameters, and symbolic inversion

The catalog is generic in its scalar type, but its constructors were not: they
validate `0 < k < h < 1`, `τ > 0` and so on, and a `Symbolics.Num` makes those
comparisons *expressions* rather than `Bool`s. Every validation is now guarded
by `Elliptic.is_hard_numeric` — the same predicate the elliptic integrals have
used since that confusion first bit — so a symbolic model simply is not
validated, and

```julia
@variables E∞ E₁ τ₁ p
carson_relaxation(zener_maxwell(E∞, E₁, τ₁), p)
```

returns an exact expression. `SymPy.Sym` works too, which needed the model type
parameters relaxed from `T <: Real` to `T <: Number` (`Sym` is not `<: Real`).

The one operation that genuinely cannot take symbols is the Kelvin ⇄ Maxwell
conversion: it locates its roots by bisection between consecutive poles, which
needs a real ordering and a real sign test. It now refuses with that reason
instead of failing later inside the bracketing, and `carson_creep` on a
symbolic Prony chain falls back on the exact `J* = 1/R*` rather than
propagating the refusal.

With `SymPy` loaded, `inverse_laplace` and `inverse_carson` accept a **symbolic
time**, which means something different from a numeric one: not the value there
but the function. They hand the transform to SymPy and return `R(t)` or `J(t)`
in closed form. That is how the Burgers relaxation function — the `cosh`/`sinh`
expression that serves as this package's numerical oracle — is now *derived*
rather than transcribed.

Where that route stops is worth stating, because it is not where one would
guess. A rational exponent inverts exactly; the Cole-Cole model at `α = 1/2`
has a genuine closed form in `erfc` and SymPy finds it. A **symbolic** exponent
does not work, and fails in the worst way — SymPy returns `nan`, which would
propagate silently — so the extension warns. A sum of several fractional powers
(Huet-Sayegh, 2S2P1D) is left unevaluated, and is warned about too. Neither is
a gap here: those inverses are Mittag-Leffler functions or worse, which is
exactly why the four numerical quadratures exist.

!!! note
    `Symbolics` exports its own `inverse_laplace` — a five-argument transform
    for solving ODEs — so `using MeanFieldHomogenization, Symbolics` makes the
    bare name ambiguous and it must be qualified. `inverse_carson`, the one
    used for anything in the catalog, is unaffected.

### Documentation

New: a theory page on the Laplace-Carson route (including the interlacing
theorem and its two asymmetric bracketing tables), a manual page for the model
catalog, a manual page on choosing an inversion algorithm, an API page, and
three tutorials generated from `scripts/63`, `64` and `65` — the Kelvin ⇄
Maxwell conversion with the interlacing drawn, the four inversion algorithms
measured against four exact pairs, and the model gallery with master curves,
Cole-Cole and Black diagrams. A hand-written page covers the symbolic route
(`scripts/66` is its terser twin, deliberately unpublished: SymPy costs ~45 s
of build time per page).

`scripts/61_freq_vs_time.jl` now compares **three** routes rather than two, and
separates the discrepancies instead of merely bounding them: the trapezoidal
error of the time route (`8e-4`), the inversion error (`1e-9`, flat across five
decades of time) and the Gaver-Stehfest budget (`1e-8` to `3e-6`) each occupy
their own column, so a regression in any one of the three would move exactly
one of them.

`docs/src/applications/bituminous.md` no longer defines its own 2S2P1D struct.
The migration is numerically empty — the inline struct's field *names* were the
reverse of the literature's, but its formula and positional order already
matched, so all twelve calibrated parameter sets carry over argument for
argument and the page still reproduces the ECHOES reference values. It gains a
closing section that inverts the mix's effective modulus back into the time
domain, which is what a structural calculation under a passing wheel actually
needs.

## v0.5.1 — a symbolic laminate is symbolic all the way down, frame included

### A stray `1.0` in an exact answer

A bilayer of symbolic isotropic layers used to come back as

```
C₃₃₃₃ = 1.0/(f/(k_A + 4*mu_A/3) - (f - 1)/(k_B + 4*mu_B/3))
C₂₃₂₃ = 1.0*mu_A*mu_B/(f*mu_B + mu_A*(1 - f))
```

That `1.0` is a `Float64` in a calculation that has no business containing one,
and it came from the **frame**, not from the kernel. `Laminate(; normal = (0,0,1))`
built a `CanonicalBasis{3,Float64}`; `_wrap4` read its third column as
`(0.0, 0.0, 1.0)`; and a `TensTI` converts its axis to the element type of its
*data* and rebuilds its components from the Walpole basis of that axis — so the
float came back as a symbolic `1.0` in front of every coefficient.

It was invisible to every existing test, and necessarily so: `Sym(1.0) == Sym(1)`
is `true`, so no `simplify(… - …) == 0` check can see it. The regression test
added here asserts `is_Integer` on the axis instead.

A canonical frame is now read for its axis **exactly** — `(0, 0, 1)` — whatever
its own element type, and the declared floor `T` sets that element type, so
`Laminate(; T = Sym)` and `Laminate(; T = Sym, normal = (0, 0, 1))` finally agree.
Neither form is required any more: symbolic moduli, thicknesses and frames are
all carried by ordinary construction, and `T` goes back to being what it says it
is — a floor.

An obliquely oriented *numeric* frame still contributes floating-point axis
components. That is correct: the geometry itself is floating point.

### A symbolic frame

`normal` and `euler_angles` now accept symbolic components, so
`Laminate(; normal = (0, sin(θ), cos(θ)))` and `Laminate(; euler_angles = (θ, 0, 0))`
are ordinary laminates. Previously an explicit `basis = …` was the only route,
and the docstrings said so.

A symbolic normal is completed into `(ℓ, m, n̂)` by plain Gram-Schmidt against a
reference axis — no trigonometry and no `atan2`, so the frame stays as readable
as the normal it came from, and the result is `tsimplify`'d once at construction
rather than leaving `sin(θ)/√(sin²θ + cos²θ)` to resurface in every component.

The in-plane pair has to be *chosen*, and non-degenerately. A numeric normal
picks whichever canonical axis is least aligned with `n̂` — a comparison, which
can never degenerate. A symbolic normal cannot answer that comparison, so it
falls back to `e₁`, and the new keyword

```julia
Laminate(; normal = (cos(θ), 0, sin(θ)), in_plane = (0, 1, 0))
```

names another reference when the normal may lie along `e₁`. The choice is
physically immaterial — the effective property is invariant under rotation about
`n`, and the test suite states that as a symbolic identity rather than to a
tolerance — but it is not automatable. `in_plane` goes with `normal` only;
`euler_angles` and `basis` already fix the whole frame, and passing it there now
raises rather than being ignored.

### Coaxiality without a tolerance

`_parallel` answered `false` for every non-`Real` element type, so a symbolic
`TensTI` layer was never recognized as coaxial with the laminate normal and the
exact `TensTI` return degraded to a generic `Tens`. Collinearity of symbolic axes
is now decided **structurally** — the same expression, up to an overall sign.
Conservative by construction, never a false positive: a laminate that *is*
transversely isotropic may still come back generic, which loses nothing.

### Smaller repairs on the same path

- `show` applied `round(float(x))` to the normal, which a symbolic frame has
  neither of.
- `interface_jump` rotated back to canonical components through `_basis_matrix`,
  which pins the frame to `Float64`; it now uses the new element-type-preserving
  `Core._frame_matrix`. `_basis_matrix` is untouched — its other callers
  (cubature, multipole sums, axis permutations) do want `Float64`.
- `_compute_period` seeded the sum with `zero(T)` of the *declared* floor, so a
  symbolic stack computed `0.0 + h₁ + h₂`. SymPy absorbs that leading float;
  `Symbolics.Num` need not. It now sums from the first thickness, which is what
  its own docstring already promised.
- `laminate_alv` is a trapezoidal Volterra discretization on a grid of times and
  is numerical by construction. It now says so, with an `ArgumentError` naming
  the elastic `Laminated` scheme, instead of failing several frames deeper on
  `Float64(::Sym)`.
- Two docstrings promised checks the code does not perform — that
  `validate_laminate` verifies `Σ f ≈ 1`, and that mixing `thickness` and
  `fraction` across layers is rejected. Neither exists, and neither could exist
  for a symbolic or `Dual` thickness. The prose now describes what the code does.

### ForwardDiff now reaches through a tilted frame

Checking the frame paths for non-regression turned up a defect that was neither
new nor in this package: `ForwardDiff` could not differentiate a laminate whose
normal is not `e₃` at all. `TensND.KM` of a `Dual`-valued `TensISO` in a
`RotatedBasis` returned a 9×9 matrix instead of 6×6, because
`TensND._store_symmetric` keyed its round-off tolerance on `AbstractFloat` and a
`Dual` is not one.

**Requires TensND 0.3.6**, where that is fixed; `[compat]` moves from `"0.3.5"`
to `"0.3.6"` accordingly — a lower bound at patch level, because the fix is a
patch release and the laminate genuinely needs it. `test_laminate_dual_compat.jl`
gains the tilted case, on a modulus and on a thickness, against central
differences.

### Compatibility

No exported name was added or removed and no numerical result changed: `1.0*x`
and `x` are the same number. Two things did change type, both of them fixes:
`Laminate(; T = Sym, normal = (0, 0, 1))` now yields a `CanonicalBasis{3,Sym}`
where it yielded a `CanonicalBasis{3,Float64}`, and a symbolic laminate of TI
layers now returns a `TensTI` where it returned a `Tens`. Code that dispatched on
either would see the difference.

## v0.5.0 — a dormant period before setting, and no proper names in the code

### Breaking changes

Nothing in `src/` changed, and no exported name was touched. The minor bump is
for a **change of results**: the two hydration chapters print different early-age
numbers than v0.4.6 did, and the setting time they report moves by five hours.
Below 1.0 the resolver treats a minor bump as breaking regardless, so a
downstream `[compat] MeanFieldHomogenization = "0.4"` must be widened.

**Requires ChemistryLab 0.12.** `docs/Project.toml` now pins
`ChemistryLab = "0.12"`: the dormant period below is passed through that package's
`induction` keyword, which 0.11 does not have.

### Parrott & Killoh has no dormant period, and the micromechanics felt it

ChemistryLab 0.12 adds an induction factor `β(t) = 1 - exp(-(t/τ)^m)` on the
clinker silicates, because `parrot_killoh_avrami` floors its Avrami argument so
the ODE can leave `ξ = 0` at all — which means the clinker starts hydrating the
instant the water does. Measured against a real CEM I calorimetry record there,
the unmodified model had released 23.8 J/g by 2.7 h where the calorimeter saw 4.7.

Here the consequence is not a heat curve, it is a **reported result**.
`paste_moduli` returns setting as a genuine zero of the self-consistent fixed
point — below percolation the hydrate foam has no stiffness and the model says so
rather than returning a small number — and a model that hydrates from `t = 0`
crosses that threshold too early:

| | setting (first non-zero modulus) |
|--- |--- |
| without the dormant period (v0.4.6 behavior) | 13.7 h |
| with it (this release) | **18.9 h** |

Both `scripts/common/` helpers carry it, and with the same `τ = 5 h`, `m = 2.5` on
the silicates only. That symmetry is deliberate: `IONIC_CALIBRATION` exists so the
ionic and stoichiometric routes differ in *what forms*, not in *how fast*, and
adding a dormant period to one alone would have broken exactly the comparison the
two chapters are built on. `waller` is excluded throughout — its sigmoid
`α(t) = 1/(1 + (τ/t)^n)` already vanishes as `t → 0`, so an SCM entered through it
carries its own onset delay and damping it again would count the delay twice. The
aluminate is excluded too: it reacts within minutes of wetting.

Long-age results are unaffected. The mean degree of hydration at 28 days and the
porosity come out the same with and without the dormant period, which is what one
would expect of a delay of a few hours in a 28-day calculation.

### Proper names out of the code, references kept

Author names have no business in file names or identifiers; they belong in the
bibliography. So:

| was | is |
|--- |--- |
| `scripts/common/lavergne_model.jl` | `scripts/common/paste_micromechanics.jl` |
| `scripts/common/lavergne_hydration.jl` | `scripts/common/stoichiometric_hydration.jl` |
| `scripts/44_lavergne_hydration_micromechanics.jl` | `scripts/44_stoichiometric_hydration_micromechanics.jl` |
| `scripts/common/pichler_model.jl` | `scripts/common/quasibrittle_strength.jl` |
| `scripts/bench_echoes/benchmark_pichler.jl` | `scripts/bench_echoes/benchmark_strength.jl` |
| `lavergne_paste_moduli`, `lavergne_concrete_moduli` | `paste_moduli`, `concrete_moduli` |
| `lavergne_semiadiabatic` | `semiadiabatic_cell` |
| `NTHETA_LAVERGNE` | `NTHETA_PASTE` |
| `LAVERGNE_HYDRATES`, `_INCLUSIONS`, `_PORES`, `_E_NU` | `PASTE_HYDRATES`, `PASTE_INCLUSIONS`, `PASTE_PORES`, `PASTE_E_NU` |
| `LAVERGNE_GROUPS` | `PASTE_PHASE_GROUPS` |
| `LAVERGNE_MIX_C100`, `_VESSEL_CP`, `_LOSS_A`, `_LOSS_B` | `CALORIMETRY_MIX_C100`, `_VESSEL_CP`, `_LOSS_A`, `_LOSS_B` |
| `pichler_strength` | `quasibrittle_strength` |
| `py_pichler` | `py_strength_reference` |

The bibliography keys `Lavergne2018` and `pichler2011`, and every citation in
prose, are untouched — that is where the names belong. Every rename is a `git mv`,
so the history follows the files. None of this touches `src/`: it was all example
material, which is why the API is unchanged.

Earlier CHANGELOG sections still name the old files. They are the record of what
was released at the time and are left alone.

### Also

- A docstring in the stoichiometric helper named a private, unpublished script
  when describing how the aluminate cascade passes each phase's kinetic remainder
  from one reaction to the next. It now describes the mechanism without naming the
  file.

## v0.4.6 — the ionic chapter is now certified end to end

`src/` is untouched. This is a documentation release, and it exists for the same
reason v0.4.5 did: the documentation is deployed on a tag, so the corrections below
cannot be published without one.

`docs/Project.toml` now requires `ChemistryLab = "0.11"`, which is where the fix
lives.

### One instant in forty was not certified, and the chapter said otherwise

`applications/ionic_hydrating_paste.md` asserts that every replayed instant is
proved optimal. That was false as published: the FIRST of the forty sampled
instants, `t = 4320 s`, fell back to an uncertified interior-point composition
holding 56 interior species where the answer has 25. ChemistryLab 0.11.0 walks the
warm-start chain up to the first requested instant, which certifies it — stationarity
9.1e-13, element balance 6.1e-15 mol — so the claim is now true rather than weakened.

The 28-day heat of the limestone paste moves from 420.2 to 420.3 J/g, which is the
size of the error the silent fallback was hiding. Every other figure of the chapter
is unchanged: 404.8 J/g without limestone, ΔT of 19.1 and 18.6 K, the ettringite
depletion, the porosity table.

### A pointer to the chemistry on its own

Sections 1 to 6 of that chapter are chemistry, and they are now also published by
ChemistryLab.jl as *The full Portland cement, through its pore solution*, backed by
`scripts/ionic_hydration.jl` there. A reader who does not need the elastic modulus
of §7 is sent to it.

## v0.4.5 — calorimetry in the ionic chapter

`src/` is untouched. This release exists because the documentation is deployed on
a tag: the chapter below cannot be published without one.

`docs/Project.toml` requires `ChemistryLab = "0.10"` and `OptimaSolver = "0.4"`.

### Heat of hydration, from the certified compositions

`applications/ionic_hydrating_paste.md` gains a calorimetry section for the same
two mixes it already compares, the CEM I 52.5 N of Lavergne et al. (2018) with
and without its 3.5 % limestone. Cumulative heat and heat rate come from
`ChemistryLab.heat_release`, i.e. from the drop in system enthalpy across the
**certified** speciations the rest of the chapter uses — Eqs. (17)–(21) of that
paper, with reactants, ions and hydrates each counted once and no reaction
stoichiometry to write down. That matters here because the hydrates are not
produced by any reaction the model declares: they are precipitated by the Gibbs
minimization.

Isothermal at 20 °C, per gram of binder: 185, 288, 347 and 420 J/g at 1, 3, 7 and
28 days with the limestone, against 184, 279, 336 and 405 J/g without, the heat
rate peaking at 4.8 and 5.0 mW/g. The limestone *raises* the heat slightly rather
than diluting it, because the carbonate is not inert — it converts the aluminate
to monocarboaluminate and stabilizes the ettringite, and both reactions are
exothermic.

The NF EN 196-9 (Langavant) cell is then integrated on that heat with the
published calibration of Lavergne et al. §4.1 — vessel 380 J/K, sand 812 J/K from
the `Qtz` entry of CEMDATA18, losses `a = 75 J/(h·K)` and `b = 0.260 J/(h·K²)`,
on their Table 11 `C100` mix. It reaches 19.1 K above ambient at 22 h, against an
adiabatic 74.5 K: the sand and the losses absorb three quarters of the heat, which
is what the test is designed to do. The one approximation — the acceleration of
the kinetics by the temperature rise is not fed back — is stated in the chapter,
along with why: under partial equilibrium the heat source inside the ODE would
require differentiating the equilibrium map, and `ChemistryLab` now refuses that
combination with a warning rather than returning the 207 K it would otherwise
produce.

`scripts/common/ionic_hydration.jl` gains `binder_mass` and `calorimeter`
arguments, and `lavergne_semiadiabatic()`, which builds that cell with the
published numbers.

## v0.4.4 — certified equilibria in the ionic chapter

`src/` is untouched.

`docs/Project.toml` requires `ChemistryLab = "0.9"`, whose `speciated_states`
certifies every instant it replays against the KKT conditions. The Gibbs problem
is convex, so those conditions are sufficient: the compositions the ionic chapter
reports are now **proved** to be the global optimum rather than accepted from an
interior-point iteration that stops on `MaxIters`. Measured on this chapter's own
OPC, every replayed instant is certified with an element balance between 1e-11
and 1e-13 mol, the six-hour instant having stood at 6.9e-2 mol before.

The visible consequence is that the no-limestone mix of §4 now behaves as the
chemistry says it should: the ettringite peaks at 6 hours and is depleted to
nothing, the AFm settling at the sulfate budget, with the porosity and the
elastic modulus monotone across all forty instants.

`docs/Project.toml` also drops the patch level from the `OptimaSolver` bound,
`"0.2.7"` becoming `"0.2"`, and loses a comment block that had drifted between the
two entries and hid `ChemistryLab` from view. `TensND = "0.3.5"` in
`Project.toml` keeps its patch level, and should: versions up to 0.3.3 rotate
incorrectly, which this package relies on being fixed.

## v0.4.3 — the ionic chapter on ChemistryLab's public API

**`src/` is untouched.** This release moves `scripts/common/ionic_hydration.jl`
off three private internals of ChemistryLab and onto `speciated_states`, public
since ChemistryLab 0.8.0, and fixes a dependency that was never declared.

- **`speciated_states` is now delegated**, not reimplemented. The guards the
  replay needs — a warm start capped at the element budget and projected back
  into `{Aₑn = bₑ, n ≥ 0}`, and a back-end instance the integration has not
  used — belong to the coupling, not to a chapter.
- **`using SciMLBase` is gone** from `ionic_hydration.jl`, and with it a real
  portability bug: SciMLBase was never declared in `docs/Project.toml` and only
  resolved transitively, so the script ran on one machine and failed on another
  with `Package SciMLBase not found in current path`. It was never an OS
  difference, only two Manifests disagreeing about an undeclared dependency.
- **`docs/Project.toml`** now requires `ChemistryLab = "0.8"`.
- **Diagrams**: the two cement chapters were missing the
  `%%{init: {"flowchart": {"useMaxWidth": false}} }%%` directive every other
  diagram in this documentation carries. Without it mermaid stretches the `<svg>`
  to the full column and the `max-width` rule in `custom.css` has nothing to act
  on, which is why their boxes came out oversized.

## v0.4.2 — coupling to ChemistryLab.jl (documentation and scripts only)

**`src/` is untouched by this release.** It exists so the two new Applications
chapters and their scripts reach the published documentation: Documenter only
rebuilds `stable/` on a tagged release, so without it the material below stays
in `dev/`. No exported API, no behavior, and no compat bound of the package
changed, so nothing downstream needs adjusting.

### Two cement chapters, computing what other chapters correlate

- **`applications/hydrating_blended_paste.md`** — the four-scale elastic model
  of Lavergne et al. (*CCR* **104**, 2018) with the volume fractions **computed
  by the chemistry**: a Parrot & Killoh + Waller kinetics run, the reaction
  stoichiometry, and the molar volumes of CEMDATA18, instead of a Powers-type
  correlation. Every other cement chapter here starts from such a correlation,
  and the repository held three mutually incompatible parameterizations of it;
  this chapter deliberately adds no fourth, since its model takes volume
  fractions and never `(w/c, α)`.

- **`applications/ionic_hydrating_paste.md`** — the same paste with the
  chemistry turned inside out. The clinker only dissolves, into Ca²⁺, SiO₂,
  AlO₂⁻, FeO₂⁻, SO₄²⁻, CO₃²⁻ and H⁺, and a Gibbs free-energy minimization at
  every accepted step decides which hydrates are stable. No sequencing rule is
  written anywhere.

  The two chapters share the micromechanics exactly, so the comparison is
  meaningful: they agree to a few percent at one day and drift to some ten
  percent at 28 days, which measures what a hand-written priority cascade costs
  against letting the thermodynamics choose. The ionic route also yields what
  the stoichiometric one cannot: the pore-solution pH (12.5–12.6 throughout, a
  computed quantity), a setting threshold at 11.6 h where the self-consistent
  fixed point collapses because the solid has not percolated, and an aluminate
  sequence that changes with the mix without a line of code changing — with
  3.5 % calcite the carbonate stabilizes the ettringite and it survives to
  28 days, without limestone it peaks at 6 hours and is depleted to nothing.

### Scripts

- `44_lavergne_hydration_micromechanics.jl`, `45_ionic_hydration_micromechanics.jl`,
  and the shared `common/lavergne_model.jl` (micromechanics),
  `common/lavergne_hydration.jl` and `common/ionic_hydration.jl` (chemistry).
  These two scripts activate `docs/` rather than the repository root — they are
  the only ones needing a package outside the MeanFieldHomogenization stack.

### Documentation environment

- **`docs/Project.toml`** gains `ChemistryLab` (compat `"0.7"`, which requires
  ChemistryLab ≥ 0.7.1 for the ionic chapter to be trustworthy), `OrdinaryDiffEq`,
  `DynamicQuantities`, `OrderedCollections` and `OptimaSolver`.
- **Bibliography**: `Lavergne2018` and `ParrotKilloh1984` added, both verified —
  the DOI against Crossref, the 1984 proceedings entry against independent
  bibliographic records.


## v0.4.1

Two `DiluteDual` bugs on the ageing-viscoelastic path, both found by a new
invariant test: at the loading time ``t_0``, the ALV effective relaxation must
equal the *elastic* estimate built on the instantaneous (glassy) moduli.

### Bug fixes

- **`homogenize_alv(…, DiluteDual(), …)` added stiffness contributions to a
  compliance.** The dual scheme accumulates in compliance space, but the
  per-phase data collected upstream are the stiffness contributions ``Ñ``;
  only the crack branch of the dispatch converted them. A **solid** inclusion
  therefore entered the sum with the wrong quantity — and the wrong dimension.
  The symptom was not subtle: a sphere at ``k = 20``, ``\mu = 8`` in a matrix at
  ``k = 10``, ``\mu = 4``, ``f = 0.25`` gave ``(K, \mu) = (0.073, 0.34)``
  instead of ``(11.78, 4.82)``. Contributions are now mapped by the exact
  identity ``H̃ = -J̃_M ∘ Ñ ∘ J̃_M`` — for a homogeneous inclusion, the
  ``(J_r - J_M) ∘ B̃^{\rm dil}`` of the elastic `dilute_dual.jl` — before
  feeding the iso / TI / ortho / generic paths. The defect is independent of the
  time grid and of the phase laws, so the elastic limit was wrong too:
  `DiluteDual` was only ever cross-checked against `homogenize` on crack-only
  RVEs, exactly where the converting branch runs. **Unaffected:** the elastic
  pipeline, every other ALV scheme, and `DiluteDual` on crack-only RVEs.
- **Order-2 `DiluteDual` was a silent alias of `Dilute`.** The conductivity
  dispatch returned `dilute_alv_order2` verbatim, with a comment stating the
  two schemes coincide "in this lightweight implementation"; it now calls
  `dilute_dual_alv_order2` with the resistivity contributions
  ``R̃_r = -R̃_0 ∘ Ñ_r ∘ R̃_0``. Unlike the order-4 defect, the old numbers were
  *plausible* — a valid `Dilute` estimate under a `DiluteDual` label — so any
  conduction or diffusion comparison of the two ALV schemes that found them
  equal was reading this bug.

### New — the glassy limit as a test

- `test/Viscoelasticity/test_glassy_limit_alv.jl`. The trapezoidal block
  ``(1,1)`` is ``\mathbb R(t_0, t_0)``, the glassy modulus, and every Volterra
  operation — product, inverse, divide, layered recurrence, scheme fixed point —
  preserves it, hence ``[R̃^{\rm hom}]_{(1,1)} = \mathbb C^{\rm hom}_{\rm
  elastic}(\{\mathbb R_r(t_0,t_0)\})`` and, in engineering form,
  ``J^E_{\rm eff}(t_0,t_0) = 1/E^{\rm hom}(t_0)``. Checked against `homogenize`
  for the nine order-4 schemes, a spheroid under isotropic orientation average,
  penny cracks in a Maxwell matrix, seven order-2 schemes, and the ageing
  solidifying composite of [sanahuja2013](@cite) in both topologies. This
  complements the existing elastic-limit tests, which feed *constant* Heaviside
  laws and constrain every diagonal block; here the laws genuinely relax and
  genuinely age, so only the first block is constrained — and that is the block
  the two `DiluteDual` bugs survived in.

### Documentation and scripts

- **The elastic reference of the ageing-creep benchmark was drawn for the wrong
  microstructure.** `scripts/53_ageing_creep_solid.jl` plotted `:layers` creep
  curves against a `:whole_pores` reference (23 % apart at ``t_0 = 2/3``), so no
  curve started on it. The reference is now built from the same topology, by the
  elastic `homogenize` on the frozen microstructure — an independent check of
  the ALV result rather than a restatement of its first point — and is sampled
  at the loading ages as well as the setting times, the function being a
  staircase between them. Same fix in
  `docs/src/applications/ageing_creep.md`, which gains an *Elastic cross-check*
  section printing the two sides.

## v0.4.0

The route from the Fourier Green operator to the COD tensor ``𝐁``, written down
and executed symbolically — first for elasticity, then for transport, where it
turns out to be analytic at *full* anisotropy. Writing the transport derivation
down is what exposed a wrong prefactor in the thermal closed forms, which is why
this is a minor rather than a patch release.

### Breaking changes

- **The thermal COD scalar `b` was too small by ``4\pi/(3\eta)`` (elliptic) and
  ``\pi^{2}/4`` (ribbon).** Every thermal crack number changes: `cod_tensor`,
  `compliance_contribution`, `delta_resistivity` and the thermal `dif` on an
  `AbstractTens{2,3}` reference. The corrected closed forms are

  ```
  b_ellipse = 4 / (3 √λ₁ 𝓔_η′)      b_iso = 4/(3 k₀ 𝓔_η)      b_penny = 8/(3π k₀)
  b_ribbon  = π / (2 √det(K₀|₍m,n₎))                          b_ribbon_iso = π/(2 k₀)
  ```

  `b` is normalized exactly like the elastic ``𝐁`` — by the in-plane half-width,
  through ``b = \chi/(b\Lambda)`` with ``\chi^{\mathcal E} = 2/3``,
  ``\chi^{\mathcal R} = \pi/4`` — so the two branches now share one convention,
  which is what the elasticity ↔ conductivity table of the theory page always
  claimed. Three independent routes pin the new values: the ``\xi_n``-integral
  chain (`scripts/16_cod_symbolic_thermal.jl`), the flattening limit
  ``\lim_{\omega\to0}\omega\,\mathbf\Lambda^{-1}``, and the textbook temperature
  jump of an insulating penny crack, ``[\![T]\!](r) = \frac{4\sigma_n a}{\pi
  k_0}\sqrt{1-r^{2}/a^{2}}``, whose surface average gives ``8/(3\pi k_0)``.
  **Unaffected:** the whole elastic branch, and `fracture_permeability` /
  `ConductiveCrack`, whose conduction side does not go through these formulas.
- **`_cod_aniso_ellipse_thermal` no longer uses the ``\mathbf K_0^{-1/2}``
  transform.** The equivalent adjugate form needs one **2×2** eigenvalue problem
  instead of `eigen` on a 3×3 plus `svdvals`, so the anisotropic thermal COD is
  now type-generic: it flows through `ForwardDiff` and evaluates on symbolic
  scalars, where it previously threw. Results agree with the old route wherever
  the old route ran (up to the prefactor above).

### Why 37 green tests did not catch it

Worth recording, because the mechanism is reusable. Nearly every assertion in
`test/Cracks/test_thermal.jl` restated the closed form the code evaluates — a
tautology that no prefactor error can fail. The one independent check, a
Hill-tensor limit, compared its oracle against
`delta_resistivity(crack, R, 1) = (4π/3) R` instead of against `R`: a `b` too
small by `4π/(3η)` is *exactly* canceled by that spurious `4π/3`, so the test
passed on two compensating errors. The file is rewritten to anchor the oracle
where the definition puts it, ``R = \lim_{\omega\to0}\omega\mathbf\Lambda^{-1}``,
and to check the **convergence rate** rather than one tolerance: the truncation
is ``O(\omega)``, so a tenfold smaller ``\omega`` must cut the error tenfold,
which a constant prefactor offset cannot fake. It also adds a
rotate-the-whole-problem invariance test on the adjugate branch. 46 assertions,
up from 37.

### Differentiability and symbolic scalars — one wrong predicate, six sites

`T <: Real` was used throughout as a proxy for "a concrete number I may compare
against a tolerance". It is not one: **`Symbolics.Num <: Real`**, yet `<`, `≈`
and `iszero` on a `Num` return *symbolic* expressions that throw in a boolean
context — while `ForwardDiff.Dual` is also `<: Real` and compares perfectly
well, so no `<: Real` test can separate the two. The predicate is now explicit,
closed and defaults to the safe answer:
`Elliptic.is_hard_numeric`, replacing the wrong test in `_agm_converged`,
`Core._sort_axes_and_basis` (both methods), `Core._classify_shape_2d/3d`,
`Cracks._classify_crack`, `Cracks._is_unit_alignment` and
`Cracks._elliptic_CS`.

- **New weak extension `MeanFieldHomogenizationSymbolicsExt`.** Without it
  `ell_K`/`ell_E` on a `Num` fell through to the AGM recursion and unrolled ~60
  nested `sqrt` into the expression tree — correct, but unusable and
  unprintable. Registered with `@register_symbolic`, they stay an unexpanded
  call on a symbolic argument while any numeric argument still reaches the fast
  `Float64` method, so `Symbolics.build_function` round-trips the result exactly.
  This is the Symbolics counterpart of the existing SymPy extension.
- **`Cracks._ti_aligned` compared `|axis·n̂|` to 1.** On a `TensTI{4, Num}` whose
  axis *is* the crack normal, `Symbolics` leaves `abs(1.0)` unevaluated — even
  `tsimplify` returns `-1 + abs(1.0)` — so the test said "not aligned" and
  `cod_tensor` silently fell through to a **numerical** back-end instead of the
  closed form. It now compares the square, which uses only `*` and does fold.
- **The numerical Green back-ends now say why they cannot run on a boxed
  scalar.** `Core._A_and_Tn` builds an `MArray`, which cannot even be
  *constructed* for a non-`isbits` eltype; the failure used to surface as
  `setindex!() with non-isbitstype eltype …` from StaticArrays, several frames
  deep. It is now an `ArgumentError` naming the type and stating which
  references do have a closed form.

Verified, rather than asserted: `ForwardDiff` through the elastic closed forms
(∂/∂E, ∂/∂ν, ∂/∂η, ribbon, aligned TI) and the thermal ones (∂/∂k₀ against the
exact `-b/k₀`, ∂/∂η, ribbon, `R₃₃`, and a full six-component gradient through the
anisotropic adjugate branch); `Symbolics.Num` through elastic iso, penny, ribbon,
symbolic aspect ratio and aligned TI, and through thermal iso and fully
anisotropic. `Float64` and `Dual` results are bit-unchanged throughout. No
`TensND` change was needed — every frame of the original failure was in this
package, so the `TensND = "0.3.5"` bound stands.

### Bug fixes

- **`cod_tensor` threw in dispatch on a symbolic transversely isotropic
  reference.** `Cracks._ti_aligned` compared the stored symmetry axis with the
  crack normal through `isapprox`, which is undefined on `SymPy.Sym` — the only
  seam of the COD chain without a `T <: Real` guard, while
  `_classify_crack` and `Core._sort_axes_and_basis` both have one. The
  comparison is now structural for non-`Real` element types.
- **`cod_tensor(PennyCrack(one(Sym)), …)` returned `NaN`, silently.**
  `_elliptic_CS` took its removable `η = 1` shortcut only for `T <: Real`, so a
  symbolic penny evaluated `𝒞 = (ℰ − η²𝒦)/k²` as `0/0`. The guard is now
  `iszero(k²)` alone: `false` for a free symbol, which is the general branch and
  the answer we want, `true` for an exact `Sym(1)`. Only non-`Real` *geometry*
  was affected — every `Float64`, `ForwardDiff.Dual` and complex-moduli result
  is bit-for-bit unchanged.

### New — from the Green operator to `𝐁`

- **New theory section** in `theory/cod_tensors.md`: the reduced kernel
  ``\hat{𝐐}^{⋆}_{nn}``, why its ``ξ_n`` integral converges, the three
  reductions (equivariance, degree-1 homogeneity, parity) that make it
  tractable, the resulting ``a_1, a_2, a_3`` for an isotropic and for an
  aligned-TI matrix, and the three master integrals that produce
  ``𝒞_η, 𝒮_η, ℰ_η``. The radical ``σ_γ`` of the published TI closed form is
  **derived** — it is the sum ``γ_1+γ_2`` of the two in-plane Stroh roots — and
  the section closes on what breaks for a general anisotropy.
- **New `scripts/09_cod_symbolic_green.jl`** — the same derivation carried out
  by SymPy end to end, with the shipped closed forms as its oracles. Not added
  to the documentation gallery: SymPy-heavy scripts are re-executed on every
  docs build (repo policy, `docs/literate.jl`).
- **First symbolic tests of the crack chain**, `test/Cracks/test_cod_symbolic.jl`
  (31 tests): `cod_tensor` on `TensISO{4,3,Sym}` and on an aligned
  `TensTI{4,Sym}`, the dispatcher's aligned/non-aligned decision, and the
  ISO-vs-aligned-TI agreement at a free symbolic aspect ratio.

### New — the same derivation for transport (order 2)

- **New section** *From the Green operator to `b`* in `theory/thermal_cracks.md`,
  and **new `scripts/16_cod_symbolic_thermal.jl`**. The order-2 acoustic form
  ``\underline{\xi}\cdot\boldsymbol{K}_0\cdot\underline{\xi}`` is a *scalar*, so
  the numerator of the kernel collapses to a constant and the ``\xi_n`` integral
  closes **at full anisotropy** — no sextic, no residues, no cubature:
  ``\hat{Q}^{\star}_{nn} = \tfrac12\sqrt{(\underline{n}\wedge\underline{\xi}^{\star})
  \cdot\mathrm{adj}\boldsymbol{K}_0\cdot(\underline{n}\wedge\underline{\xi}^{\star})}``.
  The contour integral then reduces to an **effective ellipse**: an arbitrarily
  anisotropic conductor behaves as an isotropic one of conductivity
  ``\sqrt{\lambda_1}`` around a crack of effective aspect ratio
  ``\eta' = \sqrt{\lambda_2/\lambda_1}``, with ``\lambda_{1,2}`` the eigenvalues
  of a **2×2** in-plane restriction of ``\mathrm{adj}\boldsymbol{K}_0``. Being
  2×2, that step is closed form — and symbolically evaluable, where the shipped
  ``\boldsymbol{K}_0^{-1/2}`` route needs `eigen` and `svdvals` on 3×3.
- The section records a **prefactor disagreement** with the thermal closed forms
  it precedes (``4\pi/(3\eta)`` for the ellipse, ``\pi^{2}/4`` for the ribbon),
  supported by three independent checks: the ``\xi_n``-integral chain, the
  flattening limit ``\lim_{\omega\to0}\omega\boldsymbol{\Lambda}^{-1}``, and the
  textbook temperature jump of an insulating penny crack. Nothing is changed in
  the code: resolving it moves every thermal crack result, so it is left visible
  and measured by the script rather than patched in passing.

### Documentation fixes

- The isotropic prefactor was written `8η(1−ν²)/(3E)` in the docstring of
  `_cod_iso_ellipse` and in `theory/cod_tensors.md`, where the code and the
  reference both give `8(1−ν²)/(3E)`. The `η` was parasitic and invisible at the
  penny crack. Prose only — no computed value changes.
- `theory/cod_tensors.md` claimed that `method = :auto` selects `Residue` for
  anisotropic `Float64` input. It never does, and has not since the dispatch
  rework: `:auto` picks `DECUHR` when its extension is loaded and
  `NestedQuadGK` otherwise (`src/Core/dispatch.jl`).
- **Admonition titles rendered their markup literally.** Documenter emits an
  admonition title as *plain text* — no inline markdown, no math — so
  ``` ``\boldsymbol{B}`` ``` and `` `order = 0` `` appeared with their delimiters
  in the rendered header. All 13 affected titles (7 pages, 4 docstrings, 2
  Literate scripts) are now plain prose.
- Stale self-references in the banners of `scripts/10_`…`14_`, left over from a
  renumbering.

## v0.3.1

(v0.3.0 was tagged but never registered, so everything below is folded into the
first published release. No API is removed or renamed; the "breaking" entries
are results that change because they were wrong.)

MeanFieldHomogenization as a **constitutive law inside a finite-element code**,
the poroelastic and hydraulic machinery a fractured-rock model needs, and three
correctness fixes for cracks that are not aligned with the canonical axes.

### Breaking changes

- **`fracture_permeability` now returns a genuinely self-consistent estimate.**
  The broken loop was linear in the crack density and independent of the
  fracture conductivity; the fixed one percolates. On a connected network the
  two differ by orders of magnitude, so **every fractured-permeability number
  changes**, and with it every `FracturedPoroelasticRock` transport output.
- **Numbers change for crack families that are not parallel to one another, or
  not aligned with an anisotropic reference medium.** The COD frame fix below
  is a correctness fix, so any previously published `C_hom` / `S_hom` involving
  tilted cracks was wrong and now changes — self-consistent schemes included,
  since their running estimate is anisotropic by construction. Aligned families
  in an isotropic matrix are unaffected, bit for bit.
- **`SelfConsistent` iterates are now projected onto the major-symmetric
  tensors.** Coaxial configurations move by roundoff; non-coaxial crack
  families, which used to return `NaN`, now converge.
- **`TensND ≥ 0.3.5` is required** (was `0.3`). Versions up to 0.3.3 rotate
  `TensTI` / `TensOrtho` incorrectly in `change_tens` (fixed in 0.3.4), and
  versions up to 0.3.4 drop the basis in `TensISO` products, returning a bare
  `Array` where a `Tens` is expected (fixed in 0.3.5). Both are relied on by the
  fixes above, so the bound states the first version that carries them all.

### Bug fixes

- **`fracture_permeability` returned the dilute estimate, not the
  self-consistent one.** Its fixed-point loop defaulted to `abstol = 1e-10`,
  which for a tight rock (`k ~ 1e-18 m²`) is met by the *first* iterate: the
  loop exited after one step, so the reference medium was never updated. The
  symptom was silent and diagnostic — the result came out exactly linear in the
  crack density and completely independent of the fracture conductivity, hence
  with no percolation threshold at all. `abstol` now defaults to `0` and the
  test is purely relative. Randomly oriented flowing cracks now reproduce the
  closed form `K/k_s = 1/(1 − 32d/9)` to three digits, percolating at
  `d_c = 9/32`, where the broken loop gave `1 + 32d/9`.
- **Cracks tilted with respect to the reference medium gave wrong results.**
  The four anisotropic COD kernels rotated the stiffness into the crack basis
  but passed the crack frame in *global* coordinates. Aligned cracks coincide,
  which is why a suite built on them never caught it. Fixed by
  `Cracks._crack_local_frame`, which returns both in one frame; validated
  against the ECHOES reference across crack inclinations (agreement improved
  from up to 8 % error to ~1.5·10⁻⁷, ECHOES' own quadrature accuracy).
  **Any result involving non-parallel cracks in an anisotropic reference medium
  changes**, self-consistent schemes included, since their running estimate is
  anisotropic by construction.
- **`SelfConsistent` diverged on non-coaxial crack families.** Its body
  assembles `𝔹 : 𝔸⁻¹`, a product of non-commuting tensors, which is not
  major-symmetric when families are not coaxial; the asymmetry grew from
  roundoff to 2·10⁻⁴ over a few iterations and the crack cubature then returned
  a `NaN` integrand. Each iterate is now projected back onto the
  major-symmetric tensors, as the reference implementation does. Requires
  **TensND ≥ 0.3.5**, which fixes two basis bugs of its own.

### New — finite-element coupling (`MeanFieldHomogenization.Constitutive`)

- `AbstractMFHMaterial` / `material_response`, a multi-gradient, multi-flux
  Gauss-point contract with per-point internal state and a consistent tangent.
- `HomogenizedElastic` (linear, any cell and scheme) and
  `MicrocrackedMaterial` (crack families that open and close; piecewise linear,
  so the tangent is exact, with steps split at every closure and reopening).
- `MaterialCache`, keyed on the discrete open/closed set — a thick-cylinder run
  needs **4 scheme solves for 2304 quadrature points**.
- `to_tensors` / `from_tensors` / `plane_strain_response`, the `Tensors.jl`
  boundary, and `check_material_interface`.
- `MeanFieldHomogenizationFerriteMaterialExt`, activated by `import Ferrite`
  alone (no gmsh): `mfh_states`, `mfh_element!`, `mfh_poro_element!` — the
  coupled `(u, p)` element, backward Euler, whose Jacobian *is* the four tangent
  blocks the material declares — plus `annulus_grid` and `cylinder_sector_grid`.
- The **ARMA 2011 well test**: a three-dimensional, anisotropic, coupled
  reservoir simulation driven entirely by the homogenized material
  (`scripts/fe/arma2011_welltest.jl`, static figures in the docs).

### New — poromechanics and fractured permeability

- `MeanFieldHomogenization.Poromechanics`: `biot_tensor`,
  `inverse_biot_modulus`, `poroelastic_parameters`, `undrained_stiffness`,
  `skempton_tensor`, `terzaghi_stress`.
- `ConductiveCrack`, the *flowing* crack (the preferential flow path, as
  opposed to the insulating crack), and `fracture_permeability`, its
  self-consistent effective conductivity.
- `FracturedPoroelasticRock`, the saturated fractured rock of
  [barthelemyARMA2011]: two gradients `(E, p)`, two fluxes `(Σ, φ)`, the four
  Biot tangent blocks, fractures driven by the **Terzaghi** effective stress,
  and a permeability that follows the apertures through the cubic law. The
  coupled *u–p* reservoir simulations of that paper are **not** shipped; the
  material is, and is tested.
- `crack_family_compliances` / `crack_family_residual`: the per-family
  decomposition `S_hom = S_solid + Σ (4π/3) dᵢ 𝕊ᵢ`, which is what drives an
  aperture update.
- `set_amount!`, `set_geometry!`, `set_property!`: mutating `RVE` setters for
  Gauss-point loops (`set_param` remains the immutable, AD-friendly path).

## v0.2.0

Two N-body homogenization models, the machinery they share, and a bridge that
puts every existing scheme on the new cell.

### Breaking changes

- **The documented sign convention of the transport quantities is corrected.**
  The package now states `σ ≡ −q = K·∇T` as the stress analog: on a surface
  of normal `n`, `σ·n` is in *both* theories what the exterior transmits to the
  interior. **No returned value changed** — `flux_gradient_loc`, `flux_flux_loc`
  and `gradient_flux_loc` compute exactly what they always did — but their
  docstrings previously described that quantity as the flux `q` rather than as
  `σ = −q`. Code that followed the old docstrings was interpreting the sign
  backwards; the numbers were always the ones documented now. The same applies
  to the thermal `sif` / `dif`, whose driving vector is `σ∞ ≡ −q∞`.
- **Two names were removed from the export list**: `laminate_interface` and
  `layer_fraction`. Both were exported without ever being defined, so any use
  already raised `UndefVarError`; no working code is affected.

### Added

- **`ParticleAssembly`** — a cell holding individually *located* inclusions,
  with `PeriodicBox` and `MixedBC` far-field treatments, volume fractions
  derived from the geometry (`f_a = |Ω_a|/|Ω|`), particle families, and the
  `cubic_lattice` / `random_assembly` generators.
- **`ClusterModel`** — Molinari & El Mouden (1996): the mean strain of every
  inclusion, resolved pairwise inside a cluster of radius `R_c`. Degenerates
  *exactly* onto Mori-Tanaka at `cluster_radius = 0`.
- **`EquivalentInclusion`** — Brisard, Dormieux & Sab (2014) at order `p = 0`,
  a Galerkin discretization of the weak Lippmann-Schwinger equation, with
  `eim_bound_type` reporting when the estimate is a rigorous bound.
- **`interaction_tensor` / `self_interaction_tensor`** — the two-inclusion
  tensor `𝕋^{ab}` both schemes are built on, with three back-ends
  (`:analytical` closed forms exact at any separation, `:multipole`,
  `:quadrature`) plus `lattice_interaction_tensor` for periodic images.
- **Anisotropic Green operator** — `green_operator_aniso` (Barnett line
  integral for 3D elasticity, closed form for conduction), `green_function_aniso`
  and the `green_operator` dispatcher. This is what makes it possible to chain
  an N-body estimate into a further scale, since a cluster estimate on a cubic
  array is *cubic*, not isotropic.
- **`RVE(asm)`** — forgets the positions and keeps the derived fractions, so
  **every one-site scheme runs directly on an assembly**
  (`homogenize(asm, MoriTanaka(), :C)`). One phase per particle, which is exact
  because every scheme sums over phases linearly. `matrix_geometry` and
  `distribution_shape` are forwarded through `homogenize`.
- **Nano-interfaces** — `surface_stiffness` and `equivalent_particle`
  (Dormieux, Lemarchand & Brisard, 2016): a Gurtin-Murdoch interface condensed
  into an equivalent particle stiffness, after which the classical schemes
  apply unchanged.
- **Parameter lenses for assemblies** — `center_param` and `radius_param`,
  differentiable through both the N-body and the one-site paths.

### Changed

- The Green operator follows [Brisard, Bertin & Legoll (2023)](https://doi.org/10.1016/j.cma.2023.116389),
  Eq. (9): it maps a polarization onto **minus** the induced field, so its
  Fourier symbol is positive semi-definite and its self term is
  `𝕋^{aa} = +ℙ`, coherent with the Hill tensor. Molinari & El Mouden and
  Berveiller et al. use the opposite sign, which the documentation now names
  explicitly wherever a formula is transcribed from them. (This convention
  governs API introduced in this release only.)
- `green_operator_iso` and the other public Green entry points return a `Tens`
  like every other tensor-valued function of the package; the `SArray` kernels
  they wrap are internal.
- The Theory and Manual sections of the documentation are grouped, so that the
  standard theory precedes what is built on it.

## v0.1.0 — 2026-08-07

First public release of MeanFieldHomogenization.jl — a mean-field homogenization toolkit
for elasticity, conductivity and ageing linear viscoelasticity, ported from and
cross-validated against the C++ ECHOES code.

### Core

- **`RVE` container** — representative volume element (`add_matrix!` /
  `add_phase!`); volume fractions and crack densities stored at RVE level;
  solid inclusions (`VolumeFraction`) and flat cracks (`CrackDensity`).
- **Ten homogenization schemes** — Voigt, Reuss, Dilute, DiluteDual,
  MoriTanaka, Maxwell, PonteCastanedaWillis, SelfConsistent,
  AsymmetricSelfConsistent, DifferentialScheme — via a single
  `homogenize(rve, scheme, property)` entry point.
- **Elasticity (`:C`) and conductivity (`:K`)** — order-4 and order-2 tensor
  algebra, crack contributions, and Sevostianov-style interface stiffness
  (spring / Kapitza / membrane) in every pipeline.
- **Self-consistent solvers** — symmetric Hill/Budiansky SC with a built-in
  Newton-Raphson solver (quadratic convergence, backtracking line search) or
  Anderson/Picard, positive-definite guard near percolation; NonlinearSolve.jl
  algorithms available via weak extension.

### Ageing linear viscoelasticity (ALV)

- **`Viscoelasticity` sub-module** — time-domain ALV homogenization after
  Sanahuja (2013) and Barthélémy et al. (2016, 2019): `ViscoLaw` relaxation /
  creep kernels, Sanahuja trapezoidal discretization, block-Volterra inverse,
  discrete ALV Hill kernel.
- **`homogenize_alv(rve, scheme, prop; times)`** for all schemes, order-4 and
  order-2; iso and TI Walpole-basis fast paths; BLAS/LAPACK Volterra fast path.
- **Differential scheme as an adaptive SciML ODE** on the fictitious
  incorporation time `τ ∈ [0,1]` (`Tsit5` default), with functional
  `Path` / `Sequential` / `Proportional` incorporation trajectories.
- **N-layer composite spheres** — full bulk + shear Hervé–Zaoui recurrence in
  elastic and ALV form, with perfect / spring / membrane interfaces.

### Differentiability

- **ForwardDiff throughout** — `derivative`, `gradient`, `jacobian`,
  `sensitivity` of any homogenization result w.r.t. physical, geometric,
  volume-fraction or crack-density parameters (lens API); multi-scale chain
  rule by closure composition.
- RVE-level orientation projection (`symmetrize`: iso / TI Reynolds averaging).

### Added

- **Periodic multilayer homogenization**, `MeanFieldHomogenization.Laminates` — a second
  kind of microstructure, next to the random morphologies the Eshelby-based
  schemes describe:
  - `Laminate`, a periodic unit cell of parallel layers with **no matrix, no
    auxiliary Eshelby problem and no reference medium**, solved *exactly* by
    the new `Laminated` scheme. `Voigt` and `Reuss` apply to it too, and two
    of the bracketings are equalities: a laminate saturates Voigt in the plane
    of the layers and Reuss across them, simultaneously and for arbitrary
    anisotropy.
  - Elasticity **and** transport from one implementation, dispatched on the
    order of the stored property exactly as the mean-field schemes are.
  - The four imperfect-interface models of `LayeredSpheres` reused unchanged —
    a planar interface being the curvature-free case: `SpringInterface` /
    `MembraneInterface` in elasticity, `KapitzaInterface` /
    `SurfaceConductiveInterface` in transport. They act on *complementary*
    halves of the answer (primal out of plane, dual in plane — a planar
    membrane produces no traction jump at all), and enter with the weight
    `1/L`, an interface **density**: hence a genuine size effect, and why a
    laminate stores thicknesses rather than only fractions.
  - **Anisotropic interfaces**, which a plane admits and a sphere does not:
    `AnisotropicSpringInterface` (any symmetric compliance tensor),
    `AnisotropicMembraneInterface` (any 2-D surface stiffness, six
    coefficients) and `AnisotropicSurfaceConductiveInterface`. The spherical
    recurrence needs its jump conditions to share the symmetry of the
    geometry; a plane has a normal and an arbitrary in-plane texture, so it
    does not. Both exact oracles hold unchanged with a full tensor.
    (`KapitzaInterface` needs no counterpart: `[T] = ρ qₙ` relates two
    scalars.)
  - A hand-written symbolic tutorial,
    `docs/src/tutorials/symbolic_laminate.md`, deriving the closed forms from
    the code and displaying the effective Kelvin-Mandel matrix in terms of
    five layer averages — two of which enter *inverted* (harmonic, series,
    out of plane) and two directly (arithmetic, parallel, in plane).
  - Per-layer localization (`layer_strain_localization`, …), the two Hill
    tensors (`laminate_hill`) and the interface displacement jumps
    (`interface_jump`); lenses `ThicknessParameter` and `InterfaceParameter`.
  - Ageing-viscoelastic twin (`laminate_alv`,
    `homogenize_alv(lam, Laminated(), …)`): the *same* kernel with the `3×3`
    cofactor inversion swapped for `volterra_inverse` on the out-of-plane
    restriction. The elastic limit returns the elastic laminate in every
    diagonal time block, and both exact saturations survive the transposition.
  - Symbolically evaluable end to end: `scripts/38_laminate_symbolic.jl`
    derives the Backus (1962) closed forms *from the code*. This is what pins
    the two implementation choices that make it possible — the pseudo-inverse
    is a cofactor inverse of the out-of-plane block, never an SVD-based
    `pinv`, and every intermediate is an `SMatrix`, never an `MMatrix`.
  - Documented theory (`docs/src/theory/laminate.md`), which **corrects** the
    reference note it follows: its appendix introduces a *pair* of
    "in-plane / out-of-plane pseudo-inverses", which is vacuous for the tensor
    actually involved — `⟨ℙ⟩` has an identically zero in-plane block, so its
    in-plane pseudo-inverse does not exist. One ordinary Moore-Penrose
    pseudo-inverse is all that is needed, and the unfinished general case is
    never required.

- **The homogenization-cell abstraction and declarative multiscale chaining**:
  - `AbstractHomogenizationCell` (in `Core/cells.jl`) is now the supertype of
    everything `homogenize` accepts — `RVE` and `Laminate` today. `homogenize`,
    `get_param`/`set_param` and `derivative`/`gradient`/`jacobian` are typed on
    it; every scheme kernel stays on `RVE`, so existing behavior is unchanged.
  - `Homogenized(cell, scheme)` may be stored **as a property value**: a whole
    multiscale model becomes one object, resolved lazily. With no explicit
    `property` it *inherits the key it is stored under*, so one nested cell
    answers `:C` and `:K` alike — a microstructure is described once, not once
    per physics.
  - `NestedParameter` / `nested(...)` addresses a scalar inside a nested cell,
    so `derivative`/`gradient`/`jacobian` cross every scale in a single
    `ForwardDiff` pass, replacing hand-rolled closures and explicit
    `ForwardDiff.Tag` plumbing.
  - Each `(cell, key)` pair is evaluated exactly once per `homogenize` call —
    including across the ~100 iterations of a self-consistent solve — through a
    task-local, call-scoped cache that never leaks a value between two autodiff
    evaluations.
  - Both styles are documented and compared in
    `docs/src/manual/multiscale.md`, with a worked side-by-side in
    `scripts/36_laminate_multiscale.jl`; `applications/strength.md` and
    `applications/cement_paste_diffusion.md` now show the declarative variant
    next to their production (explicit) model, and say when each is preferable.

- **Inclusions whose response is a trained neural network**,
  `MeanFieldHomogenization.NeuralInclusions` — a fourth route into the custom-inclusion
  contract, after the analytic families, the layered patterns and the
  finite-element solves:
  - `NeuralHillInclusion` (gate A, the Hill tensor) and
    `NeuralLocalizationInclusion` (gate B, both localization tensors, for a
    heterogeneous morphology that has none). Gate A keeps the contrast
    dependence and the `ℂ₁ = ℂ₀ ⟹ 𝔸 = 𝕀` limit **exact**, so the fit error is
    confined to a single tensor.
  - What a surrogate buys that the other routes cannot: it is **differentiable
    in the morphology**. `derivative(rve, scheme, geometry(:phase, :field))`
    reaches an aspect ratio, where the finite-element inclusions refuse the
    request outright because their solve would silently return zero. Smooth
    activations only (`tanh`, `softplus`, never a piecewise-linear one), so
    second derivatives exist too.
  - Enforced rather than fitted: the symmetry class and the major symmetry
    (a structured TensND type from the right number of components), homogeneity
    in the reference moduli (`ℙ(λℂ₀) = ℙ(ℂ₀)/λ`), and frame indifference.
    `AffineHill` goes further and removes `ν₀` from the inputs altogether, using
    the exact affine structure `ℙ = d·𝕌ᴬ + 𝕍ᴬ/μ₀` of the isotropic-matrix Hill
    tensor — one input fewer and an order of magnitude more accurate than
    `DimensionlessHill` at equal network size.
  - The learning system: `SampleBox` + Halton low-discrepancy sampling (no RNG
    state, so a dataset is reproducible), `generate_dataset` with a
    two-callback teacher (`geometry`, `response` — the only morphology-dependent
    part), `fit_scaling`, `train_surrogate`, `validate_surrogate`,
    `report_surrogate`.
  - `Provenance` traveling with every model — teacher, sample counts, held-out
    error — and a **domain guard** that refuses (or warns on) an evaluation
    outside the box the model was trained on. Test tolerances are derived from
    the recorded error rather than hard-coded, so a retraining cannot silently
    loosen a threshold.
  - Four models committed under `src/NeuralInclusions/models/` as reviewable
    JSON, validated against the analytic ellipsoid; `scripts/nn/train_models.jl`
    regenerates them and `scripts/84_neural_inclusion_ellipsoid.jl` is the
    published tutorial. Nothing is trained at test or documentation-build time.
- **`MeanFieldHomogenizationLuxExt`** (`Lux` + `Optimisers` + `Zygote`, weak dependencies) —
  the optimizer, and the only part of the surrogate pipeline that needs one. It
  writes its result back into a dependency-free `MLP`, so evaluating a trained
  model needs nothing beyond the package.

- **Inclusions solved by finite elements**, `MeanFieldHomogenization.FiniteElements`, both
  using the finite Eshelby cell with the first-order corrected boundary
  condition of
  [Adessina et al. 2017](https://doi.org/10.1016/j.ijengsci.2017.03.015):
  - `FEEllipticCrack` — flat elliptical crack in 3-D tetrahedra, whose COD
    tensor comes out of the solve. Declaring `shape_trait` and supplying
    `cod_tensor` is enough for ℍ, ℕ, 𝐑, 𝐍_K, the bundled pair and the four
    `delta_*` to be inherited, so it drops into every scheme, `symmetrize`
    included.
  - `FEExcenteredSphere` — sphere with an off-center spherical core, solved by
    **axisymmetric Fourier elements** (modes 0/1/2 in elasticity, 0/1 in
    transport). Three assemblies and eight solves in two dimensions where the
    three-dimensional formulation needs six solves on a mesh two orders of
    magnitude larger. Heterogeneous, so it enters through gate B with both
    localization tensors measured in one pass; it also supplies exact `Voigt`
    and `Reuss` bounds, since the geometry fixes its internal volume fractions.
  - Memoization on the reference medium in the inclusion's local frame:
    under `IsoSymmetrize` / `TISymmetrize` a whole family of orientations
    shares a single solve. `fe_assembly_count` / `fe_reset!` to inspect it.
  - Diagnostics `fe_mesh_report`, `fe_cod_breakdown`, `fe_axi_mesh_report`,
    `fe_axi_breakdown` — the last returning the **uncorrected** tensors beside
    the corrected ones, which is how the correction is shown to work.
- **A two-backend contract for the finite-element solves.** The physics —
  Fourier operators, boundary data, the polarization fixed point, the
  memoization — lives in `src/FiniteElements/`; a backend supplies only the
  discretization, through nine generics (`FEBackend`, `backends.jl`).
  - `FerriteBackend` (`MeanFieldHomogenizationFerriteExt`, needs `Ferrite`, `FerriteGmsh`,
    `Gmsh`) serves both morphologies.
  - `GridapBackend` (`MeanFieldHomogenizationGridapExt`, needs `Gridap`, `GridapGmsh`)
    serves both as well, stating the weak form directly rather than looping
    over elements: `∫( ε(v) ⊙ (σ∘ε(u)) )dΩ` for the crack,
    `∫( Eᵐ(v) ⋅ (D ⋅ Eᵐ(u)) * ρ )dΩ` for the axisymmetric modes.
  - The crack front weld — the merge of the node pairs gmsh's `Crack` plugin
    duplicates in spite of `OpenBoundaryPhysicalGroup` — moved from
    `Ferrite.Grid` surgery to a renumbering of the written `.msh`, so both
    backends read the same welded mesh.
  - `AutoBackend`, the default, resolves at the **first solve** and is then
    pinned: an inclusion can be built and stored in an `RVE` with no
    finite-element package loaded.
  - The two backends build the same discrete system and agree to round-off —
    ≈ 10⁻¹⁴ on every tensor, both physics, both morphologies — which
    `test_gridap_backend.jl` asserts at 10⁻⁸.
- **Isotropic Kelvin dipole field** in `Core`: `green_gradient_iso`,
  `dipole_displacement_iso` — the far field a polarized inclusion radiates,
  and the ingredient of the corrected boundary condition.
- **Compliance formulation of the differential scheme.**
  `DifferentialScheme(; formulation = :compliance)` integrates the dual ODE
  `dS/dτ = Σ φ̇ᵢ ℍᵢ(S)` and inverts the result, instead of
  `dC/dτ = Σ φ̇ᵢ 𝐍ᵢ(C)`. The two are analytically equivalent
  (`ℍ = −𝕊 : 𝐍 : 𝕊`) and agree to solver accuracy; they differ in which
  variable carries the error control, the compliance form being the better
  conditioned one for a medium softening towards percolation. Available in
  elasticity, conduction and ALV (both tensor orders).
- **`differential_path(rve, scheme, property)`** returns `(τ, states)` along
  the incorporation path instead of the `τ = 1` value alone. `scripts/24`
  no longer has to reach into the scheme's internals to plot `C^hom(τ)`.
- **Pair constructors for the trajectories**: `Path(:A => τ -> …, :B => …)`,
  `CustomPath(:A => values, …)`, `Sequential(:A, :B)` — the form the
  documentation already advertised.
- **Differential scheme in order-2 ALV** (viscous conduction / diffusion):
  `differential_alv_order2`, reached through
  `homogenize_alv(rve, DifferentialScheme(), :K; times)`, which previously
  raised a `MethodError`.
- **`LayeredSphere` phases in the ALV differential scheme**, through
  block-matrix (`_at`) variants of the layered-sphere ALV kernels. Mori-Tanaka
  and the self-consistent schemes already supported them.
- Unrecognized `DifferentialScheme` keywords are now forwarded to
  `OrdinaryDiffEq.solve` (`maxiters`, `dtmax`, `callback`, …) instead of being
  silently dropped.
- **New tutorial: [Comparing loading-path trajectories](docs/src/tutorials/differential_loading_paths.md)**,
  a τ-resolved companion to the existing path-dependence tutorial — four
  trajectories to the same target fractions, plotted through
  `differential_path` rather than compared at `τ = 1` only.
- **New tutorial: [Ageing viscoelastic schemes side by side](scripts/62_alv_schemes.jl)**
  (`scripts/62_alv_schemes.jl`, published as `tutorials/generated/alv_schemes.md`).
  Dilute, Mori-Tanaka, Maxwell and PCW on one ageing creep test, after
  [barthelemyIJES2019]: the collapse `MT = Maxwell = PCW` when the distribution
  shape equals the inclusion shape, the aspect-ratio sweep at fixed volume
  fraction, and the volume fraction beyond which the Maxwell/PCW estimate
  leaves the admissible domain of Ponte Castañeda & Willis.
- **`02_hill_elasticity.jl` promoted to a tutorial**
  (`tutorials/generated/hill_tensors.md`): no tutorial called
  [`hill_tensor`](@ref) as its subject, though it is the object every scheme is
  built on. Four geometries against their closed forms, two independent
  algorithms on an anisotropic matrix, the Eshelby tensor against
  [eshelby1957], and the step from `P` to a dilute estimate — checked against
  both the analytical dilute formula and `homogenize(rve, Dilute(), :C)`.
- **`59_alv_sensitivities.jl` promoted to a tutorial**
  (`tutorials/generated/alv_sensitivities.md`): `ForwardDiff` through the
  Volterra assembly, the two patterns that cover every case — the `set_param`
  lens for a parameter carried by the RVE, closure capture for one living
  inside the `ViscoLaw` — a joint gradient combining both, and a
  relaxation-time derivative that has no elastic counterpart. Every value
  validated against a central finite difference.

- **`RVE{T}(:M)` / `RVE{T,S}(:M)` constructors**, strictly equivalent to the
  `RVE(:M; T = …)` keyword form, which is kept.
- **`promote_rve(rve, T)` and `convert(RVE{T}, rve)`** to force an
  element-type floor on an already-built RVE.

- **Bundled localization helpers** — `Core.loc_and_stiffness` /
  `Core.loc_and_stress_average` (plus
  `Cracks.compliance_and_stiffness_contribution` and the `Schemes`-level
  `_phase_*_and_*` wrappers) share the single expensive `hill_tensor` /
  `cod_tensor` / layered-recurrence solve between the two quantities that
  Mori-Tanaka and the self-consistent kernels always request together. They
  used to be computed independently, i.e. twice with identical arguments.
  Results are **bitwise identical**; measured effect: −50 % time and
  allocations on an anisotropic matrix or a crack phase, −18.6 % on a 20-bin
  orientation family (40 → 20 Hill solves).

- **Benchmark suite** (`scripts/bench/bench_suite.jl` + `harness.jl`) with a
  committed baseline, three independent measurement channels (time,
  allocations, work counters), a calibrated noise floor and a bitwise
  checksum gate. See `scripts/bench/README.md` and `scripts/bench/DIAGNOSTIC.md`.

### Changed

- **Documentation reorganized along the Theory / Manual / Tutorials /
  Applications boundary**, with a stable anchor (`@id`) on every page. Two
  pages change section, so their URLs move: `applications/transport.md` →
  `tutorials/transport.md` (it demonstrates the API rather than reproducing a
  published study, unlike the seven pages that remain in Applications) and
  `tutorials/from_echoes.md` → `manual/from_echoes.md` (an API correspondence
  table, not a guided walk-through). The finite Eshelby cell, previously
  derived twice, is now stated once in
  [theory/corrected_cell.md](docs/src/theory/corrected_cell.md) and cited from
  the manual, the tutorial and the application.

- **`method = :auto` no longer selects the residue algorithm for a 3D
  anisotropic elastic reference.** It now picks a cubature — `DECUHR` when its
  extension is loaded, the type-generic `NestedQuadGK` otherwise — and
  `Residue` is reachable on an explicit `method = :residues` only, mirroring
  the `NUMINT` default of ECHOES. The reason is robustness, not taste: the
  residue acoustic polynomial degenerates whenever the reference is
  anisotropic in *type* and isotropic in *value*, and returns `NaN` /
  `DomainError` there. That reference is easy to reach — it is what the
  differential and self-consistent schemes feed back at their first step, and
  `Dilute` / `MoriTanaka` hit it too — so several RVEs that used to fail now
  work: an aligned triaxial ellipsoid or a solid phase combined with an
  aligned crack family in the differential scheme, and any scheme given an
  anisotropically-typed isotropic-valued matrix.

  Consequences to be aware of:

  - anisotropic `:auto` paths are slower (~11 ms with `DECUHR`, ~31 ms with
    `NestedQuadGK`, against ~4 ms for the residues) and their results move.
    Measured against the residue values on the benchmark suite: 7.4e-12 for a
    crack in a triclinic matrix, 1.25e-10 for Mori-Tanaka with an anisotropic
    matrix, and 1.34e-7 for a multi-axis-TI self-consistent estimate — a
    fixed-point scheme amplifies the per-evaluation cubature error (~1e-9 for
    `DECUHR`) up to its own convergence tolerance, so that is the floor on a
    converged SC value now. Pass `method = :residues` explicitly to recover
    both the previous speed and the ~1e-14 accuracy where the reference is
    known to be non-degenerate, or `:nestedquadgk` for ~1e-14 robustly;
  - loading `DECUHR` changes which cubature `:auto` picks, hence the accuracy
    the effective property is computed to. Pass `method` explicitly wherever
    that must not depend on the session;
  - non-`Float64` coefficients (`ForwardDiff.Dual`, `Complex`, symbolic) are
    unaffected: they already routed to `NestedQuadGK`.

- **A `RVE`'s element type is now a promotion floor, not a cast.** The
  `amounts` dict became heterogeneous (`Dict{Symbol,AbstractAmount}`) and
  `add_phase!` stores each amount under `promote_type(T, typeof(value))`
  instead of `convert(T, value)`. Every consequence is a widening: an amount
  narrower than the floor (`Int`, `Float32`, `Rational` under the default
  `T = Float64`) is stored exactly as before, and the only pre-existing calls
  whose stored type changes are those that used to be silently *narrowed*
  (a `BigFloat` fraction in a `Float64` RVE now stays `BigFloat`).

  - complex, `ForwardDiff.Dual` and symbolic amounts are accepted by a plain
    `RVE(:M)`, so `T = ComplexF64` / `T = typeof(f)` / `T = Sym` declarations
    are no longer needed anywhere (they remain valid, and still widen narrower
    amounts);
  - phases may carry amounts of *different* element types — a `Dual` fraction
    next to a `Float64` one, a `Dual` crack density next to a real fraction —
    promoted only where the values meet;
  - `eltype(rve)` now reports the *effective* element type (floor promoted with
    the stored amounts); `eltype(typeof(rve))` reports the declared floor;
  - `matrix_volume_fraction` takes its unit from the accumulator, so a symbolic
    RVE yields `1 - f` rather than `1.0 - f`;
  - `matrix_volume_fraction` is now a **cached field read** (`RVE.f_matrix`),
    refreshed by `add_phase!` with the same loop and the same dict iteration
    order, hence bit-identical. Do not write `rve.amounts` directly; that
    leaves the cache stale.

  **Performance.** Paired against the previous commit on the 67-case
  `scripts/bench` suite: all checksums bit-identical, allocations down on 21
  cases (−19 152 B), median |Δt| 1.1 %, every sub-microsecond scheme at or
  below the previous commit. Heterogeneous amounts cost a type refinement
  (`a isa VolumeFraction` no longer narrows to a concrete
  `VolumeFraction{Float64}`), paid back by the `f_matrix` cache
  (61 ns / 48 B → 2.4 ns / 0 B) and by `scale_by_amount(a, X)`, which puts the
  per-phase product behind a barrier dispatching on the amount's concrete
  type. The crack `delta_*` paths keep the plain `amount_value(a)`: the same
  barrier needs varargs there and measured +1.4 KB per call for no gain.

  Complex moduli never required a declaration in the first place — the moduli
  live in an untyped `Phase.properties` dict and were always promoted at the
  arithmetic. The tutorials that claimed otherwise, and the
  `fraction = ComplexF64(f)` idiom in `scripts/51`, `scripts/61` and the
  bituminous application, have been corrected.

### Fixed

- `set_param(rve, PropertyParameter(...), x)` narrowed the rebuilt property
  dict to `Dict{Symbol,TensND.AbstractTens}` although `Phase.properties` is
  `Dict{Symbol,Any}`. It therefore **threw** on any RVE carrying a non-tensor
  property — every ageing-viscoelastic RVE (`ViscoLaw`) included. Undetected
  because the ALV sensitivity tests only exercised `AmountParameter`.

- **The order-2 (transport) Hill tensor lost the orientation of a rotated
  inclusion.** `_hill_order2_3d_iso` read the Newton-potential components in the
  inclusion's own basis and wrapped them into a *canonical* `Tens`, so
  `hill_tensor(ell, K₀)` returned the tensor of the **unrotated** inclusion —
  exactly, hence silently. Both the ellipsoid and the cylinder kernels were
  affected, in the isotropic-matrix branch only; the anisotropic branch always
  built the shape tensor with the rotation in it, and elasticity was never
  affected. Consequence: any transport homogenization with *oriented*
  non-spherical inclusions was wrong off-axis, and an orientation distribution
  collapsed to a single orientation. The existing `s = ℙ·K₀` consistency test
  could not catch it — both sides dropped the rotation identically — so the new
  guard in `test/Conductivity/test_hill_order2.jl` tests rotation
  **equivariance** and cross-checks the isotropic kernel against the independent
  anisotropic derivation.

- **`Maxwell()` ignored the RVE's distribution shape on the ageing-viscoelastic
  path.** `_homogenize_alv_dispatch(::Maxwell, …)` built its Hill kernel on a
  hard-coded `Spheroid(1.0)`, while the elastic `Schemes.maxwell` reads
  `rve.distribution_shape` and the ALV `PonteCastanedaWillis` — algebraically
  the same formula — read it too. The same scheme on the same RVE therefore
  answered differently depending on which path it was called through, and the
  ALV answer silently ignored a modeling choice the user had made. Unnoticed
  because every test used the spherical default. On a 30 % / 10:1 oblate
  composite the error on `C₁₁₁₁` was 7.7 %; after the fix the ALV and elastic
  results agree to the last bit in the non-ageing limit.

- **A heterogeneous inclusion no longer contributes zero.** The generic
  contribution tensors (`stiffness_contribution` and its three siblings) went
  through `(C₁ − C₀) : 𝔸`, which is meaningless when the inclusion has no
  single `C₁` — and evaluated to zero for `FEExcenteredSphere`. They now switch
  to the exact identities `ℕ = 𝔸_σε − ℂ₀ : 𝔸_εε` and
  `ℍ = (𝔸_εε − 𝕊₀ : 𝔸_σε) : 𝕊₀` when `is_homogeneous_inclusion` is false, which
  makes gate B a complete entry point for a heterogeneous morphology.
- **`AsymmetricSelfConsistent` no longer depends on the bounds.** Nothing in
  the asymmetric algorithm needs one; only the heuristic that chooses once
  between the stiffness and the compliance form evaluated `Voigt`. It now falls
  back to the dilute estimate when the phases expose no layer-wise average, so
  the scheme is available wherever `SelfConsistent` is. New predicate
  `Schemes.has_layer_average` in place of a `try`/`catch`.
- **The isotropy guard of the axisymmetric solver was too tight.** At `rtol =
  1e-8` it refused `SelfConsistent` and `AsymmetricSelfConsistent` on a
  perfectly legitimate isotropic problem: the tensors the solver *returns* are
  isotropic only to the discretization error, a few parts in a million, so an
  iterative scheme can never feed back a reference isotropic to machine
  precision. Relaxed to `1e-4`; genuine anisotropy is orders of magnitude
  larger.
- **The boundary entities of the meshes are now fully declared.** A gmsh
  physical group has a dimension, so a group of curves does not carry its own
  bounding points and a group of surfaces does not carry its edges. A backend
  that reads boundary conditions off entity labels rather than off mesh facets
  therefore left dofs free: three on the outer sphere and six on the axis of
  the axisymmetric cell, eleven on the sphere of the crack cell. Cost on the
  axisymmetric cell: one order of convergence, under the appearance of
  discretization error.
- **Analytic sensitivity through a finite-element geometry is refused instead
  of returning zero.** `_replace_geom_field` copies non-numeric fields by
  reference, so the perturbed inclusion shared the original's `FECache` — whose
  key is the reference medium alone — and served back the unperturbed tensors.
  The derivative came out as exactly zero, with no warning. Use a finite
  difference over freshly constructed inclusions.

- **`method = :nestedquadgk` no longer raises a method ambiguity on an
  isotropic reference.** The `TensISO` disambiguation rules existed for
  `:auto`, `:residues` and `:decuhr` but had never been added for
  `:nestedquadgk`, so the one always-available cubature could not be requested
  explicitly for an isotropic matrix — for an ellipsoid or for a crack.
- **The crack dispatch no longer decides the anisotropic default on its own.**
  `Cracks._ti_crack_dispatch` ended on a hard-coded `Residue()` for a crack in
  a non-aligned TI matrix, so that path kept the residue algorithm as its
  `:auto` regardless of the shared rule — exactly the duplication
  `Core/dispatch.jl` centralizes to avoid. It now defers to
  `_aniso_default_algo`, as the ellipsoid TI refinement already did.
- **Crack densities are now diluted by the solid increments in the
  differential scheme.** Replacing `dφ` of the current medium by solid
  material also destroys the cracks that piece contained, so the volume
  balance extends to `dε_c = dφ_c^ε − ε_c Σ_{j solid} dφ_j`, inverted by the
  same Sherman-Morrison factor: `dφ_c^ε = dε_c + (ε_c/f₀) Σ_{j solid} df_j`.
  The missing term made `ε_c(τ)` an incorporation schedule rather than the
  density actually reached. Unchanged for crack-only RVEs and for
  trajectories where no solid grows while the cracks do.
- **An aligned non-spherical phase no longer crashes the differential
  scheme.** The ODE state was sized from the matrix's symmetry class alone,
  on the assumption that only cracks could leak anisotropy into the running
  estimate. An aligned spheroid does too — its dilute concentration tensor is
  transversely isotropic even between two isotropic materials — so
  `homogenize(rve, DifferentialScheme(), :C)` died on a `DimensionMismatch`
  as soon as a phase was not spherical. The state is now sized by probing the
  phase contributions, and lands in the smallest class that holds the running
  estimate (TI for aligned spheroids and cracks, rather than the full Mandel
  fallback — smaller state, and the Hill backends keep their closed-form
  paths). Isotropic RVEs keep the exact same 2-component state and results.
- **The compliance contribution of a heterogeneous inclusion no longer uses
  its declared (meaningless) phase property.** `_phase_compliance_contribution`
  applied `inv(C₁)` to a placeholder that every stiffness-side kernel ignores,
  so `DiluteDual` and the compliance-side ASC silently depended on what the
  user happened to declare for a `LayeredSphere` / `LayeredSpheroid`. It now
  goes through `ℍ = −𝕊₀ : 𝐍 : 𝕊₀`.
- **The ALV differential scheme no longer returns wrong values for an
  anisotropic running medium.** The ALV Hill kernel exists for an isotropic
  reference only, but the differential scheme evaluates it against its
  *running* medium, which an aligned non-spherical inclusion — or any crack
  without isotropic orientation average — takes out of the iso class;
  `iso_params_from_blocks` read `(α, β)` off it regardless. Such RVEs now
  raise an `ArgumentError` naming the phase and the two ways out. Cracks with
  `symmetrize = :iso` now work, where they used to fail with a cryptic "only
  iso reference is supported".
- `homogenize_alv(rve, DifferentialScheme(), prop; times)` honors `prop`
  instead of always homogenizing `:C`.

- **The n-layer sphere's shear localization `β_k` is validated against ECHOES.**
  `benchmark_nlayers.jl` had dropped the comparison, citing a 1–50 % gap blamed
  on an `echoes.layer_eE` indexing convention. The gap was ours: `β_k` was the
  bare mode-1 amplitude `a_k`, missing `b_k·F_k`, whose omission cancels in the
  degenerate configurations the fallback check used. The term was added later
  by an unrelated fix and the comparison never re-run. Restored alongside
  `α_k`: **30/30 configurations (2 to 8 layers), 4.4e-14**. The header comment
  in `shear_recurrence.jl`, which still described `β_k = a_k`, now matches the
  implementation.

- **`test_complex_moduli.jl` no longer hides scheme failures.** Every scheme
  was wrapped in `try/catch … @test_broken false`, so a scheme that stopped
  working in the complex plane vanished silently from the report. The
  assertions are now explicit and per scheme. Ground truth established by the
  de-masking: all ten schemes work with complex moduli; the single gap is
  `SelfConsistent(algorithm = NewtonDefault())`, whose ForwardDiff Jacobian
  cannot carry a `Dual` over a complex scalar (the default Anderson solver is
  the complex-capable path). This is now pinned by a `@test_throws` and
  documented in the manual.

- **Per-phase helpers now all evaluate a phase in the same reference medium.**
  `_phase_dilute_concentration` (both orders) and
  `_phase_stiffness_contribution` (4th order) evaluated their phase in
  `_project_matrix(P₀, sym)`, whereas `_phase_stiffness_contribution` (2nd
  order) and `_phase_compliance_contribution` (both orders) used the **raw**
  `P₀`. As soon as a phase carried `symmetrize ≠ NoSymmetrize`, the
  concentration tensor `A` and the contribution tensor `N` of one and the same
  phase were therefore computed in two *different* reference media, and the
  Mori-Tanaka denominator `⟨A⟩` mixed the two.

  The discrepancy is invisible on an isotropic matrix — `isotropify` is then a
  no-op to ~1e-16 — which is why every existing `symmetrize` test missed it:
  all of them use isotropic matrices *and* isotropic phase properties. It
  becomes a genuine (non-negligible) difference for an **anisotropic** matrix.
  All published isotropic-matrix results are unchanged to machine precision;
  the `echoes` cross-checks (`benchmark_pichler.jl`, `benchmark_nlayers.jl`,
  `benchmark_hill_derivative.jl`) are unaffected.

  Covered by `test/Schemes/test_loc_bundles.jl`, "every phase helper uses the
  same reference medium", which pins the corrected values on a triclinic
  matrix.

- **`SelfConsistent` with transversely-isotropic phases no longer errors.** The
  exact azimuthal average returns a `TensTI{4,T,8}` — the commutant of SO(2)
  about the axis is 8-dimensional, not 6 — while the analytical TI-coaxial Hill
  builder only had methods for the 5- and 6-parameter forms, so the running
  estimate raised `MethodError: no method matching _hill_3d_ti_coaxial(…,
  ::TensTI{4,Float64,8})`. A stiffness is major-symmetric, hence its average
  has `ℓ₇ = ℓ₈ = 0` exactly and narrows losslessly (`_ti8_to_ti6`).

- **`NewtonDefault` could not solve a fixed point richer than its starting
  guess.** The Newton parameter space was taken from `x0` (often a `TensISO`
  phase property, 2 components) while one application of the scheme can land in
  a larger symmetry class (a `TensTI{4,T,8}`, 8 components), so the residual
  subtracted vectors of different lengths (`DimensionMismatch`).
  `AndersonDefault` was unaffected, since Picard simply propagates whatever the
  step returns. The parametrization is now seeded from `step(x0)`.

### Performance

All figures below are measured against the committed baseline
(`scripts/bench/baseline.json`, 67 cases, noise floor 1.5 %); 64 of the 67
cases stay **bitwise identical**, the other three move by at most 6.9e-16
through pure floating-point reassociation.

- **Crack COD back-ends no longer allocate per quadrature node.**
  `_qnn_pair_components` (the innermost loop of the whole `Cracks` module) is
  now a pure function returning an `SMatrix{3,3,T}` instead of writing into a
  caller buffer through ~10 heap 3×3 `Matrix{T}` temporaries per α node;
  `_A_and_Tn`, `_phi_cache` and `_inv3` likewise return `StaticArrays`.
  `cod_tensor` on an elliptic crack in a triclinic matrix: **−85 % time,
  −99.4 % allocations** (18.5 MB → 111 KB). `StaticArrays` becomes a direct
  dependency (it was already in the manifest transitively).

- **Anisotropic 2D Hill uses one vector-valued quadrature.** It ran 16
  separate `quadgk` calls, each evaluating the full 16-component integrand and
  keeping one component. **−35 % time, −18.5 % allocations.**

- **DECUHR and anisotropic-cylinder Hill use the closed-form acoustic
  inverse.** Both built a symmetric 3×3 `K` with a 4-deep 81-iteration loop
  and then called generic LU `inv` per node; `_sym3_inv_acoustic` already did
  this allocation-free and Dual-safe. **−62 % time, −80.7 % allocations.**

- **Green helpers exploit the major symmetry of `C`.** `Kns` follows from
  `Vs + transpose(Vs)` at zero flops instead of an 81-iteration loop, and `Ks`
  is accumulated on its upper triangle only. **−21.5 %** on `hill_tensor`
  under `ForwardDiff.Dual`.

- **The self-consistent Newton solver stops recomputing residuals it already
  has.** The line search accepts `r_new` and the next iteration recomputed the
  residual at the same point (one full RVE pass — i.e. one `hill_tensor` per
  phase — per iteration); `Tref` was derived from a whole extra evaluation
  whose value was discarded. **−18.4 % allocations.**

- Requires TensND 0.2.6 to benefit from its `TensOrtho` `getindex` and
  `tensor_or_array` fixes (`−57 %` on `⊡` between structured operands).

### Removed

- **15 dead functions**, none reachable from `src/`, `ext/`, `test/` or
  `docs/`: `_Qnn_direct` and `_acoustic_tensor` (`Core/green_kernel.jl`),
  `Core._quadgk` (`Core/quadrature.jl`), `_masson_log`
  (`Core/green_residue.jl`), `_sc_pd_guard` and `_rve_in_compliance_space`
  (`Schemes/self_consistent.jl`), `_amounts_with_promoted_eltype` and
  `_replace_tuple_at` (`Schemes/parameters.jl`), `_mandel66_to_tens`,
  `_block_value_tensor`, `_block_value_mandel`, `_get_block`
  (`Viscoelasticity/trapezoidal.jl`), `_block_value_order2_tens` and
  `_block_value_order2_mat` (`Viscoelasticity/order2_alv.jl`),
  `_sc_alv_mt_body_against` and `_sc_alv_step_echoes`
  (`Viscoelasticity/schemes_alv_sc.jl`).

  `_Qnn_direct` was the starting point of this whole audit: it contracted the
  stiffness with the Green kernel through a six-deep `p,q,r,s,α,β` loop — 729
  iterations where the contraction factorizes into `U = (C·n̂)·ξ` then
  `B = U·K⁻¹·Uᵀ`, roughly 100 flops. The factorized form already existed twice
  in the package (`Cracks/green_residue.jl`, `Core/green_helpers.jl`), so the
  slow copy was deleted rather than optimized, and the header of
  `green_kernel.jl` now points at the two live implementations.

  `Core._quadgk` deserves a special mention: its own header described it as
  the wrapper "all downstream sub-modules should go through", and no call site
  had ever used it.

### Compatibility & validation

- Aligned with TensND 0.3 (snake_case + UPPERCASE-acronym API).
- Cross-validated against C++ ECHOES to ≤ ~1e-3 (moduli) and machine precision
  (elastic limits) across schemes, porous, layered and ALV benchmarks.
- ~3900 tests.

---

## Pre-1.0 development history

The entries below predate the first public release and use the internal
version numbers under which each feature was developed (never published outside
MPCM-Registry). Kept for reference.

### v0.8.1 — Packaging & spelling fix

#### Fixes

- Documentation: unified American English spelling throughout
  (`homogenization` consistently, replacing `homogenisation`)

#### Infrastructure

- Registered in MPCM-Registry; DECUHR.jl resolved via registry
  (no more `[sources]` local path in `Project.toml`)
- GitHub Actions workflows: CI, Documentation, Register, CompatHelper, Format, TagBot
- Multi-version documentation deployment (`docs/deploy_docs.jl`)

---

### v0.8.0 — Differential scheme as a SciML ODE on the fictitious incorporation time

**`DifferentialScheme` is now solved by an adaptive SciML ODE
integrator** (`Tsit5` default) on the fictitious incorporation time
`τ ∈ [0, 1]`, replacing the explicit Euler discretization.  The
underlying multi-phase incorporation-sequence ODE
([Norris 1985](@cite norris1985); user's hand-written DEM note) reads

```math
\frac{\mathrm d \mathbb C^{hom}}{\mathrm d \tau}
  = \sum_\alpha \frac{\mathrm d \varphi_\alpha}{\mathrm d \tau}
                (\mathbb C_\alpha - \mathbb C^{hom}):\mathbb A_\alpha^{dil}(\mathbb C^{hom})
   + \sum_c \frac{\mathrm d \varepsilon_c}{\mathrm d \tau}
            \Delta\mathbb C^{crack}_c(\mathbb C^{hom})
```

with the volume balance `df = (𝟙 − f ⊗ 𝐔) · dφ` inverted by
Sherman-Morrison so the user supplies effective volume fractions
`f_α(τ)` along the chosen `trajectory` (cracks contribute their
density derivative `dε_c/dτ` directly — no Sherman-Morrison
correction needed).  Implemented in elastic, conduction and ALV
pipelines (`src/Schemes/differential.jl`,
`src/Viscoelasticity/schemes_alv_extra.jl::differential_alv`).

`OrdinaryDiffEq` is now a strong dependency.

**New trajectory type `Path(Dict(:phase => τ -> f(τ)))`** for fully
functional incorporation paths.  Derivatives `df_α/dτ` are computed
by `ForwardDiff.derivative` automatically.  The existing
`Proportional()`, `Sequential(order)` and `CustomPath(Dict(:phase => Vector))`
trajectories are preserved for back-compat — internally rewritten
as callables on `τ`.

**`DifferentialScheme` constructor extended** :

```julia
DifferentialScheme(; trajectory = Proportional(),
                     nsteps::Int = 100,
                     abstol::Real = 1e-8,
                     reltol::Real = 1e-6,
                     alg = nothing,    # `nothing` → Tsit5()
                     kwargs...)
```

`nsteps` is now reinterpreted as the **`saveat` density** along `τ`
(the integration step is controlled by `abstol`/`reltol`).  Existing
scripts that pass `nsteps = 50` continue to work.

**ALV cracks in `differential_alv`** : the `# deferred` placeholder
is resolved.  Crack phases now contribute `dε_c/dτ ·
ΔC̃^crack_c(C̃)` to the RHS at every solver step (Sevostianov
interface-stiffness correction propagated automatically when `:Rn`
/ `:Rt` ViscoLaws are attached).

**Demo script** `scripts/46_differential_loading_paths.jl` shows the
genuine path-dependence of DEM : a 3-phase composite reaching the
same target volume fractions `f_α^∞` along four different
trajectories (`Proportional`, two `Sequential` orderings, and a
functional `Path(τ -> τ², 2τ−τ²)`) yields four different effective
moduli at `τ = 1` — the DEM iteration sees a different effective
medium at each infinitesimal phase increment depending on the
incorporation history.

### v0.7.0 — Sevostianov interface stiffness, ECHOES SC body, Newton-Raphson SC

**Sevostianov-style crack interface stiffness** is now supported in
all three pipelines (elasticity, conductivity, ALV).  The COD
compliance becomes `B_eff = B · (𝟙 + b·K·B)^{-1}` with `K` a
3×3 spring-like 2-tensor in elasticity (kept as `:K_interface` on the
crack phase), a scalar Kapitza conductance in conductivity
(`:α_interface`), and a `ViscoLaw`-valued kernel in ALV (a pair
`(:Rn, :Rt)` of normal/tangential relaxation laws).  Validation
scripts `scripts/44_alv_cracks_interface.jl` and
`scripts/45_cracks_iso_interface.jl` cross-check the implementation
against ECHOES C++ via PyCall (≤ 3·10⁻⁴ relative error across MT, SC,
ASC, PCW, Maxwell, Differential at d = 0.30 ALV / d = 0.50 elastic).

**MT and SC schemes now match ECHOES `B·A^{-1}` form for cracks**
(elastic, conduction, ALV).  The textbook symmetric Hill / Budiansky
SC iteration map collapses crack-rich RVEs to the trivial percolated
fixed point and the additive MT for cracks similarly percolates well
below the ECHOES-reported moduli.  Tracing the C++ reference shows
that ECHOES' `compute_strain_Stress` returns `A_α · S_n` for solid
inclusions but the bare `H_c` (no `S_n` factor) for void cracks, so
the trailing `S_n` cancellation only holds for solid-only RVEs.  The
MFH MT and SC bodies now mirror this :

```julia
A_E = (Σ_solids f·sym(A_α(C_n))) · S_n + Σ_cracks ε · sym(H_c(C_n))
B_E = (Σ_solids f·sym(C_α·A_α(C_n))) · S_n     # cracks → 0 (traction-free)
C_eff = B_E · A_E^{-vol}
```

Fix applied symmetrically to MT (`mori_tanaka.jl`), elastic SC
(`self_consistent.jl`), conduction SC (same dispatcher), and ALV SC
(`schemes_alv_sc.jl::_sc_alv_step_echoes_form`).  Brings SC for cracks
from 74 % off ECHOES to 2·10⁻³ at d = 0.50, and ALV SC + interface
stiffness from 1 % off to 1.4·10⁻⁴ at d = 0.30.

**ForwardDiff promoted to a strong dependency.**  The four
`derivative` / `gradient` / `jacobian` / `sensitivity` autodiff
entry points are now available out of the box (no `using ForwardDiff`
needed) — the previous weak extension `MeanFieldHomogenizationForwardDiffExt` is
removed.  This also enables the new built-in Newton-Raphson SC solver
below.

**Built-in Newton-Raphson SC solver** (`SelfConsistent(; algorithm =
NewtonDefault())`).  Replaces the prior weak-extension stub: the
solver now ships with the package and uses `ForwardDiff.jacobian` on
the iso / TI / ortho / aniso canonical components of the running
estimate.  Quadratic convergence (typically 5–10 iterations vs ~100
for `AndersonDefault` Picard) and a backtracking line search make it
robust on high-contrast configurations.  Falls back to a single
Picard step when the line search exhausts.  Available for both
[`SelfConsistent`](@ref) and [`AsymmetricSelfConsistent`](@ref).

**Eigenvalue guard `_sc_pd_guard`** for the SC running estimate :
mirrors ECHOES `homogenization_scheme.h::evaluate` by detecting a
non-positive-definite running estimate and resetting it to a tiny
positive baseline before each step.  Prevents Picard / Newton from
collapsing to the trivial `C = 0` fixed point near the percolation
threshold.

### v0.6.0 — TI ALV fast path, order-2 ALV, BLAS Volterra, ALV cracks roadmap

**TI Walpole-basis fast path** for ALV homogenization : when every
phase 4-tensor and the matrix kernel are TI 4-tensors with the
**common canonical axis n = e₃** (every 6×6 Mandel block matches the
Walpole structure), `homogenize_alv` now routes through new
`*_alv_ti(ℓ_…)` primitives that operate on **6** `n × n` Volterra
matrices `(ℓ₁, ℓ₂, ℓ₃, ℓ₄, ℓ₅, ℓ₆)` instead of the generic `(6n × 6n)`
block.  The Walpole 2×2 part `[[ℓ₁, ℓ₃]; [ℓ₄, ℓ₂]]` is packed as a
`(2n)×(2n)` block-Volterra matrix and inverted via
`volterra_inverse(_; block_size = 2)`; the two scalar shears
`(ℓ₅, ℓ₆)` go through the LAPACK scalar fast path below.

ISO inputs are subsumed automatically (iso ⊂ TI), so a TI-matrix +
iso-inclusion combination — common in layered concrete /
fiber-reinforced ALV — gets the fast path "for free".  Storage is 6 ·
n² doubles per phase (vs 36 · n² generic), and the inverse cost drops
from `O((6n)²)` to `O((2n)²) + 2·O(n²) ≈ ×3` cheaper than the generic
6n×6n path.

The TI fast path is integrated into all six schemes (Voigt / Reuss /
Dilute / DiluteDual / Mori-Tanaka / Maxwell), via a new
`_try_ti_tuples` helper analogous to the existing `_try_iso_pairs`
detection.

**Order-2 ALV** : new sub-module covering vector-tensor ageing
viscoelasticity (thermal / electrical conductivity, diffusivity,
permittivity).  Operators are stored as `(3n × 3n)` lower-block-
triangular matrices with 3×3 blocks.  Mirrors the order-4 API:

  * `homogenize_alv_order2(rve, scheme, prop; times)` — public entry
    point, dispatching on Voigt / Reuss / Dilute / DiluteDual /
    Mori-Tanaka / Maxwell schemes.
  * `hill_kernel_order2(ell, K_0_law, times)` — Hill polarization for
    iso ALV matrix + ellipsoidal inclusion (time-space decoupling
    `P̃[block(i,j)] = α₀^{-vol}[i,j] · 𝐈^A`, with `𝐈^A` from the
    existing elastic `tens_IA(ell)`).
  * `iso_order2_params_from_blocks` / `iso_order2_blocks_from_params`
    — per-component parameter extraction (single scalar α for iso
    order-2).
  * `voigt_alv_order2`, `reuss_alv_order2`, `dilute_alv_order2`,
    `dilute_dual_alv_order2`, `mori_tanaka_alv_order2`,
    `maxwell_alv_order2`.

`trapezoidal_matrix(law, times)` now accepts both order-4 (4-tensor /
6×6 Mandel) and order-2 (`TensND.AbstractTens{2,3}` / 3×3 matrix)
sample types, dispatching to the appropriate (B·n)×(B·n) layout.

The order-2 elastic-limit test verifies that ALV reduces to the
existing elastic conductivity `homogenize` to machine precision for
both spherical and spheroidal inclusions.  `scripts/bench_echoes/
bench_order2_alv.jl` provides a template for a Julia–ECHOES
crosscheck on the `fluage_echoes_maxwell_ordre2.py` setup.

**Volterra BLAS / LAPACK fast path** : `volterra_inverse`,
`volterra_left_divide` and `volterra_divide` now dispatch to LAPACK
`trtri` / `trsm` via the `LowerTriangular(...)` wrapper for
`BlasFloat` element types and `n ≥ 64` (the crossover where BLAS
overhead amortizes).  Measured speedups vs the hand-rolled forward
substitution: **×9.7** at `n = 500`, **×14.6** at `n = 1000`.  The
hand-rolled fallback is preserved for small grids and for
non-BlasFloat element types (`BigFloat`, `Sym`, `ForwardDiff.Dual`).

**ALV cracks roadmap** : new file
`src/Viscoelasticity/cracks_alv.jl` documents the planned
`cod_kernel_alv` / `compliance_contribution_alv` API, the time-space
decoupling formulas for pure penny / interface-stiffness cracks in
iso ALV matrices, and the integration points with the existing
`CrackDensity` amount in `Schemes`.  Implementation is scheduled for
v0.6.1.

### v0.5.3 — Non-uniform time grid + multi-layer ALV stability + iso fast path

**Bug fix (membrane interface convention)** : when v0.5.2 introduced
the C++-convention σ-form shear M-matrix, the elastic-limit unit test
for `MembraneInterface` started failing because the membrane jump
expressions and the M-matrix used different angular-component
normalisations.  Resolution: keep the M-matrix in the
**Christensen–Lo / SymPy convention** (matching the elastic
state-space recurrence in `LayeredSpheres`) and derive the analytic
`M^{-1}` from the C++ closed form via the diagonal conjugation

```text
M_C++ = D_row · M_Christ–Lo · D_col,
   D_row = diag(1/2, 1, 1/2, 1),  D_col = diag(1, 1, 2, 2)
⇒ M_Christ–Lo^{-1} = D_col · M_C++^{-1} · D_row
```

This gives the **best of both worlds**: numerical stability of the
closed-form (only `(3κ+4μ)^{-vol}` and `μ^{-vol}` `n×n` Volterra
inverses are needed) under the elastic-compatible mode normalization
where the Christensen–Lo membrane jumps stay correct.  The
`Membrane interface (elastic limit)` test is back to `≤ 1e-10`
tolerance with no convention asterisk.

**Performance — iso-symmetry fast path** : when every phase 4-tensor
and the matrix kernel are iso (`TensISO{4,3}`-valued),
`homogenize_alv` automatically routes through new
`*_alv_iso(αβ_…)` scheme primitives that operate on two scalar
`n × n` Volterra matrices `(α = 3K, β = 2μ)` instead of the generic
`(6n × 6n)` Mandel block matrix.  Theoretical speedups :

  - matrix-matrix product : ~108× cheaper (`216 n³` → `2 n³`)
  - matrix inverse        : ~18× cheaper
  - storage               : 18× smaller

Measured speedup on the script-37 setup (5 phases + Maxwell matrix +
spherical inclusions, MT scheme): **×2.5–7** depending on `n_times`.
The detection happens once per phase via a cheap iso-form pattern
check (`_is_iso_block`); on failure the generic 6n×6n path is
selected automatically — no API change.  New internal helpers
(`iso_schemes_alv.jl`) :
`voigt_alv_iso`, `reuss_alv_iso`, `dilute_alv_iso`,
`dilute_dual_alv_iso`, `mori_tanaka_alv_iso`, `maxwell_alv_iso`,
plus `dilute_concentration_alv_iso`, `dilute_contribution_alv_iso`
used inside `_inclusion_alv_quantities`.

**Bug fix (multi-layer ALV stability)** : on a non-uniform time grid
(e.g. `logspace`) with a matrix relaxation kernel that has multiple
time constants and/or layers with extreme modulus contrast (pores,
step-activated `ViscoLaw`s), the layered-sphere ALV recurrence
diverged from the ECHOES Python reference by 1e-3 to 1e-2.  Two
compounding root causes :

1. **Right vs left Volterra divide.**  The Hervé–Zaoui closed-form
   interface transition `T = M_b^{-1} · M_a` requires the Volterra
   inverse on the **left** of the numerator.  Our `volterra_divide`
   implemented `num · S^{-vol}` (right) ; the two products are equal
   only when `[num, S] = 0`.  Lower-triangular Volterra trapezoidal
   matrices commute pairwise iff they are Toeplitz (uniform time
   grid + same kernel structure) — non-uniform grids broke this
   assumption silently.

2. **Generic 4n×4n inversion of the shear M-matrix is FP-unstable
   for soft phases.**  Even the block forward-substitution
   `volterra_inverse(_; block_size = 4)` collapses when
   `det(M[t,t]) → 0` (pore-like or step-activated layers).
   ECHOES C++ uses a **closed-form analytic 4×4 inverse** whose only
   `n × n` Volterra inverses are `(3κ + 4μ)^{-vol}` and `μ^{-vol}` —
   both regular for any non-vacuum modulus.

**Fix** :

- Added `volterra_left_divide(S, M; block_size = 1|6)` (forward
  substitution on `S · T = M`, rows i = j..n) and switched every
  closed-form transition (perfect / spring / membrane, both bulk and
  shear) to use it.
- Added `_shear_M_inverse_alv(r, M_κ, M_μ, n)` returning the
  closed-form `M(r; κ, μ)^{-1}` in time-major 4n×4n layout, mirroring
  C++ `inclusion_sphere_nlayers.h::set_visco_inv_matrix_dev` and then
  conjugated to the Christensen–Lo convention.  Used by
  `_shear_layer_transfer_alv` (intra-layer transfer) and
  `_shear_amp_blocks_alv` (state → amplitude extraction).
- Reverted the v0.5.2 τ-scaling for the shear M-matrix — the
  closed-form inverse is naturally written in σ-form and FP
  stability now comes from the closed form rather than the rescaling.

**Impact** : `script 37 :layers` now produces smooth, monotonic
creep curves matching the Python reference figure visually
(bounded between elastic limit and matrix Maxwell), in place of
the previous unbounded / oscillating output.  Bench results :

- `bench_layered_alv.jl` (N=2 stiff elastic + Maxwell matrix,
  uniform grid) : 1e-16 (unchanged).
- `bench_layered_alv_step.jl` (N=3 step-activated layers + pore +
  Maxwell matrix, non-uniform grid) : bulk α 1e-15, shear β 1e-14.
- `bench_step_n2.jl` (N=2 step layers, no pore) : 1e-14.
- `bench_layered_alv_nopore.jl` (N=4 elastic, varied moduli) : 1e-15.

### v0.5.1 — Multi-layer sphere shear localization bug fix

**Bug fix** : on a non-uniform time grid (e.g. `logspace`) with a
matrix relaxation kernel that has multiple time constants and/or
layers with extreme modulus contrast (pores, step-activated
`ViscoLaw`s), the layered-sphere ALV recurrence diverged from the
ECHOES Python reference by 1e-3 to 1e-2.  Two compounding root
causes :

1. **Right vs left Volterra divide.**  The Hervé–Zaoui closed-form
   interface transition `T = M_b^{-1} · M_a` requires the Volterra
   inverse on the **left** of the numerator.  Our `volterra_divide`
   implemented `num · S^{-vol}` (right) ; the two products are equal
   only when `[num, S] = 0`.  Lower-triangular Volterra trapezoidal
   matrices commute pairwise iff they are Toeplitz (uniform time
   grid + same kernel structure) — non-uniform grids broke this
   assumption silently.

2. **Generic 4n×4n inversion of the shear M-matrix is FP-unstable
   for soft phases.**  Even the block forward-substitution
   `volterra_inverse(_; block_size = 4)` collapses when
   `det(M[t,t]) → 0` (pore-like or step-activated layers).
   ECHOES C++ uses a **closed-form analytic 4×4 inverse** whose only
   `n × n` Volterra inverses are `(3κ + 4μ)^{-vol}` and `μ^{-vol}` —
   both regular for any non-vacuum modulus.

**Fix** :

- Added `volterra_left_divide(S, M; block_size = 1|6)` (forward
  substitution on `S · T = M`, rows i = j..n) and switched every
  closed-form transition (perfect / spring / membrane, both bulk and
  shear) to use it.
- Added `_shear_M_inverse_alv(r, M_κ, M_μ, n)` returning the
  closed-form `M(r; κ, μ)^{-1}` in time-major 4n×4n layout, mirroring
  C++ `inclusion_sphere_nlayers.h::set_visco_inv_matrix_dev`.  Used
  by `_shear_layer_transfer_alv` (intra-layer transfer) and
  `_shear_amp_blocks_alv` (state → amplitude extraction).
- Reverted the v0.5.2 τ-scaling for the shear M-matrix — the
  closed-form inverse is naturally written in σ-form and FP
  stability now comes from the closed form rather than the rescaling.
- Rewrote `_shear_M_matrix_alv` to use the C++ ECHOES mode
  normalization (mode 1 contributes `U = a · r`, not Christensen–Lo's
  `2a · r`) so it stays consistent with the analytic `M^{-1}`
  formula.  Mode-2 dev contribution factor
  `F_k = (21/5) μ^{-vol} (3κ + μ) (r_b⁵ − r_a⁵)/(r_b³ − r_a³)` was
  unchanged ; it cancels the mode-2 amplitude scaling implicitly.

**Impact** : `script 37 :layers` now produces smooth, monotonic
creep curves matching the Python reference figure visually
(bounded between elastic limit and matrix Maxwell), in place of
the previous unbounded / oscillating output.  Bench results :

- `bench_layered_alv.jl` (N=2 stiff elastic + Maxwell matrix,
  uniform grid) : 1e-16 (unchanged).
- `bench_layered_alv_step.jl` (N=3 step-activated layers + pore +
  Maxwell matrix, non-uniform grid) : bulk α 1e-15, shear β 1e-14.
- `bench_step_n2.jl` (N=2 step layers, no pore) : 1e-14.
- `bench_layered_alv_nopore.jl` (N=4 elastic, varied moduli) : 1e-15.

### v0.5.1 — Multi-layer sphere shear localization bug fix

**Bug fix** : `LayeredSpheres._shear_localization` and the
companion ALV `shear_localization_alv` previously returned only the
mode-1 amplitude `a_k`, which is the correct per-layer dev β only
when `b_k` (mode-2 amplitude) vanishes — true for `N = 1` (single
sphere) and for the degenerate `N = 2` cases tested in
`test_christensen.jl` (shell ≡ matrix or core ≡ shell).  For
genuinely multi-layer composite spheres with distinct core, shell
and matrix moduli, `b_k` is non-zero and contributes to the
volume-averaged deviatoric strain via the mode-2 r³ profile.

The corrected per-layer dev localization is

```text
β_k = a_k + b_k · (21/5) (3κ_k + μ_k)/μ_k · (r_k⁵ − r_{k-1}⁵)/(r_k³ − r_{k-1}³)
```

(modes 3 and 4 contribute zero to the layer-volume-averaged
deviatoric strain by angular orthogonality).  This matches ECHOES
C++ `inclusion_sphere_nlayers.h::get_visco_layer_average_strain_Strain`
to machine precision.

**Impact** : the ALV `:layers` topology of `script 37` now matches
the Python `fluage_echoes_solid.py` reference for N ≥ 2.  The
elastic `stiffness_contribution(LayeredSphere, C₀)` and the ALV
`stiffness_contribution_alv(LayeredSphere, C₀_law, times)` produce
the correct effective dilute / MT moduli.

**Validation** : new cross-check benchmark
`scripts/bench_echoes/bench_layered_alv.{py,jl}` and a regression
test `shear_localization_alv — N=2 cross-check vs ECHOES Python` in
`test/Viscoelasticity/test_layered_alv.jl` pin the ALV per-layer
α(t,t') and β(t,t') Volterra blocks to ECHOES Python at machine
precision (1e-16 on the diagonal, 1e-6 on the off-diagonal blocks).

### v0.5.0 — Ageing linear viscoelasticity (ALV) module

A new `MeanFieldHomogenization.Viscoelasticity` sub-module brings time-domain
viscoelastic homogenization to the package, mirroring the capabilities
of the C++ ECHOES `viscoelasticity/visco_law.h` and
`homogenization_maxwell.h`.  Reference: Sanahuja IJSS 2013 ;
Barthélémy-Giraud-Lavergne-Sanahuja IJSS 2016 ;
Barthélémy-Giraud-Sanahuja-Sevostianov IJES 2019 ; ECHOES manual
chapter 7 and appendix `viscoelastic_hill_kernel.qmd`.

#### Highlights

- **`ViscoLaw`** : abstract relaxation `R(t,t')` or creep `J(t,t')`
  kernel, scalar- or 4-tensor-valued, with built-in convenience
  constructors `maxwell_relaxation`, `kelvin_creep`, `maxwell_iso`,
  `kelvin_iso`, `heaviside_law`.
- **`trapezoidal_matrix`** : Sanahuja-2013 trapezoidal discretization of
  the Stieltjes integral on a time grid `times`, returning a dense
  `Matrix{T}` of size `(B·n) × (B·n)` in lower-block-triangular form
  (`B = 6` for 4-tensor in Mandel convention, `B = 1` for scalar
  kernels).
- **`volterra_inverse`** : block-triangular forward-substitution that
  takes a discrete relaxation matrix to its discrete creep matrix in
  `O(B³ n²)` flops.
- **`hill_kernel`** : discrete ALV Hill polarization tensor for an
  ellipsoidal inclusion in an isotropic ALV matrix, using the
  time-space decoupling formula of the manual appendix : reuses the
  elastic auxiliary tensors `tens_UA`, `tens_VA` and combines them with
  two scalar Volterra inverses (longitudinal and shear moduli).
  Machine-precision agreement with the elastic Hill tensor in the
  Heaviside (elastic) limit.
- **`homogenize_alv(rve, scheme, prop; times)`** : public entry point
  that builds the discrete operators for every phase, computes the
  ALV Hill kernel, and dispatches to the appropriate scheme function.
  Implemented schemes : `Voigt`, `Reuss`, `Dilute`, `DiluteDual`,
  `MoriTanaka`, `Maxwell`.  Each one's output coincides with the
  corresponding elastic homogenization in the Heaviside limit (verified
  to machine precision in the test suite).
- **`Phase.properties` relaxed to `Dict{Symbol, Any}`** : a phase can now
  carry either an elastic `AbstractTens` or a `ViscoLaw` under the
  same key (`:C`).  No regression in the elastic test suite (3421/3421).
- **Scripts** : `scripts/33_visco_law_basics.jl` (kernels + plot),
  `scripts/37_fluage_echoes_solid.jl` (Sanahuja-style ageing creep of a
  solidifying composite, whole-pores topology, mirroring
  `tests/python/creep/fluage_echoes_solid.py` after [@sanahuja2013] and
  chapter 9 §"Ageing creep of solidifying cementitious materials").
- **Tests** : `test/Viscoelasticity/` adds 525 new tests across
  `test_visco_law.jl`, `test_trapezoidal.jl`, `test_volterra_inverse.jl`,
  `test_hill_alv_iso.jl`, `test_schemes_alv.jl`.  Total package test
  count : 3946/3946 PASS.

#### Self-Consistent ALV (added in 0.5.0)

- **`self_consistent_alv(rve, prop; times, abstol, reltol, maxiters,
  damping, verbose, select_best)`** — symmetric SC fixed-point iteration
  on the `(6n × 6n)` block matrix.  Each iteration recomputes the
  per-phase Hill kernels using the running estimate's iso parameters,
  computes the dilute concentration tensors, and forms the next
  iterate.  Convergence on the Frobenius norm.
- Plumbed into the dispatcher via `homogenize_alv(rve, SelfConsistent(),
  :C; times = T)`.
- Tests against the elastic SC limit pass at machine precision.

#### N-layer sphere ALV — full bulk + shear recurrence (added in 0.5.0)

- **`bulk_localization_alv(sphere::LayeredSphere, C0_law, times)`** —
  per-layer bulk localization matrices `α_k(t,t')` of size `n × n`
  (one per layer).  Extends the elastic Hervé-Zaoui bulk recurrence
  ([`LayeredSpheres/bulk_recurrence.jl`]) by replacing every scalar
  modulus with its `n × n` trapezoidal Volterra matrix, building
  `(2n × 2n)` block transfer matrices.
- **`shear_localization_alv(sphere::LayeredSphere, C0_law, times)`** —
  per-layer deviatoric (Y₂-harmonic) localization matrices `β_k(t,t')`
  of size `n × n`.  Builds the Hervé-Zaoui 1993 4×4 fundamental matrix
  in **time-major** layout (`(4n × 4n)` block-lower-triangular with
  4×4 diagonal blocks), inverts it via `volterra_inverse(_;
  block_size = 4)`, propagates two probe states through the layers,
  and selects the linear combination matching unit far-field
  `(a_{N+1}, b_{N+1}) = (I_n, 0)` via a final `(2n × 2n)`
  `block_size = 2` Volterra solve.  Verified to machine precision
  against `LayeredSpheres._shear_localization` in the Heaviside
  (elastic) limit.
- **`bulk_state_seq_alv`** / `_shear_state_seq_alv` — forward
  propagation of the discrete state vectors through every layer.
- **ALV interface transfers** (`_bulk_interface_T_alv`,
  `_shear_interface_T_alv`) cover the same set of imperfect interface
  models as the elastic counterpart : `PerfectInterface`,
  `SpringInterface(kn, kt)` (primal — displacement jump driven by
  `kn`/`kt`) and `MembraneInterface(κs, μs)` (dual — traction jump
  from surface elasticity).  Each interface parameter may be a plain
  scalar (constant in time, the elastic limit) **or** a `ViscoLaw`
  scalar kernel — in the latter case the jump itself is ageing and
  the corresponding `n × n` block is the parameter's trapezoidal
  matrix.  The `(4n × 4n)` shear block-diagonal (4×4 sense) cleanly
  reduces to the elastic 4×4 jump for scalar parameters.
- **Composite-sphere assembly**:
  `strain_strain_loc_alv(sphere, C0_law, times)` builds the volume-
  averaged strain-strain localization tensor
  `A_avg = ⟨α⟩ 𝕁 + ⟨β⟩ 𝕂` (`(6n × 6n)`), and
  `stiffness_contribution_alv(sphere, C0_law, times)` builds the
  size-independent stiffness contribution
  `N = 3 Σ_k f_k (M_κ_k − M_κ_0) ∘ α_k 𝕁
       + 2 Σ_k f_k (M_μ_k − M_μ_0) ∘ β_k 𝕂` (`(6n × 6n)`).
- **`homogenize_alv` extended to `LayeredSphere` phases**: the
  per-inclusion quantities (`A_dut`, `N_dut`) are computed via the
  layered-sphere recurrence instead of the Hill kernel + dilute
  pipeline.  Verified to machine precision against the elastic
  reference for the Dilute scheme and against the elastic MT for the
  `t = t' = 0` block of a Maxwell relaxation kernel.
- **`scripts/37_fluage_echoes_solid.jl`** now exposes a `MODEL`
  constant (`:whole_pores` / `:layers`) selecting the topology;
  the `:layers` branch reproduces the Python `sphere_nlayers(...)`
  setup of `tests/python/creep/fluage_echoes_solid.py` exactly.

#### Deferred to follow-up

- Cracks in ALV (extrapolating from the elastic `cod_tensor` /
  `compliance_contribution` infrastructure).
- Self-Consistent ALV with `LayeredSphere` phases (the current SC
  ALV iteration handles only `Ellipsoid`-geometry inclusions).
- Anisotropic ALV Hill kernel (numerical surface integral with Volterra
  inverse of the 3×3 acoustic tensor at each integration point).

### v0.4.0 — Friendly autodiff sensitivities, RVE-level symmetrize, Hill-symmetric SC

A small but expressive API exposing `ForwardDiff`-based derivatives of any
homogenization result with respect to any scalar input parameter — physical
(stiffness coefficient, conductivity), geometric (radii, semi-axes, crack
opening, distribution-shape envelope) or volume-fraction / crack-density —
*and* for arbitrary scalar fields of inclusion types defined later by the
user. The autodiff path unlocks geometric and user-type sensitivities that
were not previously practical, and the multi-scale chain rule is taken care
of automatically by composing several `homogenize` calls inside a single
closure.

The release also ships an RVE-level orientation-distribution projection
(`symmetrize`), a corrected Hill-symmetric self-consistent scheme that
percolates exactly at φ=0.5 for spherical pores, a `Spheroid` convenience
constructor, Dual-stable SC convergence, and a `select_best` mode that
mirrors the C++ reference's behavior at percolation thresholds.

#### Additions

- **Lens hierarchy** `AbstractParameter` with four concrete kinds
  (`AmountParameter`, `PropertyParameter`, `GeometryParameter`,
  `DistributionShapeParameter`) plus user-friendly helpers (`amount`,
  `property`, `geometry`, `shape_param`).
- **`get_param(rve, p)` / `set_param(rve, p, value)`** — read / immutable
  update of the scalar designated by a lens, with automatic eltype
  promotion to integrate `ForwardDiff.Dual` cleanly.
- **Public autodiff entry points** `derivative`, `gradient`, `jacobian`
  and the closure fallback `sensitivity(f, x₀)`. They become available
  after `using ForwardDiff` (weak extension `MeanFieldHomogenizationForwardDiffExt`).
- **Generic geometry-field reflection** `_replace_geom_field` based on
  `@generated` reconstruction with uniform sibling-field eltype
  promotion. User-defined inclusions whose constructor follows the
  parametric Julia auto-generated pattern (`MyType{T,B}(args...)`) are
  differentiable through their scalar fields without any library change.
- **Symbol selectors for property tensors** mapping named coefficients
  (`:bulk`, `:shear`, `:transverse`, `:axial`, `:ℓ₁`..`:ℓ₆`) to the
  positional indices of `get_data(tensor)` for `TensISO{2}`,
  `TensISO{4,3}`, `TensTI{2}` and `TensTI{4}`.
- **`MeanFieldHomogenizationForwardDiffExt`** weak extension activating the public
  API on `using ForwardDiff`. ForwardDiff is registered in `[weakdeps]`
  alongside NonlinearSolve and SymPy; no new hard dependency.
- **RVE-level orientation symmetrize** via the `symmetrize` keyword on
  `add_matrix!` / `add_phase!`. Three options:
  - `:none` (default): no projection.
  - `:iso`: Reynolds average over `SO(3)` ⇒ isotropic contribution
    (`TensISO`).
  - `:ti` / `TISymmetrize(axis)`: Reynolds average over rotations around
    `axis` ⇒ transversely-isotropic contribution (`TensTI(axis)`).
  Implemented for tensor orders 2 and 4. The TI projection currently
  routes the matrix through an iso projection during the
  localization-tensor computation (workaround for non-coaxial inclusion
  families); see [`src/Schemes/symmetrize.jl`](src/Schemes/symmetrize.jl)
  for the rationale.
- **`Spheroid(ω; euler_angles)`** convenience constructor on top of
  `Ellipsoid`, mirroring the `spheroidal(omega)` helper of the C++
  reference: `ω = c/a` with one polar semi-axis equal to `ω` and two
  equatorial ones equal to `1`. Eshelby/Hill computations are
  scale-invariant so only the aspect ratio matters.
- **`select_best` keyword on the SC fixed-point solver** — when `true`,
  the solver tracks the best iterate seen during Picard iteration
  (smallest residual on the value field) and returns it at the end.
  Useful for high-contrast iterations that oscillate around the fixed
  point near percolation thresholds; matches the C++ reference's
  `select_best=True` mode.

#### Fixes

- **Hill-symmetric self-consistent**: every phase now contributes a
  non-trivial dilute concentration `A_α = inv(I + P(C_α − C_eff))`
  computed in the iterating effective medium, including the matrix
  phase. The previous SC step treated the matrix as having `A = I`
  (Mori-Tanaka-style), which gave the upper SC branch only and
  misplaced the porous-sphere percolation threshold. With the fix,
  porous spheres percolate exactly at φ = 0.5.
- **Dual-stable SC convergence criterion** — the Picard convergence
  test now requires both the value AND every partial of the residual to
  fall below `abstol`. Without that, the value can converge while the
  partials carry residual error of order `‖∂step/∂x‖ × abstol`,
  producing numerically wrong sensitivities through the SC fixed point.
- **TI symmetrize Walpole normalization**: the `_apply_symmetrize` for
  `TISymmetrize` now divides the W₅ and W₆ projection coefficients by
  `‖W_k‖² = 2`, matching the basis-decomposition convention of
  `TensND.TensTI{4}`. Round-trip on a coaxial TI(ez) tensor is now
  exact.

#### Documentation

New manual page `manual/sensitivities.md` (motivation, lens API, closure
fallback, user-inclusion tutorial, multi-scale chain-rule example, and a
section on the `symmetrize` keyword) and auto-API page
`api/sensitivities.md`. Both wired into `docs/make.jl`.

#### Scripts

- `scripts/26_sensitivities.jl` — tour of the API (lenses + gradient +
  jacobian + cross-check vs the Christensen 1990 closed form for
  `∂k_MT/∂f`, agreement to ~1e-16).
- `scripts/27_user_inclusion_sensitivity.jl` — extensibility demo on a
  user-defined inclusion type `MyBlob{T,B}` with two numeric fields
  (`radius`, `eccentricity`).
- `scripts/28_multiscale_strength.jl` — three-scale upscaling of
  cement-paste / mortar elasticity and quasi-brittle compression
  strength following Pichler & Hellmich 2011 (SC + MT + MT). The single
  iso hydrate phase + global-μ autodiff approximation matches the
  effective moduli (k, μ, E) of the reference Python implementation to
  rtol ≈ 1e-3 across the (wc, α) grid.
- `scripts/29_porous_schemes.jl` — porous benchmark across all ten
  schemes (sphere and oblate ω = 0.2 with iso symmetrize). After the
  Hill-symmetric SC fix, spherical-pore SC percolates exactly at φ=0.5.
- `scripts/bench_echoes/benchmark_porous.jl`, `benchmark_pichler.jl`
  — PyCall cross-validation against the C++ reference; ten schemes ×
  two cases (sphere / oblate) for porous, six wc curves × twelve α
  points for Pichler. The moduli match the reference to rtol_mod ≈ 1e-3
  across both benchmarks.

#### Tests

Three new cross-cutting test files:

- `test_parameters.jl` (round-trip, type-promotion, no-mutation
  invariants on every lens kind),
- `test_sensitivities.jl` (FD vs autodiff cross-check on every scheme,
  closed-form Christensen 1990 match to `~1e-12`, closure fallback,
  `MyBlob` user-inclusion demonstration),
- `test_symmetrize.jl` (round-trip iso/TI projections on 2nd- and
  4th-order tensors, TI(ez) coaxial preservation, integration with
  `homogenize`).

Total: 3421 tests pass.

#### Breaking changes

- **SC results differ for systems near percolation** because of the
  Hill-symmetric SC fix. The pre-v0.4 SC step treated the matrix as
  Mori-Tanaka-style (A = I) and therefore selected the upper branch
  unconditionally. The new step is the textbook Hill / Budiansky 1965
  symmetric SC. Users who relied on the old upper-branch behavior for
  porous-sphere systems should switch to `MoriTanaka` (which is also
  not broken by the fix).
- **`homogenize` API** keeps the `homogenize(rve, scheme; property=:C)`
  kwarg form for backward compatibility but the recommended signature
  is now `homogenize(rve, scheme, property::Symbol)` with `property`
  required and positional.

#### Notes

- `Complex{T}` autodiff is not supported (ForwardDiff does not mix Dual
  and Complex cleanly). Symbolic differentiation goes through SymPy on
  the closed-form schemes directly (already supported).
- The `AsymmetricSelfConsistent` scheme follows the symmetric-SC fixed
  point in compliance space when matrix is stiff. The C++ reference's
  ASC uses a different formulation (compliance-form Mori-Tanaka with
  iterating reference) which converges to a different branch for
  porous oblate systems away from percolation; this is documented in
  `scripts/bench_echoes/benchmark_porous.jl`.

### v0.3.0 — RVE container + 10 homogenization schemes

New `MeanFieldHomogenization.Schemes` sub-module: a Representative Volume Element
container plus the ten classical mean-field homogenization schemes ported
from C++ ECHOES, with a few Julia-idiomatic improvements.

#### Additions

- **`RVE`** container with `add_matrix!`, `add_phase!`, helpers
  (`matrix_phase`, `inclusion_phase_names`, `phase_property`,
  `volume_fraction`, `crack_density`, `matrix_volume_fraction`,
  `validate_rve`).  Volume fractions are stored at the RVE level rather
  than on the inclusions — a single inclusion remains usable for
  localization-tensor calculations without any RVE machinery.
- **`AbstractAmount`** hierarchy with `VolumeFraction` (solid inclusions)
  and `CrackDensity` (flat cracks); the matrix amount is implicit
  (`1 − Σ f_inc`) and crack densities are excluded from that sum.
- **`AbstractDistributionShape`** hierarchy with `UniformDistribution`
  (single outer envelope, current behavior); leaves an extension hook
  for a future `PairwiseDistribution` (Willis 1982) without breaking
  the public API.
- **Ten homogenization schemes**: `Voigt`, `Reuss`, `Dilute`,
  `DiluteDual`, `MoriTanaka`, `Maxwell`, `PonteCastanedaWillis`,
  `SelfConsistent`, `AsymmetricSelfConsistent`, `DifferentialScheme`.
- **`homogenize(rve, scheme; property=:C)`** central entry point.  The
  scheme can be a type instance (`MoriTanaka()`,
  `SelfConsistent(algorithm=NewtonRaphson(), abstol=1e-12)`) or a
  `Symbol` shortcut. Canonical Symbol aliases are lowercase
  (`:mt`, `:sc`, `:diff`, …) for consistency with the algorithm-method
  symbols (`:auto`, `:residues`, `:decuhr`); CamelCase and ECHOES
  upper-case codes (`:MT`, `:DIFF`, …) are kept as backwards-compatible
  aliases.
- **Differential trajectories**: `Proportional` (default), `Sequential`
  (phase-by-phase), `CustomPath` (per-phase explicit trajectory) — all
  validated for monotonicity and boundary conditions.
- **SciML weak extension** `MeanFieldHomogenizationNonlinearSolveExt` (active once
  `NonlinearSolve.jl` is loaded into the session) makes every algorithm
  of `NonlinearSolve.jl` available to `SelfConsistent` /
  `AsymmetricSelfConsistent` via the `algorithm` keyword, through a
  `ForwardDiff`-safe implicit-function-theorem lift (no nested `Dual`s,
  regardless of the chosen algorithm). The auto-resolving
  `AutoNonlinear` marker selects a globalized SciML algorithm
  (`TrustRegion`) when the extension is active and falls back to the
  built-in `NewtonDefault` otherwise — it is not the default of
  `SelfConsistent` / `AsymmetricSelfConsistent` (which remains
  `AndersonDefault`, more robust through the porous-percolation
  bifurcation). See the
  [nonlinear solvers tutorial](docs/src/tutorials/12_nonlinear_solvers.md).
- **Conductivity (`property = :K`)** is supported by every scheme
  through 2nd-order tensor algebra (gradient-gradient localization,
  resistivity contributions for cracks).

#### Number-type compatibility

Every new scheme is fully `ForwardDiff.Dual` and `Complex{Float64}`
compatible (frequency-domain viscoelasticity); symbolic `Sym` / `Num`
work on the closed-form schemes (Voigt, Reuss, Dilute, DiluteDual,
Mori-Tanaka, Maxwell, PCW). The asymmetric SC heuristic uses the
Inf-norm rather than the SVD-based 2-norm so it works seamlessly under
`Dual`.

#### Documentation

New theory page `theory/homogenization.md`, manual page
`manual/schemes.md`, API page `api/schemes.md`. Bibliography augmented
with `mori1973`, `christensen1990`, `mclaughlin1977`, `norris1985`,
`ponte1995`, `willis1982`. New scripts `scripts/20_voigt_reuss_bounds.jl`
through `scripts/25_echoes_crosscheck.jl`. The latter cross-validates
Mori-Tanaka against the [Christensen 1990](@cite christensen1990) closed
form (exact match to 6 sig. figs. on bulk and shear at five fractions).

#### Tests

Around 270 new tests covering construction, numerical bounds, closed
forms, Dual sensitivity (every scheme), Complex moduli sweep, Symbol
shortcuts, and CustomPath validation. Total 3312 tests passing.

### v0.2.0 — alignment with TensND 0.2 (breaking)

Follow-up to TensND 0.2's API unification. MeanFieldHomogenization is iso-functional —
all outputs are unchanged — but every mention of a TensND symbol now uses
the new snake_case + UPPERCASE-acronym convention.

#### Breaking changes

- `TensND.TensWalpole` references (type annotations, dispatch rules,
  constructor calls) now use `TensND.TensTI{4}`.  The struct layout is
  identical so numerical behavior is unchanged.
- Accessor renames propagated from TensND: `getbasis` → `get_basis`,
  `tensbasis` → `tens_basis`, `invKM` → `inv_KM`, `getdata` → `get_data`,
  `getarray` → `get_array`, `getvar` → `get_var`, `getdim` → `get_dim`,
  `getorder` → `get_order`.
- Predicate renames: `isISO` → `is_ISO`, `isTI` → `is_TI`,
  `isOrtho` → `is_ORTHO`.
- Tensor factory renames in scripts and docs: `tensId2` → `tens_Id2`,
  `tensJ4` → `tens_J4`, `tensTI` → `tens_TI`, etc.

#### Additions

None — functional surface unchanged.

#### Migration guide

If you have your own code depending on MeanFieldHomogenization dispatch, apply the
same renames as listed in TensND's v0.2 changelog. All MeanFieldHomogenization tests
(2865) pass without behavioral change after migration.
