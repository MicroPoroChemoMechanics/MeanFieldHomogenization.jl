"""A MeanFieldHomogenization material as the stress of a dolfinx problem.

:class:`Stress` builds the stress as a ``FEMExternalOperator`` of
dolfinx-external-operator acting on the strain of a displacement field. The
residual is written with it in UFL as with any stress; its derivative, the
consistent tangent, is the operator's derivative. At each Newton iteration
:meth:`Stress.update` evaluates the strains at every quadrature point and makes
one Julia call that computes the stresses and the tangents together; the
tangent the Jacobian asks for is the one of that call, not a second integration.

In two dimensions the problem is plane strain: the material is integrated in
three dimensions with ε₁₃ = ε₂₃ = ε₃₃ = 0, and the in-plane components of the
stress and of the tangent enter the problem. The out-of-plane stress σ₃₃ is
kept in :attr:`Material.stress` (Mandel slot 2).
"""

import basix.ufl
import numpy as np
import ufl
from dolfinx_external_operator import (
    FEMExternalOperator,
    evaluate_external_operators,
    evaluate_operands,
    replace_external_operators,
)

from . import Material, _SLOT, _WEIGHTS, julia

__all__ = ["Newton", "NewtonError", "Poroelastic", "Stress", "quasistatic", "strain"]


def strain(u):
    """The small strain ε(u) = sym ∇u."""
    return ufl.sym(ufl.grad(u))


def _mandel_maps(d):
    """Linear maps between the ``d × d`` components and the Mandel components.

    ``to`` (6, d²) takes the row-major components of a symmetric tensor to its
    Mandel vector (the slots outside the plane stay zero); ``back`` (d², 6)
    takes a Mandel vector to the in-plane components.
    """
    to = np.zeros((6, d * d))
    back = np.zeros((d * d, 6))
    for i in range(d):
        for j in range(d):
            a = _SLOT[i, j]
            # Each off-diagonal component is met twice: (i, j) and (j, i).
            to[a, i * d + j] = _WEIGHTS[a] if i == j else _WEIGHTS[a] / 2
            back[i * d + j, a] = 1.0 / _WEIGHTS[a]
    return to, back


class Stress:
    """The stress of ``material`` on the strain of the displacement ``u``.

    ``material`` is a Julia material or Julia code that builds one;
    ``quadrature_degree`` sets the quadrature of the stress space, which the
    measure of the forms must match (``ufl.dx(metadata={"quadrature_degree": q})``,
    or :attr:`dx`). :attr:`sigma` is the stress to write the residual with;
    :attr:`dt` the time step the next :meth:`update` integrates over.
    """

    def __init__(self, material, u, quadrature_degree=2, dt=1.0, **options):
        mesh = u.function_space.mesh
        self.dim = d = mesh.geometry.dim
        if d not in (2, 3):
            raise ValueError(f"Stress: a mesh of dimension 2 or 3 is expected; got {d}")
        tdim = mesh.topology.dim
        imap = mesh.topology.index_map(tdim)
        ncells = imap.size_local + imap.num_ghosts
        element = basix.ufl.quadrature_element(mesh.basix_cell(), value_shape=(d, d), degree=quadrature_degree)
        from dolfinx import fem

        self.space = fem.functionspace(mesh, element)
        nqp = self.space.element.interpolation_points.shape[0]
        self.material = Material(material, ncells, nqp)
        if (self.material.ngradients, self.material.nfluxes) != (6, 6):
            raise ValueError(
                "Stress: a mechanical material (strain to stress) is expected; this one has "
                f"{self.material.ngradients} gradient and {self.material.nfluxes} flux components: "
                "a poroelastic material goes to Poroelastic"
            )
        self.dt = dt
        self.options = options
        self.dx = ufl.Measure("dx", domain=mesh, metadata={"quadrature_degree": quadrature_degree})
        self._to, self._back = _mandel_maps(d)
        self._strain_seen = None
        self.sigma = FEMExternalOperator(strain(u), function_space=self.space, external_function=self._external)

    def forms(self, residual, u):
        """The residual form, written with :attr:`sigma`, and its Jacobian, with the operator replaced.

        Returns ``(F, J, operators)``: the forms to compile with ``fem.form``
        and the external operators :meth:`update` evaluates.
        """
        J = ufl.derivative(residual, u, ufl.TrialFunction(u.function_space))
        J_expanded = ufl.algorithms.expand_derivatives(J)
        F_replaced, F_ops = replace_external_operators(residual)
        J_replaced, J_ops = replace_external_operators(J_expanded)
        return F_replaced, J_replaced, F_ops + J_ops

    def update(self, operators):
        """Evaluate the strains and the operators of ``operators`` (one Julia call)."""
        evaluate_external_operators(operators, evaluate_operands(operators))

    def commit(self):
        """Make the states of the last update the current ones."""
        self.material.commit()
        self._strain_seen = None

    def rollback(self):
        """Discard the states of the last update."""
        self.material.rollback()
        self._strain_seen = None

    # The external function: a derivative multi-index to a callable on the
    # evaluated operand (cells × points·d²).
    def _external(self, derivatives):
        if derivatives == (0,):
            return self._stress_values
        if derivatives == (1,):
            return self._tangent_values
        raise NotImplementedError(f"Stress: derivative {derivatives} of the stress is not available")

    def _integrate(self, eps):
        # The stress and the tangent come from one call; the Jacobian evaluated
        # at the same strains, step and states reads the tangent of that call.
        seen = self._strain_seen
        if seen is None or seen[0] != self.dt or not np.array_equal(seen[1], eps):
            d2 = self.dim * self.dim
            mandel = self._to @ eps.reshape(-1, d2).T
            self.material.update(mandel, self.dt, **self.options)
            self._strain_seen = (self.dt, eps.copy())
        return self.material.stress, self.material.tangent

    def _stress_values(self, eps):
        sigma, _ = self._integrate(eps)
        return (self._back @ sigma).T.ravel()

    def _tangent_values(self, eps):
        _, tangent = self._integrate(eps)
        return np.einsum("Ia,Jb,abn->nIJ", self._back, self._back, tangent, optimize=True).ravel()


class Poroelastic:
    """A poroelastic material on the displacement ``u`` and the pore pressure ``p``.

    ``material`` has the gradients ``(ε, p)`` and the fluxes ``(σ, φ)`` of a
    poroelastic law of MeanFieldHomogenization (``BiotPoroelasticity``,
    ``ClosingCrackPoroelasticity``): the total stress and the fluid content per
    unit volume. :attr:`sigma` and :attr:`phi` are two external operators on the
    same operands, ``ε(u)`` and ``p``, whose values and derivatives come from one
    Julia call; :attr:`phi_old` holds the fluid content of the last committed
    step, for the storage term ``(φ − φ_old) / Δt`` of the balance of the fluid.
    ``u`` and ``p`` may be the parts of a mixed function (``ufl.split``). In two
    dimensions the problem is plane strain.
    """

    def __init__(self, material, u, p, quadrature_degree=2, dt=1.0, **options):
        from dolfinx import fem

        mesh = _dolfinx_mesh(p)
        self.dim = d = mesh.geometry.dim
        if d not in (2, 3):
            raise ValueError(f"Poroelastic: a mesh of dimension 2 or 3 is expected; got {d}")
        tdim = mesh.topology.dim
        imap = mesh.topology.index_map(tdim)
        ncells = imap.size_local + imap.num_ghosts
        cell = mesh.basix_cell()
        tensor = basix.ufl.quadrature_element(cell, value_shape=(d, d), degree=quadrature_degree)
        scalar = basix.ufl.quadrature_element(cell, value_shape=(), degree=quadrature_degree)
        self.space_sigma = fem.functionspace(mesh, tensor)
        self.space_phi = fem.functionspace(mesh, scalar)
        nqp = self.space_phi.element.interpolation_points.shape[0]
        self.material = Material(material, ncells, nqp)
        if (self.material.ngradients, self.material.nfluxes) != (7, 7):
            raise ValueError(
                "Poroelastic: a material of gradients (ε, p) and fluxes (σ, φ) is expected; this one has "
                f"{self.material.ngradients} gradient and {self.material.nfluxes} flux components"
            )
        self.dt = dt
        self.options = options
        self.dx = ufl.Measure("dx", domain=mesh, metadata={"quadrature_degree": quadrature_degree})
        self._to, self._back = _mandel_maps(d)
        self._seen = None
        self.sigma = FEMExternalOperator(strain(u), p, function_space=self.space_sigma, external_function=self._external_sigma)
        self.phi = FEMExternalOperator(strain(u), p, function_space=self.space_phi, external_function=self._external_phi)
        self.phi_old = fem.Function(self.space_phi)
        self._store_phi_old()

    forms = Stress.forms
    update = Stress.update

    def commit(self):
        """Make the states of the last update the current ones, φ_old with them."""
        self.material.commit()
        self._store_phi_old()
        self._seen = None

    def rollback(self):
        """Discard the states of the last update."""
        self.material.rollback()
        self._seen = None

    def _store_phi_old(self):
        # The committed fluid content, from a step of the states as they are
        # (no strain, no pressure change, no time): the fluxes of the states.
        n = self.space_phi.dofmap.index_map.size_local + self.space_phi.dofmap.index_map.num_ghosts
        self.phi_old.x.array[:] = julia()._mfh_fenicsx_fluid_content(self.material.states, self.material.material)[:n]

    def _integrate(self, eps, p):
        seen = self._seen
        if seen is None or seen[0] != self.dt or not (np.array_equal(seen[1], eps) and np.array_equal(seen[2], p)):
            d2 = self.dim * self.dim
            gradients = np.vstack([self._to @ eps.reshape(-1, d2).T, p.reshape(1, -1)])
            self.material.update(gradients, self.dt, **self.options)
            self._seen = (self.dt, eps.copy(), p.copy())
        return self.material.fluxes, self.material.tangent

    # Derivative multi-indices on the operands (ε, p).
    def _external_sigma(self, derivatives):
        def values(eps, p):
            F, K = self._integrate(eps, p)
            if derivatives == (0, 0):
                return (self._back @ F[:6]).T.ravel()
            if derivatives == (1, 0):
                return np.einsum("Ia,Jb,abn->nIJ", self._back, self._back, K[:6, :6], optimize=True).ravel()
            if derivatives == (0, 1):
                return (self._back @ K[:6, 6]).T.ravel()
            raise NotImplementedError(f"Poroelastic: derivative {derivatives} of the stress is not available")

        return values

    def _external_phi(self, derivatives):
        def values(eps, p):
            F, K = self._integrate(eps, p)
            if derivatives == (0, 0):
                return F[6].copy()
            if derivatives == (1, 0):
                return (self._back @ K[6, :6]).T.ravel()
            if derivatives == (0, 1):
                return K[6, 6].copy()
            raise NotImplementedError(f"Poroelastic: derivative {derivatives} of the fluid content is not available")

        return values


def _dolfinx_mesh(expr):
    """The dolfinx mesh of a UFL expression on one domain."""
    from dolfinx.mesh import Mesh

    domain = ufl.domain.extract_unique_domain(expr)
    return Mesh(domain.ufl_cargo(), domain)


class NewtonError(RuntimeError):
    """Newton's method did not converge in the iterations allowed."""


class Newton:
    """Newton's method on the residual ``residual(u; v)`` written with ``stress.sigma``.

    The forms are compiled once; :meth:`solve` runs one load step from the
    current ``u``, the Dirichlet conditions ``bcs`` taking their current values,
    and returns the number of iterations. Convergence is read on the residual,
    the rows of the Dirichlet conditions removed: below ``atol``, or below
    ``rtol`` times the first one. The step is not committed: call
    ``stress.commit()`` once it is accepted, ``stress.rollback()`` otherwise.
    :attr:`history` holds the residual norms of the last step.
    """

    def __init__(self, stress, residual, u, bcs, rtol=1.0e-10, atol=1.0e-10, maxiter=25, petsc_options=None):
        from dolfinx import fem
        from petsc4py import PETSc

        F, J, self.operators = stress.forms(residual, u)
        self.F, self.J = fem.form(F), fem.form(J)
        self.stress, self.u, self.bcs = stress, u, bcs
        self.rtol, self.atol, self.maxiter = rtol, atol, maxiter
        self.du = fem.Function(u.function_space)
        # A direct solve by default, MUMPS when PETSc has it (PETSc's own LU
        # made a plastic step on 28,000 dofs of quadratic tetrahedra nine times
        # longer); `petsc_options` override it, under a prefix of this solver's own.
        self.ksp = PETSc.KSP().create(u.function_space.mesh.comm)
        self.ksp.setType("preonly")
        self.ksp.getPC().setType("lu")
        if PETSc.Sys.hasExternalPackage("mumps"):
            self.ksp.getPC().setFactorSolverType("mumps")
        prefix = f"mfh_newton_{id(self)}_"
        self.ksp.setOptionsPrefix(prefix)
        options = PETSc.Options()
        for key, value in (petsc_options or {}).items():
            options[prefix + key] = value
        self.ksp.setFromOptions()

    def residual_norm(self):
        """The norm of the residual at the current ``u``, operators evaluated."""
        from dolfinx.fem.petsc import apply_lifting, assemble_vector, set_bc
        from petsc4py import PETSc

        self.stress.update(self.operators)
        b = assemble_vector(self.F)
        apply_lifting(b, [self.J], [self.bcs], x0=[self.u.x.petsc_vec], alpha=-1.0)
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        set_bc(b, self.bcs, self.u.x.petsc_vec, -1.0)
        return b, b.norm()

    def reaction(self, dofs):
        """The sum of the residual over ``dofs`` at the current state.

        On the dofs of a Dirichlet condition that carries no external force, it
        is the reaction there: the force the condition exerts, summed over the
        processes. Call it after :meth:`solve`, whose last residual evaluated the
        stresses at the converged state.
        """
        from dolfinx.fem.petsc import assemble_vector
        from petsc4py import PETSc

        b = assemble_vector(self.F)
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        index_map = self.u.function_space.dofmap.index_map
        owned = np.asarray(dofs)[np.asarray(dofs) < index_map.size_local * self.u.function_space.dofmap.index_map_bs]
        local = float(np.sum(b.array_r[owned]))
        b.destroy()
        return self.u.function_space.mesh.comm.allreduce(local)

    def solve(self):
        """One step: Newton iterations from the current ``u``; the number of iterations."""
        from dolfinx.fem.petsc import assemble_matrix
        from petsc4py import PETSc

        self.history = []
        for iteration in range(self.maxiter + 1):
            b, norm = self.residual_norm()
            self.history.append(norm)
            first = self.history[0]
            if norm <= self.atol or norm <= self.rtol * first and iteration > 0:
                b.destroy()
                return iteration
            if iteration == self.maxiter:
                break
            A = assemble_matrix(self.J, self.bcs)
            A.assemble()
            self.ksp.setOperators(A)
            self.ksp.solve(b, self.du.x.petsc_vec)
            self.du.x.scatter_forward()
            self.u.x.array[:] -= self.du.x.array
            A.destroy()
            b.destroy()
        b.destroy()
        raise NewtonError(f"Newton: no convergence in {self.maxiter} iterations (residual {norm:.3e}, first {first:.3e})")


def _integration_failed(err):
    """Whether a Julia error is a point that failed to integrate."""
    from . import julia

    jl = julia()
    return bool(jl.isa(err.exception, jl.LocalIntegrationError))


def quasistatic(newton, times, prescribe, on_step=None, max_cuts=8):
    """Run the load steps between consecutive ``times``, cutting a step that fails.

    ``prescribe(t)`` sets the boundary values at the time ``t``; ``on_step(t)``
    is called after each accepted step. A step whose Newton iterations do not
    converge, or where a point fails to integrate, is rolled back and halved, up
    to ``max_cuts`` times. Returns the accepted times.
    """
    from juliacall import JuliaError

    stress, u = newton.stress, newton.u
    accepted = [times[0]]

    def advance(t0, t1, depth):
        u0 = u.x.array.copy()
        stress.dt = t1 - t0
        prescribe(t1)
        try:
            newton.solve()
        except (NewtonError, JuliaError) as err:
            if isinstance(err, JuliaError) and not _integration_failed(err) or depth == max_cuts:
                raise
            u.x.array[:] = u0
            stress.rollback()
            tm = (t0 + t1) / 2
            advance(t0, tm, depth + 1)
            advance(tm, t1, depth + 1)
            return
        stress.commit()
        accepted.append(t1)
        if on_step is not None:
            on_step(t1)

    for t0, t1 in zip(times[:-1], times[1:]):
        advance(t0, t1, 0)
    return accepted
