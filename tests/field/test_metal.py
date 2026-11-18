import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import firm3dpp

from firm3d.catapult.field import CatapultBoozerField, CatapultCartesianField
from firm3d.catapult.tracing import (
    save_trajectories_boozer_gpu,
    trace_particles_boozer_gpu,
    trace_particles_cartesian_gpu,
)
from firm3d.field.boozermagneticfield import (
    BoozerAnalytic,
    BoozerRadialInterpolant,
    BoozerSplineField,
    InterpolatedBoozerField,
)
from firm3d.field.tracing import (
    MaxToroidalFluxStoppingCriterion,
    trace_particles_boozer,
)
from firm3d.trajectory_helpers import compute_peta
from firm3d.util.constants import (
    ALPHA_PARTICLE_CHARGE as CHARGE,
)
from firm3d.util.constants import (
    ALPHA_PARTICLE_MASS as MASS,
)
from firm3d.util.constants import (
    FUSION_ALPHA_PARTICLE_ENERGY as ENERGY,
)
from tests.field.test_gpu import (
    CATAPULTField,
    cartesian_rhs,
    final_states,
    get_field,
    sample_test_points,
)

HAS_METAL = getattr(firm3dpp, "gpu_backend", None) == "metal"
n_test_pts = 2000


def assert_tracing_matches_cpu(
    test: unittest.TestCase,
    field: InterpolatedBoozerField,
    res: int,
    stz: np.ndarray,
    vpar: np.ndarray,
) -> None:
    """
    Final states of Metal single precision vs the CPU double tracer: the
    median and 90th percentile differences per column stay within a
    factor of the CPU tracer's own spread between tolerances 1e-8 and
    1e-9, which measures the orbits' sensitivity to any perturbation.

    The kernel stops at the first step boundary at or after tmax
    (docs/gpu_tracing.rst) while the CPU tracer's dense output lands
    exactly on tmax, so each particle is traced on the CPU to the time
    the kernel actually reached, for a comparison at matched times.
    """
    tmax = 1e-6

    cfield = CatapultBoozerField(field, res, res, res, precision="single")
    res_tys, _ = trace_particles_boozer_gpu(
        cfield,
        stz,
        vpar,
        tmax=tmax,
        mass=MASS,
        charge=CHARGE,
        Ekin=ENERGY,
        tol=1e-8,
        forget_exact_path=True,
    )
    out_gpu = final_states(res_tys)
    survived = out_gpu[:, 0] >= 0.999 * tmax
    test.assertTrue(survived.any())

    def run_cpu(i, tol):
        res_tys, _ = trace_particles_boozer(
            field,
            stz[i : i + 1],
            vpar[i : i + 1],
            tmax=float(out_gpu[i, 0]),
            mass=MASS,
            charge=CHARGE,
            Ekin=ENERGY,
            abstol=tol,
            reltol=tol,
            stopping_criteria=[MaxToroidalFluxStoppingCriterion(1.0)],
            forget_exact_path=True,
        )
        return final_states(res_tys)[0]

    idx = np.where(survived)[0]
    out_cpu = np.array([run_cpu(i, 1e-8) for i in idx])
    reference = np.array([run_cpu(i, 1e-9) for i in idx])

    columns = [1, 2, 4]
    diff = out_gpu[idx][:, columns] - out_cpu[:, columns]
    ref = reference[:, columns] - out_cpu[:, columns]
    test.assertTrue(np.any(diff != 0), "Metal single matched CPU double exactly")
    factor = 10.0
    for c in range(diff.shape[1]):
        for q in (50, 90):
            got = np.percentile(np.abs(diff[:, c]), q)
            expected = np.percentile(np.abs(ref[:, c]), q)
            test.assertLessEqual(
                got,
                factor * expected + 1e-12,
                f"column {columns[c]}, p{q}: {got:.3e} exceeds "
                f"{factor} x {expected:.3e}",
            )


@unittest.skipUnless(HAS_METAL, "Metal backend not available")
class TestMetalTracingBoozerVacuum(unittest.TestCase):
    """
    The Metal backend against the CPU reference, in single precision: the
    only precision the backend has, since Apple GPUs lack FP64. The
    acceptance levels follow docs/gpu_tracing.rst: the field and its
    derivatives agree with double precision to about 1e-5 relative, and
    traced states differ at the orbits' own sensitivity level.
    """

    vacuum = True

    def setUp(self):
        self.n_metagrid_pts = 15
        self.filename = "examples/inputs/boozmn_aten_rescaled_low_res.nc"
        self.bri, self.field, self.nfp = get_field(
            self.filename, self.n_metagrid_pts, self.vacuum
        )
        self.stz = sample_test_points(n_test_pts)

        self.VELOCITY = np.sqrt(2 * ENERGY / MASS)
        self.vpar_init = np.random.default_rng(1).uniform(
            -self.VELOCITY, self.VELOCITY, n_test_pts
        )

        self.field = CATAPULTField(
            self.field,
            ns=self.n_metagrid_pts,
            ntheta=self.n_metagrid_pts,
            nzeta=self.n_metagrid_pts,
            nfp=self.nfp,
        )

    def test_interpolant(self):
        # column-scale normalized, as in CATAPULTField.test_precision; CUDA
        # measures single-vs-double at 2e-6 to 1e-5 on an A100
        f32 = self.field.compute_gpu_interpolant(self.stz, dtype=np.float32)
        self.assertEqual(f32.dtype, np.float32)
        cpu = self.field.compute_cpu_interpolant(self.stz)
        scale = np.max(np.abs(cpu), axis=0) + 1e-16
        error = np.abs(cpu - f32.astype(np.float64)) / scale
        self.assertLess(error.max(), 1e-4)

    def test_derivatives(self):
        f32 = self.field.compute_gpu_derivatives(
            self.stz, self.vpar_init, self.VELOCITY, dtype=np.float32
        )
        self.assertEqual(f32.dtype, np.float32)
        cpu = self.field.compute_cpu_derivatives(
            self.stz, self.vpar_init, self.VELOCITY
        )
        scale = np.max(np.abs(cpu), axis=0) + 1e-16
        error = np.abs(cpu - f32.astype(np.float64)) / scale
        self.assertLess(error.max(), 1e-3)

    def test_double_precision_raises(self):
        # the double bindings exist but must refuse with a clear error rather
        # than silently downcast
        field = self.field.field
        res = self.n_metagrid_pts
        inside = self.stz[:, 0] < 1.0
        stz = self.stz[inside][:8]
        vpar = self.vpar_init[inside][:8]
        cfield = CatapultBoozerField(field, res, res, res, precision="double")
        with self.assertRaisesRegex(RuntimeError, "single precision"):
            trace_particles_boozer_gpu(
                cfield,
                stz,
                vpar,
                tmax=1e-8,
                mass=MASS,
                charge=CHARGE,
                Ekin=ENERGY,
                tol=1e-8,
                forget_exact_path=True,
            )

    def test_tracing_against_cpu(self):
        inside = self.stz[:, 0] < 1.0
        assert_tracing_matches_cpu(
            self,
            self.field.field,
            self.n_metagrid_pts,
            self.stz[inside][:64],
            self.vpar_init[inside][:64],
        )

    def test_trajectory_saving_consistent(self):
        """
        Chunked trajectory saving re-enters the kernel with dt and mu fed
        back, exercising the cross-dispatch first-same-as-last state; a
        single-chunk save must reproduce the plain trace exactly.
        """
        field = self.field.field
        res = self.n_metagrid_pts
        inside = self.stz[:, 0] < 1.0
        stz = self.stz[inside][:64]
        vpar = self.vpar_init[inside][:64]
        tmax = 1e-6

        cfield = CatapultBoozerField(field, res, res, res, precision="single")
        res_tys, _ = trace_particles_boozer_gpu(
            cfield,
            stz,
            vpar,
            tmax=tmax,
            mass=MASS,
            charge=CHARGE,
            Ekin=ENERGY,
            tol=1e-8,
            forget_exact_path=True,
        )
        traced = np.array([traj[-1] for traj in res_tys])

        saved = np.array(
            [
                traj[-1]
                for traj in save_trajectories_boozer_gpu(
                    cfield, stz, vpar, tmax, tmax, MASS, CHARGE, self.VELOCITY, 1e-8
                )
            ]
        )
        np.testing.assert_allclose(saved[:, :5], traced, rtol=1e-6, atol=1e-9)


@unittest.skipUnless(HAS_METAL, "Metal backend not available")
class TestMetalTracingBoozerFiniteBeta(TestMetalTracingBoozerVacuum):
    """
    All checks of the vacuum class, re-run against the finite-beta
    right-hand side: the kernel then reads the wider interpolant (I, K and
    their derivatives) and the wrapper selects RHS_BOOZER instead of
    RHS_BOOZER_VACUUM, a code path no vacuum test touches.
    """

    vacuum = False

    def test_selects_finite_beta_rhs(self):
        # guard against this class silently re-testing the vacuum path
        self.assertEqual(self.field.field_type, "boozer")
        res = self.n_metagrid_pts
        cfield = CatapultBoozerField(
            self.field.field, res, res, res, precision="single"
        )
        self.assertFalse(cfield.vacuum)


@unittest.skipUnless(HAS_METAL, "Metal backend not available")
class TestMetalTracingBoozerNoK(unittest.TestCase):
    """
    A no-K field (finite beta, K dropped) has no K to evaluate, so its table
    carries K = 0 and the kernel runs the finite-beta right-hand side, which
    with K = 0 is exactly the CPU's no-K equations (gc_noK). This is the
    field type of upstream's p_eta conservation example.
    """

    def setUp(self):
        self.res = 15
        self.bri = BoozerRadialInterpolant(
            "examples/inputs/boozmn_aten_rescaled_low_res.nc", 3, no_K=True
        )
        self.field = InterpolatedBoozerField(
            self.bri,
            3,
            ns_interp=self.res,
            ntheta_interp=self.res,
            nzeta_interp=self.res,
        )
        self.stz = sample_test_points(n_test_pts)
        velocity = np.sqrt(2 * ENERGY / MASS)
        self.vpar = np.random.default_rng(1).uniform(-velocity, velocity, n_test_pts)

    def test_tabulates_with_finite_beta_rhs(self):
        res = self.res
        cfield = CatapultBoozerField(self.field, res, res, res, precision="single")
        self.assertEqual(cfield.field_type, "nok")
        self.assertFalse(cfield.vacuum)

    def test_tracing_against_cpu(self):
        inside = self.stz[:, 0] < 1.0
        assert_tracing_matches_cpu(
            self, self.field, self.res, self.stz[inside][:64], self.vpar[inside][:64]
        )


@unittest.skipUnless(HAS_METAL, "Metal backend not available")
class TestMetalTracingBoozerSplineNoK(unittest.TestCase):
    """
    The conservation benchmark's field: a quasi-symmetric BoozerSplineField
    of type "nok", tabulated straight from its splines rather than through
    an InterpolatedBoozerField. In an exactly quasi-symmetric field p_eta is
    conserved, so its drift, measured on the exact field, is the tracers'
    numerical error. As in the benchmark, the CPU traces the interpolant of
    the same field, and the two median drifts must agree; they agree to
    within 10% here, and the benchmark's own tripwire is a factor of 10.
    """

    HELICITY_M, HELICITY_N = 1, -4
    FACTOR = 2.0

    def setUp(self):
        self.res = 8
        # a coarse spline grid keeps setup to seconds; it breaks the
        # symmetry slightly, which both tracers see alike
        self.spline = BoozerSplineField(
            "examples/inputs/boozmn_beta2.5_QH.nc",
            ntheta=16,
            nzeta=16,
            helicity_M=self.HELICITY_M,
            helicity_N=self.HELICITY_N,
        )
        stz = sample_test_points(128)
        # away from the edge, so that the markers survive to tmax
        self.stz = np.ascontiguousarray(stz[stz[:, 0] < 0.9][:32])
        velocity = np.sqrt(2 * ENERGY / MASS)
        self.vpar = np.random.default_rng(1).uniform(-velocity, velocity, 32)
        self.tmax = 1e-4

    def peta(self, points: np.ndarray, vpar: np.ndarray) -> np.ndarray:
        return compute_peta(
            self.spline,
            np.ascontiguousarray(points),
            np.ascontiguousarray(vpar),
            MASS,
            CHARGE,
            self.HELICITY_M,
            self.HELICITY_N,
        )

    def test_tabulates_with_finite_beta_rhs(self):
        res = self.res
        cfield = CatapultBoozerField(self.spline, res, res, res, precision="single")
        self.assertEqual(cfield.field_type, "nok")
        self.assertFalse(cfield.vacuum)

    def test_peta_drift_matches_cpu(self):
        res, tol = self.res, 1e-6
        cfield = CatapultBoozerField(self.spline, res, res, res, precision="single")
        gpu, _ = trace_particles_boozer_gpu(
            cfield,
            self.stz,
            self.vpar,
            tmax=self.tmax,
            mass=MASS,
            charge=CHARGE,
            Ekin=ENERGY,
            tol=tol,
            forget_exact_path=True,
        )
        interp = InterpolatedBoozerField(
            self.spline,
            3,
            ns_interp=res,
            ntheta_interp=res,
            nzeta_interp=res,
            nfp=self.spline.nfp,
            stellsym=self.spline.stellsym,
        )
        cpu, _ = trace_particles_boozer(
            interp,
            self.stz,
            self.vpar,
            tmax=self.tmax,
            mass=MASS,
            charge=CHARGE,
            Ekin=ENERGY,
            abstol=tol,
            reltol=tol,
            stopping_criteria=[MaxToroidalFluxStoppingCriterion(1.0)],
            forget_exact_path=True,
        )

        peta0 = self.peta(self.stz, self.vpar)
        if self.spline.psi0 is None:
            self.fail("the spline field has no psi0, the scale of p_eta")
        scale = CHARGE * abs(self.spline.psi0)
        drifts = []
        for res_tys in (gpu, cpu):
            final = np.array([ty[-1] for ty in res_tys])
            self.assertTrue(
                np.all(final[:, 0] >= 0.999 * self.tmax), "a marker was lost"
            )
            drifts.append(np.abs(self.peta(final[:, 1:4], final[:, 4]) - peta0) / scale)
        d_gpu, d_cpu = (np.median(d) for d in drifts)
        self.assertGreater(d_cpu, 0.0)
        self.assertLessEqual(
            max(d_gpu / d_cpu, d_cpu / d_gpu),
            self.FACTOR,
            f"median p_eta drift: Metal {d_gpu:.3e}, CPU {d_cpu:.3e}",
        )


class AnalyticCylindricalField:
    """
    A smooth vector field B = (B_r, B_phi, B_z) on a cylindrical grid, with
    the interface the Cartesian table builder and cartesian_rhs read from a
    simsopt InterpolatedField. The 1/r toroidal part carries a perturbation
    of amplitude eps that is periodic in phi with nfp and stellarator
    symmetric (B_r odd, B_phi and B_z even under (phi, z) -> (-phi, -z)), so
    the kernel's folding of phi and z, and the signs it restores, are tested
    against the field itself. With eps = 0 the guiding-centre orbit is known
    exactly: r and v_par stay fixed, phi turns at v_par / r, and z drifts at
    m (v_par^2 + v_perp^2 / 2) / (q b0 r0).
    """

    H = 1e-6  # central-difference step for grad |B|, relative to r0, dimensionless

    def __init__(self, b0, r0, eps, nfp, r_range, z_range, cells):
        self.b0, self.r0, self.eps, self.nfp = b0, r0, eps, nfp
        self.kz = np.pi / z_range[1]  # vertical wavenumber of the perturbation [1/m]
        self.r_range = (r_range[0], r_range[1], cells)
        self.phi_range = (0.0, 2 * np.pi / nfp, cells)
        self.z_range = (z_range[0], z_range[1], cells)
        self.rphiz = np.zeros((0, 3))

    def set_points_cyl(self, rphiz):
        self.rphiz = np.asarray(rphiz, dtype=float).reshape(-1, 3)

    def _b_cyl(self, r, phi, z):
        wave = self.eps * self.b0
        return np.column_stack(
            (
                wave * np.sin(self.nfp * phi) * np.cos(self.kz * z),
                self.b0 * self.r0 / r
                + wave * np.cos(self.nfp * phi) * np.cos(self.kz * z),
                wave * np.sin(self.nfp * phi) * np.sin(self.kz * z),
            )
        )

    def _abs_b(self, r, phi, z):
        return np.linalg.norm(self._b_cyl(r, phi, z), axis=1)

    def B_cyl(self):
        return self._b_cyl(*self.rphiz.T)

    def GradAbsB_cyl(self):
        r, phi, z = self.rphiz.T
        h = self.H * self.r0
        return np.column_stack(
            (
                (self._abs_b(r + h, phi, z) - self._abs_b(r - h, phi, z)) / (2 * h),
                (self._abs_b(r, phi + h / r, z) - self._abs_b(r, phi - h / r, z))
                / (2 * h),
                (self._abs_b(r, phi, z + h) - self._abs_b(r, phi, z - h)) / (2 * h),
            )
        )

    def _to_cartesian(self, v):
        phi = self.rphiz[:, 1]
        c, s = np.cos(phi), np.sin(phi)
        return np.column_stack(
            (c * v[:, 0] - s * v[:, 1], s * v[:, 0] + c * v[:, 1], v[:, 2])
        )

    def B(self):
        return self._to_cartesian(self.B_cyl())

    def GradAbsB(self):
        return self._to_cartesian(self.GradAbsB_cyl())


class TorusClassifier:
    """
    Stands in for a simsopt SurfaceClassifier: positive inside the torus of
    major radius r0 and minor radius a, negative outside. The quadratic
    a^2 - (r - r0)^2 - z^2 is smooth, so the tricubic table holds it exactly.
    """

    def __init__(self, r0, a):
        self.r0, self.a = r0, a

    def evaluate_rphiz(self, rphiz):
        r, z = rphiz[:, 0], rphiz[:, 2]
        return (self.a**2 - (r - self.r0) ** 2 - z**2)[:, None]


@unittest.skipUnless(HAS_METAL, "Metal backend not available")
class TestMetalTracingCartesianVacuum(unittest.TestCase):
    """
    The Cartesian kernels against analytic fields, since the CPU Cartesian
    tracer and coil fields come from simsopt: interpolation and the RHS in a
    perturbed, stellarator-symmetric field, and whole traces in a purely
    toroidal one, whose orbits and loss times are known in closed form.
    """

    B0, R0, NFP, MINOR = 5.0, 10.0, 5, 1.5  # [T], [m], dimensionless, [m]
    # interpolation cells per dimension: the perturbation's sin(nfp phi) needs
    # 16 for the table to resolve it to ~1e-5; the 1/r field of the traces 8
    CELLS_PERTURBED, CELLS_TOROIDAL = 16, 8

    def setUp(self):
        self.velocity = float(np.sqrt(2 * ENERGY / MASS))
        self.rng = np.random.default_rng(4)
        self.classifier = TorusClassifier(self.R0, self.MINOR)

    def field(self, eps, cells):
        """The analytic field and its single precision table."""
        field = AnalyticCylindricalField(
            self.B0, self.R0, eps, self.NFP, (8.0, 12.0), (0.0, 3.0), cells
        )
        return field, CatapultCartesianField(field, self.classifier, precision="single")

    def points(self, n, half_width_r, half_height):
        """(n, 3) (r, phi, z) around the torus axis, all toroidal angles."""
        return np.column_stack(
            (
                self.R0 + self.rng.uniform(-half_width_r, half_width_r, n),
                self.rng.uniform(0.0, 2 * np.pi, n),
                self.rng.uniform(-half_height, half_height, n),
            )
        )

    def test_interpolant(self):
        field, cfield = self.field(0.05, self.CELLS_PERTURBED)
        rphiz = self.points(2000, 1.5, 2.5)  # both signs of z, every phi period
        out = firm3dpp.test_gpu_interpolation(
            cfield.quad_info,
            cfield.rrange,
            cfield.phirange,
            cfield.zrange,
            np.array(rphiz, dtype=np.float32),
            "cartesian_vacuum",
            len(rphiz),
        ).reshape(len(rphiz), 7)
        field.set_points_cyl(rphiz)
        exact = np.hstack(
            (
                field.B_cyl(),
                field.GradAbsB_cyl(),
                self.classifier.evaluate_rphiz(rphiz),
            )
        )
        scale = np.max(np.abs(exact), axis=0) + 1e-16
        self.assertLess(np.max(np.abs(exact - out) / scale), 1e-4)

    def test_derivatives(self):
        field, cfield = self.field(0.05, self.CELLS_PERTURBED)
        rphiz = self.points(500, 1.5, 2.5)
        vpar = self.rng.uniform(-self.velocity, self.velocity, len(rphiz))
        out = firm3dpp.test_derivatives_cartesian(
            cfield.quad_info,
            cfield.rrange,
            cfield.phirange,
            cfield.zrange,
            np.array(rphiz, dtype=np.float32),
            vpar.astype(np.float32),
            self.velocity,
            MASS,
            CHARGE,
            len(rphiz),
        ).reshape(len(rphiz), 4)
        exact = np.array(
            [
                cartesian_rhs(rphiz[i], vpar[i], field, MASS, CHARGE, self.velocity)
                for i in range(len(rphiz))
            ]
        )
        scale = np.max(np.abs(exact), axis=0) + 1e-16
        self.assertLess(np.max(np.abs(exact - out) / scale), 1e-3)

    def trace(self, cfield, rphiz, vpar, tmax):
        """(n, 5) final rows (t, x, y, z, vpar) of a single precision trace."""
        xyz = np.column_stack(
            (
                rphiz[:, 0] * np.cos(rphiz[:, 1]),
                rphiz[:, 0] * np.sin(rphiz[:, 1]),
                rphiz[:, 2],
            )
        )
        res_tys, _ = trace_particles_cartesian_gpu(
            cfield,
            xyz,
            vpar,
            tmax=tmax,
            mass=MASS,
            charge=CHARGE,
            Ekin=ENERGY,
            tol=1e-8,
            forget_exact_path=True,
        )
        return final_states(res_tys, boozer=False)

    def drift_speed(self, vpar):
        """Vertical drift speed [m/s] in the purely toroidal field."""
        v_perp2 = self.velocity**2 - vpar**2
        return MASS * (vpar**2 + 0.5 * v_perp2) / (CHARGE * self.B0 * self.R0)

    def test_orbit_matches_analytic_drift(self):
        _, cfield = self.field(0.0, self.CELLS_TOROIDAL)
        rphiz = self.points(64, 0.5, 0.5)  # z < 0 too: z must not be wrapped
        vpar = self.rng.uniform(-0.9, 0.9, len(rphiz)) * self.velocity
        tmax = 1e-6  # [s]: ~1 rad of toroidal turn, ~7 cm of drift
        out = self.trace(cfield, rphiz, vpar, tmax)
        t = out[:, 0]
        self.assertTrue(np.all(t >= tmax), "a particle stopped early")
        r = np.hypot(out[:, 1], out[:, 2])
        turn = np.angle(np.exp(1j * (np.arctan2(out[:, 2], out[:, 1]) - rphiz[:, 1])))
        expected_turn = np.angle(np.exp(1j * vpar * t / rphiz[:, 0]))
        np.testing.assert_allclose(r, rphiz[:, 0], atol=1e-3)
        np.testing.assert_allclose(turn, expected_turn, atol=1e-4)
        np.testing.assert_allclose(
            out[:, 3], rphiz[:, 2] + self.drift_speed(vpar) * t, atol=1e-3
        )
        np.testing.assert_allclose(out[:, 4], vpar, rtol=1e-4)

    def test_loss_time_matches_analytic(self):
        _, cfield = self.field(0.0, self.CELLS_TOROIDAL)
        rphiz = self.points(64, 0.5, 0.5)
        vpar = self.rng.uniform(-0.9, 0.9, len(rphiz)) * self.velocity
        tmax = 1e-4  # [s]: every particle drifts out through the top
        out = self.trace(cfield, rphiz, vpar, tmax)
        # lost on the first step that ends outside, at most one step late
        z_exit = np.sqrt(self.MINOR**2 - (rphiz[:, 0] - self.R0) ** 2)
        t_loss = (z_exit - rphiz[:, 2]) / self.drift_speed(vpar)
        dt_max = rphiz[:, 0] * 0.5 * np.pi / self.velocity  # the kernel's step cap
        self.assertTrue(np.all(out[:, 0] < tmax), "a particle was not lost")
        self.assertTrue(np.all(out[:, 0] >= t_loss * (1 - 1e-4)))
        self.assertTrue(np.all(out[:, 0] <= t_loss + dt_max))


# Setup for the scripts that trace in a fresh interpreter: the wrapper reads
# its FIRM3D_METAL_* settings once per process. trace() returns the final
# states, and each script saves its result to the file it is given.
FRESH_TRACE_SETUP = (
    "import sys\n"
    "import numpy as np\n"
    "from firm3d.catapult.field import CatapultBoozerField\n"
    "from firm3d.catapult.tracing import trace_particles_boozer_gpu\n"
    "from firm3d.util.constants import ALPHA_PARTICLE_CHARGE as q\n"
    "from firm3d.util.constants import ALPHA_PARTICLE_MASS as m\n"
    "from firm3d.util.constants import FUSION_ALPHA_PARTICLE_ENERGY as E\n"
    "from tests.field.test_gpu import get_field, sample_test_points\n"
    "bri, field, nfp = get_field("
    "'examples/inputs/boozmn_aten_rescaled_low_res.nc', 8, False)\n"
    "cfield = CatapultBoozerField(field, 8, 8, 8, precision='single')\n"
    "stz = sample_test_points(64)\n"
    "stz = np.ascontiguousarray(stz[stz[:, 0] < 1.0][:13])\n"
    "v = np.sqrt(2 * E / m)\n"
    "vpar = np.random.default_rng(3).uniform(-v, v, len(stz))\n"
    "def trace():\n"
    "    res_tys, _ = trace_particles_boozer_gpu(cfield, stz, vpar, tmax=1e-6, "
    "mass=m, charge=q, Ekin=E, tol=1e-8, forget_exact_path=True)\n"
    "    return np.array([ty[-1] for ty in res_tys])\n"
)


def run_fresh_interpreter(script: str, env: dict[str, str]) -> tuple[np.ndarray, str]:
    """Run script in a new interpreter; return the array it saved, and stderr."""
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "out.npy"
        proc = subprocess.run(
            [sys.executable, "-c", script, str(out)],
            env=env,
            capture_output=True,
            text=True,
            cwd=Path(__file__).parents[2],
        )
        if proc.returncode != 0:
            raise AssertionError(proc.stderr)
        return np.load(out), proc.stderr


@unittest.skipUnless(HAS_METAL, "Metal backend not available")
class TestMetalDispatchChunking(unittest.TestCase):
    """
    The host splits a trace into dispatches, each sized from the GPU time of
    the last so that none holds the GPU long enough for macOS to kill it, or
    of FIRM3D_METAL_MAX_STEPS steps each when that is set, keeping every
    particle's state, step size and first-same-as-last derivative in device
    buffers in between. However the trace is split, it must come out bitwise
    the same as in one dispatch.
    """

    SCRIPT = FRESH_TRACE_SETUP + "np.save(sys.argv[1], trace())\n"
    # 20,000 markers at the setup's tolerance: a single 1000-step dispatch,
    # the host's fixed size before dispatches were timed, takes 0.19 s on an
    # M4's GPU (and 0.8 s at a benchmark's tolerance and table size)
    LONG_SCRIPT = FRESH_TRACE_SETUP + (
        "stz = sample_test_points(80000)\n"
        "stz = np.ascontiguousarray(stz[stz[:, 0] < 1.0][:20000])\n"
        "vpar = np.random.default_rng(3).uniform(-v, v, len(stz))\n"
        "trace_particles_boozer_gpu(cfield, stz, vpar, tmax=1e-5, mass=m, "
        "charge=q, Ekin=E, tol=1e-8, forget_exact_path=True)\n"
        "np.save(sys.argv[1], len(stz))\n"
    )
    # twice the wrapper's TARGET_DISPATCH_S
    LONGEST_DISPATCH_S = 0.1

    def trace(
        self, max_steps: str | None, script: str = SCRIPT
    ) -> tuple[np.ndarray, int, float]:
        """What the script saved; the trace's dispatches and longest one [s]."""
        env = dict(os.environ, FIRM3D_METAL_PROFILE="1")
        env.pop("FIRM3D_METAL_MAX_STEPS", None)
        if max_steps is not None:
            env["FIRM3D_METAL_MAX_STEPS"] = max_steps
        saved, stderr = run_fresh_interpreter(script, env)
        traces = re.findall(
            r"trace: \d+ particles, (\d+) dispatches .*, longest ([\d.]+) ms", stderr
        )
        self.assertEqual(len(traces), 1, stderr)
        return saved, int(traces[0][0]), float(traces[0][1]) / 1e3

    def test_split_trace_is_bitwise_identical(self):
        whole, n_whole, _ = self.trace("100000")
        self.assertEqual(n_whole, 1, "the reference must trace in one dispatch")
        for max_steps in (None, "1", "7"):  # None: timed dispatches
            split, n_split, _ = self.trace(max_steps)
            self.assertGreater(n_split, 1, f"FIRM3D_METAL_MAX_STEPS={max_steps}")
            np.testing.assert_array_equal(
                split, whole, err_msg=f"FIRM3D_METAL_MAX_STEPS={max_steps}"
            )

    def test_dispatches_stay_short(self):
        n, dispatches, longest = self.trace(None, self.LONG_SCRIPT)
        self.assertEqual(n, 20000)
        self.assertGreater(dispatches, 1)
        self.assertLess(longest, self.LONGEST_DISPATCH_S)


@unittest.skipUnless(HAS_METAL, "Metal backend not available")
class TestMetalFieldTable(unittest.TestCase):
    """
    The GPU reads a CatapultBoozerField's table in place, and copies any
    other array on every launch: at high resolution those copies, one per
    saved interval, were most of a trace's wall time. Both paths must trace
    bitwise alike; FIRM3D_METAL_PROFILE reports which one a launch took.
    """

    SCRIPT = FRESH_TRACE_SETUP + (
        "in_place = trace()\n"
        "# the same table one float past a page boundary, which Metal copies\n"
        "shifted = np.empty(cfield.quad_pts.size + 1, dtype=np.float32)[1:]\n"
        "shifted[:] = cfield.quad_pts\n"
        "cfield.quad_pts = shifted\n"
        "np.save(sys.argv[1], np.stack([in_place, trace()]))\n"
    )

    def test_in_place_and_copied_tables_trace_alike(self):
        env = dict(os.environ, FIRM3D_METAL_PROFILE="1")
        (in_place, copied), stderr = run_fresh_interpreter(self.SCRIPT, env)
        paths = re.findall(r"field table: \d+ bytes, ([a-z ]+)", stderr)
        self.assertEqual(paths, ["read in place", "copied"], stderr)
        np.testing.assert_array_equal(copied, in_place)


@unittest.skipUnless(HAS_METAL, "Metal backend not available")
class TestMetalTracingEdgeCases(unittest.TestCase):
    """
    Degenerate inputs exercising the kernel's masking and chunking logic
    rather than its numerics: the kernel packs 8 particles per 32-thread
    SIMD-group, so counts leaving a partial block must behave exactly like
    full ones, and out-of-domain or zero-length traces must terminate.
    """

    def setUp(self):
        self.n_metagrid_pts = 15
        filename = "examples/inputs/boozmn_aten_rescaled_low_res.nc"
        self.bri, field, self.nfp = get_field(filename, self.n_metagrid_pts, True)
        res = self.n_metagrid_pts
        self.cfield = CatapultBoozerField(field, res, res, res, precision="single")

        stz = sample_test_points(100)
        inside = stz[:, 0] < 1.0
        self.stz = stz[inside][:13]
        velocity = np.sqrt(2 * ENERGY / MASS)
        self.vpar = np.random.default_rng(2).uniform(-velocity, velocity, 13)
        self.tmax = 1e-7

    def trace(self, stz, vpar, tmax):
        res_tys, _ = trace_particles_boozer_gpu(
            self.cfield,
            stz,
            vpar,
            tmax=tmax,
            mass=MASS,
            charge=CHARGE,
            Ekin=ENERGY,
            tol=1e-8,
            forget_exact_path=True,
        )
        return res_tys

    def test_partial_simd_block_matches_full(self):
        # 13 and 5 particles both leave a partial 8-particle block; a
        # particle's trajectory must be bitwise independent of its block
        # neighbours and of the padding lanes
        full = self.trace(self.stz, self.vpar, self.tmax)
        part = self.trace(self.stz[:5], self.vpar[:5], self.tmax)
        self.assertEqual(len(full), 13)
        self.assertEqual(len(part), 5)
        for i in range(5):
            np.testing.assert_array_equal(part[i][-1], full[i][-1])

    def test_finished_neighbours_leave_a_trace_unchanged(self):
        # a particle that reaches its tmax sits out the DP5 stages while the
        # rest of its block steps on; those still stepping must come out
        # bitwise as they do when no block neighbour finishes early
        tmax = np.full(len(self.stz), self.tmax)
        tmax[1::2] = self.tmax / 10
        mixed = self.trace(self.stz, self.vpar, tmax)
        uniform = self.trace(self.stz, self.vpar, self.tmax)
        for i in range(1, len(self.stz), 2):
            self.assertLess(mixed[i][-1, 0], self.tmax)  # did finish early
        for i in range(0, len(self.stz), 2):
            np.testing.assert_array_equal(mixed[i][-1], uniform[i][-1])

    def test_single_particle(self):
        res_tys = self.trace(self.stz[:1], self.vpar[:1], self.tmax)
        self.assertEqual(len(res_tys), 1)
        self.assertGreaterEqual(res_tys[0][-1, 0], self.tmax)

    def test_tmax_zero_returns_initial_state(self):
        res_tys = self.trace(self.stz[:3], self.vpar[:3], 0.0)
        for i, ty in enumerate(res_tys):
            self.assertEqual(ty[-1, 0], 0.0)
            np.testing.assert_allclose(ty[-1, 1], self.stz[i, 0], rtol=1e-6)

    def test_born_outside_lcfs_stops_immediately(self):
        outside = self.stz[:3].copy()
        outside[:, 0] = 1.05
        res_tys = self.trace(outside, self.vpar[:3], self.tmax)
        for ty in res_tys:
            self.assertLess(ty[-1, 0], self.tmax)  # flagged lost, not traced out
            self.assertGreaterEqual(ty[-1, 1], 1.0)


@unittest.skipUnless(HAS_METAL, "Metal backend not available")
class TestMetalParticleSetup(unittest.TestCase):
    """
    What the setup kernel derives where the host passes the -1 sentinels: the
    initial step from the step bound |G|/|B| pi/(2v), and mu from v_total and
    the parallel speed. A tmax = 0 launch returns them with no step taken.
    These follow fixes upstream made to the CUDA kernel this one ports.
    """

    def setUp(self):
        self.velocity = np.float32(np.sqrt(2 * ENERGY / MASS))
        # pseudo-Cartesian (s cos theta, s sin theta, zeta) at s = 0.5
        theta = np.linspace(0.0, 2 * np.pi, 8, endpoint=False)
        self.x_inits = np.ascontiguousarray(
            np.column_stack((0.5 * np.cos(theta), 0.5 * np.sin(theta), theta)),
            dtype=np.float32,
        )

    def table(self, G0: float) -> CatapultBoozerField:
        # uniform |B| (etabar = 0): otherwise d|B|/ds ~ 1/sqrt(s) is singular
        # on the table's s = 0 layer
        field = BoozerAnalytic(etabar=0.0, B0=1.0, N=0, G0=G0, psi0=1.0, iota0=0.4)
        return CatapultBoozerField(field, 8, 8, 8, precision="single")

    def setup_only(self, cfield: CatapultBoozerField, vpar: np.ndarray) -> np.ndarray:
        """(nparticles, 7) rows (t, x1, x2, zeta, vpar, dt, mu) at tmax = 0."""
        n = len(vpar)
        out = firm3dpp.boozer_gpu_tracing(
            quad_pts=cfield.quad_pts,
            srange=cfield.srange,
            trange=cfield.trange,
            zrange=cfield.zrange,
            stz_init=self.x_inits[:n].copy(),
            m=MASS,
            q=CHARGE,
            vtotal=float(self.velocity),
            vtang=np.ascontiguousarray(vpar, dtype=np.float32),
            tmax=np.zeros(n),
            tol=1e-8,
            dt_in=-np.ones(n, dtype=np.float32),
            mu_in=-np.ones(n, dtype=np.float32),
            psi0=cfield.psi0,
            nparticles=n,
            vacuum=cfield.vacuum,
        )
        return np.reshape(out, (n, 7))

    def test_negative_G_steps_forward(self):
        # the sign of G is a convention of the equilibrium, but the bound it
        # sets is a time: a negative one seeds a negative dt, and the trace
        # then runs backwards and never reaches tmax
        vpar = np.linspace(-0.9, 0.9, len(self.x_inits)) * self.velocity
        forward = self.setup_only(self.table(G0=1.1), vpar)
        mirrored = self.setup_only(self.table(G0=-1.1), vpar)
        self.assertTrue(np.all(mirrored[:, 5] > 0), mirrored[:, 5])
        np.testing.assert_array_equal(mirrored[:, 5], forward[:, 5])

    def test_parallel_speed_rounded_above_total_gives_zero_mu(self):
        # rounding can put |v_par| a few ulps above v_total, which the Python
        # check admits; mu from v_total^2 - v_par^2 must clamp at 0, since a
        # negative mu has no physical meaning
        above = np.nextafter(self.velocity, np.float32(np.inf))
        above = np.nextafter(above, np.float32(np.inf))
        out = self.setup_only(self.table(G0=1.1), np.array([above, -above]))
        np.testing.assert_array_equal(out[:, 6], 0.0)


@unittest.skipUnless(HAS_METAL, "Metal backend not available")
class TestMetalEnvKnobs(unittest.TestCase):
    """
    The diagnostic env vars parse their value: VAR=0/false/off must leave
    the knob off (a presence-only check once turned FIRM3D_METAL_FAST_MATH=0
    into relaxed math). The wrapper reads the environment once at first GPU
    use, so each case needs a fresh interpreter.
    """

    SCRIPT = (
        "import numpy as np\n"
        "from tests.field.test_gpu import "
        "CATAPULTField, get_field, sample_test_points\n"
        "bri, field, nfp = get_field("
        "'examples/inputs/boozmn_aten_rescaled_low_res.nc', 8, True)\n"
        "f = CATAPULTField(field, ns=8, ntheta=8, nzeta=8, nfp=nfp)\n"
        "f.compute_gpu_interpolant(sample_test_points(8), dtype=np.float32)\n"
    )

    def profile_lines(self, value):
        env = dict(os.environ, FIRM3D_METAL_PROFILE=value)
        proc = subprocess.run(
            [sys.executable, "-c", self.SCRIPT],
            env=env,
            capture_output=True,
            text=True,
            cwd=Path(__file__).parents[2],
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        return proc.stderr.count("[metal-profile]")

    def test_disabled_values_are_silent(self):
        for value in ("0", "false", "off"):
            self.assertEqual(
                self.profile_lines(value), 0, f"FIRM3D_METAL_PROFILE={value}"
            )

    def test_enabled_value_prints(self):
        self.assertGreater(self.profile_lines("1"), 0)
