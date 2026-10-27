import os
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np
import firm3dpp

from firm3d.catapult.field import CatapultBoozerField
from firm3d.catapult.tracing import (
    save_trajectories_boozer_gpu,
    trace_particles_boozer_gpu,
)
from firm3d.field.tracing import (
    MaxToroidalFluxStoppingCriterion,
    trace_particles_boozer,
)
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
    final_states,
    get_field,
    sample_test_points,
)

HAS_METAL = getattr(firm3dpp, "gpu_backend", None) == "metal"
n_test_pts = 2000


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
        self.vpar_init = np.random.uniform(-self.VELOCITY, self.VELOCITY, (n_test_pts,))

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
        out_gpu = final_states(res_tys)
        survived = out_gpu[:, 0] >= 0.999 * tmax
        self.assertTrue(survived.any())

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
        self.assertTrue(np.any(diff != 0), "Metal single matched CPU double exactly")
        factor = 10.0
        for c in range(diff.shape[1]):
            for q in (50, 90):
                got = np.percentile(np.abs(diff[:, c]), q)
                expected = np.percentile(np.abs(ref[:, c]), q)
                self.assertLessEqual(
                    got,
                    factor * expected + 1e-12,
                    f"column {columns[c]}, p{q}: {got:.3e} exceeds "
                    f"{factor} x {expected:.3e}",
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
