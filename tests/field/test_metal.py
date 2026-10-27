import unittest

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

    def setUp(self):
        self.n_metagrid_pts = 15
        self.filename = "examples/inputs/boozmn_aten_rescaled_low_res.nc"
        self.vacuum = True
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
