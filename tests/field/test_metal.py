import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import firm3dpp

from firm3d.catapult.field import CatapultBoozerField
from firm3d.catapult.tracing import (
    save_trajectories_boozer_gpu,
    trace_particles_boozer_gpu,
)
from firm3d.field.boozermagneticfield import (
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
    The host splits a trace into dispatches of at most FIRM3D_METAL_MAX_STEPS
    steps each, keeping every particle's state, step size and
    first-same-as-last derivative in device buffers in between. However the
    trace is split, it must come out bitwise the same as in one dispatch.
    """

    SCRIPT = FRESH_TRACE_SETUP + "np.save(sys.argv[1], trace())\n"

    def trace(self, max_steps: str | None) -> tuple[np.ndarray, int]:
        """Final states, and the number of dispatches the trace took."""
        env = dict(os.environ, FIRM3D_METAL_PROFILE="1")
        env.pop("FIRM3D_METAL_MAX_STEPS", None)
        if max_steps is not None:
            env["FIRM3D_METAL_MAX_STEPS"] = max_steps
        final, stderr = run_fresh_interpreter(self.SCRIPT, env)
        dispatches = re.findall(r"trace: \d+ particles, (\d+) dispatches", stderr)
        self.assertEqual(len(dispatches), 1, stderr)
        return final, int(dispatches[0])

    def test_split_trace_is_bitwise_identical(self):
        whole, n_whole = self.trace(None)
        self.assertEqual(n_whole, 1, "the default must fit this trace in one dispatch")
        for max_steps in ("1", "7"):
            split, n_split = self.trace(max_steps)
            self.assertGreater(n_split, 1, f"FIRM3D_METAL_MAX_STEPS={max_steps}")
            np.testing.assert_array_equal(
                split, whole, err_msg=f"FIRM3D_METAL_MAX_STEPS={max_steps}"
            )


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
