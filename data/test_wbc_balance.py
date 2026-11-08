"""Math and MuJoCo convention checks: python -m unittest discover -s data -p 'test_wbc*.py'."""

import unittest

import mujoco
import numpy as np
from numpy.testing import assert_allclose
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

from simulate_wbc import (
    ROOT,
    SIDES,
    NZ,
    NV,
    build_inequalities,
    interpolate_rotation,
    linearized_zmp_task,
    measured_contact_wrenches,
    require_id,
    resultant_zmp,
    task_state,
)
from wbc_balance import BalanceReference, project_to_polygon


class BalanceMathTests(unittest.TestCase):
    def reference(self):
        time = np.arange(241) / 60.0
        support = np.ones((len(time), 2), dtype=bool)
        support[120:, 0] = False
        return {
            "time": time,
            "support": support,
            "com": np.tile([0.0, 0.0, 0.5], (len(time), 1)),
            "pose": {
                "left_foot": np.tile([0.0, 0.1, 0.0], (len(time), 1)),
                "right_foot": np.tile([0.0, -0.1, 0.0], (len(time), 1)),
            },
        }

    def test_preview_obeys_continuous_lip_equations(self):
        plan = BalanceReference(self.reference(), 1.0, 1.0, 0.5)
        epsilon = 1e-6
        for time in (0.123, 1.234, 1.987, 2.456):
            cm, vm, xim, *_ = plan.sample(time - epsilon)
            cp, vp, xip, *_ = plan.sample(time + epsilon)
            c, v, xi, p, *_ = plan.sample(time)
            assert_allclose((cp - cm) / (2 * epsilon), v, atol=1e-8)
            assert_allclose((vp - vm) / (2 * epsilon), plan.omega**2 * (c - p), atol=1e-8)
            assert_allclose((xip - xim) / (2 * epsilon), plan.omega * (xi - p), atol=1e-8)
            assert_allclose(xi, c + v / plan.omega, atol=1e-10)
        self.assertLess(plan.sample(1.2)[2][1], -0.01)

    def test_planned_stance_does_not_inherit_raw_foot_sliding(self):
        ref = self.reference()
        ref["pose"]["right_foot"][80:, 0] += 0.2
        plan = BalanceReference(ref, 1.0, 1.0, 0.5)
        assert_allclose(plan.feet[:, 1, 0], 0.0)

    def test_preview_is_translation_equivariant(self):
        ref = self.reference()
        first = BalanceReference(ref, 0.5, 1.0, 0.5)
        shift = np.array([0.7, -0.4])
        ref["com"][:, :2] += shift
        for positions in ref["pose"].values():
            positions[:, :2] += shift
        second = BalanceReference(ref, 0.5, 1.0, 0.5)
        assert_allclose(second.com - first.com, np.tile(shift, (len(first.com), 1)), atol=1e-12)
        assert_allclose(second.dcm - first.dcm, np.tile(shift, (len(first.com), 1)), atol=1e-12)

    def test_projection_respects_rotated_polygon_and_margin(self):
        corners = np.array([[-0.07, -0.04], [-0.07, 0.04], [0.07, -0.04], [0.07, 0.04]])
        rotation = Rotation.from_euler("z", 0.7).as_matrix()[:2, :2]
        vertices = corners @ rotation.T + [1.0, -0.2]
        equations = ConvexHull(vertices).equations
        point = project_to_polygon(np.array([1.3, 0.4]), equations, 0.005)
        self.assertTrue(np.all(equations[:, :2] @ point + equations[:, 2] <= -0.005 + 1e-9))
        inside = np.array([1.0, -0.2])
        assert_allclose(project_to_polygon(inside, equations, 0.005), inside)
        with self.assertRaises(ValueError):
            project_to_polygon(np.array([1.3, 0.4]), equations, 0.05)

    def test_rotation_interpolation_hits_noncommuting_endpoint(self):
        rotations = Rotation.from_euler("xyz", [[0.3, -0.5, 0.7], [-0.4, 0.2, 0.9]])
        assert_allclose(interpolate_rotation(rotations, 0, 0).as_matrix(), rotations[0].as_matrix())
        assert_allclose(
            interpolate_rotation(rotations, 0, 1).as_matrix(), rotations[1].as_matrix(), atol=1e-12
        )

    def test_zmp_linearization_has_no_force_scaling_gradient(self):
        numerator = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0]])
        normal = np.array([0.0, 0.0, 1.0])
        previous = np.array([15.0, -25.0, 500.0])
        target = np.array([0.03, 0.01])
        matrix, offset = linearized_zmp_task(numerator, normal, previous, 500.0, target)
        assert_allclose(matrix @ previous, 0.0, atol=1e-12)
        assert_allclose(matrix @ (0.2 * previous) + offset, offset, atol=1e-12)
        epsilon = 1e-4
        delta = np.array([1.0, 2.0, 3.0])
        shifted = previous + epsilon * delta
        exact = numerator @ shifted / (normal @ shifted) - target
        assert_allclose(matrix @ shifted + offset, exact, atol=1e-11)

    def test_resultant_zmp_is_force_weighted_on_plane(self):
        wrench = np.zeros((2, 6))
        wrench[:, 2] = [100.0, 300.0]
        positions = np.array([[0.1, -0.2, 0.0035], [0.3, 0.2, 0.0035]])
        zmp, force = resultant_zmp(wrench, positions, 0.0035)
        assert_allclose(zmp, [0.25, 0.1])
        assert_allclose(force, [0.0, 0.0, 400.0])
        zmp, _ = resultant_zmp(np.zeros((2, 6)), positions, 0.0035)
        self.assertTrue(np.isnan(zmp).all())


class ContactConventionTests(unittest.TestCase):
    def test_loaded_support_cannot_satisfy_constraints_with_zero_wrench(self):
        model = mujoco.MjModel.from_xml_path(str(ROOT / "model/szar_humanoid.xml"))
        data = mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(model, data, 0)
        mujoco.mj_forward(model, data)
        sites = {
            f"{s}_foot": require_id(model, mujoco.mjtObj.mjOBJ_SITE, f"{s}_foot") for s in SIDES
        }
        ground = require_id(model, mujoco.mjtObj.mjOBJ_GEOM, "ground")
        weight = model.body_subtreemass[0] * 9.81
        matrix, upper = build_inequalities(
            model,
            data,
            np.ones(2, dtype=bool),
            sites,
            ground,
            0.8,
            0.07,
            0.04,
            0.005,
            np.full(12, 30.0),
            0.2 * weight,
        )
        self.assertGreater(np.max(-upper), 0.1 * weight)
        solution = np.zeros(NZ)
        solution[NV + 2] = solution[NV + 8] = 0.5 * weight
        self.assertLessEqual(np.max(matrix @ solution - upper), 1e-8)
        matrix, upper = build_inequalities(
            model,
            data,
            np.zeros(2, dtype=bool),
            sites,
            ground,
            0.8,
            0.07,
            0.04,
            0.005,
            np.full(12, 30.0),
            0.2 * weight,
        )
        self.assertLessEqual(np.max(matrix @ np.zeros(NZ) - upper), 0.0)

    def test_measured_wrench_reconstructs_generalized_contact_force(self):
        model = mujoco.MjModel.from_xml_path(str(ROOT / "model/szar_humanoid.xml"))
        data = mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(model, data, 0)
        mujoco.mj_forward(model, data)
        reference = {
            "ground_geom_id": require_id(model, mujoco.mjtObj.mjOBJ_GEOM, "ground"),
            "site_ids": {
                f"{s}_foot": require_id(model, mujoco.mjtObj.mjOBJ_SITE, f"{s}_foot") for s in SIDES
            },
            "sole_geom_ids": {
                s: require_id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{s}_sole_collision") for s in SIDES
            },
        }
        jacobians = [
            task_state(model, data, f"{s}_foot", site_id=reference["site_ids"][f"{s}_foot"])[0]
            for s in SIDES
        ]
        mujoco.mj_step(model, data)
        wrench, counts = measured_contact_wrenches(model, data, reference)
        self.assertGreater(wrench[:, 2].sum(), 0.0)
        self.assertTrue(np.all(counts > 0))
        reconstructed = sum(jac.T @ force for jac, force in zip(jacobians, wrench))
        assert_allclose(reconstructed, data.qfrc_constraint, atol=1e-8)


if __name__ == "__main__":
    unittest.main()
