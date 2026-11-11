"""Behavioral checks for calibration, frame quality, contact convention and tracking."""

from pathlib import Path
import tempfile
import sys
import unittest

import mujoco
import numpy as np
import openpyxl
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from fit_xsens_human import SEGMENTS, audit_and_correct, fit_motion, quaternion_rotations
from prepare_xsens_model import prepare
from simulate_xsens_walk import contact_wrenches, make_environment, floating_base_command
from compute_joint_torques import reconstruct


class XsensPipelineTests(unittest.TestCase):
    def make_workbook(self, path, count=25):
        model = mujoco.MjModel.from_xml_path(str(ROOT / "model/ms_human_locomotion/human.xml"))
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        ref = np.stack(
            [data.xmat[model.body(body).id].reshape(3, 3).copy() for _, body, _ in SEGMENTS]
        )
        workbook = openpyxl.Workbook()
        general = workbook.active
        general.title = "General Information"
        general.append(["Frame Rate", 60])
        general.append(["Processing Quality", "synthetic test"])
        quat_sheet = workbook.create_sheet("Segment Orientation - Quat")
        pos_sheet = workbook.create_sheet("Segment Position")
        quat_sheet.append(
            ["Frame"] + [f"{segment} q{i}" for segment, _, _ in SEGMENTS for i in range(4)]
        )
        pos_sheet.append(
            ["Frame"] + [f"{segment} {a}" for segment, _, _ in SEGMENTS for a in ("x", "y", "z")]
        )
        expected = []
        proximal = [
            None,
            "right_hip_pitch",
            "right_knee_pitch",
            "right_ankle_pitch",
            "left_hip_pitch",
            "left_knee_pitch",
            "left_ankle_pitch",
        ]
        for i in range(count):
            data.qpos[:] = model.qpos0
            fraction = i / (count - 1)
            for name, angle in (
                ("right_hip_pitch", 0.15 * fraction),
                ("right_knee_pitch", 0.35 * fraction),
                ("right_ankle_pitch", -0.08 * fraction),
                ("left_hip_roll", -0.06 * fraction),
            ):
                data.qpos[model.jnt_qposadr[model.joint(name).id]] = angle
            expected.append(data.qpos.copy())
            mujoco.mj_forward(model, data)
            matrices = np.stack(
                [data.xmat[model.body(body).id].reshape(3, 3) for _, body, _ in SEGMENTS]
            )
            rotation = matrices @ ref.transpose(0, 2, 1)
            quats = Rotation.from_matrix(rotation).as_quat()[:, [3, 0, 1, 2]]
            points = np.stack(
                [
                    (
                        data.xpos[model.body("pelvis").id]
                        if name is None
                        else data.xanchor[model.joint(name).id]
                    )
                    for name in proximal
                ]
            )
            quat_sheet.append([i] + quats.ravel().tolist())
            pos_sheet.append([i] + points.ravel().tolist())
        workbook.save(path)
        workbook.close()
        return model, np.asarray(expected)

    def test_known_motion_recovered_and_root_quaternion_not_double_rotated(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            model, expected = self.make_workbook(tmp / "recording.xlsx")
            fit_motion(
                tmp / "recording.xlsx",
                ROOT / "model/ms_human_locomotion/human.xml",
                tmp / "fit.npz",
                align_heading=False,
            )
            with np.load(tmp / "fit.npz") as fit:
                np.testing.assert_allclose(fit["qpos"][:, 3:], expected[:, 3:], atol=3e-4)
                self.assertLess(float(np.max(fit["orientation_residual_deg"])), 0.01)
                self.assertTrue(fit["solver_success"].all())
                self.assertEqual(fit["tracking_qpos"].shape, (25, 22))
                self.assertTrue(np.isfinite(fit["qvel"][fit["derivative_valid"]]).all())
                self.assertTrue(np.isfinite(fit["qacc"][fit["derivative_valid"]]).all())

    def test_bad_quaternions_are_flagged_not_treated_as_valid_identity(self):
        quats = np.zeros((5, 7, 4))
        quats[..., 0] = 1
        quats[2, 4] = 0
        matrices, valid = quaternion_rotations(quats)
        self.assertFalse(valid[2, 4])
        self.assertTrue(np.isfinite(matrices).all())

    def test_common_reset_is_flagged_with_safety_interval(self):
        matrices = np.tile(np.eye(3), (121, 7, 1, 1))
        reset = Rotation.from_euler("z", 40, degrees=True).as_matrix()
        matrices[60:] = reset @ matrices[60:]
        positions = np.zeros((121, 3))
        positions[60:, 0] = 100
        corrected, position, valid, events, report = audit_and_correct(
            matrices, np.ones((121, 7), dtype=bool), positions, 60
        )
        self.assertFalse(valid[60])
        self.assertFalse(valid[45])
        self.assertTrue(valid[0])
        np.testing.assert_allclose(corrected, np.tile(np.eye(3), (121, 7, 1, 1)), atol=1e-12)
        np.testing.assert_allclose(position, 0, atol=1e-12)
        self.assertEqual(len(report), 2)

    def test_subject_mass_and_stature_are_applied_without_touching_template(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            self.make_workbook(tmp / "recording.xlsx")
            prepare(
                tmp / "recording.xlsx",
                ROOT / "model/ms_human_locomotion/human.xml",
                tmp / "human.xml",
                62.5,
                1.72,
            )
            model = mujoco.MjModel.from_xml_path(str(tmp / "human.xml"))
            data = mujoco.MjData(model)
            mujoco.mj_forward(model, data)
            self.assertAlmostEqual(float(model.body_mass.sum()), 62.5, places=10)
            head = model.geom("head_visual").id
            self.assertAlmostEqual(
                data.geom_xpos[head, 2] + model.geom_size[head, 0], 1.72, places=9
            )
            mass = np.empty((model.nv, model.nv))
            mujoco.mj_fullM(model, mass, data.qM)
            self.assertGreater(np.linalg.eigvalsh(mass)[0], 0)

    def test_world_wrenches_reconstruct_generalized_contact_force(self):
        path = ROOT / "model/ms_human_locomotion/human.xml"
        model, _, belt_dof = make_environment(path, 0.001, 0.5)
        data = mujoco.MjData(model)
        human = mujoco.MjModel.from_xml_path(str(path))
        data.qpos[: human.nq] = human.qpos0
        data.qpos[2] -= 0.005
        data.qvel[belt_dof] = -0.5
        mujoco.mj_forward(model, data)
        wrenches, counts = contact_wrenches(model, data)
        self.assertTrue((counts > 0).all())
        result = np.zeros(model.nv)
        for i, side in enumerate(("left", "right")):
            jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
            mujoco.mj_jacSite(model, data, jp, jr, model.site(f"{side}_foot").id)
            result += np.vstack((jp, jr)).T @ wrenches[i]
        np.testing.assert_allclose(result[:21], data.qfrc_constraint[:21], atol=1e-9)
        self.assertTrue((wrenches[:, 2] > 0).all())

    def test_free_controller_removes_unactuated_inverse_force(self):
        path = ROOT / "model/ms_human_locomotion/human.xml"
        model, _, _ = make_environment(path, 0.001, 0)
        data = mujoco.MjData(model)
        human = mujoco.MjModel.from_xml_path(str(path))
        data.qpos[:22] = human.qpos0
        data.qpos[2] -= 0.005
        required = floating_base_command(model, data, np.zeros(model.nv))
        self.assertTrue(np.isfinite(required).all())
        np.testing.assert_allclose(required[:6], 0, atol=1e-3)

    def test_inverse_dynamics_recovers_applied_torques_and_assistance(self):
        path = ROOT / "model/ms_human_locomotion/human.xml"
        model, _, belt_dof = make_environment(path, 0.001, 0)
        data = mujoco.MjData(model)
        human = mujoco.MjModel.from_xml_path(str(path))
        data.qpos[:22] = human.qpos0
        data.qpos[2] -= 0.005
        data.ctrl[:] = np.linspace(-3, 3, model.nu)
        data.qfrc_applied[:6] = [1, 2, 3, 4, 5, 6]
        mujoco.mj_forward(model, data)
        wrenches, _ = contact_wrenches(model, data)
        recovered = reconstruct(model, data, data.qacc, wrenches)
        dofs = np.array([model.jnt_dofadr[int(j)] for j in model.actuator_trnid[:, 0]])
        np.testing.assert_allclose(recovered[dofs], data.ctrl, atol=1e-8)
        np.testing.assert_allclose(recovered[:6], data.qfrc_applied[:6], atol=1e-8)


if __name__ == "__main__":
    unittest.main()
