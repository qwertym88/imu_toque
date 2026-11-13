"""Reference conversion tests run without the optional mjlab/GPU environment."""

from pathlib import Path
import hashlib
import json
import sys
import tempfile
import unittest

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from human_tracking.motion import interpolate_poses, prepare_reference


class MjlabReferenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)
        self.model_path = ROOT / "model/ms_human_locomotion/human.xml"
        self.model = mujoco.MjModel.from_xml_path(str(self.model_path))
        self.times = np.arange(61) / 60.0
        self.poses = np.tile(self.model.qpos0, (61, 1))
        # Nonzero root angular velocity makes COM-vs-link velocity mistakes visible.
        for i, elapsed in enumerate(self.times):
            yaw = np.array([np.cos(elapsed / 4), 0, 0, np.sin(elapsed / 4)])
            mujoco.mju_mulQuat(self.poses[i, 3:7], yaw, self.model.qpos0[3:7])
        self.valid = np.ones(61, dtype=bool)
        self.fit_path = self.directory / "fit.npz"
        self.write_fit()

    def tearDown(self):
        self.temp.cleanup()

    def write_fit(self, model_hash=None):
        metadata = {
            "model_sha256": model_hash or hashlib.sha256(self.model_path.read_bytes()).hexdigest()
        }
        np.savez_compressed(
            self.fit_path,
            time_s=self.times,
            tracking_qpos=self.poses,
            valid=self.valid,
            derivative_valid=self.valid,
            joint_names=np.asarray([self.model.joint(i).name for i in range(1, self.model.njnt)]),
            metadata_json=np.asarray(json.dumps(metadata)),
        )

    def convert(self, speed=1.0):
        return prepare_reference(
            self.model_path, self.fit_path, self.directory / "converted", 0.1, 0.8, speed=speed
        )

    def test_frame_transform_orders_and_body_origin_velocities(self):
        report = self.convert(speed=1.0)
        model = mujoco.MjModel.from_xml_path(report["model"])
        data = mujoco.MjData(model)
        with np.load(self.directory / "converted/motion.npz") as motion:
            self.assertEqual(motion["joint_pos"].shape, (41, 15))
            self.assertEqual(motion["body_pos_w"].shape, (41, 10, 3))
            self.assertEqual(report["body_names"][0], "pelvis")
            np.testing.assert_allclose(motion["qpos"][:, 0], motion["time_s"], atol=1e-12)
            np.testing.assert_allclose(
                np.linalg.norm(motion["body_quat_w"], axis=-1), 1, atol=1e-12
            )
            i, h = 20, 1e-5
            endpoints = []
            for sign in (-1, 1):
                data.qpos[:] = motion["qpos"][i]
                mujoco.mj_integratePos(model, data.qpos, motion["qvel"][i], sign * h)
                mujoco.mj_forward(model, data)
                endpoints.append(data.xpos[1:].copy())
            derivative = (endpoints[1] - endpoints[0]) / (2 * h)
            np.testing.assert_allclose(motion["body_lin_vel_w"][i], derivative, atol=1e-6)
            self.assertEqual(model.nu, 0)  # PD actuators are added by the mjlab asset.
            self.assertEqual(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor"), -1)
            self.assertAlmostEqual(float(model.body_mass.sum()), float(self.model.body_mass.sum()))

    def test_invalid_interval_is_rejected_instead_of_interpolated(self):
        self.valid[30] = False
        self.write_fit()
        with self.assertRaisesRegex(ValueError, "invalid/reset"):
            self.convert()

    def test_changed_model_is_rejected(self):
        self.write_fit(model_hash="wrong-model")
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            self.convert()

    def test_quaternion_sign_flip_does_not_create_spin(self):
        poses = self.poses[:2].copy()
        poses[1, 3:7] = -poses[0, 3:7]
        actual = interpolate_poses(np.array([0.0, 1.0]), poses, np.array([0.5]))
        np.testing.assert_allclose(actual[0, 3:7], poses[0, 3:7], atol=1e-12)

    def test_converter_refuses_to_overwrite_input_model(self):
        with self.assertRaisesRegex(ValueError, "overwrite"):
            prepare_reference(self.model_path, self.fit_path, self.model_path.parent)


if __name__ == "__main__":
    unittest.main()
