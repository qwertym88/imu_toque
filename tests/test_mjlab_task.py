"""Optional CPU checks against the actual mjlab API; no CUDA simulation required."""

import importlib.util
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
HAS_MJLAB = importlib.util.find_spec("mjlab") is not None


@unittest.skipUnless(HAS_MJLAB, "optional mjlab environment not installed")
class MjlabTaskTests(unittest.TestCase):
    def test_runtime_model_export_keeps_floor_actuators_and_prefixed_names(self):
        import mujoco
        from mjlab.scene import Scene
        from human_tracking.task import make_env_cfg
        from human_tracking.runner import export_runtime_model

        # A temporary entity file keeps this test independent of local Xsens data.
        import xml.etree.ElementTree as ET

        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            tree = ET.parse(ROOT / "model/ms_human_locomotion/human.xml")
            root = tree.getroot()
            root.remove(root.find("actuator"))
            root.remove(root.find("keyframe"))
            root.find("worldbody").remove(root.find("worldbody/geom"))
            tree.write(directory / "human.xml")
            cfg = make_env_cfg(directory, 1)
            scene = Scene(cfg.scene, device="cpu")
            model = scene.compile()
            model.opt.timestep = 0.005
            output = directory / "runtime.xml"
            export_runtime_model(scene, model, output)
            exported = mujoco.MjModel.from_xml_path(str(output))
            self.assertEqual(exported.nu, 15)
            self.assertEqual(exported.ngeom, model.ngeom)
            self.assertGreaterEqual(
                mujoco.mj_name2id(exported, mujoco.mjtObj.mjOBJ_GEOM, "robot/left_sole"), 0
            )
            self.assertAlmostEqual(exported.opt.timestep, model.opt.timestep)

    def test_fall_detection_respects_anatomical_pelvis_rotation(self):
        import torch
        from human_tracking.task import fall_detected

        upright = torch.tensor([[2**-0.5, 2**-0.5, 0.0, 0.0]])
        command = SimpleNamespace(
            robot_anchor_pos_w=torch.tensor([[0.0, 0.0, 1.0]]),
            anchor_quat_w=upright,
            robot_anchor_quat_w=upright.clone(),
        )
        robot = SimpleNamespace(
            data=SimpleNamespace(
                gravity_vec_w=torch.tensor([[0.0, 0.0, -1.0]]),
                joint_pos=torch.zeros(1, 15),
            )
        )

        class Scene(dict):
            env_origins = torch.zeros(1, 3)

        env = SimpleNamespace(
            scene=Scene(robot=robot),
            command_manager=SimpleNamespace(get_term=lambda _: command),
        )
        self.assertFalse(bool(fall_detected(env)[0]))
        command.robot_anchor_quat_w = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        self.assertTrue(bool(fall_detected(env)[0]))

    def test_training_has_no_external_force_events_and_local_logging(self):
        from human_tracking.task import make_env_cfg, make_runner_cfg, ReferenceResidualActionCfg

        cfg = make_env_cfg(ROOT / "outputs/y13/mjlab", 16)
        self.assertEqual(cfg.events, {})
        self.assertEqual(cfg.commands["motion"].body_names[0], "pelvis")
        self.assertIsInstance(cfg.actions["joint_pos"], ReferenceResidualActionCfg)
        self.assertAlmostEqual(cfg.sim.mujoco.timestep * cfg.decimation, 1 / 50)
        agent = make_runner_cfg()
        self.assertEqual(agent.logger, "tensorboard")
        self.assertFalse(agent.upload_model)


if __name__ == "__main__":
    unittest.main()
