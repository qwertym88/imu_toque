"""Mechanical acceptance checks for the MS-Human rigid-body derivation."""

from pathlib import Path
import sys
import unittest

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from build_ms_human import LEG_JOINTS, SIDES, SOURCE, body_groups, derive, load_source_reference
from inspect_human_model import inspect_model


class HumanModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = mujoco.MjModel.from_xml_path(str(ROOT / "model/ms_human_locomotion/human.xml"))
        cls.source, cls.source_data = load_source_reference()
        cls.source_qref = cls.source_data.qpos.copy()
        # Remove upstream controller regularization when comparing rigid mechanics.
        cls.source.dof_armature[:] = 0

    def test_topology_and_no_hidden_passive_or_muscle_dynamics(self):
        model = self.model
        report, data = inspect_model(model)
        self.assertEqual((model.nq, model.nv, model.nu, model.nbody - 1), (22, 21, 15, 10))
        self.assertEqual((model.na, model.ntendon, model.neq), (0, 0, 0))
        for array in (
            model.dof_armature,
            model.dof_damping,
            model.dof_frictionloss,
            model.jnt_stiffness,
        ):
            np.testing.assert_array_equal(array, 0)
        self.assertFalse(np.any(model.actuator_ctrllimited))
        self.assertFalse(np.any(model.actuator_forcelimited))
        for side in ("left", "right"):
            self.assertAlmostEqual(report["soles"][side]["bottom_z_m"], 0, places=10)
        self.assertGreater(report["soles"]["left"]["site_world_m"][1], 0)
        self.assertLess(report["soles"]["right"]["site_world_m"][1], 0)
        # Verify body tensors and mass matrix away from the neutral pose too.
        rng = np.random.default_rng(410)
        for _ in range(6):
            data.qvel[:] = rng.normal(size=model.nv)
            for j in range(1, model.njnt):
                lo, hi = model.jnt_range[j] if model.jnt_limited[j] else (-0.3, 0.3)
                data.qpos[model.jnt_qposadr[j]] = rng.uniform(lo, hi)
            mujoco.mj_forward(model, data)
            mass = np.empty((model.nv, model.nv))
            mujoco.mj_fullM(model, mass, data.qM)
            self.assertGreater(np.linalg.eigvalsh(mass)[0], 0)
            np.testing.assert_array_equal(data.qfrc_passive, 0)

    def test_mass_com_and_whole_body_inertia_preserved_in_neutral_pose(self):
        model, source, sd = self.model, self.source, self.source_data
        sd.qpos[:] = self.source_qref
        sd.qvel[:] = 0
        mujoco.mj_forward(source, sd)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        self.assertAlmostEqual(
            float(model.body_mass.sum()), float(source.body_mass.sum()), places=10
        )
        np.testing.assert_allclose(data.subtree_com[1], sd.subtree_com[1], atol=1e-12)

        def world_tensor(m, d):
            com = d.subtree_com[1]
            result = np.zeros((3, 3))
            for b in range(1, m.nbody):
                r = d.ximat[b].reshape(3, 3)
                delta = d.xipos[b] - com
                result += r @ np.diag(m.body_inertia[b]) @ r.T
                result += m.body_mass[b] * (delta @ delta * np.eye(3) - np.outer(delta, delta))
            return result

        np.testing.assert_allclose(world_tensor(model, data), world_tensor(source, sd), atol=1e-9)

    def test_frozen_source_fk_mass_and_gravity_match_over_leg_motion(self):
        """Compare against the original articulated tree with auxiliaries frozen.

        This validates the reduction, NOT equivalence to the original moving
        polynomial knee/patella mechanism, which was intentionally simplified.
        """
        model, source, sd = self.model, self.source, self.source_data
        data = mujoco.MjData(model)
        mappings = [
            (f"{side}_{alias}", f"{original}_{suffix}")
            for side, suffix in SIDES
            for alias, original in LEG_JOINTS
        ]
        target_dofs = [model.jnt_dofadr[model.joint(a).id] for a, _ in mappings]
        source_dofs = [source.jnt_dofadr[source.joint(b).id] for _, b in mappings]
        rng = np.random.default_rng(413)
        for _ in range(8):
            mujoco.mj_resetData(model, data)
            sd.qpos[:] = self.source_qref
            sd.qvel[:] = 0
            for target, original in mappings:
                tj, sj = model.joint(target).id, source.joint(original).id
                lo, hi = model.jnt_range[tj]
                angle = rng.uniform(lo * 0.5, hi * 0.5)
                data.qpos[model.jnt_qposadr[tj]] = angle
                sd.qpos[source.jnt_qposadr[sj]] = angle
            mujoco.mj_forward(model, data)
            mujoco.mj_forward(source, sd)
            for name, members in body_groups(source).items():
                b = model.body(name).id
                com = (source.body_mass[members, None] * sd.xipos[members]).sum(axis=0)
                com /= source.body_mass[members].sum()
                np.testing.assert_allclose(data.xipos[b], com, atol=1e-11)
            target_mass = np.empty((model.nv, model.nv))
            source_mass = np.empty((source.nv, source.nv))
            mujoco.mj_fullM(model, target_mass, data.qM)
            mujoco.mj_fullM(source, source_mass, sd.qM)
            np.testing.assert_allclose(
                target_mass[np.ix_(target_dofs, target_dofs)],
                source_mass[np.ix_(source_dofs, source_dofs)],
                atol=1e-10,
            )
            np.testing.assert_allclose(
                data.qfrc_bias[target_dofs], sd.qfrc_bias[source_dofs], atol=1e-9
            )

    def test_gravity_matches_potential_energy_gradient(self):
        model = self.model
        data = mujoco.MjData(model)
        for name, angle in (
            ("right_hip_pitch", 0.2),
            ("left_knee_pitch", 0.4),
            ("torso_roll", 0.1),
        ):
            data.qpos[model.jnt_qposadr[model.joint(name).id]] = angle
        mujoco.mj_forward(model, data)
        bias = data.qfrc_bias.copy()
        original = data.qpos.copy()
        step = 1e-6
        derivative = np.empty(model.nv)
        for dof in range(model.nv):
            energies = []
            for sign in (1, -1):
                data.qpos[:] = original
                velocity = np.zeros(model.nv)
                velocity[dof] = sign
                mujoco.mj_integratePos(model, data.qpos, velocity, step)
                mujoco.mj_forward(model, data)
                energies.append(-np.sum(model.body_mass[:, None] * data.xipos * model.opt.gravity))
            derivative[dof] = (energies[0] - energies[1]) / (2 * step)
        np.testing.assert_allclose(bias, derivative, atol=2e-7)

    def test_actuators_are_unit_joint_torques(self):
        model = self.model
        data = mujoco.MjData(model)
        data.ctrl[:] = np.linspace(-2, 3, model.nu)
        mujoco.mj_forward(model, data)
        expected = np.zeros(model.nv)
        for a in range(model.nu):
            j = int(model.actuator_trnid[a, 0])
            expected[model.jnt_dofadr[j]] = data.ctrl[a]
        np.testing.assert_allclose(data.qfrc_actuator, expected, atol=1e-14)

    def test_checked_in_derivation_is_reproducible(self):
        xml, _ = derive(SOURCE)
        self.assertEqual(xml, (ROOT / "model/ms_human_locomotion/human.xml").read_text())


if __name__ == "__main__":
    unittest.main()
