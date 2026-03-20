"""Utility kinematics helpers aligned with the MuJoCo UR5 environment."""

from __future__ import annotations

import dataclasses
from typing import Optional

import mujoco as mj
import numpy as np

from env import UR5P2PEnv


@dataclasses.dataclass
class IKSolution:
    q: np.ndarray
    success: bool
    iterations: int
    position_error: float
    orientation_error: float


class UR5Kinematics:
    """Kinematics driven directly by the MuJoCo model used in UR5P2PEnv."""

    def __init__(self, seed: int = 0) -> None:
        # Use target_mode='none' to avoid random goal generation noise.
        self._env = UR5P2PEnv(render_mode=None, seed=seed, target_mode="none")
        # Build model/data once by resetting.
        obs, info = self._env.reset()
        self._nq = self._env.nq
        self._ee_sid = self._env.ee_sid
        if self._ee_sid is None:
            raise RuntimeError("End-effector site not found in the MuJoCo model.")
        # Cache joint limits from the MuJoCo model.
        self._jnt_limits = self._env.model.jnt_range[: self._nq].copy()

        # Pre-allocate buffers for MuJoCo Jacobian calls.
        self._jacp = np.zeros((3, self._env.model.nv))
        self._jacr = np.zeros((3, self._env.model.nv))

    # ------------------------------------------------------------------
    # Forward kinematics helpers
    # ------------------------------------------------------------------
    def forward_kinematics(self, q: np.ndarray) -> np.ndarray:
        """Return a 4x4 pose matrix (world->EE) using MuJoCo computation."""
        self._set_configuration(q)
        if hasattr(mj, "mj_fwdPosition"):
            mj.mj_fwdPosition(self._env.model, self._env.data)
        else:
            mj.mj_forward(self._env.model, self._env.data)
        pos = self._env.data.site_xpos[self._ee_sid].copy()
        rot = self._env.data.site_xmat[self._ee_sid].reshape(3, 3).copy()
        T = np.eye(4)
        T[:3, :3] = rot
        T[:3, 3] = pos
        return T

    # ------------------------------------------------------------------
    # Inverse kinematics (damped least squares)
    # ------------------------------------------------------------------
    def inverse_kinematics(
        self,
        target_pos: np.ndarray,
        target_rot: Optional[np.ndarray] = None,
        target_z: Optional[np.ndarray] = None,
        *,
        initial_guess: Optional[np.ndarray] = None,
        tol_pos: float = 1e-3,
        tol_rot: float = 1e-2,
        iters: int = 40000,
        damping: float = 1e-3,
        step: float = 0.25,
        w_pos: float = 1.0,
        w_rot: float = 0.7,
    ) -> IKSolution:
        """Damped-least-squares IK matching position and optional orientation.

        Parameters mirror UR5P2PEnv._ik_dls for consistency.
        """

        target_pos = np.asarray(target_pos, dtype=np.float64).reshape(3)
        if target_rot is not None:
            target_rot = np.asarray(target_rot, dtype=np.float64).reshape(3, 3)
        if target_rot is None and target_z is not None:
            target_z = np.asarray(target_z, dtype=np.float64)
            if np.linalg.norm(target_z) == 0:
                target_z = None
            else:
                target_z = target_z / np.linalg.norm(target_z)

        if initial_guess is None:
            q = self._env.data.qpos[: self._nq].copy()
        else:
            q = np.asarray(initial_guess, dtype=np.float64).copy()

        # Nullspace bias toward joint mid-range when finite limits exist (avoid NaNs).
        jrange = self._jnt_limits.copy()
        finite = np.isfinite(jrange).all(axis=1)
        mid = np.zeros(self._nq)
        mid[finite] = 0.5 * (jrange[finite, 0] + jrange[finite, 1])

        lamI6 = (damping ** 2) * np.eye(6)
        lamI3 = (damping ** 2) * np.eye(3)

        # Pre-forward once to build a fixed desired orientation when using target_z.
        self._set_configuration(q)
        if hasattr(mj, "mj_fwdPosition"):
            mj.mj_fwdPosition(self._env.model, self._env.data)
        else:
            mj.mj_forward(self._env.model, self._env.data)

        R_des = None
        if target_rot is not None:
            R_des = target_rot
        elif target_z is not None:
            z_des = target_z  # already normalized above if provided
            R_curr0 = self._env.data.site_xmat[self._ee_sid].reshape(3, 3)
            # Choose x-axis close to current x but orthogonal to z_des
            x_curr0 = R_curr0[:, 0]
            x_proj0 = x_curr0 - np.dot(x_curr0, z_des) * z_des
            if np.linalg.norm(x_proj0) < 1e-6:
                x_proj0 = np.array([1.0, 0.0, 0.0]) - np.dot(np.array([1.0, 0.0, 0.0]), z_des) * z_des
            x_des = x_proj0 / np.linalg.norm(x_proj0)
            y_des = np.cross(z_des, x_des)
            R_des = np.column_stack((x_des, y_des, z_des))

        for it in range(iters):
            self._set_configuration(q)
            if hasattr(mj, "mj_fwdPosition"):
                mj.mj_fwdPosition(self._env.model, self._env.data)
            else:
                mj.mj_forward(self._env.model, self._env.data)
            ee_pos = self._env.data.site_xpos[self._ee_sid].copy()
            e_pos = target_pos - ee_pos

            use_rot = R_des is not None
            if use_rot:
                R_curr = self._env.data.site_xmat[self._ee_sid].reshape(3, 3)
                R_err = R_des @ R_curr.T
                e_rot = 0.5 * np.array([
                    R_err[2, 1] - R_err[1, 2],
                    R_err[0, 2] - R_err[2, 0],
                    R_err[1, 0] - R_err[0, 1],
                ])
            else:
                e_rot = np.zeros(3)

            if np.linalg.norm(e_pos) < tol_pos and (not use_rot or np.linalg.norm(e_rot) < tol_rot):
                return IKSolution(q=q, success=True, iterations=it + 1, position_error=np.linalg.norm(e_pos), orientation_error=np.linalg.norm(e_rot))

            mj.mj_jacSite(self._env.model, self._env.data, self._jacp, self._jacr, self._ee_sid)
            Jp = self._jacp[:, : self._nq]
            Jr = self._jacr[:, : self._nq]

            if use_rot and w_rot > 0.0:
                J6 = np.vstack((w_pos * Jp, w_rot * Jr))
                e6 = np.concatenate((w_pos * e_pos, w_rot * e_rot))
                dq = J6.T @ np.linalg.solve(J6 @ J6.T + lamI6, e6)
            else:
                dq = Jp.T @ np.linalg.solve(Jp @ Jp.T + lamI3, e_pos)

            bias = np.zeros_like(q)
            bias[finite] = mid[finite] - q[finite]
            dq += 0.05 * bias
            # Limit per-joint update to improve stability
            dq = np.clip(dq, -0.2, 0.2)

            q = q + step * dq
            q = np.clip(q, jrange[:, 0], jrange[:, 1])

        return IKSolution(
            q=q,
            success=False,
            iterations=iters,
            position_error=np.linalg.norm(e_pos),
            orientation_error=np.linalg.norm(e_rot),
        )

    # ------------------------------------------------------------------
    def _set_configuration(self, q: np.ndarray) -> None:
        q = np.asarray(q, dtype=np.float64)
        if q.shape[0] != self._nq:
            raise ValueError(f"Expected {self._nq} joints, got {q.shape[0]}")
        self._env.data.qpos[: self._nq] = q
        self._env.data.qvel[: self._nq] = 0.0


if __name__ == "__main__":
    kin = UR5Kinematics()

    target_pose = np.eye(4)
    target_pose[:3, 3] = np.array([0.4, 0.0, 0.2905])
    init_q = np.array([0.0, -3.0, 1.5, 1.0, -1.1, 0.0])

    # Desired orientation: align EE z-axis with world -z
    sol = kin.inverse_kinematics(target_pose[:3, 3], target_z=np.array([0.0, 0.0, -1.0]), initial_guess=init_q)
    print("success:", sol.success)
    print("iterations:", sol.iterations)
    print("q (rad):", sol.q)

    fk_T = kin.forward_kinematics(sol.q)
    print("FK position:", fk_T[:3, 3])
    print("FK orientation:", fk_T[:3, :3])
