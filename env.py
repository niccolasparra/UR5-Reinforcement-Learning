# ur5_p2p_env.py
import os
import math
import copy
import time
import sys
import numpy as np
import gymnasium as gym
# Ensure MuJoCo uses GLFW on macOS before importing mujoco
os.environ.setdefault("MUJOCO_GL", "glfw")
import mujoco as mj
from dm_control import mjcf

UR5_SCENE_XML = "assets/scene.xml"   # path to menagerie scene.xml

def rand_uniform(a, b):
    return np.random.uniform(a, b)

def quat_mul(q1, q2):
    # xyzw (MuJoCo also supports internal wxyz representation, but we use xyzw here)
    x1,y1,z1,w1 = q1
    x2,y2,z2,w2 = q2
    return np.array([
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2
    ])

class UR5P2PEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, render_mode=None, seed=0, target_mode: str = None, target_pos: list | tuple | np.ndarray = None):
        super().__init__()
        self.nq = 6
        self.nu = 6
        self.render_mode = render_mode
        self.rng = np.random.default_rng(seed)
        self.horizon = 1024
        self.step_i = 0

        # Action / observation spaces
        act_limit = np.ones(self.nu, dtype=np.float32) * 0.05  # per-step joint delta in radians
        self.action_space = gym.spaces.Box(-act_limit, act_limit, dtype=np.float32)

        # obs: q(6)+qd(6)+ee(3)+goal(3)+collision(1) = 19
        obs_high = np.ones(6+6+3+3+1, dtype=np.float32) * np.inf  # q(6)+qd(6)+ee(3)+goal(3)+collision(1)
        self.observation_space = gym.spaces.Box(-obs_high, obs_high, dtype=np.float32)

        self.viewer = None
        self.substeps = 10
        self.goal_quat = np.array([0,0,0,1], dtype=np.float64)  # orientation is fixed or optional
        self.start_pos = None
        self.viewer_failed = False
        self._viewer_builtin = False
        self.ctrl_mode = os.getenv("UR5_CTRL_MODE", "position")  # 'position' | 'torque' | 'velocity'
        self.debug = os.getenv("UR5_DEBUG", "1") == "1"
        self._once_logged = False

        self.prev_dist = 0.0
        self.q_target = None
        self.use_gravity_comp = True
        self.use_ik_init = os.getenv("UR5_USE_IK", "1") == "1"  # Enable IK initialization by default
        # Reward configuration (centralized for easier tuning)
        self.orient_axis = os.getenv("UR5_ORIENT_AXIS", "z")  # 'x','y','z','-x','-y','-z'
        self.reward_cfg = {
            "dist_success_thresh": 0.05,
            "time_penalty": 0.01,
            "collision_penalty": 10.0,
            "success_bonus": 100.0,
            "orient_weight": 0.0,
            "orient_power": 2.0,
            "progress_scale": 10.0,  # Increased from 5.0 for stronger progress reward
            "dist_weight": 0.5,  # NEW: continuous penalty for distance to goal
        }

        # Base MJCF load (UR5 + ground etc.), tables are rebuilt on each reset
        self.base_mjcf = mjcf.from_path(UR5_SCENE_XML)

        # UR5 end-effector site name (may differ per model)
        self.ee_site_name = "ee_site"  # adjust to match the model (or add a site on the terminal link)
        self._build_scene_and_model()

        self.target_mode = target_mode
        self.target_pos_cli = target_pos
        # Fixed random box for goal sampling: [xmin, xmax, ymin, ymax, zmin, zmax] (meters)
        self.target_random_box = np.array([
            -0.5, -0.35,   # X range
            -0.3, 0.3   ,   # Y range
            # 0.18, 0.45   # Z range (approx table top + margin)
        ], dtype=np.float64)

    # ---------- MJCF setup ----------
    def _build_scene_and_model(self):
        # Copy base_mjcf and construct the world
        scene = copy.deepcopy(self.base_mjcf)
        world = scene.worldbody

        # Ensure an EE site exists and matches TCP (attachment_site) if present
        att = scene.find('site', 'attachment_site')
        ee_site = scene.find('site', 'ee_site')
        if att is not None:
            # Align/Create ee_site on the wrist_3_link with attachment_site pose
            b = scene.find('body', 'wrist_3_link')
            if ee_site is None:
                if b is None:
                    for bname in ['tool0', 'ee_link', 'flange', 'tcp', 'wrist_3']:
                        b = scene.find('body', bname)
                        if b is not None:
                            break
                if b is not None:
                    b.add('site', name='ee_site', pos=list(att.pos),
                          quat=list(att.quat) if hasattr(att, 'quat') else None,
                          size=[0.005], rgba=[1,0,0,1])
            else:
                try:
                    ee_site.pos = list(att.pos)
                    if hasattr(att, 'quat'):
                        ee_site.quat = list(att.quat)
                except Exception:
                    pass
            self.ee_site_name = 'ee_site'
        else:
            # Fallback: try common names or create a site at the wrist_3_link origin
            if ee_site is None:
                for nm in ['ee', 'gripper_tip', 'tool_site', 'flange_site']:
                    ee_site = scene.find('site', nm)
                    if ee_site is not None:
                        self.ee_site_name = nm
                        break
            if ee_site is None:
                for bname in ['tool0', 'wrist_3_link', 'ee_link', 'flange', 'tcp', 'wrist_3']:
                    b = scene.find('body', bname)
                    if b is not None:
                        b.add('site', name='ee_site', pos=[0,0,0], size=[0.005], rgba=[1,0,0,1])
                        self.ee_site_name = 'ee_site'
                        break

        # Fixed tables (large boxes): use half-size
        table_size = np.array([0.8 / 2, 1.2 / 2, 0.1405 / 2])  # half-extent (x,y,z) = (1.0, 1.5, 0.1)m
        table_h = 0.0
        table_xy = np.array([0.3 + 0.8 / 2, 0.1405 / 2])

        table_names = ["table_1", "table_2"]
        table_centers = [
            np.array([table_xy[0], table_xy[1], table_h + table_size[2]], dtype=np.float64),
            np.array([-table_xy[0], table_xy[1], table_h + table_size[2]], dtype=np.float64),
        ]
        for name, center in zip(table_names, table_centers):
            world.add(
                "geom",
                name=name,
                type="box",
                size=table_size,
                pos=center.tolist(),
                rgba=[0.6, 0.4, 0.3, 1.0],
                friction=[rand_uniform(0.4, 0.9), 0.005, 0.0001]
            )

        self.table_names = table_names
        self.active_table = None

        # Sites to visualize start/goal positions (start: red, goal: green)
        world.add("site", name="start_site", size=[0.012], rgba=[1.0, 0.2, 0.2, 0.9], pos=[0,0,1])
        world.add("site", name="goal_site",  size=[0.012], rgba=[0.2, 1.0, 0.2, 0.9], pos=[0,0,1])

        # Compile model
        xml = scene.to_xml_string()
        assets = scene.get_assets()
        self.model = mj.MjModel.from_xml_string(xml, assets=assets)
        self.data = mj.MjData(self.model)

        # name → id caches
        self.geom_name2id = {mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_GEOM, i): i
                             for i in range(self.model.ngeom)}
        self.site_name2id = {mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_SITE, i): i
                             for i in range(self.model.nsite)}
        self.ee_sid   = self.site_name2id.get(self.ee_site_name, None)
        self.start_sid = self.site_name2id["start_site"]
        self.goal_sid  = self.site_name2id["goal_site"]

        if self.debug and not self._once_logged:
            print(f"[ENV] ee_site_name='{self.ee_site_name}', ee_sid={self.ee_sid}")
            print(f"[ENV] joints={self.nq}, actuators={self.nu}, ctrl_mode={self.ctrl_mode}")
            try:
                bid = self.model.site_bodyid[self.ee_sid] if self.ee_sid is not None else -1
                bname = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_BODY, int(bid)) if bid >= 0 else "None"
                loc = self.model.site_pos[self.ee_sid] if self.ee_sid is not None else None
                print(f"[ENV] ee_site body='{bname}' (id={bid}), local pos={loc}")
            except Exception:
                pass
            self._once_logged = True

        # PD gains (for position actuators, ctrl is interpreted as qref)
        self.kp = 2000.0
        self.kd = 400.0

    def _fallback_interactive_viewer(self):
        """Fallback loop using mujoco-python-viewer (GLFW)."""
        try:
            import mujoco_viewer
        except Exception as e:
            print("Fallback viewer unavailable:", e)
            print("Hint: fix ~/.config/mujoco_viewer permissions or install mujoco-python-viewer:")
            print("  pip install mujoco-python-viewer glfw")
            return
        viewer = None
        try:
            viewer = mujoco_viewer.MujocoViewer(self.model, self.data)
            print("[Interactive] Fallback viewer active. Press Ctrl+C to exit.")
            while True:
                if self.goal_pos is not None and hasattr(self, 'goal_sid'):
                    self.model.site_pos[self.goal_sid] = self.goal_pos
                self._pd_to_ctrl(self.data.qpos[:self.nq])
                mj.mj_step(self.model, self.data)
                viewer.render()
        except KeyboardInterrupt:
            pass
        except Exception as e:
            print("Fallback viewer closed:", e)
        finally:
            try:
                if viewer is not None:
                    viewer.close()
            except Exception:
                pass

    # ---------- Interactive viewer (blocking) ----------
    def interactive_viewer(self):
        """Open the built-in MuJoCo viewer and block until the window closes.
        Tip: Use the right panel -> Actuation/Controls sliders to move joints.
        """
        try:
            import mujoco.viewer as mjv
            # On macOS, the built-in viewer requires running under 'mjpython'
            if sys.platform == "darwin" and not os.path.basename(sys.executable).startswith("mjpython"):
                print("Tip: On macOS, run with 'mjpython env.py --interactive' for the built-in viewer.")
                return self._fallback_interactive_viewer()
        except Exception as e:
            print("Built-in viewer unavailable:", e)
            # Fallback to non-blocking path
            self.render_mode = "human"
            self.render()
            print("Fallback viewer started (non-blocking). Close the window or Ctrl+C to exit.")
            try:
                while True:
                    time.sleep(0.1)
            except KeyboardInterrupt:
                return

        # Ensure model/data exist
        if self.model is None or self.data is None:
            self._build_scene_and_model()
            mj.mj_forward(self.model, self.data)

        print("[Interactive] Use the right panel's Actuation sliders to move joints. Close the window to exit.")
        q_hold = self.data.qpos[:self.nq].copy()
        try:
            with mjv.launch_passive(self.model, self.data) as viewer:
                while viewer.is_running():
                    if self.goal_pos is not None and hasattr(self, 'goal_sid'):
                        self.model.site_pos[self.goal_sid] = self.goal_pos
                    self._pd_to_ctrl(q_hold)
                    mj.mj_step(self.model, self.data)
                    viewer.sync()
        except RuntimeError as e:
            print("Built-in viewer couldn't start:", e)
            return self._fallback_interactive_viewer()
    

    # ---------- Gym API ----------
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.step_i = 0

        # Rebuild scene (including table geometry)
        self._build_scene_and_model()

        # Initial joint configuration
        q0 = np.array([-3.483192, -1.723273, 2.236795, -2.084319, -1.570798, -1.904485], dtype=np.float64)
        q0 += self.rng.normal(0, 0.02, size=self.nq)
        
        # Use IK to initialize near start_pos if enabled
        if self.use_ik_init and self.start_pos is not None:
            try:
                q_ik = self._solve_ik(self.start_pos, q0)
                if q_ik is not None:
                    q0 = q_ik
                    if self.debug:
                        print("[RESET] IK solver succeeded, using IK solution for initial config")
                else:
                    if self.debug:
                        print("[RESET] IK solver failed, using default config")
            except Exception as e:
                if self.debug:
                    print(f"[RESET] IK solver error: {e}, using default config")
        
        self.data.qpos[:self.nq] = q0
        self.data.qvel[:] = 0.0
        mj.mj_forward(self.model, self.data)
        self.q_target = self.data.qpos[:self.nq].copy()
        # Immediately latch the current pose as the actuator reference so gravity
        # does not pull the arm down before the first env.step() call.
        self._pd_to_ctrl(self.q_target)

        # Run a short warmup to let the PD controller settle before the main loop.
        warmup_steps = max(0, int(getattr(self, "warmup_steps", 10)))
        if warmup_steps:
            self._step_sim(warmup_steps)

        # Set initial start position to a fixed point (near the center of the table)
        self.start_pos = np.array([0.4, 0.0, 0.1405 + 0.150], dtype=np.float64)
        self.goal_pos = None
        if self.start_pos is not None:
            self.model.site_pos[self.start_sid] = self.start_pos
        if self.goal_pos is not None:
            self.model.site_pos[self.goal_sid] = self.goal_pos

        # Goal sampling depending on target_mode
        mode = (self.target_mode or '').lower() if self.target_mode is not None else ''
        if mode == 'fixed':
            if self.target_pos_cli is not None:
                self.goal_pos = np.array(self.target_pos_cli, dtype=np.float64)
            else:
                self.goal_pos = np.array([-0.4, 0.0, 0.1405+0.15], dtype=np.float64)
        elif mode == 'random':
            xmin, xmax, ymin, ymax = self.target_random_box.tolist()
            x = self.rng.uniform(min(xmin, xmax), max(xmin, xmax))
            y = self.rng.uniform(min(ymin, ymax), max(ymin, ymax))
            # z = self.rng.uniform(min(zmin, zmax), max(zmin, zmax))
            z = 0.2905
            self.goal_pos = np.array([x, y, z], dtype=np.float64)
        elif mode == 'none':
            self.goal_pos = None

        if self.start_pos is not None:
            self.model.site_pos[self.start_sid] = self.start_pos
        if self.goal_pos is not None:
            self.model.site_pos[self.goal_sid] = self.goal_pos

        if self.debug:
            ee_after_reset = self._ee_pos()
            print(f"[RESET] use_ik_init={self.use_ik_init}")
            print("[RESET] start_pos=", self.start_pos)
            print("[RESET] goal_pos=", self.goal_pos)
            print(f"[RESET] ee_pos after init=", ee_after_reset)
            if self.start_pos is not None:
                print(f"[RESET] dist(ee, start)={np.linalg.norm(ee_after_reset - self.start_pos):.4f}")
            if (self.target_mode or '').lower() == 'random':
                print("[RESET] random_box=", self.target_random_box)

        # Initialize distance used for progress calculation (0 if no goal)
        
        ee_pos = self._ee_pos()
        if self.goal_pos is not None:
            self.prev_dist = np.linalg.norm(ee_pos - self.goal_pos)
        else:
            self.prev_dist = 0.0
        
        if self.render_mode == "human":
            self.render()

        obs = self._get_obs()
        info = {
            "q_start": self.data.qpos[:self.nq].copy(),
            "active_table": getattr(self, "active_table", None),
        }
        return obs, info
    

    def _pd_to_ctrl(self, q_des):
        """Drive joints depending on actuator mode.
        - position: ctrl := q_des (for position servos where ctrl is qref)
        - torque:   ctrl := Kp(q_des-q) - Kd qd
        - velocity: ctrl := Kp(q_des-q) - Kd qd (interpreted by velocity servos)
        """
        q  = self.data.qpos[:self.nq]
        qd = self.data.qvel[:self.nq]
        bias = np.zeros(self.nq, dtype=np.float64)
        if self.use_gravity_comp and self.model is not None:
            # qfrc_bias contains gravity, Coriolis, centrifugal terms at the current state.
            bias = self.data.qfrc_bias[:self.nq].copy()
        if self.ctrl_mode == "torque":
            u = self.kp * (q_des - q) - self.kd * qd + bias
            if self.model is not None:
                ctrl_range = self.model.actuator_ctrlrange[:self.nu]
                for j in range(self.nu):
                    lo, hi = ctrl_range[j]
                    if np.isfinite(lo) and np.isfinite(hi):
                        u[j] = np.clip(u[j], lo, hi)
            self.data.ctrl[:self.nu] = u
        elif self.ctrl_mode == "velocity":
            v = self.kp * (q_des - q) - self.kd * qd
            self.data.ctrl[:self.nu] = v
        else:  # position
            self.data.ctrl[:self.nu] = q_des[:self.nu]

    def _step_sim(self, n=1, hold_target=True):
        for _ in range(n):
            if hold_target and self.q_target is not None:
                self._pd_to_ctrl(self.q_target)
            mj.mj_step(self.model, self.data)

    def step(self, action):
        self.step_i += 1
        action = np.clip(action, self.action_space.low, self.action_space.high)

        if self.q_target is None or self.q_target.shape[0] != self.nq:
            self.q_target = self.data.qpos[:self.nq].copy()

        self.q_target = self.q_target.astype(np.float64) + action.astype(np.float64)

        if self.model is not None and getattr(self.model, "njnt", 0) >= self.nq:
            jrange = self.model.jnt_range[:self.nq]
            for j in range(self.nq):
                lo, hi = jrange[j]
                if np.isfinite(lo) and np.isfinite(hi):
                    self.q_target[j] = np.clip(self.q_target[j], lo, hi)

        # Update goal/start sites for visualization
        if self.goal_pos is not None:
            self.model.site_pos[self.goal_sid] = self.goal_pos
        if self.start_pos is not None:
            self.model.site_pos[self.start_sid] = self.start_pos
        self._step_sim(self.substeps)

        if self.render_mode == "human":
            self.render()

        # Collision check
        collided = self._check_collision()

        # Compute goal distance and success flag
        ee_pos = self._ee_pos()
        cfg = self.reward_cfg
        if self.goal_pos is not None:
            dist = np.linalg.norm(ee_pos - self.goal_pos)
            success = dist < cfg["dist_success_thresh"]
        else:
            dist = None
            success = False

        reward, reward_components = self._compute_reward(
            dist=dist,
            prev_dist=self.prev_dist,
            collided=collided,
            success=success,
        )
        self.prev_dist = dist if dist is not None else 0.0

        terminated = bool(success or collided)
        truncated = self.step_i >= self.horizon

        obs = self._get_obs()
        info = {
            "success": success,
            "collision": collided,
            "dist": dist,
            "reward_components": reward_components,
            "active_table": getattr(self, "active_table", None),
        }
        return obs, reward, terminated, truncated, info

    def _compute_reward(self, dist, prev_dist, collided: bool, success: bool):
        """
        Centralized reward computation.
        If dist is None, only time/collision/success terms are applied (no distance-based progress).
        """
        cfg = self.reward_cfg

        # Distance-based progress reward (relative improvement)
        progress = 0.0
        if dist is not None and prev_dist != 0.0:
            denom = max(dist, 1e-6)
            progress = cfg["progress_scale"] * (prev_dist - dist) / denom

        # Distance penalty (absolute distance to goal)
        dist_penalty = 0.0
        if dist is not None:
            dist_penalty = -cfg["dist_weight"] * dist

        reward = 0.0
        reward += progress
        reward += dist_penalty
        reward -= cfg["time_penalty"]

        # Orientation reward (align a chosen EE axis with world -Z)
        orient_reward = 0.0
        if self.ee_sid is not None and cfg["orient_weight"] != 0.0:
            try:
                R = self.data.site_xmat[self.ee_sid].reshape(3, 3)
                axis = (self.orient_axis or "z").lower().strip()
                sign = -1.0 if axis.startswith("-") else 1.0
                axis = axis[1:] if axis.startswith("-") else axis
                idx = {"x": 0, "y": 1, "z": 2}.get(axis, 2)
                v = R[:, idx] * sign
                align = float(np.dot(v, np.array([0.0, 0.0, -1.0])))
                orient_reward = cfg["orient_weight"] * (align ** cfg["orient_power"])
                reward += orient_reward
            except Exception:
                pass

        collision_reward = 0.0
        if collided:
            collision_reward = -cfg["collision_penalty"]
            reward += collision_reward

        success_reward = 0.0
        if success:
            success_reward = cfg["success_bonus"]
            reward += success_reward

        components = {
            "progress": progress,
            "dist_penalty": dist_penalty,
            "time": -cfg["time_penalty"],
            "orient": orient_reward,
            "collision": collision_reward,
            "success": success_reward,
        }
        return reward, components

    def _ee_pos(self):
        if self.ee_sid is None:
            return np.zeros(3)
        mj.mj_forward(self.model, self.data)
        return self.data.site_xpos[self.ee_sid].copy()

    def _get_obs(self, q_goal=None):
        q = self.data.qpos[:self.nq].copy()
        qd = self.data.qvel[:self.nq].copy()
        ee = self._ee_pos()
        goal = self.goal_pos if self.goal_pos is not None else np.zeros(3)
        coll = float(self._check_collision())
        obs = np.concatenate([
            q,
            qd,
            ee.astype(np.float64),
            goal.astype(np.float64),
            np.array([coll], dtype=np.float64)
        ]).astype(np.float32)
        return obs

    def _solve_ik(self, target_pos: np.ndarray, initial_guess: np.ndarray) -> np.ndarray | None:
        """Simple IK solver using damped least squares.
        Returns joint configuration or None if failed.
        """
        q = initial_guess.copy()
        target_pos = np.asarray(target_pos, dtype=np.float64)
        
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        damping = 1e-3
        lamI3 = (damping ** 2) * np.eye(3)
        
        jrange = self.model.jnt_range[:self.nq]
        
        for it in range(500):  # max iterations
            self.data.qpos[:self.nq] = q
            self.data.qvel[:] = 0.0
            mj.mj_forward(self.model, self.data)
            
            ee_pos = self.data.site_xpos[self.ee_sid].copy()
            e_pos = target_pos - ee_pos
            
            if np.linalg.norm(e_pos) < 0.01:  # 1cm tolerance
                return q
            
            mj.mj_jacSite(self.model, self.data, jacp, jacr, self.ee_sid)
            Jp = jacp[:, :self.nq]
            
            dq = Jp.T @ np.linalg.solve(Jp @ Jp.T + lamI3, e_pos)
            dq = np.clip(dq, -0.2, 0.2)
            
            q = q + 0.2 * dq
            # Clip to joint limits
            for j in range(self.nq):
                lo, hi = jrange[j]
                if np.isfinite(lo) and np.isfinite(hi):
                    q[j] = np.clip(q[j], lo, hi)
        
        # Failed to converge
        return None

    def _check_collision(self):
        mj.mj_forward(self.model, self.data)
        table_ids = {self.geom_name2id[n] for n in getattr(self, "table_names", []) if n in self.geom_name2id}
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            if c.geom1 in table_ids or c.geom2 in table_ids:
                # Ensure that at least one of the bodies is not a table (table-table contact is ignored)
                if not (c.geom1 in table_ids and c.geom2 in table_ids):
                    return True
        return False

    def render(self):
        if self.render_mode == "rgb_array":
            # Off-screen rendering for video recording
            if not hasattr(self, "_offscreen_renderer"):
                from mujoco import Renderer
                # HD resolution: 1280x720 (instead of 640x480)
                self._offscreen_renderer = Renderer(self.model, height=480, width=1280)
            
            # Update visualization sites
            if self.start_pos is not None:
                self.model.site_pos[self.start_sid] = self.start_pos
            if self.goal_pos is not None:
                self.model.site_pos[self.goal_sid] = self.goal_pos
            
            # Forward kinematics to update visualization
            mj.mj_forward(self.model, self.data)
            
            # Render and return RGB array
            self._offscreen_renderer.update_scene(self.data)
            return self._offscreen_renderer.render()
        
        elif self.render_mode == "human":
            if getattr(self, "viewer_failed", False):
                return
            if self.viewer is None:
                # 1) Try MuJoCo built-in viewer first
                try:
                    import mujoco.viewer as mjv
                    self.viewer = mjv.launch_passive(self.model, self.data)
                    self._viewer_builtin = True
                except Exception:
                    # 2) Fallback to mujoco-python-viewer
                    try:
                        import mujoco_viewer
                        self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data)
                        self._viewer_builtin = False
                    except Exception as e:
                        if not self.viewer_failed:
                            print("Viewer unavailable (will not retry):", e)
                        self.viewer_failed = True
                        return
            # Draw a frame
            try:
                if self._viewer_builtin:
                    if hasattr(self.viewer, "is_running") and not self.viewer.is_running():
                        self.viewer_failed = True
                        return
                    self.viewer.sync()
                else:
                    self.viewer.render()
            except Exception:
                self.viewer_failed = True
                return

    def close(self):
        if self.viewer is not None:
            try:
                if hasattr(self.viewer, "close"):
                    self.viewer.close()
            except Exception:
                pass
            self.viewer = None
        
        # Clean up offscreen renderer if it exists
        if hasattr(self, "_offscreen_renderer"):
            try:
                self._offscreen_renderer.close()
            except Exception:
                pass
            del self._offscreen_renderer

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--interactive", action="store_true", help="Open a blocking viewer; close the window to exit")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-mode", type=str, default="random",
                        help="How to set goal position: fixed|random|none (random uses built-in region)")
    parser.add_argument("--target-pos", type=float, nargs=3, default=None)
    args = parser.parse_args()

    env = UR5P2PEnv(render_mode="human" if args.interactive else None,
                    seed=args.seed,
                    target_mode=args.target_mode,
                    target_pos=args.target_pos)

    if args.interactive:
        env.reset()
        env.interactive_viewer()
        env.close()
    else:
        obs, info = env.reset()
        done = False
        total = 0.0
        while not done:
            a = np.zeros(env.action_space.shape, dtype=np.float32)
            obs, r, term, trunc, info = env.step(a)
            total += r
            done = term or trunc
        print("episode done. total reward:", total)
