import os
import numpy as np
import mujoco as mj
from dm_control import mjcf

# Make GL selection robust on macOS even without viewer
os.environ.setdefault("MUJOCO_GL", "glfw")


def _load_model() -> tuple[mj.MjModel, mj.MjData]:
    here = os.path.dirname(__file__)
    scene_path = os.path.abspath(os.path.join(here, "assets", "scene.xml"))
    # Build via dm_control so we can ensure ee_site exists to match env.py
    scene = mjcf.from_path(scene_path)

    # If attachment_site exists, ensure ee_site coincides with it (same body/pose)
    att = scene.find('site', 'attachment_site')
    ee = scene.find('site', 'ee_site')
    if att is not None:
        # Try to place or move ee_site onto wrist_3_link with attachment_site's pose
        b = scene.find('body', 'wrist_3_link')
        if ee is None:
            if b is None:
                # Fallback to likely names if structure differs
                for nm in ['ee_link', 'tool0', 'flange', 'wrist_3']:
                    b = scene.find('body', nm)
                    if b is not None:
                        break
            if b is None:
                raise RuntimeError("Could not find an EE body to attach ee_site")
            b.add('site', name='ee_site', pos=list(att.pos), quat=list(att.quat) if hasattr(att, 'quat') else None,
                  size=[0.005], rgba=[1, 0, 0, 1])
        else:
            # Update existing ee_site to match attachment_site pose
            try:
                ee.pos = list(att.pos)
                if hasattr(att, 'quat'):
                    ee.quat = list(att.quat)
            except Exception:
                pass
    else:
        # No attachment_site; ensure ee_site exists on wrist_3_link origin as fallback
        if ee is None:
            b = scene.find('body', 'wrist_3_link')
            if b is None:
                for nm in ['ee_link', 'tool0', 'flange', 'wrist_3']:
                    b = scene.find('body', nm)
                    if b is not None:
                        break
            if b is None:
                raise RuntimeError("Could not find a wrist end-effector body to attach ee_site")
            b.add('site', name='ee_site', pos=[0, 0, 0], size=[0.005], rgba=[1, 0, 0, 1])

    xml = scene.to_xml_string()
    assets = scene.get_assets()
    model = mj.MjModel.from_xml_string(xml, assets=assets)
    data = mj.MjData(model)
    return model, data


def _find_site_id(model: mj.MjModel, name: str) -> int:
    try:
        sid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, name)
    except Exception:
        sid = -1
    return sid


def _fk(model: mj.MjModel, data: mj.MjData, q: np.ndarray, site_id: int) -> tuple[np.ndarray, np.ndarray]:
    data.qpos[: model.nq] = q
    data.qvel[:] = 0.0
    mj.mj_fwdPosition(model, data)
    pos = data.site_xpos[site_id].copy()
    rot = data.site_xmat[site_id].reshape(3, 3).copy()
    return pos, rot


def _r_err_from_R(R_des: np.ndarray, R_curr: np.ndarray) -> np.ndarray:
    R_err = R_des @ R_curr.T
    return 0.5 * np.array([
        R_err[2, 1] - R_err[1, 2],
        R_err[0, 2] - R_err[2, 0],
        R_err[1, 0] - R_err[0, 1],
    ])


def _build_R_from_z(z_des: np.ndarray, x_hint: np.ndarray | None) -> np.ndarray:
    z = z_des / np.linalg.norm(z_des)
    if x_hint is None or np.linalg.norm(x_hint) < 1e-9:
        x_hint = np.array([1.0, 0.0, 0.0])
    # Project x_hint onto plane orthogonal to z
    x_proj = x_hint - np.dot(x_hint, z) * z
    if np.linalg.norm(x_proj) < 1e-9:
        x_proj = np.array([1.0, 0.0, 0.0]) - np.dot(np.array([1.0, 0.0, 0.0]), z) * z
    x = x_proj / np.linalg.norm(x_proj)
    y = np.cross(z, x)
    return np.column_stack((x, y, z))


def _build_R_with_axis(axis_idx: int, axis_dir: np.ndarray, hint_vec: np.ndarray | None) -> np.ndarray:
    """Build a rotation matrix whose specified column (axis_idx) aligns with axis_dir.
    The remaining columns are constructed to form a right-handed orthonormal basis,
    using hint_vec to choose a stable perpendicular direction.
    """
    v = axis_dir / np.linalg.norm(axis_dir)
    if hint_vec is None or np.linalg.norm(hint_vec) < 1e-9:
        hint_vec = np.array([1.0, 0.0, 0.0])
    # Make a vector not parallel to v
    u_proj = hint_vec - np.dot(hint_vec, v) * v
    if np.linalg.norm(u_proj) < 1e-9:
        alt = np.array([0.0, 1.0, 0.0])
        u_proj = alt - np.dot(alt, v) * v
    u = u_proj / np.linalg.norm(u_proj)

    if axis_idx == 0:  # x := v
        x = v
        z = np.cross(x, u)
        if np.linalg.norm(z) < 1e-9:
            z = np.array([0.0, 0.0, 1.0])
        z = z / np.linalg.norm(z)
        y = np.cross(z, x)
    elif axis_idx == 1:  # y := v
        y = v
        x = np.cross(u, y)
        if np.linalg.norm(x) < 1e-9:
            x = np.array([1.0, 0.0, 0.0])
        x = x / np.linalg.norm(x)
        z = np.cross(x, y)
    else:  # axis_idx == 2, z := v
        z = v
        x = u
        y = np.cross(z, x)

    return np.column_stack((x, y, z))


def ik_solve(
    model: mj.MjModel,
    data: mj.MjData,
    site_id: int,
    target_pos: np.ndarray,
    target_z: np.ndarray | None = None,
    *,
    initial_q: np.ndarray | None = None,
    iters: int = 4000,
    tol_pos: float = 1e-3,
    tol_rot: float = 1e-2,
    step: float = 0.1,
    damping: float = 5e-3,
    w_pos: float = 1.0,
    w_rot: float = 0.3,
    two_stage: bool = True,
    target_axis: str | None = None,  # 'x', 'y', 'z' optionally prefixed with '-' for opposite
) -> tuple[np.ndarray, bool, int, float, float]:
    nq = int(model.nq)
    nv = int(model.nv)

    # Initial configuration
    if initial_q is None:
        try:
            kid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_KEY, "home")
            qk = model.key_qpos
            if getattr(qk, "ndim", 1) == 2:
                q0 = np.asarray(qk[kid, :], dtype=np.float64)
            else:
                q0 = np.asarray(qk[kid * nq : (kid + 1) * nq], dtype=np.float64)
        except Exception:
            q0 = np.zeros(nq, dtype=np.float64)
    else:
        q0 = np.asarray(initial_q, dtype=np.float64)
    q = q0.reshape(nq).copy()

    target_pos = np.asarray(target_pos, dtype=np.float64).reshape(3)
    R_des = None

    # Desired orientation from target axis alignment
    if target_z is not None or target_axis is not None:
        _, R0 = _fk(model, data, q, site_id)
        x_hint = R0[:, 0]
        if target_axis is not None:
            ax = target_axis.strip().lower()
            sign = -1.0 if ax.startswith('-') else 1.0
            ax = ax[1:] if ax.startswith('-') else ax
            idx = {'x': 0, 'y': 1, 'z': 2}.get(ax, 2)
            v = sign * np.array([0.0, 0.0, -1.0])  # align chosen local axis to world -Z
            R_des = _build_R_with_axis(idx, v, x_hint)
        else:
            R_des = _build_R_from_z(np.asarray(target_z, dtype=np.float64), x_hint)

    # Jacobian buffers in DOF space
    jacp = np.zeros((3, nv))
    jacr = np.zeros((3, nv))
    lamI6 = (damping ** 2) * np.eye(6)
    lamI3 = (damping ** 2) * np.eye(3)

    # Map DOFs to joints and qpos indices
    dof_jntid = model.dof_jntid[:nv]
    # Map each DOF to its corresponding qpos index (hinge joints ⇒ 1 DoF per joint)
    dof_qposid = model.jnt_qposadr[dof_jntid]
    jrange = model.jnt_range[dof_jntid].copy()  # (nv, 2)
    finite = np.isfinite(jrange).all(axis=1)
    mid = np.zeros(nv)
    mid[finite] = 0.5 * (jrange[finite, 0] + jrange[finite, 1])

    # Optionally run a position-only stage first for better convergence
    total_iters = 0
    for stage_idx in range(2 if two_stage else 1):
        orient_on = (R_des is not None and w_rot > 0.0) and (stage_idx == 1 or not two_stage)
        for it in range(1, iters + 1):
            ee_pos, R_curr = _fk(model, data, q, site_id)
            e_pos = target_pos - ee_pos

            use_rot = orient_on
            if use_rot:
                e_rot = _r_err_from_R(R_des, R_curr)
            else:
                e_rot = np.zeros(3)

            # Only allow early return on the final (orientation-on) stage.
            if (not two_stage or stage_idx == 1):
                if np.linalg.norm(e_pos) < tol_pos and (not use_rot or np.linalg.norm(e_rot) < tol_rot):
                    total_iters += it
                    return q, True, total_iters, float(np.linalg.norm(e_pos)), float(np.linalg.norm(e_rot))

            mj.mj_jacSite(model, data, jacp, jacr, site_id)
            Jp = jacp[:, :nv]
            Jr = jacr[:, :nv]

            if use_rot:
                J6 = np.vstack((w_pos * Jp, w_rot * Jr))
                e6 = np.concatenate((w_pos * e_pos, w_rot * e_rot))
                dq = J6.T @ np.linalg.solve(J6 @ J6.T + lamI6, e6)
            else:
                dq = Jp.T @ np.linalg.solve(Jp @ Jp.T + lamI3, e_pos)

            # No nullspace bias by default to avoid attraction to mids

            dq = np.clip(dq, -0.05, 0.05)

            # Update q via DOF->qpos mapping
            q = q.copy()
            np.add.at(q, dof_qposid, step * dq)

            # Clip per-DOF against joint limits when finite
            for i in range(nv):
                if finite[i]:
                    qi = dof_qposid[i]
                    lo, hi = jrange[i]
                    q[qi] = float(np.clip(q[qi], lo, hi))

        total_iters += iters

    ee_pos, R_curr = _fk(model, data, q, site_id)
    e_pos = target_pos - ee_pos
    e_rot = _r_err_from_R(R_des, R_curr) if R_des is not None else np.zeros(3)
    return q, False, total_iters, float(np.linalg.norm(e_pos)), float(np.linalg.norm(e_rot))


def main() -> None:
    model, data = _load_model()

    # Prefer ee_site (now colocated with attachment_site), fallback list kept for robustness
    site_name_candidates = [
        "ee_site",
        "attachment_site",
        "ee",
        "gripper_tip",
        "tool_site",
        "flange_site",
    ]
    sid = -1
    for nm in site_name_candidates:
        sid = _find_site_id(model, nm)
        if sid >= 0:
            site_name = nm
            break
    if sid < 0:
        raise RuntimeError("End-effector site not found. Consider adding a site to the wrist_3_link body.")

    # Target: position and -Z orientation
    target_pos = np.array([0.4, 0.0, 0.2905], dtype=np.float64)
    target_z = np.array([0.0, 0.0, -1.0], dtype=np.float64)

    # Initial guess: use 'home' keyframe if available, else zeros
    init_q = None

    # Auto-detect which local axis best represents the TCP "down" direction initially,
    # then align that axis to world -Z for IK stability.
    try:
        kid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_KEY, "home")
        qk = model.key_qpos
        if getattr(qk, "ndim", 1) == 2:
            q_guess = np.asarray(qk[kid, :], dtype=np.float64)
        else:
            nq = int(model.nq)
            q_guess = np.asarray(qk[kid * nq : (kid + 1) * nq], dtype=np.float64)
    except Exception:
        q_guess = np.zeros(int(model.nq), dtype=np.float64)

    _, R0 = _fk(model, data, q_guess, sid)
    world_neg_z = np.array([0.0, 0.0, -1.0])
    d0 = R0.T @ world_neg_z
    axis_idx = int(np.argmax(np.abs(d0)))
    axis_map = ['x', 'y', 'z']
    chosen_axis = axis_map[axis_idx]
    print("auto target_axis:", chosen_axis, "dot:", d0.tolist())

    q, success, iters, perr, rerr = ik_solve(
        model,
        data,
        sid,
        target_pos,
        target_z=target_z,
        initial_q=init_q,
        iters=30000,
        tol_pos=8e-4,
        tol_rot=8e-3,
        # Align the detected local axis to world -Z
        target_axis=chosen_axis,
    )

    print("ee_site:", site_name)
    try:
        bid = model.site_bodyid[sid]
        bname = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, int(bid))
    except Exception:
        bid, bname = -1, "<unknown>"
    local_pos = model.site_pos[sid] if hasattr(model, 'site_pos') else None
    local_quat = model.site_quat[sid] if hasattr(model, 'site_quat') else None
    print("site_id:", int(sid), "body:", bname, "local_pos:", local_pos, "local_quat:", local_quat)
    print("success:", success)
    print("iterations:", iters)
    print("position_error:", perr)
    print("orientation_error:", rerr)
    print("q (rad):", np.array2string(q, precision=6, suppress_small=False))
    print("q (deg):", np.degrees(q))

    # FK check
    pos, R = _fk(model, data, q, sid)
    print("FK position:", pos)
    print("FK orientation (R):\n", R)
    # Axis checks vs world -Z
    world_neg_z = np.array([0.0, 0.0, -1.0])
    dots = R.T @ world_neg_z  # dot for each local axis with world -Z
    print("dot(local x,y,z with -Z):", dots)


if __name__ == "__main__":
    main()


