import math

import numpy as np
import scipy.spatial.transform as st

try:
    from loguru import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)


X_MIN = 0.2990
X_MAX = 1.0130
Y_MIN = -0.4330
Y_MAX = 0.3680
Z_MIN_BASE = 0.0715
Z_MAX = 0.6360
Z_ROT_MARGIN = 0.06


def get_safe_z_min(roll_angle):
    return Z_MIN_BASE + Z_ROT_MARGIN * abs(math.sin(roll_angle))


def _check_and_warn(x, y, z, safe_x, safe_y, safe_z):
    if abs(x - safe_x) > 1e-4 or abs(y - safe_y) > 1e-4 or abs(z - safe_z) > 1e-4:
        message = (
            "[Safety Boundary Triggered] "
            f"Target: ({x:.4f}, {y:.4f}, {z:.4f}) -> "
            f"Clipped: ({safe_x:.4f}, {safe_y:.4f}, {safe_z:.4f})"
        )
        if hasattr(logger, "opt"):
            logger.opt(colors=True).warning(f"<red>{message}</red>")
        else:
            logger.warning(message)


def clip_target_pose_10d(target_pose_10d):
    x, y, z = target_pose_10d[0], target_pose_10d[1], target_pose_10d[2]
    rx, ry, rz = target_pose_10d[3], target_pose_10d[4], target_pose_10d[5]

    try:
        rot = st.Rotation.from_rotvec([rx, ry, rz])
        roll = rot.as_euler("xyz")[0]
        z_min = get_safe_z_min(roll)
    except Exception:
        z_min = Z_MIN_BASE + Z_ROT_MARGIN

    safe_x = float(np.clip(x, X_MIN, X_MAX))
    safe_y = float(np.clip(y, Y_MIN, Y_MAX))
    safe_z = float(np.clip(z, z_min, Z_MAX))

    _check_and_warn(x, y, z, safe_x, safe_y, safe_z)

    target_pose_10d[0] = safe_x
    target_pose_10d[1] = safe_y
    target_pose_10d[2] = safe_z
    return target_pose_10d


def clip_target_pose_7d(target_tcp):
    x, y, z = target_tcp[0], target_tcp[1], target_tcp[2]
    qw, qx, qy, qz = target_tcp[3], target_tcp[4], target_tcp[5], target_tcp[6]

    try:
        rot = st.Rotation.from_quat([qx, qy, qz, qw], scalar_first=False)
        roll = rot.as_euler("xyz")[0]
        z_min = get_safe_z_min(roll)
    except Exception:
        z_min = Z_MIN_BASE + Z_ROT_MARGIN

    safe_x = float(np.clip(x, X_MIN, X_MAX))
    safe_y = float(np.clip(y, Y_MIN, Y_MAX))
    safe_z = float(np.clip(z, z_min, Z_MAX))

    _check_and_warn(x, y, z, safe_x, safe_y, safe_z)

    target_tcp[0] = safe_x
    target_tcp[1] = safe_y
    target_tcp[2] = safe_z
    return target_tcp


def is_eef_within_workspace_strict(eef_pos):
    if eef_pos is None:
        return True

    x, y, z = eef_pos[0], eef_pos[1], eef_pos[2]
    return X_MIN <= x <= X_MAX and Y_MIN <= y <= Y_MAX and Z_MIN_BASE <= z <= Z_MAX
