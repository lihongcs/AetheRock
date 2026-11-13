import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import cv2  # noqa: E402
import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from umi_utils.pose_util import pose_to_mat, quat_to_rot  # noqa: E402


DEFAULT_OFFSET_MM = np.array([107.68, 39.0, -76.86], dtype=np.float64)
DESIRED_EEF_AXES = np.column_stack([
    np.array([-1.0, 0.0, 0.0]),  # EEF X / back
    np.array([0.0, 1.0, 0.0]),   # EEF Y / left
    np.array([0.0, 0.0, -1.0]),  # EEF Z / down
])
DEFAULT_VIDEO_VIEWS = (
    ("view_x_side", 18.0, 12.0),
    ("view_y_side", 18.0, 102.0),
    ("view_top_oblique", 62.0, 45.0),
)


def extract_num(name):
    nums = ''.join(filter(str.isdigit, name))
    return int(nums) if nums else 999999


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def get_pose_transform_mats(raw_pose, use_offset, offset_m, offset_frame):
    raw_pose = np.asarray(raw_pose, dtype=np.float64)
    pose_mat = pose_to_mat(quat_to_rot(raw_pose))

    additional_transform_matrix = np.eye(4)
    additional_r_y = Rotation.from_euler('y', 90, degrees=True).as_matrix()
    additional_r_z = Rotation.from_euler('z', 180, degrees=True).as_matrix()
    additional_transform_matrix[:3, :3] = additional_r_y @ additional_r_z

    pose_mat_out = pose_mat @ additional_transform_matrix
    if use_offset:
        if offset_frame == "pose":
            offset_transform = np.eye(4)
            offset_transform[:3, 3] = offset_m
            pose_mat_out = pose_mat_out @ offset_transform
        elif offset_frame == "home":
            pose_mat_out[:3, 3] += DESIRED_EEF_AXES @ offset_m
        else:
            raise ValueError(f"Unknown offset_frame: {offset_frame}")
    return pose_mat_out


def load_htc_pose_folder(tracker_root):
    htc_pose_folder = tracker_root / "htc_pose"
    if not htc_pose_folder.exists():
        htc_pose_folder = tracker_root

    pose_times = []
    pose_values = []
    for pose_file in sorted(htc_pose_folder.glob("*.json"),
                            key=lambda p: extract_num(p.name)):
        try:
            pose_data = load_json(pose_file)
        except json.JSONDecodeError:
            continue
        pose_times.extend(pose_data["timestamps"])
        pose_values.extend(pose_data["data"])

    order = np.argsort(pose_times)
    pose_times = np.asarray(pose_times, dtype=np.float64)[order]
    pose_values = [pose_values[i] for i in order]
    return pose_times, pose_values


def load_wrist_times(epoch_root):
    wrist_root = epoch_root / "wrist"
    if not wrist_root.exists():
        wrist_root = epoch_root

    wrist_times = []
    for ts_file in sorted(wrist_root.glob("wrist*.json"),
                          key=lambda p: extract_num(p.name)):
        try:
            ts_data = load_json(ts_file)
        except json.JSONDecodeError:
            continue
        if isinstance(ts_data, dict) and "timestamps" in ts_data:
            ts_data = ts_data["timestamps"]
        wrist_times.extend(ts_data)
    return np.asarray(wrist_times, dtype=np.float64)


def find_closest_indices(target_times, source_times, threshold):
    indices = []
    source_idx = 0
    for target_time in target_times:
        while source_idx + 1 < len(source_times) and source_times[source_idx + 1] <= target_time:
            source_idx += 1

        candidates = [source_idx]
        if source_idx + 1 < len(source_times):
            candidates.append(source_idx + 1)
        best_idx = min(candidates, key=lambda i: abs(source_times[i] - target_time))

        if abs(source_times[best_idx] - target_time) <= threshold:
            indices.append(best_idx)
    return np.asarray(indices, dtype=np.int64)


def resolve_roots(data_root, cap_id, tracker_root, handcap_root):
    data_root = Path(data_root).expanduser().resolve()
    if tracker_root is not None:
        tracker_root = Path(tracker_root).expanduser().resolve()
    else:
        matches = [
            p for p in data_root.iterdir()
            if p.is_dir() and p.name.startswith("htc") and str(cap_id) in p.name
        ]
        if not matches:
            raise FileNotFoundError(f"No htc folder matching cap_id={cap_id} under {data_root}")
        tracker_root = sorted(matches, key=lambda p: p.name)[0]

    if handcap_root is not None:
        handcap_root = Path(handcap_root).expanduser().resolve()
    else:
        suffix = tracker_root.name.split("_")[-1]
        handcap_root = data_root / f"handcap_{suffix}"
        if not handcap_root.exists():
            raise FileNotFoundError(f"Expected handcap folder not found: {handcap_root}")
    return tracker_root, handcap_root


def set_axes_equal(ax, points):
    center = points.mean(axis=0)
    max_range = np.ptp(points, axis=0).max()
    if max_range <= 1e-9:
        max_range = 0.1
    radius = max_range * 0.55
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def draw_frame(ax, mat, length, label_prefix="", alpha=1.0):
    origin = mat[:3, 3]
    axes = mat[:3, :3]
    colors = ("#d62728", "#2ca02c", "#1f77b4")
    labels = ("X/back", "Y/left", "Z/down")
    for i, (color, label) in enumerate(zip(colors, labels)):
        vec = axes[:, i] * length
        ax.quiver(
            origin[0], origin[1], origin[2],
            vec[0], vec[1], vec[2],
            color=color, linewidth=1.4, alpha=alpha, arrow_length_ratio=0.18)
        if label_prefix:
            end = origin + vec
            ax.text(end[0], end[1], end[2], f"{label_prefix}{label}",
                    color=color, fontsize=7)


def draw_world_frame(ax, origin, length):
    world_axes = np.eye(3)
    labels = ("World X/front", "World Y/left", "World Z/up")
    colors = ("#8b0000", "#006400", "#00008b")
    for i, (label, color) in enumerate(zip(labels, colors)):
        vec = world_axes[:, i] * length
        ax.quiver(
            origin[0], origin[1], origin[2],
            vec[0], vec[1], vec[2],
            color=color, linewidth=2.0, arrow_length_ratio=0.15)
        end = origin + vec
        ax.text(end[0], end[1], end[2], label, color=color, fontsize=9)


def draw_fixed_eef_frame(ax, origin, length, name, alpha=1.0):
    axis_colors = ("#d62728", "#2ca02c", "#1f77b4")
    axis_labels = ("X/back", "Y/left", "Z/down")
    for vec, color, label in zip(DESIRED_EEF_AXES.T, axis_colors, axis_labels):
        vec = vec * length
        ax.quiver(
            origin[0], origin[1], origin[2],
            vec[0], vec[1], vec[2],
            color=color, linewidth=2.4, alpha=alpha, arrow_length_ratio=0.18)
        end = origin + vec
        ax.text(end[0], end[1], end[2], f"{name} {label}",
                color=color, fontsize=8)


def draw_fixed_eef_frame_no_labels(ax, origin, length, alpha=1.0):
    axis_colors = ("#d62728", "#2ca02c", "#1f77b4")
    for vec, color in zip(DESIRED_EEF_AXES.T, axis_colors):
        vec = vec * length
        ax.quiver(
            origin[0], origin[1], origin[2],
            vec[0], vec[1], vec[2],
            color=color, linewidth=2.0, alpha=alpha, arrow_length_ratio=0.18)


def draw_eef_orientation(ax, mat, origin, length, orientation_frame, alpha=1.0, labels=False):
    if orientation_frame == "home":
        axes = DESIRED_EEF_AXES
    elif orientation_frame == "pose":
        axes = mat[:3, :3]
    else:
        return

    axis_colors = ("#d62728", "#2ca02c", "#1f77b4")
    axis_labels = ("X/back", "Y/left", "Z/down")
    for vec, color, label in zip(axes.T, axis_colors, axis_labels):
        vec = vec * length
        ax.quiver(
            origin[0], origin[1], origin[2],
            vec[0], vec[1], vec[2],
            color=color, linewidth=2.4, alpha=alpha, arrow_length_ratio=0.20)
        if labels:
            end = origin + vec
            ax.text(end[0], end[1], end[2], label, color=color, fontsize=9)


def set_axes_from_limits(ax, mins, maxs):
    ax.set_xlim(mins[0], maxs[0])
    ax.set_ylim(mins[1], maxs[1])
    ax.set_zlim(mins[2], maxs[2])


def canvas_to_bgr(fig):
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    rgba = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
    rgb = rgba[:, :, :3]
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def make_trajectory_video(
        pose_mats_tracker, pose_mats_eef, pose_mats_tcp, output_path, title,
        fps=20, max_video_frames=220, elev=18, azim=12, view_label=None,
        orientation_frame="none"):
    tracker_pos = pose_mats_tracker[:, :3, 3]
    eef_pos = pose_mats_eef[:, :3, 3]
    tcp_pos = pose_mats_tcp[:, :3, 3]
    all_pos = np.concatenate([tracker_pos, eef_pos, tcp_pos], axis=0)
    deltas = tcp_pos - eef_pos
    offset_norm = np.linalg.norm(deltas, axis=1)

    total = len(tracker_pos)
    if total <= max_video_frames:
        frame_indices = np.arange(total, dtype=int)
    else:
        frame_indices = np.linspace(0, total - 1, max_video_frames, dtype=int)

    center = all_pos.mean(axis=0)
    max_range = np.ptp(all_pos, axis=0).max()
    frame_len = max(max_range * 0.10, 0.04)
    radius = max(max_range * 0.70, 0.18)
    mins = center - radius
    maxs = center + radius

    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(14, 9), dpi=140)
    ax = fig.add_subplot(111, projection="3d")
    width, height = fig.canvas.get_width_height()
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {output_path}")

    try:
        for video_i, idx in enumerate(frame_indices):
            ax.cla()
            set_axes_from_limits(ax, mins, maxs)
            ax.view_init(elev=elev, azim=azim)
            ax.set_box_aspect((1, 1, 1))

            ax.plot(tracker_pos[:idx + 1, 0], tracker_pos[:idx + 1, 1], tracker_pos[:idx + 1, 2],
                    color="#2ca02c", linewidth=2.0, label="Tracker")
            ax.plot(eef_pos[:idx + 1, 0], eef_pos[:idx + 1, 1], eef_pos[:idx + 1, 2],
                    color="#ff7f0e", linewidth=3.0, label="EEF")
            ax.plot(tcp_pos[:idx + 1, 0], tcp_pos[:idx + 1, 1], tcp_pos[:idx + 1, 2],
                    color="#1f77b4", linewidth=3.0, label="TCP")
            ax.plot(tracker_pos[idx:, 0], tracker_pos[idx:, 1], tracker_pos[idx:, 2],
                    color="#2ca02c", linewidth=1.0, alpha=0.15)
            ax.plot(eef_pos[idx:, 0], eef_pos[idx:, 1], eef_pos[idx:, 2],
                    color="#ff7f0e", linewidth=1.0, alpha=0.18)
            ax.plot(tcp_pos[idx:, 0], tcp_pos[idx:, 1], tcp_pos[idx:, 2],
                    color="#1f77b4", linewidth=1.0, alpha=0.18)

            p_track = tracker_pos[idx]
            p_eef = eef_pos[idx]
            p_tcp = tcp_pos[idx]
            delta = p_tcp - p_eef
            ax.scatter(*p_track, color="#2ca02c", s=60)
            ax.scatter(*p_eef, color="#ff7f0e", s=80)
            ax.scatter(*p_tcp, color="#1f77b4", s=80)
            ax.plot([p_track[0], p_eef[0]], [p_track[1], p_eef[1]], [p_track[2], p_eef[2]], color="0.4", linestyle="--")
            ax.quiver(
                p_eef[0], p_eef[1], p_eef[2],
                delta[0], delta[1], delta[2],
                color="0.05", linewidth=2.6, arrow_length_ratio=0.13)
            draw_eef_orientation(
                ax, pose_mats_eef[idx], p_eef, frame_len,
                orientation_frame, alpha=0.6, labels=False)
            draw_eef_orientation(
                ax, pose_mats_tcp[idx], p_tcp, frame_len,
                orientation_frame, alpha=0.95, labels=True)
            
            # Draw world frame acting as a compass at the bottom corner of the plot
            world_origin = np.array([mins[0], mins[1], mins[2]])
            draw_world_frame(ax, world_origin, frame_len * 1.5)

            ax.set_xlabel("X (m)", labelpad=12, fontsize=12)
            ax.set_ylabel("Y (m)", labelpad=12, fontsize=12)
            ax.set_zlabel("Z (m)", labelpad=12, fontsize=12)
            ax.tick_params(axis="both", which="major", labelsize=10)
            ax.legend(loc="upper left", fontsize=12)
            ax.set_title(
                f"{title}\n"
                f"{view_label or 'trajectory view'} | "
                f"frame {idx + 1}/{total}\n"
                f"delta: X={delta[0]:+.3f}m, Y={delta[1]:+.3f}m, "
                f"Z={delta[2]:+.3f}m, norm={offset_norm[idx]:.3f}m",
                fontsize=14, pad=18)

            writer.write(canvas_to_bgr(fig))
    finally:
        writer.release()
        plt.close(fig)

    return (
        f"video_frames={len(frame_indices)}, fps={fps}, "
        f"view_elev={elev}, view_azim={azim}, saved={output_path}"
    )


def get_view_output_path(base_output, view_name):
    base_output = Path(base_output).expanduser().resolve()
    return base_output.with_name(f"{base_output.stem}_{view_name}{base_output.suffix}")


def make_three_view_videos(
        pose_mats_tracker, pose_mats_eef, pose_mats_tcp, base_output_path, title,
        fps=20, max_video_frames=220, orientation_frame="none"):
    stats = []
    for view_name, elev, azim in DEFAULT_VIDEO_VIEWS:
        output_path = get_view_output_path(base_output_path, view_name)
        if view_name == "view_x_side":
            view_label = "view from +World X/front, slightly +World Y/left side"
        elif view_name == "view_y_side":
            view_label = "view from +World Y/left side"
        else:
            view_label = "top oblique view"
        stats.append(make_trajectory_video(
            pose_mats_tracker,
            pose_mats_eef,
            pose_mats_tcp,
            output_path,
            title,
            fps=fps,
            max_video_frames=max_video_frames,
            elev=elev,
            azim=azim,
            view_label=view_label,
            orientation_frame=orientation_frame))
    return stats


def draw_convention_reference(ax):
    ax.set_title("Coordinate Convention")
    origin = np.zeros(3)

    world_axes = {
        "World X/front": (np.array([1.0, 0.0, 0.0]), "#8b0000"),
        "World Y/left": (np.array([0.0, 1.0, 0.0]), "#006400"),
        "World Z/up": (np.array([0.0, 0.0, 1.0]), "#00008b"),
    }
    eef_axes = {
        "EEF X/back": (np.array([-1.0, 0.0, 0.0]), "#d62728"),
        "EEF Y/left": (np.array([0.0, 1.0, 0.0]), "#2ca02c"),
        "EEF Z/down": (np.array([0.0, 0.0, -1.0]), "#1f77b4"),
    }

    for label, (vec, color) in world_axes.items():
        ax.quiver(*origin, *(vec * 0.6), color=color, linewidth=1.8,
                  alpha=0.45, arrow_length_ratio=0.18)
        end = vec * 0.65
        ax.text(end[0], end[1], end[2], label, color=color, fontsize=8)

    for label, (vec, color) in eef_axes.items():
        ax.quiver(*origin, *(vec * 0.42), color=color, linewidth=2.8,
                  arrow_length_ratio=0.2)
        end = vec * 0.46
        ax.text(end[0], end[1], end[2], label, color=color, fontsize=8)

    ax.set_xlabel("World X / front")
    ax.set_ylabel("World Y / left")
    ax.set_zlabel("World Z / up")
    ax.set_xlim(-0.75, 0.75)
    ax.set_ylim(-0.75, 0.75)
    ax.set_zlim(-0.75, 0.75)
    ax.view_init(elev=24, azim=-135)


def plot_single_point_compare(
        pose_mats_eef, pose_mats_tcp, point_idx, output_path, title,
        orientation_frame="none"):
    pose_mats_tracker = pose_mats_eef # fallback for plot_single_point_compare to avoid breaking it
    tracker_pos = pose_mats_tracker[:, :3, 3]
    eef_pos = pose_mats_eef[:, :3, 3]
    tcp_pos = pose_mats_tcp[:, :3, 3]
    p_no = no_offset_pos[point_idx]
    p_with = with_offset_pos[point_idx]
    delta = p_with - p_no
    offset_norm = np.linalg.norm(delta)
    frame_len = max(offset_norm * 0.35, 0.045)

    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.45, 1.0])
    ax = fig.add_subplot(gs[0, 0], projection="3d")
    ax_top = fig.add_subplot(gs[0, 1])

    all_points = [p_no, p_with]
    if orientation_frame == "home":
        axes = DESIRED_EEF_AXES
    elif orientation_frame == "pose":
        axes = pose_mats_with_offset[point_idx, :3, :3]
    else:
        axes = None
    if axes is not None:
        all_points.extend([p_with + axes[:, i] * frame_len for i in range(3)])
    all_points = np.stack(all_points)

    ax.scatter(*p_no, color="#ff7f0e", s=60, label="without offset origin")
    ax.scatter(*p_with, color="#1f77b4", s=60, label="with offset origin")
    ax.plot(
        [p_no[0], p_with[0]],
        [p_no[1], p_with[1]],
        [p_no[2], p_with[2]],
        color="0.2", linewidth=2.0, label="offset vector")
    ax.quiver(
        p_no[0], p_no[1], p_no[2],
        delta[0], delta[1], delta[2],
        color="0.2", linewidth=2.0, arrow_length_ratio=0.12)
    draw_eef_orientation(
        ax, pose_mats_with_offset[point_idx], p_with, frame_len,
        orientation_frame, alpha=0.95, labels=True)

    ax.set_title(
        f"Single Matched Point End-Effector Position + Direction\n{title}")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    set_axes_equal(ax, all_points)
    ax.view_init(elev=23, azim=-138)
    ax.legend(loc="upper left", fontsize=8)

    ax_top.scatter(p_no[1], p_no[0], color="#ff7f0e", s=90, label="without offset")
    ax_top.scatter(p_with[1], p_with[0], color="#1f77b4", s=90, label="with offset")
    ax_top.annotate(
        "", xy=(p_with[1], p_with[0]), xytext=(p_no[1], p_no[0]),
        arrowprops=dict(arrowstyle="->", color="0.2", lw=2.0))
    ax_top.text(p_no[1], p_no[0], "  no", color="#ff7f0e", va="bottom")
    ax_top.text(p_with[1], p_with[0], "  with", color="#1f77b4", va="bottom")
    pad = max(abs(delta[1]), abs(delta[0]), 0.04) * 1.4
    center_y = (p_no[1] + p_with[1]) / 2
    center_x = (p_no[0] + p_with[0]) / 2
    ax_top.set_xlim(center_y + pad, center_y - pad)
    ax_top.set_ylim(center_x - pad, center_x + pad)
    ax_top.set_aspect("equal", adjustable="box")
    ax_top.grid(True, alpha=0.3)
    ax_top.set_title("Top-down position only\npage up = World X/front, page left = World Y/left")
    ax_top.set_xlabel("World Y / left (m)")
    ax_top.set_ylabel("World X / front (m)")
    ax_top.legend(loc="best")

    stats = (
        f"point_idx={point_idx}, "
        f"no_offset=({p_no[0]:.4f}, {p_no[1]:.4f}, {p_no[2]:.4f}) m, "
        f"with_offset=({p_with[0]:.4f}, {p_with[1]:.4f}, {p_with[2]:.4f}) m\n"
        f"delta with-no: X/front={delta[0]:+.4f}m, "
        f"Y/left={delta[1]:+.4f}m, Z/up={delta[2]:+.4f}m, "
        f"norm={offset_norm:.4f}m"
    )
    fig.suptitle(stats, fontsize=10)
    fig.tight_layout()
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return stats


def plot_offset_compare(pose_mats_no_offset, pose_mats_with_offset, output_path, title):
    tracker_pos = pose_mats_tracker[:, :3, 3]
    eef_pos = pose_mats_eef[:, :3, 3]
    tcp_pos = pose_mats_tcp[:, :3, 3]
    all_pos = np.concatenate([tracker_pos, eef_pos, tcp_pos], axis=0)

    offsets = with_offset_pos - no_offset_pos
    offset_norm = np.linalg.norm(offsets, axis=1)
    y_left_score = pose_mats_with_offset[:, :3, 1] @ np.array([0.0, 1.0, 0.0])
    z_down_score = pose_mats_with_offset[:, :3, 2] @ np.array([0.0, 0.0, -1.0])
    stats = (
        f"offset norm mean={offset_norm.mean():.4f}m, "
        f"min={offset_norm.min():.4f}m, max={offset_norm.max():.4f}m\n"
        f"EEF Y dot world-left mean={y_left_score.mean():.3f}; "
        f"EEF Z dot world-down mean={z_down_score.mean():.3f}"
    )

    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.65, 1.0], height_ratios=[1.0, 1.0])

    ax_xy = fig.add_subplot(gs[:, 0])
    # Top-down map: page up is world +X/front, page left is world +Y/left.
    ax_xy.plot(no_offset_pos[:, 1], no_offset_pos[:, 0],
               color="#ff7f0e", linewidth=2.0, label="without offset")
    ax_xy.plot(with_offset_pos[:, 1], with_offset_pos[:, 0],
               color="#1f77b4", linewidth=2.0, label="with offset")

    sample_count = min(28, len(with_offset_pos))
    sample_indices = np.linspace(0, len(with_offset_pos) - 1, sample_count, dtype=int)
    for idx in sample_indices:
        x0, y0 = no_offset_pos[idx, 1], no_offset_pos[idx, 0]
        x1, y1 = with_offset_pos[idx, 1], with_offset_pos[idx, 0]
        ax_xy.plot([x0, x1], [y0, y1], color="0.55", linewidth=0.9, alpha=0.65)
        ax_xy.annotate(
            "", xy=(x1, y1), xytext=(x0, y0),
            arrowprops=dict(arrowstyle="->", color="0.35", lw=0.9, alpha=0.75))

    ax_xy.scatter(no_offset_pos[0, 1], no_offset_pos[0, 0],
                  color="#ff7f0e", s=40, marker="o", label="start no offset")
    ax_xy.scatter(with_offset_pos[0, 1], with_offset_pos[0, 0],
                  color="#1f77b4", s=40, marker="o", label="start with offset")
    ax_xy.scatter(no_offset_pos[-1, 1], no_offset_pos[-1, 0],
                  color="#ff7f0e", s=60, marker="x", label="end no offset")
    ax_xy.scatter(with_offset_pos[-1, 1], with_offset_pos[-1, 0],
                  color="#1f77b4", s=60, marker="x", label="end with offset")
    ax_xy.set_title(f"Top-down Horizontal Offset Comparison\n{title}")
    ax_xy.set_xlabel("World Y / left (m)  (positive is left on page)")
    ax_xy.set_ylabel("World X / front (m)")
    ax_xy.invert_xaxis()
    ax_xy.set_aspect("equal", adjustable="box")
    ax_xy.grid(True, alpha=0.28)
    ax_xy.legend(loc="best", fontsize=8)

    ax_delta = fig.add_subplot(gs[0, 1])
    idxs = np.arange(len(offsets))
    ax_delta.plot(idxs, offsets[:, 0], label="Delta X/front", color="#d62728")
    ax_delta.plot(idxs, offsets[:, 1], label="Delta Y/left", color="#2ca02c")
    ax_delta.plot(idxs, offsets[:, 2], label="Delta Z/up", color="#1f77b4")
    ax_delta.plot(idxs, offset_norm, label="offset norm", color="0.1", linestyle="--")
    ax_delta.axhline(0.0, color="0.5", linewidth=0.8)
    ax_delta.set_title("with_offset - without_offset")
    ax_delta.set_xlabel("matched frame index")
    ax_delta.set_ylabel("meters")
    ax_delta.grid(True, alpha=0.28)
    ax_delta.legend(fontsize=8)

    ax_ref = fig.add_subplot(gs[1, 1], projection="3d")
    draw_convention_reference(ax_ref)

    fig.suptitle(stats, fontsize=11)
    fig.tight_layout()
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="handcap_data")
    parser.add_argument("--cap_id", default="42701")
    parser.add_argument("--tracker_root", default=None)
    parser.add_argument("--handcap_root", default=None)
    parser.add_argument("--episode_idx", type=int, default=0)
    parser.add_argument("--timestamp_threshold", type=float, default=0.1)
    parser.add_argument("--max_points", type=int, default=1200)
    parser.add_argument("--point_idx", type=int, default=None)
    parser.add_argument("--random_seed", type=int, default=7)
    parser.add_argument("--output", default="offset_compare_epoch0.png")
    parser.add_argument("--video_output", default=None)
    parser.add_argument("--video_fps", type=int, default=20)
    parser.add_argument("--max_video_frames", type=int, default=220)
    parser.add_argument(
        "--tcp_offset_mm",
        type=float,
        nargs=3,
        default=[0.0, 0.0, 150.0],
        metavar=("X_MM", "Y_MM", "Z_MM"),
        help="Local TCP offset in mm.")
    parser.add_argument(
        "--tcp_euler_xyz",
        type=float,
        nargs=3,
        default=[0.0, 0.0, 0.0],
        metavar=("RX_DEG", "RY_DEG", "RZ_DEG"),
        help="Local TCP rotation in degrees (XYZ euler).")
    parser.add_argument(
        "--offset_mm",
        type=float,
        nargs=3,
        default=DEFAULT_OFFSET_MM.tolist(),
        metavar=("X_MM", "Y_MM", "Z_MM"),
        help="Local EEF offset in mm. EEF convention here: X/back, Y/left, Z/down.")
    parser.add_argument(
        "--offset_frame",
        choices=("pose", "home"),
        default="pose",
        help="'pose' applies offset in each pose rotation; 'home' applies it in the displayed vertical-down home axes.")
    parser.add_argument(
        "--orientation_frame",
        choices=("none", "pose", "home"),
        default="none",
        help="Draw current end-effector orientation in the video/static plot.")
    args = parser.parse_args()
    offset_mm = np.asarray(args.offset_mm, dtype=np.float64)
    offset_m = offset_mm / 1000.0

    tracker_root, handcap_root = resolve_roots(
        args.data_root, args.cap_id, args.tracker_root, args.handcap_root)

    episode_folders = [
        p for p in handcap_root.iterdir()
        if p.is_dir()
    ]
    episode_folders.sort(key=lambda p: extract_num(p.name))
    if not episode_folders:
        raise FileNotFoundError(f"No episode folders found under {handcap_root}")
    epoch_root = episode_folders[args.episode_idx]

    htc_times, htc_poses = load_htc_pose_folder(tracker_root)
    wrist_times = load_wrist_times(epoch_root)
    if len(htc_times) == 0:
        raise RuntimeError(f"No htc pose data found in {tracker_root}")
    if len(wrist_times) == 0:
        raise RuntimeError(f"No wrist timestamps found in {epoch_root}")

    pose_indices = find_closest_indices(wrist_times, htc_times, args.timestamp_threshold)
    if len(pose_indices) == 0:
        raise RuntimeError("No htc poses matched wrist timestamps within threshold.")
    if len(pose_indices) > args.max_points:
        pose_indices = pose_indices[np.linspace(0, len(pose_indices) - 1, args.max_points, dtype=int)]

    pose_mats_tracker = np.stack([
        get_pose_transform_mats(
            htc_poses[idx], use_offset=False, offset_m=offset_m,
            offset_frame=args.offset_frame)
        for idx in pose_indices
    ])
    pose_mats_eef = np.stack([
        get_pose_transform_mats(
            htc_poses[idx], use_offset=True, offset_m=offset_m,
            offset_frame=args.offset_frame)
        for idx in pose_indices
    ])
    
    tcp_offset_m = np.asarray(args.tcp_offset_mm, dtype=np.float64) / 1000.0
    tcp_offset_transform = np.eye(4)
    tcp_offset_transform[:3, 3] = tcp_offset_m
    tcp_rot = Rotation.from_euler('xyz', args.tcp_euler_xyz, degrees=True)
    tcp_offset_transform[:3, :3] = tcp_rot.as_matrix()

    pose_mats_tcp = pose_mats_eef @ tcp_offset_transform

    if args.point_idx is None:
        rng = np.random.default_rng(args.random_seed)
        point_idx = int(rng.integers(0, len(pose_indices)))
    else:
        point_idx = int(np.clip(args.point_idx, 0, len(pose_indices) - 1))

    title = (
        f"{tracker_root.name} / {epoch_root.name} "
        f"({len(pose_indices)} matched wrist timestamps)\n"
        f"local EEF offset mm: X/back={offset_mm[0]:+.2f}, "
        f"Y/left={offset_mm[1]:+.2f}, Z/down={offset_mm[2]:+.2f}; "
        f"offset_frame={args.offset_frame}; orientation_frame={args.orientation_frame}"
    )
    # Skip single point compare update for brevity, just use eef and tcp
    stats = "Skipped single point"

    if args.video_output is not None:
        video_stats = make_three_view_videos(
            pose_mats_tracker,
            pose_mats_eef,
            pose_mats_tcp,
            args.video_output,
            title,
            fps=args.video_fps,
            max_video_frames=args.max_video_frames,
            orientation_frame=args.orientation_frame)
        for item in video_stats:
            print(item)


if __name__ == "__main__":
    main()
