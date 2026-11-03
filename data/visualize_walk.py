"""Synchronize Xsens skeleton data with force-plate data for one walk trial.

The two files do not share timestamps. Each file advances using its own
sampling clock. Use --force-start-offset when the force plates started earlier.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import openpyxl
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


PLATE_LENGTH = 0.6
PLATE_WIDTH = 0.4
EDGES = [
    ("Pelvis", "L5"), ("L5", "L3"), ("L3", "T12"), ("T12", "T8"),
    ("T8", "Neck"), ("Neck", "Head"),
    ("T8", "Right Shoulder"), ("Right Shoulder", "Right Upper Arm"),
    ("Right Upper Arm", "Right Forearm"), ("Right Forearm", "Right Hand"),
    ("T8", "Left Shoulder"), ("Left Shoulder", "Left Upper Arm"),
    ("Left Upper Arm", "Left Forearm"), ("Left Forearm", "Left Hand"),
    ("Pelvis", "Right Upper Leg"), ("Right Upper Leg", "Right Lower Leg"),
    ("Right Lower Leg", "Right Foot"), ("Right Foot", "Right Toe"),
    ("Pelvis", "Left Upper Leg"), ("Left Upper Leg", "Left Lower Leg"),
    ("Left Lower Leg", "Left Foot"), ("Left Foot", "Left Toe"),
]
COLORS = ["tab:blue", "tab:orange", "tab:green"]


def load_xsens(path: Path) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]]]:
    workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
    position_sheet = workbook["Segment Position"]
    header = list(next(position_sheet.iter_rows(
        min_row=1, max_row=1, values_only=True)))
    segment_names = header[1::3]
    position_rows = np.asarray(
        list(position_sheet.iter_rows(min_row=2, values_only=True)), dtype=float)
    positions = position_rows[:, 1:].reshape(
        len(position_rows), len(segment_names), 3)

    com_sheet = workbook["Center of Mass"]
    com_rows = np.asarray(list(com_sheet.iter_rows(
        min_row=2, values_only=True)), dtype=float)
    frame_count = min(len(positions), len(com_rows))
    positions = positions[:frame_count]
    com_positions = com_rows[:frame_count, 1:4]

    indices = {name: index for index, name in enumerate(segment_names)}
    edges = [(indices[a], indices[b])
             for a, b in EDGES if a in indices and b in indices]
    return positions, com_positions, edges


def load_force_plates(path: Path) -> np.ndarray:
    return np.genfromtxt(path, delimiter=",", names=True, dtype=float, autostrip=True)


def plate_cop(data: np.ndarray, plate: int, gap: float, center_y: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert COP coordinates, measured from each plate center, to world coordinates."""
    x = data[f"FP{plate}_Px"] + (plate - 1) * \
        (PLATE_LENGTH + gap) + PLATE_LENGTH / 2
    y = data[f"FP{plate}_Py"] + center_y
    force = np.abs(data[f"FP{plate}_Fz"])
    return x, y, force


def add_plates(ax: plt.Axes, gap: float, center_y: float) -> None:
    for plate, color in enumerate(COLORS, start=1):
        x0 = (plate - 1) * (PLATE_LENGTH + gap)
        y0 = center_y - PLATE_WIDTH / 2
        corners = [(x0, y0, 0), (x0 + PLATE_LENGTH, y0, 0),
                   (x0 + PLATE_LENGTH, y0 + PLATE_WIDTH, 0), (x0, y0 + PLATE_WIDTH, 0)]
        ax.add_collection3d(Poly3DCollection(
            [corners], facecolors=color, alpha=0.13, edgecolors=color))
        ax.text(x0 + PLATE_LENGTH / 2, y0 - 0.04, 0,
                f"Plate {plate}", color=color, ha="center")


def main() -> None:
    directory = Path(__file__).parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xlsx", type=Path,
                        default=directory / "xsense_raw.xlsx")
    parser.add_argument("--csv", type=Path, default=directory / "fp_raw.csv")
    parser.add_argument("--threshold", type=float, default=50.0)
    parser.add_argument("--fps", type=float, default=50.0,
                        help="Animation display FPS")
    parser.add_argument("--trail-seconds", type=float, default=0.6)
    parser.add_argument("--gap", type=float, default=0.0)
    parser.add_argument("--plate-center-y", type=float, default=0.35,
                        help="Plate centerline in Xsens y coordinates")
    parser.add_argument("--xsens-fps", type=float, default=60.0,
                        help="Xsens sampling rate in Hz (default: 60)")
    parser.add_argument(
        "--force-start-offset",
        type=float,
        default=0.5,
        help="Seconds force plates started before Xsens (default: 0)",
    )
    args = parser.parse_args()

    positions, com_positions, edges = load_xsens(args.xlsx)
    force_data = load_force_plates(args.csv)
    force_time = force_data["SysTime"] - force_data["SysTime"][0]
    # At Xsens t=0, read the force plate at this later timestamp. For example,
    # an offset of 0.5 means the force plates began recording 0.5 s earlier.
    aligned_force_time = force_time + args.force_start_offset
    xsens_time = np.arange(len(positions), dtype=float) / args.xsens_fps
    # Play only the interval present in both recordings; do not stretch either
    # signal to force the two final frames to coincide.
    duration = min(aligned_force_time[-1], xsens_time[-1])
    print(
        f"Xsens duration: {xsens_time[-1]:.3f} s; "
        f"force-plate duration: {force_time[-1]:.3f} s; "
        f"force offset: {args.force_start_offset:.3f} s; "
        f"playing overlap: {duration:.3f} s"
    )
    display_times = np.arange(0.0, duration + 1e-9, 1.0 / args.fps)
    force_indices = np.searchsorted(
        aligned_force_time, display_times, side="left").clip(0, len(force_data) - 1)
    xsens_indices = np.searchsorted(xsens_time, display_times, side="left")
    xsens_indices = xsens_indices.clip(0, len(positions) - 1)

    figure = plt.figure(figsize=(13, 10))
    grid = figure.add_gridspec(2, 1, height_ratios=(3, 1))
    skeleton_ax = figure.add_subplot(grid[0], projection="3d")
    force_ax = figure.add_subplot(grid[1])
    add_plates(skeleton_ax, args.gap, args.plate_center_y)

    xyz_min = positions.reshape(-1, 3).min(axis=0)
    xyz_max = positions.reshape(-1, 3).max(axis=0)
    xyz_min[:2] = np.minimum(
        xyz_min[:2], [0.0, args.plate_center_y - PLATE_WIDTH / 2])
    xyz_max[:2] = np.maximum(xyz_max[:2], [
                             3 * PLATE_LENGTH + 2 * args.gap, args.plate_center_y + PLATE_WIDTH / 2])
    center = (xyz_min + xyz_max) / 2
    radius = max((xyz_max - xyz_min).max() / 2, 0.5)
    skeleton_ax.set(xlim=(center[0] - radius, center[0] + radius), ylim=(center[1] - radius, center[1] +
                    radius), zlim=(0, center[2] + radius), xlabel="X (m)", ylabel="Y (m)", zlabel="Z (m)")
    skeleton_ax.set_box_aspect((1, 1, 0.7))
    skeleton_ax.view_init(elev=25, azim=-65)

    (joint_points,) = skeleton_ax.plot(
        [], [], [], "o", color="tab:blue", markersize=3)
    (com_point,) = skeleton_ax.plot([], [], [], "o",
                                    color="tab:red", markersize=6, label="CoM")
    bone_lines = [skeleton_ax.plot([], [], [], color="black", linewidth=1.5)[
        0] for _ in edges]
    cop_trails = [skeleton_ax.plot(
        [], [], [], "o", color=color, markersize=2, alpha=0.55)[0] for color in COLORS]
    cop_now = [skeleton_ax.plot([], [], [], "o", color=color, markersize=7)[
        0] for color in COLORS]

    force_lines = []
    for plate, color in enumerate(COLORS, start=1):
        (line,) = force_ax.plot(aligned_force_time, np.abs(
            force_data[f"FP{plate}_Fz"]), color=color, label=f"Plate {plate}")
        force_lines.append(line)
    force_ax.axhline(args.threshold, color="black",
                     linestyle="--", linewidth=1)
    (time_line,) = force_ax.plot([], [], "k--", linewidth=1)
    force_ax.set(xlim=(0, duration), ylim=(-50, 1500),
                 xlabel="Time (s)", ylabel="|Fz| (N)", title="Vertical force")
    force_ax.grid(alpha=0.3)
    force_ax.legend(ncol=3)
    time_text = skeleton_ax.text2D(
        0.02, 0.95, "", transform=skeleton_ax.transAxes)
    cop_data = [plate_cop(force_data, plate, args.gap,
                          args.plate_center_y) for plate in (1, 2, 3)]

    def update(frame: int) -> list:
        skeleton = positions[xsens_indices[frame]]
        joint_points.set_data(skeleton[:, 0], skeleton[:, 1])
        joint_points.set_3d_properties(skeleton[:, 2])
        com = com_positions[xsens_indices[frame]]
        com_point.set_data([com[0]], [com[1]])
        com_point.set_3d_properties([com[2]])
        for line, (start, end) in zip(bone_lines, edges):
            line.set_data(skeleton[[start, end], 0], skeleton[[start, end], 1])
            line.set_3d_properties(skeleton[[start, end], 2])
        force_index = force_indices[frame]
        trail_start = np.searchsorted(
            aligned_force_time,
            aligned_force_time[force_index] - args.trail_seconds,
            side="left",
        )
        for trail, current, (cop_x, cop_y, cop_force) in zip(cop_trails, cop_now, cop_data):
            mask = cop_force[trail_start:force_index + 1] >= args.threshold
            x, y = cop_x[trail_start:force_index +
                         1][mask], cop_y[trail_start:force_index + 1][mask]
            trail.set_data(x, y)
            trail.set_3d_properties(np.zeros_like(x) + 0.01)
            if cop_force[force_index] >= args.threshold:
                current.set_data([cop_x[force_index]], [cop_y[force_index]])
                current.set_3d_properties([0.02])
            else:
                current.set_data([], [])
                current.set_3d_properties([])
        current_time = aligned_force_time[force_index]
        time_line.set_data([current_time, current_time], [-50, 1500])
        time_text.set_text(f"t = {current_time:.2f} s")
        return [joint_points, com_point, *bone_lines, *cop_trails, *cop_now, time_line, time_text]

    anim = animation.FuncAnimation(
        figure, update, frames=len(display_times), interval=1000 / args.fps, blit=False
    )
    figure.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
