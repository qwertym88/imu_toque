#!/usr/bin/env python3
"""Preview the recorded Xsens skeleton and its exported center of mass."""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import openpyxl
from matplotlib.animation import FuncAnimation

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data/xsense_raw.xlsx")
    parser.add_argument("--fps", type=float, default=60.0, help="Playback frame rate")
    parser.add_argument("--frame", type=int, help="Save a single frame instead of animating")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/xsens_preview.png")
    parser.add_argument(
        "--no-show", action="store_true", help="Use with --frame for headless export"
    )
    args = parser.parse_args()
    if args.fps <= 0:
        parser.error("--fps must be positive")
    if args.no_show and args.frame is None:
        parser.error("--no-show requires --frame")
    xlsx_path = args.input

    # 读取 Xsens 段位置
    wb = openpyxl.load_workbook(xlsx_path, read_only=True, data_only=True)
    ws = wb["Segment Position"]

    header = list(next(ws.iter_rows(min_row=1, max_row=1, values_only=True)))
    segment_names = header[1::3]

    rows = []
    for row in ws.iter_rows(min_row=2, values_only=True):
        rows.append(row)

    data = np.asarray(rows, dtype=float)
    positions = data[:, 1:].reshape(len(data), len(segment_names), 3)

    # 读取质心位置
    com_ws = wb["Center of Mass"]
    com_rows = np.asarray(
        list(com_ws.iter_rows(min_row=2, values_only=True)),
        dtype=float,
    )
    com_positions = com_rows[:, 1:4]

    wb.close()

    # 保证质心和骨架帧数一致
    num_frames = min(len(positions), len(com_positions))
    positions = positions[:num_frames]
    com_positions = com_positions[:num_frames]

    name_to_idx = {name: i for i, name in enumerate(segment_names)}

    # Xsens 段之间的连接关系
    edge_names = [
        ("Pelvis", "L5"),
        ("L5", "L3"),
        ("L3", "T12"),
        ("T12", "T8"),
        ("T8", "Neck"),
        ("Neck", "Head"),
        ("T8", "Right Shoulder"),
        ("Right Shoulder", "Right Upper Arm"),
        ("Right Upper Arm", "Right Forearm"),
        ("Right Forearm", "Right Hand"),
        ("T8", "Left Shoulder"),
        ("Left Shoulder", "Left Upper Arm"),
        ("Left Upper Arm", "Left Forearm"),
        ("Left Forearm", "Left Hand"),
        ("Pelvis", "Right Upper Leg"),
        ("Right Upper Leg", "Right Lower Leg"),
        ("Right Lower Leg", "Right Foot"),
        ("Right Foot", "Right Toe"),
        ("Pelvis", "Left Upper Leg"),
        ("Left Upper Leg", "Left Lower Leg"),
        ("Left Lower Leg", "Left Foot"),
        ("Left Foot", "Left Toe"),
    ]

    edges = [
        (name_to_idx[a], name_to_idx[b])
        for a, b in edge_names
        if a in name_to_idx and b in name_to_idx
    ]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # 固定坐标范围，避免动画过程中画面跳动
    xyz_min = positions.reshape(-1, 3).min(axis=0)
    xyz_max = positions.reshape(-1, 3).max(axis=0)
    center = (xyz_min + xyz_max) / 2
    radius = max((xyz_max - xyz_min).max() / 2, 0.5)

    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    (points,) = ax.plot([], [], [], "o", color="tab:blue", markersize=4)
    (com_point,) = ax.plot([], [], [], "o", color="red", markersize=8)
    lines = [ax.plot([], [], [], color="black", linewidth=2)[0] for _ in edges]

    def update(frame):
        p = positions[frame]

        points.set_data(p[:, 0], p[:, 1])
        points.set_3d_properties(p[:, 2])

        com = com_positions[frame]
        com_point.set_data([com[0]], [com[1]])
        com_point.set_3d_properties([com[2]])

        for line, (i, j) in zip(lines, edges):
            line.set_data([p[i, 0], p[j, 0]], [p[i, 1], p[j, 1]])
            line.set_3d_properties([p[i, 2], p[j, 2]])

        ax.set_title(f"Xsens frame {frame}/{len(positions) - 1}")
        return [points, com_point, *lines]

    if args.frame is not None:
        if not 0 <= args.frame < len(positions):
            parser.error(f"--frame must be between 0 and {len(positions) - 1}")
        update(args.frame)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=150)
        print(f"Saved Xsens frame {args.frame} to {args.output}")
        if not args.no_show:
            plt.show()
        plt.close(fig)
        return

    animation = FuncAnimation(
        fig,
        update,
        frames=len(positions),
        interval=1000 / args.fps,
        blit=False,
    )

    plt.show()


if __name__ == "__main__":
    main()
