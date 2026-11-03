"""Visualize vertical force and center of pressure from fp_raw.csv.

The animated view is the default; it keeps only a short COP trail visible.

Run from this directory or any other directory:
    python visualize_force_plates.py                              # play animation
    python visualize_force_plates.py --no_animate                 # static PNG
    python visualize_force_plates.py --output out.gif             # save GIF
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

PLATE_LENGTH = 0.6  # m, x direction
PLATE_WIDTH = 0.4  # m, y direction


def load_force_plate_data(csv_path: Path) -> np.ndarray:
    """Load the CSV while tolerating the whitespace in its header."""
    return np.genfromtxt(
        csv_path,
        delimiter=",",
        names=True,
        dtype=float,
        autostrip=True,
    )


def _compute_cop_global(
    data: np.ndarray, plate_index: int, gap: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert local COP coordinates to global x/y."""
    fz = np.abs(data[f"FP{plate_index}_Fz"])
    # COP is measured from the center of each plate, while plate origins are
    # its left edges in the global drawing.
    global_x = (
        data[f"FP{plate_index}_Px"] + (plate_index - 1) * (PLATE_LENGTH + gap) + PLATE_LENGTH / 2
    )
    global_y = data[f"FP{plate_index}_Py"]
    return global_x, global_y, fz


def _draw_plate_outlines(ax: plt.Axes, gap: float) -> None:
    """Draw the three force plate rectangles and labels."""
    colors = ["tab:blue", "tab:orange", "tab:green"]
    for plate_index, color in enumerate(colors, start=1):
        origin_x = (plate_index - 1) * (PLATE_LENGTH + gap)
        ax.add_patch(
            Rectangle(
                (origin_x, -PLATE_WIDTH / 2),
                PLATE_LENGTH,
                PLATE_WIDTH,
                fill=False,
                edgecolor=color,
                linewidth=2,
            )
        )
        ax.text(
            origin_x + PLATE_LENGTH / 2,
            -PLATE_WIDTH / 2 - 0.05,
            f"Plate {plate_index}",
            color=color,
            ha="center",
        )


def make_figure(data: np.ndarray, contact_threshold: float, gap: float) -> plt.Figure:
    """Create a static overview figure (all data at once)."""
    time = data["SysTime"] - data["SysTime"][0]
    figure, (force_ax, plate_ax) = plt.subplots(2, 1, figsize=(12, 8))

    colors = ["tab:blue", "tab:orange", "tab:green"]
    for plate_index, color in enumerate(colors, start=1):
        fz = np.abs(data[f"FP{plate_index}_Fz"])
        force_ax.plot(time, fz, color=color, label=f"Plate {plate_index}")

        # Px/Py are local COP coordinates. Only draw them during loaded contact.
        contact = fz >= contact_threshold
        cop_x = (
            data[f"FP{plate_index}_Px"][contact]
            + (plate_index - 1) * (PLATE_LENGTH + gap)
            + PLATE_LENGTH / 2
        )
        cop_y = data[f"FP{plate_index}_Py"][contact]
        plate_ax.scatter(cop_x, cop_y, c=time[contact], s=5, cmap="viridis")

    _draw_plate_outlines(plate_ax, gap)

    force_ax.axhline(contact_threshold, color="black", linestyle="--", linewidth=1)
    force_ax.set(title="Vertical force |Fz|", xlabel="Time (s)", ylabel="|Fz| (N)")
    force_ax.legend(ncol=3)
    force_ax.grid(alpha=0.3)

    plate_ax.set(
        title="Center of pressure while loaded (color = time)",
        xlabel="Global x (m)",
        ylabel="Local y (m)",
        aspect="equal",
    )
    plate_ax.set_xlim(-0.1, 3 * PLATE_LENGTH + 2 * gap + 0.1)
    plate_ax.set_ylim(-PLATE_WIDTH / 2 - 0.1, PLATE_WIDTH / 2 + 0.1)
    plate_ax.grid(alpha=0.3)
    figure.tight_layout()
    return figure


def animate_data(
    data: np.ndarray,
    contact_threshold: float,
    gap: float,
    fps: float = 50,
    trail_seconds: float = 1.0,
) -> animation.FuncAnimation:
    """Create an animated visualization that reveals data over time.

    Parameters
    ----------
    data : np.ndarray
        Structured array from the CSV.
    contact_threshold : float
        Minimum |Fz| to consider a plate loaded.
    gap : float
        Gap between neighboring plates.
    fps : float
        Display frame rate. Samples are downsampled to this rate.
    trail_seconds : float
        How many seconds of COP history to keep visible.
    """
    time = data["SysTime"] - data["SysTime"][0]
    # The file is sampled at 512 Hz. Drawing every sample is unnecessary and
    # can make the wall-clock playback much slower than the recorded motion.
    frame_times = np.arange(0.0, time[-1] + 1e-9, 1.0 / fps)
    frame_indices = np.searchsorted(time, frame_times, side="left")
    frame_indices = np.clip(frame_indices, 0, len(data) - 1)

    figure, (force_ax, plate_ax) = plt.subplots(2, 1, figsize=(12, 8))

    # ---------- force subplot ----------
    colors = ["tab:blue", "tab:orange", "tab:green"]
    force_lines = []
    for plate_index, color in enumerate(colors, start=1):
        (line,) = force_ax.plot([], [], color=color, label=f"Plate {plate_index}")
        force_lines.append(line)

    force_ax.axhline(contact_threshold, color="black", linestyle="--", linewidth=1)
    force_ax.set(
        title="Vertical force |Fz| — animation",
        xlabel="Time (s)",
        ylabel="|Fz| (N)",
        xlim=(0, time[-1] * 1.05),
        ylim=(-50, 1500),
    )
    force_ax.legend(ncol=3)
    force_ax.grid(alpha=0.3)

    # ---------- plate / COP subplot ----------
    _draw_plate_outlines(plate_ax, gap)
    cop_scatters = []
    for color in colors:
        (scat,) = plate_ax.plot([], [], "o", color=color, markersize=4, alpha=0.7)
        cop_scatters.append(scat)

    plate_ax.set(
        title="Center of pressure over time",
        xlabel="Global x (m)",
        ylabel="Local y (m)",
        aspect="equal",
        xlim=(-0.1, 3 * PLATE_LENGTH + 2 * gap + 0.1),
        ylim=(-PLATE_WIDTH / 2 - 0.1, PLATE_WIDTH / 2 + 0.1),
    )
    plate_ax.grid(alpha=0.3)

    # ---------- time indicator line on force plot ----------
    (time_indicator,) = force_ax.plot([], [], "k--", linewidth=1, alpha=0.6)

    # ---------- timestamp text ----------
    timestamp_text = plate_ax.text(
        0.98,
        0.98,
        "",
        transform=plate_ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7),
    )

    figure.tight_layout()

    # Pre-compute global COP for all plates
    cop_data = [_compute_cop_global(data, i, gap) for i in (1, 2, 3)]

    def _update(frame: int) -> list:
        sample_index = frame_indices[frame]
        t = time[: sample_index + 1]
        forces = [np.abs(data[f"FP{i}_Fz"][: sample_index + 1]) for i in (1, 2, 3)]

        for line, f in zip(force_lines, forces):
            line.set_data(t, f)

        # COP — show only a short recent trail, so repeated walks remain clear.
        for i, (scat, (gx, gy, fz)) in enumerate(zip(cop_scatters, cop_data)):
            first_index = np.searchsorted(time, time[sample_index] - trail_seconds, side="left")
            gx_curr = gx[first_index : sample_index + 1]
            gy_curr = gy[first_index : sample_index + 1]
            contact = fz[first_index : sample_index + 1] >= contact_threshold
            scat.set_data(gx_curr[contact], gy_curr[contact])

        # Time indicator line
        current_time = time[frame]
        time_indicator.set_data([current_time, current_time], force_ax.get_ylim())

        # Timestamp
        timestamp_text.set_text(f"t = {current_time:.2f} s")

        return [*force_lines, *cop_scatters, time_indicator, timestamp_text]

    anim = animation.FuncAnimation(
        figure,
        _update,
        frames=len(frame_indices),
        interval=1000.0 / fps,
        blit=True,
    )
    return anim


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "csv_path",
        nargs="?",
        type=Path,
        default=Path(__file__).with_name("fp_raw.csv"),
        help="Path to fp_raw.csv",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=50.0,
        help="Contact threshold in N (default: 50)",
    )
    parser.add_argument(
        "--gap",
        type=float,
        default=0.0,
        help="Gap between neighboring plates in meters (default: 0)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("force_plate_visualization.png"),
        help="Output path (PNG for static, GIF/MP4 for animation)",
    )
    parser.add_argument("--no-show", action="store_true", help="Do not open a plot window")
    parser.add_argument(
        "--no_animate",
        action="store_true",
        help="Do not create an animated visualization instead of a static overview",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=50.0,
        help="Animation display rate; data is downsampled to this rate (default: 50)",
    )
    parser.add_argument(
        "--trail-seconds",
        type=float,
        default=3.0,
        help="Seconds of recent COP trail to display (default: 1)",
    )
    args = parser.parse_args()

    data = load_force_plate_data(args.csv_path)
    n_pts = len(data)

    if not args.no_animate:
        anim = animate_data(
            data, args.threshold, args.gap, fps=args.fps, trail_seconds=args.trail_seconds
        )

        # Save if output has a movie/gif extension
        if args.output.suffix in (".gif", ".mp4", ".mov", ".avi"):
            writer = "pillow" if args.output.suffix == ".gif" else "ffmpeg"
            anim.save(args.output, writer=writer, dpi=150)
            print(f"Saved animation to {args.output}")

        if not args.no_show:
            plt.show()
    else:
        figure = make_figure(data, args.threshold, args.gap)
        figure.savefig(args.output, dpi=150)
        print(f"Saved visualization to {args.output}")
        if not args.no_show:
            plt.show()


if __name__ == "__main__":
    main()
