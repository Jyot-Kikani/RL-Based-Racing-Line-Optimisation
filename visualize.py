# ── visualize.py ──────────────────────────────────────────────────────────────
# Plot the learned racing line as a speed heatmap over the track.
# Run: python visualize.py --trajectory outputs/trajectories/trajectory_ep1.csv
#
# Produces:
#   outputs/plots/racing_line_heatmap.png
#   outputs/plots/speed_profile.png

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.collections import LineCollection
from env.track import Track
from config import TRACK_FILE, PLOT_DIR


def plot_racing_line(traj_path: str):
    df    = pd.read_csv(traj_path)
    track = Track(TRACK_FILE)

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_aspect("equal")
    ax.set_facecolor("#1a1a2e")
    fig.patch.set_facecolor("#1a1a2e")

    # Draw track boundaries
    ax.plot(track.left_bound[:, 0],  track.left_bound[:, 1],
            color="#444466", linewidth=1.5, label="Boundaries")
    ax.plot(track.right_bound[:, 0], track.right_bound[:, 1],
            color="#444466", linewidth=1.5)

    # Draw centerline
    ax.plot(track.centerline[:, 0], track.centerline[:, 1],
            color="#666688", linewidth=1, linestyle="--", alpha=0.5,
            label="Centerline")

    # Draw RL racing line as speed heatmap
    points  = np.array([df["x"].values, df["y"].values]).T.reshape(-1, 1, 2)
    segs    = np.concatenate([points[:-1], points[1:]], axis=1)
    speeds  = df["speed_kmh"].values
    norm    = plt.Normalize(speeds.min(), speeds.max())
    lc      = LineCollection(segs, cmap="RdYlGn", norm=norm, linewidth=2.5)
    lc.set_array(speeds[:-1])
    ax.add_collection(lc)

    cbar = fig.colorbar(lc, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Speed (km/h)", color="white")
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    ax.set_title("RL Agent — Learned Racing Line", color="white", fontsize=14)
    ax.tick_params(colors="white")
    ax.legend(facecolor="#222244", edgecolor="none",
              labelcolor="white", fontsize=9)

    os.makedirs(PLOT_DIR, exist_ok=True)
    out = os.path.join(PLOT_DIR, "racing_line_heatmap.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Saved → {out}")
    plt.show()


def plot_speed_profile(traj_path: str):
    df  = pd.read_csv(traj_path)
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(df["frame"], df["speed"], color="#4CAF50", linewidth=1.2)
    ax.set_xlabel("Frame")
    ax.set_ylabel("Speed (m/s)")
    ax.set_title("Speed Profile — RL Agent")
    ax.grid(True, alpha=0.3)
    out = os.path.join(PLOT_DIR, "speed_profile.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved → {out}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory",
                        default="outputs/trajectories/trajectory_ep1.csv")
    args = parser.parse_args()
    plot_racing_line(args.trajectory)
    plot_speed_profile(args.trajectory)