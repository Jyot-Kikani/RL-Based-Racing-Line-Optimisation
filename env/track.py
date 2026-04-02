# ── env/track.py ──────────────────────────────────────────────────────────────
# Loads a TUMFTM-format CSV and exposes track geometry.
#
# TUMFTM CSV columns: x_m, y_m, w_tr_right_m, w_tr_left_m
# Download real tracks from:
#   https://github.com/TUMFTM/racetrack-database/tree/master/tracks

import numpy as np
import csv
from scipy.interpolate import splprep, splev


class Track:
    def __init__(self, csv_path: str, smooth: bool = True, n_points: int = 300):
        self.csv_path = csv_path
        self.n_points = n_points
        self._load(csv_path)
        if smooth:
            self._smooth()
        self._compute_bounds()
        self._compute_distances()

    # ── Step 2a: load CSV ─────────────────────────────────────────────────────
    def _load(self, path: str):
        xs, ys, wr, wl = [], [], [], []
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                xs.append(float(row["x_m"]))
                ys.append(float(row["y_m"]))
                wr.append(float(row["w_tr_right_m"]))
                wl.append(float(row["w_tr_left_m"]))

        # Drop duplicate closing point if present (last == first)
        if len(xs) > 1 and abs(xs[-1] - xs[0]) < 1e-6 and abs(ys[-1] - ys[0]) < 1e-6:
            xs, ys, wr, wl = xs[:-1], ys[:-1], wr[:-1], wl[:-1]

        self._raw_xy      = np.column_stack([xs, ys])
        self._raw_w_right = np.array(wr)
        self._raw_w_left  = np.array(wl)

    # ── Step 2b: smooth + resample centerline with scipy spline ──────────────
    def _smooth(self):
        xy = self._raw_xy
        # Close the loop for a periodic spline
        x = np.append(xy[:, 0], xy[0, 0])
        y = np.append(xy[:, 1], xy[0, 1])

        tck, _ = splprep([x, y], s=10.0, per=True, k=3)
        u_new  = np.linspace(0, 1, self.n_points, endpoint=False)
        sx, sy = splev(u_new, tck)
        self.centerline = np.column_stack([sx, sy])

        # Resample widths to match new number of points
        old_u = np.linspace(0, 1, len(self._raw_w_right))
        new_u = np.linspace(0, 1, self.n_points)
        self._w_right = np.interp(new_u, old_u, self._raw_w_right)
        self._w_left  = np.interp(new_u, old_u, self._raw_w_left)

    # ── Step 2c: offset centerline along normal vectors → boundaries ──────────
    def _compute_bounds(self):
        cl = self.centerline
        n  = len(cl)

        # Tangent at each point (forward difference, wrapped)
        tangents = np.roll(cl, -1, axis=0) - cl          # (N, 2)
        norms    = np.linalg.norm(tangents, axis=1, keepdims=True)
        tangents = tangents / np.where(norms > 1e-9, norms, 1e-9)

        # Normal = 90° rotation of tangent  (left-hand normal)
        normals = np.column_stack([-tangents[:, 1], tangents[:, 0]])

        self.right_bound = cl + normals * self._w_right[:, None]
        self.left_bound  = cl - normals * self._w_left[:, None]

    # ── Step 2d: cumulative arc-length for progress tracking ─────────────────
    def _compute_distances(self):
        cl   = self.centerline
        diffs = np.diff(cl, axis=0, append=cl[:1])       # wrap around
        segs  = np.linalg.norm(diffs, axis=1)
        self.distances     = np.cumsum(segs)
        self.track_length  = self.distances[-1]

    # ── Step 2e: nearest waypoint index ───────────────────────────────────────
    def nearest_waypoint(self, x: float, y: float) -> int:
        diffs = self.centerline - np.array([x, y])
        return int(np.argmin(np.sum(diffs ** 2, axis=1)))

    def progress(self, x: float, y: float) -> float:
        """Cumulative arc-length to nearest waypoint."""
        return float(self.distances[self.nearest_waypoint(x, y)])

    # ── Step 2f: on-track check via signed distance to centerline ─────────────
    def is_on_track(self, x: float, y: float) -> bool:
        """
        True if the point is within track width of the nearest centerline point.
        Uses a simple distance check — good enough for dense waypoint arrays.
        """
        idx   = self.nearest_waypoint(x, y)
        pt    = np.array([x, y])
        dist  = float(np.linalg.norm(pt - self.centerline[idx]))
        # Allow up to the narrower of the two widths at this point
        limit = min(self._w_right[idx], self._w_left[idx])
        return dist <= limit

    # ── Convenience ───────────────────────────────────────────────────────────
    @property
    def start_pos(self):
        """(x, y) of first waypoint."""
        return self.centerline[0]

    @property
    def start_heading(self):
        """Heading angle (rad) at start, pointing toward waypoint 1."""
        p0 = self.centerline[0]
        p1 = self.centerline[1]
        return float(np.arctan2(p1[1] - p0[1], p1[0] - p0[0]))