import numpy as np
from pathlib import Path

# ---------- parameters ----------
N = 600

cx, cy = 0.0, 0.0
z0 = 0.0  # put slightly behind/below marker

semi_axis_x = 0.20  # ellipse semi-axis along x [m]
semi_axis_y = 0.10  # ellipse semi-axis along y [m]
band_width = 0.055  # corridor thickness [m]

rotation_deg = 0.0  # rotate shape if needed
script_dir = Path(__file__).resolve().parent
write_obj_file = script_dir / "data/ellipse_band.obj"  # TODO

# ---------- ellipse centerline ----------
theta = np.linspace(0, 2 * np.pi, N, endpoint=False)
rot = np.deg2rad(rotation_deg)

x_local = semi_axis_x * np.cos(theta)
y_local = semi_axis_y * np.sin(theta)

x = cx + np.cos(rot) * x_local - np.sin(rot) * y_local
y = cy + np.sin(rot) * x_local + np.cos(rot) * y_local
z = z0 * np.ones_like(theta)

center = np.column_stack((x, y, z))

closed_center = np.vstack([center, center[0]])
center_path_length = np.sum(np.linalg.norm(np.diff(closed_center, axis=0), axis=1))
print(f"Center path length: {center_path_length:.6f} m")

# ---------- compute normal for ribbon/band ----------
dx = np.gradient(x)
dy = np.gradient(y)

length = np.sqrt(dx**2 + dy**2)
length[length < 1e-9] = 1e-9

tx = dx / length
ty = dy / length

nx = -ty
ny = tx

half_band = band_width / 2.0

left = center.copy()
right = center.copy()

left[:, 0] += half_band * nx
left[:, 1] += half_band * ny

right[:, 0] -= half_band * nx
right[:, 1] -= half_band * ny

# ---------- write OBJ mesh ----------
write_obj_file.parent.mkdir(parents=True, exist_ok=True)

with open(write_obj_file, "w") as f:
    f.write("# Ellipse band mesh\n")

    for p in left:
        f.write(f"v {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")

    for p in right:
        f.write(f"v {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")

    for i in range(N):
        j = (i + 1) % N

        l1 = i + 1
        l2 = j + 1
        r1 = N + i + 1
        r2 = N + j + 1

        f.write(f"f {l1} {l2} {r2} {r1}\n")
