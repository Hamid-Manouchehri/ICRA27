import numpy as np
from pathlib import Path

# ---------- parameters ----------
N = 800

cx, cy = 0.0, 0.0
z0 = 0.0  # put slightly behind/below marker

half_size_x = 0.16  # half width of squircle [m]
half_size_y = 0.16  # half height of squircle [m]
corner_roundness_power = 4.0  # 2=circle, larger=more square
band_width = 0.055  # corridor thickness [m]

rotation_deg = 0.0  # rotate shape if needed
script_dir = Path(__file__).resolve().parent
write_obj_file = script_dir / "data/squircle_band.obj"  # TODO

# ---------- squircle centerline ----------
# Superellipse:
# |x/a|^n + |y/b|^n = 1
# The parametric form below generates a square with curved corners.
theta = np.linspace(0.0, 2.0 * np.pi, N, endpoint=False)
rot = np.deg2rad(rotation_deg)

cos_t = np.cos(theta)
sin_t = np.sin(theta)

x_local = half_size_x * np.sign(cos_t) * np.abs(cos_t) ** (
    2.0 / corner_roundness_power)
y_local = half_size_y * np.sign(sin_t) * np.abs(sin_t) ** (
    2.0 / corner_roundness_power)

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
    f.write("# Squircle band mesh\n")

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
