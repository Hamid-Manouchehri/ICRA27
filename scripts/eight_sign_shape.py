import numpy as np
from pathlib import Path

# ---------- parameters ----------
N = 400
cx, cy, z0 = 0.0, 0.0, 0.

eight_width_total  = 0.40   # size of the eight shape [m]
eight_height_total = 0.15   # size of the eight shape [m]

band_width = 0.055           # line/band thickness [m]

script_dir = Path(__file__).resolve().parent

write_obj_file = script_dir / "data/eight_sign_band.obj"  # TODO

# ---------- centerline ----------
t = np.linspace(0, 2*np.pi, N, endpoint=False)

W = eight_width_total / 2
H = eight_height_total / 2

x = cx + W * np.sin(t)
y = cy + H * np.sin(2*t)
z = z0 * np.ones_like(t)

center = np.column_stack((x, y, z))

closed_center = np.vstack([center, center[0]])
center_path_length = np.sum(np.linalg.norm(np.diff(closed_center, axis=0), axis=1))
print(f"Center path length: {center_path_length:.6f} m")

# ---------- tangent and normal ----------
dx = np.gradient(x)
dy = np.gradient(y)

length = np.sqrt(dx**2 + dy**2)
tx = dx / length
ty = dy / length

# normal in xy plane
nx = -ty
ny = tx

half_band = band_width / 2

left = center.copy()
right = center.copy()

left[:, 0]  += half_band * nx
left[:, 1]  += half_band * ny

right[:, 0] -= half_band * nx
right[:, 1] -= half_band * ny

write_obj_file.parent.mkdir(parents=True, exist_ok=True)

# ---------- write OBJ mesh ----------
with open(write_obj_file, "w") as f:
    f.write("# Figure-eight band mewrite_obj_filesh\n")

    # vertices: left side then right side
    for p in left:
        f.write(f"v {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
    for p in right:
        f.write(f"v {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")

    # faces
    for i in range(N):
        j = (i + 1) % N

        l1 = i + 1
        l2 = j + 1
        r1 = N + i + 1
        r2 = N + j + 1

        f.write(f"f {l1} {l2} {r2} {r1}\n")