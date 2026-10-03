"""What the TRN camera sees at apogee: sky-view from the vehicle (azimuthal equidistant, centre = nadir).
Moon fills a disc of half-angle asin(R/(R+h)) around nadir. Camera FOV is a cone around the boresight.
Assumptions: apogee altitude 48.5 km; camera full FOV 30 deg (EDIT); boresight = body -z (exhaust side).
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R, H = 1737.4, 48.5                 # km
FOV = 30.0                          # deg, full angle (ASSUMPTION)
f = np.radians(FOV / 2)
horizon = np.degrees(np.arcsin(R / (R + H)))            # 76.6 deg from nadir
slant_horizon = np.sqrt((R + H) ** 2 - R ** 2)          # km, distance to the horizon

cases = [
    ("Gimballed, held at nadir", 0.0, "Ground fills the image"),
    ("Fixed, burn attitude", 46.7, "Ground in view, oblique (1.46× range)"),
    ("Fixed, horizontal attitude\n(e.g. apogee burn along velocity)", 90.0, None),
]

blue, orange, aqua = "#2a78d6", "#eb6834", "#1baf7a"
ink, muted, grid, surface = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"


def cone_boundary(theta_deg, half, n=400):
    th = np.radians(theta_deg)
    b = np.array([np.sin(th), 0.0, np.cos(th)])          # boresight, z = toward nadir
    u = np.cross(b, [0, 1, 0]); u /= np.linalg.norm(u)
    v = np.cross(b, u)
    ph = np.linspace(0, 2 * np.pi, n)
    pts = (np.cos(half) * b[None, :] + np.sin(half) * (np.cos(ph)[:, None] * u + np.sin(ph)[:, None] * v))
    ang = np.degrees(np.arccos(np.clip(pts[:, 2], -1, 1)))
    az = np.arctan2(pts[:, 1], pts[:, 0])
    return ang * np.cos(az), ang * np.sin(az)


def ground_fraction(theta_deg, half, n=200000, seed=0):
    rng = np.random.default_rng(seed)
    th = np.radians(theta_deg)
    b = np.array([np.sin(th), 0.0, np.cos(th)])
    u = np.cross(b, [0, 1, 0]); u /= np.linalg.norm(u)
    v = np.cross(b, u)
    cosr = 1 - rng.random(n) * (1 - np.cos(half))        # uniform over the cap
    ph = rng.random(n) * 2 * np.pi
    s = np.sqrt(1 - cosr**2)
    pts = cosr[:, None] * b + s[:, None] * (np.cos(ph)[:, None] * u + np.sin(ph)[:, None] * v)
    ang = np.degrees(np.arccos(np.clip(pts[:, 2], -1, 1)))
    return np.mean(ang < horizon)


fig, axs = plt.subplots(1, 3, figsize=(11.5, 4.6), facecolor=surface)
tt = np.linspace(0, 2 * np.pi, 400)
for ax, (title, th, note) in zip(axs, cases):
    ax.set_facecolor(surface)
    ax.set_aspect("equal")
    ax.set_xlim(-130, 130); ax.set_ylim(-175, 130)
    ax.add_patch(plt.Circle((0, 0), 125, color=grid, lw=0))                       # sky / space
    ax.add_patch(plt.Circle((0, 0), horizon, color=aqua, alpha=0.30, lw=0))       # Moon
    ax.plot(horizon * np.cos(tt), horizon * np.sin(tt), color=aqua, lw=1.5)
    x, y = cone_boundary(th, f)
    ax.fill(x, y, color=blue, alpha=0.55, lw=0)
    ax.plot(x, y, color=blue, lw=2)
    ax.plot(0, 0, "+", color=ink, ms=8)
    ax.text(0, -horizon - 8, f"Moon horizon, {horizon:.1f}° from nadir", ha="center", va="top",
            color=ink, fontsize=8)
    ax.text(0, 118, "space", ha="center", color=muted, fontsize=8)
    frac = ground_fraction(th, f)
    if note is None:
        note = f"{frac*100:.0f}% of the image is ground,\nonly at the limb, ~{slant_horizon:.0f} km away"
    ax.set_title(title, color=ink, fontsize=10, fontweight="bold")
    ax.text(0, -138, note, ha="center", va="top", color=ink, fontsize=9)
    ax.axis("off")

fig.suptitle(f"Camera view at apogee ({H} km): a fixed camera's ground view depends on attitude "
             f"(FOV {FOV:.0f}° assumed)", color=ink, fontsize=11, fontweight="bold", x=0.01, ha="left")
fig.tight_layout(rect=(0, 0.03, 1, 0.95))
fig.savefig("/home/claude/trn_fov_apogee.png", dpi=200, facecolor=surface)

print(f"horizon from nadir: {horizon:.1f} deg; slant range to horizon: {slant_horizon:.0f} km")
for title, th, _ in cases:
    print(f"{title.splitlines()[0]:30s} boresight {th:5.1f} deg  ground fraction {ground_fraction(th, f)*100:5.1f}%")
print(f"boresight angle where FOV loses ground entirely: {horizon + FOV/2:.1f} deg from nadir")
