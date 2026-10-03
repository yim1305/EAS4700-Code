"""
IMU lever-arm offset vs. position drift.

Question this answers: if the IMU sits an axial distance z from the CG
(on the roll axis), how much position error accumulates during a maneuver
if we (a) do nothing, or (b) correct it in software but only know z to within dz?

Model (IMU on roll axis, r = [0, 0, z]):
  centripetal (along axis) : a_c = z * (w_pitch^2 + w_yaw^2)
  tangential  (sideways)   : a_t = z * alpha
  total magnitude (worst)  : a   = z * sqrt(w^4 + alpha^2)
  position error over t    : dx   = 0.5 * a * t^2   (constant-acceleration worst case)

Correction: the INS subtracts  alpha x r + w x (w x r)  using the best-known r.
What is left is the same expression evaluated at the lever-arm KNOWLEDGE error dz.

EDIT THE PARAMETERS BELOW, then run:  python3 imu_lever_arm_plot.py
"""
import numpy as np
import matplotlib.pyplot as plt

# ---------------- parameters (all assumed -- replace with your own) ----------
W_DEG_S = 6.0        # pitch/yaw body rate during the maneuver [deg/s]
ALPHA_DEG_S2 = 1.0   # pitch/yaw angular acceleration [deg/s^2]
T_MANEUVER = 20.0    # duration the rate/acceleration is sustained [s]
BUDGET_M = 1.0       # position-error budget allocated to this term [m]
DZ_KNOWLEDGE = [0.02, 0.05]   # lever-arm knowledge error after calibration [m]
Z_MAX = 1.5          # largest axial offset to plot [m]
# -----------------------------------------------------------------------------

w = np.radians(W_DEG_S)
alpha = np.radians(ALPHA_DEG_S2)
coef = np.sqrt(w**4 + alpha**2)          # acceleration per metre of offset [1/s^2]


def drift(offset_m):
    """Position error [m] from a lever-arm acceleration error of offset_m."""
    return 0.5 * coef * offset_m * T_MANEUVER**2


z = np.linspace(0.001, Z_MAX, 400)

# colors: reference categorical slots 1-3, neutral gray for the budget line
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"

fig, ax = plt.subplots(figsize=(9, 5.4), dpi=160)
fig.patch.set_facecolor(SURFACE)
ax.set_facecolor(SURFACE)

# series
ax.plot(z, drift(z), color=BLUE, lw=2, label="Uncorrected")
ax.plot(z, drift(np.full_like(z, DZ_KNOWLEDGE[1])), color=AQUA, lw=2,
        label=f"Corrected, offset known to {DZ_KNOWLEDGE[1]*100:.0f} cm")
ax.plot(z, drift(np.full_like(z, DZ_KNOWLEDGE[0])), color=ORANGE, lw=2,
        label=f"Corrected, offset known to {DZ_KNOWLEDGE[0]*100:.0f} cm")
ax.axhline(BUDGET_M, color=MUTED, lw=1.2, ls="--")

# the corrected error does not depend on z, only on knowledge error -> flat lines
# direct labels
ax.text(Z_MAX, drift(Z_MAX) * 0.93, "Uncorrected", color=INK, ha="right",
        va="top", fontsize=9.5)
ax.text(Z_MAX, drift(DZ_KNOWLEDGE[1]) * 1.25,
        f"Corrected, {DZ_KNOWLEDGE[1]*100:.0f} cm knowledge error",
        color=INK, ha="right", va="bottom", fontsize=9.5)
ax.text(Z_MAX, drift(DZ_KNOWLEDGE[0]) * 0.75,
        f"Corrected, {DZ_KNOWLEDGE[0]*100:.0f} cm knowledge error",
        color=INK, ha="right", va="top", fontsize=9.5)
ax.text(0.02, BUDGET_M * 1.12, f"Budget: {BUDGET_M:g} m", color=MUTED,
        ha="left", va="bottom", fontsize=9.5)

# where does the uncorrected curve cross the budget?
z_cross = 2 * BUDGET_M / (coef * T_MANEUVER**2)
if z_cross < Z_MAX:
    ax.plot([z_cross], [BUDGET_M], "o", color=BLUE, ms=8, mec=SURFACE, mew=2)
    ax.annotate(f"Uncorrected exceeds budget\nbeyond z = {z_cross*100:.0f} cm",
                xy=(z_cross, BUDGET_M), xytext=(z_cross + 0.12, BUDGET_M * 0.28),
                color=INK, fontsize=9.5,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=1))

ax.set_yscale("log")
ax.set_xlim(0, Z_MAX)
ax.set_ylim(0.01, 20)
ax.set_xlabel("Axial offset of IMU from CG, z  [m]", color=MUTED, fontsize=10)
ax.set_ylabel("Position error from lever arm  [m]", color=MUTED, fontsize=10)
ax.set_title("IMU lever-arm offset vs. position drift",
             loc="left", color=INK, fontsize=13, fontweight="bold", pad=22)
ax.text(0, 1.03,
        f"IMU on roll axis. Assumed: {W_DEG_S:g} deg/s, {ALPHA_DEG_S2:g} deg/s^2 "
        f"sustained for {T_MANEUVER:g} s (worst case).",
        transform=ax.transAxes, color=MUTED, fontsize=9, va="bottom")

ax.grid(True, which="major", color=GRID, lw=0.8)
ax.grid(False, which="minor")
ax.tick_params(colors=MUTED, labelsize=9)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax.spines[s].set_color(GRID)

ax.legend(loc="upper left", frameon=False, fontsize=9, labelcolor=INK)
fig.tight_layout()
fig.savefig("imu_lever_arm_drift.png", facecolor=SURFACE)

print(f"coef = {coef:.4e} (m/s^2 per m of offset)")
print(f"uncorrected drift at z=0.3 m : {drift(0.3):.2f} m")
print(f"uncorrected drift at z=1.0 m : {drift(1.0):.2f} m")
print(f"uncorrected crosses {BUDGET_M:g} m budget at z = {z_cross*100:.1f} cm")
for dz in DZ_KNOWLEDGE:
    print(f"corrected, knowledge error {dz*100:.0f} cm : {drift(dz):.3f} m")
