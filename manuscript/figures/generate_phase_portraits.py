"""
Generate combined phase portrait figure for manuscript.
Two panels:
  (a) Damped decay to equilibrium (overdamped)
  (b) Orbital dynamics (underdamped/Hamiltonian)

Both evolve over 10,000 steps from t=0 to t=250.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Simulation parameters
n_steps = 10000
t_max = 250
dt = t_max / n_steps
t = np.linspace(0, t_max, n_steps)

# Initial conditions (same for both)
mu_0 = 2.0      # Initial belief
pi_0 = 1.5      # Initial momentum

# System parameters
K = 1.0         # Spring constant (evidence strength)
M = 1.0         # Mass (precision)

# ============================================================================
# Case 1: Damped decay (overdamped: gamma > 2*sqrt(K*M))
# ============================================================================
gamma_damped = 3.0  # Strong damping

mu_damped = np.zeros(n_steps)
pi_damped = np.zeros(n_steps)
mu_damped[0] = mu_0
pi_damped[0] = pi_0

for i in range(1, n_steps):
    # Hamiltonian: H = pi^2/(2M) + K*mu^2/2
    # Equations of motion with damping:
    #   dmu/dt = pi/M
    #   dpi/dt = -K*mu - gamma*pi/M

    mu = mu_damped[i-1]
    pi = pi_damped[i-1]

    # Symplectic-like integration with damping
    dmu = pi / M
    dpi = -K * mu - gamma_damped * pi / M

    mu_damped[i] = mu + dmu * dt
    pi_damped[i] = pi + dpi * dt

# ============================================================================
# Case 2: Orbit (underdamped/conservative: gamma = 0 or very small)
# ============================================================================
gamma_orbit = 0.0  # No damping - pure Hamiltonian

mu_orbit = np.zeros(n_steps)
pi_orbit = np.zeros(n_steps)
mu_orbit[0] = mu_0
pi_orbit[0] = pi_0

for i in range(1, n_steps):
    # Symplectic Euler for energy conservation
    mu = mu_orbit[i-1]
    pi = pi_orbit[i-1]

    # Update momentum first (symplectic)
    pi_new = pi - K * mu * dt
    # Update position with new momentum
    mu_new = mu + pi_new / M * dt

    mu_orbit[i] = mu_new
    pi_orbit[i] = pi_new

# ============================================================================
# Create combined figure
# ============================================================================
fig = plt.figure(figsize=(12, 5))
gs = GridSpec(1, 2, figure=fig, wspace=0.3)

# Panel (a): Damped
ax1 = fig.add_subplot(gs[0, 0])
scatter1 = ax1.scatter(mu_damped, pi_damped, c=t, cmap='viridis', s=1, alpha=0.7)
ax1.plot(mu_damped[0], pi_damped[0], 'o', color='lime', markersize=10,
         markeredgecolor='black', markeredgewidth=1.5, label='Start', zorder=5)
ax1.plot(mu_damped[-1], pi_damped[-1], 's', color='red', markersize=10,
         markeredgecolor='black', markeredgewidth=1.5, label='End', zorder=5)
ax1.axhline(0, color='gray', linestyle='--', alpha=0.5, linewidth=0.5)
ax1.axvline(0, color='gray', linestyle='--', alpha=0.5, linewidth=0.5)
ax1.set_xlabel(r'Belief $\mu$', fontsize=12)
ax1.set_ylabel(r'Momentum $\pi_\mu$', fontsize=12)
ax1.set_title(r'(a) Overdamped: $\gamma > 2\sqrt{KM}$', fontsize=12)
ax1.legend(loc='upper right', fontsize=10)
ax1.set_aspect('equal', adjustable='box')
ax1.grid(alpha=0.3)

# Panel (b): Orbit
ax2 = fig.add_subplot(gs[0, 1])
scatter2 = ax2.scatter(mu_orbit, pi_orbit, c=t, cmap='viridis', s=1, alpha=0.7)
ax2.plot(mu_orbit[0], pi_orbit[0], 'o', color='lime', markersize=10,
         markeredgecolor='black', markeredgewidth=1.5, label='Start', zorder=5)
ax2.axhline(0, color='gray', linestyle='--', alpha=0.5, linewidth=0.5)
ax2.axvline(0, color='gray', linestyle='--', alpha=0.5, linewidth=0.5)
ax2.set_xlabel(r'Belief $\mu$', fontsize=12)
ax2.set_ylabel(r'Momentum $\pi_\mu$', fontsize=12)
ax2.set_title(r'(b) Underdamped: $\gamma = 0$ (Hamiltonian)', fontsize=12)
ax2.legend(loc='upper right', fontsize=10)
ax2.set_aspect('equal', adjustable='box')
ax2.grid(alpha=0.3)

# Add shared colorbar
cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
cbar = fig.colorbar(scatter2, cax=cbar_ax)
cbar.set_label('Time $t$', fontsize=11)

plt.tight_layout(rect=[0, 0, 0.9, 1])

# Save figure
plt.savefig('/home/user/Hamiltonian-VFE/manuscript/figures/phase_portraits_combined.png',
            dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig('/home/user/Hamiltonian-VFE/manuscript/figures/phase_portraits_combined.pdf',
            bbox_inches='tight', facecolor='white')

print("Saved: phase_portraits_combined.png and .pdf")
print(f"  Damped: gamma={gamma_damped}, final position=({mu_damped[-1]:.4f}, {pi_damped[-1]:.4f})")
print(f"  Orbit:  gamma={gamma_orbit}, energy conserved (closed orbit)")

plt.show()
