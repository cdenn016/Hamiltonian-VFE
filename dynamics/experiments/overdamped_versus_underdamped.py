#!/usr/bin/env python3
"""
Overdamped vs Underdamped Comparison
=====================================

Direct comparison between:
- Overdamped (first-order): θ̇ = -∇V (current approach)
- Underdamped (second-order): Full Hamiltonian dynamics

This demonstrates why the Fisher metric as mass matters!

Key differences:
- Overdamped: Exponential relaxation to minimum
- Underdamped: Oscillatory approach with momentum

Author: Chris
Date: November 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple

from dynamics.field_theory import (
    FullFieldHamiltonian,
    FullFieldState,
    create_simple_potential,
    create_initial_state
)
from dynamics.product_manifold_integrator import ProductManifoldVerlet
from geometry.lie_algebra import LieGroup


def overdamped_dynamics(
    hamiltonian: FullFieldHamiltonian,
    state: FullFieldState,
    dt: float,
    n_steps: int,
    friction: float = 1.0
) -> dict:
    """
    Simulate overdamped (first-order) dynamics.

    θ̇ = -γ^{-1} ∇V

    This is gradient flow with friction γ.
    """
    # Storage
    t_history = [state.t]
    mu_history = [state.mu.copy()]
    Sigma_history = [state.Sigma.copy()]
    phi_history = [state.phi.copy()]
    energy_history = [hamiltonian.potential_energy(state)]

    # Compute forces (negative gradient)
    eps = 1e-6

    for step in range(n_steps):
        # Gradient of potential
        grad_mu = np.zeros_like(state.mu)
        for i in range(len(state.mu)):
            mu_plus = state.mu.copy()
            mu_plus[i] += eps
            mu_minus = state.mu.copy()
            mu_minus[i] -= eps

            V_plus = hamiltonian.potential_func(mu_plus, state.Sigma, state.phi)
            V_minus = hamiltonian.potential_func(mu_minus, state.Sigma, state.phi)

            grad_mu[i] = (V_plus - V_minus) / (2 * eps)

        # Overdamped update: θ̇ = -γ^{-1} ∇V
        state.mu = state.mu - (dt / friction) * grad_mu
        state.t += dt

        # Store
        if step % 10 == 0:
            t_history.append(state.t)
            mu_history.append(state.mu.copy())
            Sigma_history.append(state.Sigma.copy())
            phi_history.append(state.phi.copy())
            energy_history.append(hamiltonian.potential_energy(state))

    return {
        't': np.array(t_history),
        'mu': np.array(mu_history),
        'Sigma': np.array(Sigma_history),
        'phi': np.array(phi_history),
        'energy': np.array(energy_history)
    }


def underdamped_dynamics(
    hamiltonian: FullFieldHamiltonian,
    state: FullFieldState,
    dt: float,
    n_steps: int
) -> dict:
    """
    Simulate underdamped (second-order) Hamiltonian dynamics.

    Full geometric integrator on product manifold.
    """
    integrator = ProductManifoldVerlet(hamiltonian)
    t_end = dt * n_steps

    history = integrator.integrate(
        state,
        t_end=t_end,
        dt=dt,
        save_interval=10
    )

    return history


def compare_regimes(K=13):
    """
    Compare overdamped vs underdamped for same initial conditions.

    Args:
        K: Dimension of latent space (default 2 for easy visualization)
    """
    print("\n" + "="*70)
    print("OVERDAMPED VS UNDERDAMPED COMPARISON")
    print("="*70)
    print(f"Latent dimension K = {K}")

    # Setup
    gauge_group = LieGroup.SO3

    # Simple quadratic potential
    mu_target = np.random.randn(K) * 0.5
    mu_target[0] = 1.0  # Fix first component for consistency
    if K > 1:
        mu_target[1] = -0.5
    Sigma_target = 2 * np.eye(K)

    potential = create_simple_potential(
        mu_target,
        Sigma_target,
        weights=(1.0, 1.0, 1.0)  # Only mean term for simplicity
    )

    # Hamiltonian
    Sigma_prior = 3 * np.eye(K)
    hamiltonian = FullFieldHamiltonian(
        potential=potential,
        Sigma_prior=Sigma_prior,
        gauge_group=gauge_group
    )

    # Initial state (displaced from minimum)
    state_overdamped = FullFieldState(
        mu=np.zeros(K),
        Sigma=Sigma_target.copy(),
        phi=np.zeros(3),
        pi_mu=np.zeros(K),
        pi_Sigma=np.zeros((K, K)),
        pi_phi=np.zeros(3),
        t=0.0
    )

    # Same initial state for underdamped (but with momentum)
    pi_mu_init = np.random.randn(K) * 0.05
    state_underdamped = FullFieldState(
        mu=np.zeros(K),
        Sigma=Sigma_target.copy(),
        phi=np.zeros(3),
        pi_mu=pi_mu_init,  # Initial momentum
        pi_Sigma=np.zeros((K, K)),
        pi_phi=np.zeros(3),
        t=0.0
    )

    # Simulate both
    dt = 0.01
    n_steps = 2000  # 20 time units

    print("\n1. Overdamped (gradient flow)...")
    history_over = overdamped_dynamics(
        hamiltonian,
        state_overdamped,
        dt=dt,
        n_steps=n_steps,
        friction=2.0
    )

    print("2. Underdamped (Hamiltonian)...")
    history_under = underdamped_dynamics(
        hamiltonian,
        state_underdamped,
        dt=dt,
        n_steps=n_steps
    )

    # Plot comparison
    plot_comparison(history_over, history_under, mu_target)

    return history_over, history_under


def plot_comparison(history_over: dict, history_under: dict, mu_target: np.ndarray):
    """Plot overdamped vs underdamped comparison."""
    output_dir = Path("_experiments/overdamped_vs_underdamped")
    output_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # Row 1: Time evolution
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.plot(history_over['t'], history_over['mu'][:, 0], 'r-',
             linewidth=2, label='Overdamped μ₀', alpha=0.7)
    ax1.plot(history_under['t'], history_under['mu'][:, 0], 'b-',
             linewidth=2, label='Underdamped μ₀', alpha=0.7)
    ax1.axhline(mu_target[0], color='gray', linestyle='--', alpha=0.5, label='Target')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('μ₀')
    ax1.set_title('Position Evolution: Overdamped vs Underdamped')
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Energy comparison
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.plot(history_over['t'], history_over['energy'], 'r-',
             linewidth=2, label='Overdamped', alpha=0.7)
    ax2.plot(history_under['t'], history_under['energy'], 'b-',
             linewidth=2, label='Underdamped', alpha=0.7)
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Potential Energy')
    ax2.set_title('Energy Dissipation')
    ax2.legend()
    ax2.grid(alpha=0.3)
    ax2.set_yscale('log')

    # Phase space (overdamped) - plot first two dimensions
    ax3 = fig.add_subplot(gs[1, 0])
    mu_over = history_over['mu']
    K = mu_over.shape[1]

    if K == 1:
        # For 1D, plot mu vs time
        ax3.plot(history_over['t'], mu_over[:, 0], 'r-', linewidth=2, alpha=0.7)
        ax3.set_xlabel('Time')
        ax3.set_ylabel('μ₀')
    else:
        # For K >= 2, plot first two dimensions
        ax3.plot(mu_over[:, 0], mu_over[:, 1], 'r-', linewidth=2, alpha=0.7)
        ax3.plot(mu_over[0, 0], mu_over[0, 1], 'go', markersize=10, label='Start')
        ax3.plot(mu_over[-1, 0], mu_over[-1, 1], 'ro', markersize=10, label='End')
        ax3.plot(mu_target[0], mu_target[1], 'k*', markersize=15, label='Target')
        ax3.set_xlabel('μ₀')
        ax3.set_ylabel('μ₁')
        ax3.axis('equal')

    ax3.set_title(f'Overdamped: Phase Space (K={K})')
    ax3.legend()
    ax3.grid(alpha=0.3)

    # Phase space (underdamped)
    ax4 = fig.add_subplot(gs[1, 1])
    mu_under = history_under['mu']

    if K == 1:
        # For 1D, plot mu vs time
        ax4.plot(history_under['t'], mu_under[:, 0], 'b-', linewidth=2, alpha=0.7)
        ax4.set_xlabel('Time')
        ax4.set_ylabel('μ₀')
    else:
        # For K >= 2, plot first two dimensions
        ax4.plot(mu_under[:, 0], mu_under[:, 1], 'b-', linewidth=2, alpha=0.7)
        ax4.plot(mu_under[0, 0], mu_under[0, 1], 'go', markersize=10, label='Start')
        ax4.plot(mu_under[-1, 0], mu_under[-1, 1], 'ro', markersize=10, label='End')
        ax4.plot(mu_target[0], mu_target[1], 'k*', markersize=15, label='Target')
        ax4.set_xlabel('μ₀')
        ax4.set_ylabel('μ₁')
        ax4.axis('equal')

    ax4.set_title(f'Underdamped: Phase Space (K={K})')
    ax4.legend()
    ax4.grid(alpha=0.3)

    # Velocity comparison
    ax5 = fig.add_subplot(gs[1, 2])
    # Approximate velocity from position differences
    v_over = np.diff(mu_over[:, 0]) / np.diff(history_over['t'])
    v_under = np.diff(mu_under[:, 0]) / np.diff(history_under['t'])
    ax5.plot(history_over['t'][1:], v_over, 'r-', linewidth=2, label='Overdamped', alpha=0.7)
    ax5.plot(history_under['t'][1:], v_under, 'b-', linewidth=2, label='Underdamped', alpha=0.7)
    ax5.set_xlabel('Time')
    ax5.set_ylabel('μ̇₀')
    ax5.set_title('Velocity Evolution')
    ax5.legend()
    ax5.grid(alpha=0.3)

    # Distance to target
    ax6 = fig.add_subplot(gs[2, :])
    dist_over = np.linalg.norm(mu_over - mu_target, axis=1)
    dist_under = np.linalg.norm(mu_under - mu_target, axis=1)
    ax6.plot(history_over['t'], dist_over, 'r-', linewidth=2, label='Overdamped', alpha=0.7)
    ax6.plot(history_under['t'], dist_under, 'b-', linewidth=2, label='Underdamped', alpha=0.7)
    ax6.set_xlabel('Time')
    ax6.set_ylabel('||μ - μ*||')
    ax6.set_title('Distance to Target')
    ax6.legend()
    ax6.grid(alpha=0.3)
    ax6.set_yscale('log')

    plt.suptitle('Overdamped (Gradient Flow) vs Underdamped (Hamiltonian)', fontsize=16, y=0.995)

    plt.savefig(output_dir / "comparison.png", dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\n✓ Saved comparison: {output_dir / 'comparison.png'}")

    # Print summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"\nOverdamped:")
    print(f"  Final distance: {dist_over[-1]:.6f}")
    print(f"  Approach: Exponential decay (no oscillation)")
    print(f"\nUnderdamped:")
    print(f"  Final distance: {dist_under[-1]:.6f}")
    print(f"  Approach: Oscillatory (conserves energy)")
    print("\nKey difference: Momentum!")
    print("="*70 + "\n")


def main(K=3):
    """
    Run overdamped vs underdamped comparison.

    Args:
        K: Latent dimension (default 2)
    """
    compare_regimes(K=K)


if __name__ == "__main__":
    import sys
    # Allow K to be passed as command line argument
    K = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    main(K=K)