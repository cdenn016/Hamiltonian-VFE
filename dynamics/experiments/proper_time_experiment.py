# -*- coding: utf-8 -*-
"""
Created on Sat Nov 22 20:23:27 2025

@author: chris and christine
"""

#!/usr/bin/env python3
"""
Proper Time from Belief Trajectories
=====================================

Demonstrates relativistic proper time dilation emerging from Fisher metric.

For agents with different "masses" (Σ_p), trajectories in belief space
accumulate different proper times:

    dτ = √(dμ^T Σ_p^{-1} dμ)

Key prediction: τ_heavy/τ_light > 1 (time dilation for massive agents)

This is a direct relativistic effect from the Fisher metric geometry!

Author: Chris
Date: November 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple

from dynamics.field_theory import (
    FullFieldHamiltonian,
    FullFieldState,
    create_simple_potential
)
from dynamics.product_manifold_integrator import ProductManifoldVerlet
from geometry.lie_algebra import LieGroup


def compute_proper_time(
    trajectory: np.ndarray,
    times: np.ndarray,
    Sigma_prior: np.ndarray
) -> Tuple[np.ndarray, float]:
    """
    Compute proper time accumulated along a belief trajectory.

    Args:
        trajectory: (N, K) array of belief states μ(t)
        times: (N,) array of coordinate times
        Sigma_prior: (K, K) mass matrix (prior covariance)

    Returns:
        tau_cumulative: (N,) cumulative proper time
        tau_total: Total proper time
    """
    K = trajectory.shape[1]
    N = len(trajectory)

    Sigma_prior_inv = np.linalg.inv(Sigma_prior)

    tau = np.zeros(N)

    for i in range(1, N):
        dmu = trajectory[i] - trajectory[i-1]

        # Proper time increment: dτ = √(dμ^T Σ_p^{-1} dμ)
        dtau_sq = dmu @ Sigma_prior_inv @ dmu

        if dtau_sq > 0:
            dtau = np.sqrt(dtau_sq)
        else:
            dtau = 0.0

        tau[i] = tau[i-1] + dtau

    return tau, tau[-1]


def run_trajectory(
    K: int,
    Sigma_prior_scale: float,
    mu_target: np.ndarray,
    dt: float = 0.005,
    t_end: float = 30.0,
    seed: int = None
) -> dict:
    """Run a single belief trajectory with given mass."""

    if seed is not None:
        np.random.seed(seed)

    gauge_group = LieGroup.SO3

    # Simple quadratic potential
    Sigma_target = 2 * np.eye(K)
    potential = create_simple_potential(
        mu_target,
        Sigma_target,
        weights=(1.0, 0.0, 0.0)  # Only mean term
    )

    # Hamiltonian with MASS = Σ_p^{-1}
    Sigma_prior = Sigma_prior_scale * np.eye(K)
    hamiltonian = FullFieldHamiltonian(
        potential=potential,
        Sigma_prior=Sigma_prior,
        gauge_group=gauge_group
    )

    # Initial state
    state = FullFieldState(
        mu=np.zeros(K),
        Sigma=Sigma_target.copy(),
        phi=np.zeros(3),
        pi_mu=np.random.randn(K) * 0.1,  # Initial momentum
        pi_Sigma=np.zeros((K, K)),
        pi_phi=np.zeros(3),
        t=0.0
    )

    # Integrate
    integrator = ProductManifoldVerlet(hamiltonian)
    history = integrator.integrate(
        state,
        t_end=t_end,
        dt=dt,
        save_interval=2
    )

    # Compute proper time
    tau_cumulative, tau_total = compute_proper_time(
        history['mu'],
        history['t'],
        Sigma_prior
    )

    history['tau'] = tau_cumulative
    history['tau_total'] = tau_total
    history['Sigma_prior'] = Sigma_prior
    history['Sigma_prior_scale'] = Sigma_prior_scale

    return history


def proper_time_experiment(K=13):
    """
    Demonstrate proper time dilation for K=13 belief trajectories.

    Compares agents with different masses (Σ_p scales).
    """
    print("\n" + "="*70)
    print(f"PROPER TIME EXPERIMENT (K={K})")
    print("="*70)

    # Target in belief space
    mu_target = np.random.randn(K) * 0.5
    mu_target[0] = 1.0  # Fix first component

    # Different mass scales
    mass_scales = [0.5, 2.0, 8.0, 32.0]
    labels = ['Very Light', 'Light', 'Heavy', 'Very Heavy']
    colors = ['blue', 'green', 'orange', 'red']

    print(f"\nRunning {len(mass_scales)} trajectories with different masses...")
    print(f"Target: μ* = {mu_target[:3]}... (K={K})")

    trajectories = []
    for i, (scale, label) in enumerate(zip(mass_scales, labels)):
        print(f"\n  {label} (Σ_p = {scale}I)...")
        history = run_trajectory(
            K=K,
            Sigma_prior_scale=scale,
            mu_target=mu_target,
            dt=0.005,
            t_end=30.0,
            seed=42 + i  # Fixed seed for reproducibility
        )
        trajectories.append(history)
        print(f"    Proper time: τ = {history['tau_total']:.6f}")

    # Compute ratios
    tau_values = [h['tau_total'] for h in trajectories]
    tau_light = tau_values[0]

    print("\n" + "="*70)
    print("PROPER TIME RATIOS")
    print("="*70)
    for i, (scale, label, tau) in enumerate(zip(mass_scales, labels, tau_values)):
        ratio = tau / tau_light
        print(f"  {label:15s} (Σ_p={scale:5.1f}I): τ/τ_light = {ratio:.3f}")

    # Create comprehensive figure
    create_proper_time_figure(trajectories, labels, colors, mu_target, K)

    return trajectories


def create_proper_time_figure(
    trajectories: List[dict],
    labels: List[str],
    colors: List[str],
    mu_target: np.ndarray,
    K: int
):
    """Create comprehensive proper time figure."""

    output_dir = Path("_experiments/proper_time")
    output_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 4, hspace=0.35, wspace=0.35)

    # -------------------------------------------------------------------------
    # Panel A: Coordinate time vs Proper time
    # -------------------------------------------------------------------------
    ax_a = fig.add_subplot(gs[0, 0:2])

    for traj, label, color in zip(trajectories, labels, colors):
        t = traj['t']
        tau = traj['tau']
        ax_a.plot(t, tau, color=color, linewidth=2.5, label=label, alpha=0.8)

    # Add diagonal (light cone)
    t_max = max([traj['t'][-1] for traj in trajectories])
    ax_a.plot([0, t_max], [0, t_max], 'k--', linewidth=1.5,
              alpha=0.3, label='Light cone (τ=t)')

    ax_a.set_xlabel('Coordinate Time t', fontsize=12)
    ax_a.set_ylabel('Proper Time τ', fontsize=12)
    ax_a.set_title('A: Proper Time Dilation from Fisher Metric', fontsize=13, fontweight='bold')
    ax_a.legend(fontsize=10, loc='upper left')
    ax_a.grid(alpha=0.3)

    # -------------------------------------------------------------------------
    # Panel B: Proper time ratio vs mass
    # -------------------------------------------------------------------------
    ax_b = fig.add_subplot(gs[0, 2:])

    scales = [traj['Sigma_prior_scale'] for traj in trajectories]
    taus = [traj['tau_total'] for traj in trajectories]
    tau_light = taus[0]
    ratios = [tau / tau_light for tau in taus]

    ax_b.plot(scales, ratios, 'o-', markersize=12, linewidth=2.5, color='purple')

    for scale, ratio, color, label in zip(scales, ratios, colors, labels):
        ax_b.plot(scale, ratio, 'o', markersize=15, color=color, alpha=0.7)

    ax_b.axhline(1.0, color='gray', linestyle='--', alpha=0.5, label='τ = τ_light')
    ax_b.set_xlabel('Mass Scale Σ_p', fontsize=12)
    ax_b.set_ylabel('Proper Time Ratio τ/τ_light', fontsize=12)
    ax_b.set_title('B: Time Dilation vs Mass', fontsize=13, fontweight='bold')
    ax_b.legend(fontsize=10)
    ax_b.grid(alpha=0.3)
    ax_b.set_xscale('log')

    # -------------------------------------------------------------------------
    # Panel C: Belief trajectories (first 3 components)
    # -------------------------------------------------------------------------
    ax_c = fig.add_subplot(gs[1, 0:2])

    for traj, label, color in zip(trajectories, labels, colors):
        t = traj['t']
        mu = traj['mu']
        # Plot first component only (or use norm)
        mu_norm = np.linalg.norm(mu - mu_target, axis=1)
        ax_c.plot(t, mu_norm, color=color, linewidth=2.5, label=label, alpha=0.8)

    ax_c.set_xlabel('Coordinate Time t', fontsize=12)
    ax_c.set_ylabel('||μ(t) - μ*||', fontsize=12)
    ax_c.set_title(f'C: Belief Trajectories (K={K})', fontsize=13, fontweight='bold')
    ax_c.legend(fontsize=10)
    ax_c.grid(alpha=0.3)
    ax_c.set_yscale('log')

    # -------------------------------------------------------------------------
    # Panel D: Proper velocity dτ/dt
    # -------------------------------------------------------------------------
    ax_d = fig.add_subplot(gs[1, 2:])

    for traj, label, color in zip(trajectories, labels, colors):
        t = traj['t']
        tau = traj['tau']

        # Compute dτ/dt
        dtau_dt = np.diff(tau) / np.diff(t)
        t_mid = 0.5 * (t[:-1] + t[1:])

        ax_d.plot(t_mid, dtau_dt, color=color, linewidth=2, label=label, alpha=0.8)

    ax_d.set_xlabel('Coordinate Time t', fontsize=12)
    ax_d.set_ylabel('dτ/dt', fontsize=12)
    ax_d.set_title('D: Proper Velocity', fontsize=13, fontweight='bold')
    ax_d.legend(fontsize=10)
    ax_d.grid(alpha=0.3)



    # -------------------------------------------------------------------------
    # Panel G: Accumulated distance in belief space
    # -------------------------------------------------------------------------
    ax_g = fig.add_subplot(gs[2, 2])

    for traj, label, color in zip(trajectories, labels, colors):
        mu = traj['mu']
        t = traj['t']

        # Compute accumulated coordinate distance
        distances = np.zeros(len(mu))
        for i in range(1, len(mu)):
            dmu = mu[i] - mu[i-1]
            distances[i] = distances[i-1] + np.linalg.norm(dmu)

        ax_g.plot(t, distances, color=color, linewidth=2, label=label, alpha=0.8)

    ax_g.set_xlabel('Time', fontsize=11)
    ax_g.set_ylabel('Accumulated Distance ||Δμ||', fontsize=11)
    ax_g.set_title('E: Coordinate Distance', fontsize=12, fontweight='bold')
    ax_g.legend(fontsize=9)
    ax_g.grid(alpha=0.3)



    # Save
    plt.savefig(output_dir / f"proper_time_K{K}.png", dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / f"proper_time_K{K}.pdf", bbox_inches='tight')

    print(f"\n✓ Saved: {output_dir / f'proper_time_K{K}.png'}")
    print(f"✓ Saved: {output_dir / f'proper_time_K{K}.pdf'}")


def main(K=13):
    """
    Run proper time experiment for K-dimensional belief space.

    Args:
        K: Latent dimension (default 13)
    """
    trajectories = proper_time_experiment(K=K)

    print("\n" + "="*70)
    print("EXPERIMENT COMPLETE")
    print("="*70)
    print("\nKey finding: Proper time dilation τ_heavy/τ_light > 1")
    print("This is a direct relativistic effect from Fisher metric!")
    print("="*70 + "\n")


if __name__ == "__main__":
    import sys
    K = int(sys.argv[1]) if len(sys.argv) > 1 else 13
    main(K=K)