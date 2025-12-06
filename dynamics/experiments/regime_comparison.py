# -*- coding: utf-8 -*-
"""
Created on Sun Nov 23 09:34:00 2025

@author: chris and christine
"""

#!/usr/bin/env python3
"""
Dynamics Regime Comparison Experiment
======================================

Compares different dynamical regimes for belief evolution:

1. **Overdamped (Gradient Flow)** - Current VFE approach
   θ̇ = -∇F(θ)
   - First-order dissipative dynamics
   - Energy monotonically decreases
   - Fast convergence, no oscillations
   - Like particle in thick fluid

2. **Underdamped (Hamiltonian)** - Conservative dynamics
   dθ/dt = G^{-1} p
   dp/dt = -∇F
   - Second-order conservative dynamics
   - Energy approximately conserved
   - Orbital motion, can overshoot
   - Like particle in vacuum

3. **Critically Damped** - Optimal convergence
   q̈ + 2√k q̇ + k q = 0
   - Fastest convergence without overshoot
   - Balance between overdamped and underdamped
   - Like particle in optimally viscous fluid

Research Questions:
------------------
1. Which regime converges faster to minimum?
2. Does conservative dynamics explore phase space better?
3. How does energy conservation quality depend on time step?
4. Do different regimes find different local minima?
5. What's the role of metric curvature in trajectory differences?

Author: Chris & Claude
Date: November 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass

from agent.trainer import Trainer, TrainingHistory
from agent.hamiltonian_trainer import HamiltonianTrainer, HamiltonianHistory
from agent.system import MultiAgentSystem
from agent.agents import Agent
from config import SystemConfig, AgentConfig, TrainingConfig
from gradients.free_energy_clean import compute_total_free_energy


@dataclass
class RegimeComparisonResults:
    """Results from comparing different dynamical regimes."""

    regime_name: str
    history: object  # TrainingHistory or HamiltonianHistory
    final_energy: float
    convergence_time: float  # Steps to reach 95% of final energy
    energy_trajectory: np.ndarray
    computation_time: float

    # Hamiltonian-specific (None for gradient flow)
    energy_conservation: float = None
    kinetic_trajectory: np.ndarray = None


def create_test_system(
    n_agents: int = 3,
    spatial_shape: Tuple[int, ...] = (),
    K: int = 13,
    lambda_belief_align: float = 1.0,
    lambda_self: float = 1.0
) -> MultiAgentSystem:
    """
    Create test system for regime comparison.

    Args:
        n_agents: Number of agents
        spatial_shape: Spatial dimensions (empty = 0D particles)
        K: Latent dimension (must be odd for SO(3))
        lambda_belief_align: Belief alignment strength
        lambda_self: Self-energy strength

    Returns:
        system: Configured multi-agent system
    """
    # System configuration
    system_config = SystemConfig(
        lambda_self=lambda_self,
        lambda_belief_align=lambda_belief_align,
        lambda_prior_align=1.0,
        lambda_gauge_smooth=0.0
    )

    # Agent configuration
    agent_config = AgentConfig(
        spatial_shape=spatial_shape,
        K=K,
        mu_scale=0.5,
        sigma_scale=0.3,
        lr_mu_q=0.1,
        lr_sigma_q=0.005
    )

    # Create agents
    agents = []
    rng = np.random.default_rng(42)  # Fixed seed for reproducibility

    for i in range(n_agents):
        agent = Agent(i, agent_config, rng=rng)
        agents.append(agent)

    # Create system
    system = MultiAgentSystem(agents, system_config)

    return system


def run_overdamped_regime(
    system: MultiAgentSystem,
    n_steps: int = 200,
    lr_scale: float = 1.0
) -> RegimeComparisonResults:
    """
    Run overdamped gradient flow (current VFE approach).

    Dynamics: θ̇ = -∇F(θ)

    Args:
        system: Multi-agent system
        n_steps: Training steps
        lr_scale: Learning rate multiplier

    Returns:
        results: Regime comparison results
    """
    import time

    print("\n" + "="*70)
    print("REGIME: OVERDAMPED (Gradient Flow)")
    print("="*70)

    # Configure training
    config = TrainingConfig(
        n_steps=n_steps,
        lr_mu_q=0.1 * lr_scale,
        lr_sigma_q=0.005 * lr_scale,
        log_every=1,
        save_history=True
    )

    # Train
    trainer = Trainer(system, config)

    start_time = time.perf_counter()
    history = trainer.train()
    computation_time = time.perf_counter() - start_time

    # Analyze results
    energy_traj = np.array(history.total_energy)
    final_energy = energy_traj[-1]

    # Convergence time: steps to reach 95% of final energy
    initial_energy = energy_traj[0]
    target_energy = initial_energy - 0.95 * (initial_energy - final_energy)

    convergence_idx = np.where(energy_traj <= target_energy)[0]
    convergence_time = convergence_idx[0] if len(convergence_idx) > 0 else n_steps

    results = RegimeComparisonResults(
        regime_name="Overdamped",
        history=history,
        final_energy=final_energy,
        convergence_time=convergence_time,
        energy_trajectory=energy_traj,
        computation_time=computation_time
    )

    print(f"\n✓ Overdamped training complete")
    print(f"  Final energy: {final_energy:.6f}")
    print(f"  Convergence time: {convergence_time} steps")
    print(f"  Computation time: {computation_time:.2f}s")

    return results


def run_underdamped_regime(
    system: MultiAgentSystem,
    n_steps: int = 200,
    dt: float = 0.01,
    friction: float = 0.1,
    mass_scale: float = 1.0
) -> RegimeComparisonResults:
    """
    Run underdamped Hamiltonian dynamics.

    Dynamics:
        dθ/dt = G^{-1} p
        dp/dt = -∇F - γ*p  (γ = friction)

    Args:
        system: Multi-agent system
        n_steps: Training steps
        dt: Time step size
        friction: Friction coefficient (0 = conservative)
        mass_scale: Mass tensor scale

    Returns:
        results: Regime comparison results
    """
    import time

    print("\n" + "="*70)
    regime_type = "UNDERDAMPED (Hamiltonian)" if friction < 0.01 else "DAMPED (Hamiltonian)"
    print(f"REGIME: {regime_type}")
    print("="*70)

    # Configure training
    config = TrainingConfig(
        n_steps=n_steps,
        log_every=1,
        save_history=True
    )

    # Train
    trainer = HamiltonianTrainer(
        system,
        config,
        friction=friction,
        mass_scale=mass_scale
    )

    start_time = time.perf_counter()
    history = trainer.train(dt=dt)
    computation_time = time.perf_counter() - start_time

    # Analyze results
    energy_traj = np.array(history.total_energy)  # Free energy (V)
    hamiltonian_traj = np.array(history.total_hamiltonian)  # H = T + V
    kinetic_traj = np.array(history.kinetic_energy)  # T

    final_energy = energy_traj[-1]

    # Energy conservation quality
    H0 = hamiltonian_traj[0]
    H_final = hamiltonian_traj[-1]
    energy_conservation = abs(H_final - H0) / (abs(H0) + 1e-10)

    # Convergence time (based on free energy V, not total H)
    initial_energy = energy_traj[0]
    target_energy = initial_energy - 0.95 * (initial_energy - final_energy)

    convergence_idx = np.where(energy_traj <= target_energy)[0]
    convergence_time = convergence_idx[0] if len(convergence_idx) > 0 else n_steps

    results = RegimeComparisonResults(
        regime_name=f"Underdamped (γ={friction})",
        history=history,
        final_energy=final_energy,
        convergence_time=convergence_time,
        energy_trajectory=energy_traj,
        computation_time=computation_time,
        energy_conservation=energy_conservation,
        kinetic_trajectory=kinetic_traj
    )

    print(f"\n✓ Hamiltonian training complete")
    print(f"  Final free energy (V): {final_energy:.6f}")
    print(f"  Energy conservation: {energy_conservation:.2e}")
    print(f"  Convergence time: {convergence_time} steps")
    print(f"  Computation time: {computation_time:.2f}s")

    return results


def plot_regime_comparison(
    results_dict: Dict[str, RegimeComparisonResults],
    out_dir: Path
):
    """
    Create comprehensive comparison plots.

    Generates:
        1. Energy trajectories (all regimes overlaid)
        2. Phase portraits (if Hamiltonian data available)
        3. Convergence comparison
        4. Energy conservation (Hamiltonian only)

    Args:
        results_dict: Dictionary mapping regime name to results
        out_dir: Output directory
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Create figure with subplots
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # ==================================================================
    # 1. Energy Trajectories (All Regimes)
    # ==================================================================
    ax1 = fig.add_subplot(gs[0, :])

    for name, res in results_dict.items():
        steps = np.array(res.history.steps)
        energy = res.energy_trajectory

        ax1.plot(steps, energy, linewidth=2, label=name, alpha=0.8)

    ax1.set_xlabel("Step", fontsize=12)
    ax1.set_ylabel("Free Energy F(θ)", fontsize=12)
    ax1.set_title("Energy Evolution Across Regimes", fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(alpha=0.3)

    # ==================================================================
    # 2. Hamiltonian Components (if available)
    # ==================================================================
    ax2 = fig.add_subplot(gs[1, 0])

    for name, res in results_dict.items():
        if res.kinetic_trajectory is not None:
            steps = np.array(res.history.steps)
            T = res.kinetic_trajectory
            V = res.energy_trajectory
            H = T + V

            ax2.plot(steps, H, linewidth=2, label=f"{name}: H", alpha=0.7)
            ax2.plot(steps, T, '--', linewidth=1.5, label=f"{name}: T", alpha=0.6)
            ax2.plot(steps, V, ':', linewidth=1.5, label=f"{name}: V", alpha=0.6)

    ax2.set_xlabel("Step", fontsize=11)
    ax2.set_ylabel("Energy", fontsize=11)
    ax2.set_title("Hamiltonian Decomposition (H = T + V)", fontsize=12)
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)

    # ==================================================================
    # 3. Energy Conservation (Hamiltonian regimes)
    # ==================================================================
    ax3 = fig.add_subplot(gs[1, 1])

    for name, res in results_dict.items():
        if hasattr(res.history, 'energy_drift'):
            steps = np.array(res.history.steps)
            drift = np.array(res.history.energy_drift)

            ax3.semilogy(steps, drift + 1e-12, linewidth=2, label=name, alpha=0.7)

    ax3.axhline(0.01, color='red', linestyle='--', alpha=0.5, label='1% drift')
    ax3.set_xlabel("Step", fontsize=11)
    ax3.set_ylabel("|H(t) - H(0)| / H(0)", fontsize=11)
    ax3.set_title("Energy Conservation Quality", fontsize=12)
    ax3.legend(fontsize=9)
    ax3.grid(alpha=0.3, which='both')

    # ==================================================================
    # 4. Convergence Comparison (Bar Chart)
    # ==================================================================
    ax4 = fig.add_subplot(gs[1, 2])

    names = list(results_dict.keys())
    convergence_times = [results_dict[n].convergence_time for n in names]

    bars = ax4.bar(range(len(names)), convergence_times, alpha=0.7)
    ax4.set_xticks(range(len(names)))
    ax4.set_xticklabels(names, rotation=45, ha='right')
    ax4.set_ylabel("Steps to 95% Convergence", fontsize=11)
    ax4.set_title("Convergence Speed", fontsize=12)
    ax4.grid(alpha=0.3, axis='y')

    # Color bars by regime type
    for i, name in enumerate(names):
        if 'Overdamped' in name:
            bars[i].set_color('C0')
        elif 'γ=0' in name:
            bars[i].set_color('C1')
        else:
            bars[i].set_color('C2')

    # ==================================================================
    # 5. Phase Space (Hamiltonian regimes) - if available
    # ==================================================================
    ax5 = fig.add_subplot(gs[2, :])

    for name, res in results_dict.items():
        if hasattr(res.history, 'momentum_norms'):
            energy = res.energy_trajectory
            momentum = np.array(res.history.momentum_norms)

            # Trim to same length
            min_len = min(len(energy), len(momentum))
            energy = energy[:min_len]
            momentum = momentum[:min_len]

            ax5.scatter(energy, momentum, s=20, alpha=0.5, label=name)

    ax5.set_xlabel("Free Energy V(θ)", fontsize=11)
    ax5.set_ylabel("Momentum Norm ||p||", fontsize=11)
    ax5.set_title("Phase Space Portrait (V, ||p||)", fontsize=12)
    ax5.legend(fontsize=9)
    ax5.grid(alpha=0.3)

    # ==================================================================
    # Save figure
    # ==================================================================
    plt.suptitle("Dynamical Regime Comparison: Overdamped vs Underdamped",
                 fontsize=16, fontweight='bold', y=0.995)

    out_path = out_dir / "regime_comparison_dashboard.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\n✓ Saved comparison dashboard: {out_path}")


def run_comprehensive_comparison(
    n_steps: int = 300,
    out_dir: Path = Path("_results/regime_comparison")
):
    """
    Run comprehensive regime comparison experiment.

    Compares:
        1. Overdamped (gradient flow)
        2. Underdamped (Hamiltonian, γ=0)
        3. Lightly damped (Hamiltonian, γ=0.1)
        4. Critically damped (Hamiltonian, γ=1.0)

    Args:
        n_steps: Number of training steps
        out_dir: Output directory
    """
    print("\n" + "="*70)
    print("COMPREHENSIVE DYNAMICS REGIME COMPARISON")
    print("="*70)
    print(f"Steps: {n_steps}")
    print(f"Output: {out_dir}")
    print("="*70)

    # Create test system (3 agents, 0D particles, K=3)
    print("\nCreating test system...")
    system_template = create_test_system(
        n_agents=3,
        spatial_shape=(),  # 0D particles
        K=13,
        lambda_belief_align=1.0,
        lambda_self=1.0
    )

    results = {}

    # ==================================================================
    # 1. Overdamped (Gradient Flow)
    # ==================================================================
    # Fresh copy of system
    system = create_test_system(3, (), 13, 1.0, 1.0)
    results['Overdamped'] = run_overdamped_regime(system, n_steps=n_steps)

    # ==================================================================
    # 2. Underdamped (Hamiltonian, no friction)
    # ==================================================================
    system = create_test_system(3, (), 13, 1.0, 1.0)
    results['Underdamped (γ=0)'] = run_underdamped_regime(
        system, n_steps=n_steps, dt=0.01, friction=0.0
    )

    # ==================================================================
    # 3. Lightly Damped
    # ==================================================================
    system = create_test_system(3, (), 13, 1.0, 1.0)
    results['Damped (γ=0.1)'] = run_underdamped_regime(
        system, n_steps=n_steps, dt=0.01, friction=0.5
    )

    # ==================================================================
    # 4. Critically Damped
    # ==================================================================
    system = create_test_system(3, (), 13, 1.0, 1.0)
    results['Damped (γ=1.0)'] = run_underdamped_regime(
        system, n_steps=n_steps, dt=0.01, friction=1.0
    )

    # ==================================================================
    # Generate comparison plots
    # ==================================================================
    print("\n" + "="*70)
    print("GENERATING COMPARISON PLOTS")
    print("="*70)

    plot_regime_comparison(results, out_dir)

    # ==================================================================
    # Print summary table
    # ==================================================================
    print("\n" + "="*70)
    print("SUMMARY TABLE")
    print("="*70)
    print(f"{'Regime':<25} {'Final E':>12} {'Conv. Steps':>12} {'Time (s)':>10} {'E. Cons.':>10}")
    print("-" * 70)

    for name, res in results.items():
        e_cons = f"{res.energy_conservation:.2e}" if res.energy_conservation is not None else "N/A"
        print(f"{name:<25} {res.final_energy:>12.6f} {res.convergence_time:>12} "
              f"{res.computation_time:>10.2f} {e_cons:>10}")

    print("="*70)
    print("\n✓ Regime comparison complete!")
    print(f"  Results saved in: {out_dir}")
    print("="*70)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Dynamics regime comparison experiment")
    parser.add_argument("--steps", type=int, default=300, help="Number of training steps")
    parser.add_argument("--out-dir", type=str, default="_results/regime_comparison",
                        help="Output directory")
    args = parser.parse_args()

    run_comprehensive_comparison(n_steps=args.steps, out_dir=Path(args.out_dir))