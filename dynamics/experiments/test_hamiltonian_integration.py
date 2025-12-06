# -*- coding: utf-8 -*-
"""
Created on Sun Nov 23 10:13:22 2025

@author: chris and christine
"""

#!/usr/bin/env python3
"""
Test Hamiltonian Integration
============================

Simple test to validate that the Hamiltonian trainer integrates
correctly with the agent system.

This script:
1. Creates a minimal 2-agent system
2. Runs both gradient flow and Hamiltonian training
3. Validates energy conservation
4. Compares convergence

Expected behavior:
- Gradient flow: Monotonic energy decrease
- Hamiltonian (γ=0): Energy conserved to ~1%
- Hamiltonian (γ=0.1): Light damping, slight energy decrease

Author: Chris & Claude
Date: November 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from agent.trainer import Trainer
from agent.hamiltonian_trainer import HamiltonianTrainer
from agent.system import MultiAgentSystem
from agent.agents import Agent
from config import SystemConfig, AgentConfig, TrainingConfig


def create_simple_system():
    """Create minimal 2-agent system for testing."""

    # System config
    system_config = SystemConfig(
        lambda_self=1.0,
        lambda_belief_align=1.0,
        lambda_prior_align=0.0
    )

    # Agent config
    agent_config = AgentConfig(
        spatial_shape=(),  # 0D particle
        K=3,  # Must be odd for SO(3)
        mu_scale=0.5,
        sigma_scale=0.3
    )

    # Create agents
    rng = np.random.default_rng(42)
    agents = [Agent(i, agent_config, rng=rng) for i in range(2)]

    # Create system
    system = MultiAgentSystem(agents, system_config)

    return system


def test_gradient_flow():
    """Test standard gradient flow trainer."""
    print("\n" + "="*70)
    print("TEST 1: GRADIENT FLOW (Overdamped)")
    print("="*70)

    system = create_simple_system()

    config = TrainingConfig(
        n_steps=50,
        lr_mu_q=0.01,
        lr_sigma_q=0.005,
        log_every=10,
        save_history=True
    )

    trainer = Trainer(system, config)
    history = trainer.train()

    # Validate monotonic decrease
    energies = np.array(history.total_energy)
    energy_changes = np.diff(energies)

    assert np.all(energy_changes <= 1e-6), "Energy should decrease monotonically"

    print(f"\n✓ Gradient flow test PASSED")
    print(f"  Energy decreased from {energies[0]:.6f} to {energies[-1]:.6f}")
    print(f"  All steps showed energy decrease: {np.all(energy_changes <= 0)}")

    return history


def test_hamiltonian_conservative():
    """Test Hamiltonian trainer without friction."""
    print("\n" + "="*70)
    print("TEST 2: HAMILTONIAN (Conservative, γ=0)")
    print("="*70)

    system = create_simple_system()

    config = TrainingConfig(
        n_steps=100,
        log_every=20,
        save_history=True
    )

    trainer = HamiltonianTrainer(
        system, config,
        friction=0.0,  # Conservative
        mass_scale=1.0
    )

    history = trainer.train(dt=0.01)

    # Check energy conservation
    H_traj = np.array(history.total_hamiltonian)
    H0 = H_traj[0]
    H_final = H_traj[-1]

    energy_drift = abs(H_final - H0) / abs(H0)

    # Should conserve energy to within ~5% (depends on time step)
    assert energy_drift < 0.05, f"Energy drift {energy_drift:.2e} too large"

    print(f"\n✓ Conservative Hamiltonian test PASSED")
    print(f"  Energy conservation: |ΔH|/H = {energy_drift:.2e}")
    print(f"  Initial H: {H0:.6f}")
    print(f"  Final H: {H_final:.6f}")

    return history


def test_hamiltonian_damped():
    """Test Hamiltonian trainer with light damping."""
    print("\n" + "="*70)
    print("TEST 3: HAMILTONIAN (Damped, γ=0.1)")
    print("="*70)

    system = create_simple_system()

    config = TrainingConfig(
        n_steps=100,
        log_every=20,
        save_history=True
    )

    trainer = HamiltonianTrainer(
        system, config,
        friction=0.1,  # Light damping
        mass_scale=1.0
    )

    history = trainer.train(dt=0.01)

    # With damping, Hamiltonian should decrease
    H_traj = np.array(history.total_hamiltonian)
    H_changes = np.diff(H_traj)

    # Most steps should show energy decrease (but some may increase due to curvature)
    decreasing_fraction = np.sum(H_changes <= 0) / len(H_changes)

    assert decreasing_fraction > 0.5, "Most steps should show energy decrease with damping"

    print(f"\n✓ Damped Hamiltonian test PASSED")
    print(f"  Fraction of steps with energy decrease: {decreasing_fraction:.1%}")
    print(f"  Total energy change: {H_traj[-1] - H_traj[0]:.6f}")

    return history


def plot_test_results(gf_history, ham_history, out_path="test_hamiltonian.png"):
    """Create comparison plot of test results."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Energy trajectories
    ax1 = axes[0, 0]
    ax1.plot(gf_history.steps, gf_history.total_energy,
             linewidth=2, label='Gradient Flow', color='C0')
    ax1.plot(ham_history.steps, ham_history.total_energy,
             linewidth=2, label='Hamiltonian', color='C1')
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Free Energy F(θ)")
    ax1.set_title("Energy Evolution")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Hamiltonian components
    ax2 = axes[0, 1]
    ax2.plot(ham_history.steps, ham_history.total_hamiltonian,
             linewidth=2, label='H (total)', color='black')
    ax2.plot(ham_history.steps, ham_history.kinetic_energy,
             '--', linewidth=2, label='T (kinetic)', color='C2')
    ax2.plot(ham_history.steps, ham_history.potential_energy,
             ':', linewidth=2, label='V (potential)', color='C3')
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Energy")
    ax2.set_title("Hamiltonian Decomposition (H = T + V)")
    ax2.legend()
    ax2.grid(alpha=0.3)

    # Energy conservation
    ax3 = axes[1, 0]
    ax3.semilogy(ham_history.steps, np.array(ham_history.energy_drift) + 1e-12,
                 linewidth=2, color='C4')
    ax3.axhline(0.01, color='red', linestyle='--', alpha=0.5, label='1% drift')
    ax3.axhline(0.05, color='orange', linestyle='--', alpha=0.5, label='5% drift')
    ax3.set_xlabel("Step")
    ax3.set_ylabel("|H(t) - H(0)| / H(0)")
    ax3.set_title("Energy Conservation Quality")
    ax3.legend()
    ax3.grid(alpha=0.3, which='both')

    # Phase space
    ax4 = axes[1, 1]
    V = np.array(ham_history.potential_energy)
    p_norm = np.array(ham_history.momentum_norms)
    scatter = ax4.scatter(V, p_norm, c=ham_history.steps,
                          cmap='viridis', s=30, alpha=0.6)
    ax4.set_xlabel("Potential V(θ)")
    ax4.set_ylabel("Momentum ||p||")
    ax4.set_title("Phase Space Portrait")
    plt.colorbar(scatter, ax=ax4, label='Step')
    ax4.grid(alpha=0.3)

    plt.suptitle("Hamiltonian Integration Test Results",
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()

    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved test results plot: {out_path}")


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("HAMILTONIAN INTEGRATION TEST SUITE")
    print("="*70)

    # Run tests
    gf_history = test_gradient_flow()
    ham_conservative_history = test_hamiltonian_conservative()
    ham_damped_history = test_hamiltonian_damped()

    # Plot results
    plot_test_results(gf_history, ham_conservative_history)

    print("\n" + "="*70)
    print("ALL TESTS PASSED ✓")
    print("="*70)
    print("\nHamiltonian trainer successfully integrated with agent system!")
    print("\nKey findings:")
    print("  1. Gradient flow: Energy decreases monotonically")
    print("  2. Hamiltonian (γ=0): Energy conserved to ~5%")
    print("  3. Hamiltonian (γ=0.1): Light damping causes controlled energy decrease")
    print("\nNext steps:")
    print("  - Run full regime comparison: python dynamics/experiments/regime_comparison.py")
    print("  - Experiment with different friction values")
    print("  - Test on larger systems with field agents")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()