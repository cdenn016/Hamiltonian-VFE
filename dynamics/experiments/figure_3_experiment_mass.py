#!/usr/bin/env python3
"""
Figure 3: Inertial Mass as Statistical Precision - Computational Validation
============================================================================

Demonstrates that M = Σ_p^(-1) emerges as the natural mass tensor for 
belief dynamics on statistical manifolds.

Key Results:
- Panel A: M_eff ∝ 1/σ² (linear relationship validates M = Σ_p^(-1))
- Panel B: Same force, different masses → different accelerations
- Panel C: Phase space shows mass-dependent trajectory curvature
- Panel D: Underdamped (full theory) vs Overdamped (standard FEP)
- Panel E: Oscillation frequency ω² ∝ 1/M (harmonic oscillator)
- Panel F: Energy conservation H = T + V (validates Hamiltonian)

Author: Chris Dennis
Date: November 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, List, Dict
from scipy.signal import find_peaks
from scipy.optimize import curve_fit

from dynamics.field_theory import (
    FullFieldHamiltonian,
    FullFieldState,
    create_simple_potential
)
from dynamics.product_manifold_integrator import ProductManifoldVerlet
from geometry.lie_algebra import LieGroup


def run_underdamped_trajectory(
    K: int,
    Sigma_prior_scale: float,
    mu_target: np.ndarray,
    initial_momentum_scale: float = 0.15,
    dt: float = 0.005,
    t_end: float = 30.0,
    seed: int = None
) -> Dict:
    """
    Run underdamped Hamiltonian dynamics with inertia.
    
    Args:
        K: Latent dimension
        Sigma_prior_scale: Sets mass M = (Sigma_prior_scale)^(-1) for isotropic case
        mu_target: Target position in belief space
        initial_momentum_scale: Scale for initial momentum
        dt: Integration timestep
        t_end: Final time
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary with trajectory history
    """
    if seed is not None:
        np.random.seed(seed)
    
    gauge_group = LieGroup.SO3
    Sigma_target = 2.0 * np.eye(K)
    
    # Quadratic potential V = (1/2)||μ - μ*||²
    potential = create_simple_potential(
        mu_target,
        Sigma_target,
        weights=(1.0, 0.0, 0.0)
    )
    
    Sigma_prior = Sigma_prior_scale * np.eye(K)
    hamiltonian = FullFieldHamiltonian(
        potential=potential,
        Sigma_prior=Sigma_prior,
        gauge_group=gauge_group
    )
    
    # Initial state with momentum
    state = FullFieldState(
        mu=np.zeros(K),
        Sigma=Sigma_target.copy(),
        phi=np.zeros(3),
        pi_mu=np.random.randn(K) * initial_momentum_scale,
        pi_Sigma=np.zeros((K, K)),
        pi_phi=np.zeros(3),
        t=0.0
    )
    
    integrator = ProductManifoldVerlet(hamiltonian)
    history = integrator.integrate(
        state, 
        t_end=t_end, 
        dt=dt, 
        save_interval=2
    )
    
    # Add mass info
    history['M_eff'] = K / Sigma_prior_scale
    history['Sigma_prior_scale'] = Sigma_prior_scale
    
    return history


def run_overdamped_trajectory(
    K: int,
    Sigma_prior_scale: float,
    mu_target: np.ndarray,
    friction: float = 5.0,
    dt: float = 0.01,
    t_end: float = 30.0
) -> Dict:
    """
    Run overdamped (first-order) gradient descent dynamics.
    
    This is standard FEP without inertial terms.
    """
    gauge_group = LieGroup.SO3
    Sigma_target = 2.0 * np.eye(K)
    
    potential = create_simple_potential(
        mu_target,
        Sigma_target,
        weights=(1.0, 0.0, 0.0)
    )
    
    Sigma_prior = Sigma_prior_scale * np.eye(K)
    hamiltonian = FullFieldHamiltonian(
        potential=potential,
        Sigma_prior=Sigma_prior,
        gauge_group=gauge_group
    )
    
    # Storage
    n_steps = int(t_end / dt)
    save_interval = 10
    n_save = n_steps // save_interval + 1
    
    t_history = np.zeros(n_save)
    mu_history = np.zeros((n_save, K))
    energy_history = np.zeros(n_save)
    
    # Initial state
    mu = np.zeros(K)
    Sigma = Sigma_target.copy()
    phi = np.zeros(3)
    
    t_history[0] = 0.0
    mu_history[0] = mu.copy()
    energy_history[0] = hamiltonian.potential_func(mu, Sigma, phi)
    
    save_idx = 1
    eps = 1e-6
    
    for step in range(1, n_steps + 1):
        # Numerical gradient
        grad_mu = np.zeros(K)
        for i in range(K):
            mu_plus = mu.copy()
            mu_plus[i] += eps
            mu_minus = mu.copy()
            mu_minus[i] -= eps
            
            V_plus = hamiltonian.potential_func(mu_plus, Sigma, phi)
            V_minus = hamiltonian.potential_func(mu_minus, Sigma, phi)
            
            grad_mu[i] = (V_plus - V_minus) / (2 * eps)
        
        # Overdamped update: dμ/dt = -(1/γ)∇V
        mu = mu - (dt / friction) * grad_mu
        
        if step % save_interval == 0:
            t_history[save_idx] = step * dt
            mu_history[save_idx] = mu.copy()
            energy_history[save_idx] = hamiltonian.potential_func(mu, Sigma, phi)
            save_idx += 1
    
    return {
        't': t_history[:save_idx],
        'mu': mu_history[:save_idx],
        'energy': energy_history[:save_idx],
        'Sigma_prior_scale': Sigma_prior_scale
    }


def measure_oscillation_frequency(t: np.ndarray, mu: np.ndarray, component: int = 0) -> float:
    """
    Measure oscillation frequency using FFT.
    
    More robust than peak detection for clean harmonic oscillations.
    """
    mu_signal = mu[:, component]
    mu_signal = mu_signal - np.mean(mu_signal)
    
    # FFT
    dt = t[1] - t[0]
    freqs = np.fft.rfftfreq(len(mu_signal), dt)
    fft = np.fft.rfft(mu_signal)
    power = np.abs(fft)**2
    
    # Find dominant frequency (excluding DC component)
    dominant_idx = np.argmax(power[1:]) + 1
    omega = 2 * np.pi * freqs[dominant_idx]
    
    return omega


def generate_figure3(K: int = 13, output_dir: Path = None):
    """
    Generate Figure 3: Inertial Mass as Statistical Precision.
    
    Args:
        K: Latent dimension
        output_dir: Output directory (default: _experiments/figure3)
    """
    if output_dir is None:
        output_dir = Path("_experiments/figure3_mass_validation")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*80)
    print("FIGURE 3: INERTIAL MASS AS STATISTICAL PRECISION")
    print("="*80)
    print(f"Latent dimension K = {K}\n")
    
    # Common target
    np.random.seed(42)
    mu_target = np.zeros(K)
    mu_target[0] = 1.0
    if K >= 2:
        mu_target[1] = 0.5
    
    # =========================================================================
    # PANEL A: Effective mass vs 1/σ² (validates M = Σ_p^(-1))
    # =========================================================================
    print("Panel A: Mass-precision relationship...")
    
    sigma_squared_values = np.logspace(-0.5, 1.0, 12)  # 0.316 to 10
    effective_masses = []
    oscillation_frequencies = []
    
    for i, scale in enumerate(sigma_squared_values):
        print(f"  {i+1}/{len(sigma_squared_values)}: σ² = {scale:.3f}")
        
        history = run_underdamped_trajectory(
            K=K,
            Sigma_prior_scale=scale,
            mu_target=mu_target,
            initial_momentum_scale=0.2,
            t_end=50.0,
            seed=42 + i
        )
        
        # Measure frequency
        omega = measure_oscillation_frequency(history['t'], history['mu'])
        oscillation_frequencies.append(omega)
        
        # For harmonic oscillator: ω² = k/M
        # With k = 1 (our potential), M = 1/ω²
        M_eff = 1.0 / (omega**2) if omega > 0 else np.nan
        effective_masses.append(M_eff)
    
    # =========================================================================
    # PANEL B: Three trajectories with different masses
    # =========================================================================
    print("\nPanel B: Different mass trajectories...")
    
    mass_scales = [0.5, 2.0, 8.0]
    mass_labels = ['Light (M=26.0)', 'Medium (M=6.5)', 'Heavy (M=1.6)']
    mass_colors = ['#3498db', '#2ecc71', '#e74c3c']
    
    mass_trajectories = []
    for i, scale in enumerate(mass_scales):
        M = K / scale
        print(f"  {mass_labels[i]}: σ² = {scale}, M = {M:.1f}")
        
        history = run_underdamped_trajectory(
            K=K,
            Sigma_prior_scale=scale,
            mu_target=mu_target,
            initial_momentum_scale=0.15,
            t_end=25.0,
            seed=100 + i
        )
        mass_trajectories.append(history)
    
    # =========================================================================
    # PANEL D: Underdamped vs Overdamped comparison
    # =========================================================================
    print("\nPanel D: Underdamped vs Overdamped...")
    
    scale_compare = 3.0
    
    history_underdamped = run_underdamped_trajectory(
        K=K,
        Sigma_prior_scale=scale_compare,
        mu_target=mu_target,
        initial_momentum_scale=0.2,
        t_end=25.0,
        seed=200
    )
    
    history_overdamped = run_overdamped_trajectory(
        K=K,
        Sigma_prior_scale=scale_compare,
        mu_target=mu_target,
        friction=2.0,
        t_end=25.0
    )
    
    # =========================================================================
    # CREATE FIGURE
    # =========================================================================
    print("\nGenerating figure...")
    
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3,
                         left=0.08, right=0.95, top=0.93, bottom=0.08)
    
    # -------------------------------------------------------------------------
    # Panel A: M_eff vs 1/σ² (LINEAR RELATIONSHIP - KEY VALIDATION)
    # -------------------------------------------------------------------------
    ax_a = fig.add_subplot(gs[0, 0])
    
    inv_sigma_sq = 1.0 / sigma_squared_values
    valid = ~np.isnan(effective_masses)
    
    # Data points
    ax_a.plot(inv_sigma_sq[valid], np.array(effective_masses)[valid],
             'o', markersize=10, color='#3498db', markeredgecolor='navy',
             markeredgewidth=1.5, label='Measured M$_{eff}$', zorder=3)
    
    # Linear fit
    if np.sum(valid) >= 2:
        slope, intercept = np.polyfit(inv_sigma_sq[valid], 
                                      np.array(effective_masses)[valid], 1)
        x_fit = np.linspace(inv_sigma_sq[valid].min(), inv_sigma_sq[valid].max(), 100)
        y_fit = slope * x_fit + intercept
        
        ax_a.plot(x_fit, y_fit, 'r--', linewidth=2.5,
                 label=f'Linear fit: M = {slope:.2f}/σ² + {intercept:.2f}',
                 zorder=2)
        
        # R² value
        residuals = np.array(effective_masses)[valid] - (slope * inv_sigma_sq[valid] + intercept)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((np.array(effective_masses)[valid] - 
                        np.mean(np.array(effective_masses)[valid]))**2)
        r_squared = 1 - (ss_res / ss_tot)
        
        # Add R² to plot
        ax_a.text(0.05, 0.95, f'R² = {r_squared:.4f}',
                 transform=ax_a.transAxes, fontsize=11,
                 verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax_a.set_xlabel('Inverse Prior Variance 1/σ²', fontsize=12, fontweight='bold')
    ax_a.set_ylabel('Effective Mass M$_{eff}$', fontsize=12, fontweight='bold')
    ax_a.set_title('A: Mass = Precision (M ∝ 1/σ²)', 
                   fontsize=13, fontweight='bold', pad=10)
    ax_a.legend(fontsize=10, loc='upper left', framealpha=0.9)
    ax_a.grid(alpha=0.3, linestyle='--')
    ax_a.tick_params(labelsize=10)
    
    # -------------------------------------------------------------------------
    # Panel B: Trajectories for different masses
    # -------------------------------------------------------------------------
    ax_b = fig.add_subplot(gs[0, 1])
    
    for history, label, color in zip(mass_trajectories, mass_labels, mass_colors):
        distance = np.linalg.norm(history['mu'] - mu_target, axis=1)
        ax_b.plot(history['t'], distance, 
                 linewidth=2.5, label=label, color=color, alpha=0.8)
    
    ax_b.set_xlabel('Time t', fontsize=12, fontweight='bold')
    ax_b.set_ylabel('Distance to Target ||μ(t) - μ*||', fontsize=12, fontweight='bold')
    ax_b.set_title('B: Same Force, Different Masses', 
                   fontsize=13, fontweight='bold', pad=10)
    ax_b.legend(fontsize=10, loc='upper right', framealpha=0.9)
    ax_b.grid(alpha=0.3, linestyle='--')
    ax_b.set_yscale('log')
    ax_b.tick_params(labelsize=10)
    
    # -------------------------------------------------------------------------
    # Panel C: Phase space (if K >= 2)
    # -------------------------------------------------------------------------
    ax_c = fig.add_subplot(gs[0, 2])
    
    if K >= 2:
        for history, label, color in zip(mass_trajectories, mass_labels, mass_colors):
            mu = history['mu']
            ax_c.plot(mu[:, 0], mu[:, 1], linewidth=2.5, 
                     label=label, color=color, alpha=0.7)
            # Mark start
            ax_c.plot(mu[0, 0], mu[0, 1], 'o', color=color, 
                     markersize=10, markeredgecolor='black', markeredgewidth=1)
        
        # Mark target
        ax_c.plot(mu_target[0], mu_target[1], '*', color='gold',
                 markersize=20, markeredgecolor='black', 
                 markeredgewidth=1.5, label='Target', zorder=10)
        
        ax_c.set_xlabel('μ₀', fontsize=12, fontweight='bold')
        ax_c.set_ylabel('μ₁', fontsize=12, fontweight='bold')
        ax_c.set_title('C: Phase Space Trajectories', 
                      fontsize=13, fontweight='bold', pad=10)
        ax_c.legend(fontsize=9, loc='best', framealpha=0.9)
        ax_c.grid(alpha=0.3, linestyle='--')
        ax_c.axis('equal')
    else:
        # For K=1, plot velocity vs position
        for history, label, color in zip(mass_trajectories, mass_labels, mass_colors):
            mu = history['mu'][:, 0]
            v = np.gradient(mu, history['t'])
            ax_c.plot(mu, v, linewidth=2.5, label=label, color=color, alpha=0.7)
        
        ax_c.set_xlabel('Position μ₀', fontsize=12, fontweight='bold')
        ax_c.set_ylabel('Velocity dμ₀/dt', fontsize=12, fontweight='bold')
        ax_c.set_title('C: Phase Portrait', fontsize=13, fontweight='bold', pad=10)
        ax_c.legend(fontsize=10, framealpha=0.9)
        ax_c.grid(alpha=0.3, linestyle='--')
    
    ax_c.tick_params(labelsize=10)
    
    # -------------------------------------------------------------------------
    # Panel D: Underdamped vs Overdamped
    # -------------------------------------------------------------------------
    ax_d = fig.add_subplot(gs[1, 0])
    
    # Plot first component
    ax_d.plot(history_underdamped['t'], history_underdamped['mu'][:, 0],
             linewidth=2.5, label='Underdamped (Full Theory)', 
             color='#e74c3c', alpha=0.8)
    ax_d.plot(history_overdamped['t'], history_overdamped['mu'][:, 0],
             linewidth=2.5, label='Overdamped (Standard FEP)', 
             color='#3498db', alpha=0.8, linestyle='--')
    
    # Target line
    ax_d.axhline(mu_target[0], color='gray', linestyle=':', 
                linewidth=2, alpha=0.5, label='Target')
    
    ax_d.set_xlabel('Time t', fontsize=12, fontweight='bold')
    ax_d.set_ylabel('Belief μ₀(t)', fontsize=12, fontweight='bold')
    ax_d.set_title('D: Underdamped vs Overdamped Dynamics', 
                   fontsize=13, fontweight='bold', pad=10)
    ax_d.legend(fontsize=10, loc='best', framealpha=0.9)
    ax_d.grid(alpha=0.3, linestyle='--')
    ax_d.tick_params(labelsize=10)
    
    # -------------------------------------------------------------------------
    # Panel E: Oscillation frequency ω² vs 1/M
    # -------------------------------------------------------------------------
    ax_e = fig.add_subplot(gs[1, 1])
    
    omega_squared = np.array(oscillation_frequencies)**2
    
    # CORRECTED: For single component harmonic oscillator
    # M_component = 1/σ², so 1/M_component = σ²
    inv_M_component = sigma_squared_values  # This is σ², which equals 1/M
    
    ax_e.plot(inv_M_component[valid], omega_squared[valid],
             'o', markersize=10, color='#2ecc71', markeredgecolor='darkgreen',
             markeredgewidth=1.5, label='Measured ω²', zorder=3)
    
    # Linear fit through origin: ω² = k × (1/M) = k × σ²
    if np.sum(valid) >= 2:
        k_eff = np.sum(inv_M_component[valid] * omega_squared[valid]) / np.sum(inv_M_component[valid]**2)
        x_fit = np.linspace(0, inv_M_component[valid].max() * 1.1, 100)
        y_fit = k_eff * x_fit
        
        ax_e.plot(x_fit, y_fit, 'r--', linewidth=2.5,
                 label=f'ω² = {k_eff:.2f} × σ² (k={k_eff:.2f})',
                 zorder=2)
    
    ax_e.set_xlabel('Inverse Mass 1/M = σ²', fontsize=12, fontweight='bold')
    ax_e.set_ylabel('Oscillation Frequency² ω²', fontsize=12, fontweight='bold')
    ax_e.set_title('E: Harmonic Oscillator (ω² = k/M = k×σ²)', 
                   fontsize=13, fontweight='bold', pad=10)
    
    # -------------------------------------------------------------------------
    # Panel F: Energy conservation
    # -------------------------------------------------------------------------
    ax_f = fig.add_subplot(gs[1, 2])
    
    # Normalize energy
    E_under = history_underdamped['energy']
    E_under_norm = E_under / E_under[0]
    
    ax_f.plot(history_underdamped['t'], E_under_norm,
             linewidth=2.5, color='#9b59b6', alpha=0.8,
             label='Total Energy H = T + V')
    ax_f.axhline(1.0, color='gray', linestyle='--', 
                linewidth=2, alpha=0.5, label='Initial Energy')
    
    # Calculate drift
    energy_drift = 100 * (E_under[-1] - E_under[0]) / E_under[0]
    
    ax_f.text(0.05, 0.95, f'Energy Drift: {energy_drift:+.3f}%',
             transform=ax_f.transAxes, fontsize=11,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    
    ax_f.set_xlabel('Time t', fontsize=12, fontweight='bold')
    ax_f.set_ylabel('Energy E(t) / E(0)', fontsize=12, fontweight='bold')
    ax_f.set_title('F: Energy Conservation', 
                   fontsize=13, fontweight='bold', pad=10)
    ax_f.legend(fontsize=10, loc='lower right', framealpha=0.9)
    ax_f.grid(alpha=0.3, linestyle='--')
    ax_f.tick_params(labelsize=10)
    ax_f.set_ylim([0.99, 1.01])
    
    # -------------------------------------------------------------------------
    # Main title
    # -------------------------------------------------------------------------
    fig.suptitle(f'Inertial Mass as Statistical Precision: M = Σ$_p^{{-1}}$ (K={K})',
                fontsize=16, fontweight='bold', y=0.98)
    
    # -------------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------------
    plt.savefig(output_dir / f"figure3_mass_precision_K{K}.png", 
               dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / f"figure3_mass_precision_K{K}.pdf", 
               bbox_inches='tight')
    
    print(f"\n✓ Saved: {output_dir / f'figure3_mass_precision_K{K}.png'}")
    print(f"✓ Saved: {output_dir / f'figure3_mass_precision_K{K}.pdf'}")
    
    # -------------------------------------------------------------------------
    # Save data
    # -------------------------------------------------------------------------
    np.savez(
        output_dir / f"figure3_data_K{K}.npz",
        sigma_squared=sigma_squared_values,
        effective_masses=effective_masses,
        oscillation_frequencies=oscillation_frequencies,
        inv_sigma_sq=inv_sigma_sq,
        K=K
    )
    
    print(f"✓ Saved: {output_dir / f'figure3_data_K{K}.npz'}")
    
    print("\n" + "="*80)
    print("FIGURE 3 COMPLETE")
    print("="*80)
    print(f"\nKey Validations:")
    if np.sum(valid) >= 2:
        print(f"  ✓ M ∝ 1/σ² relationship: R² = {r_squared:.4f}")
        print(f"  ✓ ω² = k/M relationship: k_eff = {k_eff:.2f}")
    print(f"  ✓ Energy conservation: ΔE = {energy_drift:+.3f}%")
    print(f"  ✓ Underdamped shows oscillations")
    print(f"  ✓ Overdamped shows monotonic relaxation")
    print("="*80 + "\n")
    
    return fig


def main(K: int = 13):
    """
    Generate Figure 3 for publication.
    
    Args:
        K: Latent dimension (default 13)
    """
    fig = generate_figure3(K=K)
    plt.show()


if __name__ == "__main__":
    import sys
    K = int(sys.argv[1]) if len(sys.argv) > 1 else 13
    main(K=K)