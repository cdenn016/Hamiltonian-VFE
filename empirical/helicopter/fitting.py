# -*- coding: utf-8 -*-
"""
Model Fitting and Comparison
============================

Fit gradient and Hamiltonian models to human behavioral data and compare
their predictions.

Key Questions:
1. Does Hamiltonian dynamics better predict human updates?
2. Do humans show "inertia" effects (momentum)?
3. Is belief updating smoother than gradient descent predicts?

Author: Hamiltonian-VFE Team
Date: December 2025
"""

import sys
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable
from pathlib import Path
from scipy.optimize import minimize, differential_evolution

# Handle both module execution and direct script execution
try:
    from .data_loader import SubjectData
    from .belief_agent import (
        BeliefAgent1D,
        GradientDynamics,
        HamiltonianDynamics,
        create_gradient_agent,
        create_hamiltonian_agent
    )
except ImportError:
    _this_dir = Path(__file__).parent
    _project_root = _this_dir.parent.parent
    if str(_project_root) not in sys.path:
        sys.path.insert(0, str(_project_root))

    from empirical.helicopter.data_loader import SubjectData
    from empirical.helicopter.belief_agent import (
        BeliefAgent1D,
        GradientDynamics,
        HamiltonianDynamics,
        create_gradient_agent,
        create_hamiltonian_agent
    )


# =============================================================================
# Model Fit Results
# =============================================================================

@dataclass
class ModelFit:
    """Results from fitting a model to subject data."""
    model_type: str                     # 'gradient' or 'hamiltonian'
    subject_id: int
    parameters: Dict[str, float]        # Fitted parameters
    mse: float                          # Mean squared error on updates
    correlation: float                  # Correlation with human updates
    mae: float                          # Mean absolute error
    log_likelihood: float               # Log likelihood (if applicable)

    # Detailed predictions
    predicted_updates: np.ndarray = field(repr=False)
    human_updates: np.ndarray = field(repr=False)
    predicted_beliefs: np.ndarray = field(repr=False)
    human_beliefs: np.ndarray = field(repr=False)

    # Per-trial diagnostics
    trial_errors: np.ndarray = field(repr=False)

    @property
    def r_squared(self) -> float:
        """Coefficient of determination."""
        ss_res = np.sum((self.human_updates - self.predicted_updates)**2)
        ss_tot = np.sum((self.human_updates - np.mean(self.human_updates))**2)
        return 1 - ss_res / ss_tot if ss_tot > 0 else 0.0


@dataclass
class ModelComparison:
    """Comparison between gradient and Hamiltonian models."""
    subject_id: int
    gradient_fit: ModelFit
    hamiltonian_fit: ModelFit

    @property
    def mse_improvement(self) -> float:
        """Relative MSE improvement of Hamiltonian over Gradient."""
        return (self.gradient_fit.mse - self.hamiltonian_fit.mse) / self.gradient_fit.mse

    @property
    def correlation_improvement(self) -> float:
        """Improvement in correlation."""
        return self.hamiltonian_fit.correlation - self.gradient_fit.correlation

    @property
    def hamiltonian_wins(self) -> bool:
        """Does Hamiltonian model fit better?"""
        return self.hamiltonian_fit.mse < self.gradient_fit.mse

    def summary(self) -> str:
        """Human-readable summary."""
        winner = "Hamiltonian" if self.hamiltonian_wins else "Gradient"
        return (
            f"Subject {self.subject_id}: {winner} wins\n"
            f"  Gradient MSE: {self.gradient_fit.mse:.4f}, r={self.gradient_fit.correlation:.3f}\n"
            f"  Hamiltonian MSE: {self.hamiltonian_fit.mse:.4f}, r={self.hamiltonian_fit.correlation:.3f}\n"
            f"  MSE improvement: {100*self.mse_improvement:.1f}%"
        )


# =============================================================================
# Loss Functions
# =============================================================================

def compute_update_mse(model_updates: np.ndarray,
                       human_updates: np.ndarray) -> float:
    """Mean squared error on belief updates."""
    valid = np.isfinite(model_updates) & np.isfinite(human_updates)
    if not np.any(valid):
        return np.inf
    return np.mean((model_updates[valid] - human_updates[valid])**2)


def compute_belief_mse(model_beliefs: np.ndarray,
                       human_beliefs: np.ndarray) -> float:
    """Mean squared error on beliefs (predictions)."""
    valid = np.isfinite(model_beliefs) & np.isfinite(human_beliefs)
    if not np.any(valid):
        return np.inf
    return np.mean((model_beliefs[valid] - human_beliefs[valid])**2)


def compute_correlation(model_updates: np.ndarray,
                        human_updates: np.ndarray) -> float:
    """Pearson correlation between model and human updates."""
    valid = np.isfinite(model_updates) & np.isfinite(human_updates)
    if np.sum(valid) < 3:
        return 0.0
    return np.corrcoef(model_updates[valid], human_updates[valid])[0, 1]


# =============================================================================
# Fitting Functions
# =============================================================================

def fit_gradient_model(subject: SubjectData,
                       optimize: bool = True) -> ModelFit:
    """
    Fit gradient descent model to subject data.

    Parameters to fit:
        - learning_rate: Step size for gradient descent

    Args:
        subject: Subject data
        optimize: If True, optimize learning rate. If False, use default.

    Returns:
        ModelFit with results
    """
    arrays = subject.get_arrays()
    observations = arrays['outcome']
    human_beliefs = arrays['prediction']
    human_updates = arrays['update']
    noise_stds = arrays['noise_std']

    def loss(params):
        lr = params[0]
        if lr <= 0 or lr > 2:
            return np.inf

        dynamics = GradientDynamics(learning_rate=lr)
        # Use median noise level
        obs_noise = np.median(noise_stds)
        agent = BeliefAgent1D(
            dynamics=dynamics,
            initial_mu=human_beliefs[0] if len(human_beliefs) > 0 else 150.0,
            obs_noise_sq=obs_noise**2,
        )

        agent.run_trial_sequence(observations)
        model_updates = agent.get_updates()

        return compute_update_mse(model_updates, human_updates)

    if optimize:
        # Optimize learning rate
        result = minimize(
            loss,
            x0=[0.3],
            bounds=[(0.01, 1.5)],
            method='L-BFGS-B'
        )
        best_lr = result.x[0]
    else:
        best_lr = 0.3

    # Run with best parameters
    obs_noise = np.median(noise_stds)
    dynamics = GradientDynamics(learning_rate=best_lr)
    agent = BeliefAgent1D(
        dynamics=dynamics,
        initial_mu=human_beliefs[0] if len(human_beliefs) > 0 else 150.0,
        obs_noise_sq=obs_noise**2,
    )
    agent.run_trial_sequence(observations)

    model_updates = agent.get_updates()
    model_beliefs = agent.get_predictions()

    return ModelFit(
        model_type='gradient',
        subject_id=subject.subject_id,
        parameters={'learning_rate': best_lr},
        mse=compute_update_mse(model_updates, human_updates),
        correlation=compute_correlation(model_updates, human_updates),
        mae=np.nanmean(np.abs(model_updates - human_updates)),
        log_likelihood=0.0,  # TODO: compute properly
        predicted_updates=model_updates,
        human_updates=human_updates,
        predicted_beliefs=model_beliefs,
        human_beliefs=human_beliefs,
        trial_errors=model_updates - human_updates,
    )


def fit_hamiltonian_model(subject: SubjectData,
                          optimize: bool = True) -> ModelFit:
    """
    Fit Hamiltonian dynamics model to subject data.

    Parameters to fit:
        - mass: Inertia of belief
        - friction: Damping coefficient

    Args:
        subject: Subject data
        optimize: If True, optimize parameters. If False, use defaults.

    Returns:
        ModelFit with results
    """
    arrays = subject.get_arrays()
    observations = arrays['outcome']
    human_beliefs = arrays['prediction']
    human_updates = arrays['update']
    noise_stds = arrays['noise_std']

    def loss(params):
        mass, friction = params
        if mass <= 0 or friction < 0 or friction > 5:
            return np.inf

        dynamics = HamiltonianDynamics(mass=mass, friction=friction)
        obs_noise = np.median(noise_stds)
        agent = BeliefAgent1D(
            dynamics=dynamics,
            initial_mu=human_beliefs[0] if len(human_beliefs) > 0 else 150.0,
            obs_noise_sq=obs_noise**2,
        )

        agent.run_trial_sequence(observations)
        model_updates = agent.get_updates()

        return compute_update_mse(model_updates, human_updates)

    if optimize:
        # Optimize mass and friction using differential evolution
        # (more robust for this 2D landscape)
        result = differential_evolution(
            loss,
            bounds=[(0.1, 10.0), (0.01, 3.0)],
            seed=42,
            maxiter=100,
            polish=True,
        )
        best_mass, best_friction = result.x
    else:
        best_mass, best_friction = 1.0, 0.5

    # Run with best parameters
    obs_noise = np.median(noise_stds)
    dynamics = HamiltonianDynamics(mass=best_mass, friction=best_friction)
    agent = BeliefAgent1D(
        dynamics=dynamics,
        initial_mu=human_beliefs[0] if len(human_beliefs) > 0 else 150.0,
        obs_noise_sq=obs_noise**2,
    )
    agent.run_trial_sequence(observations)

    model_updates = agent.get_updates()
    model_beliefs = agent.get_predictions()

    return ModelFit(
        model_type='hamiltonian',
        subject_id=subject.subject_id,
        parameters={'mass': best_mass, 'friction': best_friction},
        mse=compute_update_mse(model_updates, human_updates),
        correlation=compute_correlation(model_updates, human_updates),
        mae=np.nanmean(np.abs(model_updates - human_updates)),
        log_likelihood=0.0,
        predicted_updates=model_updates,
        human_updates=human_updates,
        predicted_beliefs=model_beliefs,
        human_beliefs=human_beliefs,
        trial_errors=model_updates - human_updates,
    )


def fit_subject(subject: SubjectData, optimize: bool = True) -> ModelComparison:
    """
    Fit both models to a subject and compare.

    Args:
        subject: Subject data
        optimize: Whether to optimize parameters

    Returns:
        ModelComparison with both fits
    """
    grad_fit = fit_gradient_model(subject, optimize=optimize)
    ham_fit = fit_hamiltonian_model(subject, optimize=optimize)

    return ModelComparison(
        subject_id=subject.subject_id,
        gradient_fit=grad_fit,
        hamiltonian_fit=ham_fit,
    )


def compare_dynamics(subjects: Dict[int, SubjectData],
                     optimize: bool = True,
                     verbose: bool = True) -> List[ModelComparison]:
    """
    Compare gradient and Hamiltonian dynamics across all subjects.

    Args:
        subjects: Dictionary of subject data
        optimize: Whether to optimize parameters
        verbose: Print progress

    Returns:
        List of ModelComparison objects
    """
    comparisons = []

    for i, (subj_id, subject) in enumerate(subjects.items()):
        if verbose:
            print(f"Fitting subject {subj_id} ({i+1}/{len(subjects)})...")

        comparison = fit_subject(subject, optimize=optimize)
        comparisons.append(comparison)

        if verbose:
            print(f"  {comparison.summary()}")

    return comparisons


def summarize_comparisons(comparisons: List[ModelComparison]) -> Dict:
    """Compute summary statistics across all subjects."""
    n_subjects = len(comparisons)
    n_hamiltonian_wins = sum(1 for c in comparisons if c.hamiltonian_wins)

    grad_mses = [c.gradient_fit.mse for c in comparisons]
    ham_mses = [c.hamiltonian_fit.mse for c in comparisons]
    grad_corrs = [c.gradient_fit.correlation for c in comparisons]
    ham_corrs = [c.hamiltonian_fit.correlation for c in comparisons]

    # Extract fitted parameters
    ham_masses = [c.hamiltonian_fit.parameters['mass'] for c in comparisons]
    ham_frictions = [c.hamiltonian_fit.parameters['friction'] for c in comparisons]
    grad_lrs = [c.gradient_fit.parameters['learning_rate'] for c in comparisons]

    return {
        'n_subjects': n_subjects,
        'n_hamiltonian_wins': n_hamiltonian_wins,
        'hamiltonian_win_rate': n_hamiltonian_wins / n_subjects,

        'gradient_mean_mse': np.mean(grad_mses),
        'gradient_std_mse': np.std(grad_mses),
        'hamiltonian_mean_mse': np.mean(ham_mses),
        'hamiltonian_std_mse': np.std(ham_mses),

        'gradient_mean_corr': np.mean(grad_corrs),
        'hamiltonian_mean_corr': np.mean(ham_corrs),

        'mean_mse_improvement': np.mean([c.mse_improvement for c in comparisons]),

        # Parameter distributions
        'gradient_lr_mean': np.mean(grad_lrs),
        'gradient_lr_std': np.std(grad_lrs),
        'hamiltonian_mass_mean': np.mean(ham_masses),
        'hamiltonian_mass_std': np.std(ham_masses),
        'hamiltonian_friction_mean': np.mean(ham_frictions),
        'hamiltonian_friction_std': np.std(ham_frictions),
    }


# =============================================================================
# Main (for testing)
# =============================================================================

if __name__ == '__main__':
    from .data_loader import load_mcguire_nassar_2014

    print("Loading data...")
    subjects = load_mcguire_nassar_2014()

    # Fit first 3 subjects as a test
    test_subjects = {k: subjects[k] for k in list(subjects.keys())[:3]}

    print("\nFitting models...")
    comparisons = compare_dynamics(test_subjects, optimize=True, verbose=True)

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    summary = summarize_comparisons(comparisons)
    for key, val in summary.items():
        if isinstance(val, float):
            print(f"  {key}: {val:.4f}")
        else:
            print(f"  {key}: {val}")
