# -*- coding: utf-8 -*-
"""
1D Belief Agent for Helicopter Task
====================================

Implements a 1D belief agent that tracks a scalar value (helicopter position)
using either gradient descent or Hamiltonian dynamics.

Key insight from "Inertia of Belief" manuscript:
- Gradient descent: Beliefs update proportional to prediction error
- Hamiltonian dynamics: Beliefs have MOMENTUM, leading to smoother trajectories
  and potential overshooting after changepoints

Mathematical Framework:
-----------------------
State: (μ, σ²) - belief mean and variance
Observation: o ~ N(μ_true, σ_obs²)

Free Energy (1D Gaussian):
    F = 0.5 * [(μ - μ_prior)²/σ_prior² + log(σ_prior²/σ²) + σ²/σ_prior² - 1]
      - E_q[log p(o|μ)]

Gradient Dynamics:
    dμ/dt = -∂F/∂μ = (o - μ)/σ² - (μ - μ_prior)/σ_prior²

Hamiltonian Dynamics:
    dμ/dt = p/m           (momentum drives position)
    dp/dt = -∂F/∂μ - γp   (force with friction)

Author: Hamiltonian-VFE Team
Date: December 2025
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Literal
from abc import ABC, abstractmethod


# =============================================================================
# Dynamics Base Class
# =============================================================================

class BeliefDynamics(ABC):
    """Abstract base class for belief update dynamics."""

    @abstractmethod
    def update(self,
               mu: float,
               sigma_sq: float,
               observation: float,
               obs_noise_sq: float,
               prior_mu: float,
               prior_sigma_sq: float,
               dt: float = 1.0) -> Tuple[float, float, dict]:
        """
        Update belief given new observation.

        Args:
            mu: Current belief mean
            sigma_sq: Current belief variance
            observation: Observed value
            obs_noise_sq: Observation noise variance
            prior_mu: Prior mean
            prior_sigma_sq: Prior variance
            dt: Time step

        Returns:
            new_mu: Updated belief mean
            new_sigma_sq: Updated belief variance
            info: Dictionary with diagnostic information
        """
        pass

    @abstractmethod
    def reset(self):
        """Reset any internal state (e.g., momentum)."""
        pass


# =============================================================================
# Gradient Descent Dynamics
# =============================================================================

@dataclass
class GradientDynamics(BeliefDynamics):
    """
    Gradient descent on free energy.

    The belief updates according to:
        Δμ = -lr * ∂F/∂μ

    This is the "standard" model prediction - learning rate proportional
    to prediction error, scaled by relative precision.
    """
    learning_rate: float = 0.3  # Learning rate for μ updates
    sigma_learning_rate: float = 0.1  # Learning rate for σ² updates
    min_sigma_sq: float = 1.0  # Minimum variance (prevents collapse)

    def update(self,
               mu: float,
               sigma_sq: float,
               observation: float,
               obs_noise_sq: float,
               prior_mu: float,
               prior_sigma_sq: float,
               dt: float = 1.0) -> Tuple[float, float, dict]:
        """Gradient descent update."""

        # Prediction error
        delta = observation - mu

        # Precision-weighted prediction error (natural gradient on μ)
        # From F = 0.5 * (μ - o)²/σ_obs² + 0.5 * (μ - μ_prior)²/σ_prior²
        # ∂F/∂μ = (μ - o)/σ_obs² + (μ - μ_prior)/σ_prior²

        obs_precision = 1.0 / obs_noise_sq
        prior_precision = 1.0 / prior_sigma_sq

        # Total gradient
        grad_mu = -delta * obs_precision + (mu - prior_mu) * prior_precision

        # Update μ
        new_mu = mu - self.learning_rate * dt * grad_mu

        # Update σ² (shrink toward observation noise, grow toward prior)
        # Simplified: interpolate toward posterior variance
        posterior_precision = obs_precision + prior_precision
        posterior_sigma_sq = 1.0 / posterior_precision

        new_sigma_sq = sigma_sq + self.sigma_learning_rate * dt * (posterior_sigma_sq - sigma_sq)
        new_sigma_sq = max(new_sigma_sq, self.min_sigma_sq)

        # Compute effective learning rate for comparison
        effective_lr = (new_mu - mu) / delta if abs(delta) > 1e-6 else 0.0

        info = {
            'delta': delta,
            'grad_mu': grad_mu,
            'effective_lr': effective_lr,
            'momentum': 0.0,  # No momentum in gradient descent
        }

        return new_mu, new_sigma_sq, info

    def reset(self):
        """No internal state to reset."""
        pass


# =============================================================================
# Hamiltonian Dynamics
# =============================================================================

@dataclass
class HamiltonianDynamics(BeliefDynamics):
    """
    Hamiltonian dynamics on belief space.

    The belief has MOMENTUM, updated according to:
        dμ/dt = p/m           (velocity from momentum)
        dp/dt = -∂F/∂μ - γp   (force with friction)

    Key predictions:
    1. Beliefs have inertia - they don't instantly track observations
    2. After sudden changes, beliefs may overshoot before settling
    3. Smooth trajectories rather than noisy jumps
    4. Effective learning rate depends on momentum state
    """
    mass: float = 1.0           # Inertia of belief (higher = slower response)
    friction: float = 0.5       # Damping coefficient (γ)
    dt: float = 1.0             # Integration time step
    sigma_learning_rate: float = 0.1
    min_sigma_sq: float = 1.0

    # Internal state
    momentum: float = field(default=0.0, init=False)

    def update(self,
               mu: float,
               sigma_sq: float,
               observation: float,
               obs_noise_sq: float,
               prior_mu: float,
               prior_sigma_sq: float,
               dt: float = None) -> Tuple[float, float, dict]:
        """Hamiltonian (leapfrog) update."""

        if dt is None:
            dt = self.dt

        # Prediction error
        delta = observation - mu

        # Compute force (negative gradient of potential energy)
        obs_precision = 1.0 / obs_noise_sq
        prior_precision = 1.0 / prior_sigma_sq

        # Force = -∂V/∂μ where V = F (free energy)
        force = delta * obs_precision - (mu - prior_mu) * prior_precision

        # Leapfrog integration (symplectic)
        # Half step momentum
        p_half = self.momentum + 0.5 * dt * (force - self.friction * self.momentum)

        # Full step position
        new_mu = mu + dt * p_half / self.mass

        # Recompute force at new position
        new_delta = observation - new_mu
        new_force = new_delta * obs_precision - (new_mu - prior_mu) * prior_precision

        # Half step momentum again
        new_momentum = p_half + 0.5 * dt * (new_force - self.friction * p_half)

        self.momentum = new_momentum

        # Update σ² (same as gradient)
        posterior_precision = obs_precision + prior_precision
        posterior_sigma_sq = 1.0 / posterior_precision
        new_sigma_sq = sigma_sq + self.sigma_learning_rate * dt * (posterior_sigma_sq - sigma_sq)
        new_sigma_sq = max(new_sigma_sq, self.min_sigma_sq)

        # Compute effective learning rate
        effective_lr = (new_mu - mu) / delta if abs(delta) > 1e-6 else 0.0

        # Kinetic energy
        kinetic = 0.5 * self.momentum**2 / self.mass

        info = {
            'delta': delta,
            'force': force,
            'momentum': self.momentum,
            'kinetic_energy': kinetic,
            'effective_lr': effective_lr,
            'velocity': self.momentum / self.mass,
        }

        return new_mu, new_sigma_sq, info

    def reset(self):
        """Reset momentum to zero."""
        self.momentum = 0.0

    def set_momentum(self, momentum: float):
        """Set momentum directly (for testing)."""
        self.momentum = momentum


# =============================================================================
# 1D Belief Agent
# =============================================================================

@dataclass
class AgentState:
    """Snapshot of agent state at one time point."""
    trial: int
    mu: float
    sigma_sq: float
    observation: float
    prediction_error: float
    update: float
    effective_lr: float
    momentum: float = 0.0
    kinetic_energy: float = 0.0


@dataclass
class BeliefAgent1D:
    """
    1D Belief agent for tracking helicopter position.

    Can use either gradient or Hamiltonian dynamics.
    """
    dynamics: BeliefDynamics
    initial_mu: float = 150.0       # Start at midpoint [0, 300]
    initial_sigma_sq: float = 100.0  # Initial uncertainty
    obs_noise_sq: float = 625.0      # Observation noise (25² for high noise condition)
    prior_sigma_sq: float = 1000.0   # Prior uncertainty (weak prior)

    # Current state
    mu: float = field(init=False)
    sigma_sq: float = field(init=False)

    # History
    history: List[AgentState] = field(default_factory=list, init=False)

    def __post_init__(self):
        self.reset()

    def reset(self):
        """Reset agent to initial state."""
        self.mu = self.initial_mu
        self.sigma_sq = self.initial_sigma_sq
        self.dynamics.reset()
        self.history = []

    def observe(self, observation: float, trial: int = 0) -> AgentState:
        """
        Process one observation and update belief.

        Args:
            observation: Observed helicopter position
            trial: Trial number (for logging)

        Returns:
            AgentState with updated belief and diagnostics
        """
        old_mu = self.mu

        # Update belief using dynamics
        new_mu, new_sigma_sq, info = self.dynamics.update(
            mu=self.mu,
            sigma_sq=self.sigma_sq,
            observation=observation,
            obs_noise_sq=self.obs_noise_sq,
            prior_mu=self.mu,  # Use current belief as prior (tracking)
            prior_sigma_sq=self.prior_sigma_sq,
        )

        # Record state
        state = AgentState(
            trial=trial,
            mu=new_mu,
            sigma_sq=new_sigma_sq,
            observation=observation,
            prediction_error=info['delta'],
            update=new_mu - old_mu,
            effective_lr=info['effective_lr'],
            momentum=info.get('momentum', 0.0),
            kinetic_energy=info.get('kinetic_energy', 0.0),
        )

        self.history.append(state)

        # Update current state
        self.mu = new_mu
        self.sigma_sq = new_sigma_sq

        return state

    def run_trial_sequence(self, observations: np.ndarray) -> List[AgentState]:
        """Run agent through a sequence of observations."""
        self.reset()
        for i, obs in enumerate(observations):
            self.observe(obs, trial=i+1)
        return self.history

    def get_predictions(self) -> np.ndarray:
        """Get array of predictions (beliefs) across trials."""
        return np.array([s.mu for s in self.history])

    def get_updates(self) -> np.ndarray:
        """Get array of belief updates across trials."""
        return np.array([s.update for s in self.history])

    def get_learning_rates(self) -> np.ndarray:
        """Get array of effective learning rates across trials."""
        return np.array([s.effective_lr for s in self.history])

    def get_momenta(self) -> np.ndarray:
        """Get array of momenta (Hamiltonian only) across trials."""
        return np.array([s.momentum for s in self.history])


# =============================================================================
# Factory Functions
# =============================================================================

def create_gradient_agent(learning_rate: float = 0.3,
                          obs_noise: float = 25.0) -> BeliefAgent1D:
    """Create agent with gradient descent dynamics."""
    dynamics = GradientDynamics(learning_rate=learning_rate)
    return BeliefAgent1D(
        dynamics=dynamics,
        obs_noise_sq=obs_noise**2,
    )


def create_hamiltonian_agent(mass: float = 1.0,
                             friction: float = 0.5,
                             obs_noise: float = 25.0) -> BeliefAgent1D:
    """Create agent with Hamiltonian dynamics."""
    dynamics = HamiltonianDynamics(mass=mass, friction=friction)
    return BeliefAgent1D(
        dynamics=dynamics,
        obs_noise_sq=obs_noise**2,
    )


# =============================================================================
# Main (for testing)
# =============================================================================

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # Generate synthetic data with a changepoint
    np.random.seed(42)
    n_trials = 100
    true_means = np.concatenate([
        np.full(50, 100),
        np.full(50, 200)  # Changepoint at trial 50
    ])
    observations = true_means + np.random.randn(n_trials) * 25

    # Create agents
    grad_agent = create_gradient_agent(learning_rate=0.3)
    ham_agent = create_hamiltonian_agent(mass=2.0, friction=0.3)

    # Run both agents
    grad_agent.run_trial_sequence(observations)
    ham_agent.run_trial_sequence(observations)

    # Plot
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    trials = np.arange(1, n_trials + 1)

    # Beliefs
    axes[0].plot(trials, observations, 'o', alpha=0.3, label='Observations')
    axes[0].plot(trials, true_means, 'k--', label='True mean')
    axes[0].plot(trials, grad_agent.get_predictions(), label='Gradient')
    axes[0].plot(trials, ham_agent.get_predictions(), label='Hamiltonian')
    axes[0].axvline(50, color='red', linestyle=':', label='Changepoint')
    axes[0].set_ylabel('Position')
    axes[0].legend()
    axes[0].set_title('Belief Tracking')

    # Learning rates
    axes[1].plot(trials, grad_agent.get_learning_rates(), label='Gradient')
    axes[1].plot(trials, ham_agent.get_learning_rates(), label='Hamiltonian')
    axes[1].axvline(50, color='red', linestyle=':')
    axes[1].set_ylabel('Learning Rate')
    axes[1].legend()

    # Momentum (Hamiltonian only)
    axes[2].plot(trials, ham_agent.get_momenta(), label='Momentum')
    axes[2].axvline(50, color='red', linestyle=':')
    axes[2].axhline(0, color='k', linestyle='-', alpha=0.3)
    axes[2].set_ylabel('Momentum')
    axes[2].set_xlabel('Trial')
    axes[2].legend()

    plt.tight_layout()
    plt.savefig('belief_dynamics_demo.png', dpi=150)
    print("Saved belief_dynamics_demo.png")
