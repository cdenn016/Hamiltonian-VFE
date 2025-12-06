#!/usr/bin/env python3
"""
Trajectory Metrics: Time-Evolving Pullback Geometry
===================================================

Computes pullback metrics at multiple time points along belief trajectories,
enabling analysis of how geometric structure evolves during training.

Key Functions:
--------------
- compute_trajectory_pullback_metrics(): Compute G(t) from snapshots
- analyze_signature_evolution(): Track metric signature changes over time
- detect_lorentzian_transitions(): Find Riemannian → Lorentzian changes

Mathematical Background:
-----------------------
At each time t, agents maintain belief distributions q(c,t) = N(μ(c,t), Σ(c,t)).
The pullback metric at time t is:

    G(c, t) = (∂_μ μ)^T Σ^{-1} (∂_ν μ) + (1/2)tr(Σ^{-1}(∂_μ Σ)Σ^{-1}(∂_ν Σ))

By computing G at multiple times from training snapshots, we can track:
1. Eigenvalue evolution λ_i(t)
2. Signature transitions (e.g., (+,+,+) → (-,+,+))
3. Correlation with dynamics (energy, momentum, etc.)

Author: Chris & Claude
Date: November 2025
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass

from geometry.pullback_metrics import (
    pullback_metric_gaussian,
    InducedMetric
)
from geometry.signature_analysis import (
    analyze_metric_signature,
    MetricSignature,
    SignatureAnalysis
)


@dataclass
class TrajectoryMetrics:
    """
    Time-evolving pullback metrics along a belief trajectory.

    Attributes:
        steps: Training steps where metrics were computed
        metrics: List of InducedMetric objects, one per step
        signature_analyses: List of SignatureAnalysis objects
        agent_idx: Which agent
        point_idx: Which spatial point (if grid)
    """
    steps: List[int]
    metrics: List[InducedMetric]
    signature_analyses: List[SignatureAnalysis]
    agent_idx: int
    point_idx: int = 0

    def get_eigenvalue_trajectories(self) -> np.ndarray:
        """
        Get eigenvalue evolution over time.

        Returns:
            eigenvalues: (T, d) array where T = number of snapshots,
                        d = spatial dimensionality of base manifold
        """
        # Ensure all metrics have computed eigenvalues
        for metric in self.metrics:
            if metric.eigenvalues is None:
                metric.compute_spectral_decomposition()

        # Extract eigenvalues at the specified point
        eigenvalues = []
        for metric in self.metrics:
            if metric.G.ndim == 2:
                # Single point (0D agent)
                eigs = metric.eigenvalues
            else:
                # Field agent - extract at point_idx
                eigs = metric.eigenvalues[self.point_idx]

            eigenvalues.append(eigs)

        return np.array(eigenvalues)

    def get_signature_sequence(self) -> List[str]:
        """
        Get sequence of metric signatures over time.

        Returns:
            List of signature strings (e.g., ["riemannian", "lorentzian", ...])
        """
        return [sig.signature.value for sig in self.signature_analyses]

    def detect_signature_transitions(self) -> List[Tuple[int, str, str]]:
        """
        Detect where metric signature changes.

        Returns:
            List of (step, old_signature, new_signature) tuples
        """
        transitions = []
        signatures = self.get_signature_sequence()

        for i in range(1, len(signatures)):
            if signatures[i] != signatures[i-1]:
                transitions.append((
                    self.steps[i],
                    signatures[i-1],
                    signatures[i]
                ))

        return transitions

    def get_lorentzian_fraction(self) -> float:
        """
        Get fraction of time spent in Lorentzian regime.

        Returns:
            Fraction in [0, 1]
        """
        signatures = self.get_signature_sequence()
        n_lorentzian = sum(1 for sig in signatures if sig == "lorentzian")
        return n_lorentzian / len(signatures) if signatures else 0.0


def reconstruct_agent_state_from_snapshot(
    snapshot_data: Dict[str, Any],
    agent_idx: int
) -> Dict[str, np.ndarray]:
    """
    Extract agent state from snapshot.

    Args:
        snapshot_data: Snapshot dictionary from history
        agent_idx: Which agent to extract

    Returns:
        Dictionary with 'mu_q', 'L_q', 'mu_p', 'L_p', 'Sigma_q', 'Sigma_p'
    """
    agent_data = snapshot_data['agents'][agent_idx]

    # Extract fields
    mu_q = agent_data['mu_q']
    mu_p = agent_data['mu_p']

    # GAUGE-COVARIANT: Prefer Sigma directly (new format)
    # Fall back to computing from L (old format, backwards compat)
    if 'Sigma_q' in agent_data:
        Sigma_q = agent_data['Sigma_q']
        Sigma_p = agent_data['Sigma_p']
        # Compute L on-demand if needed (for backwards compat)
        L_q = agent_data.get('L_q', np.linalg.cholesky(Sigma_q))
        L_p = agent_data.get('L_p', np.linalg.cholesky(Sigma_p))
    else:
        # Old format: compute Sigma from L
        L_q = agent_data['L_q']
        L_p = agent_data['L_p']
        Sigma_q = np.einsum('...ij,...kj->...ik', L_q, L_q, optimize=True)
        Sigma_p = np.einsum('...ij,...kj->...ik', L_p, L_p, optimize=True)

    return {
        'mu_q': mu_q,
        'L_q': L_q,
        'mu_p': mu_p,
        'L_p': L_p,
        'Sigma_q': Sigma_q,
        'Sigma_p': Sigma_p,
        'phi': agent_data.get('phi', None)
    }


def compute_trajectory_pullback_metrics(
    history,
    agent_idx: int,
    point_idx: int = 0,
    dx: float = 1.0,
    metric_type: str = "belief",
    periodic: bool = True,
    eps: float = 1e-8
) -> TrajectoryMetrics:
    """
    Compute pullback metrics at each snapshot along trajectory.

    Args:
        history: TrainingHistory or HamiltonianHistory with agent_snapshots
        agent_idx: Which agent to analyze
        point_idx: Which spatial point (for field agents)
        dx: Grid spacing for spatial gradients
        metric_type: "belief" or "prior"
        periodic: Use periodic boundary conditions
        eps: Regularization

    Returns:
        TrajectoryMetrics object with time-evolving metrics

    Example:
        >>> config = TrainingConfig(save_snapshots=True, snapshot_every=10)
        >>> trainer = Trainer(system, config)
        >>> history = trainer.train()
        >>>
        >>> traj_metrics = compute_trajectory_pullback_metrics(
        ...     history, agent_idx=0, point_idx=0
        ... )
        >>>
        >>> # Get eigenvalue evolution
        >>> eigenvalues = traj_metrics.get_eigenvalue_trajectories()
        >>> plt.plot(traj_metrics.steps, eigenvalues)
    """
    if not hasattr(history, 'agent_snapshots') or len(history.agent_snapshots) == 0:
        raise ValueError(
            "History must have agent_snapshots. "
            "Enable with TrainingConfig(save_snapshots=True)"
        )

    # Lists to store results
    metrics = []
    signature_analyses = []
    steps = history.snapshot_steps

    print(f"\nComputing trajectory metrics...")
    print(f"  Agent: {agent_idx}")
    print(f"  Point: {point_idx}")
    print(f"  Snapshots: {len(steps)}")
    print(f"  Metric type: {metric_type}")

    for i, snapshot in enumerate(history.agent_snapshots):
        step = snapshot['step']

        # Reconstruct agent state
        agent_state = reconstruct_agent_state_from_snapshot(snapshot, agent_idx)

        # Choose field based on metric type
        if metric_type == "belief":
            mu_field = agent_state['mu_q']
            Sigma_field = agent_state['Sigma_q']
        elif metric_type == "prior":
            mu_field = agent_state['mu_p']
            Sigma_field = agent_state['Sigma_p']
        else:
            raise ValueError(f"metric_type must be 'belief' or 'prior', got {metric_type}")

        # Check if this is a 0D agent (particle)
        if mu_field.ndim == 1:
            # 0D particle - no spatial gradients, can't compute pullback
            # metric in the usual sense. Skip for now.
            print(f"  Warning: Agent {agent_idx} is 0D particle, skipping pullback metric")
            continue

        # Compute pullback metric
        try:
            induced_metric = pullback_metric_gaussian(
                mu_field=mu_field,
                Sigma_field=Sigma_field,
                dx=dx,
                metric_type=metric_type,
                periodic=periodic,
                eps=eps
            )

            # Analyze signature
            # Extract metric at specified point
            # Handle point_idx as either int (1D) or tuple (2D/3D)
            if induced_metric.G.ndim == 3:
                # 1D field: (n_points, 1, 1)
                # point_idx should be int
                if isinstance(point_idx, tuple):
                    point_idx = point_idx[0]
                G_at_point = induced_metric.G[point_idx]
            elif induced_metric.G.ndim == 4:
                # 2D field: (H, W, 2, 2)
                if isinstance(point_idx, tuple):
                    # point_idx is (x, y)
                    x, y = point_idx
                else:
                    # point_idx is flat index - convert to 2D
                    H, W = induced_metric.spatial_shape
                    x = point_idx // W
                    y = point_idx % W
                G_at_point = induced_metric.G[x, y]
            elif induced_metric.G.ndim == 5:
                # 3D field: (H, W, D, 3, 3)
                if isinstance(point_idx, tuple):
                    # point_idx is (x, y, z)
                    G_at_point = induced_metric.G[point_idx]
                else:
                    # point_idx is flat index - convert to 3D
                    H, W, D = induced_metric.spatial_shape
                    x = point_idx // (W * D)
                    y = (point_idx // D) % W
                    z = point_idx % D
                    G_at_point = induced_metric.G[x, y, z]
            else:
                raise ValueError(f"Unexpected metric shape: {induced_metric.G.shape}")

            sig_analysis = analyze_metric_signature(G_at_point)

            metrics.append(induced_metric)
            signature_analyses.append(sig_analysis)

            # Progress update
            if (i + 1) % max(1, len(steps) // 10) == 0:
                progress = 100 * (i + 1) / len(steps)
                print(f"  Progress: {progress:.0f}% ({i+1}/{len(steps)})")

        except Exception as e:
            print(f"  Warning: Failed to compute metric at step {step}: {e}")
            continue

    print(f"✓ Computed {len(metrics)} pullback metrics")

    # Check if any metrics were computed
    if len(metrics) == 0:
        raise ValueError(
            f"Failed to compute any pullback metrics for agent {agent_idx} at point {point_idx}. "
            "This likely indicates a dimension mismatch or other geometric issue. "
            "Check that point_idx matches the spatial dimensionality."
        )

    # Create trajectory metrics object
    traj_metrics = TrajectoryMetrics(
        steps=steps[:len(metrics)],  # Trim to successful computations
        metrics=metrics,
        signature_analyses=signature_analyses,
        agent_idx=agent_idx,
        point_idx=point_idx
    )

    return traj_metrics


def analyze_signature_evolution(
    traj_metrics: TrajectoryMetrics
) -> Dict[str, Any]:
    """
    Analyze how metric signature evolves over time.

    Args:
        traj_metrics: TrajectoryMetrics object

    Returns:
        Dictionary with:
            - signature_sequence: List of signature strings
            - transitions: List of (step, old_sig, new_sig)
            - eigenvalue_trajectories: (T, d) array
            - lorentzian_fraction: Fraction of time in Lorentzian regime
            - riemannian_fraction: Fraction of time in Riemannian regime
            - n_transitions: Number of signature changes
            - dominant_signature: Most common signature

    Example:
        >>> analysis = analyze_signature_evolution(traj_metrics)
        >>> print(f"Lorentzian fraction: {analysis['lorentzian_fraction']:.2%}")
        >>> print(f"Transitions: {analysis['transitions']}")
    """
    # Get basic info
    signatures = traj_metrics.get_signature_sequence()
    transitions = traj_metrics.detect_signature_transitions()
    eigenvalues = traj_metrics.get_eigenvalue_trajectories()

    # Compute signature fractions
    from collections import Counter
    sig_counts = Counter(signatures)

    # Handle empty signature case
    if len(signatures) == 0:
        lorentzian_frac = 0.0
        riemannian_frac = 0.0
        dominant_sig = None
    else:
        lorentzian_frac = sig_counts.get("lorentzian", 0) / len(signatures)
        riemannian_frac = sig_counts.get("riemannian", 0) / len(signatures)
        # Find dominant signature
        dominant_sig = max(sig_counts.items(), key=lambda x: x[1])[0] if sig_counts else None

    analysis = {
        'signature_sequence': signatures,
        'transitions': transitions,
        'eigenvalue_trajectories': eigenvalues,
        'lorentzian_fraction': lorentzian_frac,
        'riemannian_fraction': riemannian_frac,
        'n_transitions': len(transitions),
        'dominant_signature': dominant_sig,
        'signature_counts': dict(sig_counts)
    }

    return analysis


def detect_lorentzian_onset(
    traj_metrics: TrajectoryMetrics,
    min_duration: int = 5
) -> Optional[int]:
    """
    Detect when system first enters Lorentzian regime (and stays there).

    Args:
        traj_metrics: TrajectoryMetrics object
        min_duration: Minimum consecutive steps in Lorentzian to count

    Returns:
        Step number of Lorentzian onset, or None if never sustained
    """
    signatures = traj_metrics.get_signature_sequence()
    steps = traj_metrics.steps

    # Find first sustained Lorentzian period
    lorentzian_count = 0
    onset_idx = None

    for i, sig in enumerate(signatures):
        if sig == "lorentzian":
            if lorentzian_count == 0:
                onset_idx = i  # Potential onset
            lorentzian_count += 1

            if lorentzian_count >= min_duration:
                return steps[onset_idx]  # Sustained Lorentzian!
        else:
            lorentzian_count = 0
            onset_idx = None

    return None


def print_signature_summary(traj_metrics: TrajectoryMetrics):
    """Print human-readable summary of signature evolution."""

    analysis = analyze_signature_evolution(traj_metrics)

    print("\n" + "="*70)
    print("METRIC SIGNATURE EVOLUTION SUMMARY")
    print("="*70)
    print(f"Agent: {traj_metrics.agent_idx}")
    print(f"Point: {traj_metrics.point_idx}")
    print(f"Snapshots: {len(traj_metrics.steps)}")
    print()

    print("Signature Distribution:")
    for sig, count in analysis['signature_counts'].items():
        frac = count / len(traj_metrics.steps)
        print(f"  {sig:15s}: {count:4d} ({frac:6.1%})")
    print()

    print(f"Dominant signature: {analysis['dominant_signature']}")
    print(f"Number of transitions: {analysis['n_transitions']}")
    print()

    if analysis['transitions']:
        print("Signature Transitions:")
        for step, old_sig, new_sig in analysis['transitions']:
            print(f"  Step {step:4d}: {old_sig:15s} → {new_sig:15s}")
    else:
        print("No signature transitions detected")

    print()
    print("Lorentzian Analysis:")
    onset = detect_lorentzian_onset(traj_metrics)
    if onset is not None:
        print(f"  Lorentzian onset: Step {onset}")
    else:
        print(f"  No sustained Lorentzian regime")
    print(f"  Lorentzian fraction: {analysis['lorentzian_fraction']:.1%}")

    print("="*70)