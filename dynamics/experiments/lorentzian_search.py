#!/usr/bin/env python3
"""
Lorentzian Structure Search
============================

Systematic parameter space search for conditions that produce Lorentzian
pullback metric signature in variational free energy minimization.

Research Question:
-----------------
Under what conditions does the Fisher-Rao pullback metric transition from
Riemannian (all positive eigenvalues) to Lorentzian (one negative eigenvalue)?

Hypotheses to test:
------------------
1. Competing forces (belief align vs prior align with opposite signs)
2. Strong gauge dynamics (large lambda_phi)
3. Underdamped Hamiltonian dynamics (low friction)
4. Non-equilibrium/driven systems
5. High curvature regions (sharp gradients)
6. 2D vs 1D (dimensionality effects)

Author: Chris & Claude
Date: November 23, 2025
"""

import numpy as np
from pathlib import Path
import json
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
import time
from collections import Counter

from agent.system import MultiAgentSystem
from agent.agents import Agent
from agent.trainer import Trainer
from agent.hamiltonian_trainer import HamiltonianTrainer
from config import SystemConfig, AgentConfig, TrainingConfig
from dynamics.trajectory_metrics import (
    compute_trajectory_pullback_metrics,
    analyze_signature_evolution
)


@dataclass
class SearchConfig:
    """Configuration for one parameter space point."""
    name: str
    description: str
    
    # Spatial configuration
    spatial_shape: tuple
    K: int
    
    # System parameters
    lambda_self: float = 1.0
    lambda_belief_align: float = 1.0
    lambda_prior_align: float = 1.0
    lambda_phi: float = 1.0
    
    # Dynamics type
    use_hamiltonian: bool = False
    friction: float = 0.1  # For Hamiltonian only
    
    # Training
    n_steps: int = 200
    lr_mu_q: float = 0.1
    lr_sigma_q: float = 0.05
    lr_phi: float = 0.1
    
    # Snapshots
    snapshot_every: int = 1


@dataclass
class SearchResult:
    """Results from one configuration."""
    config_name: str
    lorentzian_found: bool
    lorentzian_fraction: float
    dominant_signature: str
    n_transitions: int
    
    # Detailed metrics
    max_negative_eigenvalue: Optional[float]
    steps_lorentzian: List[int]
    points_lorentzian: List[tuple]
    
    # Execution time
    duration_seconds: float


def create_search_configs() -> List[SearchConfig]:
    """Define parameter space grid to search."""
    
    configs = []
    
    # === BASELINE ===
    configs.append(SearchConfig(
        name="baseline_1d",
        description="Baseline: 1D, simple self-energy only",
        spatial_shape=(16,),
        K=7,
        lambda_self=1.0,
        lambda_belief_align=0.0
    ))
    
    configs.append(SearchConfig(
        name="baseline_2d",
        description="Baseline: 2D, simple self-energy only",
        spatial_shape=(8, 8),
        K=7,
        lambda_self=1.0,
        lambda_belief_align=0.0
    ))

    # === HYPOTHESIS 1: Strong Gauge Dynamics ===
    configs.append(SearchConfig(
        name="gauge_weak",
        description="Weak gauge field dynamics",
        spatial_shape=(8, 8),
        K=7,
        lambda_self=1.0,
        lambda_phi=0.5,
        lr_phi=0.02
    ))
    
    configs.append(SearchConfig(
        name="gauge_strong",
        description="Strong gauge field dynamics",
        spatial_shape=(8, 8),
        K=7,
        lambda_self=1.0,
        lambda_phi=2.0,
        lr_phi=0.1
    ))
    
    configs.append(SearchConfig(
        name="gauge_extreme",
        description="Extreme gauge field dynamics",
        spatial_shape=(8, 8),
        K=7,
        lambda_self=0.5,
        lambda_phi=5.0,
        lr_phi=0.1
    ))
    
    # === HYPOTHESIS 3: Hamiltonian (Underdamped) ===
    configs.append(SearchConfig(
        name="hamiltonian_light_damping",
        description="Hamiltonian with light damping",
        spatial_shape=(8, 8),
        K=5,
        lambda_self=1.0,
        lambda_belief_align=1.0,
        use_hamiltonian=True,
        friction=0.05  # Very light damping
    ))
    
    configs.append(SearchConfig(
        name="hamiltonian_underdamped",
        description="Hamiltonian underdamped (conservative-ish)",
        spatial_shape=(8, 8),
        K=5,
        lambda_self=1.0,
        lambda_belief_align=1.0,
        use_hamiltonian=True,
        friction=0.01  # Extremely light damping
    ))
    
    configs.append(SearchConfig(
        name="hamiltonian_with_gauge",
        description="Hamiltonian + strong gauge",
        spatial_shape=(8, 8),
        K=5,
        lambda_self=1.0,
        lambda_phi=2.0,
        use_hamiltonian=True,
        friction=0.02,
        lr_phi=0.1
    ))
    
    # === HYPOTHESIS 4: High Curvature (Sharp Gradients) ===
    configs.append(SearchConfig(
        name="sharp_covariances",
        description="Sharp/peaked covariances (high curvature)",
        spatial_shape=(8, 8),
        K=5,
        lambda_self=1.0,
        lambda_belief_align=1.0,
        lr_sigma_q=0.005  # Slower, sharper changes
    ))

    return configs


def run_single_config(config: SearchConfig, results_dir: Path) -> SearchResult:
    """
    Run training with one configuration and check for Lorentzian structure.
    
    Returns SearchResult with findings.
    """
    print(f"\n{'='*70}")
    print(f"CONFIG: {config.name}")
    print(f"{'='*70}")
    print(f"Description: {config.description}")
    print(f"Spatial: {config.spatial_shape}, K={config.K}")
    print(f"Dynamics: {'Hamiltonian' if config.use_hamiltonian else 'Gradient Flow'}")
    print()
    
    start_time = time.time()
    
    # Create agent
    agent_config = AgentConfig(
        spatial_shape=config.spatial_shape,
        K=config.K
    )
    
    system_config = SystemConfig(
        lambda_self=config.lambda_self,
        lambda_belief_align=config.lambda_belief_align,
        lambda_prior_align=config.lambda_prior_align,
        lambda_phi=config.lambda_phi
    )
    
    rng = np.random.default_rng(42)
    agent = Agent(0, agent_config, rng=rng)
    system = MultiAgentSystem([agent], system_config)
    
    # Train
    if config.use_hamiltonian:
        train_config = TrainingConfig(
            n_steps=config.n_steps,
            lr_mu_q=0.1,  # Not used for Hamiltonian, but required
            lr_sigma_q=0.05,
            save_snapshots=True,
            snapshot_every=config.snapshot_every,
            log_every=25
        )
        trainer = HamiltonianTrainer(system, train_config, friction=config.friction)
    else:
        train_config = TrainingConfig(
            n_steps=config.n_steps,
            lr_mu_q=config.lr_mu_q,
            lr_sigma_q=config.lr_sigma_q,
            lr_phi=config.lr_phi,
            save_snapshots=True,
            snapshot_every=config.snapshot_every,
            log_every=25
        )
        trainer = Trainer(system, train_config)
    
    print("Training...")
    history = trainer.train()
    
    print()
    print("Analyzing for Lorentzian structure...")
    
    # Check multiple spatial points for Lorentzian
    lorentzian_found = False
    lorentzian_fraction_overall = 0.0
    steps_lorentzian = []
    points_lorentzian = []
    max_neg_eigenvalue = None
    all_signatures = []
    
    # Sample points to check
    if len(config.spatial_shape) == 1:
        # 1D: check beginning, middle, end
        points_to_check = [0, config.spatial_shape[0] // 2, config.spatial_shape[0] - 1]
    else:
        # 2D: check corners and center
        H, W = config.spatial_shape
        points_to_check = [
            (0, 0), (0, W-1), (H-1, 0), (H-1, W-1),  # Corners
            (H//2, W//2)  # Center
        ]
    
    for point in points_to_check:
        try:
            traj_metrics = compute_trajectory_pullback_metrics(
                history,
                agent_idx=0,
                point_idx=point,
                metric_type="belief"
            )
            
            sig_analysis = analyze_signature_evolution(traj_metrics)
            all_signatures.append(sig_analysis)
            
            if sig_analysis['lorentzian_fraction'] > 0:
                lorentzian_found = True
                points_lorentzian.append(point)
                
                # Find which steps were Lorentzian
                for i, sig in enumerate(sig_analysis['signature_sequence']):
                    if sig == "lorentzian":
                        steps_lorentzian.append(traj_metrics.steps[i])
                
                # Track most negative eigenvalue
                eigenvalues = sig_analysis['eigenvalue_trajectories']
                min_eig = np.min(eigenvalues)
                if max_neg_eigenvalue is None or min_eig < max_neg_eigenvalue:
                    max_neg_eigenvalue = float(min_eig)
            
            lorentzian_fraction_overall += sig_analysis['lorentzian_fraction']
            
        except Exception as e:
            print(f"  Warning: Failed to analyze point {point}: {e}")
            continue
    
    # Average over all points
    lorentzian_fraction_overall /= len(points_to_check)
    
    # Get dominant signature (most common across all points)
    all_sigs_flat = []
    for analysis in all_signatures:
        all_sigs_flat.extend(analysis['signature_sequence'])
    
    from collections import Counter
    if all_sigs_flat:
        dominant_sig = Counter(all_sigs_flat).most_common(1)[0][0]
    else:
        dominant_sig = "unknown"
    
    # Count transitions
    n_transitions = sum(len(a['transitions']) for a in all_signatures)
    
    duration = time.time() - start_time
    
    result = SearchResult(
        config_name=config.name,
        lorentzian_found=lorentzian_found,
        lorentzian_fraction=lorentzian_fraction_overall,
        dominant_signature=dominant_sig,
        n_transitions=n_transitions,
        max_negative_eigenvalue=max_neg_eigenvalue,
        steps_lorentzian=list(set(steps_lorentzian)),
        points_lorentzian=points_lorentzian,
        duration_seconds=duration
    )
    
    # Print summary
    print()
    print(f"{'='*70}")
    print(f"RESULT: {config.name}")
    print(f"{'='*70}")
    if lorentzian_found:
        print(f"✨ LORENTZIAN FOUND! ✨")
        print(f"  Fraction: {lorentzian_fraction_overall:.1%}")
        print(f"  Points: {points_lorentzian}")
        print(f"  Steps: {sorted(steps_lorentzian)[:10]}{'...' if len(steps_lorentzian) > 10 else ''}")
        print(f"  Max negative eigenvalue: {max_neg_eigenvalue:.6f}")
    else:
        print(f"✗ No Lorentzian structure")
        print(f"  Dominant: {dominant_sig}")
    print(f"  Transitions: {n_transitions}")
    print(f"  Duration: {duration:.1f}s")
    print(f"{'='*70}")
    
    # Save result
    result_file = results_dir / f"{config.name}_result.json"
    with open(result_file, 'w') as f:
        json.dump(asdict(result), f, indent=2)
    
    return result


def run_search(output_dir: Optional[Path] = None):
    """
    Run complete parameter space search.
    
    Tests all configurations and reports findings.
    """
    if output_dir is None:
        output_dir = Path("_results/lorentzian_search")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70)
    print("LORENTZIAN STRUCTURE SEARCH")
    print("="*70)
    print(f"Output directory: {output_dir}")
    print()
    
    configs = create_search_configs()
    print(f"Configurations to test: {len(configs)}")
    for i, cfg in enumerate(configs, 1):
        print(f"  {i}. {cfg.name}: {cfg.description}")
    print()
    
    results = []
    lorentzian_configs = []
    
    for i, config in enumerate(configs, 1):
        print(f"\n\n[{i}/{len(configs)}] Testing: {config.name}")
        
        try:
            result = run_single_config(config, output_dir)
            results.append(result)
            
            if result.lorentzian_found:
                lorentzian_configs.append(config.name)
        
        except Exception as e:
            print(f"\n❌ ERROR in {config.name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Final summary
    print("\n\n" + "="*70)
    print("SEARCH COMPLETE")
    print("="*70)
    print(f"Configurations tested: {len(results)}/{len(configs)}")
    print(f"Lorentzian found in: {len(lorentzian_configs)} configurations")
    print()
    
    if lorentzian_configs:
        print("✨ LORENTZIAN CONFIGURATIONS:")
        for name in lorentzian_configs:
            result = next(r for r in results if r.config_name == name)
            print(f"  • {name}: {result.lorentzian_fraction:.1%} Lorentzian")
            print(f"      Points: {result.points_lorentzian}")
    else:
        print("✗ No Lorentzian structure found in any configuration")
        print()
        print("Dominant signatures across all configs:")
        sig_counts = Counter(r.dominant_signature for r in results)
        for sig, count in sig_counts.most_common():
            print(f"  {sig}: {count} configs")
    
    # Save summary
    summary_file = output_dir / "search_summary.json"
    summary = {
        'n_configs': len(configs),
        'n_tested': len(results),
        'n_lorentzian': len(lorentzian_configs),
        'lorentzian_configs': lorentzian_configs,
        'results': [asdict(r) for r in results]
    }
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nResults saved to: {output_dir}")
    print(f"Summary: {summary_file}")
    print("="*70 + "\n")
    
    return results


if __name__ == "__main__":
    results = run_search()