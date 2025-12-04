# -*- coding: utf-8 -*-
"""
Helicopter Task Data Loader
===========================

Load and preprocess Matt Nassar's helicopter task datasets.

Datasets:
    - McGuireNassar2014: Main behavioral dataset (32 subjects, ~480 trials each)
    - jNeuroBehav: Extended behavioral data
    - pupilPaper: Pupil + behavioral data

Author: Hamiltonian-VFE Team
Date: December 2025
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import scipy.io as sio


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class HelicopterTrial:
    """Single trial from helicopter task."""
    trial_num: int
    outcome: float           # Observed helicopter position
    prediction: float        # Participant's prediction
    true_mean: float         # True generative mean
    prediction_error: float  # outcome - prediction (delta)
    update: float            # Change in prediction
    is_changepoint: bool     # Whether true mean changed
    noise_std: float         # Generative noise level
    hazard_rate: float       # Changepoint probability

    @property
    def learning_rate(self) -> float:
        """Implicit learning rate: update / prediction_error."""
        if abs(self.prediction_error) < 1e-6:
            return 0.0
        return self.update / self.prediction_error


@dataclass
class SubjectData:
    """All trials for a single subject."""
    subject_id: int
    trials: List[HelicopterTrial]

    # Metadata
    n_trials: int = field(init=False)
    n_changepoints: int = field(init=False)
    noise_levels: np.ndarray = field(init=False)

    def __post_init__(self):
        self.n_trials = len(self.trials)
        self.n_changepoints = sum(1 for t in self.trials if t.is_changepoint)
        self.noise_levels = np.unique([t.noise_std for t in self.trials])

    def get_arrays(self) -> Dict[str, np.ndarray]:
        """Get trial data as numpy arrays for vectorized operations."""
        return {
            'outcome': np.array([t.outcome for t in self.trials]),
            'prediction': np.array([t.prediction for t in self.trials]),
            'true_mean': np.array([t.true_mean for t in self.trials]),
            'prediction_error': np.array([t.prediction_error for t in self.trials]),
            'update': np.array([t.update for t in self.trials]),
            'is_changepoint': np.array([t.is_changepoint for t in self.trials]),
            'noise_std': np.array([t.noise_std for t in self.trials]),
            'hazard_rate': np.array([t.hazard_rate for t in self.trials]),
            'learning_rate': np.array([t.learning_rate for t in self.trials]),
        }

    def get_changepoint_indices(self) -> np.ndarray:
        """Get trial indices where changepoints occurred."""
        return np.array([i for i, t in enumerate(self.trials) if t.is_changepoint])

    def get_trials_around_changepoints(self,
                                        before: int = 5,
                                        after: int = 10) -> List[Dict]:
        """Extract trials around each changepoint for analysis."""
        cp_indices = self.get_changepoint_indices()
        arrays = self.get_arrays()

        segments = []
        for cp_idx in cp_indices:
            start = max(0, cp_idx - before)
            end = min(self.n_trials, cp_idx + after + 1)

            segment = {
                'cp_index': cp_idx,
                'relative_trial': np.arange(start - cp_idx, end - cp_idx),
            }
            for key, arr in arrays.items():
                segment[key] = arr[start:end]

            segments.append(segment)

        return segments


# =============================================================================
# Data Loading Functions
# =============================================================================

def _find_data_dir() -> Path:
    """Find the publishedDatasetsToShare directory."""
    # Try relative to this file
    this_dir = Path(__file__).parent

    candidates = [
        this_dir.parent.parent / 'publishedDatasetsToShare',
        Path('publishedDatasetsToShare'),
        Path('/home/user/Hamiltonian-VFE/publishedDatasetsToShare'),
    ]

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        "Could not find publishedDatasetsToShare directory. "
        f"Searched: {[str(p) for p in candidates]}"
    )


def load_mcguire_nassar_2014(data_dir: Optional[Path] = None) -> Dict[int, SubjectData]:
    """
    Load McGuireNassar2014 dataset.

    This is the main behavioral dataset with 32 subjects, ~480 trials each.

    Args:
        data_dir: Path to data directory (auto-detected if None)

    Returns:
        Dictionary mapping subject_id -> SubjectData
    """
    if data_dir is None:
        data_dir = _find_data_dir()

    filepath = data_dir / 'McGuireNassar2014data.mat'
    data = sio.loadmat(str(filepath))
    struct = data['allDataStruct'][0, 0]

    # Extract arrays
    subj_nums = struct['subjNum'].flatten()
    outcomes = struct['currentOutcome'].flatten()
    predictions = struct['currentPrediction'].flatten()
    true_means = struct['currentMean'].flatten()
    deltas = struct['currentDelta'].flatten()
    updates = struct['currentUpdate'].flatten()
    changepoints = struct['isChangeTrial'].flatten()
    stds = struct['currentStd'].flatten()
    hazards = struct['currentHazard'].flatten()

    # Group by subject
    subjects = {}
    unique_subjs = np.unique(subj_nums)

    for subj_id in unique_subjs:
        mask = subj_nums == subj_id
        indices = np.where(mask)[0]

        trials = []
        for i, idx in enumerate(indices):
            trial = HelicopterTrial(
                trial_num=i + 1,
                outcome=float(outcomes[idx]),
                prediction=float(predictions[idx]),
                true_mean=float(true_means[idx]),
                prediction_error=float(deltas[idx]),
                update=float(updates[idx]),
                is_changepoint=bool(changepoints[idx]),
                noise_std=float(stds[idx]),
                hazard_rate=float(hazards[idx]),
            )
            trials.append(trial)

        subjects[int(subj_id)] = SubjectData(
            subject_id=int(subj_id),
            trials=trials
        )

    return subjects


def load_jneurobehav(data_dir: Optional[Path] = None) -> Dict[str, np.ndarray]:
    """Load jNeuroBehav dataset (raw arrays)."""
    if data_dir is None:
        data_dir = _find_data_dir()

    filepath = data_dir / 'jNeuroBehav_toSend.mat'
    data = sio.loadmat(str(filepath))
    struct = data['jNeuroBehavData'][0, 0]

    return {
        'block': struct['Block'].flatten(),
        'prediction': struct['Prediction'].flatten(),
        'session': struct['session'].flatten(),
        'outcome': struct['outcome'].flatten(),
        'std': struct['standDev'].flatten(),
        'mean': struct['distMean'].flatten(),
    }


def load_pupil_data(data_dir: Optional[Path] = None) -> Dict[str, np.ndarray]:
    """Load pupil paper behavioral data (raw arrays)."""
    if data_dir is None:
        data_dir = _find_data_dir()

    filepath = data_dir / 'pupilPaperBehavData_toSend.mat'
    data = sio.loadmat(str(filepath))
    struct = data['pupilBehavData'][0, 0]

    return {
        'outcome': struct['outcome'],
        'mean': struct['mean'],
        'prediction': struct['prediction'],
        'std': struct['stdDev'],
        'trial_number': struct['trialNumber'],
    }


def load_all_datasets(data_dir: Optional[Path] = None) -> Dict:
    """Load all available datasets."""
    return {
        'mcguire_nassar_2014': load_mcguire_nassar_2014(data_dir),
        'jneurobehav': load_jneurobehav(data_dir),
        'pupil': load_pupil_data(data_dir),
    }


# =============================================================================
# Convenience Functions
# =============================================================================

def get_subject_ids(data_dir: Optional[Path] = None) -> List[int]:
    """Get list of subject IDs in McGuireNassar2014 dataset."""
    subjects = load_mcguire_nassar_2014(data_dir)
    return sorted(subjects.keys())


def get_subject_data(subject_id: int,
                     data_dir: Optional[Path] = None) -> SubjectData:
    """Get data for a specific subject."""
    subjects = load_mcguire_nassar_2014(data_dir)
    if subject_id not in subjects:
        raise ValueError(f"Subject {subject_id} not found. "
                        f"Available: {sorted(subjects.keys())}")
    return subjects[subject_id]


# =============================================================================
# Summary Statistics
# =============================================================================

def compute_summary_stats(subjects: Dict[int, SubjectData]) -> Dict:
    """Compute summary statistics across all subjects."""
    all_learning_rates = []
    all_updates = []
    all_errors = []
    cp_learning_rates = []
    non_cp_learning_rates = []

    for subj in subjects.values():
        arrays = subj.get_arrays()

        # Filter out NaN/Inf learning rates
        lr = arrays['learning_rate']
        valid = np.isfinite(lr) & (np.abs(lr) < 10)  # Clip extreme values

        all_learning_rates.extend(lr[valid])
        all_updates.extend(arrays['update'])
        all_errors.extend(np.abs(arrays['prediction_error']))

        cp_mask = arrays['is_changepoint']
        cp_learning_rates.extend(lr[valid & cp_mask])
        non_cp_learning_rates.extend(lr[valid & ~cp_mask])

    return {
        'n_subjects': len(subjects),
        'total_trials': sum(s.n_trials for s in subjects.values()),
        'total_changepoints': sum(s.n_changepoints for s in subjects.values()),
        'mean_learning_rate': np.nanmean(all_learning_rates),
        'std_learning_rate': np.nanstd(all_learning_rates),
        'mean_lr_at_changepoint': np.nanmean(cp_learning_rates),
        'mean_lr_non_changepoint': np.nanmean(non_cp_learning_rates),
        'mean_abs_error': np.nanmean(all_errors),
        'mean_abs_update': np.nanmean(np.abs(all_updates)),
    }


# =============================================================================
# Main (for testing)
# =============================================================================

if __name__ == '__main__':
    print("Loading McGuireNassar2014 dataset...")
    subjects = load_mcguire_nassar_2014()

    print(f"\nLoaded {len(subjects)} subjects")

    # Show one subject
    subj = subjects[1]
    print(f"\nSubject 1: {subj.n_trials} trials, {subj.n_changepoints} changepoints")

    arrays = subj.get_arrays()
    print(f"Mean learning rate: {np.nanmean(arrays['learning_rate']):.3f}")

    # Summary stats
    stats = compute_summary_stats(subjects)
    print(f"\nSummary Statistics:")
    for key, val in stats.items():
        if isinstance(val, float):
            print(f"  {key}: {val:.3f}")
        else:
            print(f"  {key}: {val}")
