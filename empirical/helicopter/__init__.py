# -*- coding: utf-8 -*-
"""
Helicopter Task Framework
=========================

Framework for fitting Hamiltonian-VFE models to Nassar's helicopter task data.

Key components:
    data_loader: Load and preprocess helicopter task datasets
    run_analysis: Main analysis script (self-contained)
"""

from .data_loader import (
    load_mcguire_nassar_2014,
    load_all_datasets,
    HelicopterTrial,
    SubjectData,
    get_subject_ids,
    get_subject_data
)

from .run_analysis import run_full_analysis
