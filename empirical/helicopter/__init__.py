# -*- coding: utf-8 -*-
"""
Helicopter Task Framework
=========================

Framework for fitting Hamiltonian-VFE models to Nassar's helicopter task data.

Key components:
    data_loader: Load and preprocess helicopter task datasets
    publication_analysis: Publication-quality analysis and figures
    changepoint_analysis: Specialized changepoint analysis
"""

from .data_loader import (
    load_mcguire_nassar_2014,
    load_all_datasets,
    HelicopterTrial,
    SubjectData,
    get_subject_ids,
    get_subject_data
)

from .publication_analysis import (
    run_publication_analysis,
    fit_subject,
    fit_population,
    PopulationResults,
    SubjectResults,
    ModelFit
)
