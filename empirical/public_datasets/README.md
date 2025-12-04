# Public Datasets for Hamiltonian Belief Dynamics Analysis

This directory contains analysis scripts for testing Hamiltonian belief dynamics
on public datasets. The datasets must be downloaded manually due to network restrictions.

## Dataset Download Instructions

### 1. 26-Wave Political Psychology Panel (PRIORITY)

**Best for**: Testing oscillation and temporal dynamics

- **URL**: https://osf.io/3pwvb/
- **Paper**: https://openpsychologydata.metajnl.com/articles/10.5334/jopd.54
- **Format**: CSV, SPSS
- **Download**: Click "Files" on OSF page, download all CSV files
- **Place in**: `political_panel/`

**Key variables**:
- Political attitudes (measured every 2 weeks)
- Perceived threat
- Political identification
- 552 participants x 26 waves

### 2. Twitter Polarization Dataset

**Best for**: Testing social mass equation on networks

- **URL**: https://gvrkiran.github.io/polarizationTwitter/
- **Paper**: ICWSM-17
- **Format**: Various
- **Download**: Follow links on page
- **Place in**: `twitter_polarization/`

**Key variables**:
- 679,000 users
- Network structure (who follows whom)
- Tweet content and hashtags
- Longitudinal (2016 election cycle)

### 3. Polymarket Historical Data

**Best for**: High-frequency oscillation detection (seconds-minutes)

- **URL**: https://polymarketanalytics.com or API
- **API Docs**: https://docs.polymarket.com/
- **Format**: JSON/CSV via API
- **Place in**: `prediction_markets/`

**Key variables**:
- Price time series (= aggregated beliefs)
- Market resolution events
- Volume data

### 4. ANES Panel Studies

**Best for**: Long-term attitude dynamics

- **URL**: https://electionstudies.org/data-center/
- **Key panels**:
  - 1980 Major Panel (4 waves)
  - 2016-2020-2024 Panel (3 elections)
- **Format**: SPSS, Stata
- **Place in**: `anes/`

### 5. AUTNES Austrian Panel (2017-2024)

- **URL**: https://www.nature.com/articles/s41597-025-05848-2
- **Waves**: 23 waves, ~3000 respondents each
- **Place in**: `autnes/`

---

## Analysis Scripts

After downloading, run the analysis scripts in this order:

1. `political_panel_analysis.py` - 26-wave panel oscillation test
2. `twitter_network_analysis.py` - Social mass on networks
3. `prediction_market_analysis.py` - High-frequency dynamics

Each script will:
1. Load the data
2. Compute Hamiltonian parameters (M, γ, Λ)
3. Test for oscillation vs overdamped dynamics
4. Compare τ predictions with observed relaxation times
