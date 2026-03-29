# CO₂ Emissions and Economic Activity
### DA4 Assignment 2 — Panel Data Analysis

Does economic growth cause CO₂ emissions? This report estimates the elasticity of per-capita CO₂ emissions with respect to per-capita GDP using six econometric specifications on 50 countries over 31 years (1992–2022).

---

## Files

| File | Description |
|------|-------------|
| `DA4_Assignment2_CO2_GDP.html` | Full report — open in any browser |
| `analysis.py` | Python script that downloads, cleans, estimates, and renders the report |
| `wdi_data.csv` | Panel dataset (World Development Indicators) |

---

## How to Run

**1. Install dependencies**
```bash
pip install pandas numpy statsmodels linearmodels matplotlib
```

**2. Run the analysis**
```bash
python analysis.py
```

This will generate `DA4_Assignment2_CO2_GDP.html` with all figures and results embedded.

---

## Data

- **Source:** World Bank — [World Development Indicators](https://databank.worldbank.org/source/world-development-indicators)
- **GDP per capita:** `NY.GDP.PCAP.PP.KD` (constant 2017 PPP USD)
- **CO₂ per capita:** `EN.ATM.CO2E.PC` (metric tonnes)
- **Coverage:** 50 countries, 1992–2022

---

## Models

| Model | Specification |
|-------|--------------|
| M1 | Cross-section OLS, 2005 |
| M2 | Cross-section OLS, 2022 |
| M3 | First difference + time trend, no lags |
| M4 | First difference + time trend, 2-year lag |
| M5 | First difference + time trend, 6-year lag |
| M6 | Two-way fixed effects (country + year) |

**Key finding:** A 10% increase in per-capita income is associated with approximately 7.3–9.5% higher per-capita CO₂ emissions. Within-country estimates (M3–M6) cluster around **0.73–0.74**, robust to a trade openness control.
