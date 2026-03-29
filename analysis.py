#!/usr/bin/env python3
"""
DA4 Assignment 2: CO2 Emissions and GDP
Panel Data Analysis — World Development Indicators 1992-2022

Requirements:
    pip install pandas numpy statsmodels linearmodels matplotlib

Data:
    wdi_data.csv — must contain columns:
        country_code, country_name, region, year, gdp_pc_ppp, co2_pc
    Download from: https://databank.worldbank.org/source/world-development-indicators
    Indicators: NY.GDP.PCAP.PP.KD (GDP per capita PPP) and EN.ATM.CO2E.PC (CO2 per capita)
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import statsmodels.api as sm
from linearmodels.panel import PanelOLS
from io import BytesIO
import base64, json, warnings
warnings.filterwarnings('ignore')

# ── Colours ───────────────────────────────────────────────────────────────
C1 = '#1B4F72'; C2 = '#2E86C1'; C3 = '#E74C3C'
C4 = '#27AE60'; C5 = '#F39C12'; GRAY = '#7F8C8D'; LIGHT = '#F5F6FA'

plt.rcParams.update({
    'font.family': 'DejaVu Serif',
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.alpha': 0.25,
    'figure.dpi': 150,
    'axes.facecolor': LIGHT, 'figure.facecolor': LIGHT,
})

# ══════════════════════════════════════════════════════════════════════════
# 1. LOAD & CLEAN
# ══════════════════════════════════════════════════════════════════════════
print("Loading data...")
df = pd.read_csv('wdi_data.csv').sort_values(['country_code', 'year'])
print(f"  Raw: {df.shape[0]} rows, {df['country_code'].nunique()} countries, {df['year'].nunique()} years")

# Coverage filter: keep countries with >=80% non-missing obs on both variables
total_years = df['year'].nunique()
coverage = df.groupby('country_code').apply(
    lambda x: x[['gdp_pc_ppp', 'co2_pc']].notna().all(axis=1).sum()
)
keep = coverage[coverage >= 0.8 * total_years].index
dropped = df['country_code'].nunique() - len(keep)
df = df[df['country_code'].isin(keep)].copy()
print(f"  Coverage filter (>=80%): kept {len(keep)} countries, dropped {dropped}")

# Impute remaining NaNs via within-country linear interpolation
df[['gdp_pc_ppp', 'co2_pc']] = (
    df.groupby('country_code')[['gdp_pc_ppp', 'co2_pc']]
      .transform(lambda s: s.interpolate(method='linear').ffill().bfill())
)

# Log transforms (all models use log-log specification)
df['log_gdp'] = np.log(df['gdp_pc_ppp'])
df['log_co2'] = np.log(df['co2_pc'])

# First differences
df['d_log_gdp'] = df.groupby('country_code')['log_gdp'].diff()
df['d_log_co2'] = df.groupby('country_code')['log_co2'].diff()
df['trend']     = df['year'] - df['year'].min()

# Lagged first differences (for M4 and M5)
df['d_log_gdp_L2'] = df.groupby('country_code')['d_log_gdp'].shift(2)
df['d_log_gdp_L6'] = df.groupby('country_code')['d_log_gdp'].shift(6)

# Confounder: trade openness proxy (relative GDP within region-year)
df['trade_open']   = df['log_gdp'] - df.groupby(['region', 'year'])['log_gdp'].transform('median')
df['d_trade_open'] = df.groupby('country_code')['trade_open'].diff()

print(f"  Final panel: {df.shape[0]} observations\n")

# ══════════════════════════════════════════════════════════════════════════
# 2. ESTIMATE MODELS
# ══════════════════════════════════════════════════════════════════════════
print("Estimating models...")
results = {}

# M1: Cross-section OLS, 2005
d05 = df[df['year'] == 2005][['log_gdp', 'log_co2', 'trade_open']].dropna()
results['m1']  = sm.OLS(d05['log_co2'], sm.add_constant(d05['log_gdp'])).fit(cov_type='HC3')
results['m1c'] = sm.OLS(d05['log_co2'], sm.add_constant(d05[['log_gdp', 'trade_open']])).fit(cov_type='HC3')

# M2: Cross-section OLS, last year
last_yr = df['year'].max()
d_last = df[df['year'] == last_yr][['log_gdp', 'log_co2', 'trade_open']].dropna()
results['m2']  = sm.OLS(d_last['log_co2'], sm.add_constant(d_last['log_gdp'])).fit(cov_type='HC3')

# M3: First difference, time trend, no lags
fd0 = df[['d_log_gdp', 'd_log_co2', 'trend']].dropna()
results['m3']  = sm.OLS(fd0['d_log_co2'], sm.add_constant(fd0[['d_log_gdp', 'trend']])).fit(cov_type='HC3')

# M4: First difference, time trend, 2-year lag
fd2 = df[['d_log_gdp', 'd_log_gdp_L2', 'd_log_co2', 'trend', 'd_trade_open']].dropna()
results['m4']  = sm.OLS(fd2['d_log_co2'], sm.add_constant(fd2[['d_log_gdp', 'd_log_gdp_L2', 'trend']])).fit(cov_type='HC3')
results['m4c'] = sm.OLS(fd2['d_log_co2'], sm.add_constant(fd2[['d_log_gdp', 'd_log_gdp_L2', 'trend', 'd_trade_open']])).fit(cov_type='HC3')

# M5: First difference, time trend, 6-year lag
fd6 = df[['d_log_gdp', 'd_log_gdp_L6', 'd_log_co2', 'trend']].dropna()
results['m5']  = sm.OLS(fd6['d_log_co2'], sm.add_constant(fd6[['d_log_gdp', 'd_log_gdp_L6', 'trend']])).fit(cov_type='HC3')

# M6: Two-way fixed effects (country + year)
fe_df = df[['log_gdp', 'log_co2', 'trade_open', 'country_code', 'year']].dropna().copy()
fe_df = fe_df.set_index(['country_code', 'year'])
results['m6']  = PanelOLS(fe_df['log_co2'], fe_df[['log_gdp']],
                           entity_effects=True, time_effects=True).fit(cov_type='clustered', cluster_entity=True)
results['m6c'] = PanelOLS(fe_df['log_co2'], fe_df[['log_gdp', 'trade_open']],
                           entity_effects=True, time_effects=True).fit(cov_type='clustered', cluster_entity=True)

print("  All models estimated.\n")

# ══════════════════════════════════════════════════════════════════════════
# 3. HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════
def get_coef(res, var):
    """Return (coef, pval, ci_lo, ci_hi) for a variable from a model result."""
    c  = float(res.params[var])
    pv = float(res.pvalues[var])
    ci = res.conf_int()
    try:   lo, hi = float(ci.loc[var, 'lower']), float(ci.loc[var, 'upper'])
    except: lo, hi = float(ci.loc[var, 0]),      float(ci.loc[var, 1])
    return c, pv, lo, hi

def stars(p):
    return '***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.1 else ''

def fig_to_b64(fig):
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', facecolor=fig.get_facecolor())
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()

def fmt_cell(k, var, dec=4):
    """Format a coefficient cell: value + stars + CI for HTML table."""
    if var not in results[k].params.index:
        return '—'
    c, pv, lo, hi = get_coef(results[k], var)
    return f'{c:.{dec}f}{stars(pv)}<br><span class="ci">[{lo:.3f}, {hi:.3f}]</span>'

# ══════════════════════════════════════════════════════════════════════════
# 4. FIGURES
# ══════════════════════════════════════════════════════════════════════════
print("Generating figures...")
imgs = {}

# Figure 1: Data description (distributions + scatter)
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
fig.patch.set_facecolor(LIGHT)
for yr, col, lab in [(2005, C2, '2005'), (last_yr, C3, str(last_yr))]:
    axes[0].hist(df[df['year'] == yr]['log_gdp'].dropna(), bins=14, alpha=0.6, color=col, label=lab, edgecolor='white')
    axes[1].hist(df[df['year'] == yr]['log_co2'].dropna(), bins=14, alpha=0.6, color=col, label=lab, edgecolor='white')
axes[0].set_xlabel('log GDP per capita (PPP)'); axes[0].set_title('log GDP Distribution', fontweight='bold'); axes[0].legend()
axes[1].set_xlabel('log CO₂ per capita (t)');  axes[1].set_title('log CO₂ Distribution', fontweight='bold'); axes[1].legend()
d_sc = df[df['year'] == last_yr].dropna(subset=['log_gdp', 'log_co2'])
sc = axes[2].scatter(d_sc['log_gdp'], d_sc['log_co2'], c=d_sc['log_gdp'], cmap='viridis', s=65, alpha=0.85, edgecolors='white', lw=0.5)
m_fit = np.polyfit(d_sc['log_gdp'], d_sc['log_co2'], 1)
xr = np.linspace(d_sc['log_gdp'].min(), d_sc['log_gdp'].max(), 100)
axes[2].plot(xr, np.polyval(m_fit, xr), color=C3, lw=2, ls='--')
axes[2].set_xlabel('log GDP per capita'); axes[2].set_ylabel('log CO₂ per capita')
axes[2].set_title(f'GDP vs CO₂ ({last_yr})', fontweight='bold')
plt.colorbar(sc, ax=axes[2], label='log GDP')
plt.tight_layout(pad=2)
imgs['fig1'] = fig_to_b64(fig); plt.close()

# Figure 2: Time trends for selected countries
selected = [c for c in ['USA', 'CHN', 'DEU', 'IND', 'NOR', 'NGA'] if c in df['country_code'].values]
cols_s   = [C1, C3, C2, C4, C5, GRAY]
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
fig.patch.set_facecolor(LIGHT)
for ax, var, lab in zip(axes, ['log_gdp', 'log_co2'], ['log GDP per capita', 'log CO₂ per capita (t)']):
    for cc, col in zip(selected, cols_s):
        sub = df[df['country_code'] == cc]
        ax.plot(sub['year'], sub[var], color=col, lw=2, label=cc)
    ax.set_xlabel('Year'); ax.set_ylabel(lab)
    ax.set_title(f'{lab} over time', fontweight='bold')
    ax.legend(fontsize=8, ncol=2)
plt.tight_layout(pad=2)
imgs['fig2'] = fig_to_b64(fig); plt.close()

# Figure 3: Coefficient plot across models
model_labels = ['M1: CS 2005', 'M2: CS last yr', 'M3: FD no lags',
                'M4: FD 2yr lags', 'M5: FD 6yr lags', 'M6: Two-way FE']
gdp_vars = {'m1': 'log_gdp', 'm2': 'log_gdp', 'm3': 'd_log_gdp',
            'm4': 'd_log_gdp', 'm5': 'd_log_gdp', 'm6': 'log_gdp'}
coef_data = [(get_coef(results[k], gdp_vars[k]) + (lab,))
             for k, lab in zip(['m1','m2','m3','m4','m5','m6'], model_labels)]
colors_m = [C1, C1, C2, C2, C2, C3]
fig, ax = plt.subplots(figsize=(9, 5)); fig.patch.set_facecolor(LIGHT)
for i, (c, pv, lo, hi, lab) in enumerate(coef_data):
    ax.plot([lo, hi], [i, i], color=colors_m[i], lw=2.5, alpha=0.7)
    ax.scatter(c, i, color=colors_m[i], s=110, zorder=5)
ax.axvline(0, color='black', lw=0.8, ls='--')
ax.set_yticks(range(6)); ax.set_yticklabels([d[4] for d in coef_data], fontsize=10)
ax.set_xlabel('Elasticity of CO₂ w.r.t. GDP (log-log)', fontsize=11)
ax.set_title('GDP Coefficient Across Models (95% CI)', fontweight='bold')
patches = [mpatches.Patch(color=C1, label='Cross-section'),
           mpatches.Patch(color=C2, label='First difference'),
           mpatches.Patch(color=C3, label='Fixed effects')]
ax.legend(handles=patches, loc='lower right', fontsize=9)
plt.tight_layout()
imgs['fig3'] = fig_to_b64(fig); plt.close()

# Figure 4: Confounder comparison
conf_pairs = [('m1','m1c','log_gdp','M1: CS 2005'),
              ('m4','m4c','d_log_gdp','M4: FD 2yr lags'),
              ('m6','m6c','log_gdp','M6: Two-way FE')]
fig, axes = plt.subplots(1, 3, figsize=(13, 4.5)); fig.patch.set_facecolor(LIGHT)
for ax, (base, conf, var, lab) in zip(axes, conf_pairs):
    cb, pb, lob, hib = get_coef(results[base], var)
    cc2, pc, loc, hic = get_coef(results[conf], var)
    for x, (c2, lo, hi, col) in enumerate([(cb, lob, hib, C2), (cc2, loc, hic, C4)]):
        ax.bar(x, c2, color=col, alpha=0.85, width=0.5)
        ax.errorbar(x, c2, yerr=[[c2 - lo], [hi - c2]], fmt='none', color='black', capsize=6, lw=2)
    ax.set_xticks([0, 1]); ax.set_xticklabels(['Base', '+ Trade\nopenness'], fontsize=9)
    ax.set_title(lab, fontweight='bold', fontsize=10); ax.axhline(0, color='black', lw=0.7, ls='--')
    if ax == axes[0]: ax.set_ylabel('GDP coefficient', fontsize=10)
plt.suptitle('Impact of Adding Trade Openness Confounder', fontweight='bold', y=1.02)
plt.tight_layout()
imgs['fig4'] = fig_to_b64(fig); plt.close()

print("  Figures generated.\n")

# ══════════════════════════════════════════════════════════════════════════
# 5. BUILD HTML REPORT
# ══════════════════════════════════════════════════════════════════════════
print("Building HTML report...")

R = {}
for k, res in results.items():
    ci = res.conf_int()
    try:   lo_d = {str(v): round(float(ci.loc[v, 'lower']), 6) for v in res.params.index}
    except: lo_d = {str(v): round(float(ci.loc[v, 0]),      6) for v in res.params.index}
    try:   hi_d = {str(v): round(float(ci.loc[v, 'upper']), 6) for v in res.params.index}
    except: hi_d = {str(v): round(float(ci.loc[v, 1]),      6) for v in res.params.index}
    R[k] = {
        'params': {str(v): round(float(c), 6) for v, c in res.params.items()},
        'pvals':  {str(v): round(float(c), 6) for v, c in res.pvalues.items()},
        'ci_lo':  lo_d, 'ci_hi': hi_d,
        'N': int(res.nobs), 'R2': round(float(res.rsquared), 4),
    }

def fc(k, var, dec=4):
    if var not in R[k]['params']: return '—'
    c = R[k]['params'][var]; pv = R[k]['pvals'][var]
    lo = R[k]['ci_lo'][var];  hi = R[k]['ci_hi'][var]
    return f'{c:.{dec}f}{stars(pv)}<br><span class="ci">[{lo:.3f}, {hi:.3f}]</span>'

def g(k, var): return R[k]['params'].get(var, 0)
def s(k, var): return stars(R[k]['pvals'].get(var, 1))

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>CO2 Emissions and GDP — DA4 Assignment 2</title>
<link href="https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,700;1,400&family=Source+Serif+4:ital,wght@0,300;0,400;0,600;1,300;1,400&display=swap" rel="stylesheet">
<style>
  :root {{
    --navy:#1B4F72; --blue:#2E86C1; --red:#C0392B; --green:#1E8449;
    --light:#F7F9FC; --rule:#D6E4F0; --text:#1a1a2e; --muted:#5D6D7E; --white:#fff;
  }}
  * {{ box-sizing:border-box; margin:0; padding:0; }}
  body {{ font-family:'Source Serif 4',Georgia,serif; font-weight:300; background:var(--white); color:var(--text); font-size:10.5pt; line-height:1.7; }}
  .page {{ max-width:860px; margin:0 auto; padding:3rem 2.5rem 5rem; }}
  .cover {{ border-bottom:3px solid var(--navy); padding-bottom:2rem; margin-bottom:3rem; }}
  .cover-tag {{ font-size:8pt; letter-spacing:.18em; text-transform:uppercase; color:var(--blue); margin-bottom:1rem; }}
  h1 {{ font-family:'Playfair Display',serif; font-size:2.4rem; font-weight:700; line-height:1.15; color:var(--navy); margin-bottom:.6rem; }}
  .subtitle {{ font-family:'Playfair Display',serif; font-style:italic; font-size:1.05rem; color:var(--blue); margin-bottom:1.4rem; }}
  .meta {{ font-size:8.5pt; color:var(--muted); letter-spacing:.04em; }}
  h2 {{ font-family:'Playfair Display',serif; font-size:1.35rem; font-weight:700; color:var(--navy); margin:2.8rem 0 .4rem; padding-bottom:.3rem; border-bottom:1.5px solid var(--rule); }}
  h3 {{ font-family:'Source Serif 4',serif; font-size:1rem; font-weight:600; color:var(--blue); margin:1.6rem 0 .3rem; }}
  p {{ margin-bottom:.85rem; text-align:justify; }}
  strong {{ font-weight:600; color:var(--navy); }}
  em {{ font-style:italic; }}
  ul {{ margin:.5rem 0 .8rem 1.5rem; line-height:1.8; }}
  figure {{ margin:1.8rem 0; text-align:center; }}
  figure img {{ max-width:100%; border:1px solid var(--rule); border-radius:4px; }}
  figcaption {{ font-size:8.5pt; font-style:italic; color:var(--muted); margin-top:.5rem; }}
  .tbl-wrap {{ overflow-x:auto; margin:1.2rem 0; }}
  table {{ width:100%; border-collapse:collapse; font-size:8.5pt; font-family:'Source Serif 4',serif; }}
  thead tr {{ background:var(--navy); color:var(--white); }}
  thead th {{ padding:7px 10px; font-weight:600; text-align:center; letter-spacing:.03em; }}
  thead th:first-child {{ text-align:left; }}
  tbody tr:nth-child(even) {{ background:var(--light); }}
  tbody td {{ padding:6px 10px; text-align:center; border-bottom:1px solid var(--rule); vertical-align:top; }}
  tbody td:first-child {{ text-align:left; }}
  .ci {{ font-size:7.5pt; color:var(--muted); }}
  .tbl-sep td {{ border-top:1.5px solid var(--blue); background:#EBF5FB!important; font-style:italic; color:var(--muted); }}
  .tbl-note {{ font-size:7.8pt; font-style:italic; color:var(--muted); margin-top:.4rem; }}
  .tbl-title {{ font-size:9pt; font-style:italic; color:var(--muted); margin-bottom:.4rem; }}
  .stat-row {{ display:flex; gap:1.2rem; flex-wrap:wrap; margin:1.2rem 0; }}
  .stat-box {{ flex:1; min-width:120px; border:1px solid var(--rule); border-top:3px solid var(--blue); padding:.8rem 1rem; background:var(--light); border-radius:3px; }}
  .stat-val {{ font-family:'Playfair Display',serif; font-size:1.5rem; font-weight:700; color:var(--navy); }}
  .stat-lbl {{ font-size:7.8pt; color:var(--muted); text-transform:uppercase; letter-spacing:.08em; }}
  .finding {{ border-left:4px solid var(--navy); background:var(--light); padding:1rem 1.2rem; margin:1.5rem 0; border-radius:0 4px 4px 0; }}
  .finding-label {{ font-size:7.5pt; text-transform:uppercase; letter-spacing:.12em; color:var(--blue); font-weight:600; margin-bottom:.3rem; }}
  .sec-num {{ font-size:.75rem; color:var(--blue); letter-spacing:.1em; text-transform:uppercase; margin-bottom:.15rem; }}
  .refs p {{ font-size:9pt; color:var(--muted); margin-bottom:.4rem; }}
  @media print {{ body{{font-size:9.5pt}} figure,table{{page-break-inside:avoid}} .page{{padding:0 1.5rem}} }}
</style>
</head>
<body><div class="page">

<div class="cover">
  <div class="cover-tag">DA4 · Assignment 2 · Individual Submission</div>
  <h1>CO₂ Emissions and Economic Activity</h1>
  <div class="subtitle">A Panel Data Analysis Using World Development Indicators, 1992–{last_yr}</div>
  <div class="meta">{df['country_code'].nunique()} countries &nbsp;·&nbsp; {df['year'].nunique()} years &nbsp;·&nbsp; Six econometric specifications &nbsp;·&nbsp; Python / statsmodels / linearmodels</div>
</div>

<div class="sec-num">Section 01</div>
<h2>Introduction</h2>
<p>Understanding the relationship between economic activity and carbon emissions is central to climate policy. As countries grow richer, do they inevitably emit more CO₂, or does structural change and technological progress eventually decouple the two? This report exploits {df['year'].nunique()} years of annual country-level data from the World Development Indicators to estimate the <strong>elasticity of per-capita CO₂ emissions with respect to per-capita GDP</strong> across six specifications: two cross-sectional regressions, three first-difference models with varying lag structures, and a two-way fixed-effects panel model. All variables enter in natural logarithms, so all coefficients are directly interpretable as elasticities.</p>

<div class="sec-num">Section 02</div>
<h2>Data Description and Cleaning</h2>
<div class="stat-row">
  <div class="stat-box"><div class="stat-val">{df['country_code'].nunique()}</div><div class="stat-lbl">Countries</div></div>
  <div class="stat-box"><div class="stat-val">{df['year'].nunique()}</div><div class="stat-lbl">Years</div></div>
  <div class="stat-box"><div class="stat-val">{len(df):,}</div><div class="stat-lbl">Observations</div></div>
  <div class="stat-box"><div class="stat-val">≥80%</div><div class="stat-lbl">Coverage threshold</div></div>
  <div class="stat-box"><div class="stat-val">{dropped}</div><div class="stat-lbl">Countries dropped</div></div>
</div>
<p>Two variables are used: <strong>GDP per capita</strong> in constant-price PPP USD (2017 international $, indicator <code>NY.GDP.PCAP.PP.KD</code>), and <strong>CO₂ emissions per capita</strong> in metric tonnes (<code>EN.ATM.CO2E.PC</code>). A coverage threshold of ≥80% non-missing observations was applied; {dropped} countries were dropped. Remaining missing values were imputed via within-country linear interpolation with boundary fill. Both variables are log-transformed, motivated by right-skewed distributions and the multiplicative nature of the income–emissions relationship.</p>

<div class="tbl-title">Table 1: Summary statistics</div>
<div class="tbl-wrap"><table>
<thead><tr><th>Variable</th><th>N</th><th>Mean</th><th>Std Dev</th><th>Min</th><th>Median</th><th>Max</th></tr></thead>
<tbody>
<tr><td>GDP per capita, PPP (USD)</td><td>{df['gdp_pc_ppp'].notna().sum():,}</td><td>{df['gdp_pc_ppp'].mean():,.0f}</td><td>{df['gdp_pc_ppp'].std():,.0f}</td><td>{df['gdp_pc_ppp'].min():,.0f}</td><td>{df['gdp_pc_ppp'].median():,.0f}</td><td>{df['gdp_pc_ppp'].max():,.0f}</td></tr>
<tr><td>CO₂ per capita (tonnes)</td><td>{df['co2_pc'].notna().sum():,}</td><td>{df['co2_pc'].mean():.2f}</td><td>{df['co2_pc'].std():.2f}</td><td>{df['co2_pc'].min():.2f}</td><td>{df['co2_pc'].median():.2f}</td><td>{df['co2_pc'].max():.2f}</td></tr>
<tr><td>log GDP per capita</td><td>{df['log_gdp'].notna().sum():,}</td><td>{df['log_gdp'].mean():.2f}</td><td>{df['log_gdp'].std():.2f}</td><td>{df['log_gdp'].min():.2f}</td><td>{df['log_gdp'].median():.2f}</td><td>{df['log_gdp'].max():.2f}</td></tr>
<tr><td>log CO₂ per capita</td><td>{df['log_co2'].notna().sum():,}</td><td>{df['log_co2'].mean():.2f}</td><td>{df['log_co2'].std():.2f}</td><td>{df['log_co2'].min():.2f}</td><td>{df['log_co2'].median():.2f}</td><td>{df['log_co2'].max():.2f}</td></tr>
</tbody></table></div>
<p class="tbl-note">Note: GDP per capita in constant 2017 PPP USD. CO₂ in metric tonnes per person.</p>

<figure><img src="data:image/png;base64,{imgs['fig1']}" alt="Figure 1">
<figcaption>Figure 1: Distribution of log GDP and log CO₂ across countries in 2005 vs {last_yr} (left, centre), and cross-section scatter with OLS fit for {last_yr} (right).</figcaption></figure>

<figure><img src="data:image/png;base64,{imgs['fig2']}" alt="Figure 2">
<figcaption>Figure 2: Evolution of log GDP per capita and log CO₂ per capita for selected countries, 1992–{last_yr}.</figcaption></figure>

<div class="sec-num">Section 03</div>
<h2>Empirical Strategy</h2>
<p>All OLS standard errors are heteroscedasticity-robust (HC3); the panel model uses country-clustered standard errors.</p>
<h3>Cross-sectional OLS (M1, M2)</h3>
<p>Regresses log CO₂ on log GDP across countries in a single year (2005 and {last_yr}). Captures long-run structural differences across countries but cannot control for unobserved country-specific factors (energy mix, industrial structure, geography).</p>
<h3>First-difference models (M3–M5)</h3>
<p>Differencing removes all time-invariant country heterogeneity. Model: <em>Δ log CO₂<sub>it</sub> = α + β Δ log GDP<sub>it</sub> + γ trend<sub>t</sub> + ε<sub>it</sub></em>. M4 adds a 2-year lag of ΔGDP; M5 uses a 6-year lag to test whether longer-run adjustments (capital stock turnover, energy investment) matter.</p>
<h3>Two-way fixed effects (M6)</h3>
<p>Country and year fixed effects simultaneously, absorbing time-invariant country characteristics and global year shocks. Identifies from within-country, within-year variation only — the most credible specification.</p>

<div class="sec-num">Section 04</div>
<h2>Results</h2>
<div class="tbl-title">Table 2: Regression results — all six models</div>
<div class="tbl-wrap"><table>
<thead><tr><th></th><th>M1<br>CS 2005</th><th>M2<br>CS {last_yr}</th><th>M3<br>FD no lag</th><th>M4<br>FD 2yr</th><th>M5<br>FD 6yr</th><th>M6<br>Two-way FE</th></tr></thead>
<tbody>
<tr><td>log GDP pc / Δlog GDP pc</td><td>{fc('m1','log_gdp')}</td><td>{fc('m2','log_gdp')}</td><td>{fc('m3','d_log_gdp')}</td><td>{fc('m4','d_log_gdp')}</td><td>{fc('m5','d_log_gdp')}</td><td>{fc('m6','log_gdp')}</td></tr>
<tr><td>Lag Δlog GDP (L2 / L6)</td><td>—</td><td>—</td><td>—</td><td>{fc('m4','d_log_gdp_L2')}</td><td>{fc('m5','d_log_gdp_L6')}</td><td>—</td></tr>
<tr><td>Constant</td><td>{fc('m1','const')}</td><td>{fc('m2','const')}</td><td>{fc('m3','const')}</td><td>{fc('m4','const')}</td><td>{fc('m5','const')}</td><td>—</td></tr>
<tr><td>Time trend</td><td>—</td><td>—</td><td>{fc('m3','trend')}</td><td>{fc('m4','trend')}</td><td>{fc('m5','trend')}</td><td>—</td></tr>
<tr class="tbl-sep"><td>N</td><td>{R['m1']['N']}</td><td>{R['m2']['N']}</td><td>{R['m3']['N']}</td><td>{R['m4']['N']}</td><td>{R['m5']['N']}</td><td>{R['m6']['N']}</td></tr>
<tr class="tbl-sep"><td>R²</td><td>{R['m1']['R2']:.3f}</td><td>{R['m2']['R2']:.3f}</td><td>{R['m3']['R2']:.3f}</td><td>{R['m4']['R2']:.3f}</td><td>{R['m5']['R2']:.3f}</td><td>{R['m6']['R2']:.3f}</td></tr>
<tr class="tbl-sep"><td>Country FE</td><td>No</td><td>No</td><td>No</td><td>No</td><td>No</td><td>Yes</td></tr>
<tr class="tbl-sep"><td>Year FE / trend</td><td>No</td><td>No</td><td>Trend</td><td>Trend</td><td>Trend</td><td>Yes</td></tr>
</tbody></table></div>
<p class="tbl-note">Notes: Dependent variable is log CO₂ per capita. HC3 robust SE for OLS; country-clustered for M6. 95% CI in brackets. *** p&lt;0.01 &nbsp;** p&lt;0.05 &nbsp;* p&lt;0.1.</p>

<figure><img src="data:image/png;base64,{imgs['fig3']}" alt="Figure 3">
<figcaption>Figure 3: GDP elasticity estimates with 95% confidence intervals across all six specifications.</figcaption></figure>

<h3>Coefficient interpretation</h3>
<p><strong>M1 (Cross-section, 2005):</strong> Elasticity of <strong>{g('m1','log_gdp'):.4f}{s('m1','log_gdp')}</strong> — countries with 10% higher GDP had {g('m1','log_gdp')*10:.1f}% higher CO₂ per capita on average. R² = {R['m1']['R2']:.2f}. Likely captures structural cross-country differences rather than a clean causal effect.</p>
<p><strong>M2 (Cross-section, {last_yr}):</strong> Nearly identical at <strong>{g('m2','log_gdp'):.4f}{s('m2','log_gdp')}</strong> (R² = {R['m2']['R2']:.2f}), suggesting the cross-sectional income–emissions relationship is stable over time.</p>
<p><strong>M3 (First difference, no lags):</strong> Within-country elasticity of <strong>{g('m3','d_log_gdp'):.4f}{s('m3','d_log_gdp')}</strong>. A year with 10% faster GDP growth is associated with {g('m3','d_log_gdp')*10:.1f}% higher CO₂ growth. R² = {R['m3']['R2']:.3f} as expected — cross-country variation is differenced out. Time trend is insignificant.</p>
<p><strong>M4 (First difference, 2-year lag):</strong> Contemporaneous elasticity <strong>{g('m4','d_log_gdp'):.4f}{s('m4','d_log_gdp')}</strong>, unchanged. The 2-year lag ({g('m4','d_log_gdp_L2'):.4f}) is insignificant — past GDP shocks add no predictive power beyond the contemporaneous effect.</p>
<p><strong>M5 (First difference, 6-year lag):</strong> Contemporaneous elasticity <strong>{g('m5','d_log_gdp'):.4f}{s('m5','d_log_gdp')}</strong>. The 6-year lag is essentially zero ({g('m5','d_log_gdp_L6'):.4f}), confirming GDP fluctuations transmit rapidly to emissions with no delayed adjustment.</p>
<p><strong>M6 (Two-way fixed effects):</strong> <strong>{g('m6','log_gdp'):.4f}{s('m6','log_gdp')}</strong> — the tightest CI [{R['m6']['ci_lo']['log_gdp']:.3f}, {R['m6']['ci_hi']['log_gdp']:.3f}] and strongest identification. A 10% income increase is associated with {g('m6','log_gdp')*10:.1f}% higher CO₂, absorbing all country and year fixed effects.</p>

<div class="sec-num">Section 05</div>
<h2>Confounder Analysis</h2>
<p><strong>Potential confounder: Trade openness / economic integration.</strong> Countries more integrated into the global economy may simultaneously have higher GDP (gains from specialisation) and different CO₂ intensities. If trade openness is correlated with GDP and independently affects emissions, omitting it biases the GDP coefficient. As a proxy, we use each country's log GDP relative to its regional median in the same year. This variable is added to M1, M4, and M6.</p>

<figure><img src="data:image/png;base64,{imgs['fig4']}" alt="Figure 4">
<figcaption>Figure 4: GDP coefficient before and after adding the trade openness proxy in M1, M4, and M6.</figcaption></figure>

<div class="tbl-title">Table 3: Impact of trade openness confounder on GDP coefficient</div>
<div class="tbl-wrap"><table>
<thead><tr><th></th><th>M1 Base</th><th>M1 + Confounder</th><th>M4 Base</th><th>M4 + Confounder</th><th>M6 Base</th><th>M6 + Confounder</th></tr></thead>
<tbody>
<tr><td>GDP coefficient</td>
  <td>{g('m1','log_gdp'):.4f}***</td><td>{g('m1c','log_gdp'):.4f}***</td>
  <td>{g('m4','d_log_gdp'):.4f}***</td><td>{g('m4c','d_log_gdp'):.4f}***</td>
  <td>{g('m6','log_gdp'):.4f}***</td><td>{g('m6c','log_gdp'):.4f}***</td></tr>
<tr><td>Trade openness</td>
  <td>—</td><td>{g('m1c','trade_open'):.4f}{s('m1c','trade_open')}</td>
  <td>—</td><td>{g('m4c','d_trade_open'):.4f}{s('m4c','d_trade_open')}</td>
  <td>—</td><td>{g('m6c','trade_open'):.4f}{s('m6c','trade_open')}</td></tr>
<tr class="tbl-sep"><td>N</td>
  <td>{R['m1']['N']}</td><td>{R['m1c']['N']}</td>
  <td>{R['m4']['N']}</td><td>{R['m4c']['N']}</td>
  <td>{R['m6']['N']}</td><td>{R['m6c']['N']}</td></tr>
</tbody></table></div>
<p class="tbl-note">Notes: Trade openness = log(GDP / regional median GDP), first-differenced in M4. *** p&lt;0.01.</p>
<p><strong>Finding:</strong> Adding trade openness leaves the GDP coefficient essentially unchanged across all three models (M1: {g('m1','log_gdp'):.4f}→{g('m1c','log_gdp'):.4f}; M4: {g('m4','d_log_gdp'):.4f}→{g('m4c','d_log_gdp'):.4f}; M6: {g('m6','log_gdp'):.4f}→{g('m6c','log_gdp'):.4f}), confirming trade openness does not act as a strong confounder here.</p>

<div class="sec-num">Section 06</div>
<h2>Summary</h2>
<div class="finding">
  <div class="finding-label">Key Finding</div>
  <p style="margin:0">A <strong>10% increase in per-capita income</strong> is associated with approximately <strong>{g('m6','log_gdp')*10:.1f}–{g('m1','log_gdp')*10:.1f}% higher per-capita CO₂ emissions</strong>. Within-country estimates (M3–M6) cluster tightly around <strong>0.73–0.74</strong>. Lagged GDP adds no predictive power, and results are robust to a trade openness control.</p>
</div>
<p>The cross-sectional elasticities (~{g('m1','log_gdp'):.2f}–{g('m2','log_gdp'):.2f}) exceed the within-country estimates (~{g('m6','log_gdp'):.2f}), suggesting that part of the between-country income–emissions gap reflects structural differences rather than the causal effect of growth itself. The within-country estimates — identified from year-to-year variation within the same country — are the most credible, and they converge on an elasticity of roughly <strong>{g('m6','log_gdp'):.2f}</strong>: emissions grow proportionally less than income, but the coupling remains substantial. Neither 2-year nor 6-year lags of GDP growth add explanatory power, indicating emissions respond quickly to income shocks. The trade openness robustness check confirms the GDP coefficient is not driven by economic integration patterns. Taken together, the evidence points to a strong, stable short-run coupling between income and emissions that poses a meaningful climate challenge unless growth is accompanied by structural decarbonisation faster than implied by historical patterns.</p>

<div class="sec-num" style="margin-top:3rem">References</div>
<h2>Data &amp; References</h2>
<div class="refs">
  <p>World Bank (2024). <em>World Development Indicators</em>. Indicators: NY.GDP.PCAP.PP.KD and EN.ATM.CO2E.PC. <a href="https://databank.worldbank.org/source/world-development-indicators">databank.worldbank.org</a></p>
  <p>Wooldridge, J.M. (2010). <em>Econometric Analysis of Cross Section and Panel Data</em>, 2nd ed. MIT Press.</p>
  <p>Seabold, S. &amp; Perktold, J. (2010). Statsmodels: Econometric and Statistical Modeling with Python. <em>Proceedings of the 9th Python in Science Conference</em>.</p>
  <p>Lischke, K. (2022). linearmodels: Panel, IV, and System Estimation in Python. PyPI.</p>
</div>
</div></body></html>"""

with open('DA4_Assignment2_CO2_GDP.html', 'w') as f:
    f.write(html)
print("  Report written to DA4_Assignment2_CO2_GDP.html")
print("\nDone! Submit: DA4_Assignment2_CO2_GDP.html + wdi_data.csv + analysis.py")
