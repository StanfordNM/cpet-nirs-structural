import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import os
from scipy.signal import savgol_filter
import seaborn as sns
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
from itertools import combinations
import warnings

plt.rcParams['font.family'] = 'Times New Roman'

# --------------------------------

NMD_MAPPING = {
    'DMD': 'Structural',
    'BMD': 'Structural',
    'LGMD': 'Structural',
    'FSHD': 'Structural',
    'CM': 'Structural',
    'GNE': 'Structural',
    'DM': 'Non-structural',
    'SMA': 'Non-structural',
    'Pompe': 'Non-structural',
    'Control': 'Control'
}

DIAGNOSIS_COLORS = {'Control': '#808080', 
                    'Non-structural': '#63b467',
                    'Structural': '#b47ede'}

PEAK_LABELS = {
        'VO2_peak_L':        'Peak VO₂ (L/min)',
        'VO2_peak_mLkgmin':  'Peak VO₂ (mL/kg/min)',
        'pred_vo2_pct':      'Predicted Peak VO₂ (%)',
        'VE_VO2_peak':       'Peak VE/VO₂ (no unit)',
        'VT2_VO2_AT':        'VO₂ at VT (mL/min/Kg)',
        'peak_RER':          'Peak RER (no unit)',
        'reached_max_RER':   'Peak RER ≥ 1.0 for ≥ 60s, n (%)',
        'peak_work_rate_watts': 'Peak Work Rate (W)',
        'peak_perceived_effort': 'Peak Perceived Effort (Borg)',
        'peak_HR':           'Peak Heart Rate (bpm)',
        'pred_HR_pct':       'Predicted Heart Rate (%)',
        'O2_pulse':          'O₂ pulse (mL/bpm)',
        'VE_VCO2_slope':     'VE VCO₂ Slope (no unit)',
        'VT1_VE_VCO2_slope': 'VT1 VE VCO₂ Slope (no unit)',
        'VT2_VE_VCO2_slope': 'VT2 VE VCO₂ Slope (no unit)',
        'delta_vo2_delta_wr': 'ΔVO₂/ΔWR (mL/min/W)',
    }

PEAK_LABELS_LATEX = {
        'VO2_peak_L':        r'Peak VO$_2$ (L/min)',
        'VO2_peak_mLkgmin':  r'Peak VO$_2$ (mL/kg/min)',
        'pred_vo2_pct':      r'Predicted Peak VO$_2$ (%)',
        'VE_VO2_peak':       r'Peak VE/VO$_2$ (no unit)',
        'VT2_VO2_AT':        r'VO$_2$ at VT (mL/min/Kg)',
        'peak_RER':          'Peak RER (no unit)',
        'reached_max_RER':   'Peak RER ≥ 1.0 for ≥ 60s, n (%)',
        'peak_work_rate_watts': 'Peak Work Rate (W)',
        'peak_perceived_effort': 'Peak Perceived Effort (Borg)',
        'peak_HR':           'Peak Heart Rate (bpm)',
        'pred_HR_pct':       'Predicted Heart Rate (%)',
        'O2_pulse':          r'O$_2$ pulse (mL/bpm)',
        'VE_VCO2_slope':     r'VE VCO$_2$ Slope (no unit)',
        'VT1_VE_VCO2_slope': r'VT1 VE VCO$_2$ Slope (no unit)',
        'VT2_VE_VCO2_slope': r'VT2 VE VCO$_2$ Slope (no unit)',
        'delta_vo2_delta_wr': r'ΔVO$_2$/ΔWR (mL/min/W)',
    }

RECOVERY_LABELS = {
    'VO2_t1_2': 'T1/2 VO₂ (s)',
    'VE_t1_2': 'T1/2 VE (s)',
    'VCO2_t1_2': 'T1/2 VCO₂ (s)',
    'VO2/HR_t1_2': 'T1/2 O₂ pulse (s)',
    'HR_t1_2': 'T1/2 HR (s)',
    'RQ_overshoot_pct': 'Overshoot RQ (%)',
    'VE/VO2_overshoot_pct': 'Overshoot VE/VO₂ (%)',
    'VE/VCO2_overshoot_pct': 'Overshoot VE/VCO₂ (%)',
    'PetO2_overshoot_pct': 'Overshoot PETO₂ (%)'}

RECOVERY_LABELS_LATEX = {
    'VO2_t1_2': r'$T_{1/2}$ VO$_2$ (s)',
    'VE_t1_2': r'$T_{1/2}$ VE (s)',
    'VCO2_t1_2': r'$T_{1/2}$ VCO$_2$ (s)',
    'VO2/HR_t1_2': r'$T_{1/2}$ O$_2$ pulse (s)',
    'HR_t1_2': r'$T_{1/2}$ HR (s)',
    'RQ_overshoot_pct': 'Overshoot RQ (%)',
    'VE/VO2_overshoot_pct': r'Overshoot VE/VO$_2$ (%)',
    'VE/VCO2_overshoot_pct': r'Overshoot VE/VCO$_2$ (%)',
    'PetO2_overshoot_pct': r'Overshoot PETO$_2$ (%)'}

PHASE_ORDER = ('REST', 'WARMUP', 'EXERCISE', 'RECOVERY')
WATT_COLS = [f'cpet_watts{i}' for i in range(1, 30)]
BORG_COLS = [f'cpet_borg{i}' for i in range(1, 30)]
OPTODE_COLS = {m: [f'{m}_optode_{i}' for i in (1, 2, 3)] for m in ('quad', 'ham', 'bi')}

SLOPE_COLS = {'VT2_VO2_AT': 'VT1 VO2 (ml/kg/min)',
              'VE_VCO2_slope': 'Ve/VCO2 Slope',
              'VT1_VE_VCO2_slope': 'VT1 Ve/VCO2 Slope',
              'VT2_VE_VCO2_slope': 'VT2 Ve/VCO2 Slope'}

PRO_LABELS = {'6MWT_distance_m': '6MWT   \nDistance (m)',
              'walk_time_secs': '10MWT Walk (s)',
              'run_time_secs': '10MWT Run (s)',
              'tug_time_comfortable_s': 'TUG Time (s)',
              'tug_time_max_s': 'TUG Max Time (s)',
              'osts_time_to_stand_s': 'OSTS (s)',
              'fsts_time_to_stand_s': 'FSTS (s)',
              'naq_response': 'NAQ',
              'vsa_response': 'VSAQ'}

# --------------------------------

def median_range(s):
    s = s.dropna()
    return 'N/A' if s.empty else f"{s.median():.1f} [{s.min():.1f}, {s.max():.1f}]"

def median_iqr(s):
    s = s.dropna()
    return 'N/A' if s.empty else f"{s.median():.1f} [{s.quantile(0.25):.1f}, {s.quantile(0.75):.1f}]"

def mean_sd(s):
    s = s.dropna()
    if s.empty:
        return 'N/A'
    return f"{s.mean():.1f} ({s.std():.1f})"

def pval_str(p):
    if np.isnan(p): return 'N/A'
    return '<0.001' if p < 0.001 else f"{p:.3f}"

# ------------------------------------

def compute_pairwise_group_stats(group_data, metric_cols, pairs, fdr_correction=True):
    """Mann-Whitney U per metric for each group pair, then BH-FDR correction across all metrics tested for that pair"""
    raw_pvals = {f"{g1} vs. {g2}": [] for g1, g2 in pairs}
    for col in metric_cols:
        for g1, g2 in pairs:
            a1 = group_data[g1][col].dropna().values
            a2 = group_data[g2][col].dropna().values
            p = stats.mannwhitneyu(a1, a2, alternative='two-sided')[1] if len(a1) >= 2 and len(a2) >= 2 else np.nan
            raw_pvals[f"{g1} vs. {g2}"].append((col, p))

    adjusted_pvals = {}
    for key, metric_ps in raw_pvals.items():
        cols_, ps = zip(*metric_ps)
        ps = np.array(ps)
        valid_mask = ~np.isnan(ps)
        adj = np.full_like(ps, np.nan)
        if fdr_correction and valid_mask.sum() > 0:
            adj[valid_mask] = stats.false_discovery_control(ps[valid_mask], method='bh')
        else:
            adj[valid_mask] = ps[valid_mask]
        adjusted_pvals[key] = {col: p for col, p in zip(cols_, adj)}
    return adjusted_pvals

# ------------------------------------

def _sex_pct(s):
    s = s.dropna()
    male_count = (s.str.lower() == 'male').sum()
    female_count = (s.str.lower() == 'female').sum()
    total = male_count + female_count
    if total == 0:
        return 'N/A'
    return f"M: {male_count} ({male_count / total * 100:.0f}%) / F: {female_count} ({female_count / total * 100:.0f}%)"

def _demographic_summary(pts, group_col):
    group = pts.groupby(group_col)
    counts = group['participant_id'].count()
    others = counts.drop(labels=['Control'], errors='ignore').sort_values(ascending=False).index.tolist()
    ordered = (['Control'] if 'Control' in counts.index else []) + others
    col_labels = {g: f"{g} (N = {counts[g]})" for g in ordered}
    summary = pd.DataFrame({'Age at first visit (y), Mean (SD)': group['age_at_cpet'].apply(mean_sd),
                            'Sex (%), M/F': group['sex'].apply(_sex_pct),
                            'Height (m)': group['height'].apply(median_range),
                            'Weight (kg)': group['weight'].apply(median_range),
                            'BMI (kg/m²)': group['bmi'].apply(median_range)}).T
    summary.columns = [col_labels[c] for c in summary.columns]
    summary = summary[list(col_labels.values())]
    summary.index.name = 'Metric'
    return summary, col_labels

def get_patient_info(cpet_demographics):
    pts = cpet_demographics.drop_duplicates('participant_id')
    summary, col_labels = _demographic_summary(pts, 'nmd_type')
    counts = pts.groupby('nmd_type')['participant_id'].count()

    disease_rows = []
    for disease in sorted(pts['clinical_diagnosis'].dropna().unique()):
        if disease == 'Control':
            continue
        row = {'Metric': f"  {disease}"}
        for diag, col in col_labels.items():
            n = ((pts['clinical_diagnosis'] == disease) & (pts['nmd_type'] == diag)).sum()
            row[col] = '-' if diag == 'Control' or n == 0 else f"{n} ({n / counts[diag] * 100:.1f}%)"
        disease_rows.append(row)

    disease_df = pd.DataFrame(disease_rows).set_index('Metric')
    header = pd.DataFrame([{col: '' for col in disease_df.columns}], index=['Disease'])
    return pd.concat([summary, pd.concat([header, disease_df])])

# ------------------------------------

def get_patient_info_by_diagnosis(cpet_demographics):
    return _demographic_summary(cpet_demographics.drop_duplicates('participant_id'), 'clinical_diagnosis')[0]

# ------------------------------------

def _first_and_max_recorded(manual_df, participant_id, cols):
    """(first, max) of the manually recorded once-per-minute values in cols for one participant; first = first min of exercise"""
    rows = manual_df[manual_df['participant_id'] == participant_id]
    if rows.empty:
        return np.nan, np.nan
    vals = rows.iloc[0][cols].dropna()
    if vals.empty:
        return np.nan, np.nan
    return vals.iloc[0], vals.max()

def get_work_rate(fp, df, participant_id, keiser_watt_df):
    """
    returns (baseline_wr, peak_wr) in watts for one CPET df
    SCIFIT: work rate is the 'Power' column already in the per-second CSV.
    KEISER: manually recorded once per minute in keiser_watt_df
    """
    if 'SCIFIT' in os.path.basename(fp):
        return _get_baseline_val(df, 'Power'), max(df['Power'].dropna())
    return _first_and_max_recorded(keiser_watt_df, participant_id, WATT_COLS)

def get_perceived_effort(participant_id, keiser_watt_df):
    return _first_and_max_recorded(keiser_watt_df, participant_id, BORG_COLS)

# ------------------------------------

def build_cpet_data(all_cpet_files, cpet_demographics, slope_df, keiser_watt_df):
    """
    Build a single cross-sectional dataframe with one row per participant (one CPET file each) containing all peak and recovery metrics
    """
    demo_by_id = cpet_demographics.set_index('participant_id')
    slope_by_id = slope_df.set_index('participant_id')

    rows = []
    for fp in all_cpet_files:
        pid = os.path.basename(fp).split('_')[0]
        if pid not in demo_by_id.index:
            continue
        demo_row = demo_by_id.loc[pid]

        df = pd.read_csv(fp).sort_values('t').reset_index(drop=True)
        peak_m = extract_peak_cpet_metrics(fp, demo_row['age_at_cpet'], demo_row['sex'], demo_row['weight'])
        if not peak_m:
            continue
        rec_m = compute_recovery_metrics(df, anchor_mode='exercise_end', min_sustain_sec=0)

        ## add manual WR for keiser
        baseline_wr, peak_wr = get_work_rate(fp, df, pid, keiser_watt_df)
        _, peak_perceived_effort = get_perceived_effort(pid, keiser_watt_df)
        baseline_vo2 = _get_baseline_val(df, 'VO2')
        peak_vo2_mL = peak_m.get('VO2_peak_L') * 1000
        delta_vo2_delta_wr = (peak_vo2_mL - baseline_vo2) / (peak_wr - baseline_wr)
        peak_m = {**peak_m,
                  'peak_work_rate_watts': peak_wr,
                  'baseline_work_rate_watts': baseline_wr,
                  'delta_vo2_delta_wr': delta_vo2_delta_wr,
                  'peak_perceived_effort': peak_perceived_effort}
        ## --

        row = {'participant_id': pid, 'nmd_type': demo_row['nmd_type'], **peak_m, **rec_m}
        for out_col, slope_col in SLOPE_COLS.items():
            row[out_col] = slope_by_id.loc[pid, slope_col] if pid in slope_by_id.index else np.nan
        rows.append(row)

    data = pd.DataFrame(rows)
    data['reached_max_RER'] = (data['peak_RER'] >= 1.0).astype(float)
    return data

# ------------------------------------

def _find_vo2_peak_idx(cpet_df):
    exercise = cpet_df[cpet_df['Phase'] == 'EXERCISE']
    exercise_end_t = exercise['t'].max()
    last_minute = exercise[exercise['t'] >= exercise_end_t - 60]
    return last_minute['VO2'].idxmax()

# -------------------------------

def _find_exercise_end_idx(df):
    exercise = df[df['Phase'] == 'EXERCISE']
    return exercise['t'].idxmax()

# -------------------------------

def apply_30s_rolling_average(df, window_sec=30):
    """Centered rolling average over a time window (default 30s), frequency remains same as original (1s). NaN values ignored."""
    result = df.copy()
    columns = [col for col in df.columns if col not in ['t', 'Phase']]
    t = df['t'].values
    vals = df[columns].values
    half_window = window_sec / 2
    lo_idx = np.searchsorted(t, t - half_window, side='left')
    hi_idx = np.searchsorted(t, t + half_window, side='right')
    smoothed = np.array([np.nanmean(vals[lo:hi], axis=0) for lo, hi in zip(lo_idx, hi_idx)])
    result[columns] = smoothed
    return result

# -------------------------------

def _get_baseline_val(df, col):
    rest = df[df['Phase'] == "REST"]
    val = rest[col].median()
    return val

def _find_sustained_crossing(sub_df, col, half_val, min_sustain_sec):
    """
    First timestamp where col <= half_val AND stays <= half_val for min_sustain_sec (inclusive)
    """
    t = sub_df['t'].values
    v = sub_df[col].values
    below = v <= half_val
    for i in range(len(t)):
        if not below[i]:
            continue
        window_mask = (t >= t[i]) & (t <= t[i] + min_sustain_sec)
        if below[window_mask].all():
            return t[i]
    return None

# -------------------------------

def compute_recovery_metrics(df, anchor_mode='exercise_end', min_sustain_sec=0):
    """For a single CPET dataframe, compute both T1/2 and overshoot recovery metrics.
    anchor_mode controls where each variable's peak is read from:
      'vo2_peak' - at the timestamp of VO2's own peak (constrained to last minute of EXERCISE)
      'exercise_end' - at the very last recorded EXERCISE-phase timestamp
    T1/2 = time for a variable to decay 50% of the way from its anchor point down to its own resting baseline (to rest + (peak - rest)/2)
    overshoot = anchor point to its max during the five minutes following that point
    """
    if 'VO2/HR' not in df.columns:
        if 'HR' in df.columns:
            df = df.copy()
            df['VO2/HR'] = df['VO2'] / df['HR']
        else:
            print("no HR or VO2/HR recorded for this test, skipping HR-derived metrics")

    df = apply_30s_rolling_average(df)
    exercise_end_time = df[df['Phase'] == 'EXERCISE']['t'].max()
    df.loc[df['t'] > exercise_end_time, 'Phase'] = 'RECOVERY'

    vo2_peak_idx = _find_vo2_peak_idx(df)
    exercise_end_idx = _find_exercise_end_idx(df)

    t1_2_cols = ['VO2', 'VE', 'VCO2'] + (['VO2/HR'] if 'VO2/HR' in df.columns else []) + (['HR'] if 'HR' in df.columns else [])
    overshoot_cols = ['RQ', 'VE/VO2', 'VE/VCO2', 'PetO2']

    metrics = {}
    for col in t1_2_cols:
        anchor_idx = exercise_end_idx if anchor_mode == 'exercise_end' else vo2_peak_idx
        anchor_t = df.loc[anchor_idx, 't']
        anchor_val = df.loc[anchor_idx, col]

        baseline_val = _get_baseline_val(df, col)
        half_val = baseline_val + (anchor_val - baseline_val) / 2

        post_anchor = df[df['t'] > anchor_t]
        crossing_t = _find_sustained_crossing(post_anchor, col, half_val, min_sustain_sec)
        metrics[f'{col}_t1_2'] = np.nan if crossing_t is None else crossing_t - anchor_t
        metrics[f'{col}_baseline'] = baseline_val

    for col in overshoot_cols:
        anchor_idx = exercise_end_idx if anchor_mode == 'exercise_end' else vo2_peak_idx
        anchor_t = df.loc[anchor_idx, 't']
        anchor_val = df.loc[anchor_idx, col]
        window = df[(df['t'] >= anchor_t) & (df['t'] <= anchor_t + 300)]
        if window.empty:
            metrics[f'{col}_overshoot_pct'] = np.nan
            metrics[f'{col}_overshoot_raw'] = np.nan
        else:
            max_val = window[col].max()
            metrics[f'{col}_overshoot_raw'] = max_val - anchor_val
            metrics[f'{col}_overshoot_pct'] = np.nan if anchor_val == 0 else ((max_val - anchor_val) / abs(anchor_val)) * 100

    return metrics

# ------------------------------------

def extract_peak_cpet_metrics(fp, age, sex, weight_kg):
    df = pd.read_csv(fp).sort_values('t').reset_index(drop=True)
    df = apply_30s_rolling_average(df)
    exercise = df[df['Phase'] == 'EXERCISE']
    if exercise.empty:
        return {}

    weight_lbs = weight_kg * 2.20462
    gender_f = 1 if str(sex).lower() == 'female' else 0
    pred_vo2 = 79.9 - (0.39 * age) - (13.7 * gender_f) - (0.127 * weight_lbs)
    pred_hr = 220 - age
    peak_rer = exercise['RQ'].max()
    
    did_sustain_max_rer = exercise[exercise['RQ'] >= 1.0]
    if not did_sustain_max_rer.empty:
        did_sustain_max_rer = did_sustain_max_rer.groupby((did_sustain_max_rer['t'].diff() > 1).cumsum()).filter(lambda x: len(x) >= 60)
        sustained_max_rer = True if not did_sustain_max_rer.empty else False
    else:
        sustained_max_rer = False

    missing = [c for c in ['VO2', 'VO2/Kg', 'RQ', 'HR', 'VO2/HR'] if c not in exercise.columns]
    if missing:
        print(f"  {os.path.basename(fp)}: missing columns {missing}")

    if 'VO2/HR' not in exercise.columns and 'VO2' in exercise.columns and 'HR' in exercise.columns:
        exercise = exercise.copy()
        exercise['VO2/HR'] = exercise['VO2'] / exercise['HR']

    def peak(col):
        return exercise[col].max() if col in exercise.columns else np.nan

    peak_vo2_mL = peak('VO2') # mL/min
    peak_vo2_mLkg = peak('VO2/Kg') # mL/min/Kg
    peak_hr = peak('HR')
    return {
        'VO2_peak_L': peak_vo2_mL / 1000 if not np.isnan(peak_vo2_mL) else np.nan,
        'VO2_peak_mLkgmin': peak_vo2_mLkg,
        'pred_vo2_pct': (peak_vo2_mLkg / pred_vo2 * 100) if pred_vo2 > 0 and not np.isnan(peak_vo2_mLkg) else np.nan,
        'VE_VO2_peak': peak('VE/VO2'),
        'peak_RER': peak_rer,
        'sustained_max_RER': sustained_max_rer,
        'peak_HR': peak_hr,
        'pred_HR_pct': (peak_hr / pred_hr * 100) if pred_hr > 0 and not np.isnan(peak_hr) else np.nan,
        'O2_pulse': peak('VO2/HR'),
    }
    
# ------------------------------------

def make_peak_cpet_table(data, fdr_correction=True):
    groups = ['Control', 'Structural', 'Non-structural']
    group_data = {g: data[data['nmd_type'] == g] for g in groups}

    print("Peak RER >= 1.0:")
    for g in groups:
        gd = group_data[g]
        n_total = len(gd)
        n_max = int(gd['reached_max_RER'].sum())
        print(f"  {g}: {n_max}/{n_total} ({n_max/n_total*100:.0f}%)")

    continuous_metrics = [c for c in PEAK_LABELS if c != 'reached_max_RER']

    pairs = [('Control', 'Structural'), ('Control', 'Non-structural'), ('Structural', 'Non-structural')]
    col_order = (
        [f"{g} (N={len(group_data[g])})" for g in groups]
        + ['Control vs. Structural', 'Control vs. Non-structural', 'Structural vs. Non-structural']
    )
    adjusted_pvals = compute_pairwise_group_stats(group_data, continuous_metrics, pairs, fdr_correction=fdr_correction)
    results = []
    for col, label in PEAK_LABELS.items():
        row = {'Metric': label}
        if col == 'reached_max_RER':
            for g in groups:
                gd = group_data[g]
                n_total = len(gd)
                n_max = int(gd['reached_max_RER'].sum())
                row[f"{g} (N={n_total})"] = f"{n_max} ({n_max/n_total*100:.0f}%)" if n_total > 0 else 'N/A'
            for g1, g2 in pairs:
                key = f"{g1} vs. {g2}"
                a1 = group_data[g1]['reached_max_RER'].dropna().values
                a2 = group_data[g2]['reached_max_RER'].dropna().values
                if len(a1) >= 2 and len(a2) >= 2:
                    ct = np.array([[int((a1 == 1).sum()), int((a1 == 0).sum())],
                                   [int((a2 == 1).sum()), int((a2 == 0).sum())]])
                    _, p = stats.fisher_exact(ct)
                    row[key] = pval_str(p)
                else:
                    row[key] = 'N/A'
        else:
            for g in groups:
                row[f"{g} (N={len(group_data[g])})"] = median_iqr(group_data[g][col])
            for g1, g2 in pairs:
                key = f"{g1} vs. {g2}"
                row[key] = pval_str(adjusted_pvals[key][col])
        results.append(row)
    return pd.DataFrame(results).set_index('Metric')[col_order]

# --------------------------------

def make_recovery_table(data, fdr_correction=True):
    groups = ['Control', 'Structural', 'Non-structural']
    group_data = {g: data[data['nmd_type'] == g] for g in groups}

    pairs = [('Control', 'Structural'), ('Control', 'Non-structural'), ('Structural', 'Non-structural')]
    col_order = [f"{g} (N={len(group_data[g])})" for g in groups] + ['Control vs. Structural', 'Control vs. Non-structural', 'Structural vs. Non-structural']
    adjusted_pvals = compute_pairwise_group_stats(group_data, RECOVERY_LABELS.keys(), pairs, fdr_correction=fdr_correction)

    results = []
    for col, label in RECOVERY_LABELS.items():
        row = {'Metric': label}
        for g in groups:
            row[f"{g} (N={len(group_data[g])})"] = median_iqr(group_data[g][col])
        for g1, g2 in pairs:
            key = f"{g1} vs. {g2}"
            row[key] = pval_str(adjusted_pvals[key][col])
        results.append(row)
    return pd.DataFrame(results).set_index('Metric')[col_order]

# --------------------------------

def graph_recovery_averages(all_cpet_files, cpet_demographics, metrics, ylabels=None, anchor_mode='exercise_end', group_map=None,
                             window_sec=180, resample_n=100, min_n=3, ncols=2, figsize=None, graph_smooth_sec=30, legend=True):
    """
    anchor_mode: 'vo2_peak' (VO2's own peak, last minute of EXERCISE) or 'exercise_end' (last recorded EXERCISE-phase timestamp)
    """
    ylabels = ylabels or {}
    group_map = group_map or {}
    groups = list(DIAGNOSIS_COLORS.keys())

    common_t = np.linspace(0, window_sec, resample_n)
    series = {m: {g: [] for g in groups} for m in metrics}

    id_to_raw_diag = dict(zip(cpet_demographics['participant_id'], cpet_demographics['clinical_diagnosis']))
    for fp in all_cpet_files:
        raw_diag = id_to_raw_diag[os.path.basename(fp).split('_')[0]]
        diag = group_map.get(raw_diag, raw_diag)
        if diag not in groups:
            continue

        df = pd.read_csv(fp).sort_values('t').reset_index(drop=True)
        if 'VO2/HR' not in df.columns and 'HR' in df.columns:
            df = df.copy()
            df['VO2/HR'] = df['VO2'] / df['HR']
        df = apply_30s_rolling_average(df)
        exercise_end_time = df[df['Phase'] == 'EXERCISE']['t'].max()
        df.loc[df['t'] > exercise_end_time, 'Phase'] = 'RECOVERY'
        vo2_peak_idx = _find_vo2_peak_idx(df)
        exercise_end_idx = _find_exercise_end_idx(df)

        for metric in metrics:
            if metric not in df.columns:
                continue
            anchor_idx = exercise_end_idx if anchor_mode == 'exercise_end' else vo2_peak_idx
            anchor_t = df.loc[anchor_idx, 't']
            rec = df[(df['t'] >= anchor_t) & (df['t'] <= anchor_t + window_sec)]
            rec = rec[['t', metric]].dropna()
            if len(rec) < 2:
                continue
            rel_t = rec['t'].values - anchor_t
            trace = np.interp(common_t, rel_t, rec[metric].values, left=np.nan, right=np.nan)
            if graph_smooth_sec > 0:
                half_window = graph_smooth_sec / 2
                lo_idx = np.searchsorted(common_t, common_t - half_window, side='left')
                hi_idx = np.searchsorted(common_t, common_t + half_window, side='right')
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', category=RuntimeWarning)
                    trace = np.array([np.nanmean(trace[lo:hi]) for lo, hi in zip(lo_idx, hi_idx)])
            series[metric][diag].append(trace)

    nrows = int(np.ceil(len(metrics) / ncols))
    figsize = figsize or (4 * ncols, 2.5 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    axes = axes.flatten()

    for i, (ax, metric) in enumerate(zip(axes, metrics)):
        for diag in groups:
            traces = series[metric][diag]
            if not traces:
                continue
            arr = np.vstack(traces)
            n_valid = np.sum(~np.isnan(arr), axis=0)
            mean_trace = np.nanmean(arr, axis=0)
            sem_trace = np.nanstd(arr, ddof=1, axis=0) / np.sqrt(np.maximum(n_valid, 1))
            keep = n_valid >= min_n
            color = DIAGNOSIS_COLORS[diag]
            label = f'{diag} (n={len(traces)})' if i == 0 else diag
            ax.plot(common_t[keep], mean_trace[keep], color=color, lw=1.5, label=label, zorder=3)
            ax.fill_between(common_t[keep], (mean_trace - sem_trace)[keep], (mean_trace + sem_trace)[keep],
                            color=color, alpha=0.2, edgecolor='none', zorder=2)

        is_bottom_row = i >= len(metrics) - ncols
        if is_bottom_row:
            ax.set_xlabel('Time (seconds)', fontsize=11)
        ax.set_ylabel(ylabels.get(metric, metric), fontsize=11)
        if legend and ax == axes[0]:
            ax.legend(fontsize=10, title='Group', frameon=True)

    for ax in axes[len(metrics):]:
        ax.axis('off')

    fig.tight_layout()
    plt.close(fig)
    return fig

# --------------------------------

def check_recovery_normality(data):
    groups = ['Control', 'Structural', 'Non-structural']
    group_colors = {'Control': '#4878CF', 'Structural': '#6ACC65', 'Non-structural': '#D65F5F'}
    n_metrics, n_groups = len(RECOVERY_LABELS), len(groups)
    fig, axes = plt.subplots(n_metrics, n_groups, figsize=(3.5 * n_groups, 2.25 * n_metrics))
    for row_i, (col, label) in enumerate(RECOVERY_LABELS.items()):
        for col_j, grp in enumerate(groups):
            ax = axes[row_i, col_j]
            vals = data.loc[data['nmd_type'] == grp, col].dropna()
            n = len(vals)
            if n < 3:
                ax.text(0.5, 0.5, f'n={n}\n(insufficient)', ha='center', va='center',
                        transform=ax.transAxes, fontsize=10)
                ax.set_title(f'{grp}\n{label}', fontsize=10)
                ax.axis('off')
                continue
            _, shapiro_p = stats.shapiro(vals)
            sns.histplot(vals, kde=True, ax=ax, color=group_colors[grp], alpha=0.6, edgecolor='white')
            pstr = '<0.001' if shapiro_p < 0.001 else f'{shapiro_p:.3f}'
            normal_str = 'NORMAL' if shapiro_p >= 0.05 else 'NON-NORMAL'
            ax.set_title(f'{grp}  |  {label}  |  n={n}\nShapiro p={pstr}, {normal_str}', fontsize=9)
            ax.set_xlabel('')
            ax.set_ylabel('Count' if col_j == 0 else '')
    plt.tight_layout()
    plt.show()
    
# --------------------------------

def plot_cpet_vs_pros(long_df, pt_pro_df, diagnosis, recovery_or_peak='recovery', fdr_correction=True, save_dir=None):
    cpet_df = long_df[long_df['nmd_type'] == diagnosis]
    pro_cols = [c for c in PRO_LABELS if c in pt_pro_df.columns]
    merged = cpet_df.merge(pt_pro_df[['participant_id'] + pro_cols], on='participant_id', how='inner', validate='1:1')
    print(f"{diagnosis}: {len(merged)}/{len(cpet_df)} participants with PRO data")

    if recovery_or_peak not in ['recovery', 'peak']:
        raise ValueError("recovery_or_peak must be either 'recovery' or 'peak'")
    
    if recovery_or_peak == 'recovery':
        cpet_cols = [c for c in RECOVERY_LABELS.keys() if c in merged.columns and merged[c].notna().sum() >= 5]
    else:
        cpet_cols = [c for c in PEAK_LABELS.keys() if c in merged.columns and merged[c].notna().sum() >= 5]
        if 'reached_max_RER' in cpet_cols:
            idx = cpet_cols.index('reached_max_RER')
            cpet_cols.pop(idx)
    functional_cols = [c for c in PRO_LABELS if c in merged.columns and merged[c].notna().sum() >= 5]
    cpet_labels = [RECOVERY_LABELS_LATEX[c] for c in cpet_cols] if recovery_or_peak == 'recovery' else [PEAK_LABELS_LATEX[c] for c in cpet_cols]
    functional_labels = [PRO_LABELS[c] for c in functional_cols]

    n_rec = len(cpet_cols)
    n_func = len(functional_cols)
    corr_matrix = np.full((n_rec, n_func), np.nan)
    pval_matrix = np.full((n_rec, n_func), np.nan)
    n_matrix = np.full((n_rec, n_func), np.nan)

    for i, rc in enumerate(cpet_cols):
        for j, fc in enumerate(functional_cols):
            pair = merged[[rc, fc]].dropna()
            if len(pair) >= 5:
                corr, p = stats.spearmanr(pair[rc], pair[fc])
                corr_matrix[i, j] = corr
                pval_matrix[i, j] = p
                n_matrix[i, j] = len(pair)

    min_cols_per_row = 2  # minimum number of functional measures with n >= 5 per CPET variable
    cols_with_sufficient_n = np.sum(n_matrix >= 5, axis=1)  # count columns with n >= 5 for each row
    keep_rows = [i for i in range(len(cpet_cols)) if cols_with_sufficient_n[i] >= min_cols_per_row]

    cpet_cols = [cpet_cols[i] for i in keep_rows]
    cpet_labels = [cpet_labels[i] for i in keep_rows]
    corr_matrix = corr_matrix[keep_rows, :]
    pval_matrix = pval_matrix[keep_rows, :]
    n_matrix = n_matrix[keep_rows, :]
    n_rec = len(cpet_cols)

    flat_ps = pval_matrix.flatten()
    valid_mask = ~np.isnan(flat_ps)
    adj_ps = np.full_like(flat_ps, np.nan)
    if valid_mask.sum() > 0:
        if fdr_correction:
            adj_ps[valid_mask] = multipletests(flat_ps[valid_mask], method='fdr_bh')[1]
        else:
            adj_ps[valid_mask] = flat_ps[valid_mask]
    adj_pval_matrix = adj_ps.reshape(n_rec, n_func)

    def starify(p):
        if np.isnan(p): return ''
        if p < 0.001: return '(***)'
        if p < 0.01: return '(**)'
        if p < 0.05: return '(*)'
        return ''

    annot = np.empty((n_rec, n_func), dtype=object)
    for i in range(n_rec):
        for j in range(n_func):
            p_text = starify(adj_pval_matrix[i, j])
            if p_text == '':
                annot[i, j] = '' if np.isnan(corr_matrix[i, j]) else f"{corr_matrix[i, j]:.2f}"
            else: 
                annot[i, j] = '' if np.isnan(corr_matrix[i, j]) else f"{corr_matrix[i, j]:.2f}\n{p_text}"

    fig, ax = plt.subplots(figsize=(max(8, n_func * 1.2), max(6, n_rec * 1.0)))
    sns.heatmap(corr_matrix, annot=annot, fmt='', annot_kws={'size': 12}, ax=ax, cmap="PRGn", vmin=-1, vmax=1, xticklabels=functional_labels, 
                yticklabels=cpet_labels, cbar_kws={'label': "Spearman's ρ"}, square=True, linewidths=0.5, linecolor='white')
    ax.set_title(diagnosis, fontsize=13)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha='right', fontsize=11)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=11)
    plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        fig.savefig(os.path.join(save_dir, f'{recovery_or_peak}_vs_functional_{diagnosis}.eps'), format='eps', bbox_inches='tight')
    plt.show()
    
# -------------------------------- #

######### NIRS FUNCTIONS #########

# -------------------------------- #

def build_optode_selection_supp_table(df):
    
    def get_deepest_optodes(df, optode_cols):
        """Highest-numbered usable optode per participant, 0 if none"""
        assert df[optode_cols].notna().all().all(), "NaN optode flags would be silently cast to True"
        flags = df[optode_cols].to_numpy(dtype=bool)
        k = len(optode_cols)
        return pd.Series(np.where(flags.any(axis=1), k - np.argmax(flags[:, ::-1], axis=1), 0), index=df.index)
    
    GROUPS = ['Control', 'Structural', 'Non-structural']
    MUSCLES = {
        'Quadriceps': dict(optode_cols=OPTODE_COLS['quad'], caliper='quadriceps_caliper'),
        'Hamstrings': dict(optode_cols=OPTODE_COLS['ham'], caliper='hamstrings_caliper'),
        'Biceps':  dict(optode_cols=OPTODE_COLS['bi'], caliper='biceps_caliper'),
    }
    DEPTH_LABELS = {1: 'Optode 1 (1.5cm depth)',
                    2: 'Optode 2 (1.75cm depth)',
                    3: 'Optode 3 (2.25cm depth)',
                    0: 'No acceptable signal'}
    
    df = df[df['diagnosis'].isin(GROUPS)].copy()
    n = df['diagnosis'].value_counts().reindex(GROUPS)
    rows = []

    def cont_row(label, col):
        vals = [median_iqr(df.loc[df['diagnosis'] == g, col]) for g in GROUPS]
        samples = [df.loc[df['diagnosis'] == g, col].dropna() for g in GROUPS]
        p = stats.kruskal(*samples).pvalue
        p = '<0.001' if p < 0.001 else f'{p:.3f}'
        rows.append((label, vals, p))

    def header_row(label):
        rows.append((label, [''] * len(GROUPS), ''))

    cont_row('BMI (kg/m²)', 'bmi')
    for muscle, cfg in MUSCLES.items():
        header_row(muscle)
        cont_row('Skinfold caliper (mm)', cfg['caliper'])
        header_row('Deepest usable optode, n (%)')
        deepest = get_deepest_optodes(df, cfg['optode_cols'])
        counts = (pd.crosstab(deepest, df['diagnosis']).reindex(index=list(DEPTH_LABELS), columns=GROUPS, fill_value=0))
        pct = (counts / n * 100).round(0).astype(int)
        for d, lab in DEPTH_LABELS.items():
            rows.append((lab, [f'{counts.loc[d, g]} ({pct.loc[d, g]}%)' for g in GROUPS], ''))
        if counts.loc[0].sum() > 0:
            print(f"{muscle}: participants with no usable optode:")
            print(df.loc[deepest == 0, 'basename'].tolist())

    out = pd.DataFrame([r[1] for r in rows], columns=[f'{g} (n={n[g]})' for g in GROUPS], index=[r[0] for r in rows])
    out['p'] = [r[2] for r in rows]
    out.index.name = ''
    return out
    
# --------------------------------

def smooth_tsi(y, window=11, polyorder=2):
    # Savitzky-Golay smoothing
    y = np.asarray(y, dtype=float)
    n = len(y)
    w = min(window, n if n % 2 == 1 else n - 1)
    if w <= polyorder or w < 3:
        return y
    return savgol_filter(y, w, polyorder)

# -------------------------------

def find_dxy_window(t, y, t0, min_search_sec=2, max_search_sec=30, smooth_window=11, smooth_poly=2, min_amplitude=0.0):
    t, y = np.asarray(t, dtype=float), np.asarray(y, dtype=float)
    window_mask = (t >= t0 - min_search_sec) & (t <= t0 + max_search_sec)
    t_win, y_raw = t[window_mask], y[window_mask]

    valid = ~np.isnan(y_raw)
    if valid.sum() < smooth_window:
        return None, None
    if not valid.all():
        y_raw = np.interp(t_win, t_win[valid], y_raw[valid])
    y_win = smooth_tsi(y_raw, smooth_window, smooth_poly)

    i_max, i_min = np.argmax(y_win), np.argmin(y_win)
    if i_max == i_min:
        return None, None
    i_ext1, i_ext2 = (i_max, i_min) if i_max < i_min else (i_min, i_max)

    dy = abs(y_win[i_ext2] - y_win[i_ext1])
    if dy < min_amplitude:
        return None, None
    return t_win[i_ext1], t_win[i_ext2]

# ------------------------------

def calculate_recovery_slope(nirs_df, tsi_col, t0, min_search_sec=2, max_search_sec=30, smooth_window=11, smooth_poly=2, min_amplitude=0.0):
    """
    Slope of the dominant TSI% excursion at recovery onset (t0 = R1), using same the same dynamic search as find_dxy_window
    """
    t1, t2 = find_dxy_window(nirs_df['t'].values, nirs_df[tsi_col].values, t0,
                              min_search_sec=min_search_sec, max_search_sec=max_search_sec,
                              smooth_window=smooth_window, smooth_poly=smooth_poly,
                              min_amplitude=min_amplitude)
    if t1 is None:
        return np.nan, None, None
    window = nirs_df[(nirs_df['t'] >= t1) & (nirs_df['t'] <= t2)][['t', tsi_col]].dropna()
    slope = np.polyfit(window['t'], window[tsi_col], 1)[0] if len(window) >= 2 else np.nan
    return slope, t1, t2

# ------------------------------

def calculate_TSI_vars(nirs_df, recovery_min_search_sec=2, recovery_max_search_sec=30):
    exercise_df = nirs_df[nirs_df['Phase'] == 'EXERCISE']
    if exercise_df.empty:
        return {}
    recovery_df = nirs_df[nirs_df['Phase'] == 'RECOVERY']
    t0_rec = recovery_df['t'].iloc[0] if not recovery_df.empty else None

    results = {}
    for muscle_prefix in ['quad', 'ham', 'bi']:
        tsi_col = f'[{muscle_prefix}] TSI%'
        if tsi_col not in nirs_df.columns:
            continue
        tsi_values = nirs_df[tsi_col].dropna()
        if tsi_values.empty:
            continue
        tsi_min, tsi_max = tsi_values.min(), tsi_values.max()
        tsi_recoveryslope, t_r1, t_r2 = (np.nan, None, None)
        if t0_rec is not None:
            tsi_recoveryslope, t_r1, t_r2 = calculate_recovery_slope(nirs_df, tsi_col, t0_rec, recovery_min_search_sec, recovery_max_search_sec)
        results[muscle_prefix] = {'tsi_min': tsi_min, 'tsi_max': tsi_max, 'tsi_recoveryslope': tsi_recoveryslope, 't_rec1': t_r1, 't_rec2': t_r2}
    return results

# -------------------------------

def build_TSI_summary_df(all_NIRS_dfs):
    rows = []
    for df in all_NIRS_dfs:
        results = calculate_TSI_vars(df)
        for muscle, vals in results.items():
            rows.append({'basename': df['basename'].iloc[0], 'participant_id': df['participant_id'].iloc[0],
                         'diagnosis': df['diagnosis'].iloc[0], 'muscle': muscle, **vals})
    return pd.DataFrame(rows)

# -------------------------------

def build_delta_nirs_table(cpet_dfs, id_to_diag, muscles=('quad', 'ham', 'bi')):
    # ---------
    def compute_delta_peak_or_nadir(df, col, mode):
        # zero to rest mean (t < 0), then max/min during exercise
        df = df.sort_values('t')
        if col not in df.columns or df[col].isna().all():
            return np.nan
        baseline = df.loc[df['t'] < 0, col].mean()
        ex = df.loc[df['Phase'] == 'EXERCISE', col].dropna()
        return (ex.max() if mode == 'max' else ex.min()) - baseline
    # ---------
    rows = []
    for participant_id, df in cpet_dfs.items():
        d = apply_30s_rolling_average(df.sort_values('t').copy())
        ex = d.loc[d['Phase'] == 'EXERCISE', 'VO2'].dropna()
        vo2_peak = ex.max() if not ex.empty else np.nan
        for m in muscles:
            for metric, (sig, mode) in {'dHHb_peak': ('HHb', 'max'), 'dO2Hb_nadir': ('O2Hb', 'min')}.items():
                col = f'{sig}_{m}'
                rows.append({'participant_id': participant_id, 'diagnosis': id_to_diag.get(participant_id),
                             'muscle': m, 'metric': metric,
                             'value': compute_delta_peak_or_nadir(df, col, mode),
                             'vo2_peak': vo2_peak})
    return pd.DataFrame(rows)

# -------------------------------

def create_nirs_combined_table(long_df):
    
    MUSCLE_LABELS = {'quad': 'Quadriceps', 'ham': 'Hamstrings', 'bi': 'Biceps'}
    METRIC_LABELS = {'dO2Hb_nadir': 'ΔO2Hb_nadir (μM)',
                    'dHHb_peak': 'ΔHHb_peak (μM)',
                    'tsi_recoveryslope': 'TSI_recoveryslope (%/s)'}

    dxs = [d for d in ('Control', 'Structural', 'Non-structural') if d in long_df['diagnosis'].unique()]
    n_per_dx = long_df.groupby('diagnosis')['participant_id'].nunique()
    pairs = list(combinations(dxs, 2))

    rows, pvals, p_locs = [], [], []
    for muscle in ('quad', 'ham', 'bi'):
        for metric in METRIC_LABELS:
            sub = long_df[(long_df['muscle'] == muscle) & (long_df['metric'] == metric)]
            if sub.empty:
                continue
            row = {'Metric': f'{MUSCLE_LABELS.get(muscle, muscle)} {METRIC_LABELS.get(metric, metric)}'}
            vals = {d: sub.loc[sub['diagnosis'] == d, 'value'].dropna() for d in dxs}
            for d in dxs:
                row[f'{d} (N={n_per_dx[d]})'] = median_iqr(vals[d])
            i = len(rows)
            for d1, d2 in pairs:
                col = f'{d1} vs. {d2}'
                row[col] = np.nan_to_num(np.nan)
                pvals.append(mannwhitneyu(vals[d1], vals[d2], alternative='two-sided')[1])
                p_locs.append((i, col))
            rows.append(row)

    table = pd.DataFrame(rows)
    if pvals:
        for (i, col), p in zip(p_locs, multipletests(pvals, method='fdr_bh')[1]):
            table.loc[i, col] = round(p, 3)
    cols = ['Metric'] + [f'{d} (N={n_per_dx[d]})' for d in dxs] + [f'{a} vs. {b}' for a, b in pairs]
    return table[cols]

# -------------------------------

def _get_dx_group(df):
    return df['diagnosis'].iloc[0]

def select_qc_optode_tbt(optodes_qc_df, basename, muscle):
    if basename not in optodes_qc_df.index:
        return None
    row = optodes_qc_df.loc[basename]
    for opt_num in (3, 2, 1):
        col = f'{muscle}_optode_{opt_num}'
        if bool(row.get(col, False)):
            return opt_num
    return None

def _phase_bounds(df, phase_col='Phase', t_col='t'):
    bounds = {}
    for phase in PHASE_ORDER:
        sub = df.loc[df[phase_col] == phase, t_col]
        if not sub.empty:
            bounds[phase] = (sub.min(), sub.max())
    return bounds

def _avg_phase_durations(all_NIRS_dfs, phase_col='Phase', t_col='t'):
    durs = {p: [] for p in PHASE_ORDER}
    for df in all_NIRS_dfs:
        for p, (t0, t1) in _phase_bounds(df, phase_col, t_col).items():
            durs[p].append(t1 - t0)
    return {p: (np.mean(v) if v else 0) for p, v in durs.items()}

def _phase_band_layout(avg_durations, min_frac=0.05):
    total = sum(avg_durations.values())
    min_width = min_frac * total
    widths = {p: max(avg_durations[p], min_width) for p in PHASE_ORDER}
    starts, ends, pos = {}, {}, 0.0
    for p in PHASE_ORDER:
        starts[p] = pos
        pos += widths[p]
        ends[p] = pos
    return starts, ends

def _warp_to_display_t(df, band_starts, band_ends, phase_col='Phase', t_col='t'):
    t_vals = df[t_col].values
    phases = df[phase_col].values
    display_t = np.full(len(df), np.nan)
    for phase, (t0, t1) in _phase_bounds(df, phase_col, t_col).items():
        dur = t1 - t0
        mask = phases == phase
        frac = np.zeros(mask.sum()) if dur <= 0 else (t_vals[mask] - t0) / dur
        display_t[mask] = band_starts[phase] + frac * (band_ends[phase] - band_starts[phase])
    return display_t


def _set_tbt_phase_xticks(ax, band_starts, band_ends):
    mids = {p: (band_starts[p] + band_ends[p]) / 2 for p in PHASE_ORDER}
    ax.set_xticks([mids[p] for p in PHASE_ORDER])
    ax.set_xticklabels(['Rest', 'Warmup', 'Exercise', 'Recovery'], fontsize=10)

    ax.axvspan(band_starts['REST'], band_ends['REST'], color='gray', alpha=0.08, zorder=0)
    ax.axvspan(band_starts['WARMUP'], band_ends['WARMUP'], color='cornflowerblue', alpha=0.1, zorder=0)
    ax.axvspan(band_starts['RECOVERY'], band_ends['RECOVERY'], color='gray', alpha=0.08, zorder=0)

    ax.axvline(band_starts['WARMUP'], color='gray', ls=':', lw=0.8, alpha=0.8, zorder=2.5)
    ax.axvline(band_starts['EXERCISE'], color='gray', ls=':', lw=0.8, alpha=0.8, zorder=2.5)
    ax.axvline(band_starts['RECOVERY'], color='gray', ls=':', lw=0.8, alpha=0.6, zorder=2.5)


def _resample_traces_tbt(all_NIRS_dfs, col_name_fn, common_t, band_starts, band_ends, tx_by_basename=None):
    """col_name_fn(basename) -> column name to pull, or None to skip.
    tx_by_basename: optional dict restricting which basenames are included (QC-passed optodes)."""
    resampled = {dx: [] for dx in DIAGNOSIS_COLORS}
    for df in all_NIRS_dfs:
        basename = df['basename'].iloc[0]
        if tx_by_basename is not None and basename not in tx_by_basename:
            continue
        col_name = col_name_fn(basename)
        if col_name is None or col_name not in df.columns:
            continue
        dx = _get_dx_group(df)
        if dx not in DIAGNOSIS_COLORS:
            continue
        sub = df[['t', 'Phase', col_name]].dropna(subset=['t', col_name])
        if sub.empty:
            continue
        display_t = _warp_to_display_t(sub, band_starts, band_ends)
        valid = ~np.isnan(display_t)
        if valid.sum() < 2:
            continue
        resampled[dx].append(np.interp(common_t, display_t[valid], sub[col_name].values[valid],
                                        left=np.nan, right=np.nan))
    return resampled


def _plot_mean_sem(ax, resampled, common_t, min_n, label_fn):
    for dx, color in DIAGNOSIS_COLORS.items():
        traces = resampled.get(dx, [])
        if not traces:
            continue
        arr = np.vstack(traces)
        n_valid = np.sum(~np.isnan(arr), axis=0)
        mean_trace = np.nanmean(arr, axis=0)
        sem_trace = np.nanstd(arr, axis=0, ddof=1) / np.sqrt(np.maximum(n_valid, 1))
        keep = n_valid >= min_n
        ax.plot(common_t[keep], mean_trace[keep], color=color, lw=2.5, label=label_fn(dx, len(traces)), zorder=3)
        ax.fill_between(common_t[keep], (mean_trace - sem_trace)[keep], (mean_trace + sem_trace)[keep],
                         color=color, alpha=0.2, edgecolor='none', zorder=2)

def plot_TBT_trajectories_qc(all_NIRS_dfs, optodes_qc_df, muscles=('quad', 'ham', 'bi'),
                              figsize=(16, 12), resample_n=250, min_n=3):
    avg_durations = _avg_phase_durations(all_NIRS_dfs)
    band_starts, band_ends = _phase_band_layout(avg_durations)
    common_t = np.linspace(band_starts['REST'], band_ends['RECOVERY'], resample_n)

    metrics = ['O2Hb', 'HHb', 'TSI%']
    fig, axes = plt.subplots(len(metrics), len(muscles), figsize=figsize, squeeze=False, sharex=True)

    for col, muscle in enumerate(muscles):
        tx_by_basename = {}
        for df in all_NIRS_dfs:
            basename = df['basename'].iloc[0]
            tx = select_qc_optode_tbt(optodes_qc_df, basename, muscle)
            if tx is not None:
                tx_by_basename[basename] = tx

        for row_i, metric in enumerate(metrics):
            ax = axes[row_i, col]
            if metric == 'TSI%':
                tsi_col = f'[{muscle}] TSI%'
                resampled = _resample_traces_tbt(all_NIRS_dfs, lambda b, c=tsi_col: c, common_t, band_starts, band_ends)
            else:
                resampled = _resample_traces_tbt(all_NIRS_dfs, lambda b, tx_map=tx_by_basename, m=muscle, 
                                                 met=metric: (f'[{m}] Rx1-Tx{tx_map[b]} {met}' if b in tx_map else None), 
                                                 common_t, band_starts, band_ends, tx_by_basename=tx_by_basename)

            label_fn = (lambda dx, n: f'{dx} (n={n})') if row_i == 0 else (lambda dx, n: dx)
            _plot_mean_sem(ax, resampled, common_t, min_n=min_n, label_fn=label_fn)
            _set_tbt_phase_xticks(ax, band_starts, band_ends)

            if row_i == 0:
                ax.set_title({'quad': 'Quadriceps', 'ham': 'Hamstrings', 'bi': 'Biceps'}.get(muscle, muscle), fontweight='bold', fontsize=18)
            if row_i == len(metrics) - 1:
                ax.set_xlabel('Phase', fontsize=16)
            if metric == 'TSI%':
                ax.set_ylabel('TSI%', fontsize=16)
            else:
                ax.set_ylabel(f'{metric} (ΔµM)', fontsize=16)
                ax.axhline(0, color='lightgray', lw=1.5, ls='--', zorder=0)

    for col in range(len(muscles)):
        axes[0, col].legend(fontsize=10, loc='upper left')
    fig.tight_layout()
    plt.close(fig)
    return fig

# -------------------------------

def _stem(fp):
    return os.path.splitext(os.path.basename(fp))[0]

def load_nirs_dfs(nirs_files, id_to_diag, optodes_qc_df):
    """Read NIRS files and tag participant_id, basename (file stem incl. test type), diagnosis. 
    TSI% set to NaN for muscles with no usable optode"""
    nirs_dfs = []
    for f in nirs_files:
        basename = _stem(f)
        participant_id = basename.split('_')[0]
        nirs_df = pd.read_csv(f)
        nirs_df['participant_id'] = participant_id
        nirs_df['basename'] = basename
        nirs_df['diagnosis'] = id_to_diag[participant_id]
        if basename in optodes_qc_df.index:
            row = optodes_qc_df.loc[basename]
            for muscle, cols in OPTODE_COLS.items():
                if not row[cols].astype(bool).fillna(False).any():
                    nirs_df[f'[{muscle}] TSI%'] = np.nan
        nirs_dfs.append(nirs_df)
    return nirs_dfs

def build_cpet_nirs_dfs(cpet_files, nirs_files, optodes_qc_df):
    """{participant_id: df of t, Phase, VO2 and QC-selected O2Hb_/HHb_ per muscle}. Expects one CPET file per participant"""
    nirs_by_basename = {_stem(f): f for f in nirs_files}
    cpet_nirs_dfs = {}
    for f in cpet_files:
        basename = _stem(f)
        participant_id = basename.split('_')[0]
        assert participant_id not in cpet_nirs_dfs, f"more than one CPET file for {participant_id}"
        cpet_df = pd.read_csv(f)
        nirs_df = pd.read_csv(nirs_by_basename[basename])

        keep, rename = ['t'], {}
        for sig in ('HHb', 'O2Hb'):
            for muscle in OPTODE_COLS:
                opt = select_qc_optode_tbt(optodes_qc_df, basename, muscle)
                col = f'[{muscle}] Rx1-Tx{opt} {sig}'
                if opt is not None and col in nirs_df.columns:
                    keep.append(col)
                    rename[col] = f'{sig}_{muscle}'

        cpet_vo2_only = cpet_df[['t', 'Phase', 'VO2']] if 'VO2' in cpet_df.columns else cpet_df[['t', 'Phase']]
        merged = pd.merge(cpet_vo2_only, nirs_df[keep].rename(columns=rename), on='t', how='inner', validate='one_to_one')
        cpet_nirs_dfs[participant_id] = merged[['t', 'Phase'] + [c for c in merged.columns if c not in ['t', 'Phase']]]
    return cpet_nirs_dfs
