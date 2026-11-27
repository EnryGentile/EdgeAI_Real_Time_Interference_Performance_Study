#!/usr/bin/env python3
"""
analyze_doe_significance.py - Pipeline completa di Analisi Statistica DoE (Traccia A1).

Esegue automaticamente sui dataset raccolti in resultAGX/:
  1. Statistica descrittiva, tail latency (p95, p99) e degradazione % rispetto alla Baseline.
  2. Modello lineare DoE (ANOVA 1-way / 2-way N x S), quota di varianza spiegata (SS/SST %),
     test di normalita dei residui (Shapiro-Wilk) e test adattivo ANOVA vs Kruskal-Wallis
     (equivalente Python in batch di importance_significance.jsl).
  3. Test di significativita vs Baseline (Mann-Whitney U con correzione Holm-Bonferroni,
     Welch t-test ed Effect Size Cliff's delta).
  4. Correlazioni di Spearman tra risorse hardware (vmstat/termico) e degrado LLM.
  5. Esportazione di tabelle CSV pronte per il report e grafici comparativi CPU vs GPU.
"""

import argparse
import math
import re
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.sandbox.stats.runs import runstest_1samp
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multitest import multipletests
from statsmodels.tsa.stattools import kpss

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

N_ORDER = ["25% N_tot", "50% N_tot", "75% N_tot", "N_tot"]
S_ORDER = ["25% S_tot", "50% S_tot", "75% S_tot"]
METRICS = ["ttft_ms", "per_token_lat_ms", "throughput_tok_s", "jitter_ms"]
METRIC_LABELS = {
    "ttft_ms": "TTFT (ms)",
    "per_token_lat_ms": "TPOT (ms/tok)",
    "throughput_tok_s": "Throughput (tok/s)",
    "jitter_ms": "Jitter (ms)",
}
VMSTAT_COLS = [
    "vmstat_us", "vmstat_sy", "vmstat_id", "vmstat_wa",
    "vmstat_cs", "vmstat_in", "vmstat_free_kb", "vmstat_bi", "vmstat_bo",
    "vmstat_cpu_temp_c", "vmstat_gpu_temp_c",
]

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> Tuple[float, str]:
    """Calcola Cliff's delta non parametrico e la rispettiva magnitudine (Romano et al.)."""
    if len(x) == 0 or len(y) == 0:
        return 0.0, "negligible"
    diff = x[:, None] - y[None, :]
    delta = float((np.sum(diff > 0) - np.sum(diff < 0)) / (len(x) * len(y)))
    ad = abs(delta)
    if ad < 0.147:
        mag = "negligible"
    elif ad < 0.33:
        mag = "small"
    elif ad < 0.474:
        mag = "medium"
    else:
        mag = "large"
    return delta, mag


def load_doe_files(data_root: Path, model: str, benchmark: str, max_reps: int | None = 20) -> pd.DataFrame:
    """Carica tutti i file doe_*.csv per il modello/benchmark separando baseline e stressor (limitati a max_reps)."""
    frames: List[pd.DataFrame] = []
    pattern = f"doe_{model.lower()}_*_{benchmark.lower()}_*.csv"
    for csv_path in sorted(data_root.rglob(pattern)):
        parts = csv_path.stem.replace("doe_", "").split("_")
        device = parts[1].lower() if len(parts) > 1 else "unknown"
        df = pd.read_csv(csv_path)
        if df.empty:
            continue
        df["device"] = device
        df["source_file"] = csv_path.name
        df["size"] = df["size"].fillna("N/A").replace({"": "N/A", "nan": "N/A"})
        if max_reps is not None and max_reps > 0:
            grp_cols = [c for c in ["stressor_class", "num_workers", "size"] if c in df.columns]
            df = df.groupby(grp_cols, dropna=False, sort=False).head(max_reps).reset_index(drop=True)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def load_task_files(data_root: Path, model: str, benchmark: str, max_reps: int | None = 20) -> pd.DataFrame:
    """Carica i CSV per-task evitando doppioni (solo baseline_*_rep*_tasks e run_*_tasks), limitati a rep 1..max_reps."""
    frames: List[pd.DataFrame] = []
    re_base = re.compile(rf"^baseline_{re.escape(model)}_(cpu|gpu)_{re.escape(benchmark)}_rep(\d+)_tasks\.csv$", re.I)
    re_run = re.compile(rf"^run_{re.escape(model)}_(cpu|gpu)_{re.escape(benchmark)}_.*_rep(\d+)_tasks\.csv$", re.I)

    for csv_path in sorted(data_root.rglob("*_tasks.csv")):
        bn = csv_path.name
        if bn.lower().startswith("fallback_partial_"):
            continue
        m_b = re_base.match(bn)
        m_r = re_run.match(bn) if not m_b else None
        if not (m_b or m_r):
            continue
        rep_from_name = int((m_b or m_r).group(2))
        if max_reps is not None and max_reps > 0 and rep_from_name > max_reps:
            continue
        df = pd.read_csv(csv_path)
        if df.empty:
            continue
        if max_reps is not None and max_reps > 0 and "repetition" in df.columns:
            df = df[pd.to_numeric(df["repetition"], errors="coerce") <= max_reps].copy()
            if df.empty:
                continue
        df["device"] = df["device"].astype(str).str.lower()
        df["size"] = df["size"].fillna("N/A").replace({"": "N/A", "nan": "N/A"})
        frames.append(df)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def vargha_delaney_a12(x: np.ndarray, y: np.ndarray) -> Tuple[float, str]:
    """
    Calcola Vargha-Delaney A12 (probabilita che x > y) tramite Mann-Whitney U (Arcuri & Briand 2011).
    A12 = 0.5 indica equivalenza; magnitudine secondo Vargha & Delaney (2000).
    """
    if len(x) == 0 or len(y) == 0:
        return 0.5, "negligible"
    u_stat, _ = stats.mannwhitneyu(x, y, alternative="two-sided")
    a12 = float(u_stat / (len(x) * len(y)))
    d = abs(a12 - 0.5)
    if d < 0.06:
        mag = "negligible"
    elif d < 0.14:
        mag = "small"
    elif d < 0.21:
        mag = "medium"
    else:
        mag = "large"
    return a12, mag


def nonparametric_median_ci(vals: np.ndarray, z: float = 1.96) -> Tuple[float, float]:
    """Intervallo di confidenza al 95% non parametrico per la mediana (formula ai ranghi di Le Boudec / Hoefler)."""
    n = len(vals)
    if n == 0:
        return np.nan, np.nan
    if n <= 5:
        return float(np.min(vals)), float(np.max(vals))
    s = np.sort(vals)
    low_idx = max(0, int(math.floor((n - z * math.sqrt(n)) / 2.0)) - 1)
    high_idx = min(n - 1, int(math.ceil(1.0 + (n + z * math.sqrt(n)) / 2.0)) - 1)
    return float(s[low_idx]), float(s[high_idx])


def compute_descriptive_and_tail(df_doe: pd.DataFrame, df_tasks: pd.DataFrame) -> pd.DataFrame:
    """
    Calcola statistiche descrittive (Media, Media Armonica per rates, CoV, 95% CI Media e Mediana)
    e tail latency (p90, p95, p99) + Delta% vs Baseline (Hoefler & Belli SC15).
    """
    base_means: Dict[Tuple[str, str], float] = {}
    for dev, sub in df_doe[df_doe["stressor_class"] == "none"].groupby("device"):
        for m in METRICS:
            base_means[(dev, m)] = float(sub[m].mean())

    rows: List[Dict] = []
    grp_cols = ["device", "stressor_class", "num_workers", "size"]

    task_groups = {}
    if not df_tasks.empty:
        for key, g in df_tasks.groupby(grp_cols, dropna=False):
            task_groups[key] = g

    for key, g_doe in df_doe.groupby(grp_cols, dropna=False):
        dev, s_cls, nw, sz = key
        g_task = task_groups.get(key, g_doe)
        rec: Dict = {
            "device": dev,
            "stressor_class": s_cls,
            "num_workers": nw,
            "size": sz,
            "n_runs": len(g_doe),
            "n_tasks": len(g_task),
        }
        for m in METRICS:
            vals_run = g_doe[m].dropna().to_numpy(dtype=float)
            vals_task = g_task[m].dropna().to_numpy(dtype=float)
            mean_val = float(np.mean(vals_run))
            std_val = float(np.std(vals_run, ddof=1)) if len(vals_run) > 1 else 0.0
            cov_val = (std_val / mean_val) if mean_val != 0 else np.nan
            pos_run = vals_run[vals_run > 0]
            hmean_val = float(stats.hmean(pos_run)) if len(pos_run) > 0 else np.nan

            if len(vals_run) > 1 and std_val > 0:
                ci_low, ci_high = stats.t.interval(0.95, df=len(vals_run) - 1, loc=mean_val, scale=stats.sem(vals_run))
            else:
                ci_low, ci_high = mean_val, mean_val

            med_val = float(np.median(vals_task))
            med_ci_low, med_ci_high = nonparametric_median_ci(vals_task)
            iqr_val = float(np.percentile(vals_task, 75) - np.percentile(vals_task, 25))
            p90_val = float(np.percentile(vals_task, 90))
            p95_val = float(np.percentile(vals_task, 95))
            p99_val = float(np.percentile(vals_task, 99))
            b_ref = base_means.get((dev, m), np.nan)
            delta_pct = ((mean_val - b_ref) / b_ref * 100.0) if (b_ref and not np.isnan(b_ref)) else np.nan

            rec[f"{m}_mean"] = round(mean_val, 4)
            if m == "throughput_tok_s":
                rec[f"{m}_hmean"] = round(hmean_val, 4) if not np.isnan(hmean_val) else np.nan
            rec[f"{m}_std"] = round(std_val, 4)
            rec[f"{m}_cov"] = round(cov_val, 4) if not np.isnan(cov_val) else np.nan
            rec[f"{m}_mean_ci95_low"] = round(float(ci_low), 4)
            rec[f"{m}_mean_ci95_high"] = round(float(ci_high), 4)
            rec[f"{m}_median_task"] = round(med_val, 4)
            rec[f"{m}_median_ci95_low"] = round(med_ci_low, 4)
            rec[f"{m}_median_ci95_high"] = round(med_ci_high, 4)
            rec[f"{m}_iqr_task"] = round(iqr_val, 4)
            rec[f"{m}_p90_task"] = round(p90_val, 4)
            rec[f"{m}_p95_task"] = round(p95_val, 4)
            rec[f"{m}_p99_task"] = round(p99_val, 4)
            rec[f"{m}_delta_pct"] = round(delta_pct, 2) if not np.isnan(delta_pct) else np.nan
        rows.append(rec)

    return pd.DataFrame(rows)


def compute_doe_importance_and_significance(df_doe: pd.DataFrame) -> pd.DataFrame:
    """
    Replica in batch l'analisi di importance_significance.jsl:
      - Fit OLS (1-way N oppure 2-way N * S per mem/io) via statsmodels
      - Quota di varianza spiegata SS / SST * 100
      - Test di normalita Shapiro-Wilk sui residui e omoschedasticita di Levene via scipy.stats
      - Significativita F-test ANOVA vs Kruskal-Wallis non parametrico
    """
    records: List[Dict] = []
    df_stress = df_doe[df_doe["stressor_class"] != "none"].copy()

    for (dev, s_cls), df_sub in df_stress.groupby(["device", "stressor_class"]):
        if "duration" in df_sub.columns and "t_tot" in df_sub["duration"].values:
            df_sub = df_sub[df_sub["duration"] == "t_tot"].copy()

        # Campagna incompleta: con un solo livello di N l'ANOVA non e' definita
        if df_sub["num_workers"].nunique() < 2:
            print(f"   [SKIP] {dev}/{s_cls}: un solo livello di N ({df_sub['num_workers'].unique().tolist()}), ANOVA non calcolabile")
            continue

        valid_sizes = [s for s in df_sub["size"].unique() if s not in ("N/A", "none", "")]
        has_size = len(valid_sizes) > 1
        factors = ["num_workers", "size"] if has_size else ["num_workers"]
        cell_cols = ["num_workers", "size"] if has_size else ["num_workers"]
        # Sum-to-zero coding obbligatoria per ANOVA Type III (effetti principali marginali ortogonali)
        formula_rhs = "C(num_workers, Sum) * C(size, Sum)" if has_size else "C(num_workers)"

        for metric in METRICS:
            model = smf.ols(f"{metric} ~ {formula_rhs}", data=df_sub).fit()
            anova_tbl = sm.stats.anova_lm(model, typ=3 if has_size else 2)
            # Esclude Intercept se presente per il calcolo SST
            anova_no_int = anova_tbl.drop(index=["Intercept"], errors="ignore")
            sst = float(anova_no_int["sum_sq"].sum())
            resid = model.resid.to_numpy(dtype=float)
            sh_w, sh_p = stats.shapiro(resid)
            is_normal = bool(sh_p >= 0.05)

            cell_groups = [g[metric].dropna().to_numpy(dtype=float) for _, g in df_sub.groupby(cell_cols)]
            cell_groups = [g for g in cell_groups if len(g) > 0]
            if len(cell_groups) >= 2:
                lev_stat, lev_p = stats.levene(*cell_groups)
                is_homoscedastic = bool(lev_p >= 0.05)
            else:
                lev_stat, lev_p, is_homoscedastic = np.nan, np.nan, True

            assumptions_ok = is_normal and is_homoscedastic

            kw_results: Dict[str, Tuple[float, float]] = {}
            for f_col in factors:
                groups = [g[metric].dropna().to_numpy(dtype=float) for _, g in df_sub.groupby(f_col)]
                groups = [g for g in groups if len(g) > 0]
                if len(groups) >= 2:
                    h_stat, p_kw = stats.kruskal(*groups)
                    kw_results[f_col] = (float(h_stat), float(p_kw))

            for term_name, row in anova_no_int.iterrows():
                clean_term = (
                    term_name.replace("C(num_workers, Sum)", "num_workers")
                    .replace("C(size, Sum)", "size")
                    .replace("C(num_workers)", "num_workers")
                    .replace("C(size)", "size")
                )
                ss_val = float(row["sum_sq"])
                imp_pct = (ss_val / sst * 100.0) if sst > 0 else 0.0
                f_val = float(row["F"]) if pd.notna(row["F"]) else np.nan
                p_anova = float(row["PR(>F)"]) if pd.notna(row["PR(>F)"]) else np.nan

                kw_h, kw_p = kw_results.get(clean_term, (np.nan, np.nan))
                if clean_term == "Residual":
                    rec_test, active_p, sig = "Error", np.nan, False
                elif assumptions_ok:
                    rec_test = "ANOVA_F"
                    active_p = p_anova
                    sig = bool(pd.notna(active_p) and active_p < 0.05)
                elif not np.isnan(kw_p):
                    rec_test = "Kruskal-Wallis"
                    active_p = kw_p
                    sig = bool(pd.notna(active_p) and active_p < 0.05)
                else:
                    rec_test = "ANOVA_F (assumptions violated)"
                    active_p = p_anova
                    sig = bool(pd.notna(active_p) and active_p < 0.05)

                records.append({
                    "device": dev,
                    "stressor_class": s_cls,
                    "model_type": "2-way (N*S)" if has_size else "1-way (N)",
                    "anova_type": "Type III" if has_size else "Type II",
                    "metric": metric,
                    "term": clean_term,
                    "df": int(row["df"]),
                    "sum_sq": round(ss_val, 4),
                    "importance_pct": round(imp_pct, 2),
                    "r_squared": round(float(model.rsquared), 4),
                    "shapiro_w": round(float(sh_w), 4),
                    "shapiro_p": float(f"{sh_p:.4e}"),
                    "residuals_normal": is_normal,
                    "levene_stat": round(float(lev_stat), 4) if pd.notna(lev_stat) else np.nan,
                    "levene_p": float(f"{lev_p:.4e}") if pd.notna(lev_p) else np.nan,
                    "homoscedastic": is_homoscedastic,
                    "f_stat": round(f_val, 4) if pd.notna(f_val) else np.nan,
                    "anova_p": float(f"{p_anova:.4e}") if pd.notna(p_anova) else np.nan,
                    "kruskal_h": round(kw_h, 4) if pd.notna(kw_h) else np.nan,
                    "kruskal_p": float(f"{kw_p:.4e}") if pd.notna(kw_p) else np.nan,
                    "recommended_test": rec_test,
                    "active_p": float(f"{active_p:.4e}") if pd.notna(active_p) else np.nan,
                    "significant_005": sig,
                })

    return pd.DataFrame(records)


def compute_baseline_vs_stressor_tests(df_doe: pd.DataFrame) -> pd.DataFrame:
    """Test a coppie Mann-Whitney U (con correzione Holm-Bonferroni), Welch t-test e Cliff's delta vs Baseline."""
    rows: List[Dict] = []
    for dev in sorted(df_doe["device"].unique()):
        df_dev = df_doe[df_doe["device"] == dev]
        df_base = df_dev[df_dev["stressor_class"] == "none"]
        df_str = df_dev[df_dev["stressor_class"] != "none"]
        if df_base.empty or df_str.empty:
            continue

        for metric in METRICS:
            b_vals = df_base[metric].dropna().to_numpy(dtype=float)
            b_mean = float(np.mean(b_vals))
            raw_p_list: List[float] = []
            temp_recs: List[Dict] = []

            for (s_cls, nw, sz), g in df_str.groupby(["stressor_class", "num_workers", "size"]):
                s_vals = g[metric].dropna().to_numpy(dtype=float)
                s_mean = float(np.mean(s_vals))
                delta_pct = ((s_mean - b_mean) / b_mean * 100.0) if b_mean != 0 else np.nan
                _, p_mwu = stats.mannwhitneyu(s_vals, b_vals, alternative="two-sided")
                _, p_welch = stats.ttest_ind(s_vals, b_vals, equal_var=False)
                cdelta, cmag = cliffs_delta(s_vals, b_vals)
                a12_val, a12_mag = vargha_delaney_a12(s_vals, b_vals)
                raw_p_list.append(float(p_mwu))
                temp_recs.append({
                    "device": dev,
                    "metric": metric,
                    "stressor_class": s_cls,
                    "num_workers": nw,
                    "size": sz,
                    "baseline_mean": round(b_mean, 4),
                    "stressor_mean": round(s_mean, 4),
                    "delta_pct": round(delta_pct, 2),
                    "mwu_p_raw": float(f"{p_mwu:.4e}"),
                    "welch_p_raw": float(f"{p_welch:.4e}"),
                    "cliffs_delta": round(cdelta, 4),
                    "vargha_delaney_a12": round(a12_val, 4),
                    "effect_size": cmag,
                })

            if raw_p_list:
                reject, p_adj, _, _ = multipletests(raw_p_list, alpha=0.05, method="holm")
                for rec, rej, pa in zip(temp_recs, reject, p_adj):
                    rec["mwu_p_holm"] = float(f"{pa:.4e}")
                    rec["significant_holm_005"] = bool(rej)
                    rows.append(rec)

    return pd.DataFrame(rows)


def compute_hardware_correlations(df_tasks: pd.DataFrame) -> pd.DataFrame:
    """Calcola la correlazione di Spearman tra metriche hardware (vmstat) e metriche LLM per ogni stressor."""
    if df_tasks.empty:
        return pd.DataFrame()

    # Aggrega a livello di singola run (exp_id) per evitare pseudoreplicazione tra task della stessa run
    avail_vm = [c for c in VMSTAT_COLS if c in df_tasks.columns]
    if not avail_vm:
        return pd.DataFrame()

    agg_cols = {c: "mean" for c in METRICS + avail_vm}
    df_runs = (
        df_tasks[df_tasks["stressor_class"] != "none"]
        .groupby(["device", "stressor_class", "exp_id"], as_index=False)
        .agg(agg_cols)
    )

    rows: List[Dict] = []
    for (dev, s_cls), g in df_runs.groupby(["device", "stressor_class"]):
        for m in METRICS:
            y = g[m].to_numpy(dtype=float)
            for vm_col in avail_vm:
                x = g[vm_col].to_numpy(dtype=float)
                if np.std(x) == 0 or np.std(y) == 0:
                    continue
                rho, p_val = stats.spearmanr(x, y)
                if np.isnan(rho):
                    continue
                rows.append({
                    "device": dev,
                    "stressor_class": s_cls,
                    "llm_metric": m,
                    "hw_metric": vm_col.replace("vmstat_", ""),
                    "n_runs": len(g),
                    "spearman_rho": round(float(rho), 4),
                    "p_value": float(f"{p_val:.4e}"),
                    "significant_005": bool(p_val < 0.05),
                })

    return pd.DataFrame(rows)


def compute_stationarity_and_iid_tests(df_doe: pd.DataFrame) -> pd.DataFrame:
    """
    Verifica le ipotesi i.i.d. lungo le ripetizioni ordinate (Davis & Cucu-Grosjean LITES 2019):
      - Ljung-Box (acorr_ljungbox): assenza di autocorrelazione lag-1/lag-2
      - Wald-Wolfowitz Runs test (runstest_1samp): casualita della sequenza attorno alla mediana
      - KPSS (kpss): stazionarieta della serie (H0 = stazionaria -> p >= 0.05)
      - KS 2-sample (ks_2samp): uguaglianza distributiva tra prima e seconda meta delle ripetizioni
    """
    rows: List[Dict] = []
    grp_cols = ["device", "stressor_class", "num_workers", "size"]

    for (dev, s_cls, nw, sz), g in df_doe.groupby(grp_cols, dropna=False):
        for metric in ("ttft_ms", "per_token_lat_ms"):
            vals = g[metric].dropna().to_numpy(dtype=float)
            n = len(vals)
            if n < 8:
                continue

            # Ljung-Box (lag 1 e 2)
            try:
                lb = acorr_ljungbox(vals, lags=[1, 2], return_df=True)
                lb_p1 = float(lb["lb_pvalue"].iloc[0])
                lb_p2 = float(lb["lb_pvalue"].iloc[1])
            except Exception:
                lb_p1, lb_p2 = np.nan, np.nan

            # Runs test attorno alla mediana
            try:
                _, runs_p = runstest_1samp(vals, cutoff="median", correction=True)
                runs_p = float(runs_p)
            except Exception:
                runs_p = np.nan

            # KPSS stationarity test (H0 = serie stazionaria)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    _, kpss_p, _, _ = kpss(vals, regression="c", nlags="auto")
                kpss_p = float(kpss_p)
            except Exception:
                kpss_p = np.nan

            # Two-sample Kolmogorov-Smirnov tra prima e seconda meta delle ripetizioni
            mid = n // 2
            _, ks_p = stats.ks_2samp(vals[:mid], vals[mid:])

            rows.append({
                "device": dev,
                "stressor_class": s_cls,
                "num_workers": nw,
                "size": sz,
                "metric": metric,
                "n_reps": n,
                "ljungbox_p_lag1": round(lb_p1, 4) if pd.notna(lb_p1) else np.nan,
                "ljungbox_p_lag2": round(lb_p2, 4) if pd.notna(lb_p2) else np.nan,
                "runs_test_p": round(runs_p, 4) if pd.notna(runs_p) else np.nan,
                "kpss_p": round(kpss_p, 4) if pd.notna(kpss_p) else np.nan,
                "ks_halves_p": round(float(ks_p), 4),
                "independent_lb": bool(pd.notna(lb_p1) and lb_p1 >= 0.05),
                "stationary_kpss": bool(pd.notna(kpss_p) and kpss_p >= 0.05),
                "identically_dist_ks": bool(ks_p >= 0.05),
            })

    return pd.DataFrame(rows)


def compute_posthoc_worker_levels(df_doe: pd.DataFrame) -> pd.DataFrame:
    """
    Test post-hoc a coppie tra tutti i livelli di carico N (25%, 50%, 75%, 100%):
      - Tukey HSD (statsmodels pairwise_tukeyhsd)
      - Mann-Whitney U a coppie con correzione Holm-Bonferroni e Vargha-Delaney A12
    """
    rows: List[Dict] = []
    df_stress = df_doe[df_doe["stressor_class"] != "none"].copy()

    for (dev, s_cls), df_sub in df_stress.groupby(["device", "stressor_class"]):
        levels = [lv for lv in N_ORDER if lv in df_sub["num_workers"].unique()]
        if len(levels) < 2:
            continue

        for metric in METRICS:
            sub_m = df_sub[["num_workers", metric]].dropna()
            tukey_map: Dict[Tuple[str, str], Tuple[float, float, bool]] = {}
            try:
                tk = pairwise_tukeyhsd(endog=sub_m[metric].to_numpy(dtype=float), groups=sub_m["num_workers"].astype(str), alpha=0.05)
                for row in tk.summary().data[1:]:
                    g1, g2, meandiff, p_adj, _, _, rej = row
                    tukey_map[(str(g1), str(g2))] = (float(meandiff), float(p_adj), bool(rej))
                    tukey_map[(str(g2), str(g1))] = (-float(meandiff), float(p_adj), bool(rej))
            except Exception:
                pass

            pair_recs: List[Dict] = []
            raw_p: List[float] = []
            for i in range(len(levels)):
                for j in range(i + 1, len(levels)):
                    l1, l2 = levels[i], levels[j]
                    v1 = sub_m[sub_m["num_workers"] == l1][metric].to_numpy(dtype=float)
                    v2 = sub_m[sub_m["num_workers"] == l2][metric].to_numpy(dtype=float)
                    _, p_mwu = stats.mannwhitneyu(v1, v2, alternative="two-sided")
                    a12_val, a12_mag = vargha_delaney_a12(v2, v1)
                    tk_md, tk_p, tk_rej = tukey_map.get((l1, l2), (np.nan, np.nan, False))
                    raw_p.append(float(p_mwu))
                    pair_recs.append({
                        "device": dev,
                        "stressor_class": s_cls,
                        "metric": metric,
                        "level_a": l1,
                        "level_b": l2,
                        "mean_diff_b_minus_a": round(float(np.mean(v2) - np.mean(v1)), 4),
                        "tukey_p_adj": float(f"{tk_p:.4e}") if pd.notna(tk_p) else np.nan,
                        "tukey_sig_005": tk_rej,
                        "mwu_p_raw": float(f"{p_mwu:.4e}"),
                        "a12_b_vs_a": round(a12_val, 4),
                        "a12_magnitude": a12_mag,
                    })

            if raw_p:
                rej_h, p_h, _, _ = multipletests(raw_p, alpha=0.05, method="holm")
                for rec, r_h, pa_h in zip(pair_recs, rej_h, p_h):
                    rec["mwu_p_holm"] = float(f"{pa_h:.4e}")
                    rec["mwu_sig_holm_005"] = bool(r_h)
                    rows.append(rec)

    return pd.DataFrame(rows)


def _format_fisher_or(or_val: float) -> float:
    """Formatta l'Odds Ratio di Fisher preservando NaN (0/0) e inf (div/0)."""
    if np.isnan(or_val):
        return np.nan
    if np.isinf(or_val):
        return np.inf
    return round(float(or_val), 3)


def compute_slo_attainment_and_fisher(df_tasks: pd.DataFrame) -> pd.DataFrame:
    """
    Calcola lo SLO Attainment (%) sui singoli task (stile Splitwise/DistServe) rispetto a due soglie
    relative alla Baseline (1.5x e 2.0x p95_baseline) ed esegue il Fisher's Exact Test + Odds Ratio
    per le violazioni SLO come prescritto da Arcuri & Briand (2011) per esiti dicotomici.
    """
    if df_tasks.empty:
        return pd.DataFrame()

    rows: List[Dict] = []
    grp_cols = ["stressor_class", "num_workers", "size"]

    for dev in sorted(df_tasks["device"].unique()):
        df_dev = df_tasks[df_tasks["device"] == dev]
        df_base = df_dev[df_dev["stressor_class"] == "none"]
        if df_base.empty:
            continue

        for metric in ("ttft_ms", "per_token_lat_ms"):
            b_vals = df_base[metric].dropna().to_numpy(dtype=float)
            base_p95 = float(np.percentile(b_vals, 95))
            thr_15 = 1.5 * base_p95
            thr_20 = 2.0 * base_p95

            b_ok_15 = int(np.sum(b_vals <= thr_15))
            b_fail_15 = len(b_vals) - b_ok_15
            b_ok_20 = int(np.sum(b_vals <= thr_20))
            b_fail_20 = len(b_vals) - b_ok_20

            temp_recs: List[Dict] = []
            p_list_15: List[float] = []
            p_list_20: List[float] = []

            for (s_cls, nw, sz), g in df_dev.groupby(grp_cols, dropna=False):
                vals = g[metric].dropna().to_numpy(dtype=float)
                n_tot = len(vals)
                if n_tot == 0:
                    continue
                ok_15 = int(np.sum(vals <= thr_15))
                fail_15 = n_tot - ok_15
                ok_20 = int(np.sum(vals <= thr_20))
                fail_20 = n_tot - ok_20

                or_15, fp_15 = stats.fisher_exact([[fail_15, ok_15], [b_fail_15, b_ok_15]])
                or_20, fp_20 = stats.fisher_exact([[fail_20, ok_20], [b_fail_20, b_ok_20]])
                p_list_15.append(float(fp_15))
                p_list_20.append(float(fp_20))

                temp_recs.append({
                    "device": dev,
                    "metric": metric,
                    "stressor_class": s_cls,
                    "num_workers": nw,
                    "size": sz,
                    "n_tasks": n_tot,
                    "slo_1_5x_base_p95_ms": round(thr_15, 2),
                    "slo_attain_1_5x_pct": round(ok_15 / n_tot * 100.0, 2),
                    "fisher_or_1_5x": _format_fisher_or(float(or_15)),
                    "slo_2_0x_base_p95_ms": round(thr_20, 2),
                    "slo_attain_2_0x_pct": round(ok_20 / n_tot * 100.0, 2),
                    "fisher_or_2_0x": _format_fisher_or(float(or_20)),
                })

            if temp_recs:
                _, padj_15, _, _ = multipletests(p_list_15, alpha=0.05, method="holm")
                _, padj_20, _, _ = multipletests(p_list_20, alpha=0.05, method="holm")
                for rec, p15, p20 in zip(temp_recs, padj_15, padj_20):
                    rec["fisher_p_holm_1_5x"] = float(f"{p15:.4e}")
                    rec["fisher_p_holm_2_0x"] = float(f"{p20:.4e}")
                    rows.append(rec)

    return pd.DataFrame(rows)


def generate_comparative_plots(
    df_desc: pd.DataFrame,
    df_doe_sig: pd.DataFrame,
    df_doe: pd.DataFrame,
    out_dir: Path,
):
    """Genera i grafici riassuntivi DoE: heatmap degradazione (tutte le 4 metriche), importance DoE (4 metriche) e boxplot N_tot."""
    # 1. Heatmap degradazione % rispetto alla Baseline (mediato su S per mem/io) su tutte le 4 metriche
    plot_metrics = ["ttft_ms", "per_token_lat_ms", "throughput_tok_s", "jitter_ms"]
    devices = ["cpu", "gpu"]
    fig, axes = plt.subplots(2, len(plot_metrics), figsize=(22, 8), sharey=True)

    for r_idx, dev in enumerate(devices):
        sub_dev = df_desc[(df_desc["device"] == dev) & (df_desc["stressor_class"] != "none")].copy()
        for c_idx, m in enumerate(plot_metrics):
            ax = axes[r_idx, c_idx]
            piv = (
                sub_dev.groupby(["stressor_class", "num_workers"], as_index=False)[f"{m}_delta_pct"]
                .mean()
                .pivot(index="stressor_class", columns="num_workers", values=f"{m}_delta_pct")
                .reindex(columns=[c for c in N_ORDER if c in sub_dev["num_workers"].unique()])
            )
            cmap = "YlOrRd_r" if m == "throughput_tok_s" else "YlOrRd"
            if piv.empty or piv.isna().all().all():
                # Nessuna run stressor per questo device (es. campagna solo CPU o solo GPU)
                ax.text(0.5, 0.5, f"Nessun dato {dev.upper()}", ha="center", va="center", transform=ax.transAxes)
                ax.set_title(f"{dev.upper()} | {METRIC_LABELS[m]}", fontsize=11, fontweight="bold")
                ax.set_xticks([])
                continue
            sns.heatmap(
                piv,
                annot=True,
                fmt=".1f",
                cmap=cmap,
                ax=ax,
                cbar=True,
                cbar_kws={"label": "Delta % vs Baseline"},
            )
            ax.set_title(f"{dev.upper()} | {METRIC_LABELS[m]}", fontsize=11, fontweight="bold")
            ax.set_xlabel("Carico Worker (N)" if r_idx == 1 else "")
            ax.set_ylabel("Famiglia Stressor" if c_idx == 0 else "")

    fig.suptitle("Variazione Percentuale (%) rispetto alla Baseline — Tutte le Metriche DoE", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "heatmap_degradation_cpu_vs_gpu.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # 2. Stacked Bar dell'Importanza dei Fattori DoE (SS/SST %) per tutte e 4 le metriche
    for target_m in METRICS:
        sub_imp = df_doe_sig[df_doe_sig["metric"] == target_m].copy()
        if sub_imp.empty:
            continue
        sub_imp["term_group"] = sub_imp["term"].replace({
            "num_workers": "Fattore N (Workers)",
            "size": "Fattore S (Size)",
            "num_workers:size": "Interazione N x S",
            "Residual": "Errore Sperimentale",
        })
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
        for idx, dev in enumerate(devices):
            ax = axes[idx]
            piv_imp = (
                sub_imp[sub_imp["device"] == dev]
                .pivot_table(index="stressor_class", columns="term_group", values="importance_pct", aggfunc="sum")
                .fillna(0.0)
            )
            col_order = [c for c in ["Fattore N (Workers)", "Fattore S (Size)", "Interazione N x S", "Errore Sperimentale"] if c in piv_imp.columns]
            piv_imp = piv_imp[col_order]
            if piv_imp.empty:
                ax.text(0.5, 0.5, f"Nessun dato {dev.upper()}", ha="center", va="center", transform=ax.transAxes)
                ax.set_title(f"{dev.upper()} | {METRIC_LABELS[target_m]}", fontsize=11, fontweight="bold")
                continue
            piv_imp.plot(kind="bar", stacked=True, ax=ax, colormap="tab20c", edgecolor="black", linewidth=0.5)
            ax.set_title(f"{dev.upper()} | Varianza Spiegata (SS/SST %) — {METRIC_LABELS[target_m]}", fontsize=11, fontweight="bold")
            ax.set_ylabel("Quota Varianza (%)" if idx == 0 else "")
            ax.set_xlabel("Famiglia Stressor")
            ax.set_ylim(0, 105)
            ax.tick_params(axis="x", rotation=0)
            ax.legend(loc="upper right", fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / f"doe_importance_ss_pct_{target_m}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    # 3. Boxplot confronto diretto Baseline vs N_tot per ogni stressor (CPU vs GPU)
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    sub_n100 = df_doe[(df_doe["stressor_class"] == "none") | (df_doe["num_workers"] == "N_tot")].copy()
    sub_n100["label"] = sub_n100["stressor_class"].replace({"none": "BASELINE"})
    order_lbl = ["BASELINE"] + sorted([x for x in sub_n100["label"].unique() if x != "BASELINE"])

    for ax, m in zip(axes.flat, METRICS):
        sns.boxplot(data=sub_n100, x="label", y=m, hue="device", order=order_lbl, ax=ax, palette="Set2")
        ax.set_title(f"{METRIC_LABELS[m]} — Baseline vs Carico Massimo (N_tot)", fontsize=11, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel(METRIC_LABELS[m])
        ax.grid(True, linestyle="--", alpha=0.5)

    fig.tight_layout()
    fig.savefig(out_dir / "boxplot_baseline_vs_Ntot_cpu_gpu.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_cdf_grid(df_tasks: pd.DataFrame, out_dir: Path, complementary: bool = False):
    """Genera griglia 2x4 ECDF standard o Exceedance Function (1 - CDF) su tutte e 4 le metriche."""
    if df_tasks.empty:
        return

    sub = df_tasks[(df_tasks["stressor_class"] == "none") | (df_tasks["num_workers"] == "N_tot")].copy()
    if sub.empty:
        return

    sub["Stressor"] = sub["stressor_class"].replace({"none": "BASELINE"})
    hue_order = ["BASELINE"] + sorted([s for s in sub["Stressor"].unique() if s != "BASELINE"])

    ecdf_metrics = ["ttft_ms", "per_token_lat_ms", "throughput_tok_s", "jitter_ms"]
    devices = ["cpu", "gpu"]
    fig, axes = plt.subplots(2, len(ecdf_metrics), figsize=(22, 10))

    for r_idx, dev in enumerate(devices):
        df_dev = sub[sub["device"] == dev]
        for c_idx, m in enumerate(ecdf_metrics):
            ax = axes[r_idx, c_idx]
            df_plot = df_dev[df_dev[m] > 0].copy() if m != "throughput_tok_s" else df_dev
            if df_plot.empty:
                ax.text(0.5, 0.5, f"Nessun dato {dev.upper()}", ha="center", va="center", transform=ax.transAxes)
                ax.set_title(f"{dev.upper()} | {METRIC_LABELS[m]}", fontsize=11, fontweight="bold")
                continue
            sns.ecdfplot(
                data=df_plot,
                x=m,
                hue="Stressor",
                hue_order=hue_order,
                complementary=complementary,
                linewidth=1.8,
                ax=ax,
            )
            use_log_x = m != "throughput_tok_s"
            if use_log_x:
                ax.set_xscale("log")
            if complementary:
                ax.set_yscale("log")
                ax.set_ylim(1e-3, 1.05)
                ax.axhline(0.05, color="gray", linestyle="--", linewidth=1.0, alpha=0.8)
                ax.axhline(0.01, color="black", linestyle=":", linewidth=1.1, alpha=0.85)
                x_txt = ax.get_xlim()[0] * (1.05 if use_log_x else 1.0)
                ax.text(x_txt, 0.055, "p95 (5%)", fontsize=8, color="gray")
                ax.text(x_txt, 0.0115, "p99 (1%)", fontsize=8, color="black")
                ax.set_title(f"{dev.upper()} | Exceedance (1 - CDF) {METRIC_LABELS[m]}", fontsize=11, fontweight="bold")
                ax.set_ylabel("P(X > x) [1 - CDF]" if c_idx == 0 else "")
            else:
                ax.axhline(0.95, color="gray", linestyle="--", linewidth=1.0, alpha=0.8)
                ax.axhline(0.99, color="black", linestyle=":", linewidth=1.1, alpha=0.85)
                x_txt = ax.get_xlim()[0] * (1.05 if use_log_x else 1.0)
                ax.text(x_txt, 0.955, "p95", fontsize=8, color="gray")
                ax.text(x_txt, 0.992, "p99", fontsize=8, color="black")
                ax.set_title(f"{dev.upper()} | ECDF {METRIC_LABELS[m]} (Baseline vs N_tot)", fontsize=11, fontweight="bold")
                ax.set_ylabel("Probabilita Cumulativa" if c_idx == 0 else "")
            ax.set_xlabel(f"{METRIC_LABELS[m]} ({'scala log' if use_log_x else 'scala lineare'})")
            ax.grid(True, which="both", linestyle="--", alpha=0.4)

    if complementary:
        fig.suptitle("Exceedance Function (1 - CDF) per Task — Latenze, Throughput e Jitter (pWCET)", fontsize=14, fontweight="bold")
        out_name = "exceedance_1_minus_cdf_cpu_gpu.png"
    else:
        fig.suptitle("Empirical CDF (ECDF) per Task — Latenze, Throughput e Jitter", fontsize=14, fontweight="bold")
        out_name = "ecdf_cpu_vs_gpu_latency.png"
    fig.tight_layout()
    fig.savefig(out_dir / out_name, dpi=300, bbox_inches="tight")
    plt.close(fig)


def generate_ecdf_plots(df_tasks: pd.DataFrame, out_dir: Path):
    """Genera sia l'ECDF standard sia la Exceedance Function (1 - CDF) in scala logaritmica."""
    _plot_cdf_grid(df_tasks, out_dir, complementary=False)
    _plot_cdf_grid(df_tasks, out_dir, complementary=True)


def compute_baseline_normality(df_doe: pd.DataFrame) -> pd.DataFrame:
    """Verifica normalita (Shapiro-Wilk, skewness, kurtosis) sulle ripetizioni della Baseline."""
    rows: List[Dict] = []
    df_base = df_doe[df_doe["stressor_class"] == "none"]
    for dev, sub in df_base.groupby("device"):
        for m in METRICS:
            vals = sub[m].dropna().to_numpy(dtype=float)
            if len(vals) >= 3:
                w_stat, p_val = stats.shapiro(vals)
                sk = float(stats.skew(vals))
                ku = float(stats.kurtosis(vals))
            else:
                w_stat, p_val, sk, ku = np.nan, np.nan, np.nan, np.nan
            rows.append({
                "device": dev,
                "stressor_class": "none",
                "metric": m,
                "n_runs": len(vals),
                "shapiro_w": round(float(w_stat), 4) if pd.notna(w_stat) else np.nan,
                "shapiro_p": float(f"{p_val:.4e}") if pd.notna(p_val) else np.nan,
                "normal_005": bool(pd.notna(p_val) and p_val >= 0.05),
                "skewness": round(sk, 4) if pd.notna(sk) else np.nan,
                "excess_kurtosis": round(ku, 4) if pd.notna(ku) else np.nan,
            })
    return pd.DataFrame(rows)


def generate_residual_diagnostics(df_doe: pd.DataFrame, out_dir: Path):
    """Genera grafici diagnostici dei residui OLS (Residuals vs Fitted e Normal Q-Q Plot) per ogni device e metrica."""
    df_stress = df_doe[df_doe["stressor_class"] != "none"].copy()
    if df_stress.empty:
        return

    for dev in sorted(df_stress["device"].unique()):
        df_dev = df_stress[df_stress["device"] == dev]
        stressors = sorted(df_dev["stressor_class"].unique())
        if not stressors:
            continue

        for m in METRICS:
            n_str = len(stressors)
            nrows = math.ceil(n_str / 2)
            fig, axes = plt.subplots(nrows, 4, figsize=(20, 4.2 * nrows))
            if nrows == 1:
                axes = np.array([axes])

            for idx, s_cls in enumerate(stressors):
                r_idx = idx // 2
                c_base = (idx % 2) * 2
                ax_res = axes[r_idx, c_base]
                ax_qq = axes[r_idx, c_base + 1]

                df_sub = df_dev[df_dev["stressor_class"] == s_cls].copy()
                if "duration" in df_sub.columns and "t_tot" in df_sub["duration"].values:
                    df_sub = df_sub[df_sub["duration"] == "t_tot"].copy()

                valid_sizes = [s for s in df_sub["size"].unique() if s not in ("N/A", "none", "")]
                has_size = len(valid_sizes) > 1
                formula_rhs = "C(num_workers, Sum) * C(size, Sum)" if has_size else "C(num_workers)"
                model_lbl = "2-way Type III" if has_size else "1-way Type II"

                model = smf.ols(f"{m} ~ {formula_rhs}", data=df_sub).fit()
                fitted = model.fittedvalues.to_numpy(dtype=float)
                resid = model.resid.to_numpy(dtype=float)
                r_std = float(np.std(resid, ddof=1)) if len(resid) > 1 else 1.0
                std_resid = resid / r_std if r_std > 0 else resid

                sh_w, sh_p = stats.shapiro(resid)
                cell_cols = ["num_workers", "size"] if has_size else ["num_workers"]
                cell_groups = [g[m].dropna().to_numpy(dtype=float) for _, g in df_sub.groupby(cell_cols)]
                cell_groups = [g for g in cell_groups if len(g) > 0]
                _, lev_p = stats.levene(*cell_groups) if len(cell_groups) >= 2 else (np.nan, np.nan)

                # Residuals vs Fitted
                ax_res.scatter(fitted, std_resid, alpha=0.65, s=22, color="#1f77b4", edgecolors="none")
                ax_res.axhline(0.0, color="red", linestyle="--", linewidth=1.2)
                ax_res.axhline(3.0, color="gray", linestyle=":", linewidth=0.9)
                ax_res.axhline(-3.0, color="gray", linestyle=":", linewidth=0.9)
                ax_res.set_title(f"{s_cls.upper()} ({model_lbl}) | Residui vs Fitted", fontsize=10, fontweight="bold")
                ax_res.set_xlabel(f"Fitted {METRIC_LABELS[m]}", fontsize=9)
                ax_res.set_ylabel("Residui Standardizzati", fontsize=9)
                ax_res.grid(True, linestyle="--", alpha=0.4)

                # Normal Q-Q Plot
                (osm, osr), (slope, intercept, _) = stats.probplot(std_resid, dist="norm")
                ax_qq.scatter(osm, osr, alpha=0.65, s=22, color="#2ca02c", edgecolors="none")
                ax_qq.plot(osm, slope * osm + intercept, color="red", linestyle="--", linewidth=1.2)
                ax_qq.set_title(
                    f"{s_cls.upper()} | Q-Q (Shapiro W={sh_w:.3f}, p={sh_p:.1e} | Lev p={lev_p:.2f})",
                    fontsize=9.5,
                    fontweight="bold",
                )
                ax_qq.set_xlabel("Quantili Teorici Normali", fontsize=9)
                ax_qq.set_ylabel("Residui Ordinati (z)", fontsize=9)
                ax_qq.grid(True, linestyle="--", alpha=0.4)

            # Nasconde eventuali assi inutilizzati se stressors dispari
            for j in range(len(stressors), nrows * 2):
                axes[j // 2, (j % 2) * 2].set_visible(False)
                axes[j // 2, (j % 2) * 2 + 1].set_visible(False)

            fig.suptitle(
                f"{dev.upper()} | Diagnostica Residui OLS (Fitted & Normal Q-Q) — {METRIC_LABELS[m]}",
                fontsize=13,
                fontweight="bold",
            )
            fig.tight_layout()
            fig.savefig(out_dir / f"residuals_diagnostics_{dev}_{m}.png", dpi=250, bbox_inches="tight")
            plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Analisi statistica inferenziale DoE (CPU & GPU).")
    parser.add_argument("--data-dir", type=str, default=str(PROJECT_ROOT / "resultAGX" / "mmlu"), help="Cartella dati radice.")
    parser.add_argument("--model", type=str, default="slm-1b", help="Modello target (default: slm-1b).")
    parser.add_argument("--benchmark", type=str, default="mmlu", help="Benchmark target (default: mmlu).")
    parser.add_argument("--out-dir", type=str, default=None, help="Cartella output tabelle e grafici.")
    parser.add_argument("--max-reps", type=int, default=20, help="Numero massimo di ripetizioni (1..N) per baseline e celle stressor (default: 20; 0=tutte).")
    args = parser.parse_args()

    data_root = Path(args.data_dir)
    out_dir = Path(args.out_dir) if args.out_dir else (PROJECT_ROOT / "figures" / "statistical_reports" / f"{args.model}_{args.benchmark}")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/5] Caricamento dati DoE e Task da: {data_root} (max_reps={args.max_reps})")
    df_doe = load_doe_files(data_root, args.model, args.benchmark, max_reps=args.max_reps)
    if df_doe.empty:
        raise FileNotFoundError(f"Nessun file doe_{args.model}_*_{args.benchmark}_*.csv trovato in {data_root}")
    df_tasks = load_task_files(data_root, args.model, args.benchmark, max_reps=args.max_reps)
    print(f" -> Caricate {len(df_doe)} righe DoE e {len(df_tasks)} righe Task.")

    print("[2/5] Calcolo statistiche descrittive, p95/p99 e Delta% vs Baseline...")
    df_desc = compute_descriptive_and_tail(df_doe, df_tasks)
    df_desc.to_csv(out_dir / "1_descriptive_and_tail_latency.csv", index=False)

    print("[3/5] Analisi DoE (ANOVA 1-way/2-way, SS/SST %, Shapiro-Wilk, Kruskal-Wallis)...")
    df_doe_sig = compute_doe_importance_and_significance(df_doe)
    df_doe_sig.to_csv(out_dir / "2_doe_importance_and_significance.csv", index=False)

    print("[4/5] Test di significativita vs Baseline (Mann-Whitney U + Holm + Cliff's delta)...")
    df_vs_base = compute_baseline_vs_stressor_tests(df_doe)
    df_vs_base.to_csv(out_dir / "3_baseline_vs_stressor_significance.csv", index=False)

    print("[5/7] Correlazioni hardware vmstat <-> metriche LLM...")
    df_corr = compute_hardware_correlations(df_tasks)
    if not df_corr.empty:
        df_corr.to_csv(out_dir / "4_hardware_vmstat_correlations.csv", index=False)

    print("[6/7] Test i.i.d. / Stazionarieta (Ljung-Box, Runs, KPSS, KS) & Post-Hoc (Tukey HSD / MWU-Holm)...")
    df_iid = compute_stationarity_and_iid_tests(df_doe)
    df_iid.to_csv(out_dir / "5_stationarity_and_iid_tests.csv", index=False)
    df_posthoc = compute_posthoc_worker_levels(df_doe)
    df_posthoc.to_csv(out_dir / "6_posthoc_worker_levels.csv", index=False)

    print("[7/7] SLO Attainment (%) + Fisher's Exact Test & Grafici comparativi / ECDF / 1-CDF...")
    df_slo = compute_slo_attainment_and_fisher(df_tasks)
    if not df_slo.empty:
        df_slo.to_csv(out_dir / "7_slo_attainment_and_fisher.csv", index=False)

    df_base_norm = compute_baseline_normality(df_doe)
    if not df_base_norm.empty:
        df_base_norm.to_csv(out_dir / "8_baseline_normality_shapiro.csv", index=False)

    generate_comparative_plots(df_desc, df_doe_sig, df_doe, out_dir)
    generate_ecdf_plots(df_tasks, out_dir)
    generate_residual_diagnostics(df_doe, out_dir)
    print(f"\n[COMPLETATO] Tabelle CSV e grafici statistici salvati in: {out_dir}")


if __name__ == "__main__":
    main()
