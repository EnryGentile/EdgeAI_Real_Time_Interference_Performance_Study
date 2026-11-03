#!/usr/bin/env python3
"""
compute_sample_size.py - Calcolo del numero ottimo di campioni (n) tramite Teorema del Limite Centrale (CLT).

Partendo dai risultati della Baseline (n_0 >= 40 > 30 campioni IID), applica la formula
con valore critico della Z-table (default: Confidenza 95% -> z = 1.96) e margine di errore
relativo massimo r (default: 5% -> r = 0.05):

    n = ceil( ( (z * s_0) / (r * x_bar_0) )^2 )

Protetto con blocchi try/except e meccanismi di fallback:
  1. Ricerca multi-livello del CSV di baseline (con fallback su file parziali o di riepilogo).
  2. Filtraggio difensivo di valori NaN/Inf o righe corrotte e fallback statistico se la varianza è nulla.
  3. Aggiornamento atomico con backup di setup/run_config.json (--update-config).
"""

import os
import sys
import csv
import json
import math
import shutil
import argparse
import traceback
import statistics
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
RESULTS_DIR = PROJECT_ROOT / "results"
CONFIG_PATH = SCRIPT_DIR / "run_config.json"
CONFIG_BACKUP_PATH = SCRIPT_DIR / "run_config.json.bak"
ERROR_LOG_PATH = SCRIPT_DIR / "compute_sample_size_error.log"

# Valori critici bilaterali z_{1 - alpha/2} della distribuzione Normale Standard N(0,1)
Z_TABLE = {
    0.80: 1.2816,
    0.85: 1.4395,
    0.90: 1.6449,
    0.95: 1.9600,
    0.98: 2.3263,
    0.99: 2.5758,
}

# Metriche target del modello per il dimensionamento del campione (CLT):
# TTFT (Time To First Token), TPOT (Time Per Output Token), Throughput (token/s).
# La durata totale del task NON viene usata per dimensionare n poiché dipende
# arbitrariamente dal numero di token generati da ciascuna risposta specifica.
TARGET_LLM_METRICS = [
    "ttft_ms",
    "per_token_lat_ms",
    "throughput_tok_s",
]

METRIC_LABELS = {
    "ttft_ms": "TTFT (Time To First Token) [ms]",
    "per_token_lat_ms": "TPOT (Time Per Output Token) [ms]",
    "throughput_tok_s": "Throughput [tok/s]",
}


def log_error(context: str, exc: BaseException):
    try:
        ERROR_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(ERROR_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(f"\n=== ERRORE RUNTIME [{context}] ===\n")
            f.write("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))
            f.write("\n")
    except Exception:
        pass


def get_z_value(confidence: float) -> float:
    """Restituisce il quantile z_{1 - alpha/2} dalla Z-table con fallback a 1.96 (95%)."""
    try:
        conf_norm = confidence / 100.0 if confidence > 1.0 else confidence
        if not (0.50 <= conf_norm < 1.0):
            raise ValueError(f"Livello di confidenza fuori intervallo valido (0.50, 1.0): {confidence}")
        conf_round = round(conf_norm, 2)
        if conf_round in Z_TABLE:
            return Z_TABLE[conf_round]
        from statistics import NormalDist
        return NormalDist(0, 1).inv_cdf(1.0 - (1.0 - conf_norm) / 2.0)
    except Exception as e:
        log_error("get_z_value", e)
        print(f" [FALLBACK] Livello di confidenza '{confidence}' non valido ({e}). Utilizzato fallback 95% (z = 1.9600).")
        return 1.9600


def compute_metric_sample_size(values: List[float], z_val: float, rel_error: float) -> Optional[Dict[str, float]]:
    """
    Calcola media campionaria (x_bar), deviazione standard campionaria (s),
    coefficiente di variazione (CV = s / x_bar) e il numero di campioni richiesto n.
    """
    try:
        clean_vals = [v for v in values if v is not None and math.isfinite(v)]
        count = len(clean_vals)
        if count < 2:
            return None

        x_bar = statistics.mean(clean_vals)
        s_std = statistics.stdev(clean_vals)

        if abs(x_bar) < 1e-12:
            return {
                "n_0": count,
                "mean": 0.0,
                "stdev": round(s_std, 4),
                "cv_pct": 0.0,
                "n_raw": 0.0,
                "n_required": 1,
            }

        safe_err = rel_error if rel_error > 1e-6 else 0.05
        cv = s_std / abs(x_bar)
        n_raw = ((z_val * s_std) / (safe_err * abs(x_bar))) ** 2
        n_req = max(1, math.ceil(n_raw))

        return {
            "n_0": count,
            "mean": round(x_bar, 4),
            "stdev": round(s_std, 4),
            "cv_pct": round(cv * 100.0, 2),
            "n_raw": round(n_raw, 4),
            "n_required": n_req,
        }
    except Exception as e:
        log_error("compute_metric_sample_size", e)
        return None


def resolve_baseline_csv(
    csv_arg: Optional[str],
    model_arg: Optional[str],
    device_arg: Optional[str],
    benchmark_arg: Optional[str]
) -> Path:
    """Individua il file CSV della baseline da analizzare con catena di fallback."""
    if csv_arg:
        p = Path(csv_arg)
        if p.exists():
            return p
        print(f" [WARNING] Il file CSV specificato '{p}' non esiste. Ricerca automatica di un CSV di fallback...")

    model_id = "slm-1b"
    device = "gpu"
    bench = "mmlu"

    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            model_id = cfg.get("model_id", "SLM-1B").lower()
            device = cfg.get("device", "gpu").lower()
            bench = cfg.get("benchmark_name", "mmlu").lower()
        except Exception as e:
            log_error("read_config_in_resolve_csv", e)

    if model_arg:
        m = model_arg.strip().lower()
        model_id = "slm-1b" if m in ("slm", "1b", "slm-1b") else "llm-8b"
    if device_arg:
        device = device_arg.strip().lower()
    if benchmark_arg:
        bench = benchmark_arg.strip().lower()

    candidate = RESULTS_DIR / f"baseline_{model_id}_{device}_{bench}_tasks.csv"
    if candidate.exists():
        return candidate

    # Fallback 1: qualsiasi baseline_*_tasks.csv
    if RESULTS_DIR.exists():
        all_baselines = sorted(RESULTS_DIR.glob("baseline_*_tasks.csv"))
        if all_baselines:
            print(f" [FALLBACK] File '{candidate.name}' non trovato. Utilizzato: {all_baselines[0]}")
            return all_baselines[0]

        # Fallback 2: file parziale generato dall'emergency fallback di run_experiments.py
        partial_baselines = sorted(RESULTS_DIR.glob("fallback_partial_baseline_*_tasks.csv"))
        if partial_baselines:
            print(f" [FALLBACK] Utilizzato file baseline parziale di emergenza: {partial_baselines[0]}")
            return partial_baselines[0]

    print(
        f"\n[ERRORE] Nessun file di baseline trovato (atteso: {candidate}).\n"
        "Esegui prima la baseline (40 task senza stressor) con:\n"
        f"  python setup/run_experiments.py --model {model_id} --device {device} --benchmark {bench} --no-stressor --samples 40"
    )
    sys.exit(1)


def load_csv_columns(
    csv_path: Path,
    repetition_filter: Optional[int] = None,
    aggregate_mode: str = "median",
    warmup_drop: int = 3
) -> Tuple[int, Dict[str, List[float]], Dict[str, Any]]:
    """
    Carica le colonne numeriche dal CSV della baseline.
    Se sono presenti ripetizioni multiple (es. rep 1, 2, 3 per gli stessi 40 compiti)
    e non è specificato repetition_filter, aggrega i valori di ogni singolo task
    tramite la 'median' (default) o 'mean' per eliminare il rumore inter-run e gli outlier.
    Se warmup_drop > 0, esclude i primi compiti di cold-start hardware per valutare lo steady-state.
    """
    skip_cols = {
        "task_index", "sample_id", "exp_id", "model_id", "device", "benchmark",
        "stressor_class", "stressor_method", "num_workers", "size",
        "duration", "repetition", "timestamp_s", "ground_truth", "model_response"
    }

    raw_rows: List[Dict[str, Any]] = []
    repetitions_seen = set()

    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if not row:
                    continue
                if "repetition" in row and row["repetition"]:
                    try:
                        rep_val = int(row["repetition"])
                        repetitions_seen.add(rep_val)
                        if repetition_filter is not None and rep_val != repetition_filter:
                            continue
                    except ValueError:
                        pass
                elif repetition_filter is not None:
                    continue
                raw_rows.append(row)
    except Exception as e:
        log_error("load_csv_columns_read", e)
        raise RuntimeError(f"Impossibile leggere il file CSV {csv_path}: {e}") from e

    num_reps = len(repetitions_seen) if repetitions_seen else 1
    do_aggregation = (repetition_filter is None and num_reps > 1 and aggregate_mode in ("median", "mean"))

    columns: Dict[str, List[float]] = {}
    meta: Dict[str, Any] = {
        "raw_rows": len(raw_rows),
        "repetitions_detected": num_reps,
        "aggregate_applied": aggregate_mode if do_aggregation else "none",
        "warmup_discarded": 0,
    }

    if do_aggregation:
        # Raggruppamento per ID del task (sample_id o task_index)
        grouped_by_sample: Dict[str, Dict[str, List[float]]] = {}
        for r in raw_rows:
            s_id = r.get("sample_id") or r.get("task_index") or str(len(grouped_by_sample) + 1)
            sample_dict = grouped_by_sample.setdefault(s_id, {})
            for k, val_str in r.items():
                if k is None or k in skip_cols or val_str is None:
                    continue
                try:
                    val = float(val_str)
                    if math.isfinite(val):
                        sample_dict.setdefault(k, []).append(val)
                except (ValueError, TypeError):
                    continue

        # Esclusione del transitorio iniziale di warmup (cold-start)
        if warmup_drop > 0 and len(grouped_by_sample) > warmup_drop:
            sorted_keys = sorted(
                grouped_by_sample.keys(),
                key=lambda x: int(x) if str(x).isdigit() else str(x)
            )
            for k in sorted_keys[:warmup_drop]:
                del grouped_by_sample[k]
            meta["warmup_discarded"] = warmup_drop

        effective_count = len(grouped_by_sample)
        meta["effective_samples"] = effective_count

        for s_id, metrics in grouped_by_sample.items():
            for k, vals in metrics.items():
                if not vals:
                    continue
                agg_val = statistics.median(vals) if aggregate_mode == "median" else statistics.mean(vals)
                columns.setdefault(k, []).append(agg_val)

        return effective_count, columns, meta

    # Modalità raw senza aggregazione (es. singola ripetizione o aggregazione disattivata)
    if warmup_drop > 0 and len(raw_rows) > warmup_drop:
        raw_rows = raw_rows[warmup_drop:]
        meta["warmup_discarded"] = warmup_drop

    for r in raw_rows:
        for k, val_str in r.items():
            if k is None or k in skip_cols or val_str is None:
                continue
            try:
                val = float(val_str)
                if math.isfinite(val):
                    columns.setdefault(k, []).append(val)
            except (ValueError, TypeError):
                continue

    effective_count = len(raw_rows)
    meta["effective_samples"] = effective_count
    return effective_count, columns, meta


def update_run_config_safely(recommended_n: int):
    """Aggiorna sample_count in setup/run_config.json con scrittura atomica e backup."""
    if not CONFIG_PATH.exists():
        print(f"\n [INFO] {CONFIG_PATH} non ancora presente; usa '--samples {recommended_n}' al lancio di run_experiments.py.")
        return

    try:
        shutil.copy2(CONFIG_PATH, CONFIG_BACKUP_PATH)
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        cfg["sample_count"] = int(recommended_n)

        tmp_path = CONFIG_PATH.with_suffix(".json.tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        tmp_path.replace(CONFIG_PATH)
        print(f"\n [OK] Aggiornato 'sample_count' = {recommended_n} in {CONFIG_PATH}")
    except Exception as e:
        log_error("update_run_config_safely", e)
        print(f"\n [FALLBACK] Errore durante l'aggiornamento di {CONFIG_PATH}: {e}")
        if CONFIG_BACKUP_PATH.exists():
            shutil.copy2(CONFIG_BACKUP_PATH, CONFIG_PATH)
            print(f" [FALLBACK] Ripristinato {CONFIG_PATH} dal backup {CONFIG_BACKUP_PATH}.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Calcolo del numero di campioni (n) dalla Baseline tramite CLT e Z-table (default: IC 95%, Errore 5%)."
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Percorso al CSV della baseline (es. results/baseline_slm-1b_gpu_mmlu_tasks.csv)"
    )
    parser.add_argument("-m", "--model", type=str, default=None, help="Modello della baseline ('slm' o 'llm')")
    parser.add_argument("-d", "--device", type=str, choices=["cpu", "gpu", "CPU", "GPU"], default=None, help="Allocazione ('cpu' o 'gpu')")
    parser.add_argument("-b", "--benchmark", type=str, choices=["mmlu", "spider", "MMLU", "SPIDER"], default=None, help="Benchmark ('mmlu' o 'spider')")
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.95,
        help="Livello di confidenza (default: 0.95 per 95%% -> z = 1.96)"
    )
    parser.add_argument(
        "--error",
        type=float,
        default=0.05,
        help="Margine di errore relativo massimo (default: 0.05 per 5%%)"
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=5,
        help="Numero minimo di campioni di sicurezza da garantire per run sotto stressor (default: 5)"
    )
    parser.add_argument(
        "--repetition",
        type=int,
        default=None,
        help="Filtra una specifica ripetizione della baseline (es. 1 per valutare solo i primi 40 campioni)"
    )
    parser.add_argument(
        "--aggregate",
        type=str,
        choices=["median", "mean", "none"],
        default="median",
        help="Metodo di aggregazione per le ripetizioni multiple dello stesso task (default: 'median' per de-noising robusto agli outlier)"
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=3,
        help="Numero di compiti iniziali di cold-start hardware da escludere per valutare lo steady-state (default: 3)"
    )
    parser.add_argument(
        "--update-config",
        action="store_true",
        default=False,
        help="Aggiorna automaticamente 'sample_count' in setup/run_config.json con il valore n calcolato"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    try:
        conf = args.confidence / 100.0 if args.confidence > 1.0 else args.confidence
        err = args.error / 100.0 if args.error > 1.0 else args.error
        if err <= 0 or err >= 1.0:
            print(f" [FALLBACK] Margine di errore '{args.error}' fuori range. Impostato fallback al 5% (0.05).")
            err = 0.05

        z_val = get_z_value(conf)
        csv_path = resolve_baseline_csv(args.csv, args.model, args.device, args.benchmark)
        n_0, cols, meta = load_csv_columns(
            csv_path,
            repetition_filter=args.repetition,
            aggregate_mode=args.aggregate,
            warmup_drop=args.warmup
        )

        if n_0 < 2 or not cols:
            raise ValueError(
                f"Il file {csv_path} contiene troppi pochi campioni validi (n_0={n_0}) per calcolare la deviazione standard."
            )

        agg_desc = (
            f"mediana per-task su {meta['repetitions_detected']} ripetizioni (de-noising outlier)"
            if meta["aggregate_applied"] == "median"
            else (f"media per-task su {meta['repetitions_detected']} ripetizioni" if meta["aggregate_applied"] == "mean" else f"nessuna ({meta['raw_rows']} righe totali)")
        )

        warm_msg = f" [esclusi {meta['warmup_discarded']} task iniziali di warmup/cold-start]" if meta.get("warmup_discarded") else ""

        print("\n" + "=" * 78)
        print(" CALCOLO DIMENSIONAMENTO CAMPIONE (CLT + Z-TABLE) PER IL PIANO FATTORIALE")
        print("=" * 78)
        print(f" * File Baseline analizzato : {csv_path}")
        print(f" * Compiti Baseline (n_0)   : {n_0} prompt a regime ({meta['raw_rows']} record totali{warm_msg})")
        print(f" * Aggregazione Ripetizioni : {agg_desc}")
        print(f" * Teorema Limite Centrale  : {'[OK >= 30 campioni: CLT valido]' if n_0 >= 30 else '[WARNING < 30 campioni: raccomandati >= 30]'}")
        print(f" * Intervallo di Confidenza : {conf * 100:.1f}% (alpha = {1.0 - conf:.3f} -> z_{{1-alpha/2}} = {z_val:.4f})")
        print(f" * Errore Relativo Massimo  : {err * 100:.1f}% (r = {err:.4f})")
        print(f" * Metriche di Riferimento  : TTFT, TPOT (Latenza per-token) e Throughput (esclusa durata)")
        print(f" * Formula applicata        : n = ceil( ( ({z_val:.2f} * s_0) / ({err:.2f} * x_bar_0) )^2 )")
        print("=" * 78)

        results_summary: Dict[str, Dict[str, float]] = {}
        for col_name, values in cols.items():
            res = compute_metric_sample_size(values, z_val=z_val, rel_error=err)
            if res is not None:
                results_summary[col_name] = res

        print("\n1. METRICHE INTRINSECHE DEL MODELLO LLM (Dimensionamento campioni n):")
        print(f" {'Metrica':<36s} | {'n_0':>5s} | {'Media (x_bar)':>14s} | {'Dev.Std (s_0)':>14s} | {'CV (%)':>8s} | {'n calcolato':>11s}")
        print("-" * 102)

        target_n_candidates = []

        for m in TARGET_LLM_METRICS:
            if m in results_summary:
                r = results_summary[m]
                label = METRIC_LABELS.get(m, m)
                print(
                    f" {label:<36s} | {r['n_0']:5d} | {r['mean']:14.4f} | "
                    f"{r['stdev']:14.4f} | {r['cv_pct']:7.2f}% | {r['n_required']:11d}"
                )
                target_n_candidates.append(r["n_required"])

        if target_n_candidates:
            recommended_raw = max(target_n_candidates)
            basis_str = "Massimo tra le 3 metriche core (TTFT, TPOT, Throughput)"
        else:
            recommended_raw = 10
            basis_str = "Fallback standard"

        recommended_n = max(max(1, args.min_samples), recommended_raw)

        print("\n" + "=" * 78)
        print(f" -> NUMERO DI CAMPIONI CONSIGLIATO PER GLI STRESSOR: n = {recommended_n}")
        print(f"    (Calcolato su: {basis_str}, confidenza {conf*100:.0f}%, errore {err*100:.0f}%, min_samples={args.min_samples})")
        print("=" * 78)

        if args.update_config:
            update_run_config_safely(recommended_n)

        print("\nComando pronto per lanciare gli esperimenti con gli stressor usando questo n:")
        print(f"  python setup/run_experiments.py --use-stressor --stressor <cpu|mem|io|cache|sched|interrupt> --samples {recommended_n}\n")

    except SystemExit:
        raise
    except Exception as exc:
        log_error("main_fallback", exc)
        fallback_n = max(30, getattr(args, "min_samples", 5))
        print(f"\n[EMERGENCY FALLBACK] Errore durante l'elaborazione statistica: {exc}")
        print(f"[FALLBACK] Impostato valore cautelativo da Teorema del Limite Centrale: n = {fallback_n}")
        if getattr(args, "update_config", False):
            update_run_config_safely(fallback_n)
        print(f"Dettagli dell'errore salvati in: {ERROR_LOG_PATH}")
        sys.exit(1)


if __name__ == "__main__":
    main()
