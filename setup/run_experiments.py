#!/usr/bin/env python3
"""
run_experiments.py - Orchestratore di esecuzione Benchmark, Stressor, vmstat e Reboot (Traccia A1).

Funzionalità principali:
  1. Legge setup/run_config.json (e permette di selezionare/sovrascrivere da CLI:
     --model, --device, --benchmark, --use-stressor / --no-stressor, --stressor, --samples, --repetitions).
  2. Gestisce una coda persistente su disco (setup/experiment_state.json) per sopravvivere
     ai riavvii della board (sudo reboot al termine di ogni esperimento/ripetizione).
  3. All'avvio attende il tempo di esaurimento del transitorio post-boot (default: 90s per Jetson Orin)
     e pulisce la page cache di Linux.
  4. Avvia in background 'vmstat 1' + lettura sensori termici CPU/GPU e throttling.
  5. Esegue gli N campioni (task) del benchmark con llama-cli (e stress-ng se attivo),
     sincronizzando anche le finestre temporali di vmstat con ciascun singolo task.
  6. Salva su disco (con fsync) le serie temporali vmstat, i campioni per task e il riepilogo DoE,
     e invoca il reboot della board per proseguire con l'esperimento successivo.
"""

import os
import re
import sys
import csv
import time
import math
import json
import glob
import shutil
import signal
import argparse
import threading
import subprocess
import statistics
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from setup_run import (
    CONFIG_OUTPUT_PATH,
    instantiate_run,
    format_mmlu_prompt,
    format_spider_prompt,
)

STATE_FILE_PATH = SCRIPT_DIR / "experiment_state.json"
TIMESERIES_DIR = PROJECT_ROOT / "data_timeseries"
RESULTS_DIR = PROJECT_ROOT / "results"
MASTER_SUMMARY_CSV = PROJECT_ROOT / "master_experiment_summary.csv"

# Livelli del Piano Fattoriale (DoE)
N_LEVELS = [
    ("N25", "25% N_tot", 0.25),
    ("N50", "50% N_tot", 0.50),
    ("N75", "75% N_tot", 0.75),
    ("N100", "N_tot", 1.00),
]

S_LEVELS = [
    ("S25", "25% S_tot", 0.25),
    ("S50", "50% S_tot", 0.50),
    ("S75", "75% S_tot", 0.75),
]

D_LEVELS = ["t_prefill", "t_decode", "t_tot"]

VMSTAT_COLUMNS = [
    "timestamp_s", "r", "b", "swpd_kb", "free_kb", "buff_kb", "cache_kb",
    "si", "so", "bi", "bo", "in", "cs", "us", "sy", "id", "wa", "st",
    "cpu_temp_c", "gpu_temp_c", "throttled"
]


# ==============================================================================
# MONITORAGGIO HARDWARE IN BACKGROUND (vmstat 1 + Sensori Termici Jetson Orin)
# ==============================================================================
class SystemMonitor:
    """
    Raccoglie in background a 1 Hz le statistiche di vmstat e le metriche termiche
    della board (es. NVIDIA Jetson AGX Orin / Linux SBC).
    """

    def __init__(self):
        self.samples: List[Dict[str, float]] = []
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._start_monotonic = 0.0

    @staticmethod
    def _read_thermal_zones() -> Tuple[float, float, int]:
        """Legge le temperature CPU/GPU da /sys/class/thermal e lo stato di throttling."""
        cpu_temps = []
        gpu_temps = []
        throttled = 0

        for tz_dir in glob.glob("/sys/class/thermal/thermal_zone*"):
            try:
                type_path = os.path.join(tz_dir, "type")
                temp_path = os.path.join(tz_dir, "temp")
                if not os.path.exists(temp_path):
                    continue
                with open(temp_path, "r") as f:
                    raw_temp = float(f.read().strip())
                temp_c = raw_temp / 1000.0 if raw_temp > 200 else raw_temp

                tz_type = ""
                if os.path.exists(type_path):
                    with open(type_path, "r") as f:
                        tz_type = f.read().strip().lower()

                if any(k in tz_type for k in ["gpu", "gv11b", "ga10b"]):
                    gpu_temps.append(temp_c)
                elif any(k in tz_type for k in ["cpu", "soc", "cv", "tj", "core"]):
                    cpu_temps.append(temp_c)
                else:
                    cpu_temps.append(temp_c)
            except Exception:
                continue

        # Check eventuale throttling su sysfs (Tegra / cpufreq)
        for th_path in glob.glob("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq"):
            try:
                max_path = th_path.replace("scaling_cur_freq", "cpuinfo_max_freq")
                if os.path.exists(max_path):
                    with open(th_path, "r") as f1, open(max_path, "r") as f2:
                        cur_f = float(f1.read().strip())
                        max_f = float(f2.read().strip())
                    if max_f > 0 and (cur_f / max_f) < 0.80 and (cpu_temps and max(cpu_temps) > 75.0):
                        throttled = 1
            except Exception:
                pass

        cpu_t = round(sum(cpu_temps) / len(cpu_temps), 2) if cpu_temps else 0.0
        gpu_t = round(sum(gpu_temps) / len(gpu_temps), 2) if gpu_temps else cpu_t
        return cpu_t, gpu_t, throttled

    def _collector_loop(self):
        self._start_monotonic = time.monotonic()
        vmstat_path = shutil.which("vmstat")

        if vmstat_path:
            proc = subprocess.Popen(
                ["vmstat", "-n", "1"],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                bufsize=1
            )
            try:
                idx = 0
                for line in iter(proc.stdout.readline, ""):
                    if self._stop_event.is_set():
                        break
                    line_s = line.strip()
                    if not line_s or line_s.startswith("procs") or line_s.startswith("r"):
                        continue
                    parts = line_s.split()
                    if len(parts) < 16:
                        continue
                    try:
                        vals = [int(x) for x in parts[:17]]
                    except ValueError:
                        continue

                    idx += 1
                    now_mono = time.monotonic()
                    cpu_t, gpu_t, throttled = self._read_thermal_zones()

                    row = {
                        "timestamp_s": idx,
                        "_mono_ts": now_mono,
                        "r": vals[0],
                        "b": vals[1],
                        "swpd_kb": vals[2],
                        "free_kb": vals[3],
                        "buff_kb": vals[4],
                        "cache_kb": vals[5],
                        "si": vals[6],
                        "so": vals[7],
                        "bi": vals[8],
                        "bo": vals[9],
                        "in": vals[10],
                        "cs": vals[11],
                        "us": vals[12],
                        "sy": vals[13],
                        "id": vals[14],
                        "wa": vals[15],
                        "st": vals[16] if len(vals) > 16 else 0,
                        "cpu_temp_c": cpu_t,
                        "gpu_temp_c": gpu_t,
                        "throttled": throttled,
                    }
                    with self._lock:
                        self.samples.append(row)
            finally:
                proc.terminate()
                try:
                    proc.wait(timeout=2)
                except Exception:
                    proc.kill()
        else:
            # Fallback se eseguito su ambiente privo di vmstat (es. Windows test)
            idx = 0
            while not self._stop_event.is_set():
                time.sleep(1.0)
                idx += 1
                now_mono = time.monotonic()
                cpu_t, gpu_t, throttled = self._read_thermal_zones()
                row = {col: 0 for col in VMSTAT_COLUMNS}
                row["timestamp_s"] = idx
                row["_mono_ts"] = now_mono
                row["cpu_temp_c"] = cpu_t
                row["gpu_temp_c"] = gpu_t
                row["throttled"] = throttled
                with self._lock:
                    self.samples.append(row)

    def start(self):
        self.samples.clear()
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._collector_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)

    def get_window_average(self, t_start: float, t_end: float) -> Dict[str, float]:
        """
        Restituisce la media dei campioni vmstat caduti nella finestra [t_start, t_end]
        di un singolo task (per allineare 1:1 i campioni vmstat con i task di llama.cpp).
        Se il task è durato meno di 1s, prende il campione più vicino nel tempo.
        """
        with self._lock:
            if not self.samples:
                return {col: 0.0 for col in VMSTAT_COLUMNS if col != "timestamp_s"}

            window = [s for s in self.samples if t_start <= s["_mono_ts"] <= t_end]
            if not window:
                mid = 0.5 * (t_start + t_end)
                closest = min(self.samples, key=lambda s: abs(s["_mono_ts"] - mid))
                window = [closest]

            avg = {}
            for col in VMSTAT_COLUMNS:
                if col == "timestamp_s":
                    continue
                avg[col] = round(sum(float(x[col]) for x in window) / len(window), 4)
            return avg

    def save_timeseries_csv(self, filepath: Path):
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            rows = list(self.samples)

        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=VMSTAT_COLUMNS)
            writer.writeheader()
            for r in rows:
                clean_row = {k: r[k] for k in VMSTAT_COLUMNS}
                writer.writerow(clean_row)
            f.flush()
            os.fsync(f.fileno())


# ==============================================================================
# GESTIONE STRESSOR (stress-ng)
# ==============================================================================
def get_total_ram_mb() -> int:
    """Rileva la RAM totale del sistema in MB da /proc/meminfo (default 16384 MB)."""
    try:
        with open("/proc/meminfo", "r") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    kb = int(line.split()[1])
                    return max(512, kb // 1024)
    except Exception:
        pass
    return 16384


def build_stress_ng_cmd(
    stressor_class: str,
    stressor_method: str,
    n_fraction: float,
    s_fraction: Optional[float],
    total_cores: int
) -> List[str]:
    """Costruisce il comando stress-ng in base a classe, metodo, livello N e dimensione S."""
    workers = max(1, int(round(total_cores * n_fraction)))
    cmd = ["stress-ng", "--metrics-brief"]

    if stressor_class == "cpu":
        if stressor_method == "matrix":
            cmd += ["--matrix", str(workers)]
        elif stressor_method == "fft":
            cmd += ["--fft", str(workers)]
        else:
            cmd += ["--cpu", str(workers)]

    elif stressor_class == "mem":
        ram_mb = get_total_ram_mb()
        s_frac = s_fraction if s_fraction is not None else 0.50
        # Scala sul pool di memoria disponibile per evitare OOM killer
        alloc_mb_total = max(64, int(ram_mb * 0.50 * s_frac))
        alloc_per_worker = max(16, alloc_mb_total // workers)
        if stressor_method == "stream":
            cmd += ["--stream", str(workers), "--stream-l3-size", f"{alloc_per_worker}M"]
        elif stressor_method == "memrate":
            cmd += ["--memrate", str(workers), "--memrate-bytes", f"{alloc_per_worker}M"]
        else:
            cmd += ["--vm", str(workers), "--vm-bytes", f"{alloc_per_worker}M"]

    elif stressor_class == "cache":
        if stressor_method == "l1cache":
            cmd += ["--l1cache", str(workers)]
        else:
            cmd += ["--cache", str(workers)]

    elif stressor_class == "io":
        s_frac = s_fraction if s_fraction is not None else 0.50
        file_mb = max(16, int(512 * s_frac))
        if stressor_method == "hdd":
            cmd += ["--hdd", str(workers), "--hdd-bytes", f"{file_mb}M"]
        else:
            cmd += ["--iomix", str(workers), "--iomix-bytes", f"{file_mb}M"]

    elif stressor_class == "sched":
        if stressor_method == "fork":
            cmd += ["--fork", str(workers)]
        else:
            cmd += ["--switch", str(workers)]

    elif stressor_class == "interrupt":
        if stressor_method == "sigq":
            cmd += ["--sigq", str(workers)]
        else:
            cmd += ["--timer", str(workers)]

    else:
        cmd += ["--cpu", str(workers)]

    return cmd


def stop_process_tree(proc: Optional[subprocess.Popen]):
    """Termina in modo pulito stress-ng e tutti i suoi worker figli."""
    if proc is None or proc.poll() is not None:
        return
    try:
        if os.name != "nt":
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        else:
            proc.terminate()
        proc.wait(timeout=3)
    except Exception:
        try:
            if os.name != "nt":
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            else:
                proc.kill()
        except Exception:
            pass


# ==============================================================================
# ESECUZIONE SINGOLO TASK CON llama-cli E PARSING METRICHE
# ==============================================================================
PROMPT_EVAL_RE = re.compile(
    r"prompt eval time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*tokens\s*\(\s*([\d.]+)\s*ms per token,\s*([\d.]+)\s*tokens per second\)",
    re.IGNORECASE
)
EVAL_TIME_RE = re.compile(
    r"(?<!prompt )eval time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*(?:runs|tokens)\s*\(\s*([\d.]+)\s*ms per token,\s*([\d.]+)\s*tokens per second\)",
    re.IGNORECASE
)


def run_single_llama_task(
    config: Dict[str, Any],
    prompt: str,
    stress_cmd: Optional[List[str]] = None,
    duration_mode: str = "t_tot"
) -> Dict[str, float]:
    """
    Esegue un singolo task su llama-cli, sincronizzando l'eventuale stressor con la fase
    ('t_prefill', 't_decode', 't_tot') e misurando TTFT, Per-Token Latency, Throughput e Jitter.
    """
    cmd = [
        config["llama_cli_path"],
        "-m", config["model_path"],
        "-ngl", str(config["n_gpu_layers"]),
        "-t", str(config["n_threads"]),
        "--temp", str(config["temperature"]),
        "-s", str(config.get("seed", 42)),
        "-n", str(config["max_tokens"]),
        "--no-display-prompt",
        "-p", prompt
    ]

    stress_proc: Optional[subprocess.Popen] = None
    preexec = os.setsid if os.name != "nt" else None

    t_start_mono = time.monotonic()

    # Se lo stressor deve agire in t_prefill o t_tot, parte all'inizio del task
    if stress_cmd and duration_mode in ("t_prefill", "t_tot"):
        stress_proc = subprocess.Popen(
            stress_cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            preexec_fn=preexec
        )

    token_timestamps: List[float] = []
    first_token_seen = False

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0
        )

        stderr_chunks: List[bytes] = []

        def _read_stderr():
            while True:
                chunk = proc.stderr.read(1024)
                if not chunk:
                    break
                stderr_chunks.append(chunk)

        stderr_thread = threading.Thread(target=_read_stderr, daemon=True)
        stderr_thread.start()

        # Lettura stream stdout byte/chunk per rilevare l'istante esatto del 1° token (fine prefill)
        # e i delta temporali inter-token per il calcolo del jitter
        while True:
            b = proc.stdout.read(16)
            if not b:
                break
            now_t = time.perf_counter()
            token_timestamps.append(now_t)

            if not first_token_seen:
                first_token_seen = True
                if stress_cmd and duration_mode == "t_prefill":
                    stop_process_tree(stress_proc)
                    stress_proc = None
                elif stress_cmd and duration_mode == "t_decode":
                    stress_proc = subprocess.Popen(
                        stress_cmd,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        preexec_fn=preexec
                    )

        proc.wait()
        stderr_thread.join(timeout=2.0)
        stderr_text = b"".join(stderr_chunks).decode("utf-8", errors="replace")

    finally:
        stop_process_tree(stress_proc)

    t_end_mono = time.monotonic()

    # Parsing metriche ufficiali da stderr di llama.cpp
    ttft_ms = 0.0
    prompt_tokens = 0
    prompt_ms_per_tok = 0.0
    eval_total_ms = 0.0
    eval_tokens = 0
    per_token_lat_ms = 0.0
    throughput_tok_s = 0.0

    m_prompt = PROMPT_EVAL_RE.search(stderr_text)
    if m_prompt:
        ttft_ms = float(m_prompt.group(1))
        prompt_tokens = int(m_prompt.group(2))
        prompt_ms_per_tok = float(m_prompt.group(3))

    m_eval = EVAL_TIME_RE.search(stderr_text)
    if m_eval:
        eval_total_ms = float(m_eval.group(1))
        eval_tokens = int(m_eval.group(2))
        per_token_lat_ms = float(m_eval.group(3))
        throughput_tok_s = float(m_eval.group(4))

    # Calcolo dell'Inter-Token Jitter (deviazione standard degli inter-arrival times in decode)
    jitter_ms = 0.0
    if len(token_timestamps) >= 3:
        inter_token_deltas_ms = [
            (token_timestamps[i] - token_timestamps[i - 1]) * 1000.0
            for i in range(2, len(token_timestamps))
        ]
        if len(inter_token_deltas_ms) >= 2:
            jitter_ms = statistics.stdev(inter_token_deltas_ms)
        elif len(inter_token_deltas_ms) == 1:
            jitter_ms = abs(inter_token_deltas_ms[0] - per_token_lat_ms)

    return {
        "t_start_mono": t_start_mono,
        "t_end_mono": t_end_mono,
        "duration_s": round(t_end_mono - t_start_mono, 4),
        "prompt_tokens": prompt_tokens,
        "output_tokens": eval_tokens,
        "ttft_ms": round(ttft_ms, 4),
        "prompt_ms_per_tok": round(prompt_ms_per_tok, 4),
        "eval_total_ms": round(eval_total_ms, 4),
        "per_token_lat_ms": round(per_token_lat_ms, 4),
        "throughput_tok_s": round(throughput_tok_s, 4),
        "jitter_ms": round(jitter_ms, 4),
    }


# ==============================================================================
# TRANSITORIO POST-BOOT E REBOOT DELLA BOARD
# ==============================================================================
def wait_for_post_boot_transient(wait_seconds: int):
    """
    Attende che il transitorio del sistema operativo e termico post-boot si esaurisca
    prima di avviare la raccolta delle metriche con vmstat e llama.cpp.
    Su Jetson AGX Orin 90s permettono la stabilizzazione completa dei servizi e della temperatura.
    """
    if wait_seconds <= 0:
        return

    print("\n" + "-" * 65)
    print(f" [TRANSITORIO] Attesa di {wait_seconds}s per stabilizzazione OS e termica...")
    print("-" * 65)

    # Sincronizzazione filesystem ed eventuale pulizia page cache (se root)
    if os.name != "nt":
        try:
            subprocess.run(["sync"], check=False)
            if os.geteuid() == 0:
                with open("/proc/sys/vm/drop_caches", "w") as f:
                    f.write("3\n")
                print(" [OK] Page cache, dentries e inodes rilasciati (drop_caches=3).")
        except Exception:
            pass

    step = 15
    remaining = wait_seconds
    while remaining > 0:
        sleep_chunk = min(step, remaining)
        time.sleep(sleep_chunk)
        remaining -= sleep_chunk
        cpu_t, gpu_t, _ = SystemMonitor._read_thermal_zones()
        print(f"   ... transitorio rimanente: {remaining:3d}s | Temp CPU: {cpu_t:.1f}°C | Temp GPU: {gpu_t:.1f}°C")

    print(" [OK] Transitorio terminato. Sistema stabile e pronto per l'esperimento.\n")


def trigger_board_reboot():
    """Sincronizza i dischi ed esegue il riavvio della board."""
    print("\n" + "=" * 65)
    print(" [REBOOT] Sincronizzazione filesystem e riavvio della board in corso...")
    print("=" * 65)
    if os.name != "nt":
        subprocess.run(["sync"], check=False)
        subprocess.run(["sudo", "reboot"], check=False)
    else:
        print(" [INFO] Ambiente Windows rilevato: simulazione comando 'sudo reboot'.")
    sys.exit(0)


# ==============================================================================
# COSTRUZIONE E GESTIONE DELLA CODA ESPERIMENTI (CHECKPOINT PER REBOOT)
# ==============================================================================
def build_experiment_queue(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Costruisce la lista di esperimenti da eseguire (ciascuno ripetuto per config['repetitions'] volte).
    - Se use_stressor=False: genera le ripetizioni della Baseline (es. 40 campioni).
    - Se use_stressor=True: genera il piano fattoriale per lo stressor scelto x 3 ripetizioni.
    """
    reps = int(config.get("repetitions", 3))
    queue: List[Dict[str, Any]] = []

    if not config.get("use_stressor", False):
        for rep in range(1, reps + 1):
            exp_id = f"baseline_{config['model_id']}_{config['device']}_{config['benchmark_name'].lower()}_rep{rep}"
            queue.append({
                "exp_id": exp_id,
                "mode": "baseline",
                "stressor_class": "none",
                "stressor_method": "none",
                "n_tag": "N0",
                "n_label": "0% N_tot",
                "n_fraction": 0.0,
                "s_tag": "N/A",
                "s_label": "N/A",
                "s_fraction": None,
                "duration": "baseline",
                "repetition": rep,
                "status": "pending"
            })
    else:
        s_class = config["stressor_class"]
        s_method = config["stressor_method"]
        has_size = config.get("has_size_factor", s_class in ("mem", "io"))
        s_iter = S_LEVELS if has_size else [("N/A", "N/A", None)]

        for n_tag, n_label, n_frac in N_LEVELS:
            for s_tag, s_label, s_frac in s_iter:
                for dur in D_LEVELS:
                    for rep in range(1, reps + 1):
                        if has_size:
                            exp_id = f"run_{s_class}_{n_tag}_{dur}_{s_tag}_rep{rep}"
                        else:
                            exp_id = f"run_{s_class}_{n_tag}_{dur}_rep{rep}"

                        queue.append({
                            "exp_id": exp_id,
                            "mode": "stressor",
                            "stressor_class": s_class,
                            "stressor_method": s_method,
                            "n_tag": n_tag,
                            "n_label": n_label,
                            "n_fraction": n_frac,
                            "s_tag": s_tag,
                            "s_label": s_label,
                            "s_fraction": s_frac,
                            "duration": dur,
                            "repetition": rep,
                            "status": "pending"
                        })

    return queue


def load_or_init_state(config: Dict[str, Any], reset_state: bool = False) -> Dict[str, Any]:
    """Carica il checkpoint da disco o lo inizializza se non esiste / se richiesto --reset."""
    config_signature = {
        "model_id": config["model_id"],
        "device": config["device"],
        "benchmark_name": config["benchmark_name"],
        "sample_count": config["sample_count"],
        "use_stressor": config["use_stressor"],
        "stressor_class": config["stressor_class"],
        "stressor_method": config["stressor_method"],
        "repetitions": config["repetitions"],
    }

    if not reset_state and STATE_FILE_PATH.exists():
        try:
            with open(STATE_FILE_PATH, "r", encoding="utf-8") as f:
                state = json.load(f)
            if state.get("signature") == config_signature:
                return state
            else:
                print(" [INFO] Configurazione modificata rispetto all'ultimo checkpoint: creazione nuova coda.")
        except Exception:
            pass

    state = {
        "signature": config_signature,
        "config": config,
        "queue": build_experiment_queue(config)
    }
    save_state(state)
    return state


def save_state(state: Dict[str, Any]):
    with open(STATE_FILE_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())


# ==============================================================================
# ESECUZIONE DI UN SINGOLO ESPERIMENTO (N TASK + VMSTAT + SALVATAGGIO CSV)
# ==============================================================================
def execute_experiment(config: Dict[str, Any], exp: Dict[str, Any]):
    """Esegue tutti i sample_count task previsti per l'esperimento corrente."""
    dataset_path = Path(config["dataset_path"])
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    n_samples = int(config["sample_count"])
    bench_lower = config["benchmark_name"].strip().lower()
    total_cores = os.cpu_count() or int(config.get("n_threads", 4))

    stress_cmd: Optional[List[str]] = None
    if exp["mode"] == "stressor":
        stress_cmd = build_stress_ng_cmd(
            stressor_class=exp["stressor_class"],
            stressor_method=exp["stressor_method"],
            n_fraction=exp["n_fraction"],
            s_fraction=exp["s_fraction"],
            total_cores=total_cores
        )

    print("\n" + "=" * 65)
    print(f" ESECUZIONE ESPERIMENTO: {exp['exp_id']}")
    print("=" * 65)
    print(f" * Modello / Device : {config['model_id']} su {config['device'].upper()}")
    print(f" * Benchmark        : {config['benchmark_name']} ({n_samples} task)")
    print(f" * Ripetizione      : {exp['repetition']} / {config['repetitions']}")
    if stress_cmd:
        print(f" * Stressor         : {' '.join(stress_cmd)} (Fase: {exp['duration']})")
    else:
        print(" * Stressor         : DISATTIVO (Baseline)")
    print("=" * 65)

    monitor = SystemMonitor()
    monitor.start()

    task_records: List[Dict[str, Any]] = []
    try:
        for i in range(n_samples):
            sample_item = dataset[i % len(dataset)]
            if bench_lower == "mmlu":
                prompt = format_mmlu_prompt(sample_item)
            else:
                prompt = format_spider_prompt(sample_item)

            print(f" -> Task [{i + 1:02d}/{n_samples:02d}] in esecuzione...", end="", flush=True)
            metrics = run_single_llama_task(
                config=config,
                prompt=prompt,
                stress_cmd=stress_cmd,
                duration_mode=exp["duration"]
            )

            # Aggregazione dei campioni vmstat caduti nella finestra temporale del task i
            vm_avg = monitor.get_window_average(metrics["t_start_mono"], metrics["t_end_mono"])

            record = {
                "task_index": i + 1,
                "exp_id": exp["exp_id"],
                "model_id": config["model_id"],
                "device": config["device"],
                "benchmark": config["benchmark_name"],
                "stressor_class": exp["stressor_class"],
                "stressor_method": exp["stressor_method"],
                "num_workers": exp["n_label"],
                "size": exp["s_label"],
                "duration": exp["duration"],
                "repetition": exp["repetition"],
                "prompt_tokens": metrics["prompt_tokens"],
                "output_tokens": metrics["output_tokens"],
                "ttft_ms": metrics["ttft_ms"],
                "prompt_ms_per_tok": metrics["prompt_ms_per_tok"],
                "per_token_lat_ms": metrics["per_token_lat_ms"],
                "throughput_tok_s": metrics["throughput_tok_s"],
                "jitter_ms": metrics["jitter_ms"],
                "task_duration_s": metrics["duration_s"],
            }
            for k, v in vm_avg.items():
                record[f"vmstat_{k}"] = v

            task_records.append(record)
            print(
                f" TTFT={metrics['ttft_ms']:.1f}ms | "
                f"TPOT={metrics['per_token_lat_ms']:.2f}ms/tok | "
                f"Thr={metrics['throughput_tok_s']:.2f}tok/s | "
                f"Jitter={metrics['jitter_ms']:.2f}ms"
            )
    finally:
        monitor.stop()

    # 1. Salvataggio Time-Series vmstat (1 Hz) compatibile con plot_hardware_bottlenecks.py e distributions.jsl
    TIMESERIES_DIR.mkdir(parents=True, exist_ok=True)
    ts_filename = f"{exp['exp_id']}_timeseries.csv"
    ts_path = TIMESERIES_DIR / ts_filename
    monitor.save_timeseries_csv(ts_path)
    print(f"\n [OK] Serie temporale vmstat ({len(monitor.samples)} campioni @ 1Hz) salvata in: {ts_path}")

    # 2. Salvataggio campioni per singolo task (con vmstat allineato 1:1 per task)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if exp["mode"] == "baseline":
        tasks_csv_path = RESULTS_DIR / f"baseline_{config['model_id'].lower()}_{config['device']}_{bench_lower}_tasks.csv"
    else:
        tasks_csv_path = RESULTS_DIR / f"tasks_{exp['exp_id']}.csv"

    write_header = not tasks_csv_path.exists() or exp["mode"] != "baseline" or exp["repetition"] == 1
    mode_flag = "w" if write_header else "a"
    if task_records:
        fieldnames = list(task_records[0].keys())
        with open(tasks_csv_path, mode_flag, newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerows(task_records)
            f.flush()
            os.fsync(f.fileno())
        print(f" [OK] Metriche per-task ({len(task_records)} task) salvate in: {tasks_csv_path}")

    # 3. Aggregazione sulla Run per il DoE (media dei task della ripetizione)
    if task_records:
        mean_ttft = round(statistics.mean(r["ttft_ms"] for r in task_records), 4)
        mean_tpot = round(statistics.mean(r["per_token_lat_ms"] for r in task_records), 4)
        mean_thr = round(statistics.mean(r["throughput_tok_s"] for r in task_records), 4)
        # Se il jitter intra-task è 0 (es. risposta da 1 solo token su MMLU), usa la stdev del TPOT tra i task
        raw_jitters = [r["jitter_ms"] for r in task_records if r["jitter_ms"] > 0]
        if raw_jitters:
            mean_jitter = round(statistics.mean(raw_jitters), 4)
        elif len(task_records) >= 2:
            mean_jitter = round(statistics.stdev(r["per_token_lat_ms"] for r in task_records), 4)
        else:
            mean_jitter = 0.0

        mean_cpu_temp = round(statistics.mean(r["vmstat_cpu_temp_c"] for r in task_records), 2)
        mean_gpu_temp = round(statistics.mean(r["vmstat_gpu_temp_c"] for r in task_records), 2)

        # Salvataggio in formato compatibile con doe_example.csv e plot_llm_metrics.py
        doe_csv_path = RESULTS_DIR / f"doe_{config['model_id'].lower()}_{config['device']}_{bench_lower}_{exp['stressor_class']}.csv"
        doe_exists = doe_csv_path.exists()
        doe_row = {
            "num_workers": exp["n_label"],
            "size": exp["s_label"],
            "duration": exp["duration"],
            "ttft_ms": mean_ttft,
            "per_token_lat_ms": mean_tpot,
            "throughput_tok_s": mean_thr,
            "jitter_ms": mean_jitter,
        }
        with open(doe_csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(doe_row.keys()))
            if not doe_exists:
                writer.writeheader()
            writer.writerow(doe_row)
            f.flush()
            os.fsync(f.fileno())

        # Aggiornamento master_experiment_summary.csv
        master_exists = MASTER_SUMMARY_CSV.exists()
        master_row = {
            "exp_id": exp["exp_id"],
            "model_id": config["model_id"],
            "device": config["device"],
            "benchmark": config["benchmark_name"],
            "stressor_class": exp["stressor_class"],
            "stressor_method": exp["stressor_method"],
            "N_workers": exp["n_tag"],
            "S_size": exp["s_tag"],
            "duration_mode": exp["duration"],
            "repetition": exp["repetition"],
            "sample_count": n_samples,
            "ttft_ms": mean_ttft,
            "per_token_lat_ms": mean_tpot,
            "throughput_tok_s": mean_thr,
            "jitter_ms": mean_jitter,
            "avg_cpu_temp_c": mean_cpu_temp,
            "avg_gpu_temp_c": mean_gpu_temp,
        }
        with open(MASTER_SUMMARY_CSV, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(master_row.keys()))
            if not master_exists:
                writer.writeheader()
            writer.writerow(master_row)
            f.flush()
            os.fsync(f.fileno())

        print(f" [OK] Riepilogo DoE aggiornato in: {doe_csv_path}")


# ==============================================================================
# CLI ENTRYPOINT
# ==============================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Esecutore esperimenti Benchmark + Stressor + vmstat con gestione Reboot e 3 Ripetizioni."
    )
    parser.add_argument("-m", "--model", type=str, default=None, help="Modello ('slm' o 'llm')")
    parser.add_argument("-d", "--device", type=str, choices=["cpu", "gpu", "CPU", "GPU"], default=None, help="Allocazione ('cpu' o 'gpu')")
    parser.add_argument("-b", "--benchmark", type=str, choices=["mmlu", "spider", "MMLU", "SPIDER"], default=None, help="Benchmark ('mmlu' o 'spider')")

    stressor_grp = parser.add_mutually_exclusive_group()
    stressor_grp.add_argument("--use-stressor", dest="use_stressor", action="store_true", default=None, help="Avvia gli stressor (stress-ng)")
    stressor_grp.add_argument("--no-stressor", dest="use_stressor", action="store_false", help="Non avviare gli stressor (esegui Baseline)")

    parser.add_argument(
        "-s", "--stressor",
        type=str,
        default=None,
        help="Quale stressor avviare: 'cpu', 'mem', 'io', 'cache', 'sched', 'interrupt' (o metodo specifico: 'vm', 'stream', 'iomix', ecc.)"
    )
    parser.add_argument("-n", "--samples", type=int, default=None, help="Numero di campioni (task) da eseguire per esperimento")
    parser.add_argument("-r", "--repetitions", type=int, default=None, help="Numero di ripetizioni per esperimento (default: 3)")
    parser.add_argument("--transient-wait", type=int, default=None, help="Attesa transitorio post-boot in secondi (default: 90s per Jetson Orin)")
    parser.add_argument("--no-reboot", action="store_true", default=False, help="Non riavviare la board a fine esperimento (esegui tutta la coda in sequenza)")
    parser.add_argument("--reset", action="store_true", default=False, help="Azzera il checkpoint (experiment_state.json) e ricomincia da capo")
    parser.add_argument("--status", action="store_true", default=False, help="Mostra lo stato di avanzamento della coda di esperimenti ed esce")
    return parser.parse_args()


def main():
    args = parse_args()

    # Se l'utente passa modello/device/benchmark (o se run_config.json non esiste ancora),
    # istanzia/aggiorna automaticamente run_config.json tramite setup_run.instantiate_run
    if CONFIG_OUTPUT_PATH.exists():
        with open(CONFIG_OUTPUT_PATH, "r", encoding="utf-8") as f:
            current_cfg = json.load(f)
    else:
        current_cfg = {}

    need_reinstantiate = (
        not current_cfg
        or args.model is not None
        or args.device is not None
        or args.benchmark is not None
        or args.use_stressor is not None
        or args.stressor is not None
        or args.samples is not None
        or args.repetitions is not None
        or args.transient_wait is not None
    )

    if need_reinstantiate:
        model_in = args.model or current_cfg.get("model_id", "slm")
        device_in = args.device or current_cfg.get("device", "gpu")
        bench_in = args.benchmark or current_cfg.get("benchmark_name", "mmlu")

        if args.use_stressor is not None:
            use_stressor_in = args.use_stressor
        elif args.stressor is not None:
            use_stressor_in = args.stressor.strip().lower() not in ("none", "baseline", "off", "false")
        else:
            use_stressor_in = current_cfg.get("use_stressor", False)

        stressor_in = args.stressor or current_cfg.get("stressor_method") or current_cfg.get("stressor_class", "cpu")
        samples_in = args.samples if args.samples is not None else current_cfg.get("sample_count", 40)
        reps_in = args.repetitions if args.repetitions is not None else current_cfg.get("repetitions", 3)
        wait_in = args.transient_wait if args.transient_wait is not None else current_cfg.get("transient_wait_sec", 90)
        reboot_in = False if args.no_reboot else current_cfg.get("reboot_after_run", True)

        cfg_obj = instantiate_run(
            model_input=model_in,
            device_input=device_in,
            benchmark_input=bench_in,
            use_stressor=use_stressor_in,
            stressor_input=stressor_in,
            samples=samples_in,
            repetitions=reps_in,
            transient_wait_sec=wait_in,
            reboot_after_run=reboot_in
        )
        with open(CONFIG_OUTPUT_PATH, "r", encoding="utf-8") as f:
            config = json.load(f)
    else:
        config = current_cfg
        if args.no_reboot:
            config["reboot_after_run"] = False

    state = load_or_init_state(config, reset_state=args.reset)
    queue = state["queue"]

    completed = [e for e in queue if e["status"] == "completed"]
    pending = [e for e in queue if e["status"] == "pending"]

    print("\n" + "=" * 65)
    print(f" STATO CODA ESPERIMENTI: {len(completed)}/{len(queue)} completati ({len(pending)} rimanenti)")
    print("=" * 65)

    if args.status:
        for idx, e in enumerate(queue, 1):
            print(f"  {idx:02d}. [{e['status'].upper():9s}] {e['exp_id']}")
        return

    if not pending:
        print(" [OK] Tutti gli esperimenti in coda sono stati completati!")
        print("      Usa '--reset' se vuoi rieseguire la campagna da zero.")
        return

    while pending:
        next_exp = pending[0]

        # 1. Attesa transitorio post-boot (default 90s su Jetson Orin)
        wait_for_post_boot_transient(int(config.get("transient_wait_sec", 90)))

        # 2. Esecuzione esperimento (vmstat + llama-cli + eventuale stress-ng)
        execute_experiment(config, next_exp)

        # 3. Aggiornamento checkpoint su disco
        next_exp["status"] = "completed"
        save_state(state)

        pending = [e for e in queue if e["status"] == "pending"]
        print(f"\n [CHECKPOINT] Esperimento '{next_exp['exp_id']}' completato. Rimanenti in coda: {len(pending)}")

        # 4. Se è previsto il reboot al termine di ogni esperimento, riavvia la board
        if config.get("reboot_after_run", True):
            if pending:
                print(f" [PROSSIMO AL RIAVVIO] -> {pending[0]['exp_id']}")
            else:
                print(" [FINITO] Questo era l'ultimo esperimento della coda. Riavvio finale di pulizia.")
            trigger_board_reboot()


if __name__ == "__main__":
    main()
