#!/usr/bin/env python3
"""
run_experiments.py - Orchestratore di esecuzione Benchmark, Stressor, vmstat, SSH Sync e Reboot (Traccia A1).

Ottimizzato e protetto con gestione completa degli errori in runtime (try/except) e procedure di fallback:
  1. Rilevamento automatico dei flag supportati da llama-cli (-no-cnv, --perf) e protezione stdin=DEVNULL.
  2. Watchdog con timeout per singolo task (--task-timeout, default 180s) per evitare hang sotto stress-ng.
  3. Fallback automatico sul calcolo delle metriche (da perf_counter wall-clock se stderr di llama-cli è assente)
     e retry di fallback del task in caso di crash/OOM temporaneo.
  4. Procedura di Emergency Fallback in caso di errore critico in runtime:
     - Arresto immediato e sicuro di stress-ng, llama-cli e vmstat + pulizia file temporanei I/O.
     - Salvataggio su disco dei CSV parziali raccolti fino al momento dell'errore e del log diagnostico.
     - Aggiornamento atomico del checkpoint (experiment_state.json) per impedire boot-loop infiniti.
     - Invio di emergenza via SSH (scp) dei CSV parziali e del log errori sulla macchina dell'utente.
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
import traceback
import threading
import subprocess
import statistics
from pathlib import Path
from datetime import datetime
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
STATE_BACKUP_PATH = SCRIPT_DIR / "experiment_state.json.bak"
TIMESERIES_DIR = PROJECT_ROOT / "data_timeseries"
RESULTS_DIR = PROJECT_ROOT / "results"
MASTER_SUMMARY_CSV = PROJECT_ROOT / "master_experiment_summary.csv"
FALLBACK_ERROR_LOG = RESULTS_DIR / "runtime_error_fallback.log"
STRESS_TMP_DIR = Path("/tmp/edgeai_stress_ng_tmp") if os.name != "nt" else (PROJECT_ROOT / "tmp_stress")

# Livelli del Piano Fattoriale (DoE)
# I fattori sperimentali sono esclusivamente:
#   1. Carico worker (N_LEVELS): 25%, 50%, 75%, 100%
#   2. Working-set size (S_LEVELS): 25%, 50%, 75% (solo per mem e io)
# Il tempo è fisso e costante al tempo totale (t_tot) per l'intera durata della richiesta.
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

D_LEVELS = ["t_tot"]

VMSTAT_COLUMNS = [
    "timestamp_s", "r", "b", "swpd_kb", "free_kb", "buff_kb", "cache_kb",
    "si", "so", "bi", "bo", "in", "cs", "us", "sy", "id", "wa", "st",
    "cpu_temp_c", "gpu_temp_c", "throttled"
]

# Cache globale dei flag extra supportati dalla versione installata di llama-cli
_LLAMA_CLI_EXTRA_FLAGS: Optional[List[str]] = None


class ExperimentRuntimeError(RuntimeError):
    """Eccezione sollevata quando un task o uno stressor fallisce anche dopo il tentativo di fallback."""
    pass


def log_fallback_error(context: str, exc: BaseException, exp_id: str = "N/A"):
    """Scrive i dettagli dell'errore di runtime nel file di log di fallback."""
    try:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        ts_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(FALLBACK_ERROR_LOG, "a", encoding="utf-8") as f:
            f.write(f"\n[{ts_str}] === ERRORE RUNTIME & FALLBACK | Exp: {exp_id} | Contesto: {context} ===\n")
            f.write("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
    except Exception:
        pass


# ==============================================================================
# MONITORAGGIO HARDWARE IN BACKGROUND (vmstat 1 + Sensori Termici Jetson Orin)
# ==============================================================================
class SystemMonitor:
    """
    Raccoglie in background a 1 Hz le statistiche di vmstat e le metriche termiche
    della board (es. NVIDIA Jetson AGX Orin / Linux SBC), con fallback automatico
    su /proc/stat e /proc/meminfo se vmstat si interrompe o non è disponibile.
    """

    def __init__(self):
        self.samples: List[Dict[str, float]] = []
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._start_monotonic = 0.0
        self._vmstat_proc: Optional[subprocess.Popen] = None

    @staticmethod
    def _read_thermal_zones() -> Tuple[float, float, int]:
        """Legge le temperature CPU/GPU da /sys/class/thermal (o nvidia-smi) e lo stato di throttling."""
        cpu_temps = []
        gpu_temps = []
        throttled = 0

        try:
            for tz_dir in glob.glob("/sys/class/thermal/thermal_zone*"):
                try:
                    type_path = os.path.join(tz_dir, "type")
                    temp_path = os.path.join(tz_dir, "temp")
                    if not os.path.exists(temp_path):
                        continue
                    with open(temp_path, "r") as f:
                        raw_temp = float(f.read().strip())
                    temp_c = raw_temp / 1000.0 if raw_temp > 200 else raw_temp
                    if temp_c <= 0 or temp_c > 150:
                        continue

                    tz_type = ""
                    if os.path.exists(type_path):
                        with open(type_path, "r") as f:
                            tz_type = f.read().strip().lower()

                    if any(k in tz_type for k in ["gpu", "gv11b", "ga10b"]):
                        gpu_temps.append(temp_c)
                    elif any(k in tz_type for k in ["cpu", "soc", "cv", "tj", "core", "x86_pkg"]):
                        cpu_temps.append(temp_c)
                    else:
                        cpu_temps.append(temp_c)
                except Exception:
                    continue

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
        except Exception:
            pass

        cpu_t = round(sum(cpu_temps) / len(cpu_temps), 2) if cpu_temps else 0.0
        gpu_t = round(sum(gpu_temps) / len(gpu_temps), 2) if gpu_temps else cpu_t
        return cpu_t, gpu_t, throttled

    @staticmethod
    def _read_proc_fallback_sample(idx: int, now_mono: float) -> Dict[str, float]:
        """Fallback di lettura diretta da /proc/meminfo se vmstat non è presente o fallisce."""
        cpu_t, gpu_t, throttled = SystemMonitor._read_thermal_zones()
        free_kb, buff_kb, cache_kb, swpd_kb = 0, 0, 0, 0
        try:
            if os.path.exists("/proc/meminfo"):
                mem_info = {}
                with open("/proc/meminfo", "r") as f:
                    for line in f:
                        parts = line.split()
                        if len(parts) >= 2:
                            mem_info[parts[0].rstrip(":")] = int(parts[1])
                free_kb = mem_info.get("MemFree", 0)
                buff_kb = mem_info.get("Buffers", 0)
                cache_kb = mem_info.get("Cached", 0)
                swpd_kb = max(0, mem_info.get("SwapTotal", 0) - mem_info.get("SwapFree", 0))
        except Exception:
            pass

        row = {col: 0 for col in VMSTAT_COLUMNS}
        row.update({
            "timestamp_s": idx,
            "_mono_ts": now_mono,
            "swpd_kb": swpd_kb,
            "free_kb": free_kb,
            "buff_kb": buff_kb,
            "cache_kb": cache_kb,
            "cpu_temp_c": cpu_t,
            "gpu_temp_c": gpu_t,
            "throttled": throttled,
        })
        return row

    def _collector_loop(self):
        self._start_monotonic = time.monotonic()
        vmstat_path = shutil.which("vmstat")
        idx = 0

        if vmstat_path:
            try:
                self._vmstat_proc = subprocess.Popen(
                    ["vmstat", "-n", "1"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    text=True,
                    bufsize=1
                )
                for line in iter(self._vmstat_proc.stdout.readline, ""):
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
            except Exception as e:
                log_fallback_error("vmstat_collector_loop", e)
            finally:
                if self._vmstat_proc is not None:
                    try:
                        self._vmstat_proc.terminate()
                        self._vmstat_proc.wait(timeout=2)
                    except Exception:
                        try:
                            self._vmstat_proc.kill()
                        except Exception:
                            pass

        # Fallback: se vmstat non era disponibile o si è interrotto prima dello stop_event
        while not self._stop_event.is_set():
            time.sleep(1.0)
            idx += 1
            now_mono = time.monotonic()
            row = self._read_proc_fallback_sample(idx, now_mono)
            with self._lock:
                self.samples.append(row)

    def start(self):
        self.samples.clear()
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._collector_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._vmstat_proc is not None:
            try:
                self._vmstat_proc.terminate()
            except Exception:
                pass
        if self._thread is not None:
            self._thread.join(timeout=3.0)

    def get_window_average(self, t_start: float, t_end: float) -> Dict[str, float]:
        """
        Restituisce la media dei campioni vmstat caduti nella finestra [t_start, t_end]
        di un singolo task (allineamento 1:1 tra campioni vmstat e task di llama.cpp).
        """
        try:
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
        except Exception as e:
            log_fallback_error("get_window_average", e)
            return {col: 0.0 for col in VMSTAT_COLUMNS if col != "timestamp_s"}

    def save_timeseries_csv(self, filepath: Path):
        """Salva la serie temporale vmstat in modo atomico (file .tmp -> replace)."""
        try:
            filepath.parent.mkdir(parents=True, exist_ok=True)
            with self._lock:
                rows = list(self.samples)

            if not rows:
                rows = [self._read_proc_fallback_sample(1, time.monotonic())]

            tmp_path = filepath.with_suffix(".csv.tmp")
            with open(tmp_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=VMSTAT_COLUMNS)
                writer.writeheader()
                for r in rows:
                    clean_row = {k: r.get(k, 0) for k in VMSTAT_COLUMNS}
                    writer.writerow(clean_row)
                f.flush()
                os.fsync(f.fileno())
            tmp_path.replace(filepath)
        except Exception as e:
            log_fallback_error("save_timeseries_csv", e)
            raise


# ==============================================================================
# GESTIONE STRESSOR (stress-ng) E PULIZIA DI EMERGENZA
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
    STRESS_TMP_DIR.mkdir(parents=True, exist_ok=True)
    cmd = ["stress-ng", "--metrics-brief", "--temp-path", str(STRESS_TMP_DIR)]

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
    """Termina in modo pulito un sottoprocesso (es. stress-ng o llama-cli) e tutti i suoi figli."""
    if proc is None:
        return
    try:
        if proc.poll() is not None:
            return
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


def cleanup_orphan_stressors():
    """Uccide eventuali worker stress-ng rimasti appesi e pulisce la cartella temporanea I/O."""
    try:
        if os.name != "nt" and shutil.which("pkill"):
            subprocess.run(["pkill", "-9", "-f", "stress-ng"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        if STRESS_TMP_DIR.exists():
            shutil.rmtree(STRESS_TMP_DIR, ignore_errors=True)
    except Exception:
        pass


# ==============================================================================
# ESECUZIONE SINGOLO TASK CON llama-cli, TIMEOUT E FALLBACK AUTOMATICO
# ==============================================================================
PROMPT_EVAL_RE = re.compile(
    r"prompt eval time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*tokens\s*\(\s*([\d.]+)\s*ms per token,\s*([\d.]+)\s*tokens per second\)",
    re.IGNORECASE
)
EVAL_TIME_RE = re.compile(
    r"(?<!prompt )eval time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*(?:runs|tokens)\s*\(\s*([\d.]+)\s*ms per token,\s*([\d.]+)\s*tokens per second\)",
    re.IGNORECASE
)


def _detect_llama_cli_extra_flags(llama_cli_path: str) -> List[str]:
    """
    Rileva una sola volta i flag supportati dalla versione di llama-cli installata:
      - '-no-cnv' disabilita la modalità chat interattiva automatica nei modelli Instruct recenti.
      - '--perf' assicura la stampa delle statistiche di timing su stderr nelle build recenti.
    """
    global _LLAMA_CLI_EXTRA_FLAGS
    if _LLAMA_CLI_EXTRA_FLAGS is not None:
        return _LLAMA_CLI_EXTRA_FLAGS

    extra: List[str] = []
    try:
        res = subprocess.run(
            [llama_cli_path, "--help"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=10,
            check=False
        )
        help_text = res.stdout or ""
        if "-no-cnv" in help_text or "--no-cnv" in help_text:
            extra.append("-no-cnv")
        if "--perf" in help_text:
            extra.append("--perf")
    except Exception:
        pass

    _LLAMA_CLI_EXTRA_FLAGS = extra
    return extra


def _run_llama_cli_once(
    config: Dict[str, Any],
    prompt: str,
    stress_cmd: Optional[List[str]],
    duration_mode: str,
    task_timeout_sec: int,
    override_ngl: Optional[int] = None,
    override_threads: Optional[int] = None
) -> Dict[str, float]:
    """Esegue una singola invocazione di llama-cli con watchdog di timeout e fallback wall-clock."""
    llama_bin = config["llama_cli_path"]
    ngl = override_ngl if override_ngl is not None else config["n_gpu_layers"]
    n_threads = override_threads if override_threads is not None else config["n_threads"]

    cmd = [
        llama_bin,
        "-m", config["model_path"],
        "-ngl", str(ngl),
        "-t", str(n_threads),
        "--temp", str(config["temperature"]),
        "-s", str(config.get("seed", 42)),
        "-n", str(config["max_tokens"]),
        "--no-display-prompt",
    ] + _detect_llama_cli_extra_flags(llama_bin) + ["-p", prompt]

    stress_proc: Optional[subprocess.Popen] = None
    llama_proc: Optional[subprocess.Popen] = None
    preexec = os.setsid if os.name != "nt" else None

    t_start_mono = time.monotonic()
    t_start_perf = time.perf_counter()

    if stress_cmd and duration_mode in ("t_prefill", "t_tot"):
        try:
            stress_proc = subprocess.Popen(
                stress_cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                stdin=subprocess.DEVNULL,
                preexec_fn=preexec
            )
        except Exception as e:
            log_fallback_error("start_stress_ng_prefill", e)

    token_timestamps: List[float] = []
    stdout_chunks: List[bytes] = []
    stderr_chunks: List[bytes] = []
    first_token_seen = threading.Event()

    try:
        llama_proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL,
            bufsize=0,
            preexec_fn=preexec
        )

        def _read_stderr():
            try:
                while True:
                    chunk = llama_proc.stderr.read(1024)
                    if not chunk:
                        break
                    stderr_chunks.append(chunk)
            except Exception:
                pass

        def _read_stdout():
            nonlocal stress_proc
            try:
                while True:
                    b = llama_proc.stdout.read(16)
                    if not b:
                        break
                    now_t = time.perf_counter()
                    token_timestamps.append(now_t)
                    stdout_chunks.append(b)

                    if not first_token_seen.is_set():
                        first_token_seen.set()
                        if stress_cmd and duration_mode == "t_prefill":
                            stop_process_tree(stress_proc)
                            stress_proc = None
                        elif stress_cmd and duration_mode == "t_decode":
                            try:
                                stress_proc = subprocess.Popen(
                                    stress_cmd,
                                    stdout=subprocess.DEVNULL,
                                    stderr=subprocess.DEVNULL,
                                    stdin=subprocess.DEVNULL,
                                    preexec_fn=preexec
                                )
                            except Exception as e_dec:
                                log_fallback_error("start_stress_ng_decode", e_dec)
            except Exception:
                pass

        stderr_thread = threading.Thread(target=_read_stderr, daemon=True)
        stdout_thread = threading.Thread(target=_read_stdout, daemon=True)
        stderr_thread.start()
        stdout_thread.start()

        # Watchdog sul completamento di llama-cli
        llama_proc.wait(timeout=max(10, task_timeout_sec))
        stdout_thread.join(timeout=3.0)
        stderr_thread.join(timeout=3.0)

        stderr_text = b"".join(stderr_chunks).decode("utf-8", errors="replace")
        stdout_text = b"".join(stdout_chunks).decode("utf-8", errors="replace")

        if llama_proc.returncode != 0 and not token_timestamps:
            raise RuntimeError(
                f"llama-cli terminato con codice {llama_proc.returncode}. Stderr: {stderr_text[-400:].strip()}"
            )

    except subprocess.TimeoutExpired as te:
        stop_process_tree(llama_proc)
        raise TimeoutError(f"Task llama-cli scaduto per timeout (>{task_timeout_sec}s).") from te
    finally:
        stop_process_tree(stress_proc)
        stop_process_tree(llama_proc)

    t_end_mono = time.monotonic()
    t_end_perf = time.perf_counter()

    # 1. Parsing metriche ufficiali da stderr di llama.cpp
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

    # 2. FALLBACK WALL-CLOCK se il regex su stderr non ha trovato i timing (es. formato diverso o output troncato)
    if ttft_ms <= 0.0:
        if token_timestamps:
            ttft_ms = max(0.1, (token_timestamps[0] - t_start_perf) * 1000.0)
        else:
            ttft_ms = max(0.1, (t_end_perf - t_start_perf) * 1000.0)
        prompt_tokens = max(1, len(prompt.split()))
        prompt_ms_per_tok = ttft_ms / prompt_tokens

    if per_token_lat_ms <= 0.0:
        est_tokens = max(1, eval_tokens or len(token_timestamps) or len(stdout_text.split()))
        eval_tokens = est_tokens
        if len(token_timestamps) >= 2:
            eval_total_ms = max(0.1, (token_timestamps[-1] - token_timestamps[0]) * 1000.0)
        else:
            eval_total_ms = max(0.1, (t_end_perf - t_start_perf) * 1000.0 - ttft_ms)
        per_token_lat_ms = eval_total_ms / est_tokens
        throughput_tok_s = 1000.0 / per_token_lat_ms if per_token_lat_ms > 0 else 0.0

    # 3. Calcolo dell'Inter-Token Jitter
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


def run_single_llama_task(
    config: Dict[str, Any],
    prompt: str,
    stress_cmd: Optional[List[str]] = None,
    duration_mode: str = "t_tot",
    task_timeout_sec: int = 300,
    exp_id: str = "N/A"
) -> Optional[Dict[str, float]]:
    """
    Esegue un singolo task su llama-cli con timeout di 5 minuti (300s):
      - Se il task supera il timeout di 5 minuti (300s), il processo viene sospeso/terminato
        e la funzione restituisce None per passare immediatamente al task successivo.
      - Se il device è GPU e si verifica un errore di esecuzione GPU, lo script fallisce
        e riporta l'errore (nessun passaggio silenzioso a CPU).
    """
    try:
        return _run_llama_cli_once(
            config=config,
            prompt=prompt,
            stress_cmd=stress_cmd,
            duration_mode=duration_mode,
            task_timeout_sec=task_timeout_sec
        )
    except TimeoutError as te:
        log_fallback_error("task_timeout_suspended", te, exp_id=exp_id)
        cleanup_orphan_stressors()
        print(f"\n   [TIMEOUT > {task_timeout_sec}s] {te} -> Task sospeso, passaggio al task successivo.")
        return None
    except Exception as err:
        log_fallback_error("run_single_llama_task_error", err, exp_id=exp_id)
        cleanup_orphan_stressors()
        if str(config.get("device", "")).lower() == "gpu":
            raise ExperimentRuntimeError(
                f"Errore di esecuzione su device GPU durante il task: {err}"
            ) from err

        # Solo se siamo su CPU, effettua 1 tentativo di fallback prima di arrestare lo script
        print(f"\n   [WARNING] Errore runtime nel task su CPU ({err}). Avvio singolo tentativo di fallback...")
        time.sleep(3.0)
        fallback_threads = max(1, int(config.get("n_threads", 4)) // 2)
        try:
            res = _run_llama_cli_once(
                config=config,
                prompt=prompt,
                stress_cmd=stress_cmd,
                duration_mode=duration_mode,
                task_timeout_sec=task_timeout_sec,
                override_threads=fallback_threads
            )
            print("   [FALLBACK OK] Task recuperato con successo.")
            return res
        except TimeoutError as te2:
            log_fallback_error("task_timeout_suspended_after_retry", te2, exp_id=exp_id)
            cleanup_orphan_stressors()
            print(f"\n   [TIMEOUT > {task_timeout_sec}s] {te2} -> Task sospeso, passaggio al task successivo.")
            return None
        except Exception as second_err:
            log_fallback_error("run_single_llama_task_fallback_failed", second_err, exp_id=exp_id)
            raise ExperimentRuntimeError(
                f"Fallimento critico del task: {second_err}"
            ) from second_err


# ==============================================================================
# TRANSITORIO POST-BOOT E REBOOT DELLA BOARD
# ==============================================================================
def wait_for_post_boot_transient(wait_seconds: int):
    """
    Attende che il transitorio del sistema operativo e termico post-boot si esaurisca
    prima di avviare la raccolta delle metriche con vmstat e llama.cpp.
    """
    if wait_seconds <= 0:
        return

    print("\n" + "-" * 65)
    print(f" [TRANSITORIO] Attesa di {wait_seconds}s per stabilizzazione OS e termica...")
    print("-" * 65)

    if os.name != "nt":
        try:
            subprocess.run(["sync"], check=False, timeout=10)
            if hasattr(os, "geteuid") and os.geteuid() == 0:
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
    """Sincronizza i dischi ed esegue il riavvio della board con fallback tra sudo reboot e systemctl."""
    print("\n" + "=" * 65)
    print(" [REBOOT] Sincronizzazione filesystem e riavvio della board in corso...")
    print("=" * 65)
    cleanup_orphan_stressors()
    if os.name != "nt":
        try:
            subprocess.run(["sync"], check=False, timeout=10)
            res = subprocess.run(["sudo", "-n", "reboot"], check=False, timeout=10)
            if res.returncode != 0:
                subprocess.run(["systemctl", "reboot"], check=False, timeout=10)
        except Exception as e:
            log_fallback_error("trigger_board_reboot", e)
            print(f" [WARNING] Impossibile eseguire il reboot automatico ({e}). Proseguimento senza reboot.")
            return
    else:
        print(" [INFO] Ambiente Windows rilevato: simulazione comando 'sudo reboot'.")
    sys.exit(0)


# ==============================================================================
# INVIO AUTOMATICO DEI FILE CSV TRAMITE SSH (SCP) SULLA TUA MACCHINA
# ==============================================================================
def _build_ssh_base_opts(ssh_port: int, ssh_key: Optional[str], is_scp: bool = False) -> List[str]:
    """Costruisce le opzioni comuni per ssh / scp (non interattive per esecuzione automatica)."""
    opts = [
        "-o", "BatchMode=yes",
        "-o", "ConnectTimeout=10",
        "-o", "StrictHostKeyChecking=accept-new",
    ]
    port_flag = "-P" if is_scp else "-p"
    opts += [port_flag, str(ssh_port)]
    if ssh_key:
        expanded_key = os.path.expanduser(ssh_key)
        opts += ["-i", expanded_key]
    return opts


def send_files_via_ssh(
    config: Dict[str, Any],
    files_to_send: List[Tuple[Path, str]]
) -> bool:
    """
    Invia i file CSV prodotti dall'esperimento corrente sulla macchina remota dell'utente via SSH/SCP.
    Protetto con try/except e timeout per non bloccare mai lo script se la rete cade.
    """
    try:
        ssh_target = config.get("ssh_target") or os.environ.get("SSH_TARGET")
        if not ssh_target:
            return False

        ssh_dest = (config.get("ssh_dest") or os.environ.get("SSH_DEST") or "./EdgeAI_Results").rstrip("/")
        ssh_port = int(config.get("ssh_port") or os.environ.get("SSH_PORT") or 22)
        ssh_key = config.get("ssh_key") or os.environ.get("SSH_KEY")

        valid_items = [(local_p, rel_dir) for local_p, rel_dir in files_to_send if local_p and local_p.exists()]
        if not valid_items:
            return False

        print("\n" + "-" * 65)
        print(f" [SSH SYNC] Invio di {len(valid_items)} file a {ssh_target}:{ssh_dest} (porta {ssh_port})...")
        print("-" * 65)

        remote_subdirs = sorted({rel_dir for _, rel_dir in valid_items if rel_dir})
        remote_dirs_to_create = [ssh_dest] + [f"{ssh_dest}/{d}" for d in remote_subdirs]
        mkdir_cmd_str = "mkdir -p " + " ".join(f'"{d}"' for d in remote_dirs_to_create)

        ssh_cmd = ["ssh"] + _build_ssh_base_opts(ssh_port, ssh_key, is_scp=False) + [ssh_target, mkdir_cmd_str]
        subprocess.run(ssh_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=15, check=False)

        all_ok = True
        scp_base = ["scp"] + _build_ssh_base_opts(ssh_port, ssh_key, is_scp=True)

        for local_path, rel_subdir in valid_items:
            remote_dir = f"{ssh_dest}/{rel_subdir}" if rel_subdir else ssh_dest
            remote_spec = f"{ssh_target}:{remote_dir}/{local_path.name}"
            sent = False

            for attempt in range(1, 4):
                try:
                    res = subprocess.run(
                        scp_base + [str(local_path), remote_spec],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        timeout=30,
                        check=False
                    )
                    if res.returncode == 0:
                        print(f"   [OK] Inviato: {local_path.name} -> {remote_spec}")
                        sent = True
                        break
                    else:
                        err_msg = (res.stderr or "").strip()
                        print(f"   [RETRY {attempt}/3] Errore invio {local_path.name}: {err_msg}")
                except subprocess.TimeoutExpired:
                    print(f"   [RETRY {attempt}/3] Timeout durante l'invio SCP di {local_path.name}")
                except Exception as e_scp:
                    print(f"   [RETRY {attempt}/3] Eccezione SCP su {local_path.name}: {e_scp}")
                time.sleep(3)

            if not sent:
                all_ok = False
                print(f"   [FALLBACK LOCALE] Impossibile inviare {local_path.name} via SSH. Il file è preservato localmente in: {local_path}")

        return all_ok
    except Exception as e:
        log_fallback_error("send_files_via_ssh", e)
        print(f"   [WARNING] Errore nel modulo SSH Sync ({e}). Tutti i CSV sono comunque salvati localmente.")
        return False


def sync_all_existing_csvs_via_ssh(config: Dict[str, Any]):
    """Invia tutti i file CSV e log già prodotti verso la macchina remota."""
    files_to_send: List[Tuple[Path, str]] = []

    if TIMESERIES_DIR.exists():
        for p in sorted(TIMESERIES_DIR.glob("*.csv")):
            files_to_send.append((p, "data_timeseries"))

    if RESULTS_DIR.exists():
        for p in sorted(RESULTS_DIR.glob("*.csv")):
            files_to_send.append((p, "results"))
        if FALLBACK_ERROR_LOG.exists():
            files_to_send.append((FALLBACK_ERROR_LOG, "results"))

    if MASTER_SUMMARY_CSV.exists():
        files_to_send.append((MASTER_SUMMARY_CSV, ""))

    if STATE_FILE_PATH.exists():
        files_to_send.append((STATE_FILE_PATH, "setup"))

    if CONFIG_OUTPUT_PATH.exists():
        files_to_send.append((CONFIG_OUTPUT_PATH, "setup"))

    if not files_to_send:
        print(" [INFO] Nessun file CSV trovato da sincronizzare.")
        return

    send_files_via_ssh(config, files_to_send)


# ==============================================================================
# COSTRUZIONE E GESTIONE DELLA CODA ESPERIMENTI (CON SCRITTURA ATOMICA E BACKUP)
# ==============================================================================
def build_experiment_queue(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Costruisce la lista di esperimenti da eseguire (ciascuno ripetuto per config['repetitions'] volte)."""
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


def save_state(state: Dict[str, Any]):
    """Salva experiment_state.json in modo atomico con backup di fallback."""
    try:
        if STATE_FILE_PATH.exists():
            shutil.copy2(STATE_FILE_PATH, STATE_BACKUP_PATH)
        tmp_path = STATE_FILE_PATH.with_suffix(".json.tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        tmp_path.replace(STATE_FILE_PATH)
    except Exception as e:
        log_fallback_error("save_state", e)
        if STATE_BACKUP_PATH.exists():
            shutil.copy2(STATE_BACKUP_PATH, STATE_FILE_PATH)


def load_or_init_state(config: Dict[str, Any], reset_state: bool = False) -> Dict[str, Any]:
    """Carica il checkpoint da disco (con fallback sul file .bak se corrotto) o lo inizializza."""
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

    if not reset_state:
        for candidate in (STATE_FILE_PATH, STATE_BACKUP_PATH):
            if candidate.exists():
                try:
                    with open(candidate, "r", encoding="utf-8") as f:
                        state = json.load(f)
                    if state.get("signature") == config_signature and isinstance(state.get("queue"), list):
                        return state
                    else:
                        print(" [INFO] Configurazione modificata rispetto all'ultimo checkpoint: creazione nuova coda.")
                        break
                except Exception as e:
                    log_fallback_error(f"load_state_{candidate.name}", e)
                    print(f" [FALLBACK] Checkpoint {candidate.name} corrotto, tentativo col backup...")

    state = {
        "signature": config_signature,
        "config": config,
        "queue": build_experiment_queue(config)
    }
    save_state(state)
    return state


# ==============================================================================
# PROCEDURA DI EMERGENCY FALLBACK IN CASO DI ERRORE CRITICO RUNTIME
# ==============================================================================
def trigger_emergency_fallback(
    config: Dict[str, Any],
    state: Dict[str, Any],
    exp: Dict[str, Any],
    monitor: Optional[SystemMonitor],
    partial_task_records: List[Dict[str, Any]],
    exc: BaseException
):
    """
    Procedura di emergenza attivata quando si verifica un errore bloccante in runtime:
      1. Ferma immediatamente vmstat e qualsiasi processo stress-ng / llama-cli attivo.
      2. Salva i campioni parziali raccolti fino al momento del crash in file CSV di fallback.
      3. Registra l'errore in results/runtime_error_fallback.log e marca l'esperimento come
         'failed_fallback' nel checkpoint per evitare riavvii infiniti (boot-loop).
      4. Invia via SSH i dati parziali e il log degli errori sulla macchina dell'utente e arresta lo script.
    """
    exp_id = exp.get("exp_id", "unknown_exp")
    print("\n" + "!" * 70)
    print(f" [EMERGENCY FALLBACK] Errore di elaborazione in runtime durante '{exp_id}'!")
    print(f" Causa: {exc}")
    print(" Avvio procedura di arresto sicuro, salvataggio dati parziali e sync SSH...")
    print("!" * 70)

    # 1. Arresto immediato processi in background
    if monitor is not None:
        try:
            monitor.stop()
        except Exception:
            pass
    cleanup_orphan_stressors()

    # 2. Scrittura nel log degli errori
    log_fallback_error("trigger_emergency_fallback", exc, exp_id=exp_id)

    fallback_files: List[Tuple[Path, str]] = []
    if FALLBACK_ERROR_LOG.exists():
        fallback_files.append((FALLBACK_ERROR_LOG, "results"))

    # 3. Salvataggio serie temporale vmstat parziale
    if monitor is not None and monitor.samples:
        try:
            TIMESERIES_DIR.mkdir(parents=True, exist_ok=True)
            partial_ts_path = TIMESERIES_DIR / f"fallback_partial_{exp_id}_timeseries.csv"
            monitor.save_timeseries_csv(partial_ts_path)
            fallback_files.append((partial_ts_path, "data_timeseries"))
            print(f" [FALLBACK] Salvati {len(monitor.samples)} campioni vmstat parziali in: {partial_ts_path}")
        except Exception as e_ts:
            log_fallback_error("save_partial_timeseries", e_ts, exp_id=exp_id)

    # 4. Salvataggio record task parziali
    if partial_task_records:
        try:
            RESULTS_DIR.mkdir(parents=True, exist_ok=True)
            partial_tasks_path = RESULTS_DIR / f"fallback_partial_{exp_id}_tasks.csv"
            with open(partial_tasks_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(partial_task_records[0].keys()))
                writer.writeheader()
                writer.writerows(partial_task_records)
                f.flush()
                os.fsync(f.fileno())
            fallback_files.append((partial_tasks_path, "results"))
            print(f" [FALLBACK] Salvati {len(partial_task_records)} task parziali in: {partial_tasks_path}")
        except Exception as e_tr:
            log_fallback_error("save_partial_tasks", e_tr, exp_id=exp_id)

    # 5. Aggiornamento stato checkpoint per evitare boot-loop
    try:
        exp["status"] = "failed_fallback"
        exp["error"] = str(exc)
        save_state(state)
        fallback_files.append((STATE_FILE_PATH, "setup"))
    except Exception:
        pass

    # 6. Invio di emergenza via SSH alla macchina dell'utente
    send_files_via_ssh(config, fallback_files)

    print("\n [ARRESTO SICURO] Procedura di fallback completata. Lo script si è fermato senza riavviare la board.")
    print(f" Consulta i dettagli in: {FALLBACK_ERROR_LOG}\n")
    sys.exit(1)


# ==============================================================================
# ESECUZIONE DI UN SINGOLO ESPERIMENTO (N TASK + VMSTAT + SALVATAGGIO CSV)
# ==============================================================================
def execute_experiment(
    config: Dict[str, Any],
    state: Dict[str, Any],
    exp: Dict[str, Any],
    task_timeout_sec: int = 300
) -> List[Tuple[Path, str]]:
    """Esegue tutti i sample_count task previsti per l'esperimento corrente con protezione try/except."""
    monitor: Optional[SystemMonitor] = None
    task_records: List[Dict[str, Any]] = []

    try:
        dataset_path = Path(config["dataset_path"])
        with open(dataset_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)
        if not isinstance(dataset, list) or not dataset:
            raise ValueError(f"Il dataset {dataset_path} è vuoto o non valido.")

        n_samples = int(config["sample_count"])
        bench_lower = config["benchmark_name"].strip().lower()
        total_cores = os.cpu_count() or int(config.get("n_threads", 4))

        stress_cmd: Optional[List[str]] = None
        if exp["mode"] == "stressor":
            if shutil.which("stress-ng") is None:
                raise RuntimeError("Il binario 'stress-ng' non è installato sulla macchina ma use_stressor=True.")
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

        for i in range(n_samples):
            sample_item = dataset[i % len(dataset)]
            prompt = format_mmlu_prompt(sample_item) if bench_lower == "mmlu" else format_spider_prompt(sample_item)

            print(f" -> Task [{i + 1:02d}/{n_samples:02d}] in esecuzione...", end="", flush=True)
            metrics = run_single_llama_task(
                config=config,
                prompt=prompt,
                stress_cmd=stress_cmd,
                duration_mode=exp["duration"],
                task_timeout_sec=task_timeout_sec,
                exp_id=exp["exp_id"]
            )

            # Se il task ha superato il timeout di 5 minuti (300s), viene sospeso e si passa al successivo
            if metrics is None:
                continue

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

        monitor.stop()
        cleanup_orphan_stressors()

        # 1. Salvataggio Time-Series vmstat (1 Hz)
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

        # 3. Aggregazione sulla Run per il DoE
        generated_files: List[Tuple[Path, str]] = [
            (ts_path, "data_timeseries"),
            (tasks_csv_path, "results"),
        ]

        if task_records:
            mean_ttft = round(statistics.mean(r["ttft_ms"] for r in task_records), 4)
            mean_tpot = round(statistics.mean(r["per_token_lat_ms"] for r in task_records), 4)
            mean_thr = round(statistics.mean(r["throughput_tok_s"] for r in task_records), 4)
            raw_jitters = [r["jitter_ms"] for r in task_records if r["jitter_ms"] > 0]
            if raw_jitters:
                mean_jitter = round(statistics.mean(raw_jitters), 4)
            elif len(task_records) >= 2:
                mean_jitter = round(statistics.stdev(r["per_token_lat_ms"] for r in task_records), 4)
            else:
                mean_jitter = 0.0

            mean_cpu_temp = round(statistics.mean(r["vmstat_cpu_temp_c"] for r in task_records), 2)
            mean_gpu_temp = round(statistics.mean(r["vmstat_gpu_temp_c"] for r in task_records), 2)

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
            generated_files.append((doe_csv_path, "results"))
            generated_files.append((MASTER_SUMMARY_CSV, ""))

        return generated_files

    except BaseException as exc:
        trigger_emergency_fallback(
            config=config,
            state=state,
            exp=exp,
            monitor=monitor,
            partial_task_records=task_records,
            exc=exc
        )
        return []


# ==============================================================================
# CLI ENTRYPOINT
# ==============================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Esecutore esperimenti Benchmark + Stressor + vmstat con gestione Reboot, 3 Ripetizioni, Fallback e invio CSV via SSH."
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
    parser.add_argument("--task-timeout", type=int, default=300, help="Timeout massimo in secondi per singolo task llama-cli (default: 300s = 5 min)")
    parser.add_argument("--no-reboot", action="store_true", default=False, help="Non riavviare la board a fine esperimento (esegui tutta la coda in sequenza)")
    parser.add_argument(
        "--ssh-target",
        type=str,
        default=None,
        help="Utente e IP/host della tua macchina a cui inviare i CSV via SSH (es. 'gabri@192.168.1.50' o 'gabri@localhost')"
    )
    parser.add_argument(
        "--ssh-dest",
        type=str,
        default=None,
        help="Cartella di destinazione sulla tua macchina per i CSV ricevuti via SSH (default: ./EdgeAI_Results)"
    )
    parser.add_argument(
        "--ssh-port",
        type=int,
        default=None,
        help="Porta SSH della tua macchina (default: 22, o es. 2222 se usi tunnel SSH inverso -R)"
    )
    parser.add_argument(
        "--ssh-key",
        type=str,
        default=None,
        help="Percorso alla chiave privata SSH sulla board per invio automatico senza password (es. ~/.ssh/id_ed25519)"
    )
    parser.add_argument(
        "--sync-only",
        action="store_true",
        default=False,
        help="Invia immediatamente via SSH tutti i CSV già presenti sulla board verso la tua macchina ed esce"
    )
    parser.add_argument("--reset", action="store_true", default=False, help="Azzera il checkpoint (experiment_state.json) e ricomincia da capo")
    parser.add_argument("--status", action="store_true", default=False, help="Mostra lo stato di avanzamento della coda di esperimenti ed esce")
    return parser.parse_args()


def main():
    args = parse_args()

    try:
        current_cfg = {}
        if CONFIG_OUTPUT_PATH.exists():
            try:
                with open(CONFIG_OUTPUT_PATH, "r", encoding="utf-8") as f:
                    current_cfg = json.load(f)
            except Exception as e:
                log_fallback_error("read_run_config", e)
                print(f" [WARNING] File {CONFIG_OUTPUT_PATH} illeggibile, verrà rigenerato.")

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
            or args.ssh_target is not None
            or args.ssh_dest is not None
            or args.ssh_port is not None
            or args.ssh_key is not None
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
            ssh_target_in = args.ssh_target if args.ssh_target is not None else current_cfg.get("ssh_target")
            ssh_dest_in = args.ssh_dest if args.ssh_dest is not None else current_cfg.get("ssh_dest", "./EdgeAI_Results")
            ssh_port_in = args.ssh_port if args.ssh_port is not None else current_cfg.get("ssh_port", 22)
            ssh_key_in = args.ssh_key if args.ssh_key is not None else current_cfg.get("ssh_key")

            instantiate_run(
                model_input=model_in,
                device_input=device_in,
                benchmark_input=bench_in,
                use_stressor=use_stressor_in,
                stressor_input=stressor_in,
                samples=samples_in,
                repetitions=reps_in,
                transient_wait_sec=wait_in,
                reboot_after_run=reboot_in,
                ssh_target=ssh_target_in,
                ssh_dest=ssh_dest_in,
                ssh_port=ssh_port_in,
                ssh_key=ssh_key_in
            )
            with open(CONFIG_OUTPUT_PATH, "r", encoding="utf-8") as f:
                config = json.load(f)
        else:
            config = current_cfg
            if args.no_reboot:
                config["reboot_after_run"] = False

        if args.sync_only:
            sync_all_existing_csvs_via_ssh(config)
            return

        state = load_or_init_state(config, reset_state=args.reset)
        queue = state["queue"]

        completed = [e for e in queue if e["status"] == "completed"]
        pending = [e for e in queue if e["status"] == "pending"]
        failed = [e for e in queue if e["status"] == "failed_fallback"]

        print("\n" + "=" * 65)
        print(
            f" STATO CODA ESPERIMENTI: {len(completed)}/{len(queue)} completati | "
            f"{len(pending)} in attesa | {len(failed)} falliti"
        )
        print("=" * 65)

        if args.status:
            for idx, e in enumerate(queue, 1):
                print(f"  {idx:02d}. [{e['status'].upper():15s}] {e['exp_id']}")
            return

        if not pending:
            print(" [OK] Nessun esperimento 'pending' rimanente nella coda!")
            print("      Usa '--reset' se vuoi rieseguire la campagna da zero.")
            return

        while pending:
            next_exp = pending[0]

            # 1. Attesa transitorio post-boot (default 90s su Jetson Orin)
            wait_for_post_boot_transient(int(config.get("transient_wait_sec", 90)))

            # 2. Esecuzione esperimento (protetto da try/except e emergency fallback)
            generated_files = execute_experiment(
                config=config,
                state=state,
                exp=next_exp,
                task_timeout_sec=args.task_timeout
            )

            # 3. Aggiornamento checkpoint su disco
            next_exp["status"] = "completed"
            save_state(state)

            # 4. Invio immediato dei CSV prodotti e dello stato via SSH alla tua macchina
            generated_files.append((STATE_FILE_PATH, "setup"))
            send_files_via_ssh(config, generated_files)

            pending = [e for e in queue if e["status"] == "pending"]
            print(f"\n [CHECKPOINT] Esperimento '{next_exp['exp_id']}' completato. Rimanenti in coda: {len(pending)}")

            # 5. Se è previsto il reboot al termine di ogni esperimento, riavvia la board
            if config.get("reboot_after_run", True):
                if pending:
                    print(f" [PROSSIMO AL RIAVVIO] -> {pending[0]['exp_id']}")
                else:
                    print(" [FINITO] Questo era l'ultimo esperimento della coda. Riavvio finale di pulizia.")
                trigger_board_reboot()

    except SystemExit:
        raise
    except BaseException as fatal_exc:
        cleanup_orphan_stressors()
        log_fallback_error("main_top_level", fatal_exc)
        print(f"\n[ERRORE FATALE] {fatal_exc}. Processo arrestato in sicurezza. Log in: {FALLBACK_ERROR_LOG}")
        sys.exit(1)


if __name__ == "__main__":
    main()
