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
import gc
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
    get_hardware_cache_info,
    get_cpu_cache_levels,
    get_methods_for_class,
    STRESSOR_FAMILY_ORDER,
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
#   1. Carico worker (N_LEVELS): 25%, 50%, 75%, 100% del numero di workers totali
#   2. Working-set size (S_LEVELS): 25%, 50%, 75% della dimensione massima (solo per mem e io)
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
        """Legge le temperature CPU/GPU da /sys/class/thermal, /sys/class/hwmon, o nvidia-smi e lo stato di throttling."""
        cpu_temps = []
        gpu_temps = []
        throttled = 0

        try:
            # 1. Lettura da /sys/class/thermal (standard Linux e NVIDIA Jetson AGX Orin)
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

            # 2. Fallback per CPU tramite /sys/class/hwmon (se non rilevato in /sys/class/thermal)
            if not cpu_temps:
                for hw_path in glob.glob("/sys/class/hwmon/hwmon*/temp*_input"):
                    try:
                        with open(hw_path, "r") as f:
                            raw_t = float(f.read().strip())
                        t_c = raw_t / 1000.0 if raw_t > 200 else raw_t
                        if 10.0 <= t_c <= 120.0:
                            cpu_temps.append(t_c)
                    except Exception:
                        pass

            # 3. Fallback per GPU tramite nvidia-smi (GPU discrete su PC / Colab / Server)
            if not gpu_temps:
                try:
                    res = subprocess.run(
                        ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.DEVNULL,
                        text=True,
                        timeout=1
                    )
                    if res.returncode == 0 and res.stdout.strip():
                        for val_str in res.stdout.strip().splitlines():
                            val = float(val_str.strip())
                            if 10.0 <= val <= 120.0:
                                gpu_temps.append(val)
                except Exception:
                    pass

            # 4. Rilevamento thermal throttling
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
    stressor_method: Optional[str] = None,
    n_fraction: float = 1.0,
    s_fraction: Optional[float] = None,
    total_cores: int = 4
) -> List[str]:
    """
    Costruisce il comando stress-ng eseguendo simultaneamente tutti i metodi
    della famiglia di stressor con i fattori (N workers e S working-set)
    applicati con lo stesso valore per tutti i metodi della famiglia.
    """
    workers = max(1, int(round(total_cores * n_fraction)))
    STRESS_TMP_DIR.mkdir(parents=True, exist_ok=True)
    cmd = ["stress-ng", "--metrics-brief", "--temp-path", str(STRESS_TMP_DIR)]

    s_clean_class = str(stressor_class or "cpu").lower().strip()
    if s_clean_class in ("matrix", "fft"):
        s_clean_class = "cpu"
    elif s_clean_class in ("vm", "stream", "memrate", "mem_rate", "memoria", "memory"):
        s_clean_class = "mem"
    elif s_clean_class in ("cache", "l1cache", "l1-cache", "llc-affinity", "llc_affinity"):
        s_clean_class = "cache"
    elif s_clean_class in ("io", "i/o", "iomix", "hdd", "disk"):
        s_clean_class = "io"
    elif s_clean_class in ("os", "sched", "fork", "switch", "scheduler", "schedpolicy"):
        s_clean_class = "os"
    elif s_clean_class in ("interrupt", "interrupts", "timer", "sigq", "irq"):
        s_clean_class = "interrupt"

    s_frac = s_fraction if s_fraction is not None else 0.50

    # 1. CPU (cpu, matrix, fft eseguiti simultaneamente)
    if s_clean_class == "cpu":
        cmd += [
            "--cpu", str(workers),
            "--matrix", str(workers),
            "--fft", str(workers)
        ]

    # 2. MEMORIA (vm, stream, mem_rate con fattore S applicato all'allocazione memoria)
    elif s_clean_class == "mem":
        ram_mb = get_total_ram_mb()
        alloc_mb_total = max(64, int(ram_mb * 0.50 * s_frac))
        alloc_per_worker = max(16, alloc_mb_total // workers)
        cmd += [
            "--vm", str(workers),
            "--vm-bytes", f"{alloc_per_worker}M",
            "--stream", str(workers),
            "--memrate", str(workers)
        ]

    # 3. CACHE (cache, l1-cache, llc-affinity se livelli > 2)
    elif s_clean_class == "cache":
        cmd += [
            "--cache", str(workers),
            "--l1cache", str(workers)
        ]
        if get_cpu_cache_levels() > 2:
            cmd += ["--llc-affinity", str(workers)]

    # 4. I/O (iomix, hdd eseguiti simultaneamente con lo stesso S)
    elif s_clean_class == "io":
        file_mb = max(16, int(512 * s_frac))
        cmd += [
            "--iomix", str(workers),
            "--iomix-bytes", f"{file_mb}M",
            "--hdd", str(workers),
            "--hdd-bytes", f"{file_mb}M"
        ]

    # 5. OS / SCHEDULER (schedpolicy, fork, switch eseguiti simultaneamente)
    elif s_clean_class in ("os", "sched"):
        cmd += [
            "--schedpolicy", str(workers),
            "--fork", str(workers),
            "--switch", str(workers)
        ]

    # 6. INTERRUPT (timer, sigq eseguiti simultaneamente)
    elif s_clean_class in ("interrupt", "interrupts"):
        cmd += [
            "--timer", str(workers),
            "--sigq", str(workers)
        ]

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


def ensure_jetson_performance_mode():
    """
    Abilita la modalità di massima prestazione (nvpmodel -m 0) e blocca i clock (jetson_clocks).
    Garantisce frequenze massime a CPU (1.5 GHz), GPU (625 MHz) e memoria LPDDR5 (2133 MHz),
    abbattendo i tempi di esecuzione ed eliminando il jitter di frequenza durante il DoE.
    """
    if os.name != "nt":
        try:
            if shutil.which("nvpmodel"):
                subprocess.run(["sudo", "-n", "nvpmodel", "-m", "0"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
            if shutil.which("jetson_clocks"):
                res = subprocess.run(["sudo", "-n", "jetson_clocks"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
                if res.returncode == 0:
                    print(" [PERF] Jetson Clocks attivi: frequenze CPU, GPU e bus RAM LPDDR5 bloccate al massimo.")
        except Exception:
            pass


def reclaim_unified_memory(model_path: Optional[str] = None, kill_llama: bool = True):
    """
    Rilascia attivamente e immediatamente la memoria unificata (UMA) su NVIDIA Jetson:
      1. Se kill_llama=True, termina eventuali processi llama-cli / llama-server / stress-ng rimasti appesi.
      2. Forza il garbage collection in Python.
      3. Sincronizza i filesystem buffer (os.sync).
      4. Rimuove TUTTI i file GGUF e dataset dalla Linux VFS Page Cache tramite posix_fadvise(POSIX_FADV_DONTNEED)
         (non richiede privilegi di root e libera istantaneamente gigabyte di RAM occupata dalla page cache).
      5. Tenta il drop e la compattazione delle cache kernel (drop_caches=3, compact_memory=1).
    """
    if kill_llama:
        try:
            if os.name != "nt" and shutil.which("pkill"):
                subprocess.run(["pkill", "-9", "-f", "llama-cli"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
                subprocess.run(["pkill", "-9", "-f", "llama-server"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
                subprocess.run(["pkill", "-9", "-f", "stress-ng"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
                time.sleep(0.5)
        except Exception:
            pass

    gc.collect()

    if hasattr(os, "sync"):
        try:
            os.sync()
        except Exception:
            pass

    if hasattr(os, "posix_fadvise") and hasattr(os, "POSIX_FADV_DONTNEED"):
        targets: List[Path] = []
        if model_path:
            targets.append(Path(model_path))
        models_dir = PROJECT_ROOT / "models"
        if models_dir.exists():
            targets.extend(models_dir.glob("*.gguf"))
        datasets_dir = PROJECT_ROOT / "datasets"
        if datasets_dir.exists():
            targets.extend(datasets_dir.glob("*.json"))

        seen_paths = set()
        for p in targets:
            try:
                resolved = str(p.resolve())
                if resolved in seen_paths or not p.exists():
                    continue
                seen_paths.add(resolved)
                fd = os.open(resolved, os.O_RDONLY)
                try:
                    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                finally:
                    os.close(fd)
            except Exception:
                pass

    if kill_llama and os.name != "nt":
        try:
            subprocess.run(
                ["sudo", "-n", "sysctl", "-w", "vm.drop_caches=3"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=2,
                check=False
            )
            subprocess.run(
                ["sudo", "-n", "sysctl", "-w", "vm.compact_memory=1"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=2,
                check=False
            )
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
TURN_STATS_RE = re.compile(
    r"\[\s*Prompt:\s*([0-9]+(?:[.,][0-9]+)?)\s*t/s\s*\|\s*Generation:\s*([0-9]+(?:[.,][0-9]+)?)\s*t/s\s*\]",
    re.IGNORECASE
)


def clean_llama_response(raw_text: str, prompt: str) -> str:
    """
    Rimuove sequenze di caricamento/spinner, banner ASCII di llama.cpp,
    comandi interattivi, echo del prompt, statistiche di timing e footer 'Exiting...'
    restituendo esclusivamente la risposta generata dal modello.
    """
    if not raw_text:
        return ""

    text = raw_text

    # 1. Rimuove sequenze dello spinner di caricamento e backspaces \b (\x08)
    text = re.sub(r"Loading model\.+[\s|/\\\-\x08]*", "", text)
    text = text.replace("\x08", "").replace("\b", "")

    # 2. Se è presente il menu dei comandi interattivi (/exit, /regen, /clear, ecc.)
    if "available commands:" in text:
        idx = text.find("available commands:")
        sub = text[idx:]
        lines = sub.splitlines()
        content_lines = []
        in_banner = True
        for line in lines:
            s_line = line.strip()
            if in_banner:
                if s_line.startswith("/glob") or s_line.startswith("/read"):
                    in_banner = False
                continue
            content_lines.append(line)
        text = "\n".join(content_lines)

    # 3. Rimuove l'echo del prompt (sia multi-linea che single-line) o del comando /clear
    clean_p = prompt.strip()
    single_p = " ".join(l.strip() for l in prompt.splitlines() if l.strip())
    for candidate_p in (clean_p, single_p):
        if candidate_p and candidate_p in text:
            text = text.split(candidate_p, 1)[-1]
            break

    text = re.sub(r"^\s*>\s*/clear\s*", "", text, flags=re.IGNORECASE | re.MULTILINE)
    text = re.sub(r"^\s*/clear\s*", "", text, flags=re.IGNORECASE | re.MULTILINE)

    if text.startswith("> "):
        lines = text.splitlines()
        filtered = [l for l in lines if not l.startswith("> ")]
        text = "\n".join(filtered)

    # 4. Rimuove statistiche a fine generazione [ Prompt: ... | Generation: ... ]
    text = re.sub(r"\[\s*Prompt:.*?Generation:.*?\]", "", text, flags=re.IGNORECASE)

    # 5. Rimuove footer '> Exiting...' o 'Exiting...'
    text = re.sub(r">\s*Exiting\.+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"Exiting\.+", "", text, flags=re.IGNORECASE)

    # 6. Rimuove righe isolate contenenti solo '>'
    cleaned_lines = [l for l in text.splitlines() if l.strip() != ">"]
    return "\n".join(cleaned_lines).strip()


class LlamaPersistentSession:
    """
    Mantiene una singola istanza persistente di llama-cli per l'intero esperimento:
      - Carica il modello GGUF in VRAM/RAM una sola volta all'inizio dell'esperimento.
      - Rilascia subito dopo la Linux VFS Page Cache del file GGUF (posix_fadvise DONTNEED)
        così i pesi restano in GPU senza occupare due volte la memoria unificata (UMA).
      - Tra un task e il successivo invia '/clear' su stdin per azzerare e riallocare
        esclusivamente la KV cache e la history, senza deallocare/ricaricare il modello GGUF.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.llama_bin = str(config["llama_cli_path"])
        self.model_path = str(config["model_path"])
        self.ngl = int(config["n_gpu_layers"])
        self.n_threads = int(config["n_threads"])
        raw_ctx = int(config.get("ctx_size", 1024))
        # Su Jetson Orin Nano (8GB UMA), LLM-8B richiede KV cache di 128 MB (-c 1024),
        # e batch di calcolo ridotti (-b 128 -ub 64) per minimizzare i buffer di calcolo GPU (~60MB invece di ~250MB)
        # ed evitare frammentazione nvmap/UMA, completando il caricamento in ~7s al primo tentativo senza retry.
        is_8b = "8b" in self.model_path.lower() or "llm" in str(config.get("model_id", "")).lower()
        if self.ngl > 0 and is_8b:
            self.ctx_size = min(raw_ctx, 1024)
            self.b_size = int(config.get("b_size", 128))
            self.ub_size = int(config.get("ub_size", 64))
        else:
            self.ctx_size = raw_ctx
            self.b_size = int(config.get("b_size", 512))
            self.ub_size = int(config.get("ub_size", 256))
        self.temperature = float(config.get("temperature", 0.0))
        self.seed = int(config.get("seed", 42))
        self.max_tokens = int(config["max_tokens"])

        self.proc: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()
        self._boot_stdout_bytes = bytearray()
        self._turn_stdout_bytes = bytearray()
        self._stderr_bytes = bytearray()
        self._token_timestamps: List[float] = []
        self._first_token_seen = threading.Event()
        self._turn_done_event = threading.Event()
        self._collecting_turn = False
        self._stats_seen_at: Optional[float] = None
        self._prompt_tail: str = ""
        self._prompt_echo_passed: bool = False
        self._t_start_perf: float = 0.0
        self._stress_proc: Optional[subprocess.Popen] = None
        self._active_stress_cmd: Optional[List[str]] = None
        self._active_duration_mode: str = "t_tot"
        self.tasks_completed: int = 0

    def is_alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def start(self, load_timeout_sec: int = 180):
        """Avvia llama-cli in modalità interattiva e attende il caricamento del GGUF in GPU/CPU."""
        if self.is_alive():
            return

        # Tentativi ordinati di allocazione per evitare contesa su nvmap/UMA:
        # 1. Configurazione richiesta (es. -c 1024 -b 128 -ub 64 -> 128 MB KV cache)
        # 2. Secondo tentativo dopo pulizia memoria e attesa di coalescenza DMA
        # 3. Fallback di sicurezza a buffer ultra compatto (-c 512 -b 128 -ub 64)
        attempts = [
            (self.ctx_size, self.b_size, self.ub_size),
            (min(self.ctx_size, 1024), 128, 64),
            (min(self.ctx_size, 512), 128, 64)
        ]

        last_err: Optional[Exception] = None
        for attempt_idx, (c_s, b_s, ub_s) in enumerate(attempts):
            try:
                self._start_process(c_s, b_s, ub_s, load_timeout_sec)
                self.ctx_size = c_s
                self.b_size = b_s
                self.ub_size = ub_s
                return
            except Exception as err:
                last_err = err
                self.close()
                if attempt_idx < len(attempts) - 1:
                    next_c, next_b, next_ub = attempts[attempt_idx + 1]
                    print(
                        f"\n   [INFO UMA] Allocazione buffer GPU (-c {c_s}) contesa su nvmap ({err}). "
                        f"Pulizia memoria UMA, attesa 3s e retry con -c {next_c} -b {next_b} -ub {next_ub}..."
                    )
                    time.sleep(3.0)

        if last_err is not None:
            raise last_err

    def _start_process(self, ctx_s: int, b_s: int, ub_s: int, load_timeout_sec: int):
        reclaim_unified_memory(self.model_path, kill_llama=True)
        time.sleep(0.5)

        cmd = [
            self.llama_bin,
            "-m", self.model_path,
            "-c", str(ctx_s),
            "-b", str(b_s),
            "-ub", str(ub_s),
            "-ngl", str(self.ngl),
            "-t", str(self.n_threads),
            "--temp", str(self.temperature),
            "-s", str(self.seed),
            "-n", str(self.max_tokens),
        ]

        env = os.environ.copy()
        env.pop("GGML_CUDA_ENABLE_UNIFIED_MEMORY", None)
        env["LC_NUMERIC"] = "C"

        preexec = os.setsid if os.name != "nt" else None

        with self._lock:
            self._boot_stdout_bytes.clear()
            self._turn_stdout_bytes.clear()
            self._stderr_bytes.clear()
            self._token_timestamps.clear()
            self._first_token_seen.clear()
            self._turn_done_event.clear()
            self._collecting_turn = False
            self._stats_seen_at = None
            self.tasks_completed = 0

        self.proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
            preexec_fn=preexec,
            env=env
        )

        stdout_fd = self.proc.stdout.fileno()
        stderr_fd = self.proc.stderr.fileno()

        def _stdout_reader():
            try:
                while True:
                    chunk = os.read(stdout_fd, 1024)
                    if not chunk:
                        break
                    now_t = time.perf_counter()
                    with self._lock:
                        if not self._collecting_turn:
                            self._boot_stdout_bytes.extend(chunk)
                        else:
                            self._turn_stdout_bytes.extend(chunk)
                            curr_text = self._turn_stdout_bytes.decode("utf-8", errors="replace")
                            m_stats = TURN_STATS_RE.search(curr_text)
                            if m_stats:
                                if self._stats_seen_at is None:
                                    self._stats_seen_at = time.monotonic()
                                if ">" in curr_text[m_stats.end():]:
                                    self._turn_done_event.set()
                            else:
                                # Filtro anti-echo: ignora l'eco del prompt su stdout prima della generazione
                                if not self._prompt_echo_passed:
                                    tail = self._prompt_tail
                                    if tail and tail in curr_text:
                                        pos = curr_text.rfind(tail)
                                        post_text = curr_text[pos + len(tail):]
                                        post_clean = post_text.replace("\r", "").replace("\n", "").replace(">", "").strip()
                                        if post_clean:
                                            self._prompt_echo_passed = True
                                            self._token_timestamps.append(now_t)
                                            if not self._first_token_seen.is_set():
                                                self._first_token_seen.set()
                                                self._handle_stressor_first_token()
                                    else:
                                        # Nel caso in cui il prompt non sia stato stampato a terminale,
                                        # accetta token solo dopo almeno 30ms dall'invio (durata minima di prefill)
                                        if (now_t - self._t_start_perf) >= 0.03:
                                            chunk_str = chunk.decode("utf-8", errors="ignore").strip()
                                            if chunk_str and chunk_str != ">" and "\x08" not in chunk_str:
                                                self._prompt_echo_passed = True
                                                self._token_timestamps.append(now_t)
                                                if not self._first_token_seen.is_set():
                                                    self._first_token_seen.set()
                                                    self._handle_stressor_first_token()
                                else:
                                    chunk_str = chunk.decode("utf-8", errors="ignore").strip()
                                    if chunk_str and chunk_str != ">" and "\x08" not in chunk_str:
                                        self._token_timestamps.append(now_t)
            except Exception:
                pass

        def _stderr_reader():
            try:
                while True:
                    chunk = os.read(stderr_fd, 1024)
                    if not chunk:
                        break
                    with self._lock:
                        self._stderr_bytes.extend(chunk)
            except Exception:
                pass

        threading.Thread(target=_stdout_reader, daemon=True).start()
        threading.Thread(target=_stderr_reader, daemon=True).start()

        deadline = time.monotonic() + max(30, load_timeout_sec)
        ready = False
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                with self._lock:
                    err_txt = self._stderr_bytes.decode("utf-8", errors="replace")
                    out_txt = self._boot_stdout_bytes.decode("utf-8", errors="replace")
                details = err_txt[-2500:].strip() or out_txt[-2500:].strip()
                raise RuntimeError(
                    f"llama-cli terminato durante il caricamento del modello (codice {self.proc.returncode}). Dettagli: {details}"
                )

            with self._lock:
                boot_txt = self._boot_stdout_bytes.decode("utf-8", errors="replace")
                err_txt = self._stderr_bytes.decode("utf-8", errors="replace")

            if (
                "Error: the server exited before becoming ready" in boot_txt
                or "Error: the server exited before becoming ready" in err_txt
                or "failed to initialize the context" in err_txt
            ):
                time.sleep(0.2)
                self.close()
                details = err_txt[-2500:].strip() or boot_txt[-2500:].strip()
                raise RuntimeError(f"llama-cli non riuscito ad allocare il contesto GPU/CPU. Dettagli: {details}")

            if "available commands:" in boot_txt:
                after_cmd = boot_txt.split("available commands:", 1)[1]
                if ">" in after_cmd:
                    ready = True
                    break

            time.sleep(0.05)

        if not ready:
            self.close()
            raise TimeoutError(f"Timeout ({load_timeout_sec}s) durante il caricamento iniziale del modello GGUF.")

        # Una volta che il GGUF è stato caricato nella VRAM della GPU, rilascia la VFS Page Cache
        # del file GGUF senza terminare llama-cli (kill_llama=False).
        reclaim_unified_memory(self.model_path, kill_llama=False)

    def reset_kv_cache(self):
        """
        Invia '/clear' alla sessione interattiva di llama-cli per svuotare la KV cache
        e azzerare la history prima del nuovo task, mantenendo il modello GGUF in GPU.
        """
        if not self.is_alive() or self.proc.stdin is None:
            return

        with self._lock:
            self._collecting_turn = False
            self._boot_stdout_bytes.clear()
            self._turn_stdout_bytes.clear()

        try:
            self.proc.stdin.write(b"/clear\n")
            self.proc.stdin.flush()
        except Exception:
            return

        clear_deadline = time.monotonic() + 1.5
        while time.monotonic() < clear_deadline:
            with self._lock:
                txt = self._boot_stdout_bytes.decode("utf-8", errors="replace")
            if ">" in txt:
                time.sleep(0.05)
                break
            time.sleep(0.02)

        with self._lock:
            self._boot_stdout_bytes.clear()
            self._turn_stdout_bytes.clear()

    def _handle_stressor_first_token(self):
        """Interrompe lo stressor per t_prefill o lo avvia per t_decode non appena arriva il vero primo token."""
        if self._active_stress_cmd and self._active_duration_mode == "t_prefill":
            stop_process_tree(self._stress_proc)
            self._stress_proc = None
        elif self._active_stress_cmd and self._active_duration_mode == "t_decode":
            preexec = os.setsid if os.name != "nt" else None
            try:
                self._stress_proc = subprocess.Popen(
                    self._active_stress_cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    stdin=subprocess.DEVNULL,
                    preexec_fn=preexec
                )
            except Exception as e_dec:
                log_fallback_error("start_stress_ng_decode", e_dec)

    def run_task(
        self,
        prompt: str,
        stress_cmd: Optional[List[str]],
        duration_mode: str,
        task_timeout_sec: int
    ) -> Dict[str, Any]:
        """Esegue un singolo task nella sessione persistente, pulendo la KV cache prima dell'invio."""
        if not self.is_alive():
            self.start()

        if self.tasks_completed > 0:
            self.reset_kv_cache()

        prompt_line = " ".join(line.strip() for line in prompt.splitlines() if line.strip())
        prompt_words = prompt_line.strip().split()
        prompt_tail = prompt_words[-1] if prompt_words else ":"
        if len(prompt_line.strip()) >= 20:
            prompt_tail = prompt_line.strip()[-20:]

        preexec = os.setsid if os.name != "nt" else None
        t_start_mono = time.monotonic()
        t_start_perf = time.perf_counter()

        with self._lock:
            self._boot_stdout_bytes.clear()
            self._turn_stdout_bytes.clear()
            self._stderr_bytes.clear()
            self._token_timestamps.clear()
            self._first_token_seen.clear()
            self._turn_done_event.clear()
            self._stats_seen_at = None
            self._prompt_tail = prompt_tail
            self._prompt_echo_passed = False
            self._t_start_perf = t_start_perf
            self._active_stress_cmd = stress_cmd
            self._active_duration_mode = duration_mode
            self._collecting_turn = True

        if stress_cmd and duration_mode in ("t_prefill", "t_tot"):
            try:
                self._stress_proc = subprocess.Popen(
                    stress_cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    stdin=subprocess.DEVNULL,
                    preexec_fn=preexec
                )
            except Exception as e:
                log_fallback_error("start_stress_ng_prefill", e)

        try:
            self.proc.stdin.write((prompt_line + "\n").encode("utf-8"))
            self.proc.stdin.flush()

            deadline = time.monotonic() + max(10, task_timeout_sec)
            while True:
                if self._turn_done_event.wait(timeout=0.05):
                    break

                with self._lock:
                    stats_t = self._stats_seen_at
                if stats_t is not None and (time.monotonic() - stats_t) >= 0.25:
                    break

                if self.proc.poll() is not None:
                    with self._lock:
                        err_txt = self._stderr_bytes.decode("utf-8", errors="replace")
                        out_txt = self._turn_stdout_bytes.decode("utf-8", errors="replace")
                    details = err_txt[-2500:].strip() or out_txt[-2500:].strip()
                    raise RuntimeError(
                        f"llama-cli terminato inaspettatamente durante il task (codice {self.proc.returncode}). Dettagli: {details}"
                    )

                if time.monotonic() > deadline:
                    self.close()
                    raise TimeoutError(f"Task llama-cli scaduto per timeout (>{task_timeout_sec}s).")

        finally:
            with self._lock:
                self._collecting_turn = False
                stdout_text = self._turn_stdout_bytes.decode("utf-8", errors="replace")
                stderr_text = self._stderr_bytes.decode("utf-8", errors="replace")
                token_timestamps = list(self._token_timestamps)
            stop_process_tree(self._stress_proc)
            self._stress_proc = None

        t_end_mono = time.monotonic()
        t_end_perf = time.perf_counter()
        self.tasks_completed += 1

        clean_resp = clean_llama_response(stdout_text, prompt)
        eval_tokens = max(1, len(token_timestamps) or len(clean_resp.split()))
        est_words = max(1, len(prompt_line.split()))

        ttft_ms = 0.0
        prompt_tokens = 0
        prompt_ms_per_tok = 0.0
        eval_total_ms = 0.0
        per_token_lat_ms = 0.0
        throughput_tok_s = 0.0

        # 1. Parsing metriche ufficiali da stderr di llama.cpp (se presenti log prompt eval time)
        m_prompt = PROMPT_EVAL_RE.search(stderr_text)
        if m_prompt:
            ttft_ms = float(m_prompt.group(1))
            prompt_tokens = int(m_prompt.group(2))
            prompt_ms_per_tok = float(m_prompt.group(3))

        # 2. Parsing da [ Prompt: X t/s | Generation: Y t/s ] (formato interattivo standard llama-cli)
        m_turn = TURN_STATS_RE.search(stdout_text)
        prompt_tps = 0.0
        gen_tps = 0.0
        if m_turn:
            prompt_tps = float(m_turn.group(1).replace(",", "."))
            gen_tps = float(m_turn.group(2).replace(",", "."))

        # Calcolo del TTFT reale (priorità al timestamp hardware del primo token post-prompt)
        if ttft_ms <= 0.0:
            if token_timestamps:
                raw_ttft = (token_timestamps[0] - t_start_perf) * 1000.0
                if raw_ttft >= 5.0:
                    ttft_ms = raw_ttft
                elif prompt_tps > 0:
                    # In caso di timestamp sporcato da eco immediato, calcola il prefill reale con il timer hardware di llama.cpp
                    ttft_ms = (est_words / prompt_tps) * 1000.0
                else:
                    ttft_ms = max(5.0, raw_ttft)
            elif prompt_tps > 0:
                ttft_ms = (est_words / prompt_tps) * 1000.0
            else:
                ttft_ms = max(5.0, (t_end_perf - t_start_perf) * 1000.0)

        # Token di prompt e latenza per token di prompt
        if prompt_tps > 0:
            prompt_ms_per_tok = 1000.0 / prompt_tps
            prompt_tokens = max(1, int(round((ttft_ms / 1000.0) * prompt_tps)))
        elif prompt_tokens <= 0:
            prompt_tokens = est_words
            prompt_ms_per_tok = ttft_ms / prompt_tokens

        # Throughput di generazione e TPOT
        if gen_tps > 0:
            throughput_tok_s = gen_tps
            per_token_lat_ms = 1000.0 / gen_tps
            eval_total_ms = per_token_lat_ms * eval_tokens
        else:
            m_eval = EVAL_TIME_RE.search(stderr_text)
            if m_eval:
                eval_total_ms = float(m_eval.group(1))
                eval_tokens = int(m_eval.group(2))
                per_token_lat_ms = float(m_eval.group(3))
                throughput_tok_s = float(m_eval.group(4))
            else:
                if len(token_timestamps) >= 2:
                    eval_total_ms = max(0.1, (token_timestamps[-1] - token_timestamps[0]) * 1000.0)
                else:
                    eval_total_ms = max(0.1, (t_end_perf - t_start_perf) * 1000.0 - ttft_ms)
                per_token_lat_ms = eval_total_ms / eval_tokens
                throughput_tok_s = 1000.0 / per_token_lat_ms if per_token_lat_ms > 0 else 0.0

        # 4. Calcolo dell'Inter-Token Jitter
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
            "response_text": clean_resp,
        }

    def close(self):
        """Chiude pulitamente la sessione interattiva di llama-cli e libera la memoria UMA."""
        stop_process_tree(self._stress_proc)
        self._stress_proc = None
        if self.proc is not None:
            try:
                if self.proc.poll() is None and self.proc.stdin is not None:
                    self.proc.stdin.write(b"/exit\n")
                    self.proc.stdin.flush()
                    self.proc.wait(timeout=3)
            except Exception:
                pass
            stop_process_tree(self.proc)
            self.proc = None
        reclaim_unified_memory(self.model_path, kill_llama=True)


def _detect_llama_cli_extra_flags(llama_cli_path: str) -> List[str]:
    """
    Rileva una sola volta i flag supportati dalla versione di llama-cli installata:
      - '--single-turn' o '-st': disabilita la chat interattiva a turni multipli.
      - '-no-cnv' o '--no-conversation': disabilita la modalità conversazione automatica.
      - '--perf': assicura la stampa delle statistiche di timing su stderr.
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

        # Disabilitazione modalità chat interattiva
        if "--single-turn" in help_text:
            extra.append("--single-turn")
        elif "-st" in help_text:
            extra.append("-st")

        if "-no-cnv" in help_text:
            extra.append("-no-cnv")
        elif "--no-conversation" in help_text:
            extra.append("--no-conversation")

        # Stampa metriche prestazionali
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
    override_threads: Optional[int] = None,
    override_ctx_size: Optional[int] = None,
    override_batch: Optional[int] = None,
    override_ubatch: Optional[int] = None
) -> Dict[str, float]:
    """Esegue una singola invocazione di llama-cli con watchdog di timeout e fallback wall-clock."""
    llama_bin = config["llama_cli_path"]
    ngl = override_ngl if override_ngl is not None else config["n_gpu_layers"]
    n_threads = override_threads if override_threads is not None else config["n_threads"]
    raw_ctx = int(config.get("ctx_size", 1024))
    is_8b = "8b" in str(config.get("model_path", "")).lower() or "llm" in str(config.get("model_id", "")).lower()
    ctx_size = override_ctx_size if override_ctx_size is not None else (min(raw_ctx, 1024) if (ngl > 0 and is_8b) else raw_ctx)
    default_b = 128 if (ngl > 0 and is_8b) else 512
    default_ub = 64 if (ngl > 0 and is_8b) else 256
    b_size = override_batch if override_batch is not None else int(config.get("b_size", default_b))
    ub_size = override_ubatch if override_ubatch is not None else int(config.get("ub_size", default_ub))

    # Libera la Linux VFS Page Cache del file GGUF prima di allocare i buffer CUDA su UMA
    reclaim_unified_memory(config.get("model_path"))

    cmd = [
        llama_bin,
        "-m", config["model_path"],
        "-c", str(ctx_size),
        "-b", str(b_size),
        "-ub", str(ub_size),
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

    env = os.environ.copy()
    env.pop("GGML_CUDA_ENABLE_UNIFIED_MEMORY", None)

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
            preexec_fn=preexec,
            env=env
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
                model_ready = False
                while True:
                    b = llama_proc.stdout.read(16)
                    if not b:
                        break
                    stdout_chunks.append(b)

                    # Ignora caratteri di backspace o spinner durante il caricamento del modello
                    if not model_ready:
                        chunk_str = b.decode("utf-8", errors="ignore")
                        if b"\x08" in b or "Loading model" in chunk_str or chunk_str in ("|", "/", "-", "\\"):
                            continue
                        model_ready = True

                    now_t = time.perf_counter()
                    token_timestamps.append(now_t)

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

        # Controllo CRASH: verifica codice di ritorno e messaggi fatali del server
        server_exited_error = (
            "Error: the server exited before becoming ready" in stdout_text
            or "Error: the server exited before becoming ready" in stderr_text
            or "error: the server exited" in stdout_text.lower()
            or "error: the server exited" in stderr_text.lower()
        )
        if llama_proc.returncode != 0 or server_exited_error:
            err_msg = stderr_text[-2500:].strip() or stdout_text[-2500:].strip()
            raise RuntimeError(
                f"llama-cli terminato con errore (codice {llama_proc.returncode}). Dettagli: {err_msg}"
            )

    except subprocess.TimeoutExpired as te:
        stop_process_tree(llama_proc)
        raise TimeoutError(f"Task llama-cli scaduto per timeout (>{task_timeout_sec}s).") from te
    finally:
        stop_process_tree(stress_proc)
        stop_process_tree(llama_proc)
        reclaim_unified_memory(config.get("model_path"))

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

    clean_resp = clean_llama_response(stdout_text, prompt)
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
        "response_text": clean_resp,
    }


def run_single_llama_task(
    config: Dict[str, Any],
    prompt: str,
    stress_cmd: Optional[List[str]] = None,
    duration_mode: str = "t_tot",
    task_timeout_sec: int = 300,
    exp_id: str = "N/A",
    session: Optional[LlamaPersistentSession] = None
) -> Optional[Dict[str, Any]]:
    """
    Esegue un singolo task su llama-cli con timeout di 5 minuti (300s):
      - Se viene fornita una LlamaPersistentSession attiva, mantiene il modello GGUF
        allocato in VRAM/RAM e resetta solo la KV cache (/clear) prima del nuovo task.
      - Se il task supera il timeout di 5 minuti (300s), il processo viene sospeso/terminato
        e la funzione restituisce None per passare immediatamente al task successivo.
      - Se il device è GPU e si verifica un errore di esecuzione GPU, lo script fallisce
        e riporta l'errore (nessun passaggio silenzioso a CPU).
    """
    try:
        if session is not None:
            return session.run_task(
                prompt=prompt,
                stress_cmd=stress_cmd,
                duration_mode=duration_mode,
                task_timeout_sec=task_timeout_sec
            )
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
            print(f"\n   [WARNING] Errore esecuzione GPU nel task ({err}). Riavvio sessione GPU e retry del task...")
            if session is not None:
                session.close()
                time.sleep(2.0)
                try:
                    session.start()
                    res = session.run_task(
                        prompt=prompt,
                        stress_cmd=stress_cmd,
                        duration_mode=duration_mode,
                        task_timeout_sec=task_timeout_sec
                    )
                    print("   [RETRY GPU OK] Task completato con successo su GPU al secondo tentativo.")
                    return res
                except Exception as second_gpu_err:
                    log_fallback_error("run_single_llama_task_gpu_retry_failed", second_gpu_err, exp_id=exp_id)
                    raise ExperimentRuntimeError(
                        f"Errore di esecuzione su device GPU durante il task: {second_gpu_err}"
                    ) from second_gpu_err

            reclaim_unified_memory(config.get("model_path"))
            time.sleep(3.0)
            try:
                res = _run_llama_cli_once(
                    config=config,
                    prompt=prompt,
                    stress_cmd=stress_cmd,
                    duration_mode=duration_mode,
                    task_timeout_sec=task_timeout_sec,
                    override_ctx_size=1024,
                    override_batch=128,
                    override_ubatch=64
                )
                print("   [RETRY GPU OK] Task completato con successo su GPU al secondo tentativo.")
                return res
            except Exception as second_gpu_err:
                log_fallback_error("run_single_llama_task_gpu_retry_failed", second_gpu_err, exp_id=exp_id)
                raise ExperimentRuntimeError(
                    f"Errore di esecuzione su device GPU durante il task: {second_gpu_err}"
                ) from second_gpu_err

        # Solo se siamo su CPU, effettua 1 tentativo di fallback prima di arrestare lo script
        print(f"\n   [WARNING] Errore runtime nel task su CPU ({err}). Avvio singolo tentativo di fallback...")
        time.sleep(3.0)
        if session is not None:
            session.close()
            try:
                session.start()
                res = session.run_task(
                    prompt=prompt,
                    stress_cmd=stress_cmd,
                    duration_mode=duration_mode,
                    task_timeout_sec=task_timeout_sec
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
def wait_for_post_boot_transient(wait_seconds: int, model_path: Optional[str] = None):
    """
    Attende che il transitorio del sistema operativo e termico post-boot si esaurisca
    prima di avviare la raccolta delle metriche con vmstat e llama.cpp.
    Esegue la pulizia immediata della cache UMA e dei file GGUF all'inizio e alla fine del transitorio.
    """
    reclaim_unified_memory(model_path, kill_llama=True)

    if wait_seconds <= 0:
        return

    print("\n" + "-" * 65)
    print(f" [TRANSITORIO] Attesa di {wait_seconds}s per stabilizzazione OS e termica...")
    print("-" * 65)
    print(" [OK] Pulizia immediata memoria UMA eseguita (file GGUF e Page Cache rilasciati).")

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

    reclaim_unified_memory(model_path, kill_llama=True)
    print(" [OK] Transitorio terminato. Memoria pulita e sistema pronto per l'esperimento.\n")


def trigger_board_reboot():
    """Sincronizza i dischi ed esegue il riavvio della board con fallback a catena."""
    print("\n" + "=" * 65)
    print(" [REBOOT] Sincronizzazione filesystem e riavvio della board in corso...")
    print("=" * 65)
    cleanup_orphan_stressors()
    if os.name != "nt":
        try:
            subprocess.run(["sync"], check=False, timeout=10)

            # Tentativo 1: sudo -n reboot (non interattivo, funziona se NOPASSWD è configurato)
            res1 = subprocess.run(["sudo", "-n", "reboot"], capture_output=True, text=True, timeout=10)
            if res1.returncode == 0:
                sys.exit(0)

            # Tentativo 2: systemctl reboot
            res2 = subprocess.run(["systemctl", "reboot"], capture_output=True, text=True, timeout=10)
            if res2.returncode == 0:
                sys.exit(0)

            # Tentativo 3: sudo reboot interattivo (se presente un terminale TTY per inserire password)
            print(" [INFO] Tentativo di reboot con sudo (potrebbe richiedere la password)...")
            res3 = subprocess.run(["sudo", "reboot"], check=False, timeout=15)
            if res3.returncode == 0:
                sys.exit(0)

            # Se tutti i tentativi sono falliti:
            err_msg1 = (res1.stderr or "").strip()
            err_msg2 = (res2.stderr or "").strip()
            print("\n" + "!" * 65)
            print(" [WARNING REBOOT] Tutti i tentativi automatici di reboot sono falliti.")
            if err_msg1:
                print(f" Dettaglio sudo: {err_msg1}")
            if err_msg2:
                print(f" Dettaglio systemctl: {err_msg2}")
            print("\n Per consentire alla Jetson di riavviarsi da sola senza password tra un run e l'altro,")
            print(" esegui una volta sulla shell della board:")
            print("   echo \"$USER ALL=(ALL) NOPASSWD: /sbin/reboot, /bin/systemctl reboot\" | sudo tee /etc/sudoers.d/jetson-reboot")
            print("!" * 65 + "\n")
            return
        except Exception as e:
            log_fallback_error("trigger_board_reboot", e)
            print(f" [WARNING] Eccezione durante il reboot automatico ({e}). Proseguimento senza reboot.")
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
        if config.get("no_ssh"):
            return False

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

        # Rilevamento sistema host di destinazione (Windows con drive letter C:/... vs Linux /...)
        is_windows_dest = bool(re.match(r"^[a-zA-Z]:", ssh_dest)) or ("\\" in ssh_dest)
        if is_windows_dest:
            win_dirs = [d.replace("/", "\\") for d in remote_dirs_to_create]
            mkdir_cmd_str = "cmd.exe /c \"" + " & ".join(f'if not exist "{d}" mkdir "{d}"' for d in win_dirs) + "\""
        else:
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
    if config.get("no_ssh"):
        print(" [INFO] Sincronizzazione SSH disabilitata (--no-ssh attivo).")
        return

    files_to_send: List[Tuple[Path, str]] = []

    if TIMESERIES_DIR.exists():
        for p in sorted(TIMESERIES_DIR.glob("*.csv")):
            files_to_send.append((p, "data_timeseries"))

    if RESULTS_DIR.exists():
        for p in sorted(RESULTS_DIR.glob("*.csv")):
            files_to_send.append((p, "results"))
        for p in sorted(RESULTS_DIR.glob("responses_*.json")):
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
    """
    Costruisce la lista di esperimenti da eseguire (ciascuno ripetuto per config['repetitions'] volte).
    Ciascuna famiglia di stressor viene eseguita con tutti i suoi metodi simultaneamente:
      - Famiglie 2D (cpu, cache, os, interrupt): 4 (N) * 1 (D) * 3 reps = 12 run per famiglia
      - Famiglie 3D (mem, io): 4 (N) * 3 (S) * 1 (D) * 3 reps = 36 run per famiglia
      - Modalità all-stressor: 12 + 36 + 12 + 36 + 12 + 12 = 120 run totali
    """
    reps = int(config.get("repetitions", 3))
    queue: List[Dict[str, Any]] = []

    model_clean = str(config.get("model_id", "slm-1b")).strip().lower()
    device_clean = str(config.get("device", "gpu")).strip().lower()
    bench_clean = str(config.get("benchmark_name", "mmlu")).strip().lower()

    if not config.get("use_stressor", False) or config.get("stressor_class") in ("none", "baseline"):
        for rep in range(1, reps + 1):
            exp_id = f"baseline_{model_clean}_{device_clean}_{bench_clean}_rep{rep}"
            queue.append({
                "exp_id": exp_id,
                "mode": "baseline",
                "model_id": config.get("model_id", "slm-1b"),
                "device": config.get("device", "gpu"),
                "benchmark": config.get("benchmark_name", "mmlu"),
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
        return queue

    s_clean = str(config.get("stressor_class", "cpu")).strip().lower()

    if s_clean in ("all", "all-stressor", "all-stressors", "all_stressor", "all_stressors"):
        classes_to_run = STRESSOR_FAMILY_ORDER
    elif s_clean in ("sched", "scheduler"):
        classes_to_run = ["os"]
    elif s_clean in ("memoria", "memory"):
        classes_to_run = ["mem"]
    elif s_clean in ("matrix", "fft"):
        classes_to_run = ["cpu"]
    elif s_clean in ("vm", "stream", "memrate", "mem_rate"):
        classes_to_run = ["mem"]
    elif s_clean in ("l1cache", "l1-cache", "llc-affinity", "llc_affinity"):
        classes_to_run = ["cache"]
    elif s_clean in ("iomix", "hdd", "disk"):
        classes_to_run = ["io"]
    elif s_clean in ("fork", "switch"):
        classes_to_run = ["os"]
    elif s_clean in ("timer", "sigq", "interrupts", "irq"):
        classes_to_run = ["interrupt"]
    elif s_clean in STRESSOR_FAMILY_ORDER:
        classes_to_run = [s_clean]
    else:
        classes_to_run = ["cpu"]

    for s_class in classes_to_run:
        methods = get_methods_for_class(s_class)
        s_method_label = "+".join(methods)
        # Classi 3D (con fattore di working-set S): mem, io
        has_size = s_class in ("mem", "io")
        s_iter = S_LEVELS if has_size else [("N/A", "N/A", None)]

        for n_tag, n_label, n_frac in N_LEVELS:
            for s_tag, s_label, s_frac in s_iter:
                for dur in D_LEVELS:
                    for rep in range(1, reps + 1):
                        if has_size:
                            exp_id = f"run_{model_clean}_{device_clean}_{bench_clean}_{s_class}_{n_tag}_{dur}_{s_tag}_rep{rep}"
                        else:
                            exp_id = f"run_{model_clean}_{device_clean}_{bench_clean}_{s_class}_{n_tag}_{dur}_rep{rep}"

                        queue.append({
                            "exp_id": exp_id,
                            "mode": "stressor",
                            "model_id": config.get("model_id", "slm-1b"),
                            "device": config.get("device", "gpu"),
                            "benchmark": config.get("benchmark_name", "mmlu"),
                            "stressor_class": s_class,
                            "stressor_method": s_method_label,
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
    exc: BaseException,
    partial_responses: Optional[List[Dict[str, Any]]] = None
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

    # 4b. Salvataggio risposte del modello parziali
    if partial_responses:
        try:
            RESULTS_DIR.mkdir(parents=True, exist_ok=True)
            partial_resp_path = RESULTS_DIR / f"fallback_partial_{exp_id}_responses.json"
            with open(partial_resp_path, "w", encoding="utf-8") as f:
                json.dump(partial_responses, f, indent=2, ensure_ascii=False)
                f.flush()
                os.fsync(f.fileno())
            fallback_files.append((partial_resp_path, "results"))
            print(f" [FALLBACK] Salvate {len(partial_responses)} risposte parziali in: {partial_resp_path}")
        except Exception as e_resp:
            log_fallback_error("save_partial_responses", e_resp, exp_id=exp_id)

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
    llama_session: Optional[LlamaPersistentSession] = None
    task_records: List[Dict[str, Any]] = []
    responses_log: List[Dict[str, Any]] = []

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

        ensure_jetson_performance_mode()

        print("\n [SESSIONE GGUF] Allocazione modello GGUF in memoria (caricamento unico per l'esperimento)...")
        t_load_0 = time.perf_counter()
        llama_session = LlamaPersistentSession(config)
        llama_session.start()
        t_load_sec = time.perf_counter() - t_load_0
        print(f" [OK] Modello GGUF allocato in {t_load_sec:.1f}s. Tra un task e il successivo verrà resettata solo la KV cache (/clear).")

        monitor = SystemMonitor()
        monitor.start()

        for i in range(n_samples):
            sample_item = dataset[i % len(dataset)]
            prompt = format_mmlu_prompt(sample_item) if bench_lower == "mmlu" else format_spider_prompt(sample_item)

            # Estrazione ground truth (risposta attesa)
            ground_truth = ""
            if bench_lower == "mmlu":
                raw_ans = sample_item.get("answer")
                if isinstance(raw_ans, int) and 0 <= raw_ans < 4:
                    ground_truth = ["A", "B", "C", "D"][raw_ans]
                elif raw_ans is not None:
                    ground_truth = str(raw_ans)
            else:
                ground_truth = str(sample_item.get("query", ""))

            print(f" -> Task [{i + 1:02d}/{n_samples:02d}] in esecuzione...", end="", flush=True)
            metrics = run_single_llama_task(
                config=config,
                prompt=prompt,
                stress_cmd=stress_cmd,
                duration_mode=exp["duration"],
                task_timeout_sec=task_timeout_sec,
                exp_id=exp["exp_id"],
                session=llama_session
            )

            # Se il task ha superato il timeout di 5 minuti (300s), viene sospeso e si passa al successivo
            if metrics is None:
                continue

            vm_avg = monitor.get_window_average(metrics["t_start_mono"], metrics["t_end_mono"])
            model_resp = metrics.get("response_text", "")

            record = {
                "task_index": i + 1,
                "sample_id": sample_item.get("id", i + 1),
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
                "ground_truth": ground_truth,
                "model_response": model_resp,
            }
            for k, v in vm_avg.items():
                record[f"vmstat_{k}"] = v

            task_records.append(record)

            responses_log.append({
                "task_index": i + 1,
                "sample_id": sample_item.get("id", i + 1),
                "benchmark": config["benchmark_name"],
                "prompt": prompt,
                "model_response": model_resp,
                "ground_truth": ground_truth,
                "ttft_ms": metrics["ttft_ms"],
                "per_token_lat_ms": metrics["per_token_lat_ms"],
                "throughput_tok_s": metrics["throughput_tok_s"],
                "jitter_ms": metrics["jitter_ms"],
                "output_tokens": metrics["output_tokens"],
                "task_duration_s": metrics["duration_s"],
            })

            print(
                f" TTFT={metrics['ttft_ms']:.1f}ms | "
                f"TPOT={metrics['per_token_lat_ms']:.2f}ms/tok | "
                f"Thr={metrics['throughput_tok_s']:.2f}tok/s | "
                f"Jitter={metrics['jitter_ms']:.2f}ms"
            )

            time.sleep(0.3)
            cleanup_orphan_stressors()

        if llama_session is not None:
            llama_session.close()
            llama_session = None
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
        rep_num = exp.get("repetition", 1)
        tasks_rep_csv_path: Optional[Path] = None

        if exp["mode"] == "baseline":
            # File cumulativo di riepilogo
            tasks_csv_path = RESULTS_DIR / f"baseline_{config['model_id'].lower()}_{config['device']}_{bench_lower}_tasks.csv"
            # File specifico e isolato per la singola ripetizione
            tasks_rep_csv_path = RESULTS_DIR / f"baseline_{config['model_id'].lower()}_{config['device']}_{bench_lower}_rep{rep_num}_tasks.csv"
            responses_json_path = RESULTS_DIR / f"responses_baseline_{config['model_id'].lower()}_{config['device']}_{bench_lower}_rep{rep_num}.json"
        else:
            # Per gli stressor, l'exp_id contiene già modello, device, benchmark, classe, N, D, [S], rep
            # es. run_slm-1b_gpu_mmlu_cpu_N25_t_tot_rep1_tasks.csv
            tasks_csv_path = RESULTS_DIR / f"{exp['exp_id']}_tasks.csv"
            responses_json_path = RESULTS_DIR / f"responses_{exp['exp_id']}.json"

        # 2a. Salvataggio su file dedicato della singola ripetizione (se baseline)
        if task_records and tasks_rep_csv_path is not None:
            fieldnames = list(task_records[0].keys())
            with open(tasks_rep_csv_path, "w", newline="", encoding="utf-8") as f_rep:
                writer_rep = csv.DictWriter(f_rep, fieldnames=fieldnames)
                writer_rep.writeheader()
                writer_rep.writerows(task_records)
                f_rep.flush()
                os.fsync(f_rep.fileno())
            print(f" [OK] Metriche singola ripetizione {rep_num} ({len(task_records)} task) salvate in: {tasks_rep_csv_path}")

        # 2b. Salvataggio su tasks_csv_path (file primario o cumulativo)
        write_header = not tasks_csv_path.exists() or exp["mode"] != "baseline" or rep_num == 1
        mode_flag = "w" if (exp["mode"] != "baseline" or rep_num == 1 or not tasks_csv_path.exists()) else "a"
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

        # Salvataggio file JSON strutturato con le risposte complete del modello (specifico per singola ripetizione)
        if responses_log:
            with open(responses_json_path, "w", encoding="utf-8") as f:
                json.dump(responses_log, f, indent=2, ensure_ascii=False)
                f.flush()
                os.fsync(f.fileno())
            print(f" [OK] Risposte del modello ({len(responses_log)} risposte) salvate in: {responses_json_path}")

        # Per retrocompatibilità con script che cercano il prefisso tasks_ (es. tasks_run_*.csv)
        alt_tasks_path: Optional[Path] = None
        if exp["mode"] != "baseline" and task_records and tasks_csv_path.exists():
            try:
                alt_tasks_path = RESULTS_DIR / f"tasks_{exp['exp_id']}.csv"
                if alt_tasks_path != tasks_csv_path:
                    shutil.copy2(tasks_csv_path, alt_tasks_path)
            except Exception:
                alt_tasks_path = None

        # 3. Aggregazione sulla Run per il DoE
        generated_files: List[Tuple[Path, str]] = [
            (ts_path, "data_timeseries"),
            (tasks_csv_path, "results"),
            (responses_json_path, "results"),
        ]
        if tasks_rep_csv_path is not None:
            generated_files.append((tasks_rep_csv_path, "results"))
        if alt_tasks_path is not None and alt_tasks_path.exists():
            generated_files.append((alt_tasks_path, "results"))

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

            doe_row = {
                "stressor_class": exp["stressor_class"],
                "stressor_method": exp["stressor_method"],
                "num_workers": exp["n_label"],
                "size": exp["s_label"],
                "duration": exp["duration"],
                "ttft_ms": mean_ttft,
                "per_token_lat_ms": mean_tpot,
                "throughput_tok_s": mean_thr,
                "jitter_ms": mean_jitter,
            }

            # 1. DoE per classe/famiglia (es. doe_slm-1b_gpu_mmlu_cpu.csv)
            doe_csv_path = RESULTS_DIR / f"doe_{config['model_id'].lower()}_{config['device']}_{bench_lower}_{exp['stressor_class']}.csv"
            doe_exists = doe_csv_path.exists()
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
        if llama_session is not None:
            try:
                llama_session.close()
            except Exception:
                pass
        trigger_emergency_fallback(
            config=config,
            state=state,
            exp=exp,
            monitor=monitor,
            partial_task_records=task_records,
            exc=exc,
            partial_responses=responses_log
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
        help=(
            "Quale stressor avviare: 'cpu', 'mem', 'cache', 'io', 'os', 'interrupt', 'all-stressor' "
            "(l'esecuzione viene effettuata sull'intera classe di appartenenza)"
        )
    )
    parser.add_argument(
        "--all-stressor", "--all-stressors",
        dest="all_stressor",
        action="store_true",
        default=False,
        help="Esegue tutte le famiglie di stressor in sequenza (cpu -> mem -> cache -> io -> os -> interrupt)"
    )
    parser.add_argument("-n", "--samples", type=int, default=None, help="Numero di campioni (task) da eseguire per esperimento")
    parser.add_argument("-r", "--repetitions", type=int, default=None, help="Numero di ripetizioni per esperimento (default: 3)")
    parser.add_argument("--transient-wait", type=int, default=None, help="Attesa transitorio post-boot in secondi (default: 90s per Jetson Orin)")
    parser.add_argument("--task-timeout", type=int, default=300, help="Timeout massimo in secondi per singolo task llama-cli (default: 300s = 5 min)")
    reboot_grp = parser.add_mutually_exclusive_group()
    reboot_grp.add_argument("--reboot", dest="reboot", action="store_true", default=None, help="Forza il riavvio della board a fine esperimento")
    reboot_grp.add_argument("--no-reboot", dest="reboot", action="store_false", help="Non riavviare la board a fine esperimento (esegui tutta la coda in sequenza)")
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
    parser.add_argument(
        "-c", "--ctx-size",
        type=int,
        default=None,
        help="Dimensione della finestra di contesto per llama.cpp (default: da run_config.json o 1024)"
    )
    parser.add_argument(
        "--no-ssh",
        action="store_true",
        default=False,
        help="Disabilita l'invio dei file dei risultati via SSH/SCP al termine degli esperimenti"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Numero massimo di token in output (-n). Default: da run_config o 64 (MMLU) / 256 (Spider)"
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
            or args.all_stressor
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
            or args.ctx_size is not None
            or args.max_tokens is not None
            or args.no_ssh
        )

        if need_reinstantiate:
            model_in = args.model or current_cfg.get("model_id", "slm")
            device_in = args.device or current_cfg.get("device", "gpu")
            bench_in = args.benchmark or current_cfg.get("benchmark_name", "mmlu")

            if args.all_stressor:
                use_stressor_in = True
                stressor_in = "all-stressor"
            elif args.use_stressor is not None:
                use_stressor_in = args.use_stressor
                stressor_in = args.stressor or current_cfg.get("stressor_class", "cpu")
            elif args.stressor is not None:
                use_stressor_in = args.stressor.strip().lower() not in ("none", "baseline", "off", "false")
                stressor_in = args.stressor
            else:
                use_stressor_in = current_cfg.get("use_stressor", False)
                stressor_in = current_cfg.get("stressor_class", "cpu")
            samples_in = args.samples if args.samples is not None else current_cfg.get("sample_count", 40)
            reps_in = args.repetitions if args.repetitions is not None else current_cfg.get("repetitions", 3)
            wait_in = args.transient_wait if args.transient_wait is not None else current_cfg.get("transient_wait_sec", 90)
            reboot_in = args.reboot if args.reboot is not None else current_cfg.get("reboot_after_run", True)
            no_ssh_in = bool(args.no_ssh or current_cfg.get("no_ssh", False))
            ssh_target_in = None if no_ssh_in else (args.ssh_target if args.ssh_target is not None else current_cfg.get("ssh_target"))
            ssh_dest_in = args.ssh_dest if args.ssh_dest is not None else current_cfg.get("ssh_dest", "./EdgeAI_Results")
            ssh_port_in = args.ssh_port if args.ssh_port is not None else current_cfg.get("ssh_port", 22)
            ssh_key_in = args.ssh_key if args.ssh_key is not None else current_cfg.get("ssh_key")
            ctx_size_in = args.ctx_size if args.ctx_size is not None else current_cfg.get("ctx_size", 1024)
            max_tokens_in = args.max_tokens if args.max_tokens is not None else current_cfg.get("max_tokens")

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
                ssh_key=ssh_key_in,
                no_ssh=no_ssh_in,
                ctx_size=ctx_size_in,
                max_tokens=max_tokens_in
            )
            with open(CONFIG_OUTPUT_PATH, "r", encoding="utf-8") as f:
                config = json.load(f)
        else:
            config = current_cfg
            if args.reboot is not None:
                config["reboot_after_run"] = args.reboot
            if args.ctx_size is not None:
                config["ctx_size"] = args.ctx_size
            if args.max_tokens is not None:
                config["max_tokens"] = args.max_tokens
            if args.no_ssh:
                config["no_ssh"] = True
                config["ssh_target"] = None

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

        # Pulizia immediata preventiva all'avvio della sessione
        reclaim_unified_memory(config.get("model_path"), kill_llama=True)
        ensure_jetson_performance_mode()

        while pending:
            next_exp = pending[0]

            # 1. Attesa transitorio post-boot (default 90s su Jetson Orin)
            wait_for_post_boot_transient(int(config.get("transient_wait_sec", 90)), config.get("model_path"))

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
                    trigger_board_reboot()
                else:
                    print("\n" + "=" * 65)
                    print(" [FINITO] Tutti gli esperimenti della coda sono stati completati con successo!")
                    print("          Nessun ulteriore riavvio necessario. Campagna DoE terminata.")
                    print("=" * 65 + "\n")
            else:
                print(" [INFO] Reboot disattivato da configurazione (reboot_after_run=False).")

    except SystemExit:
        raise
    except BaseException as fatal_exc:
        cleanup_orphan_stressors()
        log_fallback_error("main_top_level", fatal_exc)
        print(f"\n[ERRORE FATALE] {fatal_exc}. Processo arrestato in sicurezza. Log in: {FALLBACK_ERROR_LOG}")
        sys.exit(1)


if __name__ == "__main__":
    main()
