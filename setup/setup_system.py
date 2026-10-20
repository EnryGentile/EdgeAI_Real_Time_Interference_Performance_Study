import os
import sys
import subprocess
import json
from pathlib import Path

# ==========================================
# 0. CHECK E INSTALLAZIONE LIBRERIE NECESSARIE
# ==========================================
def install_required_packages():
    required_packages = ["huggingface_hub", "datasets"]
    for package in required_packages:
        try:
            __import__(package)
            print(f"[OK] Libreria '{package}' già installata.")
        except ImportError:
            print(f"[INSTALL] Libreria '{package}' mancante. Installazione in corso...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", package])

    print("=== Verifica delle dipendenze Python ===")


    # Import delle librerie
    from huggingface_hub import hf_hub_download
    from datasets import load_dataset

# ==========================================
# CONFIGURAZIONE PERCORSI E PARAMETRI
# ==========================================
MODELS_DIR = Path("./models")
DATASETS_DIR = Path("./datasets")
LLAMA_CPP_DIR = Path("./llama.cpp")

# Modelli Meta Llama 3.x (Stessa famiglia/generazione tecnologica)
MODELS_CONFIG = [
    {
        "role": "SLM (Small Language Model - 1B)",
        "repo": "bartowski/Llama-3.2-1B-Instruct-GGUF",
        "file": "Llama-3.2-1B-Instruct-Q4_K_M.gguf"
    },
    {
        "role": "LLM (Large Language Model - 8B)",
        "repo": "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
        "file": "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
    }
]

def setup_directories():
    """Crea le directory di lavoro se non esistono."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)

def check_and_download_models():
    """Verifica e scarica i file GGUF dei modelli (SLM 1B e LLM 8B) da Hugging Face."""
    for model_info in MODELS_CONFIG:
        role = model_info["role"]
        model_file = model_info["file"]
        model_repo = model_info["repo"]
        model_path = MODELS_DIR / model_file

        if model_path.exists():
            print(f"[OK] Modello {role} già presente: {model_path}")
        else:
            print(f"[DOWNLOAD] Scaricando {role} ({model_file}) da Hugging Face...")
            try:
                hf_hub_download(
                    repo_id=model_repo,
                    filename=model_file,
                    local_dir=MODELS_DIR,
                    local_dir_use_symlinks=False
                )
                print(f"[OK] Modello {role} scaricato correttamente in: {model_path}")
            except Exception as e:
                print(f"[ERRORE] Impossibile scaricare il modello {role}: {e}")
                print("Nota: Assicurati di aver eseguito 'huggingface-cli login' e di aver accettato i termini d'uso dei modelli Meta.")

def check_and_download_datasets():
    """Verifica e crea i subset locali di MMLU e Spider (prime 40 richieste)."""
    mmlu_path = DATASETS_DIR / "mmlu_first_40.json"
    spider_path = DATASETS_DIR / "spider_first_40.json"

    # 1. Dataset MMLU (Prefill-heavy / Compute-bound)
    if mmlu_path.exists():
        print(f"[OK] Dataset MMLU (40 campioni) già presente: {mmlu_path}")
    else:
        print("[DOWNLOAD] Generazione subset MMLU (prime 40 richieste)...")
        try:
            mmlu_dataset = load_dataset("cais/mmlu", "all", split="test", streaming=True)
            mmlu_subset = list(mmlu_dataset.take(40))
            with open(mmlu_path, "w", encoding="utf-8") as f:
                json.dump(mmlu_subset, f, indent=2, ensure_ascii=False)
            print(f"[OK] Dataset MMLU salvato in: {mmlu_path}")
        except Exception as e:
            print(f"[ERRORE] Impossibile scaricare il dataset MMLU: {e}")

    # 2. Dataset Spider (Decode-heavy / Memory-bound)
    if spider_path.exists():
        print(f"[OK] Dataset Spider (40 campioni) già presente: {spider_path}")
    else:
        print("[DOWNLOAD] Generazione subset Spider (prime 40 richieste)...")
        try:
            spider_dataset = load_dataset("xlangai/spider", split="validation", streaming=True)
            spider_subset = list(spider_dataset.take(40))
            with open(spider_path, "w", encoding="utf-8") as f:
                json.dump(spider_subset, f, indent=2, ensure_ascii=False)
            print(f"[OK] Dataset Spider salvato in: {spider_path}")
        except Exception as e:
            print(f"[ERRORE] Impossibile scaricare il dataset Spider: {e}")

def check_and_setup_llama_cpp():
    """Verifica la presenza dei binari di llama.cpp o procede alla clonazione e build con CUDA."""
    bin_path = LLAMA_CPP_DIR / "build" / "bin" / "llama-cli"
    
    if bin_path.exists():
        print(f"[OK] Runtime llama.cpp già compilato trovato in: {bin_path}")
    else:
        print("[SETUP] llama.cpp non trovato o non compilato. Configurazione in corso...")
        
        if not LLAMA_CPP_DIR.exists():
            print("Clonazione del repository ufficiale di llama.cpp...")
            subprocess.run(["git", "clone", "https://github.com/ggerganov/llama.cpp.git", str(LLAMA_CPP_DIR)], check=True)
        
        print("Configurazione CMake e compilazione con supporto CUDA per Jetson AGX Orin...")
        build_dir = LLAMA_CPP_DIR / "build"
        build_dir.mkdir(exist_ok=True)
        
        # Configurazione con supporto CUDA attivo per l'architettura Jetson
        subprocess.run(["cmake", "-B", str(build_dir), "-S", str(LLAMA_CPP_DIR), "-DGGML_CUDA=ON"], check=True)
        
        # Compilazione parallela sfruttando tutti i core della CPU della board
        subprocess.run(["cmake", "--build", str(build_dir), "--config", "Release", "-j$(nproc)"], shell=True, check=True)
        print(f"[OK] Compilazione di llama.cpp completata con successo!")

if __name__ == "__main__":
    print("=== Avvio Setup Ambiente di Benchmark (Jetson AGX Orin) ===")
    install_required_packages()
    setup_directories()
    check_and_download_models()
    check_and_download_datasets()
    check_and_setup_llama_cpp()
    print("\n=== Setup Completato con Successo! SLM (1B), LLM (8B), Dataset e Binari pronti. ===")