# EdgeAI Real-Time Interference Performance Study (Traccia A1)

Studio sperimentale e statistico sull'interferenza da contesa hardware in tempo reale durante l'inferenza di Small/Large Language Model (**SLM/LLM**) su piattaforme embedded edge (**NVIDIA Jetson AGX Orin**).

L'obiettivo è quantificare e modellare matematicamente come la saturazione di specifiche risorse hardware (mediante `stress-ng`) degrada la reattività temporale (Time To First Token), la latenza inter-token (TPOT), il throughput, il jitter e l'accuratezza semantica, correlandoli con i contatori hardware del kernel Linux (`vmstat`), i sensori termici di bordo e i registri microarchitetturali.

---

## 1. Piano Fattoriale degli Esperimenti (DoE)

In conformità ai requisiti sperimentali, i fattori del DoE sono rigorosamente definiti:

1. **Numero di Worker ($N$)**:
   - 4 livelli: `25% N_tot`, `50% N_tot`, `75% N_tot`, `N_tot` (100% dei core fisici).
2. **Working-Set Size ($S$)**:
   - 3 livelli: `25% S_tot`, `50% S_tot`, `75% S_tot` (percentuale della RAM o dello spazio I/O).
   - **Presente ESCLUSIVAMENTE** per gli stressor 3D (`mem`, `io`). Per gli stressor 2D (`cpu`, `cache`, `sched`, `interrupt`), la size è assente (`N/A`).
3. **Durata Temporale ($D$)**:
   - **NON è un fattore variabile**: è fissa e costante al **tempo totale ($t_{\text{tot}}$)** della richiesta di inferenza, coprendo sia la fase di *prefill* (elaborazione del prompt) che la fase di *decode* (generazione autoregressiva dei token) in modo continuativo.

### Dimensionamento della Matrice Sperimentale
- **Stressor 2D** (`cpu`, `cache`, `sched`, `interrupt`): $4 \text{ livelli } (N) \times 1 \text{ durata } (t_{\text{tot}}) \times 3 \text{ ripetizioni} = \mathbf{12 \text{ esecuzioni}}$.
- **Stressor 3D** (`mem`, `io`): $4 \text{ livelli } (N) \times 3 \text{ livelli } (S) \times 1 \text{ durata } (t_{\text{tot}}) \times 3 \text{ ripetizioni} = \mathbf{36 \text{ esecuzioni}}$.
- **Baseline** (senza stressor): 1 configurazione $\times 3 \text{ ripetizioni}$ su $n_0 = 40$ task per la stima della varianza iniziale e il dimensionamento CLT.

---

## 2. Caratteristiche di Robustezza del Framework

- **Watchdog di Protezione (Timeout 300s)**: ogni singolo task su `llama-cli` è protetto da un timeout di 5 minuti. Se l'interferenza genera un hang o un blocco, il processo viene terminato in sicurezza e il framework passa al task successivo, evitando l'interruzione della campagna.
- **Cattura e Salvataggio delle Risposte del Modello**: oltre alle metriche temporali, il testo generato dal modello viene salvato in due formati:
  1. *File JSON Strutturato Dedicato* (`results/responses_<exp_id>.json`): registra prompt, output completo, ground truth di benchmark (lettera esatta MMLU o query SQL Spider) e timing per valutare accuratezza e allucinazioni.
  2. *CSV per-task* (`results/tasks_<exp_id>.csv`): colonne `model_response` e `ground_truth` affiancate alle metriche per-token.
- **Monitoraggio Termico Hardware a 1 Hz**: lettura continua dei sensori hardware Tegra (`/sys/class/thermal/thermal_zone*`: `CPU-therm`, `GPU-therm`, `SOC*-therm`, `Tdiode`) e rilevamento del thermal throttling via `cpufreq`. Fallback automatico su `hwmon` e `nvidia-smi` per ambienti desktop o Colab.
- **Reverse SSH Tunneling Automatico**: invio istantaneo e sicuro via SCP di tutti i file generati direttamente sul PC dell'utente, funzionante anche attraverso reti universitarie e NAT complessi (DESSERT Remote Access / Cloudflare Tunnel).
- **Gestione Atomica dello Stato e Reboot**: checkpoint persistente (`setup/experiment_state.json`) che tiene traccia dei run completati. Al termine di ogni run la board si riavvia, attende 90 secondi di stabilizzazione termica ed esegue il test successivo.

---

## 3. Architettura della Repository

```
├── colab_test_workflow.ipynb       # Notebook per test e debug end-to-end (GPU Tesla T4 / Pinggy SSH)
├── setup/
│   ├── setup_system.py             # Setup librerie, download GGUF (1B/8B), dataset (MMLU/Spider), build CUDA llama.cpp
│   ├── setup_run.py                # Configurazione parametri run e validazione rigida prerequisiti GPU/CPU
│   ├── compute_sample_size.py      # Calcolo dimensione campionaria n via CLT (IC 95%, errore 5%)
│   ├── run_experiments.py          # Orchestratore: DoE, stress-ng, vmstat 1Hz, watchdog, SSH sync, reboot
│   ├── experiment_config.json      # Configurazione serializzata dell'esperimento attivo
│   └── experiment_state.json       # Checkpoint atomico con avanzamento della coda esperimenti
├── statistical_analysis/
│   ├── plot_llm_metrics.py         # Grafici di degradazione metriche LLM con riferimento Baseline (300 DPI)
│   ├── plot_hardware_bottlenecks.py# Grafici colli di bottiglia hardware (CPU, Mem, I/O, Sys, Thermal)
│   ├── importance_significance.jsl # Script JMP: ANOVA (SS/SST), Shapiro-Wilk residui, Kruskal-Wallis
└── master_experiment_summary.csv   # Riepilogo master di tutte le sessioni sperimentali
```

---

## 4. Guida Operativa Passo-Passo

Segui questi passaggi per configurare ed eseguire l'intera campagna sperimentale sulla scheda **NVIDIA Jetson AGX Orin**.

---

### FASE 0: Connessione e Permessi con Tailscale (PC $\leftrightarrow$ Jetson)

Utilizzando **Tailscale**, la Jetson Orin e il tuo PC Windows risiedono sulla stessa rete privata virtuale mesh:
- La connessione è **diretta e bidirezionale**: non servono tunnel inversi (`-R`), porte forwarding sul router né servizi a scadenza come Pinggy.
- Porta SSH standard: **`22`**.
- L'IP Tailscale del tuo PC Windows è: **`100.92.177.23`** (verificabile con `ipconfig`).

#### 1. Connettiti alla Jetson tramite Tailscale
Sul tuo PC Windows, collegati direttamente all'IP Tailscale della Jetson:
```bash
ssh orinagx@<JETSON_TAILSCALE_IP>
```

#### 2. Configura l'autenticazione SSH senza password per l'invio automatico dei dati
Sulla **Jetson**, genera la coppia di chiavi SSH (se non già presente):
```bash
[ -f ~/.ssh/id_ed25519.pub ] || ssh-keygen -t ed25519 -N "" -f ~/.ssh/id_ed25519
cat ~/.ssh/id_ed25519.pub
```
Copia la chiave pubblica visualizzata (`ssh-ed25519 AAAAC3...`) e incollala nel file `C:\Users\gabri\.ssh\authorized_keys` sul tuo PC Windows. In questo modo la Jetson potrà inviarti i CSV e i JSON in background senza richiedere password.

#### 3. Abilita il Reboot Automatico senza password per l'utente `orinagx`
Sulla shell della **Jetson**, esegui una sola volta:
```bash
echo "$USER ALL=(ALL) NOPASSWD: /sbin/reboot, /bin/systemctl reboot" | sudo tee /etc/sudoers.d/orinagx-reboot
```

---

### FASE 1: Preparazione e Setup del Sistema (Sulla Jetson)

#### 1. Installa i pacchetti di sistema essenziali
```bash
sudo apt update && sudo apt install -y build-essential cmake git stress-ng sysstat python3-pip python3-dev curl
```

#### 2. Verifica la presenza del compilatore CUDA (`nvcc`)
```bash
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
nvcc --version
```

#### 3. Clona la repository ed esegui il setup automatico
```bash
cd ~
# Clona la repository (se non già fatto)
git clone https://github.com/TUO_REPO/EdgeAI_Real_Time_Interference_Performance_Study.git
cd EdgeAI_Real_Time_Interference_Performance_Study

# Avvia il setup di sistema
python3 setup/setup_system.py
```
*Lo script scaricherà i modelli GGUF (SLM-1B e LLM-8B), i dataset MMLU e Spider, e compilerà `llama-cli` con accelerazione CUDA nativa (`-DGGML_CUDA=ON`).*

---

### FASE 2: Validazione della Configurazione (`setup_run.py`)

Prima di avviare le misurazioni, verifica che runtime, modelli, benchmark e allocazione hardware GPU siano conformi:

```bash
python3 setup/setup_run.py \
  --model slm \
  --device gpu \
  --benchmark mmlu \
  --no-stressor \
  --samples 40 \
  --repetitions 1 \
  --transient-wait 90 \
  --reboot \
  --ssh-target "gabri@100.92.177.23" \
  --ssh-port 22 \
  --ssh-dest "C:/Users/gabri/OneDrive/Desktop/Universita/Magistrale/Real-Time Systems and Industrial Applications/EdgeAI_Real_Time_Interference_Performance_Study"
```
Attendi l'esito: `[SUCCESSO] Configurazione istanziata con successo`.

---

### FASE 3: Esecuzione della Baseline ($n_0 = 40$ campioni)

La baseline misura le prestazioni del modello a riposo (senza carichi di interferenza attivi) per stabilire il riferimento e calcolare la variabilità naturale:

```bash
python3 setup/run_experiments.py \
  --model slm \
  --device gpu \
  --benchmark mmlu \
  --no-stressor \
  --samples 40 \
  --repetitions 1 \
  --transient-wait 90 \
  --no-reboot \
  --ssh-target "gabri@100.92.177.23" \
  --ssh-port 22 \
  --ssh-dest "C:/Users/gabri/OneDrive/Desktop/Universita/Magistrale/Real-Time Systems and Industrial Applications/EdgeAI_Real_Time_Interference_Performance_Study"
```
*I file `baseline_slm-1b_gpu_mmlu_tasks.csv`, `doe_slm-1b_gpu_mmlu_none.csv`, `responses_*.json` e le time-series verranno trasferiti automaticamente sul tuo PC via Tailscale.*

---

### FASE 4: Calcolo della Dimensione Campionaria $n$ via CLT

Esegui lo script per calcolare analiticamente il numero minimo di campioni necessari ($n$) per garantire un **intervallo di confidenza al 95%** ($z = 1.96$) con un **errore massimo del 5%** ($r = 0.05$):

```bash
python3 setup/compute_sample_size.py \
  --model slm \
  --device gpu \
  --benchmark mmlu \
  --confidence 0.95 \
  --error 0.05 \
  --update-config
```
*Il flag `--update-config` aggiorna automaticamente `sample_count` nel file `setup/run_config.json` con il valore di $n$ calcolato (es. $n = 15$).*

---

### FASE 5: Automazione Headless e Avvio Campagna DoE con Stressor

Per consentire alla Jetson di eseguire tutti i run (12 per stressor 2D, 36 per stressor 3D) con reboot automatici senza dover intervenire manualmente dopo ogni riavvio:

#### 1. Abilita l'avvio automatico al boot con systemd (una sola volta)
Sulla Jetson, esegui il nostro script di supporto:
```bash
bash setup/setup_autostart.sh enable
```
*Questo comando crea e attiva un servizio di sistema (`edgeai-doe.service`) configurato per attendere che la rete e `tailscaled` siano online, eseguire l'esperimento successivo, inviare i dati sul PC e riavviare la board.*

#### 2. Inizializza e lancia la campagna sperimentale
Configura e avvia la campagna con lo stressor scelto (es. `cpu`):
```bash
python3 setup/run_experiments.py \
  --model slm \
  --device gpu \
  --benchmark mmlu \
  --use-stressor \
  --stressor cpu \
  --repetitions 3 \
  --transient-wait 90 \
  --reboot \
  --ssh-target "gabri@100.92.177.23" \
  --ssh-port 22 \
  --ssh-dest "C:/Users/gabri/OneDrive/Desktop/Universita/Magistrale/Real-Time Systems and Industrial Applications/EdgeAI_Real_Time_Interference_Performance_Study"
```
*Dopo la prima run, la Jetson si riavvierà e proseguirà in **completa autonomia** fino a completare tutti i run.*

#### 3. Monitoraggio Headless
In qualsiasi momento puoi collegarti via Tailscale e controllare l'avanzamento senza interrompere il processo:
- **Seguire il log in tempo reale**:
  ```bash
  tail -f headless_doe.log
  ```
- **Verificare lo stato della coda**:
  ```bash
  python3 setup/run_experiments.py --status
  ```
- **Disabilitare l'avvio automatico a fine campagna**:
  ```bash
  bash setup/setup_autostart.sh disable
  ```

---

### FASE 6: Analisi Statistica e Grafici (Sul tuo PC Host Windows)

Sul tuo computer, apri una finestra PowerShell o terminale nella cartella del progetto:

#### 1. Generazione Grafici di Degradazione Metriche LLM
```powershell
python statistical_analysis/plot_llm_metrics.py
```
*Genera i grafici per TTFT, latenza per-token, throughput e jitter in `figures/llm_metrics/`, confrontando le curve di carico $N$ e working-set $S$ con la linea verde tratteggiata della Baseline senza stress.*

#### 2. Generazione Grafici dei Colli di Bottiglia Hardware
```powershell
python statistical_analysis/plot_hardware_bottlenecks.py
```
*Genera i grafici a 1 riga in `figures/hardware_bottlenecks/` per le 6 famiglie `vmstat`: saturazione CPU (%us, %sy, %id, %wa), utilizzo memoria/swap, blocchi I/O, processi ready/blocked, context switch/interrupt e temperature.*

#### 3. Analisi Statistica Avanzata in JMP
- Apri **JMP** sul tuo PC.
- Apri ed esegui il file JSL:
  [`statistical_analysis/importance_significance.jsl`](statistical_analysis/importance_significance.jsl)
  - Calcola la quota di varianza spiegata ($SS_{\text{Effect}} / SS_{\text{Total}}$).
  - Esegue il test di Shapiro-Wilk sui residui del modello.
  - Se $p \ge 0.05$ (normalità rispettata), valuta il valore $p$ dell'ANOVA ($F$-test).
  - Se $p < 0.05$ (non-normalità), esegue automaticamente il test non parametrico di **Kruskal-Wallis** ($\chi^2$).
- Apri ed esegui [`statistical_analysis/distributions.jsl`](statistical_analysis/distributions.jsl) per ottenere istogrammi e QQ-plot univariati di tutte le metriche hardware.
