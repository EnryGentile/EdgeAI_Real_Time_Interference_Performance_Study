# EdgeAI Real-Time Interference Performance Study (Traccia A1)

Studio sperimentale e statistico sull'interferenza da contesa hardware in tempo reale durante l'inferenza di Small/Large Language Model (**SLM/LLM**) su piattaforme embedded edge (**NVIDIA Jetson Orin Nano / AGX Orin**).

L'obiettivo è quantificare e modellare matematicamente come la saturazione di specifiche risorse hardware (mediante `stress-ng`) degrada la reattività temporale (Time To First Token - TTFT), la latenza inter-token (TPOT), il throughput (token/s), il jitter e l'accuratezza semantica, correlandoli con i contatori hardware del kernel Linux (`vmstat`), i sensori termici di bordo e i registri microarchitetturali.

---

## 1. Piano Fattoriale degli Esperimenti (DoE)

In conformità ai requisiti sperimentali, i fattori del DoE sono rigorosamente definiti:

1. **Numero di Worker ($N$)**:
   - 4 livelli: `25% N_tot`, `50% N_tot`, `75% N_tot`, `N_tot` (100% dei core fisici).
2. **Working-Set Size ($S$)**:
   - 3 livelli: `25% S_tot`, `50% S_tot`, `75% S_tot` (percentuale della dimensione massima: RAM per `mem`, dimensione massima file per `io`).
   - **Presente ESCLUSIVAMENTE per gli stressor 3D** (`mem`, `io`). Per gli stressor 2D (`cpu`, `cache`, `os`, `interrupt`), il fattore size è assente (`N/A`).
3. **Durata Temporale ($D$)**:
   - **NON è un fattore variabile**: è fissa e costante al **tempo totale ($t_{\text{tot}}$)** della richiesta di inferenza, coprendo sia la fase di *prefill* (elaborazione del prompt) che la fase di *decode* (generazione autoregressiva dei token) in modo continuativo.

### Dimensionamento della Matrice Sperimentale
Ogni volta che si seleziona una famiglia di stressor, il framework avvia **tutti i metodi appartenenti alla famiglia simultaneamente** in un unico comando `stress-ng`, applicando gli **stessi identici fattori** ($N$ worker e, ove previsto, $S$ size) a ciascun metodo:
- **Famiglia `cpu`** (`cpu`, `matrix`, `fft` simultanei): 4 livelli ($N$) $\times$ 1 durata ($t_{\text{tot}}$) $\times$ 3 ripetizioni = $\mathbf{12 \text{ esecuzioni}}$.
- **Famiglia `mem`** (`vm`, `stream`, `mem_rate` simultanei): 4 livelli ($N$) $\times$ 3 size ($S$) $\times$ 1 durata ($t_{\text{tot}}$) $\times$ 3 ripetizioni = $\mathbf{36 \text{ esecuzioni}}$.
- **Famiglia `cache`** (`cache`, `l1-cache`, `llc-affinity` simultanei): 4 livelli ($N$) $\times$ 1 durata ($t_{\text{tot}}$) $\times$ 3 ripetizioni = $\mathbf{12 \text{ esecuzioni}}$.
- **Famiglia `io`** (`iomix`, `hdd` simultanei): 4 livelli ($N$) $\times$ 3 size ($S$) $\times$ 1 durata ($t_{\text{tot}}$) $\times$ 3 ripetizioni = $\mathbf{36 \text{ esecuzioni}}$.
- **Famiglia `os`** (`schedpolicy`, `fork`, `switch` simultanei): 4 livelli ($N$) $\times$ 1 durata ($t_{\text{tot}}$) $\times$ 3 ripetizioni = $\mathbf{12 \text{ esecuzioni}}$.
- **Famiglia `interrupt`** (`timer`, `sigq` simultanei): 4 livelli ($N$) $\times$ 1 durata ($t_{\text{tot}}$) $\times$ 3 ripetizioni = $\mathbf{12 \text{ esecuzioni}}$.
- **Modalità `all-stressor`**: Esegue tutte le 6 famiglie di stressor in sequenza ordinata (`cpu` $\to$ `mem` $\to$ `cache` $\to$ `io` $\to$ `os` $\to$ `interrupt`) per un totale di $\mathbf{120 \text{ esecuzioni}}$ ($12 + 36 + 12 + 36 + 12 + 12$) con checkpointing e reboot automatico.
- **Baseline** (senza stressor): 1 configurazione $\times 3 \text{ ripetizioni}$ su $n_0 = 40$ task per la stima della varianza iniziale e il dimensionamento CLT.

---

## 2. Architettura del Benchmark e Ottimizzazioni Real-Time

### A. Modelli e Benchmark Zero-Shot
La valutazione è condotta in modalità **Zero-Shot pura**, senza esempi di contesto che appesantiscano la memoria o distorcano le latenze:
* **Modello LLM (8B):** `Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf` (~4.58 GB).
* **Modello SLM (1B):** `Llama-3.2-1B-Instruct-Q4_K_M.gguf` (~800 MB).
* **Benchmark MMLU (Prefill-heavy / Compute-bound):** Quesiti a scelta multipla tratti da `datasets/mmlu_first_40.json` con prompt standard in lingua inglese:
  ```text
  Question: <question>
  Options:
  A. <option A>
  B. <option B>
  C. <option C>
  D. <option D>
  Answer with only the letter corresponding to the correct answer:
  ```
* **Benchmark Spider (Decode-heavy / Memory-bound):** Traduzione Text-to-SQL da `datasets/spider_first_40.json`:
  ```text
  Convert the following natural language request into a SQL query for the database '<db_id>'.
  Request: <request>
  SQL:
  ```
Tutti i prompt e i relativi fallback di emergenza sono formulati in lingua inglese per massimizzare la precisione di instruction-following dei modelli Llama-3.

### B. Gestione della Memoria Unificata (UMA) e KV Cache a 128 MB
Sulle schede NVIDIA Jetson con 8 GB di memoria unificata (condivisa tra CPU e GPU), l'esecuzione di LLaMA-8B richiede un bilanciamento rigoroso:
* **Sessione Persistente (`LlamaPersistentSession`):** Il modello GGUF viene caricato in memoria **una sola volta** all'inizio dell'esperimento. Tra un task e il successivo viene inviato il comando interattivo `/clear` su stdin, che resetta la KV cache e azzera la cronologia senza deallocare né ricaricare i 4.58 GB di pesi da disco.
* **KV Cache Fissata a 128 MB (`-c 1024`):**
  $$\text{KV Cache Size} = 32 \text{ layer} \times 8 \text{ heads} \times 128 \times 4 \text{ byte/token} \times 1024 \text{ token} = 128.00\text{ MiB}$$
* **Allocazione 1-Shot Istantanea (`-b 128 -ub 64`):** Riducendo il batch di calcolo da 256 a 128 e il micro-batch a 64, i buffer computazionali di `llama.cpp` scendono da ~250 MB a soli ~60 MB. Questo libera circa **190 MB di memoria contigua IOMMU/DMA**, permettendo a `cudaMalloc(128 MB)` di allocare la KV cache al primo tentativo senza fallimenti `nvmap error 12`.
* **Pulizia Attiva Page Cache (`posix_fadvise DONTNEED`):** Non appena i pesi sono caricati in GPU, la cache di lettura Linux del file GGUF viene rilasciata, evitando di occupare due volte la RAM fisica.

### C. Calcolo Preciso del Vero TTFT (Filtro Anti-Echo)
Il terminale interattivo di `llama-cli` stampa in echo il prompt prima di emettere il primo token. Il framework implementa un filtro rigoroso basato sulla coda del prompt (`rfind(_prompt_tail)`):
* Lo scanner ignora tutti i caratteri appartenenti all'echo del prompt.
* Il primo timestamp registrato è associato al **primo vero token emesso dal modello**, garantendo un Time To First Token matematicamente esatto ($T_{\text{prefill}} = t_{\text{first\_token}} - t_{\text{prompt\_sent}}$).

### D. Massime Prestazioni e Azzeramento del Jitter (`nvpmodel` e `jetson_clocks`)
Di fabbrica, Linux Tegra applica governor dinamici a risparmio energetico (`schedutil` per la CPU, `nvhost_podgov` per la GPU e dynamic scaling per il bus di memoria EMC a 204 MHz). Questo introduce jitter artificiale durante le misurazioni.
* **Modalità MAXN (15W):** `sudo nvpmodel -m 0` sblocca tutti i core CPU e il massimo budget termico/energetico.
* **Lock dei Clock (`jetson_clocks`):**
  * **CPU:** 6 core bloccati a **1.728 GHz** (Orin Super) / **1.5 GHz**.
  * **GPU:** bloccata a **1.02 GHz** (Orin Super) / **625 MHz**.
  * **Bus Memoria LPDDR5 (EMC):** bloccato a **3.199 GHz** (102 GB/s di banda) con `FreqOverride=1`.
  * **Ventola:** forzata al massimo per garantire dissipazione continua.
* **Benefici:** +35%–50% di throughput su CPU, +25% di throughput su GPU e **azzeramento completo del jitter da scaling di frequenza** (ogni variazione di latenza è imputabile al 100% agli stressor).
* **Persistenza post-reboot:** Abilitando il servizio nativo `sudo systemctl enable jetson_clocks`, i clock al massimo vengono ripristinati automaticamente ad ogni riavvio.

### E. Transitorio Post-Boot Ottimizzato (60 Secondi)
La telemetria hardware registrata dimostra che la temperatura di CPU e GPU raggiunge lo steady-state a riposo (~39.7°C–40.1°C) in meno di 30 secondi dal boot, e i demoni di sistema esauriscono il carico entro 25 secondi.
* Il transitorio predefinito è impostato a **60 secondi** (`--transient-wait 60` o `-w 60`), risparmiando oltre **1 ora di attesa** su una campagna completa da 120 run rispetto ai precedenti 90s.
* In fase di test / debug è possibile usare `--wait 10` e `--no-reboot` per iterazioni da meno di 2 minuti.

---

## 3. Storage Hardware: MicroSD

Il caricamento del modello GGUF da 4.58 GB è strettamente legato al supporto di memoria di massa:
* **Su scheda MicroSD (UHS-I SDR104):** La velocità massima di lettura fisica è di circa **87.3 MB/s**. Di conseguenza, il caricamento iniziale dei 4.58 GB richiede fisicamente **~53–58 secondi**. Grazie alla sessione persistente, questo overhead avviene **1 sola volta per tutti i 40 task** (~1.3s a task ammortizzati).
* **Ottimizzazione Readahead Linux:** Per massimizzare la lettura sequenziale anche su MicroSD, il readahead del blocco disco può essere impostato a 8 MB:
  ```bash
  sudo blockdev --setra 16384 $(df models/ | tail -1 | awk '{print $1}')
  ```

---

## 4. Architettura della Repository

```
├── colab_test_workflow.ipynb       # Notebook per test e debug end-to-end (GPU Tesla T4 / Pinggy SSH)
├── setup/
│   ├── setup_system.py             # Setup librerie, download GGUF (1B/8B), dataset (MMLU/Spider), build CUDA llama.cpp
│   ├── setup_swap.sh               # Configurazione swapfile da 16GB per prevenire OOM su UMA 8GB
│   ├── setup_autostart.sh          # Configurazione servizio systemd headless post-boot con jetson_clocks
│   ├── setup_run.py                # Configurazione parametri run, template prompt EN, validazione prerequisiti
│   ├── compute_sample_size.py      # Calcolo dimensione campionaria n via CLT (IC 95%, errore 5%)
│   ├── run_experiments.py          # Orchestratore DoE: sessione persistente, stress-ng, vmstat 1Hz, watchdog, SSH
│   ├── run_config.json             # Configurazione serializzata dell'esperimento attivo
│   └── experiment_state.json       # Checkpoint atomico con avanzamento della coda esperimenti
├── statistical_analysis/
│   ├── plot_llm_metrics.py         # Grafici di degradazione metriche LLM con riferimento Baseline (300 DPI)
│   ├── plot_hardware_bottlenecks.py# Grafici colli di bottiglia hardware (CPU, Mem, I/O, Sys, Thermal)
│   ├── importance_significance.jsl # Script JMP: ANOVA (SS/SST), Shapiro-Wilk residui, Kruskal-Wallis
│   └── distributions.jsl           # Script JMP: Distribuzioni e QQ-plot univariati delle metriche
├── datasets/                       # Dataset MMLU (40 campioni) e Spider (40 campioni)
├── models/                         # Modelli GGUF Q4_K_M (Llama-3.2-1B e Llama-3.1-8B)
├── results/                        # Risultati per-task (CSV), risposte del modello (JSON) e DoE summary
└── timeseries/                     # Tracce telemetriche complete vmstat e termiche a 1 Hz
```

---

## 5. Guida Operativa Passo-Passo

Segui questi passaggi per configurare ed eseguire l'intera campagna sperimentale sulla scheda **NVIDIA Jetson Orin**.

---

### FASE 0: Connessione e Permessi con Tailscale (PC $\leftrightarrow$ Jetson)

Utilizzando **Tailscale**, la Jetson Orin e il tuo PC host risiedono sulla stessa rete privata virtuale mesh:
- Connessione diretta e bidirezionale su porta standard **`22`**.
- L'IP Tailscale dell'host ricevente è es. `100.92.177.23`.

#### 1. Connettiti alla Jetson tramite Tailscale
```bash
ssh king@<JETSON_TAILSCALE_IP>
```

#### 2. Configura l'autenticazione SSH senza password per l'invio dei dati
Sulla **Jetson**, genera la coppia di chiavi SSH (se non già presente):
```bash
[ -f ~/.ssh/id_ed25519.pub ] || ssh-keygen -t ed25519 -N "" -f ~/.ssh/id_ed25519
cat ~/.ssh/id_ed25519.pub
```
Copia la chiave pubblica visualizzata e incollala in `~/.ssh/authorized_keys` sul tuo PC host. In questo modo la Jetson potrà inviarti i file CSV e JSON in background senza richiedere password.

#### 3. Abilita i permessi sudoers NOPASSWD per Reboot e Gestione Clock
Sulla shell della **Jetson**, esegui una sola volta:
```bash
sudo bash -c 'echo "$USER ALL=(ALL) NOPASSWD: /sbin/reboot, /bin/systemctl reboot, /usr/sbin/nvpmodel, /usr/bin/jetson_clocks" > /etc/sudoers.d/$USER-reboot && chmod 0440 /etc/sudoers.d/$USER-reboot'
```

#### 4. Forza i Clock Hardware al massimo e abilita la persistenza al boot
```bash
# Attiva subito la modalità 15W MAXN e blocca i clock al massimo
sudo nvpmodel -m 0
sudo jetson_clocks

# Per garantire che i clock rimangano bloccati al massimo ad OGNI reboot (anche senza autostart):
sudo bash setup/setup_clocks_service.sh
```
Verifica lo stato con `sudo jetson_clocks --show`.

---

### FASE 1: Preparazione e Setup del Sistema (Sulla Jetson)

#### 1. Installa i pacchetti essenziali
```bash
sudo apt update && sudo apt install -y build-essential cmake git stress-ng sysstat python3-pip python3-dev curl
```

#### 2. Configura il Swapfile da 16 GB e l'ambiente CUDA
```bash
cd ~/EdgeAI_Real_Time_Interference_Performance_Study

# Configura 16GB di swap su disco per garantire stabilità contro OOM
sudo bash setup/setup_swap.sh

# Verifica CUDA nvcc
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
nvcc --version
```

#### 3. Avvia il setup automatico di asset e compilazione
```bash
python3 setup/setup_system.py
```
*Lo script verifica o scarica i modelli GGUF (SLM-1B e LLM-8B), i dataset MMLU e Spider, e compila `llama-cli` con accelerazione CUDA nativa (`-DGGML_CUDA=ON`).*

---

### FASE 2: Validazione della Configurazione (`setup_run.py`)

Prima di avviare le misurazioni, valida i prerequisiti di sistema e istanzia la configurazione desiderata:

```bash
# Esempio: Modello LLM-8B su GPU, Benchmark MMLU, Baseline (senza stressor), 40 campioni, 3 ripetizioni
python3 setup/setup_run.py \
  --model llm \
  --device gpu \
  --benchmark mmlu \
  --no-stressor \
  --samples 40 \
  --repetitions 3 \
  --wait 60 \
  --reboot \
  --ssh-target "gabri@100.92.177.23" \
  --ssh-port 22 \
  --ssh-dest "C:/Users/gabri/OneDrive/Desktop/Universita/Magistrale/Real-Time Systems and Industrial Applications/EdgeAI_Real_Time_Interference_Performance_Study"
```
*I flag `-w` / `--wait` impostano la durata del transitorio post-boot (default: 60s). Per test rapidi si possono usare `--wait 10` e `--no-reboot`.*

---

### FASE 3: Esecuzione della Baseline ($n_0 = 40$ campioni)

La baseline misura le prestazioni del modello a riposo (senza interferenze) per stabilire il riferimento e calcolare la variabilità naturale:

```bash
python3 setup/run_experiments.py --reset
```
*Al termine della sessione, i file `baseline_*_tasks.csv`, `doe_*_none.csv`, `responses_*.json` e le time-series a 1 Hz verranno sincronizzati automaticamente sul tuo PC via SSH.*

---

### FASE 4: Calcolo della Dimensione Campionaria $n$ via CLT

Esegui lo script per calcolare analiticamente il numero minimo di campioni necessari ($n$) per garantire un **intervallo di confidenza al 95%** ($z = 1.96$) con un **errore massimo del 5%** ($r = 0.05$):

* Lo script estrae la **mediana** per prompt dalle 3 ripetizioni della baseline per rimuovere gli outlier e calcola $n$ sulle 3 metriche chiave del modello: **TTFT**, **TPOT** e **Throughput**.

```bash
python3 setup/compute_sample_size.py \
  --model llm \
  --device gpu \
  --benchmark mmlu \
  --confidence 0.95 \
  --error 0.05 \
  --update-config
```
*Il flag `--update-config` aggiorna automaticamente `sample_count` nel file `setup/run_config.json` con il valore di $n$ calcolato.*

---

### FASE 5: Automazione Headless e Avvio Campagna DoE con Stressor

Per consentire alla Jetson di eseguire tutti i run (12 per stressor 2D, 36 per stressor 3D, oppure 120 per `--all-stressor`) con reboot automatici in totale autonomia:

#### 1. Abilita l'avvio automatico al boot con systemd
```bash
bash setup/setup_autostart.sh enable
```
*Il servizio di sistema `edgeai-doe.service` ripristina automaticamente i clock al massimo (`ExecStartPre=+/usr/bin/jetson_clocks`), attende la rete e riprende l'esperimento successivo in coda ad ogni riavvio.*

#### 2. Inizializza la campagna desiderata ed eseguila
```bash
# Esempio: Campagna completa con tutte le 6 famiglie di stressor (120 run)
python3 setup/setup_run.py \
  --model llm \
  --device gpu \
  --benchmark mmlu \
  --use-stressor \
  --all-stressor \
  --repetitions 3 \
  --wait 60 \
  --reboot \
  --ssh-target "gabri@100.92.177.23" \
  --ssh-port 22 \
  --ssh-dest "C:/Users/gabri/OneDrive/Desktop/Universita/Magistrale/Real-Time Systems and Industrial Applications/EdgeAI_Real_Time_Interference_Performance_Study"

# Avvio prima esecuzione
python3 setup/run_experiments.py --reset
```

#### 3. Monitoraggio Headless
In qualsiasi momento puoi collegarti via Tailscale e verificare l'avanzamento:
* **Seguire il log in tempo reale:**
  ```bash
  tail -f headless_doe.log
  ```
* **Verificare quanti esperimenti mancano:**
  ```bash
  python3 setup/run_experiments.py --status
  ```
* **Disabilitare l'avvio automatico al termine:**
  ```bash
  bash setup/setup_autostart.sh disable
  ```

---

### FASE 6: Analisi Statistica e Grafici (Sul PC Host)

Sul computer host Windows (nella cartella di ricezione dei file CSV/JSON):

#### 1. Generazione Grafici di Degradazione Metriche LLM
```powershell
python statistical_analysis/plot_llm_metrics.py
```
*Genera i grafici per TTFT, latenza per-token (TPOT), throughput e jitter in `figures/llm_metrics/`, confrontando le curve di carico $N$ e working-set $S$ con la linea verde tratteggiata della Baseline senza stress.*

#### 2. Generazione Grafici dei Colli di Bottiglia Hardware
```powershell
python statistical_analysis/plot_hardware_bottlenecks.py
```
*Genera i grafici a 1 riga in `figures/hardware_bottlenecks/` per le metriche `vmstat` a 1 Hz: saturazione CPU (%us, %sy, %id, %wa), utilizzo memoria/swap, blocchi I/O, context switch, interrupt e temperature termiche.*

#### 3. Analisi Statistica Avanzata in JMP
- Apri **JMP** sul PC ed esegui [`statistical_analysis/importance_significance.jsl`](statistical_analysis/importance_significance.jsl):
  - Calcola la quota di varianza spiegata ($SS_{\text{Effect}} / SS_{\text{Total}}$).
  - Esegue il test di Shapiro-Wilk sui residui del modello.
  - Se $p \ge 0.05$ (normalità rispettata), valuta il valore $p$ dell'ANOVA ($F$-test).
  - Se $p < 0.05$ (non-normalità), esegue automaticamente il test non parametrico di **Kruskal-Wallis** ($\chi^2$).
- Esegui [`statistical_analysis/distributions.jsl`](statistical_analysis/distributions.jsl) per ottenere istogrammi e QQ-plot univariati di tutte le metriche hardware e di sistema.
