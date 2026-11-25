#!/usr/bin/env bash
# ==============================================================================
# setup_autostart.sh - Unico punto per automatizzare gli esperimenti al boot (Traccia A1)
# ==============================================================================
# Installa un solo servizio systemd (edgeai-doe.service) che a ogni boot:
#   1. applica nvpmodel -m 0 e jetson_clocks;
#   2. esegue il prossimo esperimento in coda;
#   3. riavvia la board (con reboot_after_run / --reboot) e ripete fino a fine coda.
#
# Due modalità:
#   single : esegue setup/run_experiments.py con il run_config.json corrente
#            (prima configura con: python3 setup/run_experiments.py <flag> --reboot --status)
#   mmlu   : esegue la campagna completa run_mmlu_campaign.sh (baseline + stressor,
#            tutti i modelli x device, una ripetizione per boot)
#
# Uso:
#   bash setup/setup_autostart.sh enable [single|mmlu]   # default: single
#   bash setup/setup_autostart.sh start                  # avvia subito la prima run
#   bash setup/setup_autostart.sh status                 # servizio, avanzamento, log
#   bash setup/setup_autostart.sh disable                # ferma e rimuove il servizio
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
SERVICE_NAME="edgeai-doe.service"
SERVICE_PATH="/etc/systemd/system/${SERVICE_NAME}"
LEGACY_SERVICES=("edgeai-mmlu.service")
LOG_FILE="${PROJECT_DIR}/headless_doe.log"
MODE_FILE="${PROJECT_DIR}/setup/autostart_mode.txt"
CAMPAIGN_SCRIPT="${PROJECT_DIR}/run_mmlu_campaign.sh"
CURRENT_USER="$(whoami)"
PYTHON_BIN="$(command -v python3 || echo "/usr/bin/python3")"

usage() {
    echo "Uso: $0 {enable [single|mmlu]|start|status|disable}"
    echo "  enable single : al boot esegue run_experiments.py con il run_config.json corrente (default)"
    echo "  enable mmlu   : al boot esegue la campagna completa run_mmlu_campaign.sh"
    echo "  start         : avvia subito il servizio (prima run)"
    echo "  status        : stato del servizio, avanzamento della coda e ultime righe del log"
    echo "  disable       : ferma e rimuove il servizio"
    exit 1
}

# Rimuove i servizi di avvio installati in passato da altri script, per non avere due run in parallelo
remove_legacy_services() {
    local svc
    for svc in "${LEGACY_SERVICES[@]}"; do
        if [ -f "/etc/systemd/system/${svc}" ]; then
            sudo systemctl stop "${svc}" 2>/dev/null || true
            sudo systemctl disable "${svc}" 2>/dev/null || true
            sudo rm -f "/etc/systemd/system/${svc}"
            echo "       [OK] Vecchio servizio ${svc} rimosso."
        fi
    done
}

enable_service() {
    local mode="${1:-single}" exec_cmd description
    case "${mode}" in
        single)
            exec_cmd="${PYTHON_BIN} -u setup/run_experiments.py"
            description="EdgeAI DoE runner (run_experiments.py, una run per boot)"
            if [ ! -f "${PROJECT_DIR}/setup/run_config.json" ]; then
                echo "[ERRORE] setup/run_config.json non esiste: configura prima la run, es."
                echo "  python3 setup/run_experiments.py -m slm -d gpu -b mmlu --no-stressor -n 30 -r 20 --reboot --status"
                exit 1
            fi
            ;;
        mmlu)
            exec_cmd="/bin/bash ${CAMPAIGN_SCRIPT} run"
            description="EdgeAI campagna MMLU (run_mmlu_campaign.sh, una ripetizione per boot)"
            if [ ! -f "${CAMPAIGN_SCRIPT}" ]; then
                echo "[ERRORE] ${CAMPAIGN_SCRIPT} non trovato."
                exit 1
            fi
            ;;
        *)
            usage
            ;;
    esac

    echo "================================================================="
    echo " ABILITAZIONE ESECUZIONE AUTOMATICA AL BOOT (systemd)"
    echo "================================================================="
    echo " * Modalità            : ${mode}"
    echo " * Utente di esecuzione: ${CURRENT_USER}"
    echo " * Cartella progetto   : ${PROJECT_DIR}"
    echo " * Comando             : ${exec_cmd}"
    echo " * File di log         : ${LOG_FILE}"
    echo "================================================================="

    # 1. Permessi sudo senza password per reboot e clock
    SUDOERS_FILE="/etc/sudoers.d/${CURRENT_USER}-reboot"
    if [ ! -f "${SUDOERS_FILE}" ]; then
        echo " [1/4] Configurazione permessi sudo per reboot senza password..."
        echo "${CURRENT_USER} ALL=(ALL) NOPASSWD: /sbin/reboot, /bin/systemctl reboot, /usr/sbin/nvpmodel, /usr/bin/jetson_clocks" | sudo tee "${SUDOERS_FILE}" > /dev/null
        sudo chmod 0440 "${SUDOERS_FILE}"
        echo "       [OK] File ${SUDOERS_FILE} creato."
    else
        echo " [1/4] [OK] Permessi sudo già configurati in ${SUDOERS_FILE}."
    fi

    # 2. Un solo servizio di avvio attivo
    echo " [2/4] Rimozione di eventuali servizi di avvio precedenti..."
    remove_legacy_services

    # 3. File di servizio
    echo " [3/4] Generazione file di servizio ${SERVICE_PATH}..."
    sudo tee "${SERVICE_PATH}" > /dev/null <<EOF
[Unit]
Description=${description}
After=network-online.target tailscaled.service
Wants=network-online.target

[Service]
Type=simple
User=${CURRENT_USER}
WorkingDirectory=${PROJECT_DIR}
Environment="PATH=/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin:${HOME}/.local/bin"
Environment="LD_LIBRARY_PATH=/usr/local/cuda/lib64"
Environment="PYTHONUNBUFFERED=1"
ExecStartPre=-+/usr/sbin/nvpmodel -m 0
ExecStartPre=-+/usr/bin/jetson_clocks
ExecStart=${exec_cmd}
Restart=no
StandardOutput=append:${LOG_FILE}
StandardError=append:${LOG_FILE}

[Install]
WantedBy=multi-user.target
EOF
    echo "${mode}" > "${MODE_FILE}"

    # 4. Attivazione
    echo " [4/4] Registrazione e attivazione servizio systemd..."
    sudo systemctl daemon-reload
    sudo systemctl enable "${SERVICE_NAME}"

    echo ""
    echo "================================================================="
    echo " [SUCCESSO] ${SERVICE_NAME} abilitato in modalità '${mode}'."
    echo "================================================================="
    echo "   bash setup/setup_autostart.sh start    # avvia subito la prima run"
    echo "   bash setup/setup_autostart.sh status   # avanzamento"
    echo "   tail -f ${LOG_FILE}"
    echo "================================================================="
}

start_service() {
    if [ ! -f "${SERVICE_PATH}" ]; then
        echo "[ERRORE] Servizio non installato: esegui prima '$0 enable [single|mmlu]'."
        exit 1
    fi
    sudo systemctl start --no-block "${SERVICE_NAME}"
    echo " [OK] ${SERVICE_NAME} avviato. Segui il log con: tail -f ${LOG_FILE}"
}

disable_service() {
    echo "================================================================="
    echo " DISABILITAZIONE ESECUZIONE AUTOMATICA"
    echo "================================================================="
    if [ -f "${SERVICE_PATH}" ]; then
        sudo systemctl stop "${SERVICE_NAME}" 2>/dev/null || true
        sudo systemctl disable "${SERVICE_NAME}" 2>/dev/null || true
        sudo rm -f "${SERVICE_PATH}"
        echo " [OK] ${SERVICE_NAME} rimosso."
    else
        echo " [INFO] ${SERVICE_NAME} non era installato."
    fi
    remove_legacy_services
    sudo systemctl daemon-reload
    rm -f "${MODE_FILE}"
}

show_status() {
    local mode
    mode="$(cat "${MODE_FILE}" 2>/dev/null || echo "?")"
    echo "================================================================="
    echo " SERVIZIO ${SERVICE_NAME} (modalità: ${mode})"
    echo "================================================================="
    if [ -f "${SERVICE_PATH}" ]; then
        echo " Abilitato al boot: $(systemctl is-enabled "${SERVICE_NAME}" 2>/dev/null || echo no)"
        echo " Stato attuale    : $(systemctl is-active "${SERVICE_NAME}" 2>/dev/null || echo inactive)"
    else
        echo " Non installato."
    fi
    for svc in "${LEGACY_SERVICES[@]}"; do
        [ -f "/etc/systemd/system/${svc}" ] && echo " [ATTENZIONE] È ancora installato ${svc}: esegui '$0 disable' e poi 'enable'."
    done

    echo ""
    echo "================================================================="
    echo " AVANZAMENTO"
    echo "================================================================="
    cd "${PROJECT_DIR}"
    if [ "${mode}" = "mmlu" ]; then
        bash "${CAMPAIGN_SCRIPT}" status || true
    else
        "${PYTHON_BIN}" setup/run_experiments.py --status < /dev/null || true
    fi

    echo ""
    echo "================================================================="
    echo " ULTIME RIGHE DEL LOG (${LOG_FILE})"
    echo "================================================================="
    if [ -f "${LOG_FILE}" ]; then
        tail -n 25 "${LOG_FILE}"
    else
        echo " Nessun log ancora."
    fi
}

case "${1:-}" in
    enable)  enable_service "${2:-single}" ;;
    start)   start_service ;;
    disable) disable_service ;;
    status)  show_status ;;
    *)       usage ;;
esac
