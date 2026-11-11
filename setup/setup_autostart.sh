#!/usr/bin/env bash
# ==============================================================================
# setup_autostart.sh - Configurazione esecuzione automatica headless post-boot (Traccia A1)
# ==============================================================================
# Consente alla Jetson AGX Orin di eseguire autonomamente la campagna DoE:
#   1. Al boot attende la rete e Tailscale.
#   2. Esegue l'esperimento successivo in coda.
#   3. Invia i risultati CSV/JSON via SSH sulla tua macchina.
#   4. Riavvia la board e ripete il ciclo fino al completamento della coda.
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
SERVICE_NAME="edgeai-doe.service"
SERVICE_PATH="/etc/systemd/system/${SERVICE_NAME}"
LOG_FILE="${PROJECT_DIR}/headless_doe.log"
CURRENT_USER="$(whoami)"
PYTHON_BIN="$(which python3 || echo "/usr/bin/python3")"

usage() {
    echo "Uso: $0 {enable|disable|status}"
    echo "  enable  : Crea e attiva il servizio systemd per avviare gli esperimenti al boot"
    echo "  disable : Ferma e disabilita il servizio systemd"
    echo "  status  : Mostra lo stato del servizio e le ultime righe del log"
    exit 1
}

enable_service() {
    echo "================================================================="
    echo " ABILITAZIONE ESECUZIONE HEADLESS AUTO-START (systemd)"
    echo "================================================================="
    echo " * Utente di esecuzione: ${CURRENT_USER}"
    echo " * Cartella progetto   : ${PROJECT_DIR}"
    echo " * Interprete Python   : ${PYTHON_BIN}"
    echo " * File di log         : ${LOG_FILE}"
    echo "================================================================="

    # 1. Configurazione permessi sudoers per reboot senza password
    SUDOERS_FILE="/etc/sudoers.d/${CURRENT_USER}-reboot"
    if [ ! -f "${SUDOERS_FILE}" ]; then
        echo " [1/3] Configurazione permessi sudo per reboot senza password..."
        echo "${CURRENT_USER} ALL=(ALL) NOPASSWD: /sbin/reboot, /bin/systemctl reboot, /usr/sbin/nvpmodel, /usr/bin/jetson_clocks" | sudo tee "${SUDOERS_FILE}" > /dev/null
        sudo chmod 0440 "${SUDOERS_FILE}"
        echo "       [OK] File ${SUDOERS_FILE} creato."
    else
        echo " [1/3] [OK] Permessi sudo per reboot già configurati in ${SUDOERS_FILE}."
    fi

    # 2. Generazione del file di servizio systemd
    echo " [2/3] Generazione file di servizio ${SERVICE_PATH}..."
    sudo tee "${SERVICE_PATH}" > /dev/null <<EOF
[Unit]
Description=EdgeAI Real-Time Interference DoE Runner
After=network-online.target tailscaled.service
Wants=network-online.target tailscaled.service

[Service]
Type=simple
User=${CURRENT_USER}
WorkingDirectory=${PROJECT_DIR}
Environment="PATH=/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin:${HOME}/.local/bin"
Environment="LD_LIBRARY_PATH=/usr/local/cuda/lib64"
Environment="PYTHONUNBUFFERED=1"
ExecStart=${PYTHON_BIN} setup/run_experiments.py
Restart=no
StandardOutput=append:${LOG_FILE}
StandardError=append:${LOG_FILE}

[Install]
WantedBy=multi-user.target
EOF

    # 3. Ricaricamento e attivazione
    echo " [3/3] Registrazione e attivazione servizio systemd..."
    sudo systemctl daemon-reload
    sudo systemctl enable "${SERVICE_NAME}"

    echo ""
    echo "================================================================="
    echo " [SUCCESSO] Servizio '${SERVICE_NAME}' abilitato con successo!"
    echo "================================================================="
    echo " La Jetson eseguira' automaticamente la campagna ad ogni riavvio."
    echo ""
    echo " Comandi utili per monitorare l'avanzamento:"
    echo "   tail -f ${LOG_FILE}               # Segui l'output in tempo reale"
    echo "   python3 setup/run_experiments.py --status  # Verifica quanti run mancano"
    echo "   sudo systemctl status ${SERVICE_NAME}      # Controlla lo stato del demone"
    echo ""
    echo " Per avviare subito la prima run puoi eseguire:"
    echo "   sudo systemctl start ${SERVICE_NAME}"
    echo "================================================================="
}

disable_service() {
    echo "================================================================="
    echo " DISABILITAZIONE ESECUZIONE HEADLESS"
    echo "================================================================="
    if [ -f "${SERVICE_PATH}" ]; then
        echo " Arresto e disabilitazione servizio ${SERVICE_NAME}..."
        sudo systemctl stop "${SERVICE_NAME}" 2>/dev/null || true
        sudo systemctl disable "${SERVICE_NAME}" 2>/dev/null || true
        sudo rm -f "${SERVICE_PATH}"
        sudo systemctl daemon-reload
        echo " [OK] Servizio rimosso con successo."
    else
        echo " [INFO] Il servizio ${SERVICE_NAME} non era presente."
    fi
}

show_status() {
    echo "================================================================="
    echo " STATO SERVIZIO HEADLESS (${SERVICE_NAME})"
    echo "================================================================="
    if [ -f "${SERVICE_PATH}" ]; then
        sudo systemctl status "${SERVICE_NAME}" --no-pager || true
    else
        echo " Il servizio ${SERVICE_NAME} non e' installato o e' disabilitato."
    fi

    echo ""
    echo "================================================================="
    echo " ULTIME RIGHE DEL LOG (${LOG_FILE})"
    echo "================================================================="
    if [ -f "${LOG_FILE}" ]; then
        tail -n 25 "${LOG_FILE}"
    else
        echo " Nessun file di log trovato in ${LOG_FILE}."
    fi
}

case "$1" in
    enable)
        enable_service
        ;;
    disable)
        disable_service
        ;;
    status)
        show_status
        ;;
    *)
        usage
        ;;
esac

