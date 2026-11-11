#!/bin/bash
# ==============================================================================
# setup_swap.sh - Configurazione Swap File da 16GB su NVIDIA Jetson AGX / Nano Orin
# ==============================================================================
# Su Jetson Orin con 8GB di memoria LPDDR5 unificata, i modelli LLM da 8B
# (come LLaMA-3.1-8B-Instruct, ~4.92GB) necessitano di memoria swap su disco NVMe
# per evitare Out-Of-Memory (OOM Killer) durante il caricamento e l'inferenza CPU/GPU.
#
# Uso:
#   sudo bash setup/setup_swap.sh
# ==============================================================================

set -e

if [ "$(id -u)" -ne 0 ]; then
    echo "[ERRORE] Questo script deve essere eseguito con privilegi di root (es. sudo bash setup/setup_swap.sh)."
    exit 1
fi

SWAP_FILE="/swapfile"
# Calcola lo spazio disco disponibile sulla partizione root
AVAIL_DISK_MB=$(df -m / | tail -1 | awk '{print $4}')
if [ "$AVAIL_DISK_MB" -lt 19000 ]; then
    # Se lo spazio libero è inferiore a 19 GB (es. MicroSD da 64GB con 11GB liberi),
    # usa 8GB di swap per non saturare il disco e lasciare sempre almeno 3-4 GB liberi.
    SWAP_SIZE_GB=8
    echo "[INFO] Spazio disco libero limitato (${AVAIL_DISK_MB} MB). Dimensione swap impostata a 8 GB per sicurezza."
else
    SWAP_SIZE_GB=16
fi

echo "===================================================================="
echo " Verifica e configurazione Swap da ${SWAP_SIZE_GB}GB per Edge AI su Jetson Orin"
echo "===================================================================="

CURRENT_SWAP_KB=$(grep SwapTotal /proc/meminfo | awk '{print $2}')
CURRENT_SWAP_MB=$((CURRENT_SWAP_KB / 1024))
echo "[INFO] Swap attuale rilevata nel sistema: ${CURRENT_SWAP_MB} MB"

if [ -f "$SWAP_FILE" ]; then
    echo "[INFO] File swap $SWAP_FILE già esistente."
    if swapon --show | grep -q "$SWAP_FILE"; then
        echo "[OK] Lo swapfile $SWAP_FILE è già attivo!"
        swapon --show
        exit 0
    else
        echo "[INFO] Attivazione dello swapfile esistente..."
        swapon "$SWAP_FILE" || true
        swapon --show
        exit 0
    fi
fi

echo "[1/4] Creazione swapfile da ${SWAP_SIZE_GB}GB in $SWAP_FILE (allocazione su storage NVMe/SSD)..."
if command -v fallocate &> /dev/null; then
    fallocate -l "${SWAP_SIZE_GB}G" "$SWAP_FILE" || dd if=/dev/zero of="$SWAP_FILE" bs=1G count="$SWAP_SIZE_GB" status=progress
else
    dd if=/dev/zero of="$SWAP_FILE" bs=1G count="$SWAP_SIZE_GB" status=progress
fi

echo "[2/4] Impostazione permessi di sicurezza (chmod 600)..."
chmod 600 "$SWAP_FILE"

echo "[3/4] Formattazione swap (mkswap)..."
mkswap "$SWAP_FILE"

echo "[4/4] Attivazione dello swapfile (swapon)..."
swapon "$SWAP_FILE"

# Persistenza al riavvio in /etc/fstab
if ! grep -q "$SWAP_FILE" /etc/fstab; then
    echo "$SWAP_FILE none swap sw 0 0" >> /etc/fstab
    echo "[OK] Aggiunta voce a /etc/fstab per persistenza automatica al boot."
fi

# Ottimizzazione swappiness per prevenire swap aggressivo non necessario
sysctl vm.swappiness=30 > /dev/null || true
if ! grep -q "vm.swappiness" /etc/sysctl.conf 2>/dev/null; then
    echo "vm.swappiness=30" >> /etc/sysctl.conf
fi

echo "===================================================================="
echo "[SUCCESSO] Configurazione Swap completata con successo!"
swapon --show
free -h
echo "===================================================================="

