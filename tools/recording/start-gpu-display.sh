#!/bin/sh
# Starts a screen-less 4K display on the Spark's GPU (display :1), for recording demo videos.
# Run with sudo. Stop it with: sudo systemctl stop rec-xorg
# A software display (Xvfb) only manages ~6 fps at 4K; on the GPU the page runs at 60.
D=$(cd "$(dirname "$0")" && pwd)
[ -f "$D/xauth" ] || { touch "$D/xauth"; chmod 600 "$D/xauth"; xauth -f "$D/xauth" add :1 . "$(mcookie)"; chown "${SUDO_USER:-root}" "$D/xauth"; }
BUS=$(nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader | head -1)   # e.g. 0000000F:01:00.0
sed "s|BusID .*|BusID \"PCI:$(printf %d 0x$(echo $BUS | cut -d: -f2))@$(printf %d 0x$(echo $BUS | cut -d: -f1)):$(printf %d 0x$(echo $BUS | cut -d: -f3 | cut -d. -f1)):$(echo $BUS | cut -d. -f2)\"|" "$D/xorg-gpu.conf" > "$D/xorg-gpu.local.conf"
exec systemd-run --unit=rec-xorg /usr/bin/Xorg :1 -config "$D/xorg-gpu.local.conf" -auth "$D/xauth" -nolisten tcp -noreset -novtswitch -logfile "$D/xorg.log"
