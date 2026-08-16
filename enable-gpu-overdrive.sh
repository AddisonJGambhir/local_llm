#!/usr/bin/env bash
# Enables amdgpu OverDrive (PP_OVERDRIVE_MASK, bit 0x4000) on top of the
# current default ppfeaturemask, so the 7900 XTX's OD_FAN_CURVE sysfs
# interface becomes available. Run with sudo. Does NOT reboot automatically.
set -euo pipefail

GRUB_FILE=/etc/default/grub
BACKUP_FILE=/etc/default/grub.bak-before-ppfeaturemask
TARGET_MASK="0xfff7ffff"

if [[ $EUID -ne 0 ]]; then
    echo "Run this with sudo: sudo bash $0" >&2
    exit 1
fi

if grep -q "amdgpu.ppfeaturemask=" "$GRUB_FILE"; then
    echo "amdgpu.ppfeaturemask is already set in $GRUB_FILE:"
    grep "GRUB_CMDLINE_LINUX_DEFAULT" "$GRUB_FILE"
    echo "Not modifying. Edit $GRUB_FILE by hand if you need a different mask."
    exit 0
fi

if [[ ! -f "$BACKUP_FILE" ]]; then
    cp "$GRUB_FILE" "$BACKUP_FILE"
    echo "Backed up $GRUB_FILE -> $BACKUP_FILE"
fi

sed -i "s/^GRUB_CMDLINE_LINUX_DEFAULT=\"\(.*\)\"/GRUB_CMDLINE_LINUX_DEFAULT=\"\1 amdgpu.ppfeaturemask=${TARGET_MASK}\"/" "$GRUB_FILE"

echo "Updated GRUB_CMDLINE_LINUX_DEFAULT:"
grep "GRUB_CMDLINE_LINUX_DEFAULT" "$GRUB_FILE"

update-grub

echo
echo "Done. Reboot when ready with: sudo reboot"
echo "After reboot, verify with:"
echo "  cat /sys/module/amdgpu/parameters/ppfeaturemask   # should show ${TARGET_MASK}"
echo "  ls /sys/class/drm/card1/device/pp_od_clk_voltage  # should now exist"
