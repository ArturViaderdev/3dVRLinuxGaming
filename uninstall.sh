#!/bin/sh
# Desinstala vrlinux (los perfiles en ~/.config/vrlinux se conservan).
D="${XDG_DATA_HOME:-$HOME/.local/share}"
rm -f "$D/vulkan/implicit_layer.d/vrlinux.json" "$HOME/.local/bin/vrlinux-run" "$HOME/.local/bin/vrlinux-host"
rm -rf "$HOME/.local/lib/vrlinux"
FP="$HOME/.var/app/com.valvesoftware.Steam"
rm -f "$FP/data/vulkan/implicit_layer.d/vrlinux.json"; rm -rf "$FP/data/vrlinux-lib"
[ -f /etc/udev/rules.d/60-vrlinux-uinput.rules ] && echo "Regla udev: sudo rm /etc/udev/rules.d/60-vrlinux-uinput.rules /etc/modules-load.d/vrlinux-uinput.conf"
echo "vrlinux desinstalado (perfiles conservados en ~/.config/vrlinux)."
