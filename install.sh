#!/bin/sh
# Instala vrlinux para tu usuario (no necesita root).
#   ./install.sh            capa Vulkan + vrlinux-host + lanzador vrlinux-run
#   ./install.sh --uinput   ademas da permiso a /dev/uinput (solo para look = mouse, pide sudo)
set -e
cd "$(dirname "$0")"

if [ "$1" = "--uinput" ]; then
  echo 'KERNEL=="uinput", SUBSYSTEM=="misc", TAG+="uaccess", OPTIONS+="static_node=uinput"' |
    sudo tee /etc/udev/rules.d/60-vrlinux-uinput.rules >/dev/null
  sudo modprobe uinput 2>/dev/null || true
  echo uinput | sudo tee /etc/modules-load.d/vrlinux-uinput.conf >/dev/null
  sudo udevadm control --reload-rules && sudo udevadm trigger --name-match=uinput 2>/dev/null || true
  echo "Regla udev instalada: /dev/uinput accesible para tu usuario (si no, cierra sesion y vuelve a entrar)."
fi

if command -v g++ >/dev/null 2>&1; then
  ./build.sh || echo "Aviso: no se pudo recompilar, uso los binarios incluidos."
fi
[ -f build/libvrlinux_layer.so ] && [ -f build/vrlinux-host ] || { echo "Faltan binarios (instala g++ y ejecuta ./build.sh)"; exit 1; }

install_to() {   # $1 = data dir (contiene vulkan/implicit_layer.d), $2 = lib dir
  mkdir -p "$1/vulkan/implicit_layer.d" "$2"
  cp build/libvrlinux_layer.so "$2/"
  sed "s#@LIB@#$2/libvrlinux_layer.so#" vrlinux_layer.json.in > "$1/vulkan/implicit_layer.d/vrlinux.json"
  echo "  capa -> $1/vulkan/implicit_layer.d/vrlinux.json"
}

echo "Instalando vrlinux..."
install_to "${XDG_DATA_HOME:-$HOME/.local/share}" "$HOME/.local/lib/vrlinux"
FP="$HOME/.var/app/com.valvesoftware.Steam"
if [ -d "$FP" ]; then
  install_to "$FP/data" "$FP/data/vrlinux-lib"
  echo "  (detectado Steam Flatpak: arranca vrlinux-host a mano antes del juego)"
fi

mkdir -p "$HOME/.local/bin"
cp build/vrlinux-host "$HOME/.local/bin/vrlinux-host.new" && mv -f "$HOME/.local/bin/vrlinux-host.new" "$HOME/.local/bin/vrlinux-host"
cp tools/vrlinux-run "$HOME/.local/bin/vrlinux-run"
chmod +x "$HOME/.local/bin/vrlinux-run" "$HOME/.local/bin/vrlinux-host"
echo "  host     -> $HOME/.local/bin/vrlinux-host"
echo "  lanzador -> $HOME/.local/bin/vrlinux-run"

CFG="${XDG_CONFIG_HOME:-$HOME/.config}/vrlinux"
mkdir -p "$CFG"
update_default() {  # $1 = config dir: new defaults replace an older default.conf (kept as .old)
  if [ -f "$1/default.conf" ] && ! grep -q "vrlinux-config-version 7" "$1/default.conf"; then
    mv -f "$1/default.conf" "$1/default.conf.old"
    echo "  default.conf actualizado (el anterior queda en default.conf.old)"
  fi
  [ -f "$1/default.conf" ] || cp default.conf.example "$1/default.conf"
}
update_default "$CFG"
[ -d "$FP" ] && mkdir -p "$FP/config/vrlinux" && update_default "$FP/config/vrlinux"


cat <<EOT

Listo. Opciones de lanzamiento en Steam (Propiedades del juego > General):

  $HOME/.local/bin/vrlinux-run %command%

Con SteamVR en vez de WiVRn:

  VRLINUX_RUNTIME=steamvr $HOME/.local/bin/vrlinux-run %command%

Antes: conecta las Pico 4 a WiVRn (o arranca SteamVR). Recentrar: manten L3+R3 en el mando.
Teclas Ctrl+Shift: F1/F2 FOV | F3/F4 escala del mundo | F10 pantalla/inmersivo | F11 giro con la cabeza | F9 guardar
(--uinput solo hace falta si usas look = mouse)
Lee README.md para todo lo demas.
EOT
