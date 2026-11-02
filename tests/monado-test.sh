#!/bin/sh
# End-to-end test without a headset: Xvfb + Mesa lavapipe + Monado simulated HMD + test app.
# Ubuntu/Debian packages: xvfb monado-service mesa-vulkan-drivers libvulkan-dev libx11-dev imagemagick
# Result: tests/out/headset.png (what the headset shows: left eye | right eye).
set -e
cd "$(dirname "$0")/.."
./build.sh
g++ -std=c++17 -O1 -Ithird_party/include -Itests/app tests/app/app.cpp -o build/test-app -lvulkan -lX11
OUT="$PWD/tests/out"
mkdir -p "$OUT/home/.local/share/vulkan/implicit_layer.d" "$OUT/xdg"
chmod 700 "$OUT/xdg"
sed "s#@LIB@#$PWD/build/libvrlinux_layer.so#" vrlinux_layer.json.in > "$OUT/home/.local/share/vulkan/implicit_layer.d/vrlinux.json"
export HOME="$OUT/home" XDG_RUNTIME_DIR="$OUT/xdg" DISPLAY=:77
Xvfb :77 -screen 0 1920x1080x24 >/dev/null 2>&1 & XV=$!
sleep 1
# monado-service needs a stdin it can poll: give it a fifo we keep open
rm -f "$OUT/stdin"; mkfifo "$OUT/stdin"
XRT_COMPOSITOR_FORCE_XCB=1 P_OVERRIDE_ACTIVE_CONFIG=simulated monado-service < "$OUT/stdin" > "$OUT/monado.log" 2>&1 & MON=$!
exec 3> "$OUT/stdin"
sleep 3
./build/vrlinux-host -v > "$OUT/host.log" 2>&1 & HOST=$!
sleep 3
APP_X=700 ENABLE_VRLINUX=1 ./build/test-app 900 640 480 > "$OUT/app.log" 2>&1 & APP=$!
sleep 10
import -window root -crop 640x360+0+0 "$OUT/headset.png" || true
wait $APP || true
kill $HOST 2>/dev/null; sleep 2
kill $MON 2>/dev/null || true
exec 3>&-
kill $XV 2>/dev/null || true
grep -E "connected|fps" "$OUT/app.log" | tail -4
echo "-> $OUT/headset.png"
