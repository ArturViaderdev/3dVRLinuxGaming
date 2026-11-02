#!/bin/sh
# Compila vrlinux (capa Vulkan + vrlinux-host). Solo necesita g++ (C++17).
# Las cabeceras de Vulkan, OpenXR y X11 van incluidas en third_party/.
set -e
cd "$(dirname "$0")"
mkdir -p build
CXX=${CXX:-g++}
FLAGS="-std=c++17 -O2 -g0 -Wall -Wno-missing-field-initializers -Ithird_party/include"
$CXX $FLAGS -fPIC -shared -fvisibility=hidden \
  src/layer.cpp src/spirv_patch.cpp src/config.cpp src/vrlink.cpp \
  -o build/libvrlinux_layer.so.new \
  -static-libstdc++ -static-libgcc -Wl,--no-undefined -Wl,-z,defs -ldl -pthread
mv -f build/libvrlinux_layer.so.new build/libvrlinux_layer.so
$CXX $FLAGS host/host.cpp -o build/vrlinux-host.new -static-libstdc++ -static-libgcc -ldl -pthread
mv -f build/vrlinux-host.new build/vrlinux-host   # works even while it is running
echo "OK -> build/libvrlinux_layer.so build/vrlinux-host"
