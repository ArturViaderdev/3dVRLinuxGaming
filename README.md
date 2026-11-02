<a id="english"></a>

# vrlinux — play normal (flat) Linux games in VR

**English** · [Español](#español)

A Vulkan layer + a small OpenXR program that put ordinary games in your headset in **real geometric 3D**: each eye is a full render from its own position (your real IPD, read from the headset), not a depth-buffer trick.

Two ways to play, switched in game with **Ctrl+Shift+F10**:

- **Immersive** (default): the game image **fills the whole headset view**. Moving your head (lean, crouch, get closer) moves the camera: real 6DOF parallax. Head rotation is off by default (the image stays in front of you and you look around inside it); **Ctrl+Shift+F11** makes the game camera turn with your head.
- **Giant 3D screen**: a cinema screen floating in front of you, with real 3D and perspective when you move your head.

Your **Xbox controller keeps working normally** (nothing is injected, the game reads it directly). Hold **L3 + R3** (press both sticks) to recenter.

Works with **WiVRn** (recommended, e.g. Pico 4) and **SteamVR**, or any OpenXR runtime (Monado…). Games: Windows games on Proton/Wine (DX9/10/11 through DXVK, DX12 through VKD3D-Proton, Vulkan) and native Linux Vulkan games. 64-bit only; OpenGL-only games are not supported.

Based on [stereo3d](https://github.com/ArturViaderdev/Stereo3DSBSLinux) (same shader patching and "replay" engine).

## How it works

```
 game (Proton/DXVK)                                    vrlinux-host (outside the Steam container)
 ┌────────────────────────────────┐   shared GPU      ┌──────────────────────────────────────┐
 │ vrlinux Vulkan layer           │   memory (no CPU  │ OpenXR app (WiVRn / SteamVR)          │
 │  - every vertex shader gets a  │   copies)         │  - shows each eye with the exact pose │
 │    per-eye camera transform    │ ────────────────▶ │    it was rendered with (timewarp)    │
 │  - frame rendered twice:       │                   │  - head pose ─▶ layer                 │
 │    left eye, right eye         │ ◀──────────────── │  - gamepad L3+R3 ─▶ recenter          │
 └────────────────────────────────┘  head pose (shm)  └──────────────────────────────────────┘
```

## Installation

```sh
./install.sh
```

Installs the layer, `vrlinux-host` and the launcher `vrlinux-run` for your user (no root). If `g++` is available it rebuilds for your system (all headers are included). Run it again after every update: an old `~/.config/vrlinux/default.conf` is replaced by the new defaults and kept as `default.conf.old`. Uninstall with `./uninstall.sh`.

## Usage

1. Connect the headset: open WiVRn on the Pico 4 and connect to your PC (or start SteamVR).
2. Steam → game Properties → General → Launch options:

```
/home/YOUR_USER/.local/bin/vrlinux-run %command%
```

With **SteamVR** instead of WiVRn: `VRLINUX_RUNTIME=steamvr /home/YOUR_USER/.local/bin/vrlinux-run %command%` (`VRLINUX_RUNTIME=wivrn` forces WiVRn; without it the active OpenXR runtime is used).

Native Linux game from a terminal: `~/.local/bin/vrlinux-run ./game`.

The launcher starts `vrlinux-host` if it is not running, runs the game with the layer and stops the host at the end. With **Flatpak Steam** start `vrlinux-host` yourself and use `ENABLE_VRLINUX=1 %command%`.

3. Put the headset on, look straight ahead and **recenter** (L3+R3 on the controller, or Ctrl+Shift+Home).

### In-game settings

| Setting | Value | Why |
|---|---|---|
| Field of view | **as high as the game allows**, and the same value in `fov` (Ctrl+Shift+F1/F2) | more image around you; needed for head rotation |
| Resolution (Proton games) | **1920x1080 or less**, 4:3 / 5:4 if possible (1600x1200, 1280x1024) | each eye is rendered: 4K per eye is far too heavy. Native Vulkan games get a suitable resolution automatically (`render_size`) |
| Anti-aliasing | FXAA/SMAA/none, **never TAA/DLSS/FSR2+** | they blend in the other eye's frame |
| Motion blur, frame generation | Off | |
| V-sync | Off | vrlinux paces the game to the headset |

## Controls

| Gamepad | Action |
|---|---|
| hold **L3 + R3** (both sticks, 0.6 s) | recenter |

Keyboard (**Ctrl + Shift + …**; keys can be held to change values continuously):

| Key | Immersive mode | Screen mode |
|---|---|---|
| **F10** | switch to the giant screen | switch to immersive |
| **F11** | camera turns with your head on/off | — |
| F1 / F2 | game FOV − / + | game FOV − / + |
| **F3 / F4** | **less / more 3D** (world scale) | **less / more 3D** (separation) |
| F5 / F6 | mouse sensitivity (only `look = mouse`) | convergence − / + (which distance sits on the screen) |
| F7 | swap eyes | swap eyes |
| F8 | 3D on/off | 3D on/off |
| F9 | **save this game's profile** | save |
| Insert | HUD fixed with the game ⇄ HUD follows your head | — |
| Delete | head turn limit: off → 30° → 45° → 60° → 90° → off | — |
| PgUp / PgDn | HUD closer / farther | screen closer / farther |
| Home | recenter | recenter |
| End | head movement (6DOF) on/off | — |

### 3D strength

- **Screen mode**: F3/F4 change the separation between the eyes (like stereo3d); F5/F6 the convergence.
- **Immersive mode**: F3/F4 change the world scale (`world_scale`): less 3D makes the world look bigger and farther, more 3D smaller and closer (toy-like). The right value is about 1 in most engines and about 100 in Unreal Engine.
- Save with Ctrl+Shift+F9 when it looks right.

### Turning the camera with your head (Ctrl+Shift+F11)

The game camera is re-projected in the shaders to your head orientation, on top of what the stick/mouse do. The game only draws what its own camera sees, so when you turn your head far, objects can be missing or broken at the edges: raise the in-game FOV, or set a limit with Ctrl+Shift+Delete (beyond it the image stops turning instead of showing broken graphics). The HUD (scores, markers) stays fixed with the game by default (`hud_follow = 0`); Ctrl+Shift+Insert makes it follow your head. In deferred-rendered games the lighting may not follow the rotation exactly.

## Configuration

- `~/.config/vrlinux/default.conf`: all games (every option is explained in `default.conf.example`).
- `~/.config/vrlinux/<Game.exe>.conf`: per-game profile (Ctrl+Shift+F9). Edits are reloaded live.
- Built-in presets: Unreal Engine games (`<Name>-Win64-Shipping.exe`) start with `world_scale = 100`; Forza Horizon 3 / Azimuth (`hide_extensions = 0`). A saved profile wins (a profile saved with `world_scale = 1` in an Unreal game makes it look flat: set 100).

Main options:

| Option | Default | |
|---|---|---|
| `display` | `vr` | `vr` immersive, `screen` giant 3D screen |
| `head_look` | `0` | immersive: camera turns with your head |
| `look` | `shader` | how head rotation turns the camera: `shader`, `mouse` (virtual mouse, needs `install.sh --uinput`), `none` |
| `look_limit` | `0` | max head rotation drawn (degrees, 0 = none) |
| `hud_follow` | `0` | HUD follows your head |
| `positional` | `1` | head movement moves the camera |
| `fov` / `fov_axis` | `90` / `h` | the game's FOV (horizontal or vertical) |
| `world_scale` | `1.0` (Unreal: `100`) | immersive 3D strength (game units per metre) |
| `offscreen_mono` | `1` | shadow maps and other square / 2:1 off-screen passes are drawn without the eye offset |
| `separation` / `convergence` | `0.015` / `1.5` | screen mode 3D |
| `screen_distance` / `screen_width` | `3.0` / `5.5` | screen size and distance (m) |
| `render_size` | `auto` | resolution of native Vulkan games (`auto`, `off`, `1832x2016`…) |
| `mirror` | `0` | also show the game on the desktop window (can cause stutter) |
| `mode` | `replay` | `replay` both eyes each frame; `alternate` one frame per eye |
| `flip_y` | `auto` | set 1/0 if looking up moves the world the wrong way |

Environment variables (highest priority): `VRLINUX_DISPLAY`, `VRLINUX_HEAD_LOOK`, `VRLINUX_LOOK`, `VRLINUX_FOV`, `VRLINUX_WORLD_SCALE`, `VRLINUX_SEP`, `VRLINUX_CONV`, `VRLINUX_RENDER_SIZE`, `VRLINUX_MIRROR`, `VRLINUX_MODE`, `VRLINUX_HIDE_EXT`, `VRLINUX_LOOK_LIMIT`, `VRLINUX_HUD_FOLLOW`, `VRLINUX_SCREEN_WIDTH`, `VRLINUX_SCREEN_DISTANCE`, `VRLINUX_LOG_LEVEL`, `VRLINUX_IGNORE=Launcher.exe`. Launcher: `VRLINUX_RUNTIME=steamvr|wivrn|/path/runtime.json`.

## Logs and troubleshooting

- Game: `~/.config/vrlinux/logs/<Game.exe>.log` (every 5 s: game fps, frames sent to the headset, replay stats).
- Host: `~/.config/vrlinux/logs/vrlinux-host.log` (`vrlinux-host -v` also prints the head pose).

| Problem | Fix |
|---|---|
| "no OpenXR runtime found" / "xrCreateInstance failed" | WiVRn not connected / SteamVR not running. The host retries every 3 s. |
| Freezes every few seconds / low fps | keep `mirror = 0`, lower the game resolution (each eye is rendered). |
| "Unsupported graphics card" | try `VRLINUX_HIDE_EXT=0` (Azimuth already has it in its preset). |
| Broken graphics when turning the head | raise the in-game FOV or set a limit with Ctrl+Shift+Delete; or turn head rotation off (Ctrl+Shift+F11). |
| World drifts when turning the head | `fov` does not match the game's FOV: adjust with Ctrl+Shift+F1/F2. |
| Black or 2D frames now and then | the log line `replay 3D x (relaxed y) / 2D z` counts them; if `2D` keeps growing try `mode = alternate` (half the fps per eye). |
| Shadows in the wrong place / different in each eye | keep `offscreen_mono = 1`; the log line `off-screen passes` shows how many were found. |
| Unreal Engine game looks flat / eyes mixed (Air Twister...) | first check `world_scale` is 100 (log line `scale`). Then: |
| Unreal Engine TAA | TAA mixes both eyes. In `Engine.ini` (`.../Saved/Config/Windows/` inside the Proton prefix) add `[SystemSettings]` `r.AntiAliasingMethod=1` (FXAA) and `r.DefaultFeature.AntiAliasing=1`. |
| Game closes or hangs at the logo | send `~/.config/vrlinux/logs/<Game.exe>.log`; `DISABLE_VRLINUX=1` runs it without the layer to compare. |
| Double image | TAA / DLSS / FSR2 / motion blur in the game: disable. |
| "different GPU/driver" | the game and the VR runtime must use the same GPU; Flatpak Steam on AMD/Intel ships its own Mesa: use native Steam. |

## Testing without a headset

`tests/monado-test.sh` runs everything with Monado's simulated headset and Mesa's software Vulkan (lavapipe) under Xvfb.

---

<a id="español"></a>

# vrlinux — juegos normales (planos) de Linux en realidad virtual

[English](#english) · **Español**

Una capa Vulkan + un pequeño programa OpenXR que meten los juegos normales en tu visor en **3D geométrico real**: cada ojo es un render completo desde su propia posición (tu IPD real, leída del visor), no un truco con el búfer de profundidad.

Dos formas de jugar, que se cambian en el juego con **Ctrl+Shift+F10**:

- **Inmersivo** (por defecto): la imagen del juego **llena todo el visor**. Mover la cabeza (inclinarte, agacharte, acercarte) mueve la cámara: paralaje 6DOF real. El giro de cabeza viene apagado (la imagen queda delante y miras dentro de ella); **Ctrl+Shift+F11** hace que la cámara del juego gire con tu cabeza.
- **Pantalla 3D gigante**: una pantalla de cine flotando delante de ti, con 3D real y perspectiva al mover la cabeza.

El **mando de Xbox sigue funcionando normal** (no se inyecta nada, el juego lo lee directamente). Mantén **L3 + R3** (pulsar los dos sticks) para recentrar.

Funciona con **WiVRn** (recomendado, p.ej. Pico 4) y **SteamVR**, o cualquier runtime OpenXR (Monado…). Juegos: juegos de Windows con Proton/Wine (DX9/10/11 con DXVK, DX12 con VKD3D-Proton, Vulkan) y juegos Vulkan nativos de Linux. Solo 64 bits; no funciona con juegos solo OpenGL.

Basado en [stereo3d](https://github.com/ArturViaderdev/Stereo3DSBSLinux) (mismo parcheo de shaders y motor "replay").

## Instalación

```sh
./install.sh
```

Instala la capa, `vrlinux-host` y el lanzador `vrlinux-run` para tu usuario (sin root). Si tienes `g++` se recompila para tu sistema (todas las cabeceras van incluidas). Vuelve a ejecutarlo tras cada actualización: un `~/.config/vrlinux/default.conf` antiguo se sustituye por los nuevos valores y se guarda como `default.conf.old`. Para desinstalar: `./uninstall.sh`.

## Uso

1. Conecta el visor: abre WiVRn en las Pico 4 y conéctate al PC (o arranca SteamVR).
2. Steam → Propiedades del juego → General → Opciones de lanzamiento:

```
/home/TU_USUARIO/.local/bin/vrlinux-run %command%
```

Con **SteamVR** en vez de WiVRn: `VRLINUX_RUNTIME=steamvr /home/TU_USUARIO/.local/bin/vrlinux-run %command%` (`VRLINUX_RUNTIME=wivrn` fuerza WiVRn; sin la variable se usa el runtime OpenXR activo).

Juego nativo desde terminal: `~/.local/bin/vrlinux-run ./juego`.

El lanzador arranca `vrlinux-host` si no está en marcha, ejecuta el juego con la capa y para el host al terminar. Con **Steam Flatpak** arranca tú `vrlinux-host` y usa `ENABLE_VRLINUX=1 %command%`.

3. Ponte el visor, mira al frente y **recentra** (L3+R3 en el mando, o Ctrl+Shift+Inicio).

### Opciones dentro del juego

| Opción | Valor | Por qué |
|---|---|---|
| Campo de visión (FOV) | **el máximo que permita**, y el mismo valor en `fov` (Ctrl+Shift+F1/F2) | más imagen a tu alrededor; necesario para el giro de cabeza |
| Resolución (juegos Proton) | **1920x1080 o menos**, 4:3 / 5:4 si se puede (1600x1200, 1280x1024) | se dibuja cada ojo: 4K por ojo es demasiado. Los juegos Vulkan nativos reciben solos una resolución adecuada (`render_size`) |
| Antialiasing | FXAA/SMAA/ninguno, **nunca TAA/DLSS/FSR2+** | mezclan el fotograma del otro ojo |
| Desenfoque de movimiento, generación de fotogramas | Desactivados | |
| V-sync | Desactivado | vrlinux ajusta el ritmo del juego al visor |

## Controles

| Mando | Acción |
|---|---|
| mantener **L3 + R3** (los dos sticks, 0,6 s) | recentrar |

Teclado (**Ctrl + Shift + …**; se pueden mantener pulsadas para cambiar valores de forma continua):

| Tecla | Modo inmersivo | Modo pantalla |
|---|---|---|
| **F10** | cambiar a pantalla gigante | cambiar a inmersivo |
| **F11** | la cámara gira con la cabeza sí/no | — |
| F1 / F2 | FOV del juego − / + | FOV del juego − / + |
| **F3 / F4** | **menos / más 3D** (escala del mundo) | **menos / más 3D** (separación) |
| F5 / F6 | sensibilidad del ratón (solo `look = mouse`) | convergencia − / + (qué distancia queda en la pantalla) |
| F7 | invertir ojos | invertir ojos |
| F8 | 3D sí/no | 3D sí/no |
| F9 | **guardar el perfil del juego** | guardar |
| Insert | HUD fijo con el juego ⇄ HUD sigue a la cabeza | — |
| Supr | límite de giro de cabeza: sin límite → 30° → 45° → 60° → 90° → sin límite | — |
| RePág / AvPág | HUD más cerca / más lejos | pantalla más cerca / más lejos |
| Inicio | recentrar | recentrar |
| Fin | movimiento de cabeza (6DOF) sí/no | — |

### Fuerza del 3D

- **Modo pantalla**: F3/F4 cambian la separación entre ojos (como en stereo3d); F5/F6 la convergencia.
- **Modo inmersivo**: F3/F4 cambian la escala del mundo (`world_scale`): con menos 3D el mundo se ve más grande y lejano; con más 3D, más pequeño y cercano (de juguete). Lo correcto suele estar cerca de 1 en casi todos los motores y de 100 en Unreal Engine.
- Guarda con Ctrl+Shift+F9 cuando te guste.

### Girar la cámara con la cabeza (Ctrl+Shift+F11)

La cámara del juego se reproyecta en los shaders a la orientación de tu cabeza, sumándose a lo que hagan el stick o el ratón. El juego solo dibuja lo que ve su propia cámara, así que si giras mucho la cabeza pueden faltar objetos o verse gráficos rotos en los bordes: sube el FOV del juego, o pon un límite con Ctrl+Shift+Supr (más allá la imagen deja de girar en vez de verse rota). El HUD (puntuación, marcadores) se queda fijo con el juego por defecto (`hud_follow = 0`); Ctrl+Shift+Insert hace que siga a tu cabeza. En juegos con render diferido la iluminación puede no seguir exactamente el giro.

## Configuración

- `~/.config/vrlinux/default.conf`: todos los juegos (cada opción está explicada en `default.conf.example`).
- `~/.config/vrlinux/<Juego.exe>.conf`: perfil de cada juego (Ctrl+Shift+F9). Se recarga solo si lo editas.
- Perfiles incluidos: los juegos Unreal Engine (`<Nombre>-Win64-Shipping.exe`) arrancan con `world_scale = 100`; Forza Horizon 3 / Azimuth (`hide_extensions = 0`). Un perfil guardado tiene prioridad (un perfil guardado con `world_scale = 1` en un juego Unreal lo deja plano: pon 100).

Opciones principales:

| Opción | Por defecto | |
|---|---|---|
| `display` | `vr` | `vr` inmersivo, `screen` pantalla 3D gigante |
| `head_look` | `0` | inmersivo: la cámara gira con tu cabeza |
| `look` | `shader` | cómo gira la cámara con la cabeza: `shader`, `mouse` (ratón virtual, necesita `install.sh --uinput`), `none` |
| `look_limit` | `0` | giro máximo de cabeza que se dibuja (grados, 0 = sin límite) |
| `hud_follow` | `0` | el HUD sigue a tu cabeza |
| `positional` | `1` | mover la cabeza mueve la cámara |
| `fov` / `fov_axis` | `90` / `h` | FOV del juego (horizontal o vertical) |
| `world_scale` | `1.0` (Unreal: `100`) | fuerza del 3D en inmersivo (unidades del juego por metro) |
| `offscreen_mono` | `1` | los mapas de sombras y otras pasadas fuera de pantalla (cuadradas / 2:1) se dibujan sin el desplazamiento de cada ojo |
| `separation` / `convergence` | `0.015` / `1.5` | 3D del modo pantalla |
| `screen_distance` / `screen_width` | `3.0` / `5.5` | distancia y tamaño de la pantalla (m) |
| `render_size` | `auto` | resolución de los juegos Vulkan nativos (`auto`, `off`, `1832x2016`…) |
| `mirror` | `0` | mostrar también el juego en la ventana del escritorio (puede dar tirones) |
| `mode` | `replay` | `replay` los dos ojos en cada fotograma; `alternate` un fotograma por ojo |
| `flip_y` | `auto` | pon 1/0 si al mirar arriba el mundo se mueve al revés |

Variables de entorno (máxima prioridad): `VRLINUX_DISPLAY`, `VRLINUX_HEAD_LOOK`, `VRLINUX_LOOK`, `VRLINUX_FOV`, `VRLINUX_WORLD_SCALE`, `VRLINUX_SEP`, `VRLINUX_CONV`, `VRLINUX_RENDER_SIZE`, `VRLINUX_MIRROR`, `VRLINUX_MODE`, `VRLINUX_HIDE_EXT`, `VRLINUX_LOOK_LIMIT`, `VRLINUX_HUD_FOLLOW`, `VRLINUX_SCREEN_WIDTH`, `VRLINUX_SCREEN_DISTANCE`, `VRLINUX_LOG_LEVEL`, `VRLINUX_IGNORE=Launcher.exe`. Lanzador: `VRLINUX_RUNTIME=steamvr|wivrn|/ruta/runtime.json`.

## Registros y problemas

- Juego: `~/.config/vrlinux/logs/<Juego.exe>.log` (cada 5 s: fps del juego, fotogramas enviados al visor, estadísticas de replay).
- Host: `~/.config/vrlinux/logs/vrlinux-host.log` (`vrlinux-host -v` muestra también la posición de la cabeza).

| Problema | Solución |
|---|---|
| "no OpenXR runtime found" / "xrCreateInstance failed" | WiVRn no conectado / SteamVR no arrancado. El host lo reintenta cada 3 s. |
| Parones cada pocos segundos / pocos fps | deja `mirror = 0` y baja la resolución del juego (se dibuja cada ojo). |
| "Tarjeta gráfica no soportada" | prueba `VRLINUX_HIDE_EXT=0` (Azimuth ya lo lleva en su perfil incluido). |
| Gráficos rotos al girar la cabeza | sube el FOV del juego o pon un límite con Ctrl+Shift+Supr; o apaga el giro (Ctrl+Shift+F11). |
| El mundo se desplaza al girar la cabeza | el `fov` no coincide con el del juego: ajústalo con Ctrl+Shift+F1/F2. |
| Fotogramas negros o en 2D de vez en cuando | la línea `replay 3D x (relaxed y) / 2D z` del registro los cuenta; si `2D` no para de subir prueba `mode = alternate` (la mitad de fps por ojo). |
| Sombras mal colocadas / distintas en cada ojo | deja `offscreen_mono = 1`; la línea `off-screen passes` del registro dice cuántas se encontraron. |
| Juego Unreal Engine plano / ojos mezclados (Air Twister...) | primero comprueba que `world_scale` es 100 (línea `scale` del registro). Después: |
| TAA de Unreal Engine | el TAA mezcla los dos ojos. En `Engine.ini` (`.../Saved/Config/Windows/` dentro del prefijo de Proton) añade `[SystemSettings]` `r.AntiAliasingMethod=1` (FXAA) y `r.DefaultFeature.AntiAliasing=1`. |
| El juego se cierra o se cuelga en el logo | envía `~/.config/vrlinux/logs/<Juego.exe>.log`; `DISABLE_VRLINUX=1` lo arranca sin la capa para comparar. |
| Imagen doble | TAA / DLSS / FSR2 / desenfoque de movimiento en el juego: desactívalos. |
| "different GPU/driver" | el juego y el runtime VR deben usar la misma GPU; Steam Flatpak con AMD/Intel lleva su propio Mesa: usa Steam nativo. |

## Probar sin visor

`tests/monado-test.sh` lo ejecuta todo con el visor simulado de Monado y Vulkan por software de Mesa (lavapipe) en Xvfb.
