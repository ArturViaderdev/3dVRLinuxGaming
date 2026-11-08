// vrlinux - runtime configuration, per-game profiles and hotkeys (based on stereo3d)
#pragma once
#include <cstdint>
#include <string>

namespace stereo3d {

struct Config {
  // ---- VR
  int display = 0;             // 0 = vr (immersive, default), 1 = screen (giant 3D screen). Ctrl+Shift+F10
  int look = 2;                // 2 = shader (head rotation drawn in the shaders, stick/mouse untouched), 1 = mouse, 0 = none
  bool headLook = false;       // head turns the game camera (look = shader/mouse). Ctrl+Shift+F11
  bool lookPitch = true;       // head pitch also sent to the game (mouse Y)
  float mouseSens = 20.0f;     // mouse counts per degree of camera rotation (calibrate per game)
  float pitchLimit = 80.0f;    // degrees
  int lookLatency = 0;         // frames the game takes to apply mouse input (0-4): image placed with an older pose
  bool invertY = false;        // game uses inverted mouse Y
  bool positional = true;      // head translation moves the game camera (lean, crouch...)
  float posLimit = 0.5f;       // max head translation from the recentred point (m)
  float fov = 90.0f;           // game field of view (degrees), as set in the game
  int fovAxis = 0;             // 0 = fov is horizontal, 1 = vertical
  float worldScale = 1.0f;     // game units per metre (1 most engines, 100 Unreal)
  float hudDistance = 0.0f;    // metres at which 2D HUD is shown (0 = infinity, no shift)
  float hudScale = 1.0f;       // scale of 2D HUD around the centre (1 = unchanged)
  bool hudFollow = false;      // shader look: HUD follows the head (false = fixed with the game camera)
  float lookLimit = 0.0f;      // shader look: max head rotation drawn (degrees, 0 = no limit); beyond it the image stays put
  int flipY = -1;              // clip-space Y points down: -1 auto (viewport sign), 0 no, 1 yes
  float overscan = 1.05f;      // shader look: render a bit more than the headset FOV (for reprojection)
  int fixShadows = 2;          // screen-space position rebuilds corrected (Unreal shadows): 0 off, 1 on, 2 auto
  bool fixShadowsOn = true;    // runtime switch (Ctrl+Shift+Pause) to compare with / without
  bool offscreenMono = true;   // passes whose shape differs from the game frame (shadow maps) get no eye transform
  bool depthClamp = true;      // shader look: keep geometry beside/behind the game camera inside the depth range
  int renderW = -1, renderH = -1;  // size reported to native Vulkan games (-1 auto, 0 off)
  bool mirror = false;         // show the game on the desktop window too (can stall on some drivers)
  float screenDist = 3.0f;     // screen mode: distance (m)
  float screenWidth = 5.5f;    // screen mode: width (m)
  // ---- classic stereo (screen mode)
  float separation = 0.015f;   // per-eye shift at infinity, in NDC units
  float convergence = 1.5f;    // view depth (game units) that lands on the screen plane
  float hudDepth = 0.0f;       // screen mode HUD depth (0 = screen plane)
  // ---- common
  bool swapEyes = false;
  bool stereo = true;          // false = both eyes see the same image
  int presentMode = 1;         // desktop mirror window present mode (1 = mailbox: never waits for the monitor)
  bool hideExtensions = true;
  bool patchShaders = true;
  int logLevel = 1;
  bool hotkeys = true;
  bool passthrough = false;    // launcher/helper process: layer does nothing
  int mode = 1;                // 1 replay (both eyes every game frame), 0 alternate, 2 depth, 3 hybrid
  float fpsLimit = 0.0f;       // stereo frames per second. 0 = headset refresh rate, <0 = no limit
  bool autoSwitch = true;
  bool replayRelaxed = true;   // replay even when the game needs GPU results mid-frame (readbacks, vkd3d)
};

Config& Cfg();
float HudFactor();            // screen mode: 1 - convergence / hud_depth
void LoadConfig();
void SaveConfig();
const std::string& GameName();
const char* ModeName(int mode);
std::string ProfilePath();
uint32_t RecenterRequests();  // incremented by Ctrl+Shift+Home

void Log(int level, const char* fmt, ...) __attribute__((format(printf, 2, 3)));

// Polls hotkeys (X11, Ctrl+Shift+...) and config-file changes. Returns true if something changed.
bool PollControls();

}  // namespace stereo3d
