// vrlinux - runtime configuration, per-game profiles and hotkeys (based on stereo3d)
#include "config.h"

#include <dlfcn.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>

namespace stereo3d {

static Config g_cfg;
static std::string g_game;
static std::string g_profile;
static time_t g_profileMtime = 0;
static FILE* g_logFile = nullptr;
static std::mutex g_logMutex;
static uint32_t g_recenter = 0;
static std::string g_preset;

Config& Cfg() { return g_cfg; }
float HudFactor() {
  if (g_cfg.hudDepth <= 0.0f) return 0.0f;
  return 1.0f - g_cfg.convergence / g_cfg.hudDepth;
}
const std::string& GameName() { return g_game; }
std::string ProfilePath() { return g_profile; }
uint32_t RecenterRequests() { return g_recenter; }

void Log(int level, const char* fmt, ...) {
  if (level > g_cfg.logLevel) return;
  char buf[1024];
  va_list ap;
  va_start(ap, fmt);
  vsnprintf(buf, sizeof(buf), fmt, ap);
  va_end(ap);
  std::lock_guard<std::mutex> l(g_logMutex);
  fprintf(stderr, "[vrlinux] %s\n", buf);
  if (g_logFile) {
    fprintf(g_logFile, "%s\n", buf);
    fflush(g_logFile);
  }
}

// --- small parsing helpers (avoid strtol: glibc 2.38 C23 symbol versioning) -------
static bool ParseFloat(const char* s, float& out) {
  if (!s || !*s) return false;
  char* end = nullptr;
  float v = strtof(s, &end);
  if (end == s) return false;
  out = v;
  return true;
}
static bool ParseBool(const char* s, bool& out) {
  if (!s) return false;
  if (!strcmp(s, "1") || !strcasecmp(s, "true") || !strcasecmp(s, "on") || !strcasecmp(s, "yes")) {
    out = true; return true;
  }
  if (!strcmp(s, "0") || !strcasecmp(s, "false") || !strcasecmp(s, "off") || !strcasecmp(s, "no")) {
    out = false; return true;
  }
  return false;
}
static int ParsePresentMode(const char* s, int def) {
  if (!s) return def;
  if (!strcasecmp(s, "fifo") || !strcasecmp(s, "vsync")) return 2;
  if (!strcasecmp(s, "mailbox")) return 1;
  if (!strcasecmp(s, "immediate")) return 0;
  if (!strcasecmp(s, "fifo_relaxed")) return 3;
  return def;
}
static const char* PresentModeName(int m) {
  switch (m) { case 0: return "immediate"; case 1: return "mailbox"; case 3: return "fifo_relaxed"; default: return "fifo"; }
}

static void ApplyKey(const char* key, const char* val) {
  float f;
  bool b;
  Config& c = g_cfg;
  if (!strcmp(key, "display")) c.display = !strcasecmp(val, "screen") ? 1 : 0;
  else if (!strcmp(key, "look"))
    c.look = (!strcasecmp(val, "mouse") || !strcmp(val, "1")) ? 1 : (!strcasecmp(val, "shader") || !strcmp(val, "2")) ? 2 : 0;
  else if (!strcmp(key, "head_look") && ParseBool(val, b)) c.headLook = b;
  else if (!strcmp(key, "look_pitch") && ParseBool(val, b)) c.lookPitch = b;
  else if (!strcmp(key, "mouse_sens") && ParseFloat(val, f) && f > 0) c.mouseSens = f;
  else if (!strcmp(key, "look_latency") && ParseFloat(val, f)) c.lookLatency = std::clamp(int(f + 0.5f), 0, 4);
  else if (!strcmp(key, "pitch_limit") && ParseFloat(val, f)) c.pitchLimit = std::clamp(f, 0.0f, 89.0f);
  else if (!strcmp(key, "invert_y") && ParseBool(val, b)) c.invertY = b;
  else if (!strcmp(key, "positional") && ParseBool(val, b)) c.positional = b;
  else if (!strcmp(key, "positional_limit") && ParseFloat(val, f)) c.posLimit = std::max(0.0f, f);
  else if (!strcmp(key, "fov") && ParseFloat(val, f)) c.fov = std::clamp(f, 20.0f, 170.0f);
  else if (!strcmp(key, "fov_axis")) c.fovAxis = (!strcasecmp(val, "v") || !strcasecmp(val, "vertical")) ? 1 : 0;
  else if (!strcmp(key, "world_scale") && ParseFloat(val, f) && f > 0) c.worldScale = f;
  else if (!strcmp(key, "hud_distance") && ParseFloat(val, f)) c.hudDistance = std::max(0.0f, f);
  else if (!strcmp(key, "hud_scale") && ParseFloat(val, f)) c.hudScale = std::clamp(f, 0.2f, 2.0f);
  else if (!strcmp(key, "flip_y")) {
    if (!strcasecmp(val, "auto")) c.flipY = -1;
    else if (ParseBool(val, b)) c.flipY = b ? 1 : 0;
  }
  else if (!strcmp(key, "hud_follow") && ParseBool(val, b)) c.hudFollow = b;
  else if (!strcmp(key, "look_limit") && ParseFloat(val, f)) c.lookLimit = std::clamp(f, 0.0f, 180.0f);
  else if (!strcmp(key, "overscan") && ParseFloat(val, f)) c.overscan = std::clamp(f, 1.0f, 1.5f);
  else if (!strcmp(key, "depth_clamp") && ParseBool(val, b)) c.depthClamp = b;
  else if (!strcmp(key, "mirror") && ParseBool(val, b)) c.mirror = b;
  else if (!strcmp(key, "render_size")) {
    unsigned w = 0, h = 0;
    if (!strcasecmp(val, "auto")) c.renderW = c.renderH = -1;
    else if (!strcasecmp(val, "off") || !strcmp(val, "0")) c.renderW = c.renderH = 0;
    else if (sscanf(val, "%ux%u", &w, &h) == 2 && w >= 64 && h >= 64) { c.renderW = int(w); c.renderH = int(h); }
  }
  else if (!strcmp(key, "screen_distance") && ParseFloat(val, f)) c.screenDist = std::clamp(f, 0.5f, 50.0f);
  else if (!strcmp(key, "screen_width") && ParseFloat(val, f)) c.screenWidth = std::clamp(f, 0.3f, 100.0f);
  else if (!strcmp(key, "separation") && ParseFloat(val, f)) c.separation = f;
  else if (!strcmp(key, "convergence") && ParseFloat(val, f)) c.convergence = f;
  else if (!strcmp(key, "hud_depth") && ParseFloat(val, f)) c.hudDepth = f;
  else if (!strcmp(key, "swap_eyes") && ParseBool(val, b)) c.swapEyes = b;
  else if (!strcmp(key, "stereo") && ParseBool(val, b)) c.stereo = b;
  else if (!strcmp(key, "present_mode")) c.presentMode = ParsePresentMode(val, c.presentMode);
  else if (!strcmp(key, "auto_switch") && ParseBool(val, b)) c.autoSwitch = b;
  else if (!strcmp(key, "fps_limit")) {
    if (!strcasecmp(val, "auto")) c.fpsLimit = 0.0f;
    else if (!strcasecmp(val, "off") || !strcasecmp(val, "none")) c.fpsLimit = -1.0f;
    else if (ParseFloat(val, f)) c.fpsLimit = f;
  }
  else if (!strcmp(key, "mode")) c.mode = !strcasecmp(val, "replay") ? 1 : 0;
  else if (!strcmp(key, "hide_extensions") && ParseBool(val, b)) c.hideExtensions = b;
  else if (!strcmp(key, "patch_shaders") && ParseBool(val, b)) c.patchShaders = b;
  else if (!strcmp(key, "log_level") && ParseFloat(val, f)) c.logLevel = int(f);
  else if (!strcmp(key, "hotkeys") && ParseBool(val, b)) c.hotkeys = b;
}

static bool LoadFile(const std::string& path) {
  FILE* f = fopen(path.c_str(), "r");
  if (!f) return false;
  char line[512];
  while (fgets(line, sizeof(line), f)) {
    char* p = line;
    while (*p == ' ' || *p == '\t') ++p;
    if (*p == '#' || *p == ';' || *p == '\n' || !*p) continue;
    char* eq = strchr(p, '=');
    if (!eq) continue;
    *eq = 0;
    char* k = p;
    char* v = eq + 1;
    auto trim = [](char* s) {
      size_t n = strlen(s);
      while (n && (s[n - 1] == '\n' || s[n - 1] == '\r' || s[n - 1] == ' ' || s[n - 1] == '\t')) s[--n] = 0;
      while (*s == ' ' || *s == '\t') ++s;
      return s;
    };
    ApplyKey(trim(k), trim(v));
  }
  fclose(f);
  return true;
}

static std::string ConfigDir() {
  if (const char* x = getenv("XDG_CONFIG_HOME"); x && *x) return std::string(x) + "/vrlinux";
  if (const char* h = getenv("HOME"); h && *h) return std::string(h) + "/.config/vrlinux";
  return "/tmp/vrlinux";
}

static std::string DetectGameName() {
  // Under Proton argv[0] is usually the Windows exe path (Z:\...\Game.exe or C:\...).
  std::string name;
  FILE* f = fopen("/proc/self/cmdline", "rb");
  if (f) {
    char buf[4096];
    size_t n = fread(buf, 1, sizeof(buf) - 1, f);
    fclose(f);
    buf[n] = 0;
    size_t off = 0;
    std::string first;
    while (off < n) {
      std::string arg(buf + off);
      if (first.empty()) first = arg;
      size_t len = arg.size();
      if (len > 4 && !strcasecmp(arg.c_str() + len - 4, ".exe")) { name = arg; break; }
      off += len + 1;
    }
    if (name.empty()) name = first;
  }
  size_t slash = name.find_last_of("/\\");
  if (slash != std::string::npos) name = name.substr(slash + 1);
  for (char& c : name)
    if (c == ' ' || c == ':' || c == '"' || c == '\'') c = '_';
  if (name.empty()) name = "unknown";
  return name;
}

static void LoadEnv() {
  static const struct { const char* env; const char* key; } kMap[] = {
      {"VRLINUX_DISPLAY", "display"},         {"VRLINUX_LOOK", "look"}, {"VRLINUX_HEAD_LOOK", "head_look"},
      {"VRLINUX_LOOK_PITCH", "look_pitch"},   {"VRLINUX_MOUSE_SENS", "mouse_sens"},
      {"VRLINUX_INVERT_Y", "invert_y"},       {"VRLINUX_LOOK_LATENCY", "look_latency"},       {"VRLINUX_POSITIONAL", "positional"},
      {"VRLINUX_FOV", "fov"},                 {"VRLINUX_FOV_AXIS", "fov_axis"},
      {"VRLINUX_WORLD_SCALE", "world_scale"}, {"VRLINUX_HUD_DISTANCE", "hud_distance"},
      {"VRLINUX_HUD_SCALE", "hud_scale"},     {"VRLINUX_FLIP_Y", "flip_y"},
      {"VRLINUX_RENDER_SIZE", "render_size"}, {"VRLINUX_MIRROR", "mirror"},
      {"VRLINUX_OVERSCAN", "overscan"},       {"VRLINUX_HUD_FOLLOW", "hud_follow"},
      {"VRLINUX_LOOK_LIMIT", "look_limit"},     {"VRLINUX_DEPTH_CLAMP", "depth_clamp"},
      {"VRLINUX_SCREEN_WIDTH", "screen_width"}, {"VRLINUX_SCREEN_DISTANCE", "screen_distance"},
      {"VRLINUX_SEP", "separation"},          {"VRLINUX_CONV", "convergence"},
      {"VRLINUX_SWAP", "swap_eyes"},          {"VRLINUX_STEREO", "stereo"},
      {"VRLINUX_PRESENT_MODE", "present_mode"}, {"VRLINUX_AUTO_SWITCH", "auto_switch"},
      {"VRLINUX_FPS_LIMIT", "fps_limit"},     {"VRLINUX_MODE", "mode"},
      {"VRLINUX_HIDE_EXT", "hide_extensions"}, {"VRLINUX_PATCH", "patch_shaders"},
      {"VRLINUX_LOG_LEVEL", "log_level"},     {"VRLINUX_HOTKEYS", "hotkeys"},
  };
  for (auto& m : kMap)
    if (const char* v = getenv(m.env); v && *v) ApplyKey(m.key, v);
}

void LoadConfig() {
  static bool loaded = false;
  if (loaded) return;
  loaded = true;
  if (const char* lf = getenv("VRLINUX_LOG"); lf && *lf) g_logFile = fopen(lf, "a");
  g_game = DetectGameName();
  std::string dir = ConfigDir();
  LoadFile(dir + "/default.conf");
  if (const char* p = getenv("VRLINUX_CONFIG"); p && *p) g_profile = p;
  else g_profile = dir + "/" + g_game + ".conf";
  // built-in presets for games known to need something (a saved profile still wins)
  static const struct { const char* game; const char* keys; } kPresets[] = {
      {"azimuth.exe", "hide_extensions=0;mode=alternate"},   // Forza Horizon 3 (Azimuth): needs GPL, emulated GPU
      {"EDF6.exe", "mode=alternate"},                        // reads back from the GPU every frame
      {"EDF5.exe", "mode=alternate"},
  };
  for (auto& pr : kPresets)
    if (!strcasecmp(g_game.c_str(), pr.game)) {
      std::string k = pr.keys;
      size_t a = 0;
      while (a < k.size()) {
        size_t e = k.find(';', a);
        if (e == std::string::npos) e = k.size();
        std::string kv = k.substr(a, e - a);
        size_t eq = kv.find('=');
        if (eq != std::string::npos) ApplyKey(kv.substr(0, eq).c_str(), kv.substr(eq + 1).c_str());
        a = e + 1;
      }
      g_preset = pr.keys;
    }
  bool had = LoadFile(g_profile);
  struct stat st;
  if (stat(g_profile.c_str(), &st) == 0) g_profileMtime = st.st_mtime;
  LoadEnv();
  // launchers, overlays, Wine helpers and VR runtimes: never touch them
  static const char* kIgnore[] = {"explorer.exe", "xalia.exe", "upc.exe", "UbisoftConnect.exe", "UplayWebCore.exe",
                                  "UbisoftGameLauncher.exe", "UbisoftGameLauncher64.exe", "EpicGamesLauncher.exe",
                                  "EpicWebHelper.exe", "EOSOverlayRenderer-Win64-Shipping.exe",
                                  "EOSOverlayRenderer-Win32-Shipping.exe", "steamwebhelper.exe", "steam.exe",
                                  "EADesktop.exe", "EABackgroundService.exe", "Link2EA.exe", "RockstarService.exe",
                                  "LauncherPatcher.exe", "Launcher.exe", "steamwebhelper", "steam", "services.exe",
                                  "winedevice.exe", "plugplay.exe", "rpcss.exe", "tabtip.exe", "conhost.exe",
                                  "vrlinux-host", "wivrn-server", "wivrn-dashboard", "monado-service", "vrcompositor",
                                  "vrserver", "vrmonitor", "vrdashboard", "vrstartup", "vrwebhelper"};
  for (const char* n : kIgnore)
    if (!strcasecmp(g_game.c_str(), n)) g_cfg.passthrough = true;
  if (const char* ig = getenv("VRLINUX_IGNORE"); ig && strcasestr(ig, g_game.c_str())) g_cfg.passthrough = true;
  if (!g_cfg.passthrough && !g_logFile) {  // always keep a per-game log
    std::string ldir = dir + "/logs";
    mkdir(dir.c_str(), 0755);
    mkdir(ldir.c_str(), 0755);
    g_logFile = fopen((ldir + "/" + g_game + ".log").c_str(), "w");
  }
  if (g_cfg.passthrough) {
    g_cfg.patchShaders = false;
    g_cfg.mode = 0;
    g_cfg.hideExtensions = false;
    g_cfg.fpsLimit = -1;
    Log(1, "%s: launcher/helper, vrlinux inactive", g_game.c_str());
    return;
  }
  if (!g_preset.empty()) Log(1, "built-in preset for %s: %s", g_game.c_str(), g_preset.c_str());
  Log(1, "game=%s profile=%s%s display=%s look=%s fov=%.1f(%s) world_scale=%.3f mouse_sens=%.2f mode=%s",
      g_game.c_str(), g_profile.c_str(), had ? "" : " (new)", g_cfg.display ? "screen" : "vr",
      g_cfg.look == 2 ? "shader" : g_cfg.look ? "mouse" : "none", g_cfg.fov, g_cfg.fovAxis ? "v" : "h", g_cfg.worldScale, g_cfg.mouseSens,
      g_cfg.mode ? "replay" : "alternate");
}

void SaveConfig() {
  std::string dir = ConfigDir();
  size_t pos = 0;
  while ((pos = dir.find('/', pos + 1)) != std::string::npos) mkdir(dir.substr(0, pos).c_str(), 0755);
  mkdir(dir.c_str(), 0755);
  FILE* f = fopen(g_profile.c_str(), "w");
  if (!f) { Log(0, "cannot write %s", g_profile.c_str()); return; }
  const Config& c = g_cfg;
  fprintf(f, "# vrlinux profile for %s\n", g_game.c_str());
  fprintf(f, "display = %s\nlook = %s\nlook_pitch = %d\nmouse_sens = %.4f\npitch_limit = %.1f\ninvert_y = %d\n",
          c.display ? "screen" : "vr", c.look == 2 ? "shader" : c.look ? "mouse" : "none", c.lookPitch, c.mouseSens, c.pitchLimit, c.invertY);
  fprintf(f, "look_latency = %d\nhead_look = %d\n", c.lookLatency, c.headLook);
  fprintf(f, "positional = %d\npositional_limit = %.3f\nfov = %.2f\nfov_axis = %s\nworld_scale = %.5f\n",
          c.positional, c.posLimit, c.fov, c.fovAxis ? "v" : "h", c.worldScale);
  fprintf(f, "hud_follow = %d\nlook_limit = %.1f\n", c.hudFollow, c.lookLimit);
  fprintf(f, "flip_y = %s\noverscan = %.3f\ndepth_clamp = %d\nmirror = %d\n",
          c.flipY < 0 ? "auto" : c.flipY ? "1" : "0", c.overscan, c.depthClamp, c.mirror);
  if (c.renderW < 0) fprintf(f, "render_size = auto\n");
  else if (c.renderW == 0) fprintf(f, "render_size = off\n");
  else fprintf(f, "render_size = %dx%d\n", c.renderW, c.renderH);
  fprintf(f, "hud_distance = %.3f\nhud_scale = %.3f\nscreen_distance = %.3f\nscreen_width = %.3f\n",
          c.hudDistance, c.hudScale, c.screenDist, c.screenWidth);
  fprintf(f, "separation = %.5f\nconvergence = %.5f\nhud_depth = %.4f\nswap_eyes = %d\nstereo = %d\n", c.separation,
          c.convergence, c.hudDepth, c.swapEyes, c.stereo);
  fprintf(f, "mode = %s\nauto_switch = %d\npresent_mode = %s\n", c.mode ? "replay" : "alternate", c.autoSwitch,
          PresentModeName(c.presentMode));
  if (c.fpsLimit == 0) fprintf(f, "fps_limit = auto\n");
  else if (c.fpsLimit < 0) fprintf(f, "fps_limit = off\n");
  else fprintf(f, "fps_limit = %.1f\n", c.fpsLimit);
  fclose(f);
  struct stat st;
  if (stat(g_profile.c_str(), &st) == 0) g_profileMtime = st.st_mtime;
  Log(1, "saved %s", g_profile.c_str());
}

// --- X11 hotkeys via dlopen (no link-time dependency) ---------------------------------
namespace {
struct X11 {
  void* lib = nullptr;
  void* dpy = nullptr;
  void* (*OpenDisplay)(const char*) = nullptr;
  int (*QueryKeymap)(void*, char[32]) = nullptr;
  unsigned char (*KeysymToKeycode)(void*, unsigned long) = nullptr;
  bool tried = false;
  bool ok = false;
  unsigned char kc[24] = {};
};
X11 g_x;

enum Key { kCtrlL, kCtrlR, kShiftL, kShiftR, kF1, kF2, kF3, kF4, kF5, kF6, kF7, kF8, kF9, kF10, kF11,
           kPgUp, kPgDn, kHome, kEnd, kIns, kDel, kCount };
const unsigned long kSyms[kCount] = {0xFFE3, 0xFFE4, 0xFFE1, 0xFFE2, 0xFFBE, 0xFFBF, 0xFFC0, 0xFFC1, 0xFFC2, 0xFFC3,
                                     0xFFC4, 0xFFC5, 0xFFC6, 0xFFC7, 0xFFC8, 0xFF55, 0xFF56, 0xFF50, 0xFF57, 0xFF63, 0xFFFF};

bool InitX11() {
  if (g_x.tried) return g_x.ok;
  g_x.tried = true;
  g_x.lib = dlopen("libX11.so.6", RTLD_NOW | RTLD_LOCAL);
  if (!g_x.lib) { Log(1, "hotkeys disabled (libX11 not found)"); return false; }
  g_x.OpenDisplay = (void* (*)(const char*))dlsym(g_x.lib, "XOpenDisplay");
  g_x.QueryKeymap = (int (*)(void*, char[32]))dlsym(g_x.lib, "XQueryKeymap");
  g_x.KeysymToKeycode = (unsigned char (*)(void*, unsigned long))dlsym(g_x.lib, "XKeysymToKeycode");
  if (!g_x.OpenDisplay || !g_x.QueryKeymap || !g_x.KeysymToKeycode) return false;
  g_x.dpy = g_x.OpenDisplay(nullptr);
  if (!g_x.dpy) { Log(1, "hotkeys disabled (no X display)"); return false; }
  for (int i = 0; i < kCount; ++i) g_x.kc[i] = g_x.KeysymToKeycode(g_x.dpy, kSyms[i]);
  g_x.ok = true;
  Log(1, "hotkeys (Ctrl+Shift+): F1/F2 FOV -/+ | F3/F4 world scale (screen: separation) | F5/F6 mouse sens "
         "(screen: convergence) | F7 swap eyes | F8 3D on/off | F9 save | F10 screen/immersive | F11 head rotation on/off | "
         "Home recenter | End positional on/off | PgUp/PgDn HUD (screen: distance) | Insert HUD follows head | "
         "Delete head turn limit");
  return true;
}

bool Down(const char km[32], int k) {
  unsigned char c = g_x.kc[k];
  return c && (km[c >> 3] & (1 << (c & 7)));
}
}  // namespace

bool PollControls() {
  bool changed = false;
  using clk = std::chrono::steady_clock;
  static clk::time_point lastFileCheck = clk::now();
  static clk::time_point lastKeyPoll = clk::now();
  auto now = clk::now();

  if (now - lastFileCheck > std::chrono::seconds(1)) {
    lastFileCheck = now;
    struct stat st;
    if (stat(g_profile.c_str(), &st) == 0 && st.st_mtime != g_profileMtime) {
      g_profileMtime = st.st_mtime;
      LoadFile(g_profile);
      Log(1, "reloaded %s", g_profile.c_str());
      changed = true;
    }
  }

  if (!g_cfg.hotkeys || !InitX11()) return changed;
  float dt = std::chrono::duration<float>(now - lastKeyPoll).count();
  lastKeyPoll = now;
  if (dt > 0.1f) dt = 0.1f;

  char km[32];
  g_x.QueryKeymap(g_x.dpy, km);
  static bool prev[kCount] = {};
  bool cur[kCount];
  for (int i = 0; i < kCount; ++i) cur[i] = Down(km, i);
  bool mod = (cur[kCtrlL] || cur[kCtrlR]) && (cur[kShiftL] || cur[kShiftR]);
  auto pressed = [&](int k) { return mod && cur[k] && !prev[k]; };
  float rate = 1.0f + 1.2f * dt;   // ~ x2.2 per second while held
  float slow = 1.0f + 0.25f * dt;  // ~ x1.25 per second (FOV, sensitivity)
  Config& c = g_cfg;
  const bool screen = c.display == 1;
  if (mod) {
    if (cur[kF1]) { c.fov = std::max(20.0f, c.fov / slow); changed = true; }
    if (cur[kF2]) { c.fov = std::min(170.0f, c.fov * slow); changed = true; }
    if (cur[kF3]) { if (screen) c.separation /= rate; else c.worldScale /= rate; changed = true; }
    if (cur[kF4]) { if (screen) c.separation *= rate; else c.worldScale *= rate; changed = true; }
    if (cur[kF5]) { if (screen) c.convergence /= rate; else c.mouseSens /= slow; changed = true; }
    if (cur[kF6]) { if (screen) c.convergence *= rate; else c.mouseSens *= slow; changed = true; }
    if (screen) {
      if (cur[kPgUp]) { c.screenDist = std::max(0.5f, c.screenDist / rate); changed = true; }
      if (cur[kPgDn]) { c.screenDist = std::min(50.0f, c.screenDist * rate); changed = true; }
    } else {
      if ((cur[kPgUp] || cur[kPgDn]) && c.hudDistance <= 0.0f) c.hudDistance = 10.0f;
      if (cur[kPgUp]) { c.hudDistance = std::max(0.3f, c.hudDistance / rate); changed = true; }  // closer
      if (cur[kPgDn]) { c.hudDistance = std::min(100.0f, c.hudDistance * rate); changed = true; }
    }
  }
  if (pressed(kF7)) { c.swapEyes = !c.swapEyes; changed = true; Log(1, "swap_eyes=%d", c.swapEyes); }
  if (pressed(kF8)) { c.stereo = !c.stereo; changed = true; Log(1, "stereo=%d", c.stereo); }
  if (pressed(kF9)) SaveConfig();
  if (pressed(kF10)) { c.display = c.display ? 0 : 1; changed = true; Log(1, "display=%s", c.display ? "screen" : "vr (immersive)"); }
  if (pressed(kF11)) { c.headLook = !c.headLook; changed = true; Log(1, "head look %s", c.headLook ? "ON" : "OFF"); }
  if (pressed(kHome)) { ++g_recenter; Log(1, "recenter"); }
  if (pressed(kIns)) { c.hudFollow = !c.hudFollow; changed = true; Log(1, "HUD %s", c.hudFollow ? "follows the head" : "fixed with the game camera"); }
  if (pressed(kDel)) {  // head rotation limit: 30 45 60 90 off
    static const float kL[] = {30, 45, 60, 90, 0};
    int i = 0;
    while (i < 4 && kL[i] != c.lookLimit) ++i;
    c.lookLimit = (kL[i] == c.lookLimit) ? kL[(i + 1) % 5] : 0.0f;
    changed = true;
    if (c.lookLimit > 0) Log(1, "look_limit=%.0f degrees", c.lookLimit); else Log(1, "look_limit off");
  }
  if (pressed(kEnd)) { c.positional = !c.positional; changed = true; Log(1, "positional=%d", c.positional); }
  static bool wasAdjusting = false;
  bool adjusting = mod && (cur[kF1] || cur[kF2] || cur[kF3] || cur[kF4] || cur[kF5] || cur[kF6] || cur[kPgUp] ||
                           cur[kPgDn]);
  if (wasAdjusting && !adjusting)
    Log(1, "fov=%.1f world_scale=%.4f mouse_sens=%.3f hud_distance=%.2f | screen: sep=%.4f conv=%.3f dist=%.2f",
        c.fov, c.worldScale, c.mouseSens, c.hudDistance, c.separation, c.convergence, c.screenDist);
  wasAdjusting = adjusting;
  for (int i = 0; i < kCount; ++i) prev[i] = cur[i];
  return changed;
}

}  // namespace stereo3d
