// vrlinux-host - shows the game (rendered by the vrlinux Vulkan layer) in the headset through OpenXR.
//
//  * Works with any OpenXR runtime: WiVRn (default on most systems), SteamVR, Monado.
//    Runtime selection like any OpenXR app: active_runtime.json, or XR_RUNTIME_JSON=<file>.
//  * Receives the eye images from the game process (shared GPU memory), and shows each frame
//    with the exact pose it was rendered with: the runtime reprojects head rotation.
//  * Head tracking -> game camera through a virtual mouse (uinput) and a pose block the layer
//    reads every frame (eye positions, head movement).
//  * Gamepad: hold L3 + R3 (both sticks pressed) to recenter. The gamepad is only read,
//    never grabbed: the game keeps using it normally.
//
// Build: g++ -std=c++17 -O2 -Ithird_party/include host/host.cpp -o vrlinux-host -ldl -pthread
#define XR_USE_GRAPHICS_API_VULKAN
#include <vulkan/vulkan.h>
#include <openxr/openxr.h>
#include <openxr/openxr_platform.h>
#include <openxr/openxr_loader_negotiation.h>

#include <dirent.h>
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <linux/input.h>
#include <linux/uinput.h>
#include <poll.h>
#include <signal.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/un.h>
#include <time.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "../src/shared.h"
#include "../src/vrmath.h"

using namespace vrlinux;

namespace {

// ------------------------------------------------------------------------------------
// Logging
// ------------------------------------------------------------------------------------
FILE* g_log = nullptr;
bool g_verbose = false;
volatile sig_atomic_t g_quit = 0;

void Log(const char* fmt, ...) __attribute__((format(printf, 1, 2)));
void Log(const char* fmt, ...) {
  char buf[1024];
  va_list ap;
  va_start(ap, fmt);
  vsnprintf(buf, sizeof(buf), fmt, ap);
  va_end(ap);
  time_t t = time(nullptr);
  char ts[32];
  strftime(ts, sizeof(ts), "%H:%M:%S", localtime(&t));
  fprintf(stderr, "[vrlinux-host %s] %s\n", ts, buf);
  if (g_log) {
    fprintf(g_log, "%s %s\n", ts, buf);
    fflush(g_log);
  }
}

double Now() {
  timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + ts.tv_nsec * 1e-9;
}

std::string ConfigDir() {
  if (const char* x = getenv("XDG_CONFIG_HOME"); x && *x) return std::string(x) + "/vrlinux";
  if (const char* h = getenv("HOME"); h && *h) return std::string(h) + "/.config/vrlinux";
  return "/tmp/vrlinux";
}

// ------------------------------------------------------------------------------------
// OpenXR runtime loading (no libopenxr_loader needed; falls back to it if present)
// ------------------------------------------------------------------------------------
PFN_xrGetInstanceProcAddr g_xrGIPA = nullptr;

std::string ReadFile(const std::string& p) {
  std::string s;
  if (FILE* f = fopen(p.c_str(), "rb")) {
    char b[4096];
    size_t n;
    while ((n = fread(b, 1, sizeof(b), f)) > 0) s.append(b, n);
    fclose(f);
  }
  return s;
}

std::string JsonString(const std::string& js, const char* key) {
  std::string k = std::string("\"") + key + "\"";
  size_t p = js.find(k);
  if (p == std::string::npos) return "";
  p = js.find(':', p + k.size());
  if (p == std::string::npos) return "";
  p = js.find('"', p);
  if (p == std::string::npos) return "";
  std::string out;
  for (size_t i = p + 1; i < js.size() && js[i] != '"'; ++i) {
    if (js[i] == '\\' && i + 1 < js.size()) ++i;
    out.push_back(js[i]);
  }
  return out;
}

std::vector<std::string> RuntimeJsonCandidates() {
  std::vector<std::string> c;
  if (const char* e = getenv("XR_RUNTIME_JSON"); e && *e) c.push_back(e);
  std::string cfg;
  if (const char* x = getenv("XDG_CONFIG_HOME"); x && *x) cfg = x;
  else if (const char* h = getenv("HOME"); h && *h) cfg = std::string(h) + "/.config";
  if (!cfg.empty()) c.push_back(cfg + "/openxr/1/active_runtime.json");
  std::string dirs = "/etc/xdg";
  if (const char* x = getenv("XDG_CONFIG_DIRS"); x && *x) dirs = x;
  size_t s = 0;
  while (s <= dirs.size()) {
    size_t e = dirs.find(':', s);
    if (e == std::string::npos) e = dirs.size();
    if (e > s) c.push_back(dirs.substr(s, e - s) + "/openxr/1/active_runtime.json");
    s = e + 1;
  }
  c.push_back("/etc/openxr/1/active_runtime.json");
  return c;
}

bool LoadRuntime() {
  for (const std::string& path : RuntimeJsonCandidates()) {
    std::string js = ReadFile(path);
    if (js.empty()) continue;
    std::string lib = JsonString(js, "library_path");
    if (lib.empty()) continue;
    if (lib[0] != '/') {
      char real[4096];
      std::string base = realpath(path.c_str(), real) ? std::string(real) : path;
      lib = base.substr(0, base.find_last_of('/') + 1) + lib;
    }
    void* h = dlopen(lib.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!h) {
      Log("cannot load OpenXR runtime %s (from %s): %s", lib.c_str(), path.c_str(), dlerror());
      continue;
    }
    auto neg = (PFN_xrNegotiateLoaderRuntimeInterface)dlsym(h, "xrNegotiateLoaderRuntimeInterface");
    if (!neg) {
      Log("%s: no xrNegotiateLoaderRuntimeInterface", lib.c_str());
      continue;
    }
    XrNegotiateLoaderInfo li{XR_LOADER_INTERFACE_STRUCT_LOADER_INFO, XR_LOADER_INFO_STRUCT_VERSION,
                             sizeof(XrNegotiateLoaderInfo)};
    li.minInterfaceVersion = 1;
    li.maxInterfaceVersion = XR_CURRENT_LOADER_RUNTIME_VERSION;
    li.minApiVersion = XR_MAKE_VERSION(1, 0, 0);
    li.maxApiVersion = XR_MAKE_VERSION(1, 0x3ff, 0xfff);
    XrNegotiateRuntimeRequest rr{XR_LOADER_INTERFACE_STRUCT_RUNTIME_REQUEST, XR_RUNTIME_INFO_STRUCT_VERSION,
                                 sizeof(XrNegotiateRuntimeRequest)};
    if (XR_FAILED(neg(&li, &rr)) || !rr.getInstanceProcAddr) {
      Log("%s: runtime negotiation failed", lib.c_str());
      continue;
    }
    g_xrGIPA = rr.getInstanceProcAddr;
    Log("OpenXR runtime: %s (%s)", lib.c_str(), path.c_str());
    return true;
  }
  // system loader as a last resort
  for (const char* n : {"libopenxr_loader.so.1", "libopenxr_loader.so"}) {
    if (void* h = dlopen(n, RTLD_NOW | RTLD_LOCAL)) {
      g_xrGIPA = (PFN_xrGetInstanceProcAddr)dlsym(h, "xrGetInstanceProcAddr");
      if (g_xrGIPA) {
        Log("OpenXR: using %s", n);
        return true;
      }
    }
  }
  return false;
}

// ------------------------------------------------------------------------------------
// Function tables
// ------------------------------------------------------------------------------------
#define XR_GLOBAL(X) X(xrEnumerateInstanceExtensionProperties) X(xrCreateInstance)
#define XR_INST(X)                                                                                         \
  X(xrDestroyInstance) X(xrGetInstanceProperties) X(xrGetSystem) X(xrGetSystemProperties)                  \
  X(xrEnumerateViewConfigurationViews) X(xrGetVulkanGraphicsRequirements2KHR) X(xrCreateVulkanInstanceKHR) \
  X(xrGetVulkanGraphicsDevice2KHR) X(xrCreateVulkanDeviceKHR) X(xrCreateSession) X(xrDestroySession)       \
  X(xrCreateReferenceSpace) X(xrDestroySpace) X(xrEnumerateSwapchainFormats) X(xrCreateSwapchain)          \
  X(xrDestroySwapchain) X(xrEnumerateSwapchainImages) X(xrAcquireSwapchainImage) X(xrWaitSwapchainImage)   \
  X(xrReleaseSwapchainImage) X(xrPollEvent) X(xrBeginSession) X(xrEndSession) X(xrRequestExitSession)      \
  X(xrWaitFrame) X(xrBeginFrame) X(xrEndFrame) X(xrLocateViews) X(xrLocateSpace) X(xrResultToString)       \
  X(xrEnumerateReferenceSpaces)

struct XrFns {
#define D(n) PFN_##n n = nullptr;
  XR_GLOBAL(D)
  XR_INST(D)
#undef D
} xr;

#define VK_INST_FNS(X)                                                                                  \
  X(vkDestroyInstance) X(vkGetPhysicalDeviceQueueFamilyProperties) X(vkGetPhysicalDeviceProperties)     \
  X(vkGetPhysicalDeviceProperties2) X(vkGetPhysicalDeviceMemoryProperties) X(vkGetDeviceProcAddr)       \
  X(vkEnumerateDeviceExtensionProperties) X(vkGetPhysicalDeviceFormatProperties)
#define VK_DEV_FNS(X)                                                                                    \
  X(vkDestroyDevice) X(vkGetDeviceQueue) X(vkCreateCommandPool) X(vkDestroyCommandPool)                  \
  X(vkAllocateCommandBuffers) X(vkBeginCommandBuffer) X(vkEndCommandBuffer) X(vkResetCommandBuffer)      \
  X(vkCmdPipelineBarrier) X(vkCmdCopyImage) X(vkCmdBlitImage) X(vkCmdClearColorImage) X(vkQueueSubmit)   \
  X(vkCreateFence) X(vkDestroyFence) X(vkWaitForFences) X(vkResetFences) X(vkCreateImage)                \
  X(vkDestroyImage) X(vkGetImageMemoryRequirements) X(vkAllocateMemory) X(vkFreeMemory)                  \
  X(vkBindImageMemory) X(vkDeviceWaitIdle) X(vkGetFenceStatus)

struct VkFns {
  PFN_vkGetInstanceProcAddr vkGetInstanceProcAddr = nullptr;
#define D(n) PFN_##n n = nullptr;
  VK_INST_FNS(D)
  VK_DEV_FNS(D)
#undef D
} vk;

bool LoadVulkan() {
  void* h = dlopen("libvulkan.so.1", RTLD_NOW | RTLD_LOCAL);
  if (!h) h = dlopen("libvulkan.so", RTLD_NOW | RTLD_LOCAL);
  if (!h) return false;
  vk.vkGetInstanceProcAddr = (PFN_vkGetInstanceProcAddr)dlsym(h, "vkGetInstanceProcAddr");
  return vk.vkGetInstanceProcAddr != nullptr;
}

// ------------------------------------------------------------------------------------
// Virtual mouse (uinput) and gamepad reader
// ------------------------------------------------------------------------------------
struct Mouse {
  int fd = -1;
  bool failed = false;
  bool Open() {
    if (fd >= 0) return true;
    if (failed) return false;
    fd = open("/dev/uinput", O_WRONLY | O_NONBLOCK | O_CLOEXEC);
    if (fd < 0) {
      failed = true;
      Log("WARNING: cannot open /dev/uinput (%s): head tracking cannot turn the game camera.", strerror(errno));
      Log("         Fix: sudo install.sh --uinput (adds a udev rule), or install the 'steam-devices' package.");
      return false;
    }
    ioctl(fd, UI_SET_EVBIT, EV_KEY);
    ioctl(fd, UI_SET_KEYBIT, BTN_LEFT);
    ioctl(fd, UI_SET_KEYBIT, BTN_RIGHT);
    ioctl(fd, UI_SET_EVBIT, EV_REL);
    ioctl(fd, UI_SET_RELBIT, REL_X);
    ioctl(fd, UI_SET_RELBIT, REL_Y);
    uinput_setup us{};
    us.id.bustype = BUS_VIRTUAL;
    us.id.vendor = 0x1d6b;
    us.id.product = 0x5652;
    snprintf(us.name, sizeof(us.name), "vrlinux head tracking mouse");
    if (ioctl(fd, UI_DEV_SETUP, &us) < 0 || ioctl(fd, UI_DEV_CREATE) < 0) {
      Log("WARNING: uinput device creation failed (%s)", strerror(errno));
      close(fd);
      fd = -1;
      failed = true;
      return false;
    }
    Log("virtual mouse created (head tracking -> game camera)");
    return true;
  }
  // false if the virtual mouse is not available (the game camera did not move)
  bool Move(int dx, int dy) {
    if (!Open()) return false;
    if (!dx && !dy) return true;
    input_event ev[3]{};
    int n = 0;
    if (dx) { ev[n].type = EV_REL; ev[n].code = REL_X; ev[n].value = dx; ++n; }
    if (dy) { ev[n].type = EV_REL; ev[n].code = REL_Y; ev[n].value = dy; ++n; }
    ev[n].type = EV_SYN; ev[n].code = SYN_REPORT; ++n;
    ssize_t w = write(fd, ev, sizeof(input_event) * n);
    return w > 0;
  }
  ~Mouse() {
    if (fd >= 0) {
      ioctl(fd, UI_DEV_DESTROY);
      close(fd);
    }
  }
} g_mouse;

struct Gamepads {
  struct Pad { int fd; std::string path; bool l3 = false, r3 = false; };
  std::vector<Pad> pads;            // used by the frame loop only
  std::mutex newLock;
  std::vector<Pad> found;           // filled by the scanner thread
  std::vector<std::string> known;   // paths already opened (scanner thread)
  std::thread scanner;
  double comboStart = -1;
  bool comboFired = false;

  static bool TestBit(const unsigned long* bits, int b) {
    return (bits[b / (8 * sizeof(long))] >> (b % (8 * sizeof(long)))) & 1;
  }
  // Opening /dev/input devices can be slow: never do it in the headset frame loop.
  void Start() {
    scanner = std::thread([this]() {
      while (!g_quit) {
        Scan();
        for (int i = 0; i < 50 && !g_quit; ++i) usleep(100000);
      }
    });
  }
  void Stop() {
    if (scanner.joinable()) scanner.join();
  }
  void Scan() {
    DIR* d = opendir("/dev/input");
    if (!d) return;
    while (dirent* e = readdir(d)) {
      if (strncmp(e->d_name, "event", 5)) continue;
      std::string p = std::string("/dev/input/") + e->d_name;
      if (std::find(known.begin(), known.end(), p) != known.end()) continue;
      int fd = open(p.c_str(), O_RDONLY | O_NONBLOCK | O_CLOEXEC);
      if (fd < 0) continue;
      known.push_back(p);
      unsigned long keys[(KEY_MAX + 1) / (8 * sizeof(long)) + 1] = {};
      ioctl(fd, EVIOCGBIT(EV_KEY, sizeof(keys)), keys);
      if (TestBit(keys, BTN_THUMBL) && TestBit(keys, BTN_THUMBR) && TestBit(keys, BTN_SOUTH)) {
        char name[128] = "?";
        ioctl(fd, EVIOCGNAME(sizeof(name)), name);
        Log("gamepad: %s (%s) - hold L3+R3 to recenter", name, p.c_str());
        std::lock_guard<std::mutex> l(newLock);
        found.push_back({fd, p});
      } else {
        close(fd);
      }
    }
    closedir(d);
  }
  void Forget(const std::string& path) {
    std::lock_guard<std::mutex> l(newLock);
    forget.push_back(path);
  }
  std::vector<std::string> forget;
  // returns true when L3+R3 have been held for 0.6 s
  bool Poll() {
    {
      std::lock_guard<std::mutex> l(newLock);
      for (auto& p : found) pads.push_back(p);
      found.clear();
      for (auto& f : forget) known.erase(std::remove(known.begin(), known.end(), f), known.end());
      forget.clear();
    }
    for (size_t i = 0; i < pads.size();) {
      input_event ev[32];
      ssize_t r;
      bool dead = false;
      while ((r = read(pads[i].fd, ev, sizeof(ev))) > 0) {
        for (size_t k = 0; k < size_t(r) / sizeof(input_event); ++k) {
          if (ev[k].type != EV_KEY) continue;
          if (ev[k].code == BTN_THUMBL) pads[i].l3 = ev[k].value != 0;
          if (ev[k].code == BTN_THUMBR) pads[i].r3 = ev[k].value != 0;
        }
      }
      if (r < 0 && errno != EAGAIN) dead = true;
      if (dead) {
        close(pads[i].fd);
        std::lock_guard<std::mutex> l(newLock);
        known.erase(std::remove(known.begin(), known.end(), pads[i].path), known.end());
        pads.erase(pads.begin() + long(i));
      } else {
        ++i;
      }
    }
    bool held = false;
    for (auto& p : pads) held |= p.l3 && p.r3;
    if (!held) {
      comboStart = -1;
      comboFired = false;
      return false;
    }
    if (comboStart < 0) comboStart = Now();
    if (!comboFired && Now() - comboStart > 0.6) {
      comboFired = true;
      return true;
    }
    return false;
  }
} g_pads;

// ------------------------------------------------------------------------------------
// Connection with the game (layer)
// ------------------------------------------------------------------------------------
struct Client {
  int fd = -1;
  int memfd = -1;
  SharedState* shm = nullptr;
  int pid = 0;
  std::string game;
};

struct Imported {
  bool valid = false;
  uint32_t w = 0, h = 0;
  VkFormat format = VK_FORMAT_UNDEFINED;
  VkImage image[kSlots] = {};
  VkDeviceMemory mem[kSlots] = {};
};

// ------------------------------------------------------------------------------------
// The host
// ------------------------------------------------------------------------------------
struct Host {
  // ipc
  int listenFd = -1;
  Client client;
  Imported imp;
  uint32_t lastRecenterReq = 0;
  bool recenterPending = true;

  // xr
  XrInstance instance = XR_NULL_HANDLE;
  XrSystemId systemId = XR_NULL_SYSTEM_ID;
  XrSession session = XR_NULL_HANDLE;
  XrSessionState state = XR_SESSION_STATE_UNKNOWN;
  bool running = false;
  XrSpace local = XR_NULL_HANDLE, view = XR_NULL_HANDLE;
  XrSwapchain swapchain = XR_NULL_HANDLE;
  std::vector<VkImage> swImages;
  VkFormat swFormat = VK_FORMAT_UNDEFINED;
  uint32_t swW = 0, swH = 0;
  double lastInitTry = -100;
  std::string runtimeName;

  // vulkan
  VkInstance vkInst = VK_NULL_HANDLE;
  VkPhysicalDevice phys = VK_NULL_HANDLE;
  VkDevice dev = VK_NULL_HANDLE;
  uint32_t qfam = 0;
  VkQueue queue = VK_NULL_HANDLE;
  VkCommandPool pool = VK_NULL_HANDLE;
  VkCommandBuffer cmd = VK_NULL_HANDLE;
  VkFence fence = VK_NULL_HANDLE;
  uint8_t devUUID[16] = {}, drvUUID[16] = {};
  VkPhysicalDeviceMemoryProperties memProps{};

  // pose state
  float refYaw = 0;
  V3 refPos{};
  float sentYaw = 0, sentPitch = 0;  // radians, camera rotation relative to the reference
  float remX = 0, remY = 0;          // fractional mouse counts
  Q headQ{};
  V3 headP{};
  V3 eyeHead[2]{{-0.032f, 0, 0}, {0.032f, 0, 0}};
  Q eyeHeadQ[2]{};
  XrFovf eyeFov[2]{{-0.9f, 0.9f, 0.9f, -0.9f}, {-0.9f, 0.9f, 0.9f, -0.9f}};
  V3 eyeWorld[2]{};
  bool havePose = false;

  // frames
  bool haveFrame = false;
  SlotMeta shown{};
  uint64_t lastFrame = 0;
  uint64_t statShown = 0, statXr = 0;
  double statStart = Now();
  double lastVerbose = 0;

  // ---------------------------------------------------------------- ipc
  bool Listen() {
    listenFd = socket(AF_UNIX, SOCK_SEQPACKET | SOCK_CLOEXEC | SOCK_NONBLOCK, 0);
    sockaddr_un a{};
    a.sun_family = AF_UNIX;
    char name[64];
    int n = snprintf(name, sizeof(name), "vrlinux-%u", unsigned(getuid()));
    memcpy(a.sun_path + 1, name, size_t(n));
    socklen_t len = socklen_t(offsetof(sockaddr_un, sun_path) + 1 + n);
    if (bind(listenFd, (sockaddr*)&a, len) != 0) {
      Log("another vrlinux-host is already running (%s)", strerror(errno));
      return false;
    }
    listen(listenFd, 4);
    return true;
  }

  void DropClient(const char* why) {
    if (client.fd < 0) return;
    Log("game disconnected (%s): %s", client.game.c_str(), why);
    close(client.fd);
    if (client.shm) munmap(client.shm, sizeof(SharedState));
    if (client.memfd >= 0) close(client.memfd);
    client = Client{};
    FreeImported();
    haveFrame = false;
  }

  void Accept() {
    int fd = accept4(listenFd, nullptr, nullptr, SOCK_CLOEXEC | SOCK_NONBLOCK);
    if (fd < 0) return;
    if (client.fd >= 0) DropClient("a new game connected");
    int memfd = int(syscall(SYS_memfd_create, "vrlinux-state", 1u /*MFD_CLOEXEC*/));
    if (memfd < 0 || ftruncate(memfd, sizeof(SharedState)) != 0) {
      close(fd);
      if (memfd >= 0) close(memfd);
      return;
    }
    void* m = mmap(nullptr, sizeof(SharedState), PROT_READ | PROT_WRITE, MAP_SHARED, memfd, 0);
    if (m == MAP_FAILED) {
      close(fd);
      close(memfd);
      return;
    }
    memset(m, 0, sizeof(SharedState));
    auto* s = (SharedState*)m;
    s->magic = kMagic;
    s->version = kVersion;
    s->settings.look = 2;
    s->settings.headLook = 1;
    s->settings.lookPitch = 1;
    s->settings.mouseSens = 20;
    s->settings.pitchLimit = 80;
    s->settings.positional = 1;
    s->settings.posLimit = 0.5f;
    s->settings.screenDist = 3;
    s->settings.screenWidth = 5.5f;
    client.fd = fd;
    client.memfd = memfd;
    client.shm = s;
    PublishStatic();
    WritePose();
    // send the memfd
    char data[4] = {'V', 'R', 'L', 'X'};
    iovec iov{data, sizeof(data)};
    char ctrl[CMSG_SPACE(sizeof(int))] = {};
    msghdr msg{};
    msg.msg_iov = &iov;
    msg.msg_iovlen = 1;
    msg.msg_control = ctrl;
    msg.msg_controllen = sizeof(ctrl);
    cmsghdr* c = CMSG_FIRSTHDR(&msg);
    c->cmsg_level = SOL_SOCKET;
    c->cmsg_type = SCM_RIGHTS;
    c->cmsg_len = CMSG_LEN(sizeof(int));
    memcpy(CMSG_DATA(c), &memfd, sizeof(int));
    if (sendmsg(fd, &msg, MSG_NOSIGNAL) < 0) {
      DropClient("send failed");
      return;
    }
    lastRecenterReq = 0;
    recenterPending = true;
    Log("game connected");
  }

  void ReadClient() {
    for (;;) {
      MsgHello h;
      iovec iov{&h, sizeof(h)};
      char ctrl[CMSG_SPACE(sizeof(int) * kSlots)] = {};
      msghdr msg{};
      msg.msg_iov = &iov;
      msg.msg_iovlen = 1;
      msg.msg_control = ctrl;
      msg.msg_controllen = sizeof(ctrl);
      ssize_t r = recvmsg(client.fd, &msg, MSG_DONTWAIT | MSG_CMSG_CLOEXEC);
      if (r == 0) {
        DropClient("closed");
        return;
      }
      if (r < 0) {
        if (errno != EAGAIN && errno != EWOULDBLOCK) DropClient(strerror(errno));
        return;
      }
      std::vector<int> fds;
      for (cmsghdr* c = CMSG_FIRSTHDR(&msg); c; c = CMSG_NXTHDR(&msg, c))
        if (c->cmsg_level == SOL_SOCKET && c->cmsg_type == SCM_RIGHTS) {
          size_t n = (c->cmsg_len - CMSG_LEN(0)) / sizeof(int);
          for (size_t i = 0; i < n; ++i) {
            int fd;
            memcpy(&fd, CMSG_DATA(c) + i * sizeof(int), sizeof(int));
            fds.push_back(fd);
          }
        }
      if (r >= ssize_t(sizeof(uint32_t)) && h.type == MsgHelloType && r == ssize_t(sizeof(h))) {
        client.pid = h.pid;
        client.game.assign(h.game, strnlen(h.game, sizeof(h.game)));
        Import(h, fds);
      } else {
        for (int fd : fds) close(fd);
      }
    }
  }

  void FreeImported() {
    FinishPending(true);
    pending = false;
    if (dev) {
      if (imp.valid) vk.vkDeviceWaitIdle(dev);
      for (int i = 0; i < kSlots; ++i) {
        if (imp.image[i]) vk.vkDestroyImage(dev, imp.image[i], nullptr);
        if (imp.mem[i]) vk.vkFreeMemory(dev, imp.mem[i], nullptr);
      }
    }
    imp = Imported{};
  }

  uint32_t MemType(uint32_t bits) {
    for (uint32_t i = 0; i < memProps.memoryTypeCount; ++i)
      if ((bits & (1u << i)) && (memProps.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) return i;
    for (uint32_t i = 0; i < memProps.memoryTypeCount; ++i)
      if (bits & (1u << i)) return i;
    return 0;
  }

  void Import(const MsgHello& h, std::vector<int>& fds) {
    FreeImported();
    haveFrame = false;
    if (!dev) {  // no headset yet: we import when the game re-sends (it does on every reconnection)
      Log("game '%s' %ux%u per eye: waiting for the headset before importing", client.game.c_str(), h.width, h.height);
      for (int fd : fds) close(fd);
      pendingHello = h;
      havePendingHello = false;  // the fds are gone; ask the game to reconnect when the headset is ready
      needReconnect = true;
      return;
    }
    if (memcmp(h.deviceUUID, devUUID, 16) || memcmp(h.driverUUID, drvUUID, 16)) {
      Log("ERROR: the game renders on a different GPU/driver than the VR runtime uses. Use the same GPU for both "
          "(e.g. DRI_PRIME / __NV_PRIME_RENDER_OFFLOAD) and, with Flatpak Steam on Mesa, the same Mesa version.");
      for (int fd : fds) close(fd);
      return;
    }
    if (fds.size() != size_t(kSlots)) {
      Log("bad hello (%zu fds)", fds.size());
      for (int fd : fds) close(fd);
      return;
    }
    bool ok = true;
    for (int i = 0; i < kSlots; ++i) {
      VkExternalMemoryImageCreateInfo emi{VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO};
      emi.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
      VkImageCreateInfo ici{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, &emi};
      ici.flags = h.imageFlags;
      ici.imageType = VK_IMAGE_TYPE_2D;
      ici.format = VkFormat(h.format);
      ici.extent = {h.width * 2, h.height, 1};
      ici.mipLevels = 1;
      ici.arrayLayers = 1;
      ici.samples = VK_SAMPLE_COUNT_1_BIT;
      ici.tiling = VK_IMAGE_TILING_OPTIMAL;
      ici.usage = h.imageUsage;
      ici.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
      ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
      if (!ok || vk.vkCreateImage(dev, &ici, nullptr, &imp.image[i]) != VK_SUCCESS) {
        ok = false;
        close(fds[i]);
        continue;
      }
      VkMemoryRequirements req;
      vk.vkGetImageMemoryRequirements(dev, imp.image[i], &req);
      VkMemoryDedicatedAllocateInfo ded{VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO};
      ded.image = imp.image[i];
      VkImportMemoryFdInfoKHR ifd{VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR, &ded};
      ifd.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
      ifd.fd = fds[i];
      VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, &ifd};
      ai.allocationSize = h.allocSize[i];
      ai.memoryTypeIndex = MemType(req.memoryTypeBits);
      if (vk.vkAllocateMemory(dev, &ai, nullptr, &imp.mem[i]) != VK_SUCCESS) {  // fd still ours on failure
        close(fds[i]);
        ok = false;
        continue;
      }
      if (vk.vkBindImageMemory(dev, imp.image[i], imp.mem[i], 0) != VK_SUCCESS) ok = false;
    }
    if (!ok) {
      Log("ERROR: cannot import the game's eye images");
      FreeImported();
      return;
    }
    imp.valid = true;
    imp.w = h.width;
    imp.h = h.height;
    imp.format = VkFormat(h.format);
    Log("game '%s' (pid %d): %ux%u per eye", client.game.c_str(), h.pid, h.width, h.height);
    if (!CreateSwapchain(h.width * 2, h.height)) {
      FreeImported();
      return;
    }
    recenterPending = true;
  }
  MsgHello pendingHello{};
  bool havePendingHello = false;
  bool needReconnect = false;

  void PublishStatic() {
    if (!client.shm) return;
    client.shm->sessionRunning.store(running ? 1u : 0u);
    client.shm->displayHzX100.store(uint32_t(displayHz * 100));
    client.shm->recW = recW;
    client.shm->recH = recH;
  }

  // ---------------------------------------------------------------- xr init
  float displayHz = 90.0f;
  uint32_t recW = 0, recH = 0;

  bool XrCheck(XrResult r, const char* what) {
    if (XR_SUCCEEDED(r)) return true;
    char s[XR_MAX_RESULT_STRING_SIZE] = "?";
    if (instance && xr.xrResultToString) xr.xrResultToString(instance, r, s);
    Log("%s failed: %s (%d)", what, s, int(r));
    return false;
  }

  bool InitXr() {
    lastInitTry = Now();
    if (!g_xrGIPA && !LoadRuntime()) {
      Log("no OpenXR runtime found. Start WiVRn (or SteamVR) and make it the active runtime, or set XR_RUNTIME_JSON.");
      return false;
    }
#define G(n) g_xrGIPA(XR_NULL_HANDLE, #n, (PFN_xrVoidFunction*)&xr.n);
    XR_GLOBAL(G)
#undef G
    if (!xr.xrCreateInstance || !xr.xrEnumerateInstanceExtensionProperties) return false;
    uint32_t n = 0;
    xr.xrEnumerateInstanceExtensionProperties(nullptr, 0, &n, nullptr);
    std::vector<XrExtensionProperties> props(n, {XR_TYPE_EXTENSION_PROPERTIES});
    xr.xrEnumerateInstanceExtensionProperties(nullptr, n, &n, props.data());
    bool hasVk2 = false;
    for (auto& p : props) hasVk2 |= !strcmp(p.extensionName, XR_KHR_VULKAN_ENABLE2_EXTENSION_NAME);
    if (!hasVk2) {
      Log("the OpenXR runtime does not support XR_KHR_vulkan_enable2");
      return false;
    }
    const char* exts[] = {XR_KHR_VULKAN_ENABLE2_EXTENSION_NAME};
    XrInstanceCreateInfo ci{XR_TYPE_INSTANCE_CREATE_INFO};
    snprintf(ci.applicationInfo.applicationName, XR_MAX_APPLICATION_NAME_SIZE, "vrlinux");
    snprintf(ci.applicationInfo.engineName, XR_MAX_ENGINE_NAME_SIZE, "vrlinux");
    ci.applicationInfo.applicationVersion = 1;
    ci.applicationInfo.apiVersion = XR_MAKE_VERSION(1, 0, 0);
    ci.enabledExtensionCount = 1;
    ci.enabledExtensionNames = exts;
    XrResult r = xr.xrCreateInstance(&ci, &instance);
    if (XR_FAILED(r)) {
      Log("xrCreateInstance failed (%d): is WiVRn/SteamVR running?", int(r));
      instance = XR_NULL_HANDLE;
      return false;
    }
#define I(n) g_xrGIPA(instance, #n, (PFN_xrVoidFunction*)&xr.n);
    XR_INST(I)
#undef I
    XrInstanceProperties ip{XR_TYPE_INSTANCE_PROPERTIES};
    xr.xrGetInstanceProperties(instance, &ip);
    runtimeName = ip.runtimeName;
    XrSystemGetInfo sgi{XR_TYPE_SYSTEM_GET_INFO};
    sgi.formFactor = XR_FORM_FACTOR_HEAD_MOUNTED_DISPLAY;
    r = xr.xrGetSystem(instance, &sgi, &systemId);
    if (XR_FAILED(r)) {
      Log("%s: no headset yet (%d), retrying...", runtimeName.c_str(), int(r));
      ShutdownXr();
      return false;
    }
    XrSystemProperties sp{XR_TYPE_SYSTEM_PROPERTIES};
    xr.xrGetSystemProperties(instance, systemId, &sp);
    uint32_t vc = 0;
    xr.xrEnumerateViewConfigurationViews(instance, systemId, XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO, 0, &vc, nullptr);
    std::vector<XrViewConfigurationView> vcv(vc, {XR_TYPE_VIEW_CONFIGURATION_VIEW});
    xr.xrEnumerateViewConfigurationViews(instance, systemId, XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO, vc, &vc,
                                         vcv.data());
    if (vc) {
      recW = vcv[0].recommendedImageRectWidth;
      recH = vcv[0].recommendedImageRectHeight;
    }
    Log("runtime %s, headset '%s', recommended %ux%u per eye", runtimeName.c_str(), sp.systemName, recW, recH);
    if (!InitVulkan()) {
      ShutdownXr();
      return false;
    }
    XrGraphicsBindingVulkan2KHR gb{XR_TYPE_GRAPHICS_BINDING_VULKAN2_KHR};
    gb.instance = vkInst;
    gb.physicalDevice = phys;
    gb.device = dev;
    gb.queueFamilyIndex = qfam;
    gb.queueIndex = 0;
    XrSessionCreateInfo sci{XR_TYPE_SESSION_CREATE_INFO, &gb};
    sci.systemId = systemId;
    if (!XrCheck(xr.xrCreateSession(instance, &sci, &session), "xrCreateSession")) {
      ShutdownXr();
      return false;
    }
    XrReferenceSpaceCreateInfo rs{XR_TYPE_REFERENCE_SPACE_CREATE_INFO};
    rs.poseInReferenceSpace.orientation.w = 1;
    rs.referenceSpaceType = XR_REFERENCE_SPACE_TYPE_LOCAL;
    if (!XrCheck(xr.xrCreateReferenceSpace(session, &rs, &local), "local space")) {
      ShutdownXr();
      return false;
    }
    rs.referenceSpaceType = XR_REFERENCE_SPACE_TYPE_VIEW;
    if (!XrCheck(xr.xrCreateReferenceSpace(session, &rs, &view), "view space")) {
      ShutdownXr();
      return false;
    }
    Log("OpenXR session created, waiting for the headset to start it");
    if (client.fd >= 0) DropClient("headset (re)started: the game reconnects with fresh images");
    return true;
  }

  bool InitVulkan() {
    if (!vk.vkGetInstanceProcAddr && !LoadVulkan()) {
      Log("cannot load libvulkan.so.1");
      return false;
    }
    XrGraphicsRequirementsVulkan2KHR req{XR_TYPE_GRAPHICS_REQUIREMENTS_VULKAN2_KHR};
    if (!XrCheck(xr.xrGetVulkanGraphicsRequirements2KHR(instance, systemId, &req), "vulkan requirements")) return false;
    VkApplicationInfo ai{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    ai.pApplicationName = "vrlinux-host";
    ai.apiVersion = VK_API_VERSION_1_1;
    uint32_t maxV = VK_MAKE_API_VERSION(0, XR_VERSION_MAJOR(req.maxApiVersionSupported),
                                        XR_VERSION_MINOR(req.maxApiVersionSupported), 0);
    uint32_t minV = VK_MAKE_API_VERSION(0, XR_VERSION_MAJOR(req.minApiVersionSupported),
                                        XR_VERSION_MINOR(req.minApiVersionSupported), 0);
    if (ai.apiVersion < minV) ai.apiVersion = minV;
    if (maxV && ai.apiVersion > maxV && maxV >= VK_API_VERSION_1_1) ai.apiVersion = maxV;
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &ai;
    XrVulkanInstanceCreateInfoKHR xci{XR_TYPE_VULKAN_INSTANCE_CREATE_INFO_KHR};
    xci.systemId = systemId;
    xci.pfnGetInstanceProcAddr = vk.vkGetInstanceProcAddr;
    xci.vulkanCreateInfo = &ici;
    VkResult vr = VK_SUCCESS;
    if (!XrCheck(xr.xrCreateVulkanInstanceKHR(instance, &xci, &vkInst, &vr), "xrCreateVulkanInstanceKHR") ||
        vr != VK_SUCCESS) {
      Log("vkCreateInstance via OpenXR: %d", vr);
      return false;
    }
#define L(n) vk.n = (PFN_##n)vk.vkGetInstanceProcAddr(vkInst, #n);
    VK_INST_FNS(L)
#undef L
    XrVulkanGraphicsDeviceGetInfoKHR gi{XR_TYPE_VULKAN_GRAPHICS_DEVICE_GET_INFO_KHR};
    gi.systemId = systemId;
    gi.vulkanInstance = vkInst;
    if (!XrCheck(xr.xrGetVulkanGraphicsDevice2KHR(instance, &gi, &phys), "xrGetVulkanGraphicsDevice2KHR")) return false;
    VkPhysicalDeviceIDProperties idp{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES};
    VkPhysicalDeviceProperties2 p2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &idp};
    vk.vkGetPhysicalDeviceProperties2(phys, &p2);
    memcpy(devUUID, idp.deviceUUID, 16);
    memcpy(drvUUID, idp.driverUUID, 16);
    vk.vkGetPhysicalDeviceMemoryProperties(phys, &memProps);
    Log("GPU: %s", p2.properties.deviceName);
    uint32_t nq = 0;
    vk.vkGetPhysicalDeviceQueueFamilyProperties(phys, &nq, nullptr);
    std::vector<VkQueueFamilyProperties> qf(nq);
    vk.vkGetPhysicalDeviceQueueFamilyProperties(phys, &nq, qf.data());
    qfam = UINT32_MAX;
    for (uint32_t i = 0; i < nq; ++i)
      if (qf[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) { qfam = i; break; }
    if (qfam == UINT32_MAX) return false;
    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qci.queueFamilyIndex = qfam;
    qci.queueCount = 1;
    qci.pQueuePriorities = &prio;
    const char* dexts[] = {"VK_KHR_external_memory_fd", "VK_KHR_external_memory"};
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    dci.enabledExtensionCount = 2;
    dci.ppEnabledExtensionNames = dexts;
    XrVulkanDeviceCreateInfoKHR xdi{XR_TYPE_VULKAN_DEVICE_CREATE_INFO_KHR};
    xdi.systemId = systemId;
    xdi.pfnGetInstanceProcAddr = vk.vkGetInstanceProcAddr;
    xdi.vulkanPhysicalDevice = phys;
    xdi.vulkanCreateInfo = &dci;
    if (!XrCheck(xr.xrCreateVulkanDeviceKHR(instance, &xdi, &dev, &vr), "xrCreateVulkanDeviceKHR") || vr != VK_SUCCESS) {
      Log("vkCreateDevice via OpenXR: %d", vr);
      dev = VK_NULL_HANDLE;
      return false;
    }
#define L(n) vk.n = (PFN_##n)vk.vkGetDeviceProcAddr(dev, #n);
    VK_DEV_FNS(L)
#undef L
    vk.vkGetDeviceQueue(dev, qfam, 0, &queue);
    VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pci.queueFamilyIndex = qfam;
    vk.vkCreateCommandPool(dev, &pci, nullptr, &pool);
    VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cai.commandPool = pool;
    cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cai.commandBufferCount = 1;
    vk.vkAllocateCommandBuffers(dev, &cai, &cmd);
    VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    vk.vkCreateFence(dev, &fci, nullptr, &fence);
    return true;
  }

  void DestroySwapchain() {
    if (swapchain) xr.xrDestroySwapchain(swapchain);
    swapchain = XR_NULL_HANDLE;
    swImages.clear();
    swW = swH = 0;
  }

  bool CreateSwapchain(uint32_t w, uint32_t h) {
    if (swapchain && swW == w && swH == h) return true;
    if (dev) vk.vkDeviceWaitIdle(dev);
    DestroySwapchain();
    uint32_t n = 0;
    xr.xrEnumerateSwapchainFormats(session, 0, &n, nullptr);
    std::vector<int64_t> fmts(n);
    xr.xrEnumerateSwapchainFormats(session, n, &n, fmts.data());
    // same channel order in sRGB: raw copy of the (sRGB-encoded) game bytes
    VkFormat want = imp.format == VK_FORMAT_B8G8R8A8_UNORM ? VK_FORMAT_B8G8R8A8_SRGB : VK_FORMAT_R8G8B8A8_SRGB;
    VkFormat other = want == VK_FORMAT_B8G8R8A8_SRGB ? VK_FORMAT_R8G8B8A8_SRGB : VK_FORMAT_B8G8R8A8_SRGB;
    swFormat = VK_FORMAT_UNDEFINED;
    for (int64_t f : fmts)
      if (f == want) swFormat = want;
    if (!swFormat)
      for (int64_t f : fmts)
        if (f == other) swFormat = other;
    if (!swFormat && n) swFormat = VkFormat(fmts[0]);
    XrSwapchainCreateInfo sci{XR_TYPE_SWAPCHAIN_CREATE_INFO};
    sci.usageFlags = XR_SWAPCHAIN_USAGE_TRANSFER_DST_BIT | XR_SWAPCHAIN_USAGE_COLOR_ATTACHMENT_BIT;
    sci.format = swFormat;
    sci.sampleCount = 1;
    sci.width = w;
    sci.height = h;
    sci.faceCount = 1;
    sci.arraySize = 1;
    sci.mipCount = 1;
    if (!XrCheck(xr.xrCreateSwapchain(session, &sci, &swapchain), "xrCreateSwapchain")) {
      swapchain = XR_NULL_HANDLE;
      return false;
    }
    uint32_t ni = 0;
    xr.xrEnumerateSwapchainImages(swapchain, 0, &ni, nullptr);
    std::vector<XrSwapchainImageVulkan2KHR> imgs(ni, {XR_TYPE_SWAPCHAIN_IMAGE_VULKAN2_KHR});
    xr.xrEnumerateSwapchainImages(swapchain, ni, &ni, (XrSwapchainImageBaseHeader*)imgs.data());
    for (auto& i : imgs) swImages.push_back(i.image);
    swW = w;
    swH = h;
    Log("headset swapchain %ux%u format %d (%u images)%s", w, h, swFormat, ni,
        (swFormat == want) ? "" : " (channel order differs: blit)");
    return true;
  }

  void ShutdownXr() {
    if (dev) vk.vkDeviceWaitIdle(dev);
    FreeImported();
    DestroySwapchain();
    if (local) xr.xrDestroySpace(local);
    if (view) xr.xrDestroySpace(view);
    local = view = XR_NULL_HANDLE;
    if (session) xr.xrDestroySession(session);
    session = XR_NULL_HANDLE;
    if (dev) {
      if (fence) vk.vkDestroyFence(dev, fence, nullptr);
      if (pool) vk.vkDestroyCommandPool(dev, pool, nullptr);
      vk.vkDestroyDevice(dev, nullptr);
    }
    fence = VK_NULL_HANDLE;
    pool = VK_NULL_HANDLE;
    cmd = VK_NULL_HANDLE;
    dev = VK_NULL_HANDLE;
    if (vkInst) vk.vkDestroyInstance(vkInst, nullptr);
    vkInst = VK_NULL_HANDLE;
    if (instance) xr.xrDestroyInstance(instance);
    instance = XR_NULL_HANDLE;
    running = false;
    state = XR_SESSION_STATE_UNKNOWN;
    haveFrame = false;
    PublishStatic();
  }

  // ---------------------------------------------------------------- events
  void PollEvents() {
    XrEventDataBuffer ev{XR_TYPE_EVENT_DATA_BUFFER};
    while (instance && xr.xrPollEvent(instance, &ev) == XR_SUCCESS) {
      if (ev.type == XR_TYPE_EVENT_DATA_SESSION_STATE_CHANGED) {
        auto* e = (XrEventDataSessionStateChanged*)&ev;
        state = e->state;
        Log("session state %d", int(state));
        if (state == XR_SESSION_STATE_READY) {
          XrSessionBeginInfo bi{XR_TYPE_SESSION_BEGIN_INFO};
          bi.primaryViewConfigurationType = XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO;
          if (XrCheck(xr.xrBeginSession(session, &bi), "xrBeginSession")) {
            running = true;
            recenterPending = true;
            PublishStatic();
            Log("headset is showing vrlinux");
          }
        } else if (state == XR_SESSION_STATE_STOPPING) {
          xr.xrEndSession(session);
          running = false;
          PublishStatic();
        } else if (state == XR_SESSION_STATE_EXITING || state == XR_SESSION_STATE_LOSS_PENDING) {
          Log("session ended by the runtime: will reconnect");
          ShutdownXr();
          return;
        }
      } else if (ev.type == XR_TYPE_EVENT_DATA_INSTANCE_LOSS_PENDING) {
        Log("runtime going away: will reconnect");
        ShutdownXr();
        return;
      }
      ev = {XR_TYPE_EVENT_DATA_BUFFER};
    }
  }

  // ---------------------------------------------------------------- pose
  void Recenter() {
    float yaw, pitch;
    YawPitch(headQ, yaw, pitch);
    refYaw = yaw;
    refPos = headP;
    sentYaw = sentPitch = 0;
    remX = remY = 0;
    if (client.shm) client.shm->recenterCount.fetch_add(1);
    Log("recentered (look straight ahead, game camera level)");
  }

  void UpdatePose(XrTime t) {
    XrSpaceLocation loc{XR_TYPE_SPACE_LOCATION};
    if (XR_SUCCEEDED(xr.xrLocateSpace(view, local, t, &loc)) &&
        (loc.locationFlags & XR_SPACE_LOCATION_ORIENTATION_VALID_BIT)) {
      headQ = Norm(Q{loc.pose.orientation.x, loc.pose.orientation.y, loc.pose.orientation.z, loc.pose.orientation.w});
      if (loc.locationFlags & XR_SPACE_LOCATION_POSITION_VALID_BIT)
        headP = {loc.pose.position.x, loc.pose.position.y, loc.pose.position.z};
      havePose = true;
    }
    XrViewLocateInfo vli{XR_TYPE_VIEW_LOCATE_INFO};
    vli.viewConfigurationType = XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO;
    vli.displayTime = t;
    vli.space = view;  // eye offsets relative to the head
    XrViewState vs{XR_TYPE_VIEW_STATE};
    XrView views[2] = {{XR_TYPE_VIEW}, {XR_TYPE_VIEW}};
    uint32_t nv = 0;
    if (XR_SUCCEEDED(xr.xrLocateViews(session, &vli, &vs, 2, &nv, views)) && nv == 2 &&
        (vs.viewStateFlags & XR_VIEW_STATE_POSITION_VALID_BIT)) {
      for (int i = 0; i < 2; ++i) {
        V3 p{views[i].pose.position.x, views[i].pose.position.y, views[i].pose.position.z};
        if (Len(p) < 0.2f) eyeHead[i] = p;  // sanity
        const XrQuaternionf& o = views[i].pose.orientation;
        if (vs.viewStateFlags & XR_VIEW_STATE_ORIENTATION_VALID_BIT) eyeHeadQ[i] = Norm(Q{o.x, o.y, o.z, o.w});
        if (views[i].fov.angleRight > views[i].fov.angleLeft && views[i].fov.angleUp > views[i].fov.angleDown)
          eyeFov[i] = views[i].fov;
      }
    }
  }

  void UpdateCamera() {
    SharedState* s = client.shm;
    if (!s) return;
    const LayerSettings st = s->settings;
    if (st.recenter != lastRecenterReq) {
      lastRecenterReq = st.recenter;
      recenterPending = true;
    }
    if (g_pads.Poll()) recenterPending = true;
    if (recenterPending && havePose) {
      recenterPending = false;
      Recenter();
    }
    const Q qRef = Yaw(refYaw);
    const Q headRel = Mul(Conj(qRef), headQ);
    float hy, hp;
    YawPitch(headRel, hy, hp);
    const bool mouse = st.display == 0 && st.look == 1 && st.headLook && imp.valid && running &&
                       (state == XR_SESSION_STATE_FOCUSED || state == XR_SESSION_STATE_VISIBLE);
    if (mouse && st.mouseSens > 0.0f) {
      const float deg = 180.0f / 3.14159265f, lim = std::clamp(st.pitchLimit, 0.0f, 89.0f) / deg;
      float targetPitch = st.lookPitch ? std::clamp(hp, -lim, lim) : 0.0f;
      float dYaw = WrapPi(hy - sentYaw) * deg;
      float dPitch = (targetPitch - sentPitch) * deg;
      float cx = -dYaw * st.mouseSens + remX;
      float cy = -dPitch * st.mouseSens * (st.invertY ? -1.0f : 1.0f) + remY;
      int ix = int(std::clamp(cx, -4000.0f, 4000.0f));
      int iy = int(std::clamp(cy, -4000.0f, 4000.0f));
      if (g_mouse.Move(ix, iy)) {  // only what really reached the game counts
        remX = std::clamp(cx - float(ix), -1.0f, 1.0f);
        remY = std::clamp(cy - float(iy), -1.0f, 1.0f);
        sentYaw = WrapPi(sentYaw - float(ix) / st.mouseSens / deg);
        sentPitch -= float(iy) / st.mouseSens / deg * (st.invertY ? -1.0f : 1.0f);
      }
    } else if (st.look != 1 || st.display == 1) {
      sentYaw = sentPitch = 0;
    }
    camQ = Mul(qRef, Mul(Yaw(sentYaw), Pitch(sentPitch)));
  }
  Q camQ{};

  void WritePose() {
    SharedState* s = client.shm;
    if (!s) return;
    const LayerSettings st = s->settings;
    HostPose p{};
    p.camQ[0] = camQ.x; p.camQ[1] = camQ.y; p.camQ[2] = camQ.z; p.camQ[3] = camQ.w;
    p.headQ[0] = headQ.x; p.headQ[1] = headQ.y; p.headQ[2] = headQ.z; p.headQ[3] = headQ.w;
    p.headPos[0] = headP.x; p.headPos[1] = headP.y; p.headPos[2] = headP.z;
    Q rel = Norm(Mul(Conj(camQ), headQ));
    p.headRelQ[0] = rel.x; p.headRelQ[1] = rel.y; p.headRelQ[2] = rel.z; p.headRelQ[3] = rel.w;
    for (int i = 0; i < 2; ++i) {
      p.eyeHeadQ[i][0] = eyeHeadQ[i].x; p.eyeHeadQ[i][1] = eyeHeadQ[i].y;
      p.eyeHeadQ[i][2] = eyeHeadQ[i].z; p.eyeHeadQ[i][3] = eyeHeadQ[i].w;
      p.eyeFov[i][0] = eyeFov[i].angleLeft; p.eyeFov[i][1] = eyeFov[i].angleRight;
      p.eyeFov[i][2] = eyeFov[i].angleUp; p.eyeFov[i][3] = eyeFov[i].angleDown;
    }
    V3 d = headP - refPos;
    float L = Len(d);
    if (L > st.posLimit && L > 1e-6f) d = d * (st.posLimit / L);
    for (int i = 0; i < 2; ++i) {
      V3 eo = Rot(headQ, eyeHead[i]);         // eye offset in LOCAL axes
      V3 world = headP + eo;
      eyeWorld[i] = world;
      V3 rel = st.positional ? d + eo : eo;  // relative to the recentred origin
      V3 c = Rot(Conj(camQ), rel);
      p.eyePos[i][0] = world.x; p.eyePos[i][1] = world.y; p.eyePos[i][2] = world.z;
      p.eyeCam[i][0] = c.x; p.eyeCam[i][1] = c.y; p.eyeCam[i][2] = c.z;
      p.eyeHead[i][0] = eyeHead[i].x; p.eyeHead[i][1] = eyeHead[i].y; p.eyeHead[i][2] = eyeHead[i].z;
    }
    s->poseSeq.fetch_add(1, std::memory_order_acq_rel);
    std::atomic_thread_fence(std::memory_order_release);
    memcpy((void*)&s->pose, &p, sizeof(p));
    std::atomic_thread_fence(std::memory_order_release);
    s->poseSeq.fetch_add(1, std::memory_order_acq_rel);
  }

  // ---------------------------------------------------------------- frames
  // Claims the newest game frame and copies it into the headset swapchain.
  // The copy of the previous frame: finished? (wait = block until it is)
  bool pending = false;
  uint64_t pendingFrame = 0;
  void FinishPending(bool wait) {
    if (!pending || !dev) return;
    if (wait) vk.vkWaitForFences(dev, 1, &fence, VK_TRUE, 1000000000ull);
    else if (vk.vkGetFenceStatus(dev, fence) != VK_SUCCESS) return;
    pending = false;
    if (client.shm) client.shm->done.store(pendingFrame);
  }

  void TakeFrame() {
    SharedState* s = client.shm;
    if (!s || !imp.valid || !swapchain) return;
    FinishPending(false);
    uint64_t l = s->latest.load();
    if (!l || FrameOf(l) == lastFrame) return;
    FinishPending(true);  // never hold two of the game's slots at once
    for (int i = 0; i < 8; ++i) {  // claim it (the layer never rewrites the slot we claim)
      s->inuse.store(l);
      uint64_t l2 = s->latest.load();
      if (l2 == l) break;
      l = l2;
    }
    const uint32_t slot = SlotOf(l);
    const uint64_t frame = FrameOf(l);
    if (slot >= uint32_t(kSlots)) return;
    SlotMeta meta;
    memcpy(&meta, (const void*)&s->meta[slot], sizeof(meta));
    if (meta.frame != frame) {  // metadata not for this frame: skip
      s->done.store(frame);
      return;
    }
    uint32_t idx = 0;
    XrSwapchainImageAcquireInfo ai{XR_TYPE_SWAPCHAIN_IMAGE_ACQUIRE_INFO};
    if (XR_FAILED(xr.xrAcquireSwapchainImage(swapchain, &ai, &idx)) || idx >= swImages.size()) {
      s->done.store(frame);
      return;
    }
    XrSwapchainImageWaitInfo wi{XR_TYPE_SWAPCHAIN_IMAGE_WAIT_INFO};
    wi.timeout = 100000000;  // 100 ms
    if (XR_FAILED(xr.xrWaitSwapchainImage(swapchain, &wi))) {
      s->done.store(frame);
      XrSwapchainImageReleaseInfo ri{XR_TYPE_SWAPCHAIN_IMAGE_RELEASE_INFO};
      xr.xrReleaseSwapchainImage(swapchain, &ri);
      return;
    }
    VkImage dst = swImages[idx], src = imp.image[slot];
    vk.vkResetCommandBuffer(cmd, 0);
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vk.vkBeginCommandBuffer(cmd, &bi);
    VkImageMemoryBarrier b[2] = {{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER}, {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER}};
    // acquire the game's image from the external queue family
    b[0].srcAccessMask = 0;
    b[0].dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    b[0].oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    b[0].newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    b[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_EXTERNAL;
    b[0].dstQueueFamilyIndex = qfam;
    b[0].image = src;
    b[0].subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    b[1].srcAccessMask = 0;
    b[1].dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    b[1].oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    b[1].newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    b[1].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b[1].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b[1].image = dst;
    b[1].subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vk.vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0,
                            nullptr, 2, b);
    const bool raw = (swFormat == VK_FORMAT_B8G8R8A8_SRGB && imp.format == VK_FORMAT_B8G8R8A8_UNORM) ||
                     (swFormat == VK_FORMAT_R8G8B8A8_SRGB && imp.format == VK_FORMAT_R8G8B8A8_UNORM) ||
                     swFormat == imp.format;
    if (raw) {
      VkImageCopy c{};
      c.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
      c.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
      c.extent = {imp.w * 2, imp.h, 1};
      vk.vkCmdCopyImage(cmd, src, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, dst, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &c);
    } else {
      VkImageBlit bl{};
      bl.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
      bl.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
      bl.srcOffsets[1] = {int32_t(imp.w * 2), int32_t(imp.h), 1};
      bl.dstOffsets[1] = {int32_t(swW), int32_t(swH), 1};
      vk.vkCmdBlitImage(cmd, src, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, dst, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1,
                        &bl, VK_FILTER_NEAREST);
    }
    // give the game's image back, and the swapchain image to the compositor
    b[0].srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    b[0].dstAccessMask = 0;
    b[0].oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    b[0].newLayout = VK_IMAGE_LAYOUT_GENERAL;
    b[0].srcQueueFamilyIndex = qfam;
    b[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_EXTERNAL;
    b[1].srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    b[1].dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_READ_BIT;
    b[1].oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    b[1].newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    vk.vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0,
                            nullptr, 2, b);
    vk.vkEndCommandBuffer(cmd);
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd;
    vk.vkResetFences(dev, 1, &fence);
    if (vk.vkQueueSubmit(queue, 1, &si, fence) == VK_SUCCESS) {
      pending = true;  // completion checked next frame: no CPU wait in the headset loop
      pendingFrame = frame;
    } else {
      s->done.store(frame);
    }
    XrSwapchainImageReleaseInfo ri{XR_TYPE_SWAPCHAIN_IMAGE_RELEASE_INFO};
    xr.xrReleaseSwapchainImage(swapchain, &ri);
    shown = meta;
    lastFrame = frame;
    haveFrame = true;
    ++statShown;
  }

  void Frame() {
    XrFrameWaitInfo fwi{XR_TYPE_FRAME_WAIT_INFO};
    XrFrameState fs{XR_TYPE_FRAME_STATE};
    if (!XrCheck(xr.xrWaitFrame(session, &fwi, &fs), "xrWaitFrame")) {
      usleep(10000);
      return;
    }
    if (fs.predictedDisplayPeriod > 0) {
      float hz = float(1e9 / double(fs.predictedDisplayPeriod));
      if (std::fabs(hz - displayHz) > 0.5f) {
        displayHz = hz;
        PublishStatic();
        Log("headset refresh %.1f Hz", hz);
      }
    }
    XrFrameBeginInfo fbi{XR_TYPE_FRAME_BEGIN_INFO};
    xr.xrBeginFrame(session, &fbi);
    FinishPending(false);
    UpdatePose(fs.predictedDisplayTime);
    UpdateCamera();
    WritePose();
    if (fs.shouldRender) TakeFrame();
    if (client.fd < 0) haveFrame = false;

    XrCompositionLayerProjection proj{XR_TYPE_COMPOSITION_LAYER_PROJECTION};
    XrCompositionLayerProjectionView pv[2] = {{XR_TYPE_COMPOSITION_LAYER_PROJECTION_VIEW},
                                              {XR_TYPE_COMPOSITION_LAYER_PROJECTION_VIEW}};
    XrCompositionLayerQuad quad[2] = {{XR_TYPE_COMPOSITION_LAYER_QUAD}, {XR_TYPE_COMPOSITION_LAYER_QUAD}};
    const XrCompositionLayerBaseHeader* layers[2];
    uint32_t nl = 0;
    const bool screen = client.shm && client.shm->settings.display == 1;
    if (fs.shouldRender && haveFrame && swapchain) {
      for (int i = 0; i < 2; ++i) {
        XrSwapchainSubImage sub{};
        sub.swapchain = swapchain;
        sub.imageRect.offset = {int32_t(i * imp.w), 0};
        sub.imageRect.extent = {int32_t(imp.w), int32_t(imp.h)};
        sub.imageArrayIndex = 0;
        if (!screen) {
          pv[i].pose.orientation = {shown.camQ[i][0], shown.camQ[i][1], shown.camQ[i][2], shown.camQ[i][3]};
          pv[i].pose.position = {shown.eyePos[i][0], shown.eyePos[i][1], shown.eyePos[i][2]};
          pv[i].fov = {shown.fov[i][0], shown.fov[i][1], shown.fov[i][2], shown.fov[i][3]};
          pv[i].subImage = sub;
        } else {
          const LayerSettings& st = client.shm->settings;
          Q q = Yaw(refYaw);
          V3 c = refPos + Rot(q, V3{0, 0, -std::max(0.5f, st.screenDist)});
          quad[i].space = local;
          quad[i].eyeVisibility = i == 0 ? XR_EYE_VISIBILITY_LEFT : XR_EYE_VISIBILITY_RIGHT;
          quad[i].subImage = sub;
          quad[i].pose.orientation = {q.x, q.y, q.z, q.w};
          quad[i].pose.position = {c.x, c.y, c.z};
          float wdt = std::max(0.3f, st.screenWidth);
          quad[i].size = {wdt, wdt * float(imp.h) / float(std::max(1u, imp.w))};
          layers[nl++] = (const XrCompositionLayerBaseHeader*)&quad[i];
        }
      }
      if (!screen) {
        proj.space = local;
        proj.viewCount = 2;
        proj.views = pv;
        layers[nl++] = (const XrCompositionLayerBaseHeader*)&proj;
      }
    }
    XrFrameEndInfo fei{XR_TYPE_FRAME_END_INFO};
    fei.displayTime = fs.predictedDisplayTime;
    fei.environmentBlendMode = XR_ENVIRONMENT_BLEND_MODE_OPAQUE;
    fei.layerCount = nl;
    fei.layers = layers;
    XrCheck(xr.xrEndFrame(session, &fei), "xrEndFrame");
    ++statXr;
    double now = Now();
    if (g_verbose && now - lastVerbose > 2.0) {
      lastVerbose = now;
      float y, p;
      YawPitch(headQ, y, p);
      Log("head yaw %.1f pitch %.1f pos %.3f %.3f %.3f | camera yaw %.1f pitch %.1f | eyes %.4f %.4f", y * 57.2958f,
          p * 57.2958f, headP.x, headP.y, headP.z, sentYaw * 57.2958f, sentPitch * 57.2958f, eyeHead[0].x,
          eyeHead[1].x);
    }
    if (now - statStart >= 10.0) {
      if (client.fd >= 0)
        Log("headset %.1f fps, new game frames %.1f fps (%s)", statXr / (now - statStart), statShown / (now - statStart),
            client.game.c_str());
      statXr = statShown = 0;
      statStart = now;
    }
  }

  // ---------------------------------------------------------------- main loop
  void Run() {
    while (!g_quit) {
      // ipc
      pollfd p{listenFd, POLLIN, 0};
      if (poll(&p, 1, 0) > 0) Accept();
      if (client.fd >= 0) ReadClient();
      if (needReconnect && dev && client.fd >= 0) {  // the game re-sends its images on reconnection
        needReconnect = false;
        DropClient("re-sync with the headset");
      }
      if (!instance) {
        if (Now() - lastInitTry > 3.0) InitXr();
        if (!instance) {
          if (client.shm) PublishStatic();
          usleep(50000);
          continue;
        }
      }
      PollEvents();
      if (!instance) continue;
      if (running) {
        Frame();
      } else {
        // keep the game informed about the head even if the session is idle
        usleep(20000);
      }
    }
    if (session && running) {
      xr.xrRequestExitSession(session);
      // drain a few events so the runtime ends us cleanly
      for (int i = 0; i < 50 && running; ++i) {
        PollEvents();
        if (running) Frame();
      }
    }
    if (client.fd >= 0) DropClient("vrlinux-host exiting");
    ShutdownXr();
  }
};

void OnSignal(int) { g_quit = 1; }

}  // namespace

int main(int argc, char** argv) {
  for (int i = 1; i < argc; ++i) {
    if (!strcmp(argv[i], "-v") || !strcmp(argv[i], "--verbose")) g_verbose = true;
    else if (!strcmp(argv[i], "--ping")) {  // exit 0 if a host is already running
      int fd = socket(AF_UNIX, SOCK_SEQPACKET | SOCK_CLOEXEC, 0);
      sockaddr_un a{};
      a.sun_family = AF_UNIX;
      char name[64];
      int n = snprintf(name, sizeof(name), "vrlinux-%u", unsigned(getuid()));
      memcpy(a.sun_path + 1, name, size_t(n));
      int r = connect(fd, (sockaddr*)&a, socklen_t(offsetof(sockaddr_un, sun_path) + 1 + n));
      close(fd);
      return r == 0 ? 0 : 1;
    } else if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) {
      printf("vrlinux-host: shows vrlinux games in the headset (OpenXR: WiVRn, SteamVR, Monado).\n"
             "  XR_RUNTIME_JSON=<file>  use a specific OpenXR runtime\n"
             "  --ping                  exit 0 if a host is already running\n");
      return 0;
    }
  }
  std::string dir = ConfigDir();
  mkdir(dir.c_str(), 0755);
  mkdir((dir + "/logs").c_str(), 0755);
  g_log = fopen((dir + "/logs/vrlinux-host.log").c_str(), "w");
  signal(SIGINT, OnSignal);
  signal(SIGTERM, OnSignal);
  signal(SIGHUP, OnSignal);
  signal(SIGPIPE, SIG_IGN);
  static Host host;
  if (!host.Listen()) return 1;
  g_pads.Start();
  Log("waiting for games (socket @vrlinux-%u) and for the headset", unsigned(getuid()));
  host.Run();
  g_pads.Stop();
  Log("bye");
  return 0;
}
