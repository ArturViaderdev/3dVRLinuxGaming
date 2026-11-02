// vrlinux - Vulkan layer: normal (flat) Linux games in VR, with head tracking.
// Based on stereo3d (real geometric stereo 3D layer).
//
// How it works
//  * Every vertex / tess-eval / geometry shader is patched so that its clip-space position
//    goes through a per-eye transform read from a tiny GPU buffer (see spirv_patch.h):
//    each eye is rendered from its real position (IPD from the headset) plus the movement
//    of your head since the last recenter (6DOF parallax).
//  * replay mode: the game renders every frame once (left eye); at present the layer copies
//    the image, switches the buffer to the right eye and submits the same command buffers
//    again -> both eyes of the same instant at the game's frame rate.
//  * Both eyes are copied into an exported GPU image (OPAQUE_FD) shared with vrlinux-host,
//    which shows them in the headset through OpenXR (WiVRn / SteamVR / Monado) with the pose
//    they were rendered with, so the runtime reprojects head rotation without latency.
//  * Head rotation turns the game camera through a virtual mouse driven by vrlinux-host.
#include <vulkan/vk_layer.h>
#include <vulkan/vulkan.h>

#include <algorithm>
#include <dirent.h>
#include <dlfcn.h>
#include <signal.h>
#include <pthread.h>
#include <unistd.h>
#include <cstdio>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <functional>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "config.h"
#include "shared.h"
#include "spirv_patch.h"
#include "vrlink.h"
#include "vrmath.h"

#define LAYER_NAME "VK_LAYER_VRLINUX_vr"
#define EXPORT extern "C" __attribute__((visibility("default")))

using namespace stereo3d;
using vrlinux::Link;
using Clock = std::chrono::steady_clock;

namespace {

// ------------------------------------------------------------------------------------
// Dispatch tables
// ------------------------------------------------------------------------------------
#define INST_FUNCS(X)                                                                     \
  X(DestroyInstance) X(EnumeratePhysicalDevices) X(EnumeratePhysicalDeviceGroups)         \
  X(EnumeratePhysicalDeviceGroupsKHR) X(EnumerateDeviceExtensionProperties)               \
  X(GetPhysicalDeviceSurfaceCapabilitiesKHR) X(GetPhysicalDeviceSurfaceCapabilities2KHR)  \
  X(GetPhysicalDeviceSurfacePresentModesKHR) X(GetPhysicalDeviceMemoryProperties)         \
  X(GetPhysicalDeviceQueueFamilyProperties) X(GetPhysicalDeviceFeatures2)                 \
  X(GetPhysicalDeviceFeatures2KHR) X(GetPhysicalDeviceProperties)                         \
  X(GetPhysicalDeviceFormatProperties) X(DestroySurfaceKHR) X(GetPhysicalDeviceSurfaceSupportKHR)          \
  X(GetPhysicalDeviceSurfaceFormatsKHR) X(GetPhysicalDeviceProperties2) X(GetPhysicalDeviceProperties2KHR)

#define DEV_FUNCS(X)                                                                        \
  X(DestroyDevice) X(GetDeviceQueue) X(GetDeviceQueue2) X(CreateShaderModule)               \
  X(DestroyShaderModule) X(CreateGraphicsPipelines) X(QueueSubmit) X(QueueSubmit2)          \
  X(QueueSubmit2KHR) X(QueueWaitIdle) X(QueueBindSparse) X(DeviceWaitIdle)                  \
  X(CreateSwapchainKHR) X(DestroySwapchainKHR) X(GetSwapchainImagesKHR)                     \
  X(AcquireNextImageKHR) X(AcquireNextImage2KHR) X(QueuePresentKHR) X(CreateImage)          \
  X(DestroyImage) X(GetImageMemoryRequirements) X(AllocateMemory) X(FreeMemory)             \
  X(BindImageMemory) X(CreateBuffer) X(DestroyBuffer) X(GetBufferMemoryRequirements)        \
  X(BindBufferMemory) X(MapMemory) X(UnmapMemory) X(GetBufferDeviceAddress)                 \
  X(GetBufferDeviceAddressKHR) X(CreateCommandPool) X(DestroyCommandPool)                   \
  X(AllocateCommandBuffers) X(FreeCommandBuffers) X(BeginCommandBuffer) X(EndCommandBuffer) \
  X(ResetCommandBuffer) X(CmdPipelineBarrier) X(CmdCopyImage) X(CmdBlitImage)               \
  X(CmdUpdateBuffer) X(CmdClearColorImage) X(CmdCopyImageToBuffer) X(CreateFence) X(DestroyFence) X(WaitForFences)  \
  X(ResetFences) X(CreateSemaphore) X(DestroySemaphore) X(WaitSemaphores) X(WaitSemaphoresKHR) X(GetMemoryFdKHR) X(GetFenceStatus) X(ResetCommandPool) X(DestroyImageView) X(DestroyBufferView) \
  X(DestroySampler) X(DestroyPipeline) X(DestroyFramebuffer) X(DestroyQueryPool) X(DestroyDescriptorPool) \
  X(ResetDescriptorPool) X(FreeDescriptorSets) X(DestroyEvent) \
  X(CmdSetViewport) X(CmdSetViewportWithCount) X(CmdSetViewportWithCountEXT) \
  X(CmdBeginRenderPass) X(CmdBeginRenderPass2) X(CmdBeginRenderPass2KHR) X(CmdBeginRendering) \
  X(CmdBeginRenderingKHR) X(CmdCopyBuffer) X(GetSemaphoreCounterValue) X(GetSemaphoreCounterValueKHR)

struct InstDispatch {
#define DECL(n) PFN_vk##n n = nullptr;
  INST_FUNCS(DECL)
};
struct DevDispatch {
  PFN_vkGetDeviceProcAddr GetDeviceProcAddr = nullptr;
  DEV_FUNCS(DECL)
#undef DECL
};

inline void* Key(const void* dispatchable) { return *(void* const*)dispatchable; }

struct InstanceData {
  VkInstance instance = VK_NULL_HANDLE;
  PFN_vkGetInstanceProcAddr gipa = nullptr;
  InstDispatch d;
  uint32_t apiVersion = VK_API_VERSION_1_0;
  bool translation = false;  // DXVK / VKD3D-Proton: the game picks its own resolution
};

struct QueueInfo {
  uint32_t family = UINT32_MAX;
  std::mutex m;
};

struct VImage {
  VkImage image = VK_NULL_HANDLE;
  VkDeviceMemory mem = VK_NULL_HANDLE;
  VkFence fence = VK_NULL_HANDLE;
  VkCommandBuffer cmd = VK_NULL_HANDLE;
  VkCommandBuffer cmd2 = VK_NULL_HANDLE;  // replay mode: right-eye copy
  VkSemaphore acqSem = VK_NULL_HANDLE;
  VkSemaphore doneSem = VK_NULL_HANDLE;
};

// Per-eye shader parameters + what the host needs to show that frame.
struct EyeParams {
  float m[stereo3d::kParamFloats];
};
struct FrameView {
  EyeParams p[2];
  float camQ[2][4];
  float eyePos[2][3];
  float fov[4];
  float efov[2][4] = {};   // per-eye FOV when it differs (shader look: headset FOV)
  bool perEyeFov = false;
  bool vr = false;  // rendered with headset data
};

struct VrSlot {
  VkImage image = VK_NULL_HANDLE;
  VkDeviceMemory mem = VK_NULL_HANDLE;
  VkDeviceSize size = 0;
  VkFence fence = VK_NULL_HANDLE;  // signalled when the copy into this slot is finished
  int state = 0;                   // 0 free, 1 writing, 2 waiting for the GPU (publisher), 3 published
  uint64_t frame = 0;
  vrlinux::SlotMeta meta{};
};

// Frames exported to vrlinux-host for one swapchain
struct VrExport {
  uint32_t w = 0, h = 0;  // one eye
  VkFormat format = VK_FORMAT_UNDEFINED;
  bool rawCopy = true;    // vkCmdCopyImage (same size class) or blit
  VrSlot slots[vrlinux::kSlots];
  uint32_t helloGen = 0;  // link generation the hello was sent for
  int writing = -1;       // alternate mode: slot being filled (left eye done)
  uint64_t frameNo = 0;
  uint32_t nextSlot = 0;
  VkImageUsageFlags usage = 0;
};

struct Swapchain {
  VkSwapchainKHR handle = VK_NULL_HANDLE;  // == real swapchain handle
  VkFormat format = VK_FORMAT_UNDEFINED;
  VkExtent2D vext{}, rext{};
  std::vector<VkImage> realImages;
  std::vector<VImage> v;
  VkCommandPool pool = VK_NULL_HANDLE;
  uint32_t poolFamily = UINT32_MAX;
  bool canBlit = true;
  bool canBlitSelf = true;
  uint32_t nextAcquire = 0;
  Clock::time_point statStart = Clock::now();
  uint64_t statFrames = 0;
  uint64_t statVr = 0;
  VrExport* ex = nullptr;
  bool exportFailed = false;
};

struct ModuleEntry {
  std::vector<uint32_t> code;
  struct Variant { uint32_t model; std::string entry; VkShaderModule patched; };
  std::vector<Variant> variants;
};

struct DeviceData {
  VkDevice device = VK_NULL_HANDLE;
  VkPhysicalDevice phys = VK_NULL_HANDLE;
  InstanceData* inst = nullptr;
  DevDispatch d;
  PFN_vkSetDeviceLoaderData setLoaderData = nullptr;
  uint32_t gfxFamily = UINT32_MAX;
  uint32_t apiVersion = VK_API_VERSION_1_0;
  VkPhysicalDeviceMemoryProperties memProps{};

  std::mutex queueLock;
  std::unordered_map<VkQueue, std::unique_ptr<QueueInfo>> queues;
  VkQueue gfxQueue = VK_NULL_HANDLE;

  bool canExport = false;          // VK_KHR_external_memory_fd enabled
  uint8_t deviceUUID[16] = {}, driverUUID[16] = {};
  FrameView cur{};                 // view of the frame being rendered
  std::atomic<uint32_t> vpNeg{0}, vpPos{0};  // big viewports with negative / positive height
  float camHist[8][4] = {};        // camera orientations of the last frames (look_latency)
  uint32_t camHistPos = 0;
  // publisher: waits for the GPU copy into a slot, then hands it to the host
  std::thread publisher;
  std::mutex pubLock;
  std::condition_variable pubCv;
  std::vector<std::pair<VrExport*, int>> pubQueue;
  bool pubStop = false;
  std::mutex exportLock;           // protects VrExport slot states (present thread vs publisher)
  VrExport* pubCurrent = nullptr;  // job the publisher is working on (under pubLock)
  VrExport* activeEx = nullptr;    // export whose images the host currently has

  bool stereoShaders = false;
  VkBuffer paramBuf = VK_NULL_HANDLE;
  VkDeviceMemory paramMem = VK_NULL_HANDLE;
  VkDeviceAddress paramAddr = 0;

  std::mutex shaderLock;
  std::unordered_map<VkShaderModule, ModuleEntry> modules;
  std::atomic<uint32_t> patchedCount{0}, skippedCount{0}, pipelineCount{0};

  std::mutex swapLock;
  std::unordered_map<VkSwapchainKHR, Swapchain*> swapchains;
  uint64_t frame = 0;  // presents to virtual swapchains; even = left eye
  Clock::time_point lastPresent{};  // for the frame limiter

  // ---- replay mode: the frame's command buffers are submitted a second time for the right eye
  std::atomic<bool> replay{false};
  std::atomic<const char*> stage{"init"};     // what the layer is doing (for the stall watchdog)
  std::thread watchdog;
  std::atomic<bool> stopWatchdog{false};
  int badReplayWindows = 0;
  std::mutex recLock;                         // held by submits on the present queue and by present
  std::vector<VkCommandBuffer> recCbs;        // this frame's command buffers, in order
  std::unordered_map<VkSemaphore, uint64_t> heldTimeline;  // signals delayed until after the replay
  std::vector<VkFence> heldFences;
  bool recMono = false;                       // replay impossible this frame (app needed a signal early)
  std::mutex semLock;
  std::unordered_set<VkSemaphore> timelineSems;
  std::atomic<uint32_t> monoFrames{0}, replayFrames{0};
  std::atomic<uint64_t> presentCount{0};
  std::atomic<int> swapCount{0};
  std::atomic<uint64_t> lastAbortPresent{0};
  // ---- relaxed replay: the app may get a frame's signals before the right eye is replayed
  // (games that read GPU results mid-frame, vkd3d's present thread). Command buffers and
  // resources of that frame must then survive until the replay is done.
  std::mutex cbLock;                          // order: recLock -> cbLock
  std::condition_variable cbCv;
  std::unordered_map<VkCommandBuffer, VkCommandPool> cbPool;
  std::unordered_map<VkCommandPool, std::unordered_set<VkCommandBuffer>> poolCbs;
  std::unordered_set<VkCommandBuffer> recSet;  // recCbs of the frame being recorded
  std::unordered_map<VkCommandBuffer, uint64_t> cbSerial;  // frame serial whose replay uses the cb
  uint64_t frameSerial = 1;                   // frame being recorded
  uint64_t submittedSerial = 0;               // last frame whose replay was submitted (or dropped)
  uint64_t relaxedSerial = 0;                 // last frame whose signals were released early
  bool dropFrame = false;                     // a recorded cb was reset before present: no replay
  VkFence serialFence[8] = {};
  uint64_t serialOf[8] = {};
  std::atomic<uint32_t> relaxedFrames{0};
  struct Deferred { uint64_t serial; std::function<void()> fn; };
  std::vector<Deferred> deferred;             // destroys waiting for a relaxed replay
  // ---- off-screen passes (shadow maps, reflection cubes...): drawn without the eye transform.
  // paramBuf: [0] params the shaders read, [256] this eye's params, [512] identity
  std::atomic<uint32_t> gameW{0}, gameH{0};   // size the game renders its frame at
  std::mutex passLock;
  std::unordered_set<VkCommandBuffer> monoCbs;  // command buffers left with identity params active
  std::atomic<uint32_t> monoPasses{0};
  // ---- games polling (timeout 0 / vkGetFenceStatus) for a signal the replay holds
  std::mutex pollLock;
  uint64_t pollPresent = UINT64_MAX;
  Clock::time_point pollStart{};
};

std::mutex g_lock;
std::unordered_map<void*, InstanceData*> g_instByKey;
std::unordered_map<VkPhysicalDevice, InstanceData*> g_instByPhys;
std::unordered_map<void*, DeviceData*> g_devByKey;

InstanceData* Inst(const void* h) {
  std::lock_guard<std::mutex> l(g_lock);
  auto it = g_instByKey.find(Key(h));
  return it == g_instByKey.end() ? nullptr : it->second;
}
InstanceData* InstFromPhys(VkPhysicalDevice p) {
  std::lock_guard<std::mutex> l(g_lock);
  auto it = g_instByPhys.find(p);
  if (it != g_instByPhys.end()) return it->second;
  auto it2 = g_instByKey.find(Key(p));
  if (it2 != g_instByKey.end()) return it2->second;
  if (g_instByKey.size() == 1) return g_instByKey.begin()->second;
  return nullptr;
}
DeviceData* Dev(const void* h) {
  std::lock_guard<std::mutex> l(g_lock);
  auto it = g_devByKey.find(Key(h));
  return it == g_devByKey.end() ? nullptr : it->second;
}
// lock-free lookup for hot per-command hooks (games normally have one device)
std::atomic<void*> g_fastKey{nullptr};
std::atomic<DeviceData*> g_fastDev{nullptr};
DeviceData* DevFast(const void* h) {
  void* k = Key(h);
  if (g_fastKey.load(std::memory_order_acquire) == k) {
    DeviceData* d = g_fastDev.load(std::memory_order_acquire);
    if (d && g_fastKey.load(std::memory_order_acquire) == k) return d;
  }
  DeviceData* d = Dev(h);
  g_fastKey.store(nullptr);
  g_fastDev.store(d);
  g_fastKey.store(k);
  return d;
}

QueueInfo* GetQueueInfo(DeviceData* dd, VkQueue q) {
  std::lock_guard<std::mutex> l(dd->queueLock);
  auto& p = dd->queues[q];
  if (!p) p.reset(new QueueInfo());
  return p.get();
}

// Presentation extensions can never work with the virtual swapchain (the left-eye frame is
// never presented on its own, present IDs / waits would never complete): always hidden.
const char* const kAlwaysHidden[] = {
    "VK_KHR_present_wait",           "VK_KHR_present_wait2",          "VK_KHR_present_id",
    "VK_KHR_present_id2",            "VK_EXT_swapchain_maintenance1", "VK_KHR_swapchain_maintenance1",
    "VK_NV_low_latency2",            "VK_AMD_anti_lag",               "VK_EXT_full_screen_exclusive",
    "VK_GOOGLE_display_timing",      "VK_EXT_present_timing",
    // pipelines built from cached shader identifiers or shader objects carry no SPIR-V to patch
    "VK_EXT_shader_module_identifier", "VK_EXT_shader_object",
};
// Hidden by default (hide_extensions = 0 shows it): library pipelines are patched too, but less tested
const char* const kShaderPathHidden[] = {
    "VK_EXT_graphics_pipeline_library",
};
bool IsHidden(const char* name) {
  for (const char* h : kAlwaysHidden)
    if (!strcmp(h, name)) return true;
  if (Cfg().hideExtensions)
    for (const char* h : kShaderPathHidden)
      if (!strcmp(h, name)) return true;
  return false;
}

uint32_t FindMemType(DeviceData* dd, uint32_t bits, VkMemoryPropertyFlags want,
                     VkMemoryPropertyFlags prefer = 0) {
  for (int pass = 0; pass < 2; ++pass) {
    VkMemoryPropertyFlags f = want | (pass == 0 ? prefer : 0);
    for (uint32_t i = 0; i < dd->memProps.memoryTypeCount; ++i)
      if ((bits & (1u << i)) && (dd->memProps.memoryTypes[i].propertyFlags & f) == f) return i;
  }
  for (uint32_t i = 0; i < dd->memProps.memoryTypeCount; ++i)
    if (bits & (1u << i)) return i;
  return UINT32_MAX;
}

// ------------------------------------------------------------------------------------
// Instance hooks
// ------------------------------------------------------------------------------------
VKAPI_ATTR VkResult VKAPI_CALL hook_CreateInstance(const VkInstanceCreateInfo* ci,
                                                   const VkAllocationCallbacks* alloc,
                                                   VkInstance* pInstance) {
  auto* chain = (VkLayerInstanceCreateInfo*)ci->pNext;
  while (chain && !(chain->sType == VK_STRUCTURE_TYPE_LOADER_INSTANCE_CREATE_INFO &&
                    chain->function == VK_LAYER_LINK_INFO))
    chain = (VkLayerInstanceCreateInfo*)chain->pNext;
  if (!chain) return VK_ERROR_INITIALIZATION_FAILED;
  PFN_vkGetInstanceProcAddr gipa = chain->u.pLayerInfo->pfnNextGetInstanceProcAddr;
  chain->u.pLayerInfo = chain->u.pLayerInfo->pNext;
  auto create = (PFN_vkCreateInstance)gipa(VK_NULL_HANDLE, "vkCreateInstance");
  LoadConfig();
  // our threads (watchdog, publisher) may outlive the instance (games often leak the device at
  // exit): never let the loader unload this library under them
  static bool pinned = false;
  if (!pinned) {
    pinned = true;
    Dl_info info;
    if (dladdr((void*)&hook_CreateInstance, &info) && info.dli_fname) dlopen(info.dli_fname, RTLD_NOW | RTLD_NODELETE);
  }
  VkResult r = create(ci, alloc, pInstance);
  const VkApplicationInfo* ai = ci->pApplicationInfo;
  Log(r == VK_SUCCESS ? 1 : 0, "vkCreateInstance%s: app '%s' engine '%s' api %u.%u", r == VK_SUCCESS ? "" : " FAILED",
      ai && ai->pApplicationName ? ai->pApplicationName : "", ai && ai->pEngineName ? ai->pEngineName : "",
      ai ? VK_API_VERSION_MAJOR(ai->apiVersion) : 1, ai ? VK_API_VERSION_MINOR(ai->apiVersion) : 0);
  if (r != VK_SUCCESS) return r;

  auto* id = new InstanceData();
  id->instance = *pInstance;
  id->gipa = gipa;
  if (ci->pApplicationInfo && ci->pApplicationInfo->apiVersion)
    id->apiVersion = ci->pApplicationInfo->apiVersion;
  if (ci->pApplicationInfo && ci->pApplicationInfo->pEngineName)
    id->translation = strcasestr(ci->pApplicationInfo->pEngineName, "dxvk") ||
                      strcasestr(ci->pApplicationInfo->pEngineName, "vkd3d");
#define LOAD(n) id->d.n = (PFN_vk##n)gipa(*pInstance, "vk" #n);
  INST_FUNCS(LOAD)
#undef LOAD
  std::lock_guard<std::mutex> l(g_lock);
  g_instByKey[Key(*pInstance)] = id;
  return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL hook_DestroyInstance(VkInstance instance, const VkAllocationCallbacks* a) {
  InstanceData* id = Inst(instance);
  if (!id) return;
  id->d.DestroyInstance(instance, a);
  std::lock_guard<std::mutex> l(g_lock);
  g_instByKey.erase(Key(instance));
  for (auto it = g_instByPhys.begin(); it != g_instByPhys.end();)
    it = (it->second == id) ? g_instByPhys.erase(it) : std::next(it);
  delete id;
}

VKAPI_ATTR VkResult VKAPI_CALL hook_EnumeratePhysicalDevices(VkInstance instance, uint32_t* count,
                                                             VkPhysicalDevice* devs) {
  InstanceData* id = Inst(instance);
  VkResult r = id->d.EnumeratePhysicalDevices(instance, count, devs);
  if (devs && (r == VK_SUCCESS || r == VK_INCOMPLETE)) {
    std::lock_guard<std::mutex> l(g_lock);
    for (uint32_t i = 0; i < *count; ++i) g_instByPhys[devs[i]] = id;
  }
  return r;
}

VkResult EnumGroups(PFN_vkEnumeratePhysicalDeviceGroups fn, InstanceData* id, VkInstance instance,
                    uint32_t* count, VkPhysicalDeviceGroupProperties* props) {
  VkResult r = fn(instance, count, props);
  if (props && (r == VK_SUCCESS || r == VK_INCOMPLETE)) {
    std::lock_guard<std::mutex> l(g_lock);
    for (uint32_t i = 0; i < *count; ++i)
      for (uint32_t j = 0; j < props[i].physicalDeviceCount; ++j)
        g_instByPhys[props[i].physicalDevices[j]] = id;
  }
  return r;
}
VKAPI_ATTR VkResult VKAPI_CALL hook_EnumeratePhysicalDeviceGroups(
    VkInstance instance, uint32_t* count, VkPhysicalDeviceGroupProperties* props) {
  InstanceData* id = Inst(instance);
  return EnumGroups(id->d.EnumeratePhysicalDeviceGroups, id, instance, count, props);
}
VKAPI_ATTR VkResult VKAPI_CALL hook_EnumeratePhysicalDeviceGroupsKHR(
    VkInstance instance, uint32_t* count, VkPhysicalDeviceGroupProperties* props) {
  InstanceData* id = Inst(instance);
  return EnumGroups(id->d.EnumeratePhysicalDeviceGroupsKHR, id, instance, count, props);
}

VKAPI_ATTR VkResult VKAPI_CALL hook_EnumerateDeviceExtensionProperties(
    VkPhysicalDevice phys, const char* layerName, uint32_t* count, VkExtensionProperties* props) {
  if (layerName && !strcmp(layerName, LAYER_NAME)) {
    *count = 0;
    return VK_SUCCESS;
  }
  InstanceData* id = InstFromPhys(phys);
  if (!id) return VK_ERROR_INITIALIZATION_FAILED;
  if (layerName || Cfg().passthrough)
    return id->d.EnumerateDeviceExtensionProperties(phys, layerName, count, props);

  uint32_t n = 0;
  VkResult r = id->d.EnumerateDeviceExtensionProperties(phys, nullptr, &n, nullptr);
  if (r != VK_SUCCESS) return r;
  std::vector<VkExtensionProperties> all(n);
  r = id->d.EnumerateDeviceExtensionProperties(phys, nullptr, &n, all.data());
  if (r != VK_SUCCESS && r != VK_INCOMPLETE) return r;
  all.resize(n);
  std::vector<VkExtensionProperties> kept;
  std::string hidden;
  for (auto& e : all) {
    if (!IsHidden(e.extensionName)) kept.push_back(e);
    else hidden += std::string(" ") + e.extensionName;
  }
  static std::atomic<bool> logged{false};
  if (!hidden.empty() && !logged.exchange(true)) Log(1, "hiding from the game:%s", hidden.c_str());
  if (!props) {
    *count = uint32_t(kept.size());
    return VK_SUCCESS;
  }
  uint32_t c = std::min<uint32_t>(*count, uint32_t(kept.size()));
  std::copy(kept.begin(), kept.begin() + c, props);
  *count = c;
  return c < kept.size() ? VK_INCOMPLETE : VK_SUCCESS;
}

// ------------------------------------------------------------------------------------
// Render size: native Vulkan games are told the window has the size that suits the headset
// (they render at the window size). DXVK / VKD3D games keep the resolution chosen in the game.
// ------------------------------------------------------------------------------------
bool ForcedRenderSize(InstanceData* id, VkExtent2D& out) {
  const Config& c = Cfg();
  if (c.passthrough || !id || id->translation || c.renderW == 0) return false;
  static std::mutex m;
  static VkExtent2D decided{0, 0};
  std::lock_guard<std::mutex> l(m);
  if (!decided.width) {
    if (c.renderW > 0) {
      decided = {uint32_t(c.renderW), uint32_t(c.renderH)};
    } else if (c.display == 1) {
      decided = {1920, 1080};
    } else {
      Link().Poll();
      uint32_t w = 2160, h = 2376;  // Pico 4 class headset if the host does not know yet
      if (auto* s = Link().Shm(); s && s->recW && s->recH) { w = s->recW; h = s->recH; }
      decided = {(uint32_t(w * 0.85f) + 7) & ~7u, (uint32_t(h * 0.85f) + 7) & ~7u};
    }
    Log(1, "render size for this game: %ux%u (render_size = auto|off|WxH)", decided.width, decided.height);
  }
  out = decided;
  return true;
}
void AdjustCaps(InstanceData* id, VkSurfaceCapabilitiesKHR& c) {
  VkExtent2D e;
  if (!ForcedRenderSize(id, e)) return;
  c.currentExtent = e;
  c.minImageExtent.width = std::min(c.minImageExtent.width, e.width);
  c.minImageExtent.height = std::min(c.minImageExtent.height, e.height);
  c.maxImageExtent.width = std::max(c.maxImageExtent.width, e.width);
  c.maxImageExtent.height = std::max(c.maxImageExtent.height, e.height);
}
VKAPI_ATTR VkResult VKAPI_CALL hook_GetPhysicalDeviceSurfaceCapabilitiesKHR(VkPhysicalDevice phys, VkSurfaceKHR surface,
                                                                           VkSurfaceCapabilitiesKHR* caps) {
  InstanceData* id = InstFromPhys(phys);
  VkResult r = id->d.GetPhysicalDeviceSurfaceCapabilitiesKHR(phys, surface, caps);
  if (r == VK_SUCCESS) AdjustCaps(id, *caps);
  return r;
}
VKAPI_ATTR VkResult VKAPI_CALL hook_GetPhysicalDeviceSurfaceCapabilities2KHR(VkPhysicalDevice phys,
                                                                            const VkPhysicalDeviceSurfaceInfo2KHR* info,
                                                                            VkSurfaceCapabilities2KHR* caps) {
  InstanceData* id = InstFromPhys(phys);
  VkResult r = id->d.GetPhysicalDeviceSurfaceCapabilities2KHR(phys, info, caps);
  if (r == VK_SUCCESS) AdjustCaps(id, caps->surfaceCapabilities);
  return r;
}

// ------------------------------------------------------------------------------------
// Stall watchdog: if the game stops presenting, say where we are and release held signals
// ------------------------------------------------------------------------------------
void AbortReplayLocked(DeviceData* dd, const char* why);

// What every thread of the process is doing (name, state, kernel wait channel)
void LogThreads() {
  DIR* d = opendir("/proc/self/task");
  if (!d) return;
  int n = 0;
  while (dirent* e = readdir(d)) {
    if (e->d_name[0] == '.') continue;
    std::string base = std::string("/proc/self/task/") + e->d_name;
    auto readFile = [](const std::string& p) {
      std::string s;
      if (FILE* f = fopen(p.c_str(), "r")) {
        char buf[256];
        size_t k = fread(buf, 1, sizeof(buf) - 1, f);
        buf[k] = 0;
        s = buf;
        fclose(f);
      }
      while (!s.empty() && (s.back() == '\n' || s.back() == ' ')) s.pop_back();
      return s;
    };
    std::string comm = readFile(base + "/comm"), wchan = readFile(base + "/wchan"), stat = readFile(base + "/stat");
    std::string sys = readFile(base + "/syscall");
    char state = '?';
    size_t rp = stat.rfind(')');
    if (rp != std::string::npos && rp + 2 < stat.size()) state = stat[rp + 2];
    if (sys.size() > 12) sys = sys.substr(0, sys.find(' '));
    Log(0, "  thread %s %-16s state %c wchan %-24s syscall %s", e->d_name, comm.c_str(), state, wchan.c_str(),
        sys.c_str());
    if (++n >= 64) break;
  }
  closedir(d);
}

void WatchdogMain(DeviceData* dd) {
  uint64_t last = dd->presentCount.load();
  auto lastChange = Clock::now();
  int level = 0;
  while (!dd->stopWatchdog.load()) {
    std::this_thread::sleep_for(std::chrono::milliseconds(250));
    uint64_t pc = dd->presentCount.load();
    auto now = Clock::now();
    if (pc != last || dd->swapCount.load() == 0) {
      if (level > 0 && pc != last) Log(0, "stall over, presenting again");
      last = pc;
      lastChange = now;
      level = 0;
      continue;
    }
    double st = std::chrono::duration<double>(now - lastChange).count();
    if ((level == 0 && st > 2.0) || (level == 1 && st > 6.0)) {
      ++level;
      Log(0, "STALL %.0f s without presenting (frame %llu). Layer state: %s. mode=%s", st,
          (unsigned long long)pc, dd->stage.load(), dd->replay ? "replay" : "alternate");
      LogThreads();
      if (dd->replay) {
        if (dd->recLock.try_lock()) {
          Log(0, "  held: %zu fences, %zu timeline semaphores, %zu command buffers -> releasing",
              dd->heldFences.size(), dd->heldTimeline.size(), dd->recCbs.size());
          AbortReplayLocked(dd, "watchdog");
          if (level == 2) {
            Log(0, "  still stalled: switching to alternate mode");
            dd->replay = false;
            dd->recMono = false;
          }
          dd->recLock.unlock();
        } else {
          Log(0, "  replay lock busy (the present thread is blocked: %s)", dd->stage.load());
        }
      }
    }
  }
}

// ------------------------------------------------------------------------------------
// Device creation
// ------------------------------------------------------------------------------------
EyeParams IdentityParams() {
  EyeParams e{};
  e.m[0] = e.m[5] = e.m[10] = 1.0f;  // x, y, w rows
  e.m[14] = e.m[15] = 1.0f;            // HUD scale x, y
  return e;
}

bool CreateParamBuffer(DeviceData* dd) {
  VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  bci.size = 1024;
  bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
              VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
  bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  if (dd->d.CreateBuffer(dd->device, &bci, nullptr, &dd->paramBuf) != VK_SUCCESS) return false;
  VkMemoryRequirements req;
  dd->d.GetBufferMemoryRequirements(dd->device, dd->paramBuf, &req);
  uint32_t type = FindMemType(dd, req.memoryTypeBits,
                              VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                              VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  VkMemoryAllocateFlagsInfo fi{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO};
  fi.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
  VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, &fi};
  ai.allocationSize = req.size;
  ai.memoryTypeIndex = type;
  if (type == UINT32_MAX || dd->d.AllocateMemory(dd->device, &ai, nullptr, &dd->paramMem) != VK_SUCCESS)
    return false;
  dd->d.BindBufferMemory(dd->device, dd->paramBuf, dd->paramMem, 0);
  void* map = nullptr;
  if (dd->d.MapMemory(dd->device, dd->paramMem, 0, VK_WHOLE_SIZE, 0, &map) == VK_SUCCESS) {
    EyeParams id = IdentityParams();
    for (int o : {0, 256, 512}) memcpy((char*)map + o, id.m, sizeof(id.m));
    dd->d.UnmapMemory(dd->device, dd->paramMem);
  }
  VkBufferDeviceAddressInfo bai{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
  bai.buffer = dd->paramBuf;
  auto gbda = (dd->apiVersion >= VK_API_VERSION_1_2 && dd->d.GetBufferDeviceAddress)
                  ? dd->d.GetBufferDeviceAddress : dd->d.GetBufferDeviceAddressKHR;
  if (!gbda) gbda = dd->d.GetBufferDeviceAddress;
  if (!gbda) return false;
  dd->paramAddr = gbda(dd->device, &bai);
  return dd->paramAddr != 0;
}

void PublisherMain(DeviceData* dd);
void StopThreadsAtExit();
void RunDeferred(DeviceData* dd, bool all);
void DestroyExport(DeviceData* dd, VrExport* ex);

VKAPI_ATTR VkResult VKAPI_CALL hook_CreateDevice(VkPhysicalDevice phys, const VkDeviceCreateInfo* pci,
                                                 const VkAllocationCallbacks* alloc, VkDevice* pDevice) {
  InstanceData* inst = InstFromPhys(phys);
  auto* chain = (VkLayerDeviceCreateInfo*)pci->pNext;
  while (chain && !(chain->sType == VK_STRUCTURE_TYPE_LOADER_DEVICE_CREATE_INFO &&
                    chain->function == VK_LAYER_LINK_INFO))
    chain = (VkLayerDeviceCreateInfo*)chain->pNext;
  if (!chain || !inst) return VK_ERROR_INITIALIZATION_FAILED;
  PFN_vkSetDeviceLoaderData setLoaderData = nullptr;
  for (auto* c = (VkLayerDeviceCreateInfo*)pci->pNext; c; c = (VkLayerDeviceCreateInfo*)c->pNext)
    if (c->sType == VK_STRUCTURE_TYPE_LOADER_DEVICE_CREATE_INFO && c->function == VK_LOADER_DATA_CALLBACK)
      setLoaderData = c->u.pfnSetDeviceLoaderData;

  PFN_vkGetInstanceProcAddr gipa = chain->u.pLayerInfo->pfnNextGetInstanceProcAddr;
  PFN_vkGetDeviceProcAddr gdpa = chain->u.pLayerInfo->pfnNextGetDeviceProcAddr;
  chain->u.pLayerInfo = chain->u.pLayerInfo->pNext;
  auto create = (PFN_vkCreateDevice)gipa(inst->instance, "vkCreateDevice");

  // ---- can we do shader stereo? (buffer device address + int64) ----
  VkPhysicalDeviceProperties props{};
  inst->d.GetPhysicalDeviceProperties(phys, &props);
  uint32_t api = std::min(props.apiVersion, inst->apiVersion);
  bool supported = false;
  auto gpf2 = inst->d.GetPhysicalDeviceFeatures2 ? inst->d.GetPhysicalDeviceFeatures2
                                                 : inst->d.GetPhysicalDeviceFeatures2KHR;
  if (gpf2) {
    VkPhysicalDeviceBufferDeviceAddressFeatures bda{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES};
    VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &bda};
    gpf2(phys, &f2);
    supported = bda.bufferDeviceAddress && f2.features.shaderInt64;
  }
  // Proton (DXVK / VKD3D-Proton) always uses Vulkan 1.3. For 1.0 apps BDA would need extra
  // instance extensions, so only Vulkan >= 1.1 is handled (STEREO3D_LEGACY=1 forces it).
  const char* legacy = getenv("STEREO3D_LEGACY");
  bool apiOk = api >= VK_API_VERSION_1_1 || (legacy && legacy[0] == '1');
  bool wantStereo = Cfg().patchShaders && supported && apiOk;

  VkDeviceCreateInfo ci = *pci;
  std::vector<const char*> exts;
  for (uint32_t i = 0; i < pci->enabledExtensionCount; ++i) {
    bool always = false;
    for (const char* h : kAlwaysHidden) always |= !strcmp(h, pci->ppEnabledExtensionNames[i]);
    if (always && !Cfg().passthrough) {
      Log(1, "not enabling %s (incompatible with the stereo swapchain)", pci->ppEnabledExtensionNames[i]);
      continue;
    }
    exts.push_back(pci->ppEnabledExtensionNames[i]);
  }
  // export the eye images to vrlinux-host
  bool canExport = false;
  if (!Cfg().passthrough && api >= VK_API_VERSION_1_1) {
    uint32_t n = 0;
    inst->d.EnumerateDeviceExtensionProperties(phys, nullptr, &n, nullptr);
    std::vector<VkExtensionProperties> avail(n);
    inst->d.EnumerateDeviceExtensionProperties(phys, nullptr, &n, avail.data());
    for (auto& e : avail) canExport |= !strcmp(e.extensionName, "VK_KHR_external_memory_fd");
    if (canExport) {
      bool have = false;
      for (const char* e : exts) have |= !strcmp(e, "VK_KHR_external_memory_fd");
      if (!have) exts.push_back("VK_KHR_external_memory_fd");
    } else {
      Log(0, "VK_KHR_external_memory_fd not available: cannot send frames to the headset");
    }
  }
  ci.enabledExtensionCount = uint32_t(exts.size());
  ci.ppEnabledExtensionNames = exts.data();
  VkPhysicalDeviceFeatures featCopy{};
  VkPhysicalDeviceBufferDeviceAddressFeatures ourBda{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES};
  if (wantStereo) {
    if (api < VK_API_VERSION_1_2) {
      bool have = false;
      for (const char* e : exts) have |= !strcmp(e, "VK_KHR_buffer_device_address");
      if (!have) exts.push_back("VK_KHR_buffer_device_address");
      ci.enabledExtensionCount = uint32_t(exts.size());
      ci.ppEnabledExtensionNames = exts.data();
    }
    bool haveBda = false, haveI64 = false;
    for (auto* p = (VkBaseOutStructure*)ci.pNext; p; p = p->pNext) {
      switch (p->sType) {
        case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES:
          ((VkPhysicalDeviceVulkan12Features*)p)->bufferDeviceAddress = VK_TRUE;
          haveBda = true;
          break;
        case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES:
          ((VkPhysicalDeviceBufferDeviceAddressFeatures*)p)->bufferDeviceAddress = VK_TRUE;
          haveBda = true;
          break;
        case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2:
          ((VkPhysicalDeviceFeatures2*)p)->features.shaderInt64 = VK_TRUE;
          haveI64 = true;
          break;
        default: break;
      }
    }
    if (!haveI64) {
      if (ci.pEnabledFeatures) featCopy = *ci.pEnabledFeatures;
      featCopy.shaderInt64 = VK_TRUE;
      ci.pEnabledFeatures = &featCopy;
    }
    if (!haveBda) {
      ourBda.bufferDeviceAddress = VK_TRUE;
      ourBda.pNext = (void*)ci.pNext;
      ci.pNext = &ourBda;
    }
  } else {
    Log(0, "shader stereo NOT available (bda+int64 supported=%d, api>=1.1=%d, patch=%d)",
        supported, apiOk, Cfg().patchShaders);
  }

  VkResult r = create(phys, &ci, alloc, pDevice);
  if (r != VK_SUCCESS) {
    std::string e;
    for (uint32_t i = 0; i < ci.enabledExtensionCount; ++i) e += std::string(" ") + ci.ppEnabledExtensionNames[i];
    Log(0, "vkCreateDevice FAILED (VkResult %d) on '%s'. stereo=%d export=%d extensions:%s", (int)r,
        props.deviceName, wantStereo, canExport, e.c_str());
    return r;
  }

  auto* dd = new DeviceData();
  dd->device = *pDevice;
  dd->phys = phys;
  dd->inst = inst;
  dd->setLoaderData = setLoaderData;
  dd->apiVersion = api;
  dd->canExport = canExport;
  dd->d.GetDeviceProcAddr = gdpa;
#define LOAD(n) dd->d.n = (PFN_vk##n)gdpa(*pDevice, "vk" #n);
  DEV_FUNCS(LOAD)
#undef LOAD
  inst->d.GetPhysicalDeviceMemoryProperties(phys, &dd->memProps);
  dd->canExport = dd->canExport && dd->d.GetMemoryFdKHR;
  {
    auto gp2 = inst->d.GetPhysicalDeviceProperties2 ? inst->d.GetPhysicalDeviceProperties2
                                                    : inst->d.GetPhysicalDeviceProperties2KHR;
    if (gp2) {
      VkPhysicalDeviceIDProperties idp{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES};
      VkPhysicalDeviceProperties2 p2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &idp};
      gp2(phys, &p2);
      memcpy(dd->deviceUUID, idp.deviceUUID, 16);
      memcpy(dd->driverUUID, idp.driverUUID, 16);
    } else {
      dd->canExport = false;
    }
  }

  uint32_t qfCount = 0;
  inst->d.GetPhysicalDeviceQueueFamilyProperties(phys, &qfCount, nullptr);
  std::vector<VkQueueFamilyProperties> qf(qfCount);
  inst->d.GetPhysicalDeviceQueueFamilyProperties(phys, &qfCount, qf.data());
  for (uint32_t i = 0; i < pci->queueCreateInfoCount; ++i) {
    uint32_t fam = pci->pQueueCreateInfos[i].queueFamilyIndex;
    if (fam < qfCount && (qf[fam].queueFlags & VK_QUEUE_GRAPHICS_BIT)) {
      dd->gfxFamily = fam;
      break;
    }
  }

  if (wantStereo) {
    dd->stereoShaders = CreateParamBuffer(dd);
    dd->replay = dd->stereoShaders && Cfg().mode == 1;
    if (!dd->stereoShaders) Log(0, "failed to create stereo parameter buffer");
  }
  Log(1, "device '%s' api %u.%u: shader stereo %s, mode %s, param @0x%llx", props.deviceName,
      VK_API_VERSION_MAJOR(api), VK_API_VERSION_MINOR(api), dd->stereoShaders ? "ON" : "OFF",
      dd->replay ? "replay (both eyes every frame)" : "alternate (L/R frames)", (unsigned long long)dd->paramAddr);

  if (!Cfg().passthrough) {
    // our helper threads must never receive process signals: under Wine they would run Wine's
    // handlers on a thread Wine doesn't know (crash at exit: "nested exception on signal stack")
    static bool atexitSet = false;
    if (!atexitSet) {
      atexitSet = true;
      atexit(StopThreadsAtExit);
    }
    sigset_t all, old;
    sigfillset(&all);
    pthread_sigmask(SIG_SETMASK, &all, &old);
    dd->watchdog = std::thread(WatchdogMain, dd);
    if (dd->canExport) dd->publisher = std::thread(PublisherMain, dd);
    pthread_sigmask(SIG_SETMASK, &old, nullptr);
  }
  std::lock_guard<std::mutex> l(g_lock);
  g_devByKey[Key(*pDevice)] = dd;
  return VK_SUCCESS;
}

// Process exit with the device still alive (most games never destroy it): stop our threads before
// the C++ runtime tears down the globals they use. Registered after our static objects, so it runs
// before their destructors.
void StopThreadsAtExit() {
  std::vector<DeviceData*> devs;
  {
    std::lock_guard<std::mutex> l(g_lock);
    for (auto& kv : g_devByKey) devs.push_back(kv.second);
  }
  for (DeviceData* dd : devs) {
    dd->stopWatchdog = true;
    {
      std::lock_guard<std::mutex> l(dd->pubLock);
      dd->pubStop = true;
    }
    dd->pubCv.notify_all();
  }
  for (DeviceData* dd : devs) {
    if (dd->watchdog.joinable()) dd->watchdog.join();
    if (dd->publisher.joinable()) dd->publisher.join();
  }
}

VKAPI_ATTR void VKAPI_CALL hook_DestroyDevice(VkDevice device, const VkAllocationCallbacks* a) {
  DeviceData* dd = Dev(device);
  if (!dd) return;
  g_fastKey.store(nullptr);
  g_fastDev.store(nullptr);
  dd->stopWatchdog = true;
  if (dd->watchdog.joinable()) dd->watchdog.join();
  {
    std::lock_guard<std::mutex> l(dd->pubLock);
    dd->pubStop = true;
  }
  dd->pubCv.notify_all();
  if (dd->publisher.joinable()) dd->publisher.join();
  dd->d.DeviceWaitIdle(device);
  RunDeferred(dd, true);
  for (VkFence f : dd->serialFence)
    if (f) dd->d.DestroyFence(device, f, nullptr);
  {
    std::lock_guard<std::mutex> l(dd->shaderLock);
    for (auto& m : dd->modules)
      for (auto& v : m.second.variants)
        if (v.patched) dd->d.DestroyShaderModule(device, v.patched, nullptr);
    dd->modules.clear();
  }
  if (dd->paramBuf) dd->d.DestroyBuffer(device, dd->paramBuf, nullptr);
  if (dd->paramMem) dd->d.FreeMemory(device, dd->paramMem, nullptr);
  dd->d.DestroyDevice(device, a);
  {
    std::lock_guard<std::mutex> l(g_lock);
    g_devByKey.erase(Key(device));
  }
  delete dd;
}

void RegisterQueue(DeviceData* dd, VkQueue q, uint32_t family) {
  QueueInfo* qi = GetQueueInfo(dd, q);
  qi->family = family;
  if (family == dd->gfxFamily && !dd->gfxQueue) dd->gfxQueue = q;
}

VKAPI_ATTR void VKAPI_CALL hook_GetDeviceQueue(VkDevice device, uint32_t family, uint32_t index, VkQueue* q) {
  DeviceData* dd = Dev(device);
  dd->d.GetDeviceQueue(device, family, index, q);
  if (*q) RegisterQueue(dd, *q, family);
}
VKAPI_ATTR void VKAPI_CALL hook_GetDeviceQueue2(VkDevice device, const VkDeviceQueueInfo2* info, VkQueue* q) {
  DeviceData* dd = Dev(device);
  dd->d.GetDeviceQueue2(device, info, q);
  if (*q) RegisterQueue(dd, *q, info->queueFamilyIndex);
}

// ------------------------------------------------------------------------------------
// Queue ops. In replay mode the frame's command buffers are recorded and the timeline
// signals / fences of the present queue are held back until the right-eye replay is done.
// ------------------------------------------------------------------------------------
bool IsTimeline(DeviceData* dd, VkSemaphore sem) {
  std::lock_guard<std::mutex> l(dd->semLock);
  return dd->timelineSems.count(sem) != 0;
}

// caller holds dd->recLock and the queue lock of q
bool Tracing() { static int t = getenv("VRLINUX_TRACE") ? 1 : 0; return t; }
void ReleaseHeldLocked(DeviceData* dd, VkQueue q) {
  if (dd->heldTimeline.empty() && dd->heldFences.empty()) return;
  if (getenv("VRLINUX_TRACE")) {
    std::string o = "release held:";
    for (auto& kv : dd->heldTimeline) o += " " + std::to_string((uintptr_t)kv.first % 100000) + "=" + std::to_string(kv.second);
    Log(0, "%s fences %zu", o.c_str(), dd->heldFences.size());
  }
  std::vector<VkSemaphore> sems;
  std::vector<uint64_t> vals;
  for (auto& kv : dd->heldTimeline) { sems.push_back(kv.first); vals.push_back(kv.second); }
  VkTimelineSemaphoreSubmitInfo tl{VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO};
  tl.signalSemaphoreValueCount = uint32_t(vals.size());
  tl.pSignalSemaphoreValues = vals.data();
  VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO, sems.empty() ? nullptr : &tl};
  si.signalSemaphoreCount = uint32_t(sems.size());
  si.pSignalSemaphores = sems.data();
  size_t fi = 0;
  VkFence f0 = dd->heldFences.empty() ? VK_NULL_HANDLE : dd->heldFences[fi++];
  dd->d.QueueSubmit(q, 1, &si, f0);
  for (; fi < dd->heldFences.size(); ++fi) {
    VkSubmitInfo e{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    dd->d.QueueSubmit(q, 1, &e, dd->heldFences[fi]);
  }
  dd->heldTimeline.clear();
  dd->heldFences.clear();
}

// The app needs a held signal before the frame is complete: no replay for this frame.
// caller holds dd->recLock (not the queue lock)
void AbortReplayLocked(DeviceData* dd, const char* why) {
  if (!dd->recMono) {
    dd->recMono = true;
    ++dd->monoFrames;
    dd->lastAbortPresent = dd->presentCount.load();
    Log(2, "replay skipped this frame: %s", why);
  }
  dd->recCbs.clear();
  {
    std::lock_guard<std::mutex> l(dd->cbLock);
    dd->recSet.clear();
  }
  dd->cbCv.notify_all();
  if (dd->gfxQueue) {
    std::lock_guard<std::mutex> l(GetQueueInfo(dd, dd->gfxQueue)->m);
    ReleaseHeldLocked(dd, dd->gfxQueue);
  }
}

// Relaxed: the app needs a signal of this frame now. Release the held signals but keep
// recording; the frame is still replayed at present. caller holds dd->recLock.
void RelaxReplayLocked(DeviceData* dd, const char* why) {
  if (!Cfg().replayRelaxed || dd->recMono) {
    AbortReplayLocked(dd, why);
    return;
  }
  {
    std::lock_guard<std::mutex> l(dd->cbLock);
    if (dd->relaxedSerial != dd->frameSerial) {
      dd->relaxedSerial = dd->frameSerial;
      ++dd->relaxedFrames;
      dd->lastAbortPresent = dd->presentCount.load();
      Log(2, "relaxed replay this frame: %s", why);
    }
  }
  if (dd->gfxQueue) {
    std::lock_guard<std::mutex> l(GetQueueInfo(dd, dd->gfxQueue)->m);
    ReleaseHeldLocked(dd, dd->gfxQueue);
  }
}

// The app is about to reset / free command buffers or destroy objects. If they belong to a
// frame whose replay is not done yet, wait for it (the replay must not use freed objects).
// 'cbs' may be empty (destroy of other objects: only waits when the frame was relaxed).
void WaitReplayDone(DeviceData* dd, const VkCommandBuffer* cbs, uint32_t n, bool anyObject) {
  if (!dd->replay) return;
  std::unique_lock<std::mutex> l(dd->cbLock);
  uint64_t need = 0;
  for (uint32_t i = 0; i < n; ++i) {
    VkCommandBuffer cb = cbs[i];
    if (dd->recSet.count(cb)) {  // recorded this frame, not presented yet: wait for the present
      if (!dd->cbCv.wait_for(l, std::chrono::milliseconds(100), [&]() { return !dd->recSet.count(cb); })) {
        dd->recSet.erase(cb);  // the present does not come: give up the replay of this frame
        dd->dropFrame = true;
        Log(2, "replay dropped: command buffer reused before present");
      }
    }
    auto it = dd->cbSerial.find(cb);
    if (it != dd->cbSerial.end()) {
      need = std::max(need, it->second);
      dd->cbSerial.erase(it);
    }
  }
  if (anyObject && dd->relaxedSerial) {  // descriptor pool reset while a relaxed frame is pending
    uint64_t want = dd->relaxedSerial;
    if (want >= dd->frameSerial) {
      if (!dd->cbCv.wait_for(l, std::chrono::milliseconds(100), [&]() { return dd->submittedSerial >= want; })) {
        dd->dropFrame = true;
        Log(2, "replay dropped: descriptors reset before present");
      } else {
        need = std::max(need, want);
      }
    } else {
      need = std::max(need, want);
    }
  }
  if (!need) return;
  if (dd->submittedSerial < need)
    dd->cbCv.wait_for(l, std::chrono::milliseconds(250), [&]() { return dd->submittedSerial >= need; });
  int k = int(need % 8);
  VkFence f = dd->serialOf[k] == need ? dd->serialFence[k] : VK_NULL_HANDLE;
  l.unlock();
  if (f) dd->d.WaitForFences(dd->device, 1, &f, VK_TRUE, 1000000000ull);
}

bool HeldLocked(DeviceData* dd, VkSemaphore sem, uint64_t value) {
  auto it = dd->heldTimeline.find(sem);
  return it != dd->heldTimeline.end() && value <= it->second;
}
bool WaitsOnHeldLocked(DeviceData* dd, uint32_t n, const VkSubmitInfo* s) {
  if (dd->heldTimeline.empty()) return false;
  for (uint32_t i = 0; i < n; ++i) {
    const VkTimelineSemaphoreSubmitInfo* tl = nullptr;
    for (auto* p = (const VkBaseInStructure*)s[i].pNext; p; p = p->pNext)
      if (p->sType == VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO) tl = (const VkTimelineSemaphoreSubmitInfo*)p;
    if (!tl || !tl->pWaitSemaphoreValues) continue;
    for (uint32_t w = 0; w < s[i].waitSemaphoreCount && w < tl->waitSemaphoreValueCount; ++w)
      if (HeldLocked(dd, s[i].pWaitSemaphores[w], tl->pWaitSemaphoreValues[w])) return true;
  }
  return false;
}
bool WaitsOnHeldLocked(DeviceData* dd, uint32_t n, const VkSubmitInfo2* s) {
  if (dd->heldTimeline.empty()) return false;
  for (uint32_t i = 0; i < n; ++i)
    for (uint32_t w = 0; w < s[i].waitSemaphoreInfoCount; ++w)
      if (HeldLocked(dd, s[i].pWaitSemaphoreInfos[w].semaphore, s[i].pWaitSemaphoreInfos[w].value)) return true;
  return false;
}

bool RecordingQueue(DeviceData* dd, VkQueue q) {
  return dd->replay && q == dd->gfxQueue && dd->swapCount.load() > 0;
}

VKAPI_ATTR VkResult VKAPI_CALL hook_QueueSubmit(VkQueue q, uint32_t n, const VkSubmitInfo* s, VkFence f) {
  DeviceData* dd = Dev(q);
  if (Tracing()) Log(0, "submit1 q=%lu n=%u fence=%d", (unsigned long)((uintptr_t)q % 100000), n, f != VK_NULL_HANDLE);
  if (!dd->replay) {
    std::lock_guard<std::mutex> l(GetQueueInfo(dd, q)->m);
    return dd->d.QueueSubmit(q, n, s, f);
  }
  std::unique_lock<std::mutex> rl(dd->recLock);
  if (WaitsOnHeldLocked(dd, n, s)) RelaxReplayLocked(dd, "submit waits on a held signal");
  if (!RecordingQueue(dd, q) || dd->recMono) {
    rl.unlock();
    std::lock_guard<std::mutex> l(GetQueueInfo(dd, q)->m);
    return dd->d.QueueSubmit(q, n, s, f);
  }
  {
    std::lock_guard<std::mutex> cl(dd->cbLock);
    for (uint32_t i = 0; i < n; ++i)
      for (uint32_t c = 0; c < s[i].commandBufferCount; ++c) dd->recSet.insert(s[i].pCommandBuffers[c]);
  }
  // semaphores this same submission waits on (a later batch waiting for an earlier one):
  // their signals can never be held, the GPU would wait forever
  std::vector<VkSemaphore> waited;
  for (uint32_t i = 0; i < n; ++i)
    waited.insert(waited.end(), s[i].pWaitSemaphores, s[i].pWaitSemaphores + s[i].waitSemaphoreCount);
  auto isWaited = [&](VkSemaphore x) { return std::find(waited.begin(), waited.end(), x) != waited.end(); };
  std::vector<VkSubmitInfo> si(s, s + n);
  std::vector<VkTimelineSemaphoreSubmitInfo> tls(n);
  std::vector<std::vector<VkSemaphore>> sigs(n);
  std::vector<std::vector<uint64_t>> vals(n);
  for (uint32_t i = 0; i < n; ++i) {
    dd->recCbs.insert(dd->recCbs.end(), s[i].pCommandBuffers, s[i].pCommandBuffers + s[i].commandBufferCount);
    const VkTimelineSemaphoreSubmitInfo* tl = nullptr;
    for (auto* p = (const VkBaseInStructure*)s[i].pNext; p; p = p->pNext)
      if (p->sType == VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO) tl = (const VkTimelineSemaphoreSubmitInfo*)p;
    bool changed = false;
    for (uint32_t k = 0; k < s[i].signalSemaphoreCount; ++k) {
      VkSemaphore sem = s[i].pSignalSemaphores[k];
      uint64_t v = (tl && tl->pSignalSemaphoreValues && k < tl->signalSemaphoreValueCount)
                       ? tl->pSignalSemaphoreValues[k] : 0;
      if (IsTimeline(dd, sem) && !isWaited(sem)) {
        uint64_t& h = dd->heldTimeline[sem];
        h = std::max(h, v);
        changed = true;
      } else {
        sigs[i].push_back(sem);
        vals[i].push_back(v);
      }
    }
    if (changed) {
      si[i].signalSemaphoreCount = uint32_t(sigs[i].size());
      si[i].pSignalSemaphores = sigs[i].data();
      if (tl) {
        tls[i] = *tl;
        tls[i].pNext = nullptr;
        tls[i].signalSemaphoreValueCount = uint32_t(vals[i].size());
        tls[i].pSignalSemaphoreValues = vals[i].data();
        si[i].pNext = &tls[i];
      }
    }
  }
  if (f) dd->heldFences.push_back(f);
  std::lock_guard<std::mutex> l(GetQueueInfo(dd, q)->m);
  return dd->d.QueueSubmit(q, n, si.data(), VK_NULL_HANDLE);
}

void TraceSubmit2(DeviceData* dd, VkQueue q, uint32_t n, const VkSubmitInfo2* s, VkFence f) {
  std::string o = "submit2 q=" + std::to_string((uintptr_t)q % 100000) + (q == dd->gfxQueue ? "(gfx)" : "");
  for (uint32_t i = 0; i < n; ++i) {
    o += " [cbs " + std::to_string(s[i].commandBufferInfoCount);
    for (uint32_t w = 0; w < s[i].waitSemaphoreInfoCount; ++w)
      o += " W" + std::to_string((uintptr_t)s[i].pWaitSemaphoreInfos[w].semaphore % 100000) + "=" + std::to_string(s[i].pWaitSemaphoreInfos[w].value);
    for (uint32_t w = 0; w < s[i].signalSemaphoreInfoCount; ++w)
      o += " S" + std::to_string((uintptr_t)s[i].pSignalSemaphoreInfos[w].semaphore % 100000) + "=" + std::to_string(s[i].pSignalSemaphoreInfos[w].value);
    o += "]";
  }
  if (f) o += " fence";
  Log(0, "%s", o.c_str());
}

VkResult Submit2Common(DeviceData* dd, VkQueue q, uint32_t n, const VkSubmitInfo2* s, VkFence f,
                       PFN_vkQueueSubmit2 fn) {
  if (Tracing()) TraceSubmit2(dd, q, n, s, f);
  if (!dd->replay) {
    std::lock_guard<std::mutex> l(GetQueueInfo(dd, q)->m);
    return fn(q, n, s, f);
  }
  std::unique_lock<std::mutex> rl(dd->recLock);
  if (WaitsOnHeldLocked(dd, n, s)) RelaxReplayLocked(dd, "submit waits on a held signal");
  if (!RecordingQueue(dd, q) || dd->recMono) {
    rl.unlock();
    std::lock_guard<std::mutex> l(GetQueueInfo(dd, q)->m);
    return fn(q, n, s, f);
  }
  {
    std::lock_guard<std::mutex> cl(dd->cbLock);
    for (uint32_t i = 0; i < n; ++i)
      for (uint32_t c = 0; c < s[i].commandBufferInfoCount; ++c) dd->recSet.insert(s[i].pCommandBufferInfos[c].commandBuffer);
  }
  std::vector<VkSemaphore> waited;
  for (uint32_t i = 0; i < n; ++i)
    for (uint32_t w = 0; w < s[i].waitSemaphoreInfoCount; ++w) waited.push_back(s[i].pWaitSemaphoreInfos[w].semaphore);
  auto isWaited = [&](VkSemaphore x) { return std::find(waited.begin(), waited.end(), x) != waited.end(); };
  std::vector<VkSubmitInfo2> si(s, s + n);
  std::vector<std::vector<VkSemaphoreSubmitInfo>> sigs(n);
  for (uint32_t i = 0; i < n; ++i) {
    for (uint32_t c = 0; c < s[i].commandBufferInfoCount; ++c)
      dd->recCbs.push_back(s[i].pCommandBufferInfos[c].commandBuffer);
    bool changed = false;
    for (uint32_t k = 0; k < s[i].signalSemaphoreInfoCount; ++k) {
      const VkSemaphoreSubmitInfo& ss = s[i].pSignalSemaphoreInfos[k];
      if (IsTimeline(dd, ss.semaphore) && !isWaited(ss.semaphore)) {
        uint64_t& h = dd->heldTimeline[ss.semaphore];
        h = std::max(h, ss.value);
        changed = true;
      } else {
        sigs[i].push_back(ss);
      }
    }
    if (changed) {
      si[i].signalSemaphoreInfoCount = uint32_t(sigs[i].size());
      si[i].pSignalSemaphoreInfos = sigs[i].data();
    }
  }
  if (f) dd->heldFences.push_back(f);
  std::lock_guard<std::mutex> l(GetQueueInfo(dd, q)->m);
  return fn(q, n, si.data(), VK_NULL_HANDLE);
}
VKAPI_ATTR VkResult VKAPI_CALL hook_QueueSubmit2(VkQueue q, uint32_t n, const VkSubmitInfo2* s, VkFence f) {
  DeviceData* dd = Dev(q);
  return Submit2Common(dd, q, n, s, f, dd->d.QueueSubmit2);
}
VKAPI_ATTR VkResult VKAPI_CALL hook_QueueSubmit2KHR(VkQueue q, uint32_t n, const VkSubmitInfo2* s, VkFence f) {
  DeviceData* dd = Dev(q);
  return Submit2Common(dd, q, n, s, f, dd->d.QueueSubmit2KHR);
}
VKAPI_ATTR VkResult VKAPI_CALL hook_QueueWaitIdle(VkQueue q) {
  DeviceData* dd = Dev(q);
  if (Tracing()) Log(0, "queue wait idle q=%lu", (unsigned long)((uintptr_t)q % 100000));
  if (dd->replay && q == dd->gfxQueue) {
    std::lock_guard<std::mutex> rl(dd->recLock);
    if (!dd->heldFences.empty() || !dd->heldTimeline.empty() || !dd->recCbs.empty())
      AbortReplayLocked(dd, "vkQueueWaitIdle");
  }
  std::lock_guard<std::mutex> l(GetQueueInfo(dd, q)->m);
  return dd->d.QueueWaitIdle(q);
}
VKAPI_ATTR VkResult VKAPI_CALL hook_QueueBindSparse(VkQueue q, uint32_t n, const VkBindSparseInfo* b, VkFence f) {
  DeviceData* dd = Dev(q);
  std::lock_guard<std::mutex> l(GetQueueInfo(dd, q)->m);
  return dd->d.QueueBindSparse(q, n, b, f);
}
VKAPI_ATTR VkResult VKAPI_CALL hook_DeviceWaitIdle(VkDevice device) {
  DeviceData* dd = Dev(device);
  if (!dd->gfxQueue) return dd->d.DeviceWaitIdle(device);
  if (dd->replay) {
    std::lock_guard<std::mutex> rl(dd->recLock);
    if (!dd->heldFences.empty() || !dd->heldTimeline.empty() || !dd->recCbs.empty())
      AbortReplayLocked(dd, "vkDeviceWaitIdle");
  }
  std::lock_guard<std::mutex> l(GetQueueInfo(dd, dd->gfxQueue)->m);
  return dd->d.DeviceWaitIdle(device);
}

// CPU waits: if the app blocks on a held signal and no present arrives, it is waiting for
// something mid-frame (e.g. a GPU readback) -> release and skip the replay for this frame.
// The game polls (timeout 0, vkGetFenceStatus, vkGetSemaphoreCounterValue) for a signal the
// replay holds and no frame is being presented: release the signals (relaxed replay).
template <class HeldFn>
void PolledNotReady(DeviceData* dd, HeldFn heldFn) {
  if (!dd->replay) return;
  uint64_t pc = dd->presentCount.load();
  auto now = Clock::now();
  {
    std::lock_guard<std::mutex> l(dd->pollLock);
    if (dd->pollPresent != pc) {
      dd->pollPresent = pc;
      dd->pollStart = now;
      return;
    }
    bool recentAbort = pc - dd->lastAbortPresent.load() < 120;
    if (now - dd->pollStart < std::chrono::milliseconds(recentAbort ? 1 : 40)) return;
    dd->pollStart = now;
  }
  std::lock_guard<std::mutex> rl(dd->recLock);
  if (heldFn()) RelaxReplayLocked(dd, "app polls a held signal mid-frame");
}
template <class WaitFn, class HeldFn>
VkResult WaitWithReplay(DeviceData* dd, uint64_t timeout, WaitFn waitFn, HeldFn heldFn) {
  if (!dd->replay) return waitFn(timeout);
  if (timeout == 0) {
    VkResult r = waitFn(0);
    if (r == VK_TIMEOUT) PolledNotReady(dd, heldFn);
    return r;
  }
  const auto begin = Clock::now();
  auto stallStart = begin;
  uint64_t seenPresent = dd->presentCount.load();
  for (;;) {
    auto now = Clock::now();
    uint64_t total = uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(now - begin).count());
    uint64_t slice = 1000000ull;  // 1 ms
    if (timeout != UINT64_MAX) {
      if (total >= timeout) return VK_TIMEOUT;
      slice = std::min(slice, timeout - total);
    }
    VkResult r = waitFn(slice);
    if (r != VK_TIMEOUT) return r;
    uint64_t pc = dd->presentCount.load();
    if (pc != seenPresent) {  // frames are still being presented: a normal wait
      seenPresent = pc;
      stallStart = Clock::now();
      continue;
    }
    uint64_t stalled = uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - stallStart).count());
    bool recentAbort = pc - dd->lastAbortPresent.load() < 120;
    if (stalled > (recentAbort ? 1000000ull : 40000000ull)) {
      std::lock_guard<std::mutex> rl(dd->recLock);
      if (heldFn()) RelaxReplayLocked(dd, "CPU waits for a held signal mid-frame");
      stallStart = Clock::now();
    }
  }
}

VKAPI_ATTR VkResult VKAPI_CALL hook_WaitForFences(VkDevice device, uint32_t n, const VkFence* fences, VkBool32 all,
                                                  uint64_t timeout) {
  DeviceData* dd = Dev(device);
  return WaitWithReplay(
      dd, timeout, [&](uint64_t t) { return dd->d.WaitForFences(device, n, fences, all, t); },
      [&]() {
        for (uint32_t i = 0; i < n; ++i)
          if (std::find(dd->heldFences.begin(), dd->heldFences.end(), fences[i]) != dd->heldFences.end())
            return true;
        return false;
      });
}
VkResult WaitSemCommon(DeviceData* dd, VkDevice device, const VkSemaphoreWaitInfo* info, uint64_t timeout,
                       PFN_vkWaitSemaphores fn) {
  if (Tracing() && timeout) {
    std::string o = "cpu wait";
    for (uint32_t i = 0; i < info->semaphoreCount; ++i)
      o += " " + std::to_string((uintptr_t)info->pSemaphores[i] % 100000) + "=" + std::to_string(info->pValues[i]);
    Log(0, "%s", o.c_str());
  }
  return WaitWithReplay(
      dd, timeout, [&](uint64_t t) { return fn(device, info, t); },
      [&]() {
        for (uint32_t i = 0; i < info->semaphoreCount; ++i)
          if (HeldLocked(dd, info->pSemaphores[i], info->pValues[i])) return true;
        return false;
      });
}
VKAPI_ATTR VkResult VKAPI_CALL hook_WaitSemaphores(VkDevice device, const VkSemaphoreWaitInfo* info, uint64_t t) {
  DeviceData* dd = Dev(device);
  return WaitSemCommon(dd, device, info, t, dd->d.WaitSemaphores);
}
VKAPI_ATTR VkResult VKAPI_CALL hook_WaitSemaphoresKHR(VkDevice device, const VkSemaphoreWaitInfo* info, uint64_t t) {
  DeviceData* dd = Dev(device);
  return WaitSemCommon(dd, device, info, t, dd->d.WaitSemaphoresKHR);
}

VKAPI_ATTR VkResult VKAPI_CALL hook_GetFenceStatus(VkDevice device, VkFence f) {
  DeviceData* dd = Dev(device);
  VkResult r = dd->d.GetFenceStatus(device, f);
  if (r == VK_NOT_READY)
    PolledNotReady(dd, [&]() { return std::find(dd->heldFences.begin(), dd->heldFences.end(), f) != dd->heldFences.end(); });
  return r;
}
VkResult CounterCommon(DeviceData* dd, VkDevice device, VkSemaphore sem, uint64_t* v, PFN_vkGetSemaphoreCounterValue fn) {
  VkResult r = fn(device, sem, v);
  if (r == VK_SUCCESS && dd->replay)
    PolledNotReady(dd, [&]() {
      auto it = dd->heldTimeline.find(sem);
      return it != dd->heldTimeline.end() && it->second > *v;
    });
  return r;
}
VKAPI_ATTR VkResult VKAPI_CALL hook_GetSemaphoreCounterValue(VkDevice device, VkSemaphore sem, uint64_t* v) {
  DeviceData* dd = Dev(device);
  return CounterCommon(dd, device, sem, v, dd->d.GetSemaphoreCounterValue);
}
VKAPI_ATTR VkResult VKAPI_CALL hook_GetSemaphoreCounterValueKHR(VkDevice device, VkSemaphore sem, uint64_t* v) {
  DeviceData* dd = Dev(device);
  return CounterCommon(dd, device, sem, v, dd->d.GetSemaphoreCounterValueKHR);
}

VKAPI_ATTR VkResult VKAPI_CALL hook_CreateSemaphore(VkDevice device, const VkSemaphoreCreateInfo* ci,
                                                    const VkAllocationCallbacks* a, VkSemaphore* sem) {
  DeviceData* dd = Dev(device);
  VkResult r = dd->d.CreateSemaphore(device, ci, a, sem);
  if (r == VK_SUCCESS)
    for (auto* p = (const VkBaseInStructure*)ci->pNext; p; p = p->pNext)
      if (p->sType == VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO &&
          ((const VkSemaphoreTypeCreateInfo*)p)->semaphoreType == VK_SEMAPHORE_TYPE_TIMELINE) {
        std::lock_guard<std::mutex> l(dd->semLock);
        dd->timelineSems.insert(*sem);
      }
  return r;
}
VKAPI_ATTR void VKAPI_CALL hook_DestroySemaphore(VkDevice device, VkSemaphore sem, const VkAllocationCallbacks* a) {
  DeviceData* dd = Dev(device);
  {
    std::lock_guard<std::mutex> l(dd->semLock);
    dd->timelineSems.erase(sem);
  }
  dd->d.DestroySemaphore(device, sem, a);
}

VKAPI_ATTR VkResult VKAPI_CALL hook_BeginCommandBuffer(VkCommandBuffer cb, const VkCommandBufferBeginInfo* bi) {
  DeviceData* dd = Dev(cb);
  {
    std::lock_guard<std::mutex> l(dd->passLock);
    dd->monoCbs.erase(cb);
  }
  if (!dd->replay) return dd->d.BeginCommandBuffer(cb, bi);
  WaitReplayDone(dd, &cb, 1, false);
  VkCommandBufferBeginInfo b = *bi;  // the frame is submitted twice
  b.flags &= ~VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  b.flags |= VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;
  return dd->d.BeginCommandBuffer(cb, &b);
}

VKAPI_ATTR VkResult VKAPI_CALL hook_ResetCommandBuffer(VkCommandBuffer cb, VkCommandBufferResetFlags f) {
  DeviceData* dd = Dev(cb);
  WaitReplayDone(dd, &cb, 1, false);
  return dd->d.ResetCommandBuffer(cb, f);
}
VKAPI_ATTR VkResult VKAPI_CALL hook_AllocateCommandBuffers(VkDevice device, const VkCommandBufferAllocateInfo* ai,
                                                           VkCommandBuffer* cbs) {
  DeviceData* dd = Dev(device);
  VkResult r = dd->d.AllocateCommandBuffers(device, ai, cbs);
  if (r == VK_SUCCESS) {
    std::lock_guard<std::mutex> l(dd->cbLock);
    auto& set = dd->poolCbs[ai->commandPool];
    for (uint32_t i = 0; i < ai->commandBufferCount; ++i) {
      dd->cbPool[cbs[i]] = ai->commandPool;
      set.insert(cbs[i]);
    }
  }
  return r;
}
std::vector<VkCommandBuffer> PoolCbs(DeviceData* dd, VkCommandPool pool) {
  std::lock_guard<std::mutex> l(dd->cbLock);
  auto it = dd->poolCbs.find(pool);
  if (it == dd->poolCbs.end()) return {};
  return std::vector<VkCommandBuffer>(it->second.begin(), it->second.end());
}
VKAPI_ATTR void VKAPI_CALL hook_FreeCommandBuffers(VkDevice device, VkCommandPool pool, uint32_t n,
                                                   const VkCommandBuffer* cbs) {
  DeviceData* dd = Dev(device);
  WaitReplayDone(dd, cbs, n, false);
  {
    std::lock_guard<std::mutex> l(dd->cbLock);
    auto it = dd->poolCbs.find(pool);
    for (uint32_t i = 0; i < n; ++i) {
      dd->cbPool.erase(cbs[i]);
      if (it != dd->poolCbs.end()) it->second.erase(cbs[i]);
    }
  }
  dd->d.FreeCommandBuffers(device, pool, n, cbs);
}
VKAPI_ATTR VkResult VKAPI_CALL hook_ResetCommandPool(VkDevice device, VkCommandPool pool, VkCommandPoolResetFlags f) {
  DeviceData* dd = Dev(device);
  std::vector<VkCommandBuffer> v = PoolCbs(dd, pool);
  WaitReplayDone(dd, v.data(), uint32_t(v.size()), false);
  return dd->d.ResetCommandPool(device, pool, f);
}
VKAPI_ATTR void VKAPI_CALL hook_DestroyCommandPool(VkDevice device, VkCommandPool pool, const VkAllocationCallbacks* a) {
  DeviceData* dd = Dev(device);
  std::vector<VkCommandBuffer> v = PoolCbs(dd, pool);
  WaitReplayDone(dd, v.data(), uint32_t(v.size()), false);
  {
    std::lock_guard<std::mutex> l(dd->cbLock);
    for (VkCommandBuffer cb : v) dd->cbPool.erase(cb);
    dd->poolCbs.erase(pool);
  }
  dd->d.DestroyCommandPool(device, pool, a);
}
// Objects destroyed while a relaxed frame still has to be replayed: destroyed later, once the
// replay that may use them has finished (never blocks the game).
void RunDeferred(DeviceData* dd, bool all) {
  std::vector<std::function<void()>> run;
  {
    std::lock_guard<std::mutex> l(dd->cbLock);
    if (dd->deferred.empty()) return;
    std::vector<DeviceData::Deferred> keep;
    for (auto& d : dd->deferred) {
      bool ready = all;
      if (!ready && d.serial <= dd->submittedSerial) {
        int k = int(d.serial % 8);
        ready = dd->serialOf[k] != d.serial || dd->d.GetFenceStatus(dd->device, dd->serialFence[k]) == VK_SUCCESS;
      }
      if (ready) run.push_back(std::move(d.fn));
      else keep.push_back(std::move(d));
    }
    dd->deferred.swap(keep);
  }
  for (auto& f : run) f();
}
bool DeferIfRelaxed(DeviceData* dd, std::function<void()> fn) {
  if (!dd->replay) return false;
  {
    std::lock_guard<std::mutex> l(dd->cbLock);
    uint64_t r = dd->relaxedSerial;
    bool pending = false;
    if (r && r >= dd->frameSerial) pending = true;  // this frame was relaxed and is not replayed yet
    else if (r && r <= dd->submittedSerial) {
      int k = int(r % 8);
      pending = dd->serialOf[k] == r && dd->d.GetFenceStatus(dd->device, dd->serialFence[k]) != VK_SUCCESS;
    }
    if (pending) {
      dd->deferred.push_back({r, std::move(fn)});
      return true;
    }
  }
  return false;
}
#define DESTROY_HOOK(Name, T)                                                                  \
  VKAPI_ATTR void VKAPI_CALL hook_##Name(VkDevice device, T h, const VkAllocationCallbacks* a) { \
    DeviceData* dd = Dev(device);                                                              \
    if (h && DeferIfRelaxed(dd, [dd, device, h, a]() { dd->d.Name(device, h, a); })) return;   \
    dd->d.Name(device, h, a);                                                                  \
  }
DESTROY_HOOK(DestroyBuffer, VkBuffer)
DESTROY_HOOK(DestroyImage, VkImage)
DESTROY_HOOK(FreeMemory, VkDeviceMemory)
DESTROY_HOOK(DestroyImageView, VkImageView)
DESTROY_HOOK(DestroyBufferView, VkBufferView)
DESTROY_HOOK(DestroySampler, VkSampler)
DESTROY_HOOK(DestroyPipeline, VkPipeline)
DESTROY_HOOK(DestroyFramebuffer, VkFramebuffer)
DESTROY_HOOK(DestroyQueryPool, VkQueryPool)
DESTROY_HOOK(DestroyDescriptorPool, VkDescriptorPool)
DESTROY_HOOK(DestroyEvent, VkEvent)
#undef DESTROY_HOOK
VKAPI_ATTR VkResult VKAPI_CALL hook_ResetDescriptorPool(VkDevice device, VkDescriptorPool p, VkDescriptorPoolResetFlags f) {
  DeviceData* dd = Dev(device);
  WaitReplayDone(dd, nullptr, 0, true);
  return dd->d.ResetDescriptorPool(device, p, f);
}
VKAPI_ATTR VkResult VKAPI_CALL hook_FreeDescriptorSets(VkDevice device, VkDescriptorPool p, uint32_t n,
                                                       const VkDescriptorSet* sets) {
  DeviceData* dd = Dev(device);
  WaitReplayDone(dd, nullptr, 0, true);
  return dd->d.FreeDescriptorSets(device, p, n, sets);
}

// ------------------------------------------------------------------------------------
// Shader patching
// ------------------------------------------------------------------------------------
VKAPI_ATTR VkResult VKAPI_CALL hook_CreateShaderModule(VkDevice device, const VkShaderModuleCreateInfo* ci,
                                                       const VkAllocationCallbacks* a, VkShaderModule* m) {
  DeviceData* dd = Dev(device);
  VkResult r = dd->d.CreateShaderModule(device, ci, a, m);
  if (r == VK_SUCCESS && dd->stereoShaders && ci->codeSize % 4 == 0 &&
      HasPreRasterEntryPoint(ci->pCode, ci->codeSize / 4)) {
    std::lock_guard<std::mutex> l(dd->shaderLock);
    dd->modules[*m].code.assign(ci->pCode, ci->pCode + ci->codeSize / 4);
  }
  return r;
}

VKAPI_ATTR void VKAPI_CALL hook_DestroyShaderModule(VkDevice device, VkShaderModule m, const VkAllocationCallbacks* a) {
  DeviceData* dd = Dev(device);
  {
    std::lock_guard<std::mutex> l(dd->shaderLock);
    auto it = dd->modules.find(m);
    if (it != dd->modules.end()) {
      for (auto& v : it->second.variants)
        if (v.patched) dd->d.DestroyShaderModule(device, v.patched, nullptr);
      dd->modules.erase(it);
    }
  }
  dd->d.DestroyShaderModule(device, m, a);
}

uint32_t StageToModel(VkShaderStageFlagBits s) {
  switch (s) {
    case VK_SHADER_STAGE_VERTEX_BIT: return 0;
    case VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT: return 2;
    case VK_SHADER_STAGE_GEOMETRY_BIT: return 3;
    default: return UINT32_MAX;
  }
}

bool PatchCode(DeviceData* dd, const uint32_t* code, size_t words, uint32_t model, const char* entry,
               std::vector<uint32_t>& out) {
  std::string log;
  PatchResult pr = PatchStereo(code, words, entry, model, dd->paramAddr, out, &log);
  if (pr == PatchResult::Patched) {
    uint32_t n = ++dd->patchedCount;
    Log(2, "patched shader #%u (model %u, %s)", n, model, log.c_str());
    return true;
  }
  ++dd->skippedCount;
  Log(2, "shader not patched: %s (model %u)", PatchResultName(pr), model);
  return false;
}

VkShaderModule GetPatchedModule(DeviceData* dd, VkShaderModule m, uint32_t model, const char* entry) {
  std::lock_guard<std::mutex> l(dd->shaderLock);
  auto it = dd->modules.find(m);
  if (it == dd->modules.end()) return VK_NULL_HANDLE;
  for (auto& v : it->second.variants)
    if (v.model == model && v.entry == entry) return v.patched;
  std::vector<uint32_t> out;
  VkShaderModule pm = VK_NULL_HANDLE;
  if (PatchCode(dd, it->second.code.data(), it->second.code.size(), model, entry, out)) {
    VkShaderModuleCreateInfo ci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    ci.codeSize = out.size() * 4;
    ci.pCode = out.data();
    if (dd->d.CreateShaderModule(dd->device, &ci, nullptr, &pm) != VK_SUCCESS) pm = VK_NULL_HANDLE;
  }
  it->second.variants.push_back({model, entry, pm});
  return pm;
}

// Viewport sign tells the clip-space Y convention (needed to rotate the camera with the head)
inline void CountViewports(DeviceData* dd, uint32_t n, const VkViewport* v) {
  if (!dd || !v) return;
  for (uint32_t i = 0; i < n; ++i) {
    if (v[i].height <= -200.0f) dd->vpNeg.fetch_add(1, std::memory_order_relaxed);
    else if (v[i].height >= 200.0f) dd->vpPos.fetch_add(1, std::memory_order_relaxed);
  }
}
// ---- off-screen render passes ----
// Shadow maps, reflection / environment cubes and other views that are not the game camera must
// not get the eye transform (shadows would land in the wrong place, differently in each eye).
// They are recognised by a render area whose shape differs from the game frame (square shadow
// atlases...). Before such a pass the identity parameters are copied into the slot the shaders
// read, and the eye's parameters are copied back before the next camera pass / at the end of the
// command buffer. Copies are recorded in the game's command buffer, so they also run in the replay.
bool IsOffscreenArea(DeviceData* dd, VkExtent2D e) {
  if (!Cfg().offscreenMono || !dd->stereoShaders) return false;
  if (e.width < 64 || e.height < 64) return false;  // small post-process targets: full-screen passes anyway
  uint32_t gw = dd->gameW.load(std::memory_order_relaxed), gh = dd->gameH.load(std::memory_order_relaxed);
  if (e.width == gw && e.height == gh) return false;
  // shadow maps / atlases and cube faces: 1:1, 2:1 or 4:1 (the game frame may be any size: DXVK
  // renders at the resolution chosen in the game, not the window's)
  float r = float(std::max(e.width, e.height)) / float(std::min(e.width, e.height));
  bool shadowShape = std::fabs(r - 1.0f) < 0.01f || std::fabs(r - 2.0f) < 0.02f || std::fabs(r - 4.0f) < 0.04f;
  if (!shadowShape) return false;
  if (gw && gh) {
    float ga = float(gw) / float(gh), a = float(e.width) / float(e.height);
    if (std::fabs(a / ga - 1.0f) < 0.04f) return false;
  }
  return true;
}
void CopyParams(DeviceData* dd, VkCommandBuffer cb, VkDeviceSize from) {
  VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
  mb.srcAccessMask = VK_ACCESS_MEMORY_READ_BIT;
  mb.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  dd->d.CmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &mb, 0,
                           nullptr, 0, nullptr);
  VkBufferCopy r{from, 0, sizeof(EyeParams::m)};
  dd->d.CmdCopyBuffer(cb, dd->paramBuf, dd->paramBuf, 1, &r);
  mb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_MEMORY_READ_BIT;
  dd->d.CmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &mb, 0,
                           nullptr, 0, nullptr);
}
// called outside any render pass
void BeforePass(DeviceData* dd, VkCommandBuffer cb, VkExtent2D e) {
  if (!dd->stereoShaders || !dd->d.CmdCopyBuffer) return;
  bool off = IsOffscreenArea(dd, e);
  if (Tracing()) Log(0, "pass %ux%u cb %p off %d", e.width, e.height, (void*)cb, off);
  bool isMono;
  {
    std::lock_guard<std::mutex> l(dd->passLock);
    isMono = dd->monoCbs.count(cb) != 0;
    if (off == isMono) return;
    if (off) dd->monoCbs.insert(cb);
    else dd->monoCbs.erase(cb);
  }
  if (off) ++dd->monoPasses;
  CopyParams(dd, cb, off ? 512 : 256);
}
VKAPI_ATTR void VKAPI_CALL hook_CmdBeginRenderPass(VkCommandBuffer cb, const VkRenderPassBeginInfo* b,
                                                   VkSubpassContents c) {
  DeviceData* dd = DevFast(cb);
  BeforePass(dd, cb, b->renderArea.extent);
  dd->d.CmdBeginRenderPass(cb, b, c);
}
VKAPI_ATTR void VKAPI_CALL hook_CmdBeginRenderPass2(VkCommandBuffer cb, const VkRenderPassBeginInfo* b,
                                                    const VkSubpassBeginInfo* s) {
  DeviceData* dd = DevFast(cb);
  BeforePass(dd, cb, b->renderArea.extent);
  dd->d.CmdBeginRenderPass2(cb, b, s);
}
VKAPI_ATTR void VKAPI_CALL hook_CmdBeginRenderPass2KHR(VkCommandBuffer cb, const VkRenderPassBeginInfo* b,
                                                       const VkSubpassBeginInfo* s) {
  DeviceData* dd = DevFast(cb);
  BeforePass(dd, cb, b->renderArea.extent);
  dd->d.CmdBeginRenderPass2KHR(cb, b, s);
}
VKAPI_ATTR void VKAPI_CALL hook_CmdBeginRendering(VkCommandBuffer cb, const VkRenderingInfo* ri) {
  DeviceData* dd = DevFast(cb);
  if (!(ri->flags & VK_RENDERING_RESUMING_BIT)) BeforePass(dd, cb, ri->renderArea.extent);
  dd->d.CmdBeginRendering(cb, ri);
}
VKAPI_ATTR void VKAPI_CALL hook_CmdBeginRenderingKHR(VkCommandBuffer cb, const VkRenderingInfo* ri) {
  DeviceData* dd = DevFast(cb);
  if (!(ri->flags & VK_RENDERING_RESUMING_BIT)) BeforePass(dd, cb, ri->renderArea.extent);
  dd->d.CmdBeginRenderingKHR(cb, ri);
}
VKAPI_ATTR VkResult VKAPI_CALL hook_EndCommandBuffer(VkCommandBuffer cb) {
  DeviceData* dd = DevFast(cb);
  bool restore;
  {
    std::lock_guard<std::mutex> l(dd->passLock);
    restore = dd->monoCbs.erase(cb) != 0;
  }
  if (restore) CopyParams(dd, cb, 256);  // the next command buffer starts with the eye's params
  return dd->d.EndCommandBuffer(cb);
}

VKAPI_ATTR void VKAPI_CALL hook_CmdSetViewport(VkCommandBuffer cb, uint32_t first, uint32_t n, const VkViewport* v) {
  DeviceData* dd = DevFast(cb);
  CountViewports(dd, n, v);
  dd->d.CmdSetViewport(cb, first, n, v);
}
VKAPI_ATTR void VKAPI_CALL hook_CmdSetViewportWithCount(VkCommandBuffer cb, uint32_t n, const VkViewport* v) {
  DeviceData* dd = DevFast(cb);
  CountViewports(dd, n, v);
  dd->d.CmdSetViewportWithCount(cb, n, v);
}
VKAPI_ATTR void VKAPI_CALL hook_CmdSetViewportWithCountEXT(VkCommandBuffer cb, uint32_t n, const VkViewport* v) {
  DeviceData* dd = DevFast(cb);
  CountViewports(dd, n, v);
  dd->d.CmdSetViewportWithCountEXT(cb, n, v);
}

VKAPI_ATTR VkResult VKAPI_CALL hook_CreateGraphicsPipelines(VkDevice device, VkPipelineCache cache, uint32_t count,
                                                            const VkGraphicsPipelineCreateInfo* infos,
                                                            const VkAllocationCallbacks* a, VkPipeline* out) {
  DeviceData* dd = Dev(device);
  if (!dd->stereoShaders) return dd->d.CreateGraphicsPipelines(device, cache, count, infos, a, out);

  for (uint32_t i = 0; i < count; ++i)
    if (infos[i].pViewportState && infos[i].pViewportState->pViewports)
      CountViewports(dd, infos[i].pViewportState->viewportCount, infos[i].pViewportState->pViewports);
  std::vector<VkGraphicsPipelineCreateInfo> ci(infos, infos + count);
  std::vector<std::vector<VkPipelineShaderStageCreateInfo>> stages(count);
  struct Restore { VkShaderModuleCreateInfo* smci; const uint32_t* code; size_t size; };
  std::vector<Restore> restores;
  std::vector<std::unique_ptr<std::vector<uint32_t>>> inlineCode;

  for (uint32_t i = 0; i < count; ++i) {
    if (!ci[i].pStages || !ci[i].stageCount) continue;
    int best = -1, bestRank = -1;
    for (uint32_t s = 0; s < ci[i].stageCount; ++s) {
      int rank = -1;
      switch (ci[i].pStages[s].stage) {
        case VK_SHADER_STAGE_VERTEX_BIT: rank = 0; break;
        case VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT: rank = 1; break;
        case VK_SHADER_STAGE_GEOMETRY_BIT: rank = 2; break;
        default: break;
      }
      if (rank > bestRank) { bestRank = rank; best = int(s); }
    }
    if (best < 0) continue;
    ++dd->pipelineCount;
    const VkPipelineShaderStageCreateInfo& st = ci[i].pStages[best];
    uint32_t model = StageToModel(st.stage);
    if (st.module != VK_NULL_HANDLE) {
      VkShaderModule pm = GetPatchedModule(dd, st.module, model, st.pName);
      if (pm) {
        stages[i].assign(ci[i].pStages, ci[i].pStages + ci[i].stageCount);
        stages[i][best].module = pm;
        ci[i].pStages = stages[i].data();
      }
    } else {
      // VK_KHR_maintenance5: shader code chained inline
      for (auto* p = (VkBaseOutStructure*)st.pNext; p; p = p->pNext) {
        if (p->sType != VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO) continue;
        auto* smci = (VkShaderModuleCreateInfo*)p;
        auto buf = std::make_unique<std::vector<uint32_t>>();
        if (PatchCode(dd, smci->pCode, smci->codeSize / 4, model, st.pName, *buf)) {
          restores.push_back({smci, smci->pCode, smci->codeSize});
          smci->pCode = buf->data();
          smci->codeSize = buf->size() * 4;
          inlineCode.push_back(std::move(buf));
        }
        break;
      }
    }
  }
  VkResult r = dd->d.CreateGraphicsPipelines(device, cache, count, ci.data(), a, out);
  for (auto& rs : restores) {
    rs.smci->pCode = rs.code;
    rs.smci->codeSize = rs.size;
  }
  return r;
}

// ------------------------------------------------------------------------------------
// Head pose -> per-eye shader parameters
// ------------------------------------------------------------------------------------
constexpr float kPi = 3.14159265358979f;

void EyeFov(const Swapchain* sc, float& hf, float& vf) {
  const Config& c = Cfg();
  float W = float(sc->vext.width), H = float(std::max(1u, sc->vext.height));
  float f = c.fov * kPi / 180.0f;
  if (c.fovAxis == 0) {
    hf = f;
    vf = 2.0f * std::atan(std::tan(hf * 0.5f) * H / W);
  } else {
    vf = f;
    hf = 2.0f * std::atan(std::tan(vf * 0.5f) * W / H);
  }
}

// +1: clip-space Y points up (D3D through DXVK/VKD3D: negative viewport height), -1: down
float ClipYSign(DeviceData* dd) {
  int f = Cfg().flipY;
  if (f == 1) return -1.0f;
  if (f == 0) return 1.0f;
  uint32_t n = dd->vpNeg.load(std::memory_order_relaxed), p = dd->vpPos.load(std::memory_order_relaxed);
  return p > n ? -1.0f : 1.0f;
}

// Pose sampled now, used to render the next game frame.
FrameView ComputeView(DeviceData* dd, const Swapchain* sc) {
  FrameView v{};
  const Config& c = Cfg();
  v.p[0] = v.p[1] = IdentityParams();
  float hf, vf;
  EyeFov(sc, hf, vf);
  v.fov[0] = -hf * 0.5f; v.fov[1] = hf * 0.5f; v.fov[2] = vf * 0.5f; v.fov[3] = -vf * 0.5f;
  for (int e = 0; e < 2; ++e) {
    v.camQ[e][0] = v.camQ[e][1] = v.camQ[e][2] = 0.0f;
    v.camQ[e][3] = 1.0f;
  }
  auto& link = Link();
  vrlinux::HostPose hp{};
  bool have = link.Connected() && link.Shm()->sessionRunning.load() && link.ReadPose(hp);
  if (!have) return v;
  v.vr = true;
  // the game may apply the mouse a few frames late: show the image with the camera it really had
  dd->camHist[dd->camHistPos % 8][0] = hp.camQ[0];
  dd->camHist[dd->camHistPos % 8][1] = hp.camQ[1];
  dd->camHist[dd->camHistPos % 8][2] = hp.camQ[2];
  dd->camHist[dd->camHistPos % 8][3] = hp.camQ[3];
  ++dd->camHistPos;
  const float* camQ = hp.camQ;
  if (c.lookLatency > 0 && dd->camHistPos > uint32_t(c.lookLatency))
    camQ = dd->camHist[(dd->camHistPos - 1 - uint32_t(c.lookLatency)) % 8];
  for (int e = 0; e < 2; ++e) {
    memcpy(v.camQ[e], camQ, sizeof(hp.camQ));
    memcpy(v.eyePos[e], hp.eyePos[e], sizeof(hp.eyePos[e]));
  }
  if (c.display == 1) {  // giant 3D screen: classic stereo (separation / convergence)
    for (int e = 0; e < 2; ++e) {
      float k = (e ? 1.0f : -1.0f) * c.separation;
      if (c.swapEyes) k = -k;
      if (!c.stereo) k = 0.0f;
      v.p[e].m[2] = k;
      v.p[e].m[3] = -k * c.convergence;
      v.p[e].m[12] = k * HudFactor();
    }
    return v;
  }
  const float sx = 1.0f / std::tan(hf * 0.5f), sy = 1.0f / std::tan(vf * 0.5f);
  const float ys = ClipYSign(dd), U = c.worldScale;
  vrlinux::V3 cam[2], head[2];
  for (int e = 0; e < 2; ++e) {
    cam[e] = {hp.eyeCam[e][0], hp.eyeCam[e][1], hp.eyeCam[e][2]};
    head[e] = {hp.eyeHead[e][0], hp.eyeHead[e][1], hp.eyeHead[e][2]};
  }
  if (!c.stereo) {  // both eyes from the centre of the head
    cam[0] = cam[1] = (cam[0] + cam[1]) * 0.5f;
    head[0] = head[1] = (head[0] + head[1]) * 0.5f;
  }
  const bool shader = c.look == 2 && hp.eyeFov[0][1] > hp.eyeFov[0][0];
  for (int e = 0; e < 2; ++e) {
    int s = c.swapEyes ? 1 - e : e;
    EyeParams& p = v.p[e];
    if (!shader) {  // camera moved only: image with the game's own FOV
      p.m[3] = -sx * U * cam[s].x;   // camera right  -> scene left
      p.m[7] = -ys * sy * U * cam[s].y;
      p.m[11] = U * cam[s].z;         // camera back   -> scene farther
      if (c.hudDistance > 0.0f) {
        p.m[12] = -sx * head[s].x / c.hudDistance;
        p.m[13] = -ys * sy * head[s].y / c.hudDistance;
      }
      p.m[14] = p.m[15] = c.hudScale;
      continue;
    }
    // ---- shader look: the game camera is re-projected to the eye's real orientation and to the
    // headset's field of view:  clip' = P_eye * R^T * (unproject_game(clip) - U*t)
    vrlinux::Q rel = c.headLook ? vrlinux::Q{hp.headRelQ[0], hp.headRelQ[1], hp.headRelQ[2], hp.headRelQ[3]}
                                : vrlinux::Q{};
    // beyond look_limit the game has nothing drawn: stop turning, the image just stays where it is
    if (c.lookLimit > 0.0f) {
      rel = vrlinux::Norm(rel);
      if (rel.w < 0) rel = {-rel.x, -rel.y, -rel.z, -rel.w};
      float ang = 2.0f * std::acos(std::min(1.0f, rel.w));
      float lim = c.lookLimit * kPi / 180.0f;
      if (ang > lim) {
        float sn = std::sqrt(std::max(1e-12f, 1.0f - rel.w * rel.w));
        float k = std::sin(lim * 0.5f) / sn;
        rel = {rel.x * k, rel.y * k, rel.z * k, std::cos(lim * 0.5f)};
      }
    }
    vrlinux::Q eyeQ = vrlinux::Norm(vrlinux::Mul(rel, vrlinux::Q{hp.eyeHeadQ[s][0], hp.eyeHeadQ[s][1],
                                                                 hp.eyeHeadQ[s][2], hp.eyeHeadQ[s][3]}));
    float R[3][3];
    vrlinux::ToMat(eyeQ, R);
    const float os = c.overscan;
    float L = std::tan(hp.eyeFov[s][0]) * os, Rt = std::tan(hp.eyeFov[s][1]) * os;
    float Up = std::tan(hp.eyeFov[s][2]) * os, Dn = std::tan(hp.eyeFov[s][3]) * os;
    // E: (x, y, w, 1) -> camera space (OpenXR axes, metres*U): (x/sx, ys*y/sy, -w) - U*t
    float E[3][4] = {{1.0f / sx, 0, 0, -U * cam[s].x}, {0, ys / sy, 0, -U * cam[s].y}, {0, 0, -1.0f, -U * cam[s].z}};
    // A = R^T * E
    float A[3][4];
    for (int r = 0; r < 3; ++r)
      for (int k = 0; k < 4; ++k) A[r][k] = R[0][r] * E[0][k] + R[1][r] * E[1][k] + R[2][r] * E[2][k];
    // P: x_c = a X + b Z, y_c = ys (cY + dZ), w_c = -Z
    const float a = 2.0f / (Rt - L), b = (Rt + L) / (Rt - L), cc = 2.0f / (Up - Dn), d = (Up + Dn) / (Up - Dn);
    for (int k = 0; k < 4; ++k) {
      p.m[k] = a * A[0][k] + b * A[2][k];
      p.m[4 + k] = ys * (cc * A[1][k] + d * A[2][k]);
      p.m[8 + k] = -A[2][k];
    }
    // HUD: head-locked, the size the game screen had, in the middle of the headset view
    float shiftX = 0, shiftY = 0;
    if (c.hudDistance > 0.0f) {
      shiftX = -head[s].x / c.hudDistance;
      shiftY = -head[s].y / c.hudDistance;
    }
    const float tgh = 1.0f / sx, tgv = 1.0f / sy;
    p.m[14] = a * tgh * c.hudScale;
    p.m[12] = a * shiftX - b;
    p.m[15] = cc * tgv * c.hudScale;
    p.m[13] = ys * (cc * shiftY - d);
    p.m[16] = c.depthClamp ? 1.0f : 0.0f;
    // HUD fixed with the game camera (default): drawn as a plane at hud_distance (or far) in front of it
    if (!c.hudFollow) {
      float D = (c.hudDistance > 0.0f ? c.hudDistance : 50.0f) * U;
      p.m[17] = 1.0f;
      p.m[18] = D;
      p.m[19] = D * c.hudScale;
    }
    // the frame is shown with the orientation and FOV it was rendered with
    vrlinux::Q cq{camQ[0], camQ[1], camQ[2], camQ[3]};
    vrlinux::Q world = vrlinux::Norm(vrlinux::Mul(cq, eyeQ));
    v.camQ[e][0] = world.x; v.camQ[e][1] = world.y; v.camQ[e][2] = world.z; v.camQ[e][3] = world.w;
    v.efov[e][0] = std::atan(L); v.efov[e][1] = std::atan(Rt); v.efov[e][2] = std::atan(Up); v.efov[e][3] = std::atan(Dn);
    v.perEyeFov = true;
  }
  return v;
}

void WriteSettings() {
  auto& link = Link();
  vrlinux::SharedState* s = link.Shm();
  if (!s) return;
  const Config& c = Cfg();
  vrlinux::LayerSettings& o = s->settings;
  o.display = c.display;
  o.look = c.look;
  o.headLook = c.headLook;
  o.lookPitch = c.lookPitch;
  o.mouseSens = c.mouseSens;
  o.pitchLimit = c.pitchLimit;
  o.invertY = c.invertY;
  o.positional = c.positional;
  o.posLimit = c.posLimit;
  o.screenDist = c.screenDist;
  o.screenWidth = c.screenWidth;
  o.fovH = c.fovAxis == 0 ? c.fov : 0.0f;
  o.fovV = c.fovAxis == 1 ? c.fov : 0.0f;
  o.recenter = RecenterRequests();
  s->layerAlive.fetch_add(1);
}

// ------------------------------------------------------------------------------------
// Frames exported to vrlinux-host
// ------------------------------------------------------------------------------------
void BarrierQF(DeviceData* dd, VkCommandBuffer cb, VkImage img, VkImageLayout from, VkImageLayout to,
               VkPipelineStageFlags srcStage, VkAccessFlags srcAccess, VkPipelineStageFlags dstStage,
               VkAccessFlags dstAccess, uint32_t srcQF, uint32_t dstQF) {
  VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
  b.srcAccessMask = srcAccess;
  b.dstAccessMask = dstAccess;
  b.oldLayout = from;
  b.newLayout = to;
  b.srcQueueFamilyIndex = srcQF;
  b.dstQueueFamilyIndex = dstQF;
  b.image = img;
  b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
  dd->d.CmdPipelineBarrier(cb, srcStage, dstStage, 0, 0, nullptr, 0, nullptr, 1, &b);
}

// 8-bit format with the same channel order: copied raw (sRGB-encoded bytes stay as they are)
VkFormat SlotFormat(VkFormat game, bool& raw) {
  raw = true;
  switch (game) {
    case VK_FORMAT_B8G8R8A8_UNORM: case VK_FORMAT_B8G8R8A8_SRGB: return VK_FORMAT_B8G8R8A8_UNORM;
    case VK_FORMAT_R8G8B8A8_UNORM: case VK_FORMAT_R8G8B8A8_SRGB: return VK_FORMAT_R8G8B8A8_UNORM;
    case VK_FORMAT_A8B8G8R8_UNORM_PACK32: case VK_FORMAT_A8B8G8R8_SRGB_PACK32: return VK_FORMAT_R8G8B8A8_UNORM;
    default: raw = false; return VK_FORMAT_R8G8B8A8_UNORM;  // 10-bit / float: blit
  }
}

void FreeSlot(DeviceData* dd, VrSlot& s) {
  if (s.fence) dd->d.DestroyFence(dd->device, s.fence, nullptr);
  if (s.image) dd->d.DestroyImage(dd->device, s.image, nullptr);
  if (s.mem) dd->d.FreeMemory(dd->device, s.mem, nullptr);
  s = VrSlot{};
}

void DestroyExport(DeviceData* dd, VrExport* ex) {
  if (!ex) return;
  {
    std::unique_lock<std::mutex> l(dd->pubLock);
    dd->pubQueue.erase(std::remove_if(dd->pubQueue.begin(), dd->pubQueue.end(),
                                      [&](const std::pair<VrExport*, int>& j) { return j.first == ex; }),
                       dd->pubQueue.end());
    dd->pubCv.wait(l, [&]() { return dd->pubCurrent != ex; });
  }
  for (auto& s : ex->slots) {
    if (s.fence) dd->d.WaitForFences(dd->device, 1, &s.fence, VK_TRUE, 2000000000ull);
    FreeSlot(dd, s);
  }
  if (dd->activeEx == ex) dd->activeEx = nullptr;
  delete ex;
}

VrExport* CreateExport(DeviceData* dd, Swapchain* sc) {
  auto* ex = new VrExport();
  ex->w = sc->vext.width;
  ex->h = sc->vext.height;
  ex->format = SlotFormat(sc->format, ex->rawCopy);
  if (!ex->rawCopy && !sc->canBlit) {
    Log(0, "VR: swapchain format %d cannot be converted (no blit support)", sc->format);
    delete ex;
    return nullptr;
  }
  ex->usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
  bool ok = true;
  for (auto& s : ex->slots) {
    VkExternalMemoryImageCreateInfo emi{VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO};
    emi.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
    VkImageCreateInfo ici{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, &emi};
    ici.imageType = VK_IMAGE_TYPE_2D;
    ici.format = ex->format;
    ici.extent = {ex->w * 2, ex->h, 1};
    ici.mipLevels = 1;
    ici.arrayLayers = 1;
    ici.samples = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling = VK_IMAGE_TILING_OPTIMAL;
    ici.usage = ex->usage;
    ici.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    if (dd->d.CreateImage(dd->device, &ici, nullptr, &s.image) != VK_SUCCESS) { ok = false; break; }
    VkMemoryRequirements req;
    dd->d.GetImageMemoryRequirements(dd->device, s.image, &req);
    VkMemoryDedicatedAllocateInfo ded{VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO};
    ded.image = s.image;
    VkExportMemoryAllocateInfo exi{VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO, &ded};
    exi.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, &exi};
    ai.allocationSize = req.size;
    ai.memoryTypeIndex = FindMemType(dd, req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (dd->d.AllocateMemory(dd->device, &ai, nullptr, &s.mem) != VK_SUCCESS) { ok = false; break; }
    if (dd->d.BindImageMemory(dd->device, s.image, s.mem, 0) != VK_SUCCESS) { ok = false; break; }
    s.size = req.size;
    VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    fci.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    if (dd->d.CreateFence(dd->device, &fci, nullptr, &s.fence) != VK_SUCCESS) { ok = false; break; }
  }
  if (!ok) {
    Log(0, "VR: cannot create the exported eye images");
    for (auto& s : ex->slots) FreeSlot(dd, s);
    delete ex;
    return nullptr;
  }
  Log(1, "VR: exporting %ux%u per eye (format %d -> %d, %s)", ex->w, ex->h, sc->format, ex->format,
      ex->rawCopy ? "copy" : "blit");
  return ex;
}

// Makes sure this swapchain's frames can go to the host; (re)sends the hello when needed.
VrExport* EnsureExport(DeviceData* dd, Swapchain* sc) {
  if (!dd->canExport || !Link().Connected()) return nullptr;
  if (!sc->ex) {
    if (sc->exportFailed) return nullptr;
    sc->ex = CreateExport(dd, sc);
    if (!sc->ex) { sc->exportFailed = true; return nullptr; }
  }
  VrExport* ex = sc->ex;
  if (dd->activeEx != ex) ex->helloGen = 0;  // another swapchain was shown before
  if (ex->helloGen != Link().Generation()) {
    vrlinux::MsgHello h;
    h.width = ex->w;
    h.height = ex->h;
    h.format = ex->format;
    h.imageFlags = 0;
    h.imageUsage = ex->usage;
    h.slots = vrlinux::kSlots;
    memcpy(h.deviceUUID, dd->deviceUUID, 16);
    memcpy(h.driverUUID, dd->driverUUID, 16);
    h.pid = getpid();
    snprintf(h.game, sizeof(h.game), "%s", GameName().c_str());
    std::vector<int> fds;
    bool ok = true;
    for (int i = 0; i < vrlinux::kSlots; ++i) {
      h.allocSize[i] = ex->slots[i].size;
      VkMemoryGetFdInfoKHR gi{VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR};
      gi.memory = ex->slots[i].mem;
      gi.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
      int fd = -1;
      if (dd->d.GetMemoryFdKHR(dd->device, &gi, &fd) != VK_SUCCESS || fd < 0) ok = false;
      else fds.push_back(fd);
    }
    if (ok) ok = Link().SendHello(h, fds);
    for (int fd : fds) close(fd);
    if (!ok) {
      Log(0, "VR: could not send the eye images to the host");
      return nullptr;
    }
    std::lock_guard<std::mutex> l(dd->exportLock);
    for (auto& s : ex->slots)
      if (s.state == 3) s.state = 0;
    ex->writing = -1;
    ex->helloGen = Link().Generation();
    dd->activeEx = ex;
    Log(1, "VR: eye images sent to the host (%ux%u per eye)", ex->w, ex->h);
  }
  return ex;
}

// A slot that is neither the latest one offered to the host nor the one it is reading.
int ChooseSlot(DeviceData* dd, VrExport* ex) {
  vrlinux::SharedState* s = Link().Shm();
  if (!s) return -1;
  uint64_t latest = s->latest.load(), inuse = s->inuse.load(), done = s->done.load();
  std::lock_guard<std::mutex> l(dd->exportLock);
  for (int i = 0; i < vrlinux::kSlots; ++i) {
    int k = int((ex->nextSlot + i) % vrlinux::kSlots);
    VrSlot& sl = ex->slots[k];
    if (sl.state == 1 || sl.state == 2) continue;
    if (latest && int(vrlinux::SlotOf(latest)) == k) continue;
    if (inuse && int(vrlinux::SlotOf(inuse)) == k && vrlinux::FrameOf(inuse) == sl.frame &&
        done < vrlinux::FrameOf(inuse))
      continue;
    sl.state = 1;
    sl.frame = ++ex->frameNo;
    ex->nextSlot = uint32_t(k + 1);
    dd->d.ResetFences(dd->device, 1, &sl.fence);
    return k;
  }
  return -1;
}

void CopyToSlot(DeviceData* dd, VkCommandBuffer cb, Swapchain* sc, VrExport* ex, int slot, VkImage src, int eye,
                bool first) {
  VkImage dst = ex->slots[slot].image;
  const VkPipelineStageFlags XFER = VK_PIPELINE_STAGE_TRANSFER_BIT;
  if (first)
    BarrierQF(dd, cb, dst, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
              VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, 0, XFER, VK_ACCESS_TRANSFER_WRITE_BIT, VK_QUEUE_FAMILY_IGNORED,
              VK_QUEUE_FAMILY_IGNORED);
  else
    BarrierQF(dd, cb, dst, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, XFER,
              VK_ACCESS_TRANSFER_WRITE_BIT, XFER, VK_ACCESS_TRANSFER_WRITE_BIT, VK_QUEUE_FAMILY_IGNORED,
              VK_QUEUE_FAMILY_IGNORED);
  const int32_t x = eye ? int32_t(ex->w) : 0;
  if (ex->rawCopy) {
    VkImageCopy c{};
    c.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    c.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    c.dstOffset = {x, 0, 0};
    c.extent = {ex->w, ex->h, 1};
    dd->d.CmdCopyImage(cb, src, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, dst, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &c);
  } else {
    VkImageBlit b{};
    b.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    b.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    b.srcOffsets[1] = {int32_t(sc->vext.width), int32_t(sc->vext.height), 1};
    b.dstOffsets[0] = {x, 0, 0};
    b.dstOffsets[1] = {x + int32_t(ex->w), int32_t(ex->h), 1};
    dd->d.CmdBlitImage(cb, src, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, dst, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &b,
                       VK_FILTER_NEAREST);
  }
}

// hand the slot over to the host (queue family release -> EXTERNAL)
void ReleaseSlot(DeviceData* dd, VkCommandBuffer cb, VrExport* ex, int slot) {
  BarrierQF(dd, cb, ex->slots[slot].image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0,
            dd->gfxFamily, VK_QUEUE_FAMILY_EXTERNAL);
}

void FillMeta(vrlinux::SlotMeta& m, const FrameView& v, int eye) {
  memcpy(m.camQ[eye], v.camQ[eye], sizeof(m.camQ[eye]));
  memcpy(m.eyePos[eye], v.eyePos[eye], sizeof(m.eyePos[eye]));
  memcpy(m.fov[eye], v.perEyeFov ? v.efov[eye] : v.fov, sizeof(m.fov[eye]));
}

// caller holds the queue lock; signals the slot fence after everything submitted so far
void SubmitSlotFence(DeviceData* dd, VkQueue q, VrExport* ex, int slot) {
  VkSubmitInfo e{VK_STRUCTURE_TYPE_SUBMIT_INFO};
  dd->d.QueueSubmit(q, 1, &e, ex->slots[slot].fence);
}

void Publish(DeviceData* dd, VrExport* ex, int slot) {
  {
    std::lock_guard<std::mutex> l(dd->exportLock);
    ex->slots[slot].meta.frame = ex->slots[slot].frame;
    ex->slots[slot].state = 2;
  }
  {
    std::lock_guard<std::mutex> l(dd->pubLock);
    dd->pubQueue.push_back({ex, slot});
  }
  dd->pubCv.notify_all();
}

// Waits for the GPU copy of each slot, then offers it to the host (no latency added to the game thread).
void PublisherMain(DeviceData* dd) {
  for (;;) {
    std::pair<VrExport*, int> job;
    {
      std::unique_lock<std::mutex> l(dd->pubLock);
      dd->pubCv.wait(l, [&]() { return dd->pubStop || !dd->pubQueue.empty(); });
      if (dd->pubStop) return;
      job = dd->pubQueue.front();
      dd->pubQueue.erase(dd->pubQueue.begin());
      dd->pubCurrent = job.first;
    }
    VrSlot& s = job.first->slots[job.second];
    VkResult r = VK_TIMEOUT;
    for (int i = 0; i < 50 && r == VK_TIMEOUT; ++i) r = dd->d.WaitForFences(dd->device, 1, &s.fence, VK_TRUE, 100000000ull);
    if (r == VK_SUCCESS) {
      std::lock_guard<std::mutex> ll(Link().Mutex());
      if (vrlinux::SharedState* sh = Link().Shm()) {
        memcpy((void*)&sh->meta[job.second], &s.meta, sizeof(s.meta));
        sh->latest.store(vrlinux::PackFS(s.frame, uint32_t(job.second)));
      }
    }
    {
      std::lock_guard<std::mutex> l(dd->exportLock);
      s.state = r == VK_SUCCESS ? 3 : 0;
    }
    {
      std::lock_guard<std::mutex> l(dd->pubLock);
      dd->pubCurrent = nullptr;
    }
    dd->pubCv.notify_all();
  }
}

// ------------------------------------------------------------------------------------
// Virtual swapchain: the game renders into our images; at present they go to the headset
// (and the right eye is mirrored to the game window on the desktop).
// ------------------------------------------------------------------------------------
Swapchain* FindSwapchain(DeviceData* dd, VkSwapchainKHR h);

bool CreateImageMem(DeviceData* dd, VkImageCreateInfo ici, VkImage* img, VkDeviceMemory* mem) {
  VkResult r = dd->d.CreateImage(dd->device, &ici, nullptr, img);
  if (r != VK_SUCCESS && (ici.usage & VK_IMAGE_USAGE_STORAGE_BIT)) {
    ici.usage &= ~VK_IMAGE_USAGE_STORAGE_BIT;
    r = dd->d.CreateImage(dd->device, &ici, nullptr, img);
  }
  if (r != VK_SUCCESS) return false;
  VkMemoryRequirements req;
  dd->d.GetImageMemoryRequirements(dd->device, *img, &req);
  VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
  ai.allocationSize = req.size;
  ai.memoryTypeIndex = FindMemType(dd, req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  if (dd->d.AllocateMemory(dd->device, &ai, nullptr, mem) != VK_SUCCESS) return false;
  return dd->d.BindImageMemory(dd->device, *img, *mem, 0) == VK_SUCCESS;
}

void DestroySwapchainResources(DeviceData* dd, Swapchain* sc) {
  VkDevice dev = dd->device;
  for (auto& v : sc->v)
    if (v.fence) dd->d.WaitForFences(dev, 1, &v.fence, VK_TRUE, 2000000000ull);
  for (auto& v : sc->v) {
    if (v.cmd) dd->d.FreeCommandBuffers(dev, sc->pool, 1, &v.cmd);
    if (v.cmd2) dd->d.FreeCommandBuffers(dev, sc->pool, 1, &v.cmd2);
    if (v.fence) dd->d.DestroyFence(dev, v.fence, nullptr);
    if (v.acqSem) dd->d.DestroySemaphore(dev, v.acqSem, nullptr);
    if (v.doneSem) dd->d.DestroySemaphore(dev, v.doneSem, nullptr);
    if (v.image) dd->d.DestroyImage(dev, v.image, nullptr);
    if (v.mem) dd->d.FreeMemory(dev, v.mem, nullptr);
  }
  if (sc->pool) dd->d.DestroyCommandPool(dev, sc->pool, nullptr);
  DestroyExport(dd, sc->ex);
  sc->ex = nullptr;
}

VKAPI_ATTR VkResult VKAPI_CALL hook_CreateSwapchainKHR(VkDevice device, const VkSwapchainCreateInfoKHR* ci,
                                                       const VkAllocationCallbacks* a, VkSwapchainKHR* pSwapchain) {
  DeviceData* dd = Dev(device);
  InstanceData* inst = dd->inst;
  if (Cfg().passthrough) return dd->d.CreateSwapchainKHR(device, ci, a, pSwapchain);
  VkSurfaceCapabilitiesKHR caps{};
  inst->d.GetPhysicalDeviceSurfaceCapabilitiesKHR(dd->phys, ci->surface, &caps);
  if (!(caps.supportedUsageFlags & VK_IMAGE_USAGE_TRANSFER_DST_BIT) || dd->gfxFamily == UINT32_MAX) {
    Log(0, "surface does not support transfer dst / no graphics queue: VR disabled for this swapchain");
    return dd->d.CreateSwapchainKHR(device, ci, a, pSwapchain);
  }
  VkSwapchainCreateInfoKHR rci = *ci;
  rci.imageUsage |= VK_IMAGE_USAGE_TRANSFER_DST_BIT;
  uint32_t nm = 0;
  inst->d.GetPhysicalDeviceSurfacePresentModesKHR(dd->phys, ci->surface, &nm, nullptr);
  std::vector<VkPresentModeKHR> modes(nm);
  inst->d.GetPhysicalDeviceSurfacePresentModesKHR(dd->phys, ci->surface, &nm, modes.data());
  auto has = [&](VkPresentModeKHR m) { return std::find(modes.begin(), modes.begin() + nm, m) != modes.begin() + nm; };
  VkPresentModeKHR want = VkPresentModeKHR(Cfg().presentMode);
  // the desktop window must never slow the headset down: mailbox, else immediate, else fifo
  if (!has(want)) want = has(VK_PRESENT_MODE_MAILBOX_KHR) ? VK_PRESENT_MODE_MAILBOX_KHR
                         : has(VK_PRESENT_MODE_IMMEDIATE_KHR) ? VK_PRESENT_MODE_IMMEDIATE_KHR : VK_PRESENT_MODE_FIFO_KHR;
  rci.presentMode = want;
  rci.minImageCount = std::max(ci->minImageCount, caps.minImageCount);
  if (caps.maxImageCount) rci.minImageCount = std::min(rci.minImageCount, caps.maxImageCount);
  // the real (desktop) swapchain follows the real window; the game may render at another size
  VkExtent2D rext = caps.currentExtent;
  if (rext.width == 0xFFFFFFFFu) {
    rext = ci->imageExtent;
    rext.width = std::clamp(rext.width, caps.minImageExtent.width, caps.maxImageExtent.width);
    rext.height = std::clamp(rext.height, caps.minImageExtent.height, caps.maxImageExtent.height);
  }
  rci.imageExtent = rext;

  VkResult r = dd->d.CreateSwapchainKHR(device, &rci, a, pSwapchain);
  if (r != VK_SUCCESS) return r;

  auto* sc = new Swapchain();
  sc->handle = *pSwapchain;
  sc->format = ci->imageFormat;
  sc->vext = ci->imageExtent;
  dd->gameW = ci->imageExtent.width;
  dd->gameH = ci->imageExtent.height;
  sc->rext = rext;
  uint32_t n = 0;
  dd->d.GetSwapchainImagesKHR(device, sc->handle, &n, nullptr);
  sc->realImages.resize(n);
  dd->d.GetSwapchainImagesKHR(device, sc->handle, &n, sc->realImages.data());

  VkFormatProperties fp{};
  inst->d.GetPhysicalDeviceFormatProperties(dd->phys, sc->format, &fp);
  VkFormatProperties fpd{};
  inst->d.GetPhysicalDeviceFormatProperties(dd->phys, VK_FORMAT_R8G8B8A8_UNORM, &fpd);
  sc->canBlit = (fp.optimalTilingFeatures & VK_FORMAT_FEATURE_BLIT_SRC_BIT) &&
                (fpd.optimalTilingFeatures & VK_FORMAT_FEATURE_BLIT_DST_BIT);
  sc->canBlitSelf = (fp.optimalTilingFeatures & VK_FORMAT_FEATURE_BLIT_SRC_BIT) &&
                    (fp.optimalTilingFeatures & VK_FORMAT_FEATURE_BLIT_DST_BIT);

  bool ok = true;
  VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
  pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  pci.queueFamilyIndex = dd->gfxFamily;
  sc->poolFamily = dd->gfxFamily;
  ok &= dd->d.CreateCommandPool(device, &pci, nullptr, &sc->pool) == VK_SUCCESS;

  VkImageFormatListCreateInfo fmtList{VK_STRUCTURE_TYPE_IMAGE_FORMAT_LIST_CREATE_INFO};
  bool haveFmtList = false;
  for (auto* p = (const VkBaseInStructure*)ci->pNext; p; p = p->pNext)
    if (p->sType == VK_STRUCTURE_TYPE_IMAGE_FORMAT_LIST_CREATE_INFO) {
      fmtList = *(const VkImageFormatListCreateInfo*)p;
      fmtList.pNext = nullptr;
      haveFmtList = true;
    }
  VkImageCreateInfo ici{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
  if (ci->flags & VK_SWAPCHAIN_CREATE_MUTABLE_FORMAT_BIT_KHR) {
    ici.flags |= VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT | VK_IMAGE_CREATE_EXTENDED_USAGE_BIT;
    if (haveFmtList) ici.pNext = &fmtList;
  }
  ici.imageType = VK_IMAGE_TYPE_2D;
  ici.format = ci->imageFormat;
  ici.extent = {ci->imageExtent.width, ci->imageExtent.height, 1};
  ici.mipLevels = 1;
  ici.arrayLayers = std::max(1u, ci->imageArrayLayers);
  ici.samples = VK_SAMPLE_COUNT_1_BIT;
  ici.tiling = VK_IMAGE_TILING_OPTIMAL;
  ici.usage = ci->imageUsage | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
  ici.sharingMode = ci->imageSharingMode;
  ici.queueFamilyIndexCount = ci->queueFamilyIndexCount;
  ici.pQueueFamilyIndices = ci->pQueueFamilyIndices;
  ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

  uint32_t count = std::max(ci->minImageCount, 3u);
  sc->v.resize(count);
  for (auto& v : sc->v) {
    ok &= CreateImageMem(dd, ici, &v.image, &v.mem);
    VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    fci.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    ok &= dd->d.CreateFence(device, &fci, nullptr, &v.fence) == VK_SUCCESS;
    VkSemaphoreCreateInfo sci{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    ok &= dd->d.CreateSemaphore(device, &sci, nullptr, &v.acqSem) == VK_SUCCESS;
    ok &= dd->d.CreateSemaphore(device, &sci, nullptr, &v.doneSem) == VK_SUCCESS;
    VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cai.commandPool = sc->pool;
    cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cai.commandBufferCount = 1;
    ok &= sc->pool && dd->d.AllocateCommandBuffers(device, &cai, &v.cmd) == VK_SUCCESS;
    ok &= sc->pool && dd->d.AllocateCommandBuffers(device, &cai, &v.cmd2) == VK_SUCCESS;
    for (VkCommandBuffer c : {v.cmd, v.cmd2}) {
      if (c && dd->setLoaderData) dd->setLoaderData(device, c);
      else if (c) *(void**)c = *(void**)device;
    }
  }
  if (!ok) {
    Log(0, "failed to create virtual swapchain resources");
    DestroySwapchainResources(dd, sc);
    delete sc;
    dd->d.DestroySwapchainKHR(device, *pSwapchain, a);
    *pSwapchain = VK_NULL_HANDLE;
    return VK_ERROR_OUT_OF_DEVICE_MEMORY;
  }
  {
    std::lock_guard<std::mutex> l(dd->swapLock);
    dd->swapchains[sc->handle] = sc;
  }
  ++dd->swapCount;
  dd->cur = ComputeView(dd, sc);
  Log(1, "swapchain: game %ux%u format %d, %u virtual / %u real images, desktop present mode %d", sc->vext.width,
      sc->vext.height, sc->format, count, n, rci.presentMode);
  return VK_SUCCESS;
}

Swapchain* FindSwapchain(DeviceData* dd, VkSwapchainKHR h) {
  std::lock_guard<std::mutex> l(dd->swapLock);
  auto it = dd->swapchains.find(h);
  return it == dd->swapchains.end() ? nullptr : it->second;
}

VKAPI_ATTR void VKAPI_CALL hook_DestroySwapchainKHR(VkDevice device, VkSwapchainKHR h, const VkAllocationCallbacks* a) {
  DeviceData* dd = Dev(device);
  Swapchain* sc = nullptr;
  {
    std::lock_guard<std::mutex> l(dd->swapLock);
    auto it = dd->swapchains.find(h);
    if (it != dd->swapchains.end()) {
      sc = it->second;
      dd->swapchains.erase(it);
    }
  }
  if (sc) --dd->swapCount;
  if (sc && dd->replay) {
    std::lock_guard<std::mutex> rl(dd->recLock);
    AbortReplayLocked(dd, "swapchain destroyed");
  }
  if (sc && dd->gfxQueue) {
    std::lock_guard<std::mutex> l(GetQueueInfo(dd, dd->gfxQueue)->m);
    dd->d.QueueWaitIdle(dd->gfxQueue);
  }
  dd->d.DestroySwapchainKHR(device, h, a);
  if (sc) {
    DestroySwapchainResources(dd, sc);
    delete sc;
  }
}

VKAPI_ATTR VkResult VKAPI_CALL hook_GetSwapchainImagesKHR(VkDevice device, VkSwapchainKHR h, uint32_t* count,
                                                          VkImage* images) {
  DeviceData* dd = Dev(device);
  Swapchain* sc = FindSwapchain(dd, h);
  if (!sc) return dd->d.GetSwapchainImagesKHR(device, h, count, images);
  uint32_t n = uint32_t(sc->v.size());
  if (!images) {
    *count = n;
    return VK_SUCCESS;
  }
  uint32_t c = std::min(*count, n);
  for (uint32_t i = 0; i < c; ++i) images[i] = sc->v[i].image;
  *count = c;
  return c < n ? VK_INCOMPLETE : VK_SUCCESS;
}

VkResult AcquireVirtual(DeviceData* dd, Swapchain* sc, uint64_t timeout, VkSemaphore sem, VkFence fence,
                        uint32_t* idx) {
  uint32_t i = sc->nextAcquire;
  dd->stage = "game acquire: waiting for the previous copy of this image";
  VkResult w = dd->d.WaitForFences(dd->device, 1, &sc->v[i].fence, VK_TRUE, timeout);
  dd->stage = "game acquire: done";
  if (w == VK_TIMEOUT) return timeout == 0 ? VK_NOT_READY : VK_TIMEOUT;
  if (w != VK_SUCCESS) return w;
  sc->nextAcquire = (i + 1) % uint32_t(sc->v.size());
  if (sem != VK_NULL_HANDLE || fence != VK_NULL_HANDLE) {
    VkQueue q = dd->gfxQueue;
    if (!q) return VK_ERROR_DEVICE_LOST;
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.signalSemaphoreCount = sem ? 1 : 0;
    si.pSignalSemaphores = &sem;
    dd->stage = "game acquire: signalling (waiting for queue lock)";
    std::lock_guard<std::mutex> l(GetQueueInfo(dd, q)->m);
    VkResult r = dd->d.QueueSubmit(q, 1, &si, fence);
    dd->stage = "game is rendering";
    if (r != VK_SUCCESS) return r;
  }
  *idx = i;
  return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL hook_AcquireNextImageKHR(VkDevice device, VkSwapchainKHR h, uint64_t timeout,
                                                        VkSemaphore sem, VkFence fence, uint32_t* idx) {
  DeviceData* dd = Dev(device);
  Swapchain* sc = FindSwapchain(dd, h);
  if (!sc) return dd->d.AcquireNextImageKHR(device, h, timeout, sem, fence, idx);
  return AcquireVirtual(dd, sc, timeout, sem, fence, idx);
}
VKAPI_ATTR VkResult VKAPI_CALL hook_AcquireNextImage2KHR(VkDevice device, const VkAcquireNextImageInfoKHR* info,
                                                         uint32_t* idx) {
  DeviceData* dd = Dev(device);
  Swapchain* sc = FindSwapchain(dd, info->swapchain);
  if (!sc) return dd->d.AcquireNextImage2KHR(device, info, idx);
  return AcquireVirtual(dd, sc, info->timeout, info->semaphore, info->fence, idx);
}

void Barrier(DeviceData* dd, VkCommandBuffer cb, VkImage img, VkImageLayout from, VkImageLayout to,
             VkPipelineStageFlags srcStage, VkAccessFlags srcAccess, VkPipelineStageFlags dstStage,
             VkAccessFlags dstAccess) {
  BarrierQF(dd, cb, img, from, to, srcStage, srcAccess, dstStage, dstAccess, VK_QUEUE_FAMILY_IGNORED,
            VK_QUEUE_FAMILY_IGNORED);
}

// game image (TRANSFER_SRC) -> desktop window image, left in PRESENT_SRC
void MirrorToWindow(DeviceData* dd, Swapchain* sc, VkCommandBuffer cb, VkImage src, VkImage real) {
  const VkPipelineStageFlags ALL = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, XFER = VK_PIPELINE_STAGE_TRANSFER_BIT;
  Barrier(dd, cb, real, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, ALL, 0, XFER,
          VK_ACCESS_TRANSFER_WRITE_BIT);
  if ((sc->vext.width != sc->rext.width || sc->vext.height != sc->rext.height) && sc->canBlitSelf) {
    VkImageBlit b{};
    b.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    b.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    b.srcOffsets[1] = {int32_t(sc->vext.width), int32_t(sc->vext.height), 1};
    b.dstOffsets[1] = {int32_t(sc->rext.width), int32_t(sc->rext.height), 1};
    dd->d.CmdBlitImage(cb, src, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, real, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &b,
                       VK_FILTER_LINEAR);
  } else {
    VkImageCopy c{};
    c.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    c.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    c.extent = {std::min(sc->vext.width, sc->rext.width), std::min(sc->vext.height, sc->rext.height), 1};
    dd->d.CmdCopyImage(cb, src, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, real, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &c);
  }
  Barrier(dd, cb, real, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, XFER,
          VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0);
}

VkResult Worse(VkResult a, VkResult b) {
  if (a < 0) return a;
  if (b < 0) return b;
  if (a == VK_SUBOPTIMAL_KHR || b == VK_SUBOPTIMAL_KHR) return VK_SUBOPTIMAL_KHR;
  return VK_SUCCESS;
}

void UpdateParams(DeviceData* dd, VkCommandBuffer cb, const EyeParams& p) {
  const VkPipelineStageFlags ALL = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, XFER = VK_PIPELINE_STAGE_TRANSFER_BIT;
  dd->d.CmdPipelineBarrier(cb, ALL, XFER, 0, 0, nullptr, 0, nullptr, 0, nullptr);
  dd->d.CmdUpdateBuffer(cb, dd->paramBuf, 0, sizeof(p.m), p.m);
  dd->d.CmdUpdateBuffer(cb, dd->paramBuf, 256, sizeof(p.m), p.m);
  VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
  mb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_MEMORY_READ_BIT;
  dd->d.CmdPipelineBarrier(cb, XFER, ALL, 0, 1, &mb, 0, nullptr, 0, nullptr);
}
void FullBarrier(DeviceData* dd, VkCommandBuffer cb) {
  VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
  mb.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
  mb.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
  dd->d.CmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &mb, 0,
                           nullptr, 0, nullptr);
}

void Stats(DeviceData* dd, Swapchain* sc) {
  auto now = Clock::now();
  ++sc->statFrames;
  double el = std::chrono::duration<double>(now - sc->statStart).count();
  if (el < 5.0) return;
  uint32_t rf = dd->replayFrames.exchange(0), mf = dd->monoFrames.exchange(0), xf = dd->relaxedFrames.exchange(0);
  auto* sh = Link().Shm();
  if (dd->replay && Cfg().autoSwitch && mf > rf && rf + mf >= 30) {
    if (++dd->badReplayWindows >= 2) {
      Log(0, "this game reads back from the GPU every frame (%u of %u frames 2D): switching to alternate mode "
          "(each eye gets half the frames). Use mode = alternate in its profile to skip this.", mf, rf + mf);
      AbortReplayLocked(dd, "switching to alternate");  // caller (PresentReplay) holds recLock
      dd->replay = false;
      dd->recMono = false;
      dd->frame = 0;
    }
  } else {
    dd->badReplayWindows = 0;
  }
  Log(1, "game %.1f fps | to headset %.1f fps [%s] | replay 3D %u (relaxed %u) / 2D %u | shaders %u | off-screen passes %u | %s fov %.1f scale %.3f sens %.2f",
      sc->statFrames / el, sc->statVr / el,
      !sh ? "host not running" : sh->sessionRunning.load() ? "headset on" : "waiting for headset", rf, xf, mf,
      dd->patchedCount.load(), dd->monoPasses.exchange(0), Cfg().display ? "screen" : "vr", Cfg().fov, Cfg().worldScale, Cfg().mouseSens);
  sc->statFrames = 0;
  sc->statVr = 0;
  sc->statStart = now;
}

// Desktop window (mirror): optional and never allowed to block the headset.
// Returns <0 only for errors the game must react to (window resized / lost).
VkResult AcquireMirror(DeviceData* dd, Swapchain* sc, VImage& vi, uint32_t* realIdx) {
  *realIdx = UINT32_MAX;
  const bool vr = Link().Connected();
  if (vr && !Cfg().mirror) return VK_SUCCESS;  // in the headset: nothing on the desktop
  VkResult r = dd->d.AcquireNextImageKHR(dd->device, sc->handle, vr ? 0 : 50000000ull, vi.acqSem, VK_NULL_HANDLE,
                                         realIdx);
  if (r == VK_SUCCESS || r == VK_SUBOPTIMAL_KHR) return r;
  *realIdx = UINT32_MAX;
  if (r == VK_NOT_READY || r == VK_TIMEOUT) return VK_SUCCESS;
  return r;
}

// Alternate mode: one game frame per eye (even = left).
VkResult PresentOne(DeviceData* dd, VkQueue queue, Swapchain* sc, uint32_t idx, const VkSemaphore* waits,
                    uint32_t waitCount) {
  if (idx >= sc->v.size()) return VK_ERROR_OUT_OF_DATE_KHR;
  VImage& vi = sc->v[idx];
  const uint64_t frame = dd->frame;
  const int eye = int(frame & 1);
  const FrameView cur = dd->cur;
  QueueInfo* qi = GetQueueInfo(dd, queue);

  uint32_t realIdx = UINT32_MAX;
  dd->stage = "present(alternate): acquiring window image";
  VkResult acq = AcquireMirror(dd, sc, vi, &realIdx);

  VrExport* ex = cur.vr ? EnsureExport(dd, sc) : nullptr;
  int slot = -1;
  bool first = false;
  if (ex) {
    if (eye == 0) {
      if (ex->writing >= 0) {  // left eye written but never completed: give the slot back
        std::lock_guard<std::mutex> l(dd->exportLock);
        ex->slots[ex->writing].state = 0;
        ex->writing = -1;
      }
      slot = ChooseSlot(dd, ex);
      ex->writing = slot;
      first = true;
    } else {
      slot = ex->writing;
    }
  }
  FrameView next = ComputeView(dd, sc);
  // next frame's eye uses the freshly sampled pose
  const int nextEye = int((frame + 1) & 1);

  dd->d.WaitForFences(dd->device, 1, &vi.fence, VK_TRUE, UINT64_MAX);
  dd->d.ResetFences(dd->device, 1, &vi.fence);
  VkCommandBuffer cb = vi.cmd;
  dd->d.ResetCommandBuffer(cb, 0);
  VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  dd->d.BeginCommandBuffer(cb, &bi);
  const VkPipelineStageFlags ALL = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, XFER = VK_PIPELINE_STAGE_TRANSFER_BIT;
  Barrier(dd, cb, vi.image, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, ALL,
          VK_ACCESS_MEMORY_WRITE_BIT, XFER, VK_ACCESS_TRANSFER_READ_BIT);
  if (slot >= 0) {
    CopyToSlot(dd, cb, sc, ex, slot, vi.image, eye, first);
    FillMeta(ex->slots[slot].meta, cur, eye);
    ex->slots[slot].meta.mono = 0;
    if (eye == 1) ReleaseSlot(dd, cb, ex, slot);
  }
  if (realIdx != UINT32_MAX) MirrorToWindow(dd, sc, cb, vi.image, sc->realImages[realIdx]);
  Barrier(dd, cb, vi.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, XFER, 0,
          VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0);
  if (dd->stereoShaders) UpdateParams(dd, cb, next.p[nextEye]);
  dd->d.EndCommandBuffer(cb);

  std::vector<VkSemaphore> ws(waits, waits + waitCount);
  if (realIdx != UINT32_MAX) ws.push_back(vi.acqSem);
  std::vector<VkPipelineStageFlags> wstages(ws.size(), ALL);
  VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
  si.waitSemaphoreCount = uint32_t(ws.size());
  si.pWaitSemaphores = ws.data();
  si.pWaitDstStageMask = wstages.data();
  si.commandBufferCount = 1;
  si.pCommandBuffers = &cb;
  si.signalSemaphoreCount = realIdx != UINT32_MAX ? 1 : 0;
  si.pSignalSemaphores = &vi.doneSem;
  VkQueue subQueue = (qi->family == sc->poolFamily || !dd->gfxQueue) ? queue : dd->gfxQueue;
  VkResult result = VK_SUCCESS;
  {
    dd->stage = "present(alternate): submitting copy";
    std::lock_guard<std::mutex> l(GetQueueInfo(dd, subQueue)->m);
    VkResult sr = dd->d.QueueSubmit(subQueue, 1, &si, vi.fence);
    if (sr != VK_SUCCESS) return sr;
    if (slot >= 0 && eye == 1) SubmitSlotFence(dd, subQueue, ex, slot);
  }
  if (slot >= 0 && eye == 1) {
    Publish(dd, ex, slot);
    ex->writing = -1;
    ++sc->statVr;
  }
  if (realIdx != UINT32_MAX) {
    dd->stage = "present(alternate): presenting window";
    std::lock_guard<std::mutex> l(qi->m);
    VkPresentInfoKHR pi{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
    pi.waitSemaphoreCount = 1;
    pi.pWaitSemaphores = &vi.doneSem;
    pi.swapchainCount = 1;
    pi.pSwapchains = &sc->handle;
    pi.pImageIndices = &realIdx;
    result = dd->d.QueuePresentKHR(queue, &pi);
  }
  if (acq < 0) result = acq;
  else if (acq == VK_SUBOPTIMAL_KHR) result = Worse(result, acq);
  dd->frame = frame + 1;
  // the frame after this one is rendered for nextEye with the pose sampled now; keep the
  // other eye's data of cur so a slot always gets both halves' metadata
  dd->cur = next;
  Stats(dd, sc);
  return result;
}

// Replay mode: the game rendered this frame for the left eye. Copy it out, switch the shaders
// to the right eye, submit the same command buffers again, copy the right eye and present.
// Caller holds dd->recLock.
VkResult PresentReplay(DeviceData* dd, VkQueue queue, Swapchain* sc, uint32_t idx, const VkSemaphore* waits,
                       uint32_t waitCount) {
  QueueInfo* qi = GetQueueInfo(dd, queue);
  std::vector<VkCommandBuffer> cbs;
  cbs.swap(dd->recCbs);
  const FrameView cur = dd->cur;
  const bool stereoWanted = cur.vr && Cfg().stereo;
  bool doReplay = !dd->recMono && !cbs.empty() && stereoWanted && queue == dd->gfxQueue;
  dd->recMono = false;
  // frame serial: command buffers / objects of this frame must live until its replay is done
  uint64_t ser;
  {
    std::lock_guard<std::mutex> l(dd->cbLock);
    ser = dd->frameSerial++;
    if (dd->dropFrame) doReplay = false;
    dd->dropFrame = false;
    if (doReplay)
      for (VkCommandBuffer cb : cbs) dd->cbSerial[cb] = ser;
    dd->recSet.clear();
  }
  struct SerialDone {  // on every return path: the replay of 'ser' is submitted (or not done)
    DeviceData* dd; uint64_t ser;
    ~SerialDone() {
      {
        std::lock_guard<std::mutex> l(dd->cbLock);
        dd->submittedSerial = std::max(dd->submittedSerial, ser);
      }
      dd->cbCv.notify_all();
      RunDeferred(dd, false);
    }
  } serialDone{dd, ser};
  if (idx >= sc->v.size()) {
    std::lock_guard<std::mutex> l(qi->m);
    ReleaseHeldLocked(dd, queue);
    return VK_ERROR_OUT_OF_DATE_KHR;
  }
  VImage& vi = sc->v[idx];
  uint32_t realIdx = UINT32_MAX;
  dd->stage = "present(replay): acquiring window image";
  VkResult acq = AcquireMirror(dd, sc, vi, &realIdx);
  const bool mirror = realIdx != UINT32_MAX;
  const VkPipelineStageFlags ALL = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, XFER = VK_PIPELINE_STAGE_TRANSFER_BIT;
  std::vector<VkPipelineStageFlags> allStages(waitCount + 1, ALL);
  if (acq < 0) {
    std::lock_guard<std::mutex> l(qi->m);
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.waitSemaphoreCount = waitCount;
    si.pWaitSemaphores = waits;
    si.pWaitDstStageMask = allStages.data();
    dd->d.QueueSubmit(queue, 1, &si, VK_NULL_HANDLE);
    ReleaseHeldLocked(dd, queue);
    return acq;
  }
  VrExport* ex = cur.vr ? EnsureExport(dd, sc) : nullptr;
  int slot = ex ? ChooseSlot(dd, ex) : -1;
  FrameView next = ComputeView(dd, sc);

  dd->stage = "present(replay): waiting for previous copy";
  dd->d.WaitForFences(dd->device, 1, &vi.fence, VK_TRUE, UINT64_MAX);
  dd->d.ResetFences(dd->device, 1, &vi.fence);
  dd->stage = "present(replay): recording";
  VkImage real = mirror ? sc->realImages[realIdx] : VK_NULL_HANDLE;
  VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

  // A: left eye -> slot (+ right half too and the window if there is no replay this frame)
  VkCommandBuffer a = vi.cmd;
  dd->d.ResetCommandBuffer(a, 0);
  dd->d.BeginCommandBuffer(a, &bi);
  Barrier(dd, a, vi.image, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, ALL,
          VK_ACCESS_MEMORY_WRITE_BIT, XFER, VK_ACCESS_TRANSFER_READ_BIT);
  if (slot >= 0) {
    CopyToSlot(dd, a, sc, ex, slot, vi.image, 0, true);
    if (!doReplay) CopyToSlot(dd, a, sc, ex, slot, vi.image, 1, false);
  }
  if (!doReplay && mirror) MirrorToWindow(dd, sc, a, vi.image, real);
  Barrier(dd, a, vi.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, XFER, 0, ALL, 0);
  if (doReplay) {
    UpdateParams(dd, a, cur.p[1]);  // right eye
    FullBarrier(dd, a);
  } else {
    if (slot >= 0) ReleaseSlot(dd, a, ex, slot);
    if (dd->stereoShaders) UpdateParams(dd, a, next.p[0]);  // left eye of the next frame
  }
  dd->d.EndCommandBuffer(a);

  // B: right eye -> slot and window, shaders back to the left eye (next frame's pose)
  VkCommandBuffer b = vi.cmd2;
  if (doReplay) {
    dd->d.ResetCommandBuffer(b, 0);
    dd->d.BeginCommandBuffer(b, &bi);
    FullBarrier(dd, b);
    Barrier(dd, b, vi.image, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, ALL,
            VK_ACCESS_MEMORY_WRITE_BIT, XFER, VK_ACCESS_TRANSFER_READ_BIT);
    if (slot >= 0) {
      CopyToSlot(dd, b, sc, ex, slot, vi.image, 1, false);
      ReleaseSlot(dd, b, ex, slot);
    }
    if (mirror) MirrorToWindow(dd, sc, b, vi.image, real);
    Barrier(dd, b, vi.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, XFER, 0, ALL, 0);
    UpdateParams(dd, b, next.p[0]);
    dd->d.EndCommandBuffer(b);
  }

  std::vector<VkSemaphore> ws(waits, waits + waitCount);
  if (mirror) ws.push_back(vi.acqSem);
  VkSubmitInfo si[3] = {{VK_STRUCTURE_TYPE_SUBMIT_INFO}, {VK_STRUCTURE_TYPE_SUBMIT_INFO}, {VK_STRUCTURE_TYPE_SUBMIT_INFO}};
  si[0].waitSemaphoreCount = uint32_t(ws.size());
  si[0].pWaitSemaphores = ws.data();
  si[0].pWaitDstStageMask = allStages.data();
  si[0].commandBufferCount = 1;
  si[0].pCommandBuffers = &a;
  uint32_t nsi = 1;
  if (doReplay) {
    si[1].commandBufferCount = uint32_t(cbs.size());
    si[1].pCommandBuffers = cbs.data();
    si[2].commandBufferCount = 1;
    si[2].pCommandBuffers = &b;
    nsi = 3;
  }
  si[nsi - 1].signalSemaphoreCount = mirror ? 1 : 0;
  si[nsi - 1].pSignalSemaphores = &vi.doneSem;

  VkFence serFence = VK_NULL_HANDLE;
  const int sk = int(ser % 8);
  if (doReplay) {
    if (!dd->serialFence[sk]) {
      VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
      fci.flags = VK_FENCE_CREATE_SIGNALED_BIT;
      dd->d.CreateFence(dd->device, &fci, nullptr, &dd->serialFence[sk]);
    }
    serFence = dd->serialFence[sk];
    dd->d.WaitForFences(dd->device, 1, &serFence, VK_TRUE, 2000000000ull);
    dd->d.ResetFences(dd->device, 1, &serFence);
  }
  VkResult result;
  {
    dd->stage = "present(replay): submitting replay";
    std::lock_guard<std::mutex> l(qi->m);
    VkResult sr = dd->d.QueueSubmit(queue, nsi, si, vi.fence);
    if (sr == VK_SUCCESS && slot >= 0) SubmitSlotFence(dd, queue, ex, slot);
    if (serFence) {
      VkSubmitInfo e{VK_STRUCTURE_TYPE_SUBMIT_INFO};
      dd->d.QueueSubmit(queue, 1, &e, serFence);
      std::lock_guard<std::mutex> cl(dd->cbLock);
      dd->serialOf[sk] = ser;
    }
    dd->stage = "present(replay): presenting window";
    ReleaseHeldLocked(dd, queue);
    if (sr != VK_SUCCESS) return sr;
    result = VK_SUCCESS;
    if (mirror) {
      VkPresentInfoKHR pi{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
      pi.waitSemaphoreCount = 1;
      pi.pWaitSemaphores = &vi.doneSem;
      pi.swapchainCount = 1;
      pi.pSwapchains = &sc->handle;
      pi.pImageIndices = &realIdx;
      result = dd->d.QueuePresentKHR(queue, &pi);
    }
  }
  if (slot >= 0) {
    vrlinux::SlotMeta& m = ex->slots[slot].meta;
    FillMeta(m, cur, 0);
    if (doReplay) {
      FillMeta(m, cur, 1);
      m.mono = 0;
    } else {  // same (left) image for both eyes
      memcpy(m.camQ[1], m.camQ[0], sizeof(m.camQ[0]));
      memcpy(m.eyePos[1], m.eyePos[0], sizeof(m.eyePos[0]));
      memcpy(m.fov[1], m.fov[0], sizeof(m.fov[0]));
      m.mono = 1;
    }
    Publish(dd, ex, slot);
    ++sc->statVr;
  }
  if (acq == VK_SUBOPTIMAL_KHR) result = Worse(result, acq);
  if (doReplay) ++dd->replayFrames;
  ++dd->frame;
  dd->cur = next;
  Stats(dd, sc);
  return result;
}

VKAPI_ATTR VkResult VKAPI_CALL hook_QueuePresentKHR(VkQueue queue, const VkPresentInfoKHR* pi) {
  DeviceData* dd = Dev(queue);
  std::vector<Swapchain*> scs(pi->swapchainCount, nullptr);
  bool any = false;
  for (uint32_t i = 0; i < pi->swapchainCount; ++i) any |= (scs[i] = FindSwapchain(dd, pi->pSwapchains[i])) != nullptr;
  if (!any) {
    std::lock_guard<std::mutex> l(GetQueueInfo(dd, queue)->m);
    return dd->d.QueuePresentKHR(queue, pi);
  }
  PollControls();
  Link().Poll();
  WriteSettings();

  VkResult overall = VK_SUCCESS;
  for (uint32_t i = 0; i < pi->swapchainCount; ++i) {
    const VkSemaphore* waits = i == 0 ? pi->pWaitSemaphores : nullptr;
    uint32_t wc = i == 0 ? pi->waitSemaphoreCount : 0;
    VkResult r;
    if (scs[i] && dd->replay) {
      std::lock_guard<std::mutex> rl(dd->recLock);
      r = PresentReplay(dd, queue, scs[i], pi->pImageIndices[i], waits, wc);
    } else if (scs[i]) {
      r = PresentOne(dd, queue, scs[i], pi->pImageIndices[i], waits, wc);
    } else {
      VkPresentInfoKHR one{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
      one.waitSemaphoreCount = wc;
      one.pWaitSemaphores = waits;
      one.swapchainCount = 1;
      one.pSwapchains = &pi->pSwapchains[i];
      one.pImageIndices = &pi->pImageIndices[i];
      std::lock_guard<std::mutex> l(GetQueueInfo(dd, queue)->m);
      r = dd->d.QueuePresentKHR(queue, &one);
    }
    if (pi->pResults) pi->pResults[i] = r;
    overall = Worse(overall, r);
  }
  ++dd->presentCount;
  dd->stage = "game is rendering (between presents)";
  // frame limiter: headset refresh rate by default (stereo frames per second)
  float lim = Cfg().fpsLimit;
  if (lim == 0.0f) {
    auto* sh = Link().Shm();
    lim = (sh && sh->sessionRunning.load()) ? sh->displayHzX100.load() / 100.0f : 0.0f;
  }
  if (!dd->replay) lim *= 2.0f;  // alternate: one present per eye
  auto now = Clock::now();
  if (lim > 1.0f && dd->lastPresent != Clock::time_point{}) {
    auto period = std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(1.0 / lim));
    auto target = dd->lastPresent + period;
    if (target > now && target - now < std::chrono::milliseconds(100)) {
      if (target - now > std::chrono::microseconds(1500))
        std::this_thread::sleep_until(target - std::chrono::microseconds(1000));
      while (Clock::now() < target) std::this_thread::yield();
      now = target;
    }
    dd->lastPresent = (now - dd->lastPresent < 2 * period) ? std::max(target, dd->lastPresent) : now;
    if (dd->lastPresent < now - std::chrono::milliseconds(50)) dd->lastPresent = now;
  } else {
    dd->lastPresent = now;
  }
  return overall;
}

// ------------------------------------------------------------------------------------
// Proc addr
// ------------------------------------------------------------------------------------
#define DEVICE_HOOKS(X)                                                                          \
  X(DestroyDevice) X(GetDeviceQueue) X(GetDeviceQueue2) X(QueueSubmit) X(QueueSubmit2)          \
  X(QueueSubmit2KHR) X(QueueWaitIdle) X(QueueBindSparse) X(DeviceWaitIdle) X(CreateShaderModule) \
  X(DestroyShaderModule) X(CreateGraphicsPipelines) X(CreateSwapchainKHR) X(DestroySwapchainKHR) \
  X(GetSwapchainImagesKHR) X(AcquireNextImageKHR) X(AcquireNextImage2KHR) X(QueuePresentKHR)          \
  X(CmdSetViewport) X(CmdSetViewportWithCount) X(CmdSetViewportWithCountEXT) X(BeginCommandBuffer)            \
  X(EndCommandBuffer) X(CmdBeginRenderPass) X(CmdBeginRenderPass2) X(CmdBeginRenderPass2KHR) X(CmdBeginRendering)  \
  X(CmdBeginRenderingKHR)
#define REPLAY_HOOKS(X)                                                                                        \
  X(WaitForFences) X(WaitSemaphores) X(WaitSemaphoresKHR) X(CreateSemaphore) X(DestroySemaphore) X(GetFenceStatus) \
  X(GetSemaphoreCounterValue) X(GetSemaphoreCounterValueKHR) \
  X(ResetCommandBuffer) X(AllocateCommandBuffers) X(FreeCommandBuffers) X(ResetCommandPool) X(DestroyCommandPool)     \
  X(DestroyBuffer) X(DestroyImage) X(FreeMemory) X(DestroyImageView) X(DestroyBufferView) X(DestroySampler)          \
  X(DestroyPipeline) X(DestroyFramebuffer) X(DestroyQueryPool) X(DestroyDescriptorPool) X(DestroyEvent)              \
  X(ResetDescriptorPool) X(FreeDescriptorSets)

PFN_vkVoidFunction DeviceHook(const char* name) {
#define CHECK(n) if (!strcmp(name, "vk" #n)) return (PFN_vkVoidFunction)&hook_##n;
  if (Cfg().passthrough) return nullptr;  // launchers / helpers: no hooks at all
  if (Cfg().mode == 1) {                 // replay mode needs to see waits and command buffer setup
    REPLAY_HOOKS(CHECK)
  }
  DEVICE_HOOKS(CHECK)
#undef CHECK
  return nullptr;
}

}  // namespace

EXPORT VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL vrlinux_GetDeviceProcAddr(VkDevice device, const char* name);

EXPORT VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL vrlinux_GetInstanceProcAddr(VkInstance instance,
                                                                             const char* name) {
  if (!strcmp(name, "vkGetInstanceProcAddr")) return (PFN_vkVoidFunction)&vrlinux_GetInstanceProcAddr;
  if (!strcmp(name, "vkGetDeviceProcAddr")) return (PFN_vkVoidFunction)&vrlinux_GetDeviceProcAddr;
  if (!strcmp(name, "vkCreateInstance")) return (PFN_vkVoidFunction)&hook_CreateInstance;
  if (!strcmp(name, "vkCreateDevice")) return (PFN_vkVoidFunction)&hook_CreateDevice;
  if (!instance) return nullptr;
  InstanceData* id = Inst(instance);
  if (!id) return nullptr;
  PFN_vkVoidFunction next = id->gipa(instance, name);
  if (!next) return nullptr;
#define CHECK(n) if (!strcmp(name, "vk" #n)) return (PFN_vkVoidFunction)&hook_##n;
  CHECK(DestroyInstance)
  CHECK(EnumeratePhysicalDevices)
  CHECK(EnumeratePhysicalDeviceGroups)
  CHECK(EnumeratePhysicalDeviceGroupsKHR)
  CHECK(EnumerateDeviceExtensionProperties)
  if (!Cfg().passthrough) {
    CHECK(GetPhysicalDeviceSurfaceCapabilitiesKHR)
    CHECK(GetPhysicalDeviceSurfaceCapabilities2KHR)
  }
#undef CHECK
  if (PFN_vkVoidFunction h = DeviceHook(name)) return h;
  return next;
}

EXPORT VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL vrlinux_GetDeviceProcAddr(VkDevice device, const char* name) {
  if (!strcmp(name, "vkGetDeviceProcAddr")) return (PFN_vkVoidFunction)&vrlinux_GetDeviceProcAddr;
  DeviceData* dd = Dev(device);
  if (!dd) return nullptr;
  PFN_vkVoidFunction next = dd->d.GetDeviceProcAddr(device, name);
  if (!next) return nullptr;
  if (PFN_vkVoidFunction h = DeviceHook(name)) return h;
  return next;
}

EXPORT VKAPI_ATTR VkResult VKAPI_CALL vrlinux_NegotiateLoaderLayerInterfaceVersion(VkNegotiateLayerInterface* p) {
  if (!p || p->sType != LAYER_NEGOTIATE_INTERFACE_STRUCT) return VK_ERROR_INITIALIZATION_FAILED;
  if (p->loaderLayerInterfaceVersion >= 2) {
    p->pfnGetInstanceProcAddr = vrlinux_GetInstanceProcAddr;
    p->pfnGetDeviceProcAddr = vrlinux_GetDeviceProcAddr;
    p->pfnGetPhysicalDeviceProcAddr = nullptr;
    p->loaderLayerInterfaceVersion = 2;
  }
  return VK_SUCCESS;
}
