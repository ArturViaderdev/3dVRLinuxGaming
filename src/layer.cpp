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
#include <random>
#include <vector>

#include "config.h"
#include "depth3d_spv.h"
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
  X(CmdBeginRenderingKHR) X(CmdCopyBuffer) X(GetSemaphoreCounterValue) X(GetSemaphoreCounterValueKHR) \
  X(CreateImageView) X(CmdEndRendering) X(CmdEndRenderingKHR) X(CmdDraw) X(CmdDrawIndexed) X(CmdDrawIndirect) \
  X(CmdDrawIndexedIndirect) X(CmdDrawIndirectCount) X(CmdDrawIndexedIndirectCount) X(CmdPipelineBarrier2) \
  X(CmdPipelineBarrier2KHR) X(CmdBindPipeline) X(CmdPushConstants) X(CmdDispatch) X(CmdFillBuffer) \
  X(CmdCopyBufferToImage) X(CreatePipelineLayout) X(DestroyPipelineLayout) X(CreateComputePipelines) \
  X(CmdCopyImage2) X(CmdCopyImage2KHR) X(CmdBlitImage2) X(CmdBlitImage2KHR) X(CmdResolveImage) \
  X(CmdResolveImage2) X(CmdResolveImage2KHR) X(CmdExecuteCommands) X(CmdDrawMultiEXT) X(CmdDrawMultiIndexedEXT)

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
  // ---- depth / hybrid modes: eyes rebuilt from one game frame
  VkBuffer work = VK_NULL_HANDLE;          // colour | keys (2 eyes) | eyes (2) | last real eyes (2, hybrid)
  VkDeviceMemory workMem = VK_NULL_HANDLE;
  VkDeviceAddress workAddr = 0;
  bool prevValid[2] = {false, false};      // hybrid: the last real image of that eye is in the work buffer
  VkImage synth[2] = {VK_NULL_HANDLE, VK_NULL_HANDLE};  // the two eyes, game format and size
  VkDeviceMemory synthMem[2] = {VK_NULL_HANDLE, VK_NULL_HANDLE};
  bool depthBroken = false;
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
  bool screenFix = false;    // fix_shadows: screen-space position rebuilds corrected (fragment / compute)
  bool unshiftD16 = false;   // fix_shadows: 16-bit depth-only pipelines (Unreal shadow maps) left unpatched
  std::atomic<uint32_t> screenFixCount{0}, depthOnlyCount{0};
  uint32_t lastFixLog[2] = {UINT32_MAX, UINT32_MAX};
  bool lastFixOn = true;
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

  // ---- depth / hybrid modes (from 3dvisionlinux): the game renders one image per frame; the eyes
  // are rebuilt from it and its depth buffer (copied inside the game's command buffers)
  bool depthMode = false;
  bool hybrid = false;        // each frame renders one real eye, the other one is rebuilt
  void* paramMap = nullptr;   // persistently mapped parameter buffer (projection samples)
  std::atomic<uint64_t> lastDepthFrame{UINT64_MAX};
  std::atomic<uint32_t> calibGroup{0};     // stages so far in the frame being recorded (reset every frame)
  VkImage stageImg[4] = {};                // the depth image of each stage (under depthLock)
  uint32_t curStage = 0;                   // stage of the last scene pass (under depthLock)
  std::atomic<uint64_t> stageFrameAt{UINT64_MAX};  // last frame whose start was seen in the passes
  std::atomic<uint32_t> lastCopyGroup{0};
  std::atomic<uint32_t> copiedStages{0};
  uint64_t lastClassFrame = 0, lastFitFrame = 0, lastFlipCheck = 0;
  uint32_t depthFlip = 0;
  int flipVotes = 0;
  std::mutex depthLock;
  struct DepthImg { VkFormat fmt; VkExtent2D ext; bool single; };
  std::unordered_map<VkImage, DepthImg> depthImgs;
  struct ViewInfo { VkImage img; uint32_t mip, layer; };
  std::unordered_map<VkImageView, ViewInfo> depthViews;
  // HUD separation (depth / hybrid): the game's own back buffer gets the finished scene first and
  // the HUD on top; a copy taken in between is the scene without HUD (under depthLock)
  struct ColorImg { VkFormat fmt; VkExtent2D ext; uint32_t samples; };
  std::unordered_map<VkImage, ColorImg> colorImgs;
  std::unordered_map<VkImageView, VkImage> colorViews;
  std::unordered_set<VkImage> presentedImgs;   // the layer's virtual swapchain images
  VkExtent2D gameExt{};
  VkFormat scFormat = VK_FORMAT_UNDEFINED;
  VkImage lastColorImg = VK_NULL_HANDLE;       // colour target of the last game-sized pass
  VkImage bbCand = VK_NULL_HANDLE, bbImg = VK_NULL_HANDLE;
  int bbVotes = 0;
  bool bbUsable = false, bbSwapRB = false, bb10 = false;
  VkImage bbResolve = VK_NULL_HANDLE;          // a multisampled back buffer is resolved into this first
  VkDeviceMemory bbResolveMem = VK_NULL_HANDLE;
  VkFormat bbResolveFmt = VK_FORMAT_UNDEFINED;
  VkExtent2D bbResolveExt{};
  uint32_t bbSamples = 1;
  std::vector<std::pair<VkImage, VkDeviceMemory>> retiredImgs;
  VkBuffer snapBuf = VK_NULL_HANDLE;
  VkDeviceMemory snapMem = VK_NULL_HANDLE;
  VkDeviceAddress snapAddr = 0;
  VkDeviceSize snapSize = 0, snapSlotBytes = 0;  // kSnapSlots copies (the last frames')
  std::atomic<uint32_t> snapNext{0};
  std::atomic<uint64_t> slotFrame[3] = {{0}, {0}, {0}};
  std::atomic<bool> snapDone{false}, bbFilled{false};
  std::atomic<uint32_t> bbDraws{0};           // draws (and copies) into the back buffer in this frame so far
  std::atomic<uint64_t> depthFrameAt{UINT64_MAX};  // last frame the scene depth was cleared
  std::atomic<uint32_t> snapTaken{0};
  bool snapBad = false, loggedSnap = false;
  uint32_t autoSplit = 1, snapTried = 0, snapMatched = 0;  // the copy is taken after back buffer draw 'split'
  int lastHudCfg = -1;
  uint64_t snapBadUntil = 0;
  VkBuffer depthBuf = VK_NULL_HANDLE;
  VkDeviceMemory depthMem = VK_NULL_HANDLE;
  VkDeviceAddress depthAddr = 0;
  VkDeviceSize depthCap = 0;
  VkBuffer metaBuf = VK_NULL_HANDLE;
  VkDeviceMemory metaMem = VK_NULL_HANDLE;
  VkDeviceAddress metaAddr = 0;
  std::vector<std::pair<VkBuffer, VkDeviceMemory>> retiredBufs;
  std::atomic<float> depthAspect{0.0f}, depthArea{0.0f};
  std::atomic<uint32_t> depthCopies{0};
  std::atomic<uint32_t> rebuiltFrames{0};
  std::atomic<bool> loggedRenderPass{false};
  VkPipelineLayout synthLayout = VK_NULL_HANDLE;
  VkPipeline synthPipe[3] = {VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE};
  struct Proj {  // projection of each scene stage: z_ndc = A + B / w
    float A = 0, B = 0;
    bool ok = false;
    int switchVotes = 0;
    uint32_t inliers = 0, samples = 0;
  } proj[4];
  std::unordered_map<uint64_t, uint64_t> classDraws;  // under depthLock
  std::atomic<uint64_t> mainClass{0};
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

// Unreal Engine games ship as <Project>-Win64-Shipping.exe (also -WinGDK-, -Linux-...)
bool IsUnrealGame() {
  const std::string& g = GameName();
  return g.size() > 13 && strcasecmp(g.c_str() + g.size() - 13, "-Shipping.exe") == 0;
}
float ScreenFixFactor() { return (Cfg().fixShadows != 0 && Cfg().fixShadowsOn) ? 1.0f : 0.0f; }

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
  bci.size = 4096;  // params | identity | projection samples (kCalibOffset) | stats (kStatsOffset)
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
    memset(map, 0, 4096);
    EyeParams id = IdentityParams();
    for (int o : {0, 256, 512}) memcpy((char*)map + o, id.m, sizeof(id.m));
    float f = ScreenFixFactor();
    memcpy((char*)map + kFixFactorOffset, &f, 4);
    dd->paramMap = map;  // stays mapped: the projection samples are read from it
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
bool InitDepthMode(DeviceData* dd);
void DestroyDepthMode(DeviceData* dd);
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
  bool supported = false, vtxStores = false;
  auto gpf2 = inst->d.GetPhysicalDeviceFeatures2 ? inst->d.GetPhysicalDeviceFeatures2
                                                 : inst->d.GetPhysicalDeviceFeatures2KHR;
  if (gpf2) {
    VkPhysicalDeviceBufferDeviceAddressFeatures bda{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES};
    VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &bda};
    gpf2(phys, &f2);
    supported = bda.bufferDeviceAddress && f2.features.shaderInt64;
    vtxStores = f2.features.vertexPipelineStoresAndAtomics;
  }
  // Proton (DXVK / VKD3D-Proton) always uses Vulkan 1.3. For 1.0 apps BDA would need extra
  // instance extensions, so only Vulkan >= 1.1 is handled (STEREO3D_LEGACY=1 forces it).
  const char* legacy = getenv("STEREO3D_LEGACY");
  bool apiOk = api >= VK_API_VERSION_1_1 || (legacy && legacy[0] == '1');
  bool wantStereo = Cfg().patchShaders && supported && apiOk;
  // depth / hybrid: patched shaders store projection samples (vertex-stage stores), the eyes are
  // rebuilt by compute shaders (SPIR-V 1.5: Vulkan 1.2)
  const bool depthLike = Cfg().mode == 2 || Cfg().mode == 3;
  const bool wantDepth = wantStereo && depthLike && vtxStores && api >= VK_API_VERSION_1_2 && !Cfg().passthrough;
  if (wantStereo && depthLike && !wantDepth && !Cfg().passthrough)
    Log(0, "%s mode needs Vulkan 1.2 and vertexPipelineStoresAndAtomics: using alternate", ModeName(Cfg().mode));

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
          if (wantDepth) ((VkPhysicalDeviceFeatures2*)p)->features.vertexPipelineStoresAndAtomics = VK_TRUE;
          haveI64 = true;
          break;
        default: break;
      }
    }
    if (!haveI64) {
      if (ci.pEnabledFeatures) featCopy = *ci.pEnabledFeatures;
      featCopy.shaderInt64 = VK_TRUE;
      if (wantDepth) featCopy.vertexPipelineStoresAndAtomics = VK_TRUE;
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
    dd->depthMode = wantDepth;
    dd->hybrid = wantDepth && Cfg().mode == 3;
    dd->stereoShaders = CreateParamBuffer(dd);
    dd->replay = dd->stereoShaders && Cfg().mode == 1;
    // fix_shadows: screen-space rebuilds corrected in every game (the pattern is specific); 16-bit
    // depth-only pipelines (shadow maps) left unpatched only in Unreal Engine games (or fix_shadows = 1)
    dd->screenFix = dd->stereoShaders && dd->paramMap && Cfg().fixShadows != 0;
    dd->unshiftD16 = dd->screenFix && (Cfg().fixShadows == 1 || IsUnrealGame());
    if (dd->unshiftD16) Log(1, "fix_shadows: Unreal Engine shadow maps (16-bit depth) without eye transform");
    if (!dd->stereoShaders) {
      Log(0, "failed to create stereo parameter buffer");
      dd->depthMode = dd->hybrid = false;
    }
    if (dd->depthMode && (!dd->paramMap || !InitDepthMode(dd))) {
      Log(0, "depth mode: setup failed, using alternate");
      dd->depthMode = dd->hybrid = false;
    }
  }
  Log(1, "device '%s' api %u.%u: shader stereo %s, mode %s, param @0x%llx", props.deviceName,
      VK_API_VERSION_MAJOR(api), VK_API_VERSION_MINOR(api), dd->stereoShaders ? "ON" : "OFF",
      dd->hybrid ? "hybrid (L/R frames, the other eye of each frame rebuilt from its depth)"
      : dd->depthMode ? "depth (both eyes from each game frame + its depth buffer)"
      : dd->replay ? "replay (both eyes every frame)" : "alternate (L/R frames)", (unsigned long long)dd->paramAddr);

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
  if (dd->depthMode) DestroyDepthMode(dd);
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
void NoteDestroyImage(DeviceData* dd, VkImage img);
void NoteDestroyImageView(DeviceData* dd, VkImageView v);
inline void NoteDestroy(DeviceData*, VkBuffer) {}
inline void NoteDestroy(DeviceData* dd, VkImage h) { NoteDestroyImage(dd, h); }
inline void NoteDestroy(DeviceData* dd, VkImageView h) { NoteDestroyImageView(dd, h); }
template <class T> inline void NoteDestroy(DeviceData*, T) {}
#define DESTROY_HOOK(Name, T)                                                                  \
  VKAPI_ATTR void VKAPI_CALL hook_##Name(VkDevice device, T h, const VkAllocationCallbacks* a) { \
    DeviceData* dd = Dev(device);                                                              \
    NoteDestroy(dd, h);                                                                        \
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
      (dd->screenFix || HasPreRasterEntryPoint(ci->pCode, ci->codeSize / 4))) {
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
  if (model == 4 || model == 5) {  // fragment / compute: screen-space position rebuilds
    PatchResult pr = PatchScreenSpace(code, words, dd->paramAddr, out, &log);
    if (pr != PatchResult::Patched) return false;
    uint32_t n = ++dd->screenFixCount;
    Log(1, "fix_shadows: %s shader #%u corrected (%s)", model == 4 ? "fragment" : "compute", n, log.c_str());
    return true;
  }
  PatchResult pr = PatchStereo(code, words, entry, model, dd->paramAddr, out, &log, dd->depthMode);
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
// ------------------------------------------------------------------------------------
// Depth / hybrid modes (ported from 3dvisionlinux). The game renders one image per frame (depth:
// from the middle of the eyes; hybrid: from one eye, alternating). The scene's depth buffer is
// copied right after the main scene pass inside the game's own command buffers; the depth ->
// distance mapping comes from (clip w, clip z) samples the patched vertex shaders store into the
// parameter buffer. At present two compute passes rebuild the eyes (src/shaders/depth3d.comp).
// Only dynamic rendering (DXVK, VKD3D-Proton) is followed.
// ------------------------------------------------------------------------------------
VkDeviceAddress BufAddr(DeviceData* dd, VkBuffer buf) {
  VkBufferDeviceAddressInfo bai{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
  bai.buffer = buf;
  auto gbda = (dd->apiVersion >= VK_API_VERSION_1_2 && dd->d.GetBufferDeviceAddress)
                  ? dd->d.GetBufferDeviceAddress : dd->d.GetBufferDeviceAddressKHR;
  if (!gbda) gbda = dd->d.GetBufferDeviceAddress;
  return gbda ? gbda(dd->device, &bai) : 0;
}

bool CreateDevBuf(DeviceData* dd, VkDeviceSize size, VkBufferUsageFlags usage, VkBuffer* buf, VkDeviceMemory* mem,
                  VkDeviceAddress* addr) {
  VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  bci.size = size;
  bci.usage = usage | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
  bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  if (dd->d.CreateBuffer(dd->device, &bci, nullptr, buf) != VK_SUCCESS) return false;
  VkMemoryRequirements req;
  dd->d.GetBufferMemoryRequirements(dd->device, *buf, &req);
  VkMemoryAllocateFlagsInfo fi{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO};
  fi.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
  VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, &fi};
  ai.allocationSize = req.size;
  ai.memoryTypeIndex = FindMemType(dd, req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  if (ai.memoryTypeIndex == UINT32_MAX || dd->d.AllocateMemory(dd->device, &ai, nullptr, mem) != VK_SUCCESS) {
    dd->d.DestroyBuffer(dd->device, *buf, nullptr);
    *buf = VK_NULL_HANDLE;
    return false;
  }
  dd->d.BindBufferMemory(dd->device, *buf, *mem, 0);
  if (addr) *addr = BufAddr(dd, *buf);
  return true;
}

bool IsDepthFormat(VkFormat f) {
  switch (f) {
    case VK_FORMAT_D16_UNORM: case VK_FORMAT_X8_D24_UNORM_PACK32: case VK_FORMAT_D32_SFLOAT:
    case VK_FORMAT_D16_UNORM_S8_UINT: case VK_FORMAT_D24_UNORM_S8_UINT: case VK_FORMAT_D32_SFLOAT_S8_UINT:
      return true;
    default:
      return false;
  }
}
bool HasStencil(VkFormat f) {
  return f == VK_FORMAT_D16_UNORM_S8_UINT || f == VK_FORMAT_D24_UNORM_S8_UINT || f == VK_FORMAT_D32_SFLOAT_S8_UINT;
}
// how the depth aspect looks in a buffer copy: 0 float32, 1 unorm24 in 32 bits, 2 unorm16
uint32_t DepthCode(VkFormat f) {
  if (f == VK_FORMAT_D16_UNORM || f == VK_FORMAT_D16_UNORM_S8_UINT) return 2;
  if (f == VK_FORMAT_X8_D24_UNORM_PACK32 || f == VK_FORMAT_D24_UNORM_S8_UINT) return 1;
  return 0;
}
bool Is32bpp(VkFormat f) {
  switch (f) {
    case VK_FORMAT_B8G8R8A8_UNORM: case VK_FORMAT_B8G8R8A8_SRGB: case VK_FORMAT_R8G8B8A8_UNORM:
    case VK_FORMAT_R8G8B8A8_SRGB: case VK_FORMAT_A8B8G8R8_UNORM_PACK32: case VK_FORMAT_A8B8G8R8_SRGB_PACK32:
    case VK_FORMAT_A2B10G10R10_UNORM_PACK32: case VK_FORMAT_A2R10G10B10_UNORM_PACK32:
      return true;
    default:
      return false;
  }
}

struct SynthPC {  // push constants of depth3d.comp (std430 offsets)
  uint64_t color, depth, keys, outc, meta;
  uint32_t W, H;
  float a, b, maxShift;
  uint32_t eyes;
  uint64_t prev;
  uint32_t hyb, pad;
  uint64_t stats;
  uint64_t snap;        // HUD separation: copies of the game's back buffer before the HUD
  uint32_t snapFlags;   // 1 copies available, 2 red / blue swapped, 4 10-bit copies
  uint32_t snapStride;  // words between copies
  uint32_t snapMask;    // slots holding a recent copy
  uint32_t pad2;
};
static_assert(sizeof(SynthPC) == 112, "push constant layout");
constexpr uint32_t kSnapSlots = 3;
constexpr uint32_t kMetaWords = 68;  // see depth3d.comp

bool InitDepthMode(DeviceData* dd) {
  if (!CreateDevBuf(dd, 512, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, &dd->metaBuf,
                    &dd->metaMem, &dd->metaAddr) || !dd->metaAddr)
    return false;
  VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(SynthPC)};
  VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  plci.pushConstantRangeCount = 1;
  plci.pPushConstantRanges = &pcr;
  if (dd->d.CreatePipelineLayout(dd->device, &plci, nullptr, &dd->synthLayout) != VK_SUCCESS) return false;
  const uint32_t* code[3] = {kDepthSplatSpv, kDepthResolveSpv, kDepthMatchSpv};
  const size_t size[3] = {sizeof(kDepthSplatSpv), sizeof(kDepthResolveSpv), sizeof(kDepthMatchSpv)};
  for (int i = 0; i < 3; ++i) {
    VkShaderModuleCreateInfo smci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    smci.codeSize = size[i];
    smci.pCode = code[i];
    VkShaderModule m;
    if (dd->d.CreateShaderModule(dd->device, &smci, nullptr, &m) != VK_SUCCESS) return false;
    VkComputePipelineCreateInfo cpci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cpci.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpci.stage.module = m;
    cpci.stage.pName = "main";
    cpci.layout = dd->synthLayout;
    VkResult r = dd->d.CreateComputePipelines(dd->device, VK_NULL_HANDLE, 1, &cpci, nullptr, &dd->synthPipe[i]);
    dd->d.DestroyShaderModule(dd->device, m, nullptr);
    if (r != VK_SUCCESS) return false;
  }
  return true;
}

void DestroyDepthMode(DeviceData* dd) {
  VkDevice dev = dd->device;
  for (auto& p : dd->synthPipe)
    if (p) dd->d.DestroyPipeline(dev, p, nullptr);
  if (dd->synthLayout) dd->d.DestroyPipelineLayout(dev, dd->synthLayout, nullptr);
  auto freeBuf = [&](VkBuffer b, VkDeviceMemory m) {
    if (b) dd->d.DestroyBuffer(dev, b, nullptr);
    if (m) dd->d.FreeMemory(dev, m, nullptr);
  };
  freeBuf(dd->depthBuf, dd->depthMem);
  freeBuf(dd->snapBuf, dd->snapMem);
  freeBuf(dd->metaBuf, dd->metaMem);
  for (auto& r : dd->retiredBufs) freeBuf(r.first, r.second);
  dd->retiredBufs.clear();
  if (dd->bbResolve) dd->retiredImgs.push_back({dd->bbResolve, dd->bbResolveMem});
  dd->bbResolve = VK_NULL_HANDLE;
  for (auto& r : dd->retiredImgs) {
    dd->d.DestroyImage(dev, r.first, nullptr);
    if (r.second) dd->d.FreeMemory(dev, r.second, nullptr);
  }
  dd->retiredImgs.clear();
}

// the depth copy targets: one per scene stage (4), each big enough for 1.5x the game image in each direction
void EnsureDepthBuffer(DeviceData* dd, VkExtent2D game) {
  VkDeviceSize need = (VkDeviceSize(game.width) * game.height * 4 * 9 / 4 + 255) & ~VkDeviceSize(255);
  std::lock_guard<std::mutex> l(dd->depthLock);
  dd->depthAspect = float(game.width) / float(std::max(1u, game.height));
  dd->depthArea = float(game.width) * float(game.height);
  if (need <= dd->depthCap) return;
  VkBuffer b;
  VkDeviceMemory m;
  VkDeviceAddress a;
  if (!CreateDevBuf(dd, need * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, &b, &m, &a)) {
    Log(0, "depth mode: cannot allocate %llu bytes for the depth copies", (unsigned long long)need * 4);
    return;
  }
  if (dd->depthBuf) dd->retiredBufs.push_back({dd->depthBuf, dd->depthMem});  // may be in recorded command buffers
  dd->depthBuf = b;
  dd->depthMem = m;
  dd->depthAddr = a;
  dd->depthCap = need;
}

// per command buffer: the depth attachment of the current dynamic rendering pass
struct CbDepth {
  VkImage img = VK_NULL_HANDLE;
  VkFormat fmt = VK_FORMAT_UNDEFINED;
  VkExtent2D ext{};
  VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED;
  uint32_t mip = 0, layer = 0, draws = 0;
  bool active = false, suspending = false, secondary = false;
  bool clears = false;  // the pass clears the depth (loadOp CLEAR)
  int flag = -1;        // calibration flag last set in this command buffer (-1: not yet)
  // the scene depth to copy: once per stage, just before the game clears it for the next stage or
  // when the command buffer ends (not after every pass: EDF 6 has ~30 scene passes per frame)
  bool pending = false;
  VkImage pImg = VK_NULL_HANDLE;
  VkFormat pFmt = VK_FORMAT_UNDEFINED;
  VkExtent2D pExt{};
  VkImageLayout pLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  uint32_t pMip = 0, pLayer = 0, pStage = 0;
  uint32_t stage = 0;  // scene stage of the current pass
  // the current pass draws into the game's back buffer (HUD separation)
  bool bbPass = false, bbKhr = false, bbCanSplit = false, bbSecondary = false;
  VkImage bbImg = VK_NULL_HANDLE;
  VkImageLayout bbLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  VkRenderingInfo bbInfo{};
  std::vector<VkRenderingAttachmentInfo> bbColors;
  VkRenderingAttachmentInfo bbDepth{}, bbStencil{};
};
std::mutex g_cbLock;
std::unordered_map<VkCommandBuffer, CbDepth> g_cbDepth;
struct TlsCb { VkCommandBuffer cb = VK_NULL_HANDLE; CbDepth* st = nullptr; DeviceData* dd = nullptr; };
thread_local TlsCb t_cb;

inline TlsCb& CbState(VkCommandBuffer cb) {
  if (t_cb.cb != cb) {
    DeviceData* dd = Dev(cb);
    std::lock_guard<std::mutex> l(g_cbLock);
    t_cb.st = &g_cbDepth[cb];  // (entries stay: games reuse their command buffers)
    t_cb.cb = cb;
    t_cb.dd = dd;
  }
  return t_cb;
}

VKAPI_ATTR VkResult VKAPI_CALL hook_CreateImage(VkDevice device, const VkImageCreateInfo* ci,
                                                const VkAllocationCallbacks* a, VkImage* img) {
  DeviceData* dd = Dev(device);
  bool depth = dd->depthMode && IsDepthFormat(ci->format) && ci->imageType == VK_IMAGE_TYPE_2D &&
               (ci->usage & VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT);
  bool color = dd->depthMode && !depth && ci->imageType == VK_IMAGE_TYPE_2D &&
               (ci->usage & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT) && ci->mipLevels >= 1;
  if (color) {  // possibly the game's back buffer: the layer may copy it (scene before the HUD)
    VkImageCreateInfo c = *ci;
    c.usage |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;  // (multisampled: the source of a resolve)
    VkResult r = dd->d.CreateImage(device, &c, a, img);
    if (r != VK_SUCCESS) r = dd->d.CreateImage(device, ci, a, img);
    if (r != VK_SUCCESS) return r;
    std::lock_guard<std::mutex> l(dd->depthLock);
    dd->colorImgs[*img] = {ci->format, {ci->extent.width, ci->extent.height}, uint32_t(ci->samples)};
    return r;
  }
  if (!depth) return dd->d.CreateImage(device, ci, a, img);
  VkImageCreateInfo c = *ci;
  c.usage |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;  // so the layer can copy it
  VkResult r = dd->d.CreateImage(device, &c, a, img);
  if (r != VK_SUCCESS) return dd->d.CreateImage(device, ci, a, img);
  std::lock_guard<std::mutex> l(dd->depthLock);
  dd->depthImgs[*img] = {ci->format, {ci->extent.width, ci->extent.height}, ci->samples == VK_SAMPLE_COUNT_1_BIT};
  return r;
}
void NoteDestroyImage(DeviceData* dd, VkImage img) {
  if (!dd->depthMode || !img) return;
  std::lock_guard<std::mutex> l(dd->depthLock);
  dd->depthImgs.erase(img);
  dd->colorImgs.erase(img);
  if (img == dd->bbImg) dd->bbImg = VK_NULL_HANDLE, dd->bbUsable = false;
  if (img == dd->bbCand) dd->bbCand = VK_NULL_HANDLE;
  if (img == dd->lastColorImg) dd->lastColorImg = VK_NULL_HANDLE;
}
void NoteDestroyImageView(DeviceData* dd, VkImageView v) {
  if (!dd->depthMode || !v) return;
  std::lock_guard<std::mutex> l(dd->depthLock);
  dd->depthViews.erase(v);
  dd->colorViews.erase(v);
}
VKAPI_ATTR VkResult VKAPI_CALL hook_CreateImageView(VkDevice device, const VkImageViewCreateInfo* ci,
                                                    const VkAllocationCallbacks* a, VkImageView* view) {
  DeviceData* dd = Dev(device);
  VkResult r = dd->d.CreateImageView(device, ci, a, view);
  if (r == VK_SUCCESS && dd->depthMode) {
    std::lock_guard<std::mutex> l(dd->depthLock);
    if (dd->depthImgs.count(ci->image))
      dd->depthViews[*view] = {ci->image, ci->subresourceRange.baseMipLevel, ci->subresourceRange.baseArrayLayer};
    else if (dd->colorImgs.count(ci->image) || dd->presentedImgs.count(ci->image))
      dd->colorViews[*view] = ci->image;
  }
  return r;
}

void BeginRenderingCommon(TlsCb& t, const VkRenderingInfo* ri) {
  CbDepth& st = *t.st;
  DeviceData* dd = t.dd;
  bool resuming = (ri->flags & VK_RENDERING_RESUMING_BIT) != 0;
  if (!resuming) {
    st.active = false;
    st.draws = 0;
  }
  st.suspending = (ri->flags & VK_RENDERING_SUSPENDING_BIT) != 0;
  st.secondary = (ri->flags & VK_RENDERING_CONTENTS_SECONDARY_COMMAND_BUFFERS_BIT) != 0;
  if (resuming || !dd || !dd->depthMode || !ri->pDepthAttachment || !ri->pDepthAttachment->imageView) return;
  std::lock_guard<std::mutex> l(dd->depthLock);
  auto v = dd->depthViews.find(ri->pDepthAttachment->imageView);
  if (v == dd->depthViews.end()) return;
  auto im = dd->depthImgs.find(v->second.img);
  if (im == dd->depthImgs.end() || !im->second.single) return;
  VkExtent2D e{std::max(1u, im->second.ext.width >> v->second.mip), std::max(1u, im->second.ext.height >> v->second.mip)};
  float asp = float(e.width) / float(e.height), want = dd->depthAspect.load();
  // the scene's depth has (about) the shape and size of the game image; shadow maps, reflections
  // and other small passes do not (up to 30% off: a 16:10 or 4:3 scene with black bars in 16:9)
  if (want <= 0 || std::fabs(asp / want - 1.0f) > 0.30f) return;
  if (float(e.width) * float(e.height) < 0.4f * dd->depthArea.load()) return;
  VkDeviceSize bytes = VkDeviceSize(e.width) * e.height * (DepthCode(im->second.fmt) == 2 ? 2 : 4);
  if (bytes > dd->depthCap || e.width < 320) return;
  st.img = v->second.img;
  st.fmt = im->second.fmt;
  st.ext = e;
  st.layout = ri->pDepthAttachment->imageLayout;
  st.clears = ri->pDepthAttachment->loadOp == VK_ATTACHMENT_LOAD_OP_CLEAR;
  st.mip = v->second.mip;
  st.layer = v->second.layer;
  st.active = true;
  // The game clears its scene depth: a new frame begins (recorded in the game's order, unlike the
  // present: DXVK records that on its own thread, often after the next frame's first draws). The
  // HUD separation takes its next copy in this frame.
  const uint64_t cls = (uint64_t(e.width) << 40) | (uint64_t(e.height) << 20) | uint64_t(st.fmt & 0xFFFFF);
  const uint64_t main = dd->mainClass.load();
  if (st.clears && main && main == cls) {
    dd->depthFrameAt = dd->frame;
    if (dd->snapDone.load()) {
      dd->snapDone = false;
      dd->bbFilled = false;
      dd->bbDraws = 0;
    }
  }
}

// copy a scene depth into the layer's buffer, inside the game's command buffer
void RecordDepthCopy(DeviceData* dd, VkCommandBuffer cb, const CbDepth& st, uint32_t stage) {
  VkBuffer dst;
  VkDeviceSize stride;
  {
    std::lock_guard<std::mutex> l(dd->depthLock);
    dst = dd->depthBuf;
    stride = dd->depthCap;
  }
  if (!dst) return;
  const VkImageAspectFlags aspects = VK_IMAGE_ASPECT_DEPTH_BIT | (HasStencil(st.fmt) ? VK_IMAGE_ASPECT_STENCIL_BIT : 0);
  const bool general = st.layout == VK_IMAGE_LAYOUT_GENERAL;
  const VkImageLayout copyLayout = general ? VK_IMAGE_LAYOUT_GENERAL : VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
  const VkPipelineStageFlags DEPTH_STAGES =
      VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
  VkImageMemoryBarrier ib{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
  ib.srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
  ib.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
  ib.oldLayout = st.layout;
  ib.newLayout = copyLayout;
  ib.srcQueueFamilyIndex = ib.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  ib.image = st.img;
  ib.subresourceRange = {aspects, st.mip, 1, st.layer, 1};
  VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};  // earlier copies / readers of our buffers
  mb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT;
  mb.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  dd->d.CmdPipelineBarrier(cb, DEPTH_STAGES | VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                           VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &mb, 0, nullptr, 1, &ib);
  stage &= 3;
  VkBufferImageCopy r{};
  r.bufferOffset = stride * stage;
  r.imageSubresource = {VK_IMAGE_ASPECT_DEPTH_BIT, st.mip, st.layer, 1};
  r.imageExtent = {st.ext.width, st.ext.height, 1};
  dd->d.CmdCopyImageToBuffer(cb, st.img, copyLayout, dst, 1, &r);
  uint32_t meta[3] = {st.ext.width, st.ext.height, DepthCode(st.fmt)};
  dd->d.CmdUpdateBuffer(cb, dd->metaBuf, 16 * stage, sizeof(meta), meta);
  dd->copiedStages |= 1u << stage;
  ib.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
  ib.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT |
                     VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT;
  ib.oldLayout = copyLayout;
  ib.newLayout = st.layout;
  mb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
  dd->d.CmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &mb, 0,
                           nullptr, 1, &ib);
  ++dd->depthCopies;
  dd->lastDepthFrame = dd->frame;
  dd->lastCopyGroup = stage;
}

void FlushDepthCopy(DeviceData* dd, VkCommandBuffer cb, CbDepth& st) {
  if (!st.pending) return;
  st.pending = false;
  CbDepth c;
  c.img = st.pImg;
  c.fmt = st.pFmt;
  c.ext = st.pExt;
  c.layout = st.pLayout;
  c.mip = st.pMip;
  c.layer = st.pLayer;
  RecordDepthCopy(dd, cb, c, st.pStage);
}
// the game changes the layout of the pending depth image: follow it (or drop it if discarded)
void NoteDepthBarrier(CbDepth& st, VkImage img, VkImageLayout oldL, VkImageLayout newL) {
  if (!st.pending || img != st.pImg) return;
  if (oldL == VK_IMAGE_LAYOUT_UNDEFINED) st.pending = false;
  else st.pLayout = newL;
}

// calibration flag: 1 + stage inside the scene's passes, 0 elsewhere (shadow maps, reflections...
// have other cameras). Recorded in the game's command buffer, outside render passes.
void SetCalibFlag(DeviceData* dd, VkCommandBuffer cb, float v) {
  VkMemoryBarrier pre{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
  pre.srcAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
  pre.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  dd->d.CmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &pre, 0,
                           nullptr, 0, nullptr);
  dd->d.CmdUpdateBuffer(cb, dd->paramBuf, kCalibFlagOffset, 4, &v);
  VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
  mb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
  dd->d.CmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &mb, 0,
                           nullptr, 0, nullptr);
}

inline uint64_t DepthClass(const CbDepth& st) {
  return (uint64_t(st.ext.width) << 40) | (uint64_t(st.ext.height) << 20) | uint64_t(st.fmt & 0xFFFFF);
}
void EndRenderingCommon(TlsCb& t, VkCommandBuffer cb) {
  CbDepth& st = *t.st;
  if (!st.active || st.suspending) return;
  st.active = false;
  DeviceData* dd = t.dd;
  const uint64_t cls = DepthClass(st);
  {
    std::lock_guard<std::mutex> l(dd->depthLock);
    dd->classDraws[cls] += st.draws + (st.secondary ? 8 : 0);
  }
  const uint64_t main = dd->mainClass.load();
  const bool readOnly = st.layout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL ||
                        st.layout == VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_STENCIL_ATTACHMENT_OPTIMAL ||
                        st.layout == VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL ||
                        st.layout == VK_IMAGE_LAYOUT_READ_ONLY_OPTIMAL;
  if ((main == 0 || main == cls) && (st.draws >= 8 || st.secondary) && !readOnly) {
    if (st.pending && st.pImg != st.img) FlushDepthCopy(dd, cb, st);  // another image: keep that one too
    st.pending = true;
    st.pImg = st.img;
    st.pFmt = st.fmt;
    st.pExt = st.ext;
    st.pLayout = st.layout;
    st.pMip = st.mip;
    st.pLayer = st.layer;
    st.pStage = st.stage;
  }
}
// every few frames: the depth class with the most draws is the scene
void UpdateMainDepthClass(DeviceData* dd) {
  std::lock_guard<std::mutex> l(dd->depthLock);
  uint64_t best = 0, bestDraws = 0;
  for (auto& kv : dd->classDraws)
    if (kv.second > bestDraws) { best = kv.first; bestDraws = kv.second; }
  if (best && best != dd->mainClass.load()) {
    Log(1, "depth mode: scene depth is %llux%llu", (unsigned long long)(best >> 40),
        (unsigned long long)((best >> 20) & 0xFFFFF));
    dd->mainClass = best;
  }
  dd->classDraws.clear();
}

// flag = -2: the passes drawing the HUD over the copied scene (HUD separation) render with the
// parameters of the middle of the eyes (kHudParamsOffset, read by the patched vertex shaders), so the
// HUD has no disparity in the real eyes either: the rebuilt eye shows it where the real one does.
constexpr int kFlagNoShift = -2;
void SetHudFlag(DeviceData* dd, VkCommandBuffer cb, CbDepth& st, bool on) {
  if (!dd->paramBuf || !dd->depthMode) return;
  const int want = on ? kFlagNoShift : 0;
  if (st.flag == want || (!on && st.flag != kFlagNoShift)) return;
  SetCalibFlag(dd, cb, float(want));
  st.flag = want;
}

// ------------------------------------------------------------------------------------
// HUD separation (depth / hybrid). DXVK / VKD3D games draw into their own back buffer: first the
// finished scene (one full-screen draw or a copy), then the HUD on top; at present that back buffer
// is copied into the image the layer presents. The layer learns which image that is, copies it
// right after the scene arrived (splitting the game's pass after its first draw if needed), and
// the eyes are rebuilt from that copy, with what was drawn after it (HUD, crosshair, markers,
// menus) put back on the screen plane.
// ------------------------------------------------------------------------------------
inline bool IsRGBA8(VkFormat f) {
  return f == VK_FORMAT_R8G8B8A8_UNORM || f == VK_FORMAT_R8G8B8A8_SRGB || f == VK_FORMAT_A8B8G8R8_UNORM_PACK32 ||
         f == VK_FORMAT_A8B8G8R8_SRGB_PACK32;
}
inline bool IsBGRA8(VkFormat f) { return f == VK_FORMAT_B8G8R8A8_UNORM || f == VK_FORMAT_B8G8R8A8_SRGB; }
inline bool Is10Bit(VkFormat f) {
  return f == VK_FORMAT_A2B10G10R10_UNORM_PACK32 || f == VK_FORMAT_A2R10G10B10_UNORM_PACK32;
}
// red in the lowest bits of the 32-bit pixel
inline bool LowIsRed(VkFormat f) { return IsRGBA8(f) || f == VK_FORMAT_A2B10G10R10_UNORM_PACK32; }
// a format a back buffer can have (what is shown); HDR scene targets (16-bit float, B10G11R11) are not
inline bool IsDisplayFormat(VkFormat f) {
  return IsRGBA8(f) || IsBGRA8(f) || f == VK_FORMAT_A2B10G10R10_UNORM_PACK32 || f == VK_FORMAT_A2R10G10B10_UNORM_PACK32;
}

bool CreateImageMem(DeviceData* dd, VkImageCreateInfo ici, VkImage* img, VkDeviceMemory* mem);
// an image whose content went into the presented image (caller holds depthLock)
void VoteBackbufferLocked(DeviceData* dd, VkImage cand) {
  if (!cand || dd->presentedImgs.count(cand)) return;
  auto cf = dd->colorImgs.find(cand);
  if (cf == dd->colorImgs.end() || !IsDisplayFormat(cf->second.fmt)) return;
  if (cand == dd->bbCand) ++dd->bbVotes;
  else dd->bbCand = cand, dd->bbVotes = 1;
  if (dd->bbVotes < 3 || cand == dd->bbImg) return;
  dd->bbImg = cand;
  auto it = dd->colorImgs.find(cand);
  if (it == dd->colorImgs.end()) { dd->bbUsable = false; return; }
  const DeviceData::ColorImg& ci = it->second;
  // (8-bit copies, or 10-bit ones turned into 8 bits by the shader, against an 8-bit presented image)
  bool fmtOk = (IsDisplayFormat(ci.fmt) && (IsRGBA8(dd->scFormat) || IsBGRA8(dd->scFormat))) ||
               (ci.fmt == dd->scFormat && IsDisplayFormat(ci.fmt));
  bool extOk = ci.ext.width == dd->gameExt.width && ci.ext.height == dd->gameExt.height;
  dd->bbSamples = ci.samples;
  if (fmtOk && extOk && ci.samples > 1 &&
      (!dd->bbResolve || dd->bbResolveFmt != ci.fmt || dd->bbResolveExt.width != ci.ext.width ||
       dd->bbResolveExt.height != ci.ext.height)) {
    if (dd->bbResolve) dd->retiredImgs.push_back({dd->bbResolve, dd->bbResolveMem});
    dd->bbResolve = VK_NULL_HANDLE;
    dd->bbResolveMem = VK_NULL_HANDLE;
    VkImageCreateInfo ic{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ic.imageType = VK_IMAGE_TYPE_2D;
    ic.format = ci.fmt;
    ic.extent = {ci.ext.width, ci.ext.height, 1};
    ic.mipLevels = 1;
    ic.arrayLayers = 1;
    ic.samples = VK_SAMPLE_COUNT_1_BIT;
    ic.tiling = VK_IMAGE_TILING_OPTIMAL;
    ic.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    VkImage im = VK_NULL_HANDLE;
    VkDeviceMemory mem = VK_NULL_HANDLE;
    if (CreateImageMem(dd, ic, &im, &mem)) {
      dd->bbResolve = im, dd->bbResolveMem = mem, dd->bbResolveFmt = ci.fmt, dd->bbResolveExt = ci.ext;
    } else {
      if (im) dd->retiredImgs.push_back({im, mem});
    }
  }
  bool resolveOk = ci.samples == 1 || dd->bbResolve;
  dd->bbUsable = fmtOk && extOk && resolveOk && dd->snapBuf;
  dd->bb10 = Is10Bit(ci.fmt) && !Is10Bit(dd->scFormat);
  dd->bbSwapRB = fmtOk && LowIsRed(ci.fmt) != LowIsRed(dd->scFormat);
  Log(1, "HUD separation: the game's back buffer is a %ux%u image (format %d, %u samples)%s", ci.ext.width,
      ci.ext.height, int(ci.fmt), ci.samples,
      !dd->bbUsable ? (!fmtOk ? ": format not supported" : !extOk ? ": another size than the game image, not used"
                                                             : ": could not make its resolve image, not used")
      : !Cfg().hudSeparation ? " (hud_separation = 0: not used)" : "");
}

// copies the back buffer (= the scene without HUD) into snapBuf, outside any render pass
void RecordSnapshot(DeviceData* dd, VkCommandBuffer cb, VkImage img, VkImageLayout layout) {
  VkBuffer dst;
  VkExtent2D ext;
  VkImage rimg;
  uint32_t samples;
  VkDeviceSize slotBytes;
  {
    std::lock_guard<std::mutex> l(dd->depthLock);
    dst = dd->snapBuf;
    ext = dd->gameExt;
    rimg = dd->bbResolve;
    samples = dd->bbSamples;
    slotBytes = dd->snapSlotBytes;
  }
  if (!dst || !slotBytes || layout == VK_IMAGE_LAYOUT_UNDEFINED || (samples > 1 && !rimg) ||
      dd->snapDone.exchange(true))
    return;
  const uint32_t slot = dd->snapNext.fetch_add(1) % kSnapSlots;
  ++dd->snapTaken;
  dd->slotFrame[slot] = dd->frame + 1;  // (+1: 0 = never written)
  const VkDeviceSize dstOff = slot * slotBytes;
  if (samples > 1) {  // multisampled: resolve into the layer's image, then copy that one
    VkImageMemoryBarrier bs[2] = {{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER}, {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER}};
    const VkImageLayout sl = layout == VK_IMAGE_LAYOUT_GENERAL ? VK_IMAGE_LAYOUT_GENERAL : VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    for (auto& b : bs) {
      b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    }
    bs[0].image = img;
    bs[0].oldLayout = layout;
    bs[0].newLayout = sl;
    bs[0].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    bs[0].dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    bs[1].image = rimg;
    bs[1].oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    bs[1].newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    bs[1].srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    bs[1].dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    dd->d.CmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 2, bs);
    VkImageResolve rr{};
    rr.srcSubresource = rr.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    rr.extent = {ext.width, ext.height, 1};
    dd->d.CmdResolveImage(cb, img, sl, rimg, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &rr);
    bs[0].oldLayout = sl;
    bs[0].newLayout = layout;
    bs[0].srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    bs[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                          VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    bs[1].oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    bs[1].newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    bs[1].srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    bs[1].dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};  // earlier readers of snapBuf (the eyes' compute)
    mb.srcAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    dd->d.CmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &mb, 0,
                             nullptr, 2, bs);
    VkBufferImageCopy r{};
    r.bufferOffset = dstOff;
    r.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    r.imageExtent = {ext.width, ext.height, 1};
    dd->d.CmdCopyImageToBuffer(cb, rimg, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, dst, 1, &r);
    mb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    dd->d.CmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &mb, 0,
                             nullptr, 0, nullptr);
    return;
  }
  const VkImageLayout cl = layout == VK_IMAGE_LAYOUT_GENERAL ? VK_IMAGE_LAYOUT_GENERAL : VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
  VkImageMemoryBarrier ib{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
  ib.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
  ib.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
  ib.oldLayout = layout;
  ib.newLayout = cl;
  ib.srcQueueFamilyIndex = ib.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  ib.image = img;
  ib.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
  VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};  // earlier readers of snapBuf (the eyes' compute)
  mb.srcAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
  mb.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  dd->d.CmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT |
                               VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                           VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &mb, 0, nullptr, 1, &ib);
  VkBufferImageCopy r{};
  r.bufferOffset = dstOff;
  r.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
  r.imageExtent = {ext.width, ext.height, 1};
  dd->d.CmdCopyImageToBuffer(cb, img, cl, dst, 1, &r);
  ib.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
  ib.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                     VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
  ib.oldLayout = cl;
  ib.newLayout = layout;
  mb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
  dd->d.CmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &mb, 0,
                           nullptr, 1, &ib);
}
// the copy is taken once 'split' draws went into the back buffer this frame
uint32_t SplitDraw(DeviceData* dd) { return Cfg().hudSplit > 0 ? uint32_t(Cfg().hudSplit) : dd->autoSplit; }
inline bool SnapDue(DeviceData* dd) { return !dd->snapDone.load() && dd->bbDraws.load() >= SplitDraw(dd); }

// before the game's pass begins: does it draw into the presented image (then the last game-sized
// target is the back buffer candidate) or into the back buffer?
void NoteColorPass(TlsCb& t, VkCommandBuffer cb, const VkRenderingInfo* ri, bool khr) {
  CbDepth& st = *t.st;
  DeviceData* dd = t.dd;
  if (ri->flags & VK_RENDERING_RESUMING_BIT) return;  // a resumed pass keeps its state
  st.bbPass = false;
  if (!dd->depthMode || !ri->colorAttachmentCount || !ri->pColorAttachments || !ri->pColorAttachments[0].imageView)
    return;
  const VkRenderingAttachmentInfo& ca = ri->pColorAttachments[0];
  VkImage img;
  bool isBB;
  {
    std::lock_guard<std::mutex> l(dd->depthLock);
    // a multisampled target is resolved into a normal image at the end of the pass: that one is the output
    VkImageView outView = (ca.resolveMode != VK_RESOLVE_MODE_NONE && ca.resolveImageView) ? ca.resolveImageView
                                                                                       : ca.imageView;
    auto v = dd->colorViews.find(outView);
    if (v == dd->colorViews.end()) return;
    img = v->second;
    if (dd->presentedImgs.count(img)) {
      VoteBackbufferLocked(dd, dd->lastColorImg);
      // the game's frame is finished (when there is no scene depth to tell: menus, 2D games)
      if (dd->depthFrameAt.load() == UINT64_MAX || dd->frame > dd->depthFrameAt.load() + 10) {
        dd->snapDone = false;
        dd->bbFilled = false;
        dd->bbDraws = 0;
      }
      return;
    }
    // (only targets in a display format: the game may already be drawing the next frame's HDR scene
    // when the present copy is recorded on another thread)
    auto c = dd->colorImgs.find(img);
    if (c != dd->colorImgs.end() && c->second.ext.width == dd->gameExt.width &&
        c->second.ext.height == dd->gameExt.height && IsDisplayFormat(c->second.fmt))
      dd->lastColorImg = img;
    isBB = img == dd->bbImg && dd->bbUsable && !dd->snapBad && Cfg().hudSeparation;
  }
  if (!isBB) return;
  st.bbPass = true;
  st.bbKhr = khr;
  st.bbImg = img;
  st.bbLayout = ca.imageLayout;
  st.bbCanSplit = !(ri->flags & VK_RENDERING_SUSPENDING_BIT);
  st.bbSecondary = (ri->flags & VK_RENDERING_CONTENTS_SECONDARY_COMMAND_BUFFERS_BIT) != 0;
  // the same pass, continued (for the split after the scene's draw)
  st.bbInfo = *ri;
  st.bbInfo.pNext = nullptr;
  st.bbColors.assign(ri->pColorAttachments, ri->pColorAttachments + ri->colorAttachmentCount);
  for (auto& a : st.bbColors) a.pNext = nullptr, a.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
  st.bbInfo.pColorAttachments = st.bbColors.data();
  if (ri->pDepthAttachment) {
    st.bbDepth = *ri->pDepthAttachment;
    st.bbDepth.pNext = nullptr;
    st.bbDepth.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
    st.bbInfo.pDepthAttachment = &st.bbDepth;
  }
  if (ri->pStencilAttachment) {
    st.bbStencil = *ri->pStencilAttachment;
    st.bbStencil.pNext = nullptr;
    st.bbStencil.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
    st.bbInfo.pStencilAttachment = &st.bbStencil;
  }
  // enough of the frame is already in the back buffer (earlier passes, or the scene came in by a copy:
  // CopyResource, a blit) and this pass adds to it: copy now
  if (SnapDue(dd) && ca.loadOp == VK_ATTACHMENT_LOAD_OP_LOAD) RecordSnapshot(dd, cb, img, ca.imageLayout);
  if (dd->snapDone.load()) SetHudFlag(dd, cb, st, true);  // this pass draws HUD only
}

// before a draw: the back buffer pass already has its first draw (the finished scene): copy it now,
// before the HUD, by ending the pass and continuing it
void BeforeDraw(TlsCb& t, VkCommandBuffer cb) {
  CbDepth& st = *t.st;
  if (!st.bbPass || st.bbSecondary) return;
  DeviceData* dd = t.dd;
  if (!st.bbCanSplit || !SnapDue(dd)) {
    ++dd->bbDraws;
    return;
  }
  (st.bbKhr && dd->d.CmdEndRenderingKHR ? dd->d.CmdEndRenderingKHR : dd->d.CmdEndRendering)(cb);
  RecordSnapshot(dd, cb, st.bbImg, st.bbLayout);
  SetHudFlag(dd, cb, st, true);
  (st.bbKhr && dd->d.CmdBeginRenderingKHR ? dd->d.CmdBeginRenderingKHR : dd->d.CmdBeginRendering)(cb, &st.bbInfo);
  ++dd->bbDraws;
}
void AfterColorPass(TlsCb& t, VkCommandBuffer cb) {
  CbDepth& st = *t.st;
  if (!st.bbPass) return;
  st.bbPass = false;
  SetHudFlag(t.dd, cb, st, false);  // (after the pass: the next passes are the game's again)
  if (st.bbCanSplit && SnapDue(t.dd)) RecordSnapshot(t.dd, cb, st.bbImg, st.bbLayout);
}
// the pass's draws are in secondary command buffers (D3D11 games on DXVK may record them on worker
// threads): the first one with draws is the scene; the pass is split after it, as after a first draw
VKAPI_ATTR void VKAPI_CALL hook_CmdExecuteCommands(VkCommandBuffer cb, uint32_t n, const VkCommandBuffer* cbs) {
  TlsCb& t = CbState(cb);
  DeviceData* dd = t.dd;
  CbDepth* st = t.st;
  std::vector<uint32_t> draws(n, 0);
  {
    std::lock_guard<std::mutex> l(g_cbLock);
    for (uint32_t i = 0; i < n; ++i) {
      auto it = g_cbDepth.find(cbs[i]);
      if (it != g_cbDepth.end()) draws[i] = it->second.draws;
    }
  }
  // the copy is taken between two of them, once 'split' draws went in (it cannot be inside one)
  uint32_t start = 0;
  for (uint32_t i = 0; i < n; ++i) {
    st->draws += draws[i];
    if (!st->bbPass || !st->bbSecondary) continue;
    if (st->bbCanSplit && draws[i] && SnapDue(dd)) {
      if (i > start) dd->d.CmdExecuteCommands(cb, i - start, cbs + start);
      start = i;
      (st->bbKhr && dd->d.CmdEndRenderingKHR ? dd->d.CmdEndRenderingKHR : dd->d.CmdEndRendering)(cb);
      RecordSnapshot(dd, cb, st->bbImg, st->bbLayout);
      SetHudFlag(dd, cb, *st, true);
      (st->bbKhr && dd->d.CmdBeginRenderingKHR ? dd->d.CmdBeginRenderingKHR : dd->d.CmdBeginRendering)(cb, &st->bbInfo);
    }
    dd->bbDraws += draws[i];
  }
  if (start < n) dd->d.CmdExecuteCommands(cb, n - start, cbs + start);
}
void NoteTransfer(DeviceData* dd, VkImage src, VkImage dst) {
  if (!dd->depthMode) return;
  std::lock_guard<std::mutex> l(dd->depthLock);
  if (dd->presentedImgs.count(dst)) {
    VoteBackbufferLocked(dd, src);
    if (dd->depthFrameAt.load() == UINT64_MAX || dd->frame > dd->depthFrameAt.load() + 10) {
      dd->snapDone = false;
      dd->bbFilled = false;
      dd->bbDraws = 0;
    }
  }
  if (dst && dst == dd->bbImg) {  // a copy into the back buffer counts as a draw
    dd->bbFilled = true;
    ++dd->bbDraws;
  }
}
VKAPI_ATTR void VKAPI_CALL hook_CmdCopyImage(VkCommandBuffer cb, VkImage src, VkImageLayout sl, VkImage dst,
                                             VkImageLayout dl, uint32_t n, const VkImageCopy* r) {
  TlsCb& t = CbState(cb);
  NoteTransfer(t.dd, src, dst);
  t.dd->d.CmdCopyImage(cb, src, sl, dst, dl, n, r);
}
VKAPI_ATTR void VKAPI_CALL hook_CmdBlitImage(VkCommandBuffer cb, VkImage src, VkImageLayout sl, VkImage dst,
                                             VkImageLayout dl, uint32_t n, const VkImageBlit* r, VkFilter f) {
  TlsCb& t = CbState(cb);
  NoteTransfer(t.dd, src, dst);
  t.dd->d.CmdBlitImage(cb, src, sl, dst, dl, n, r, f);
}
VKAPI_ATTR void VKAPI_CALL hook_CmdResolveImage(VkCommandBuffer cb, VkImage src, VkImageLayout sl, VkImage dst,
                                                VkImageLayout dl, uint32_t n, const VkImageResolve* r) {
  TlsCb& t = CbState(cb);
  NoteTransfer(t.dd, src, dst);
  t.dd->d.CmdResolveImage(cb, src, sl, dst, dl, n, r);
}
VKAPI_ATTR void VKAPI_CALL hook_CmdResolveImage2(VkCommandBuffer cb, const VkResolveImageInfo2* i) {
  TlsCb& t = CbState(cb);
  NoteTransfer(t.dd, i->srcImage, i->dstImage);
  (t.dd->d.CmdResolveImage2 ? t.dd->d.CmdResolveImage2 : t.dd->d.CmdResolveImage2KHR)(cb, i);
}
VKAPI_ATTR void VKAPI_CALL hook_CmdResolveImage2KHR(VkCommandBuffer cb, const VkResolveImageInfo2* i) {
  TlsCb& t = CbState(cb);
  NoteTransfer(t.dd, i->srcImage, i->dstImage);
  (t.dd->d.CmdResolveImage2KHR ? t.dd->d.CmdResolveImage2KHR : t.dd->d.CmdResolveImage2)(cb, i);
}
VKAPI_ATTR void VKAPI_CALL hook_CmdCopyImage2(VkCommandBuffer cb, const VkCopyImageInfo2* i) {
  TlsCb& t = CbState(cb);
  NoteTransfer(t.dd, i->srcImage, i->dstImage);
  (t.dd->d.CmdCopyImage2 ? t.dd->d.CmdCopyImage2 : t.dd->d.CmdCopyImage2KHR)(cb, i);
}
VKAPI_ATTR void VKAPI_CALL hook_CmdCopyImage2KHR(VkCommandBuffer cb, const VkCopyImageInfo2* i) {
  TlsCb& t = CbState(cb);
  NoteTransfer(t.dd, i->srcImage, i->dstImage);
  (t.dd->d.CmdCopyImage2KHR ? t.dd->d.CmdCopyImage2KHR : t.dd->d.CmdCopyImage2)(cb, i);
}
VKAPI_ATTR void VKAPI_CALL hook_CmdBlitImage2(VkCommandBuffer cb, const VkBlitImageInfo2* i) {
  TlsCb& t = CbState(cb);
  NoteTransfer(t.dd, i->srcImage, i->dstImage);
  (t.dd->d.CmdBlitImage2 ? t.dd->d.CmdBlitImage2 : t.dd->d.CmdBlitImage2KHR)(cb, i);
}
VKAPI_ATTR void VKAPI_CALL hook_CmdBlitImage2KHR(VkCommandBuffer cb, const VkBlitImageInfo2* i) {
  TlsCb& t = CbState(cb);
  NoteTransfer(t.dd, i->srcImage, i->dstImage);
  (t.dd->d.CmdBlitImage2KHR ? t.dd->d.CmdBlitImage2KHR : t.dd->d.CmdBlitImage2)(cb, i);
}


// before the game's pass begins (outside render passes)
void CalibBeforePass(TlsCb& t, VkCommandBuffer cb, const VkRenderingInfo* ri) {
  CbDepth& st = *t.st;
  DeviceData* dd = t.dd;
  if (!dd->depthMode) return;
  // the game is about to clear (or replace) the scene depth: copy the previous stage first
  if (st.pending && ri->pDepthAttachment && ri->pDepthAttachment->loadOp != VK_ATTACHMENT_LOAD_OP_LOAD &&
      !(ri->flags & VK_RENDERING_RESUMING_BIT)) {
    bool same;
    {
      std::lock_guard<std::mutex> l(dd->depthLock);
      auto v = dd->depthViews.find(ri->pDepthAttachment->imageView);
      same = v != dd->depthViews.end() && v->second.img == st.pImg;
    }
    if (same) FlushDepthCopy(dd, cb, st);
  }
  if ((ri->flags & VK_RENDERING_RESUMING_BIT) || !dd->paramBuf) return;
  int want;
  const uint64_t main = dd->mainClass.load();
  if (st.active && (main == 0 || main == DepthClass(st))) {
    // A new stage each time the scene depth is cleared (another camera), or when the scene goes on in
    // another depth image. A pass that goes back to an image of an earlier stage without clearing it
    // continues that stage (EDF 6: the main scene, a far-away layer in another image with its own
    // projection, then more of the main scene): its samples and its depth belong to that camera.
    uint32_t g;
    {
      std::lock_guard<std::mutex> l(dd->depthLock);
      g = dd->calibGroup.load();
      // the first stage's image cleared again after the frame went on to later stages and came back:
      // the next frame begins (seen in the game's recording order; the present can come late)
      if (st.clears && g > dd->curStage && dd->stageImg[1] == st.img) {
        dd->calibGroup = g = 0;
        for (auto& im : dd->stageImg) im = VK_NULL_HANDLE;
        dd->stageFrameAt = dd->frame;
      }
      if (!dd->stageImg[g]) dd->stageImg[g] = st.img;
      if (st.clears || dd->stageImg[g] != st.img) {
        int back = -1;
        if (!st.clears)
          for (int s = int(g); s >= 0; --s)
            if (dd->stageImg[s] == st.img) { back = s; break; }
        if (back >= 0) {
          g = uint32_t(back);
        } else if (g < 3) {
          dd->calibGroup = ++g;
          dd->stageImg[g] = st.img;
        }
      }
      dd->curStage = g;
    }
    st.stage = g;
    want = int(1 + g);
  } else if (ri->pDepthAttachment && ri->pDepthAttachment->imageView) {
    want = 0;
  } else {
    return;
  }
  if (st.flag == want) return;
  SetCalibFlag(dd, cb, float(want));
  st.flag = want;
}

VKAPI_ATTR void VKAPI_CALL hook_CmdPipelineBarrier(VkCommandBuffer cb, VkPipelineStageFlags src,
                                                   VkPipelineStageFlags dst, VkDependencyFlags dep, uint32_t mc,
                                                   const VkMemoryBarrier* mb, uint32_t bc,
                                                   const VkBufferMemoryBarrier* bb, uint32_t ic,
                                                   const VkImageMemoryBarrier* ib) {
  TlsCb& t = CbState(cb);
  for (uint32_t i = 0; i < ic; ++i) NoteDepthBarrier(*t.st, ib[i].image, ib[i].oldLayout, ib[i].newLayout);
  t.dd->d.CmdPipelineBarrier(cb, src, dst, dep, mc, mb, bc, bb, ic, ib);
}
void NoteDependency(TlsCb& t, const VkDependencyInfo* di) {
  for (uint32_t i = 0; di && i < di->imageMemoryBarrierCount; ++i)
    NoteDepthBarrier(*t.st, di->pImageMemoryBarriers[i].image, di->pImageMemoryBarriers[i].oldLayout,
                     di->pImageMemoryBarriers[i].newLayout);
}
VKAPI_ATTR void VKAPI_CALL hook_CmdPipelineBarrier2(VkCommandBuffer cb, const VkDependencyInfo* di) {
  TlsCb& t = CbState(cb);
  NoteDependency(t, di);
  (t.dd->d.CmdPipelineBarrier2 ? t.dd->d.CmdPipelineBarrier2 : t.dd->d.CmdPipelineBarrier2KHR)(cb, di);
}
VKAPI_ATTR void VKAPI_CALL hook_CmdPipelineBarrier2KHR(VkCommandBuffer cb, const VkDependencyInfo* di) {
  TlsCb& t = CbState(cb);
  NoteDependency(t, di);
  (t.dd->d.CmdPipelineBarrier2KHR ? t.dd->d.CmdPipelineBarrier2KHR : t.dd->d.CmdPipelineBarrier2)(cb, di);
}
VKAPI_ATTR void VKAPI_CALL hook_CmdEndRendering(VkCommandBuffer cb) {
  TlsCb& t = CbState(cb);
  t.dd->d.CmdEndRendering(cb);
  EndRenderingCommon(t, cb);
  AfterColorPass(t, cb);
}
VKAPI_ATTR void VKAPI_CALL hook_CmdEndRenderingKHR(VkCommandBuffer cb) {
  TlsCb& t = CbState(cb);
  (t.dd->d.CmdEndRenderingKHR ? t.dd->d.CmdEndRenderingKHR : t.dd->d.CmdEndRendering)(cb);
  EndRenderingCommon(t, cb);
  AfterColorPass(t, cb);
}
void NoteRenderPass(DeviceData* dd) {
  if (dd->depthMode && !dd->loggedRenderPass.exchange(true))
    Log(0, "depth mode: this game uses classic render passes; only dynamic rendering (DXVK, VKD3D-Proton) "
        "is supported, the rebuilt eyes will have no depth");
}
VKAPI_ATTR void VKAPI_CALL hook_CmdDraw(VkCommandBuffer cb, uint32_t vc, uint32_t ic, uint32_t fv, uint32_t fi) {
  TlsCb& t = CbState(cb);
  BeforeDraw(t, cb);
  ++t.st->draws;
  t.dd->d.CmdDraw(cb, vc, ic, fv, fi);
}
VKAPI_ATTR void VKAPI_CALL hook_CmdDrawIndexed(VkCommandBuffer cb, uint32_t ic, uint32_t inst, uint32_t fi, int32_t vo,
                                               uint32_t fin) {
  TlsCb& t = CbState(cb);
  BeforeDraw(t, cb);
  ++t.st->draws;
  t.dd->d.CmdDrawIndexed(cb, ic, inst, fi, vo, fin);
}
VKAPI_ATTR void VKAPI_CALL hook_CmdDrawIndirect(VkCommandBuffer cb, VkBuffer b, VkDeviceSize o, uint32_t n, uint32_t s) {
  TlsCb& t = CbState(cb);
  BeforeDraw(t, cb);
  t.st->draws += std::max(1u, n);
  t.dd->d.CmdDrawIndirect(cb, b, o, n, s);
}
VKAPI_ATTR void VKAPI_CALL hook_CmdDrawIndexedIndirect(VkCommandBuffer cb, VkBuffer b, VkDeviceSize o, uint32_t n,
                                                       uint32_t s) {
  TlsCb& t = CbState(cb);
  BeforeDraw(t, cb);
  t.st->draws += std::max(1u, n);
  t.dd->d.CmdDrawIndexedIndirect(cb, b, o, n, s);
}
// DXVK batches consecutive draws into one vkCmdDrawMultiEXT (VK_EXT_multi_draw)
VKAPI_ATTR void VKAPI_CALL hook_CmdDrawMultiEXT(VkCommandBuffer cb, uint32_t n, const VkMultiDrawInfoEXT* vi,
                                                uint32_t inst, uint32_t fi, uint32_t stride) {
  TlsCb& t = CbState(cb);
  BeforeDraw(t, cb);
  t.st->draws += std::max(1u, n);
  t.dd->d.CmdDrawMultiEXT(cb, n, vi, inst, fi, stride);
}
VKAPI_ATTR void VKAPI_CALL hook_CmdDrawMultiIndexedEXT(VkCommandBuffer cb, uint32_t n, const VkMultiDrawIndexedInfoEXT* ii,
                                                       uint32_t inst, uint32_t fi, uint32_t stride, const int32_t* vo) {
  TlsCb& t = CbState(cb);
  BeforeDraw(t, cb);
  t.st->draws += std::max(1u, n);
  t.dd->d.CmdDrawMultiIndexedEXT(cb, n, ii, inst, fi, stride, vo);
}
VKAPI_ATTR void VKAPI_CALL hook_CmdDrawIndirectCount(VkCommandBuffer cb, VkBuffer b, VkDeviceSize o, VkBuffer cbuf,
                                                     VkDeviceSize co, uint32_t mx, uint32_t s) {
  TlsCb& t = CbState(cb);
  BeforeDraw(t, cb);
  t.st->draws += 8;  // count unknown on the CPU
  t.dd->d.CmdDrawIndirectCount(cb, b, o, cbuf, co, mx, s);
}
VKAPI_ATTR void VKAPI_CALL hook_CmdDrawIndexedIndirectCount(VkCommandBuffer cb, VkBuffer b, VkDeviceSize o,
                                                            VkBuffer cbuf, VkDeviceSize co, uint32_t mx, uint32_t s) {
  TlsCb& t = CbState(cb);
  BeforeDraw(t, cb);
  t.st->draws += 8;
  t.dd->d.CmdDrawIndexedIndirectCount(cb, b, o, cbuf, co, mx, s);
}

// Fit clip z = A' * w + B' to the samples (RANSAC: other cameras' samples are outliers); z_ndc = A' + B' / w
void FitStage(DeviceData::Proj& P, std::vector<std::pair<double, double>>& pts, uint32_t stage) {
  P.samples = uint32_t(pts.size());
  if (pts.size() < 8) return;
  static std::mt19937 rng(1234);
  std::uniform_int_distribution<size_t> pick(0, pts.size() - 1);
  auto inl = [&](double A, double B, std::vector<size_t>* out) {
    size_t n = 0;
    for (size_t i = 0; i < pts.size(); ++i) {
      double pred = A * pts[i].first + B;
      if (std::fabs(pts[i].second - pred) <= 2e-4 * (std::fabs(pts[i].second) + std::fabs(B)) + 1e-7) {
        ++n;
        if (out) out->push_back(i);
      }
    }
    return n;
  };
  size_t best = 0;
  double bA = 0, bB = 0;
  for (int it = 0; it < 200; ++it) {
    size_t i = pick(rng), j = pick(rng);
    double w1 = pts[i].first, w2 = pts[j].first;
    if (std::fabs(w1 - w2) <= 1e-5 * (w1 + w2)) continue;
    double A = (pts[i].second - pts[j].second) / (w1 - w2), B = pts[i].second - A * w1;
    size_t n = inl(A, B, nullptr);
    if (n > best) { best = n; bA = A; bB = B; }
  }
  auto refit = [&](double& A0, double& B0, const std::vector<size_t>& idx) {
    double sw = 0, sz = 0, sww = 0, swz = 0, n = double(idx.size());
    for (size_t i : idx) {
      sw += pts[i].first; sz += pts[i].second;
      sww += pts[i].first * pts[i].first; swz += pts[i].first * pts[i].second;
    }
    double den = n * sww - sw * sw;
    if (n < 6 || std::fabs(den) <= 1e-12 * n * sww) return false;
    A0 = (n * swz - sw * sz) / den;
    B0 = (sz - A0 * sw) / n;
    return std::fabs(B0) > 1e-9;
  };
  // several cameras can feed samples: keep the current projection while it is about as good as
  // the best, switch only when another one is clearly better several fits in a row
  if (P.ok) {
    double cA = double(P.A), cB = double(P.B);
    size_t cur = inl(cA, cB, nullptr);
    P.inliers = uint32_t(cur);
    if (cur * 3 >= best * 2 && cur >= 4) {  // still the main one: refine it a little
      P.switchVotes = 0;
      std::vector<size_t> idx;
      inl(cA, cB, &idx);
      double A, B;
      if (refit(A, B, idx) && std::fabs(B / cB - 1.0) < 0.01) {
        P.A = float(cA + (A - cA) * 0.2);
        P.B = float(cB + (B - cB) * 0.2);
      }
      return;
    }
    bool clear = best >= std::max<size_t>(8, pts.size() * 6 / 10) && cur * 4 < pts.size();
    if (!clear && (best < std::max<size_t>(6, pts.size() * 3 / 10) || ++P.switchVotes < 3)) return;
  } else if (best < std::max<size_t>(6, pts.size() * 3 / 10) || std::fabs(bB) < 1e-9) {
    return;
  }
  std::vector<size_t> idx;
  inl(bA, bB, &idx);
  double A, B;
  if (refit(A, B, idx)) { bA = A; bB = B; }
  if (std::fabs(bB) < 1e-9) return;
  Log(1, "depth mode: projection of scene stage %u %s (z = %.6g + %.6g / w, %zu of %zu samples)", stage,
      P.ok ? "changed" : "found", bA, bB, best, pts.size());
  P.A = float(bA);
  P.B = float(bB);
  P.ok = true;
  P.inliers = uint32_t(best);
  P.switchVotes = 0;
}

void FitProjection(DeviceData* dd) {
  if (!dd->paramMap) return;
  volatile uint64_t* slots = (volatile uint64_t*)((char*)dd->paramMap + kCalibOffset);
  std::vector<std::pair<double, double>> pts[4];  // (w, z) per stage
  for (uint32_t i = 0; i < kCalibSlots; ++i) {
    uint64_t v = slots[i];
    if (!v) continue;
    slots[i] = 0;
    float w, z;
    uint32_t wb = uint32_t(v), zb = uint32_t(v >> 32);
    memcpy(&w, &wb, 4);
    memcpy(&z, &zb, 4);
    if (std::isfinite(w) && std::isfinite(z) && w > 1e-6f) pts[(i / kCalibGroupSlots) & 3].push_back({w, z});
  }
  if (Tracing()) Log(0, "fit: samples %zu %zu %zu %zu flag %g", pts[0].size(), pts[1].size(), pts[2].size(), pts[3].size(),
                     *(float*)((char*)dd->paramMap + kCalibFlagOffset));
  for (uint32_t g = 0; g < 4; ++g)
    if (!pts[g].empty()) FitStage(dd->proj[g], pts[g], g);
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
  NoteRenderPass(dd);
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
void BeginRenderingAll(VkCommandBuffer cb, const VkRenderingInfo* ri, bool khr) {
  DeviceData* dd = DevFast(cb);
  if (dd->depthMode) {
    TlsCb& t = CbState(cb);
    BeginRenderingCommon(t, ri);
    CalibBeforePass(t, cb, ri);
    NoteColorPass(t, cb, ri, khr);
  }
  if (!(ri->flags & VK_RENDERING_RESUMING_BIT)) BeforePass(dd, cb, ri->renderArea.extent);
}
VKAPI_ATTR void VKAPI_CALL hook_CmdBeginRendering(VkCommandBuffer cb, const VkRenderingInfo* ri) {
  BeginRenderingAll(cb, ri, false);
  DevFast(cb)->d.CmdBeginRendering(cb, ri);
}
VKAPI_ATTR void VKAPI_CALL hook_CmdBeginRenderingKHR(VkCommandBuffer cb, const VkRenderingInfo* ri) {
  BeginRenderingAll(cb, ri, true);
  DeviceData* dd = DevFast(cb);
  (dd->d.CmdBeginRenderingKHR ? dd->d.CmdBeginRenderingKHR : dd->d.CmdBeginRendering)(cb, ri);
}
VKAPI_ATTR VkResult VKAPI_CALL hook_EndCommandBuffer(VkCommandBuffer cb) {
  DeviceData* dd = DevFast(cb);
  if (dd->depthMode) {
    TlsCb& t = CbState(cb);
    FlushDepthCopy(dd, cb, *t.st);  // the last scene stage of the frame
  }
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

  // patches stage s of pipeline i (shader module or VK_KHR_maintenance5 inline code)
  auto patchStage = [&](uint32_t i, uint32_t s, uint32_t model) {
    const VkPipelineShaderStageCreateInfo& st = ci[i].pStages[s];
    if (st.module != VK_NULL_HANDLE) {
      VkShaderModule pm = GetPatchedModule(dd, st.module, model, st.pName);
      if (pm) {
        if (stages[i].empty()) {
          stages[i].assign(ci[i].pStages, ci[i].pStages + ci[i].stageCount);
          ci[i].pStages = stages[i].data();
        }
        stages[i][s].module = pm;
      }
      return;
    }
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
  };

  for (uint32_t i = 0; i < count; ++i) {
    if (!ci[i].pStages || !ci[i].stageCount) continue;
    int best = -1, bestRank = -1, frag = -1;
    for (uint32_t s = 0; s < ci[i].stageCount; ++s) {
      int rank = -1;
      switch (ci[i].pStages[s].stage) {
        case VK_SHADER_STAGE_VERTEX_BIT: rank = 0; break;
        case VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT: rank = 1; break;
        case VK_SHADER_STAGE_GEOMETRY_BIT: rank = 2; break;
        case VK_SHADER_STAGE_FRAGMENT_BIT: frag = int(s); break;
        default: break;
      }
      if (rank > bestRank) { bestRank = rank; best = int(s); }
    }
    if (best < 0) continue;
    // screen-space position rebuilds (Unreal shadow projection, lights...): see PatchScreenSpace
    if (frag >= 0 && dd->screenFix) patchStage(i, uint32_t(frag), 4);
    if (dd->unshiftD16 && ci[i].renderPass == VK_NULL_HANDLE) {
      // Unreal Engine draws its shadow maps in 16-bit depth without colour, the scene's depth
      // pre-pass in D32S8 / D24S8: the shadow maps get no eye transform (they are seen from the light)
      bool noColor = true;
      int depthFmt = -1;
      for (auto* p = (const VkBaseInStructure*)ci[i].pNext; p; p = p->pNext)
        if (p->sType == VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO) {
          noColor = ((const VkPipelineRenderingCreateInfo*)p)->colorAttachmentCount == 0;
          depthFmt = int(((const VkPipelineRenderingCreateInfo*)p)->depthAttachmentFormat);
        }
      if (noColor && depthFmt == VK_FORMAT_D16_UNORM) {
        ++dd->depthOnlyCount;
        continue;
      }
    }
    ++dd->pipelineCount;
    patchStage(i, uint32_t(best), StageToModel(ci[i].pStages[best].stage));
  }
  VkResult r = dd->d.CreateGraphicsPipelines(device, cache, count, ci.data(), a, out);
  for (auto& rs : restores) {
    rs.smci->pCode = rs.code;
    rs.smci->codeSize = rs.size;
  }
  return r;
}

// screen-space position rebuilds done in compute (tiled lighting, SSAO...)
VKAPI_ATTR VkResult VKAPI_CALL hook_CreateComputePipelines(VkDevice device, VkPipelineCache cache, uint32_t count,
                                                           const VkComputePipelineCreateInfo* infos,
                                                           const VkAllocationCallbacks* a, VkPipeline* out) {
  DeviceData* dd = Dev(device);
  if (!dd->screenFix || !dd->stereoShaders) return dd->d.CreateComputePipelines(device, cache, count, infos, a, out);
  std::vector<VkComputePipelineCreateInfo> ci(infos, infos + count);
  struct Restore { VkShaderModuleCreateInfo* smci; const uint32_t* code; size_t size; };
  std::vector<Restore> restores;
  std::vector<std::unique_ptr<std::vector<uint32_t>>> inlineCode;
  for (uint32_t i = 0; i < count; ++i) {
    VkPipelineShaderStageCreateInfo& st = ci[i].stage;
    if (st.stage != VK_SHADER_STAGE_COMPUTE_BIT) continue;
    if (st.module != VK_NULL_HANDLE) {
      if (VkShaderModule pm = GetPatchedModule(dd, st.module, 5, st.pName)) st.module = pm;
      continue;
    }
    for (auto* p = (VkBaseOutStructure*)st.pNext; p; p = p->pNext) {
      if (p->sType != VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO) continue;
      auto* smci = (VkShaderModuleCreateInfo*)p;
      auto buf = std::make_unique<std::vector<uint32_t>>();
      if (PatchCode(dd, smci->pCode, smci->codeSize / 4, 5, st.pName, *buf)) {
        restores.push_back({smci, smci->pCode, smci->codeSize});
        smci->pCode = buf->data();
        smci->codeSize = buf->size() * 4;
        inlineCode.push_back(std::move(buf));
      }
      break;
    }
  }
  VkResult r = dd->d.CreateComputePipelines(device, cache, count, ci.data(), a, out);
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
  // depth / hybrid: both eyes share one frustum (the union of the two), so an eye differs from the
  // other only by its position and can be rebuilt from the other's image by horizontal shifts
  float commonFov[4] = {0, 0, 0, 0};
  if (dd->depthMode && shader)
    for (int k = 0; k < 4; ++k)
      commonFov[k] = (k == 0 || k == 3) ? std::min(hp.eyeFov[0][k], hp.eyeFov[1][k])
                                        : std::max(hp.eyeFov[0][k], hp.eyeFov[1][k]);
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
    const float* ef = dd->depthMode ? commonFov : hp.eyeFov[s];
    float L = std::tan(ef[0]) * os, Rt = std::tan(ef[1]) * os;
    float Up = std::tan(ef[2]) * os, Dn = std::tan(ef[3]) * os;
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
  if (dd->depthMode) {
    std::lock_guard<std::mutex> l(dd->depthLock);
    for (auto& v : sc->v) dd->presentedImgs.erase(v.image);
  }
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
  for (int e = 0; e < 2; ++e) {
    if (sc->synth[e]) dd->d.DestroyImage(dev, sc->synth[e], nullptr);
    if (sc->synthMem[e]) dd->d.FreeMemory(dev, sc->synthMem[e], nullptr);
  }
  if (sc->work) dd->d.DestroyBuffer(dev, sc->work, nullptr);
  if (sc->workMem) dd->d.FreeMemory(dev, sc->workMem, nullptr);
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
  if (dd->depthMode) {
    EnsureDepthBuffer(dd, sc->vext);
    // HUD separation: copies of the game's back buffer (kSnapSlots, one per recent frame)
    VkDeviceSize snapNeed = VkDeviceSize(sc->vext.width) * sc->vext.height * 4;
    VkBuffer nb = VK_NULL_HANDLE;
    VkDeviceMemory nm = VK_NULL_HANDLE;
    VkDeviceAddress na = 0;
    bool grow = dd->snapSize < snapNeed * kSnapSlots;
    if (grow && !CreateDevBuf(dd, snapNeed * kSnapSlots,
                              VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, &nb, &nm, &na))
      grow = false;
    std::lock_guard<std::mutex> l(dd->depthLock);
    dd->gameExt = sc->vext;
    dd->scFormat = sc->format;
    for (auto& v : sc->v) dd->presentedImgs.insert(v.image);
    if (grow) {
      if (dd->snapBuf) dd->retiredBufs.push_back({dd->snapBuf, dd->snapMem});
      dd->snapBuf = nb;
      dd->snapMem = nm;
      dd->snapAddr = na;
      dd->snapSize = snapNeed * kSnapSlots;
    }
    dd->snapSlotBytes = dd->snapSize >= snapNeed * kSnapSlots ? snapNeed : 0;
    for (auto& f : dd->slotFrame) f = 0;
    dd->bbImg = dd->bbCand = dd->lastColorImg = VK_NULL_HANDLE;  // learnt again for this swapchain
    dd->bbUsable = false;
    dd->bbVotes = 0;
  }
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
  if (dd->screenFix && dd->paramMap && Cfg().fixShadowsOn != dd->lastFixOn) {  // Ctrl+Shift+Pause
    dd->lastFixOn = Cfg().fixShadowsOn;
    float f = ScreenFixFactor();
    memcpy((char*)dd->paramMap + kFixFactorOffset, &f, 4);
  }
  double el = std::chrono::duration<double>(now - sc->statStart).count();
  if (el < 5.0) return;
  if (dd->screenFix) {  // only when something changed
    uint32_t a = dd->screenFixCount.load(), b = dd->depthOnlyCount.load();
    if ((a || b) && (a != dd->lastFixLog[0] || b != dd->lastFixLog[1])) {
      dd->lastFixLog[0] = a;
      dd->lastFixLog[1] = b;
      Log(1, "fix_shadows: %u shaders corrected, %u shadow-map pipelines without eye transform (Ctrl+Shift+Pause: %s)",
          a, b, Cfg().fixShadowsOn ? "on" : "off");
    }
  }
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
  if (dd->depthMode) {
    uint32_t rb = dd->rebuiltFrames.exchange(0);
    const DeviceData::Proj& pj = dd->proj[dd->lastCopyGroup.load() & 3];
    Log(1, "game %.1f fps | to headset %.1f fps [%s] | %s: eyes rebuilt from depth %u of %u frames%s | shaders %u | "
        "off-screen passes %u | %s fov %.1f scale %.3f",
        sc->statFrames / el, sc->statVr / el,
        !sh ? "host not running" : sh->sessionRunning.load() ? "headset on" : "waiting for headset",
        dd->hybrid ? "hybrid" : "depth", rb, uint32_t(sc->statFrames), pj.ok ? "" : " (projection not found yet)",
        dd->patchedCount.load(), dd->monoPasses.exchange(0), Cfg().display ? "screen" : "vr", Cfg().fov, Cfg().worldScale);
  } else {
    Log(1, "game %.1f fps | to headset %.1f fps [%s] | replay 3D %u (relaxed %u) / 2D %u | shaders %u | off-screen passes %u | %s fov %.1f scale %.3f sens %.2f",
        sc->statFrames / el, sc->statVr / el,
        !sh ? "host not running" : sh->sessionRunning.load() ? "headset on" : "waiting for headset", rf, xf, mf,
        dd->patchedCount.load(), dd->monoPasses.exchange(0), Cfg().display ? "screen" : "vr", Cfg().fov, Cfg().worldScale, Cfg().mouseSens);
  }
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

// ---- depth / hybrid modes -------------------------------------------------------------
bool EnsureSynth(DeviceData* dd, Swapchain* sc) {
  if (sc->depthBroken) return false;
  if (sc->work) return true;
  const uint32_t W = sc->vext.width, H = sc->vext.height;
  if (!Is32bpp(sc->format) || W > 4096) {
    Log(0, "depth mode: unsupported game image (%ux%u, format %d): the eyes will have no depth", W, H, sc->format);
    sc->depthBroken = true;
    return false;
  }
  VkDeviceSize px = VkDeviceSize(W) * H * 4;
  bool ok = CreateDevBuf(dd, px * 7,  // colour | keys x2 | eyes x2 | last real eyes x2 (hybrid)
                         VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                             VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                         &sc->work, &sc->workMem, &sc->workAddr) && sc->workAddr;
  for (int e = 0; ok && e < 2; ++e) {
    VkImageCreateInfo ici{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ici.imageType = VK_IMAGE_TYPE_2D;
    ici.format = sc->format;
    ici.extent = {W, H, 1};
    ici.mipLevels = 1;
    ici.arrayLayers = 1;
    ici.samples = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling = VK_IMAGE_TILING_OPTIMAL;
    ici.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    ok = CreateImageMem(dd, ici, &sc->synth[e], &sc->synthMem[e]);
  }
  if (!ok) {
    Log(0, "depth mode: cannot allocate the eye buffers: the eyes will have no depth");
    sc->depthBroken = true;
    return false;
  }
  return true;
}

void BufBarrier(DeviceData* dd, VkCommandBuffer cb, VkPipelineStageFlags src, VkAccessFlags sa,
                VkPipelineStageFlags dst, VkAccessFlags da) {
  VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
  mb.srcAccessMask = sa;
  mb.dstAccessMask = da;
  dd->d.CmdPipelineBarrier(cb, src, dst, 0, 1, &mb, 0, nullptr, 0, nullptr);
}

// depth mode renders from the middle of the eyes: the average of both eye transforms
EyeParams CenterParams(const FrameView& v) {
  EyeParams c;
  for (int i = 0; i < kParamFloats; ++i) c.m[i] = 0.5f * (v.p[0].m[i] + v.p[1].m[i]);
  c.m[16] = v.p[0].m[16];
  c.m[17] = v.p[0].m[17];
  return c;
}
// the transform the next game frame is drawn with
EyeParams RenderParams(DeviceData* dd, const FrameView& v, uint64_t frame) {
  return dd->hybrid ? v.p[frame & 1] : CenterParams(v);
}

// Depth / hybrid: every game frame becomes a whole pair. depth: the frame was drawn from the
// middle of the eyes and both eyes are rebuilt; hybrid: it was drawn for one eye (alternating) and
// the other eye is rebuilt from its depth (the uncovered background comes from that eye's last
// real image).
VkResult PresentDepth(DeviceData* dd, VkQueue queue, Swapchain* sc, uint32_t idx, const VkSemaphore* waits,
                      uint32_t waitCount) {
  if (idx >= sc->v.size()) return VK_ERROR_OUT_OF_DATE_KHR;
  VImage& vi = sc->v[idx];
  const uint64_t frame = dd->frame;
  const FrameView cur = dd->cur;
  QueueInfo* qi = GetQueueInfo(dd, queue);

  uint32_t realIdx = UINT32_MAX;
  dd->stage = "present(depth): acquiring window image";
  VkResult acq = AcquireMirror(dd, sc, vi, &realIdx);
  VrExport* ex = cur.vr ? EnsureExport(dd, sc) : nullptr;
  int slot = ex ? ChooseSlot(dd, ex) : -1;
  FrameView next = ComputeView(dd, sc);

  const bool synth = EnsureSynth(dd, sc);
  if (frame >= dd->lastFitFrame + 3 || !dd->proj[dd->lastCopyGroup & 3].ok) {
    FitProjection(dd);
    dd->lastFitFrame = frame;
  }
  if (frame >= dd->lastClassFrame + 15) {
    UpdateMainDepthClass(dd);
    dd->lastClassFrame = frame;
  }
  const uint64_t sfa = dd->stageFrameAt.load();
  if (sfa == UINT64_MAX || frame > sfa + 10) {  // (unless the passes tell where frames begin)
    dd->calibGroup = 0;  // the next frame's scene stages count from 0
    std::lock_guard<std::mutex> l(dd->depthLock);
    for (auto& im : dd->stageImg) im = VK_NULL_HANDLE;
  }
  const uint32_t copied = dd->copiedStages.exchange(0);

  dd->stage = "present(depth): waiting for previous copy";
  dd->d.WaitForFences(dd->device, 1, &vi.fence, VK_TRUE, UINT64_MAX);
  dd->d.ResetFences(dd->device, 1, &vi.fence);
  dd->stage = "present(depth): recording";
  VkCommandBuffer cb = vi.cmd;
  dd->d.ResetCommandBuffer(cb, 0);
  VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  dd->d.BeginCommandBuffer(cb, &bi);
  const VkPipelineStageFlags ALL = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, XFER = VK_PIPELINE_STAGE_TRANSFER_BIT,
                             CS = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
  Barrier(dd, cb, vi.image, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, ALL,
          VK_ACCESS_MEMORY_WRITE_BIT, XFER, VK_ACCESS_TRANSFER_READ_BIT);
  // the scene's depth must come from THIS frame (menus and loading screens draw no 3D scene)
  const uint64_t lastDepth = dd->lastDepthFrame.load();
  const bool fresh = lastDepth != UINT64_MAX && lastDepth + 1 >= frame;
  const int realEye = int(frame & 1), otherEye = 1 - realEye;
  const bool stereo = Cfg().stereo;
  bool first = true;
  auto toSlot = [&](VkImage img, int e) {
    if (slot < 0) return;
    CopyToSlot(dd, cb, sc, ex, slot, img, e, first);
    first = false;
  };
  bool rebuilt = false;
  if (synth) {
    const uint32_t W = sc->vext.width, H = sc->vext.height;
    const VkDeviceSize px = VkDeviceSize(W) * H * 4;
    BufBarrier(dd, cb, XFER | CS, VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_SHADER_READ_BIT, XFER | CS,
               VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_SHADER_WRITE_BIT);
    VkBufferImageCopy r{};
    r.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    r.imageExtent = {W, H, 1};
    dd->d.CmdCopyImageToBuffer(cb, vi.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, sc->work, 1, &r);
    dd->d.CmdFillBuffer(cb, sc->work, px, px * 2, 0);
    SynthPC pc{};
    pc.color = sc->workAddr;
    VkDeviceSize strideBytes;
    {
      std::lock_guard<std::mutex> l(dd->depthLock);
      pc.depth = dd->depthAddr;
      strideBytes = dd->depthCap;
    }
    pc.keys = sc->workAddr + px;
    pc.outc = sc->workAddr + px * 3;
    pc.meta = dd->metaAddr;
    pc.prev = sc->workAddr + px * 5;
    pc.stats = dd->paramAddr + kStatsOffset;
    {
      std::lock_guard<std::mutex> l(dd->depthLock);
      // the setting changed (profile, Ctrl+Shift+Scroll Lock): learn again
      const int hudCfg = (Cfg().hudSeparation ? 1 : 0) | (Cfg().hudSplit << 1);
      if (hudCfg != dd->lastHudCfg) {
        dd->lastHudCfg = hudCfg;
        dd->snapBad = false;
        dd->loggedSnap = false;
        dd->autoSplit = 1;
        dd->snapTried = dd->snapMatched = 0;
      }
      if (dd->snapBad && frame >= dd->snapBadUntil) dd->snapBad = false;  // try again now and then
      pc.snap = dd->snapAddr;
      // the copies of the last few frames: the shader uses the one that matches this image
      for (uint32_t k = 0; k < kSnapSlots; ++k) {
        const uint64_t f = dd->slotFrame[k].load();
        if (f && f + 6 >= frame + 1) pc.snapMask |= 1u << k;
      }
      pc.snapStride = uint32_t(dd->snapSlotBytes / 4);
      const bool use = Cfg().hudSeparation && dd->bbUsable && !dd->snapBad && pc.snapMask &&
                       dd->snapSlotBytes == VkDeviceSize(W) * H * 4;
      pc.snapFlags = (use ? 1u : 0u) | (dd->bbSwapRB ? 2u : 0u) | (dd->bb10 ? 4u : 0u);
      if (use && !dd->loggedSnap) {
        dd->loggedSnap = true;
        Log(1, "HUD separation: on (scene copied after back buffer draw %u, before the HUD; the HUD stays where the "
               "game draws it, the same in both eyes)", SplitDraw(dd));
      }
    }
    pc.W = W;
    pc.H = H;
    // stages copied in this frame that have a known projection: the shader combines them
    uint32_t stages = copied;
    uint32_t tail[kMetaWords - 16] = {};  // words 16..67
    for (uint32_t g = 0; g < 4; ++g) {
      const DeviceData::Proj& pj = dd->proj[g];
      if (!pj.ok) stages &= ~(1u << g);
      memcpy(&tail[g], &pj.A, 4);
      memcpy(&tail[4 + g], &pj.B, 4);
    }
    tail[8] = stages;
    tail[9] = uint32_t(strideBytes / 4);
    // orientation: compare how many depth edges also are colour edges, as is and upside down
    if (dd->paramMap && frame >= dd->lastFlipCheck + 30) {
      dd->lastFlipCheck = frame;
      volatile uint32_t* st = (volatile uint32_t*)((char*)dd->paramMap + kStatsOffset);
      uint32_t eA = st[0], hA = st[1], eB = st[2], hB = st[3], sTot = st[4], sHud = st[5], matched = st[16],
               still = st[17];
      st[0] = st[1] = st[2] = st[3] = st[4] = st[5] = st[16] = st[17] = 0;
      {
        std::lock_guard<std::mutex> l(dd->depthLock);
        dd->snapMatched += matched;
        dd->snapTried -= std::min(dd->snapTried, still);  // still images do not tell
        // HUD separation sanity: the "HUD" (what changed after the copy) must be a small part of the
        // image, and some copy must match the frames; else the copy was taken at the wrong point
        // (automatic: try after the next draw) or this game does not draw "scene, then HUD"
        const bool tooMuchHud = sTot >= 200 && sHud * 10 > sTot * 4;
        const bool noMatch = dd->snapTried >= 120 && dd->snapMatched * 5 < dd->snapTried;
        if (dd->snapTried >= 120 && !tooMuchHud && !noMatch) {
          if (dd->snapMatched * 10 < dd->snapTried * 9)
            Log(1, "HUD separation: a copy matched %u of %u frames (split after back buffer draw %u; %u copies taken)",
                dd->snapMatched, dd->snapTried, SplitDraw(dd), dd->snapTaken.exchange(0));
          dd->snapTried = dd->snapMatched = 0;
          dd->snapTaken = 0;
        }
        if ((tooMuchHud || noMatch) && !dd->snapBad) {
          const char* why = tooMuchHud ? "too much of the image changed after the copy" : "no copy matches the frames";
          if (Cfg().hudSplit == 0 && dd->autoSplit < 6) {
            ++dd->autoSplit;
            Log(1, "HUD separation: %s; now trying the copy after back buffer draw %u", why, dd->autoSplit);
          } else {
            Log(1, "HUD separation: off for now (%s: not a scene + HUD back buffer, or another split point is needed: "
                   "Ctrl+Shift+Scroll Lock)", why);
            dd->snapBad = true;
            dd->snapBadUntil = frame + 1800;
            dd->autoSplit = 1;
          }
          dd->loggedSnap = false;
          dd->snapTried = dd->snapMatched = 0;
        }
      }
      if (eA >= 200 && eB >= 200) {
        double rA = double(hA) / eA, rB = double(hB) / eB;
        int vote = rB > rA * 1.3 + 0.02 ? 1 : rA > rB * 1.3 + 0.02 ? 0 : -1;
        if (vote < 0 || uint32_t(vote) == dd->depthFlip) {
          dd->flipVotes = 0;
        } else if (++dd->flipVotes >= 3) {
          dd->depthFlip = uint32_t(vote);
          dd->flipVotes = 0;
          Log(1, "depth mode: the depth buffer is %s (edges matching the image: %.0f%% as is, %.0f%% flipped)",
              vote ? "upside down: flipped" : "upright", rA * 100, rB * 100);
        }
      }
    }
    tail[10] = dd->depthFlip;
    float ys = ClipYSign(dd);
    memcpy(&tail[11], &ys, 4);
    tail[12] = Cfg().depthNearest ? 1u : 0u;  // depth_layers = nearest
    // the transform this frame was drawn with and the one of each eye (perspective rows)
    const EyeParams src = dd->hybrid ? cur.p[realEye] : CenterParams(cur);
    memcpy(&tail[16], src.m, 12 * 4);
    memcpy(&tail[28], cur.p[0].m, 12 * 4);
    memcpy(&tail[40], cur.p[1].m, 12 * 4);
    dd->d.CmdUpdateBuffer(cb, dd->metaBuf, 64, sizeof(tail), tail);
    const bool on = stages != 0 && pc.depth && dd->depthCopies.load() > 0 && fresh && stereo;
    pc.a = 0.0f;
    pc.b = on ? 1.0f : 0.0f;
    pc.eyes = dd->hybrid ? (1u << otherEye) : 3u;
    if (dd->hybrid)
      pc.hyb = (sc->prevValid[otherEye] ? 1u : 0u) | (sc->prevValid[realEye] ? 2u : 0u) | (uint32_t(realEye) << 4);
    // (hybrid: from one eye to the other; near objects need big shifts, and a clamped shift would
    // not match the real eye: they would tremble)
    pc.maxShift = float(W) * (dd->hybrid ? 0.5f : 0.25f);
    static const bool show = getenv("VRLINUX_DEPTH_SHOW") && getenv("VRLINUX_DEPTH_SHOW")[0] == '1';
    if (show) {
      pc.maxShift = -1.0f;
      pc.eyes = 3u;
    }
    const bool compute = !dd->hybrid || on || show;
    if (compute && (pc.snapFlags & 1u) && dd->paramBuf) {
      dd->d.CmdFillBuffer(cb, dd->paramBuf, kStatsOffset + 32, 32, 0);  // this frame's match counters
      std::lock_guard<std::mutex> l(dd->depthLock);
      ++dd->snapTried;
    } else {
      pc.snapFlags &= ~1u;
    }
    if (compute) {
      BufBarrier(dd, cb, XFER | ALL, VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_MEMORY_WRITE_BIT, CS,
                 VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
      const uint32_t gx = (W + 15) / 16, gy = (H + 15) / 16;
      if (pc.snapFlags & 1u) {  // which copy belongs to this frame (one sample per 16x16 pixels)
        dd->d.CmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, dd->synthPipe[2]);
        dd->d.CmdPushConstants(cb, dd->synthLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        dd->d.CmdDispatch(cb, (gx + 15) / 16, (gy + 15) / 16, 1);
        BufBarrier(dd, cb, CS, VK_ACCESS_SHADER_WRITE_BIT, CS, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
      }
      for (int pass = 0; pass < 2; ++pass) {
        dd->d.CmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, dd->synthPipe[pass]);
        dd->d.CmdPushConstants(cb, dd->synthLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        dd->d.CmdDispatch(cb, gx, gy, 1);
        BufBarrier(dd, cb, CS, VK_ACCESS_SHADER_WRITE_BIT, pass == 0 ? CS : XFER,
                   pass == 0 ? (VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT) : VK_ACCESS_TRANSFER_READ_BIT);
      }
      rebuilt = on;
    } else {
      BufBarrier(dd, cb, XFER, VK_ACCESS_TRANSFER_WRITE_BIT, XFER, VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT);
    }
    // eye e from buffer offset 'off' through the synth image
    auto fromBuf = [&](int e, VkDeviceSize off) {
      Barrier(dd, cb, sc->synth[e], VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, XFER,
              VK_ACCESS_TRANSFER_READ_BIT, XFER, VK_ACCESS_TRANSFER_WRITE_BIT);
      VkBufferImageCopy o{};
      o.bufferOffset = off;
      o.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
      o.imageExtent = {W, H, 1};
      dd->d.CmdCopyBufferToImage(cb, sc->work, sc->synth[e], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &o);
      Barrier(dd, cb, sc->synth[e], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, XFER,
              VK_ACCESS_TRANSFER_WRITE_BIT, XFER, VK_ACCESS_TRANSFER_READ_BIT);
      toSlot(sc->synth[e], e);
    };
    if (!dd->hybrid) {
      for (int e = 0; e < 2; ++e) fromBuf(e, px * 3 + px * VkDeviceSize(e));
    } else {
      toSlot(vi.image, realEye);  // the real eye is the game image itself
      if (on || show)
        fromBuf(otherEye, px * 3 + px * VkDeviceSize(otherEye));  // rebuilt from this frame's depth
      else if (sc->prevValid[otherEye])
        fromBuf(otherEye, px * 5 + px * VkDeviceSize(otherEye));  // no 3D scene: plain alternate
      else
        toSlot(vi.image, otherEye);
      // keep this real image for the next frames (holes of this eye, menus)
      BufBarrier(dd, cb, XFER | CS, VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_SHADER_WRITE_BIT, XFER,
                 VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT);
      VkBufferCopy keep{0, px * 5 + px * VkDeviceSize(realEye), px};
      dd->d.CmdCopyBuffer(cb, sc->work, sc->work, 1, &keep);
      sc->prevValid[realEye] = true;
    }
  } else {
    toSlot(vi.image, 0);
    toSlot(vi.image, 1);
  }
  if (slot >= 0) {
    FillMeta(ex->slots[slot].meta, cur, 0);
    FillMeta(ex->slots[slot].meta, cur, 1);
    ex->slots[slot].meta.mono = (synth && stereo) ? 0 : 1;
    ReleaseSlot(dd, cb, ex, slot);
  }
  if (realIdx != UINT32_MAX) MirrorToWindow(dd, sc, cb, vi.image, sc->realImages[realIdx]);
  Barrier(dd, cb, vi.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, XFER, 0,
          VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0);
  // the next frame: hybrid renders the other eye, depth the middle (with the pose sampled now).
  // flat_menus (hybrid): screens without a 3D scene (title, menus, loading) are drawn from the middle
  // of the eyes too (games that draw their 2D menus with a perspective camera)
  EyeParams nextP = RenderParams(dd, next, frame + 1);
  const uint64_t ld = dd->lastDepthFrame.load();
  if (dd->hybrid && Cfg().flatMenus && (ld == UINT64_MAX || ld + 10 < frame)) nextP = CenterParams(next);
  UpdateParams(dd, cb, nextP);
  {  // the HUD passes of the HUD separation (flag -2) use the middle of the eyes
    EyeParams hudP = CenterParams(next);
    dd->d.CmdUpdateBuffer(cb, dd->paramBuf, kHudParamsOffset, sizeof(hudP.m), hudP.m);
    BufBarrier(dd, cb, XFER, VK_ACCESS_TRANSFER_WRITE_BIT, ALL, VK_ACCESS_SHADER_READ_BIT);
  }
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
    dd->stage = "present(depth): submitting";
    std::lock_guard<std::mutex> l(GetQueueInfo(dd, subQueue)->m);
    VkResult sr = dd->d.QueueSubmit(subQueue, 1, &si, vi.fence);
    if (sr != VK_SUCCESS) return sr;
    if (slot >= 0) SubmitSlotFence(dd, subQueue, ex, slot);
  }
  if (slot >= 0) {
    Publish(dd, ex, slot);
    ++sc->statVr;
  }
  if (rebuilt) ++dd->rebuiltFrames;
  if (realIdx != UINT32_MAX) {
    dd->stage = "present(depth): presenting window";
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
    } else if (scs[i] && dd->depthMode) {
      r = PresentDepth(dd, queue, scs[i], pi->pImageIndices[i], waits, wc);
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
  if (!dd->replay && !dd->depthMode) lim *= 2.0f;  // alternate: one present per eye
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

#define DEPTH_HOOKS(X)                                                                                        \
  X(CreateImage) X(CreateImageView) X(DestroyImage) X(DestroyImageView) X(CmdEndRendering) X(CmdEndRenderingKHR) \
  X(CmdDraw) X(CmdDrawIndexed) X(CmdDrawIndirect) X(CmdDrawIndexedIndirect) X(CmdDrawIndirectCount)               \
  X(CmdDrawIndexedIndirectCount) X(CmdPipelineBarrier) X(CmdPipelineBarrier2) X(CmdPipelineBarrier2KHR)               \
  X(CmdCopyImage) X(CmdBlitImage) X(CmdCopyImage2) X(CmdCopyImage2KHR) X(CmdBlitImage2) X(CmdBlitImage2KHR)       \
  X(CmdResolveImage) X(CmdResolveImage2) X(CmdResolveImage2KHR) X(CmdExecuteCommands) X(CmdDrawMultiEXT)              \
  X(CmdDrawMultiIndexedEXT)

PFN_vkVoidFunction DeviceHook(const char* name) {
#define CHECK(n) if (!strcmp(name, "vk" #n)) return (PFN_vkVoidFunction)&hook_##n;
  if (Cfg().passthrough) return nullptr;  // launchers / helpers: no hooks at all
  if (Cfg().mode == 2 || Cfg().mode == 3) {  // depth / hybrid: follow the scene's depth buffer
    DEPTH_HOOKS(CHECK)
  }
  if (Cfg().fixShadows != 0) {  // screen-space rebuilds in compute shaders
    CHECK(CreateComputePipelines)
  }
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
