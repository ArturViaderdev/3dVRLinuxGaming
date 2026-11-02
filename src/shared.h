// vrlinux - protocol shared by the Vulkan layer (inside the game) and vrlinux-host (OpenXR).
//
//  * The host listens on the abstract unix socket "\0vrlinux-<uid>" (works across the
//    Steam Linux Runtime container: it shares the network namespace).
//  * On connect the host sends one message with a memfd: the SharedState below, mapped by both.
//  * The layer sends MsgHello with the file descriptors of the exported frame slots
//    (one VkDeviceMemory per slot, OPAQUE_FD). Each slot is one image of 2W x H: left eye | right eye.
//  * Frames are handed over lock-free through SharedState::latest / inuse / done.
//    The layer publishes a slot only after its GPU copy finished (fence), and never rewrites
//    the latest slot nor the slot the host is reading, so no GPU semaphores cross processes.
#pragma once
#include <atomic>
#include <cstdint>

namespace vrlinux {

constexpr uint32_t kMagic = 0x56524C58;  // "VRLX"
constexpr uint32_t kVersion = 4;
constexpr int kSlots = 3;

enum MsgType : uint32_t { MsgHelloType = 1, MsgByeType = 2 };

struct MsgHello {
  uint32_t type = MsgHelloType;
  uint32_t version = kVersion;
  uint32_t width = 0, height = 0;   // ONE eye
  uint32_t format = 0;              // VkFormat of the slot images (2*width x height)
  uint32_t imageFlags = 0;          // VkImageCreateFlags used by the exporter
  uint32_t imageUsage = 0;          // VkImageUsageFlags used by the exporter
  uint32_t slots = kSlots;
  uint64_t allocSize[kSlots] = {};
  uint8_t deviceUUID[16] = {};
  uint8_t driverUUID[16] = {};
  int32_t pid = 0;
  char game[64] = {};
};

struct MsgBye {
  uint32_t type = MsgByeType;
};

// Seqlock-protected pose block written by the host every headset frame.
struct HostPose {
  float camQ[4];        // orientation the game camera has / should have (LOCAL space, x y z w)
  float eyePos[2][3];   // eye positions (LOCAL space, meters)
  float eyeCam[2][3];   // eye positions relative to the recentred origin, in camera axes (m)
  float eyeHead[2][3];  // eye offsets relative to the head, head axes (IPD / 2 ...)
  float eyeHeadQ[2][4]; // eye view orientation relative to the head (canted displays)
  float eyeFov[2][4];   // headset field of view of each eye: left right up down (radians)
  float headRelQ[4];    // head orientation relative to the recentred forward direction
  float headQ[4];       // raw head orientation (LOCAL)
  float headPos[3];
};

struct SlotMeta {
  uint64_t frame;
  float camQ[2][4];     // per eye: orientation the eye was rendered with
  float eyePos[2][3];   // per eye: position the eye was rendered from
  float fov[2][4];      // per eye: left right up down (radians, left/down negative)
  uint32_t mono;        // 1 = both halves hold the same (left) image
  uint32_t pad;
};

// Settings chosen in the game (profiles / hotkeys) that the host needs.
struct LayerSettings {
  int32_t display;      // 0 = vr (projection, head tracked), 1 = screen (giant 3D screen)
  int32_t look;         // 0 = none (image fixed in the world), 1 = mouse (virtual mouse), 2 = shader (head rotation drawn in the shaders)
  int32_t headLook;     // runtime toggle of look (Ctrl+Shift+F10)
  int32_t lookPitch;    // 1 = pitch also driven by the mouse
  float mouseSens;      // mouse counts per degree
  float pitchLimit;     // degrees
  int32_t invertY;
  int32_t positional;   // head translation -> game camera
  float posLimit;       // meters
  float screenDist;     // screen mode: meters
  float screenWidth;    // screen mode: meters
  float fovH, fovV;     // degrees (game FOV, as configured)
  uint32_t recenter;    // incremented by the layer on Ctrl+Shift+Home
  uint32_t pad[7];
};

struct SharedState {
  uint32_t magic;
  uint32_t version;
  // ---- host -> layer
  std::atomic<uint32_t> poseSeq;  // odd while writing
  HostPose pose;
  std::atomic<uint32_t> sessionRunning;
  std::atomic<uint32_t> displayHzX100;  // headset refresh rate * 100
  std::atomic<uint32_t> recenterCount;  // host-side recentres (for logs)
  uint32_t recW, recH;                  // runtime recommended per-eye size
  float runtimeFov[4];                  // left eye fov reported by the runtime (radians)
  // ---- layer -> host
  LayerSettings settings;
  std::atomic<uint32_t> layerAlive;     // incremented by the layer every present
  SlotMeta meta[kSlots];
  std::atomic<uint64_t> latest;         // (frame << 4) | slot ; 0 = none
  std::atomic<uint64_t> inuse;          // host claimed (frame << 4) | slot
  std::atomic<uint64_t> done;           // host finished reading this frame number
};

inline uint64_t PackFS(uint64_t frame, uint32_t slot) { return (frame << 4) | slot; }
inline uint64_t FrameOf(uint64_t v) { return v >> 4; }
inline uint32_t SlotOf(uint64_t v) { return uint32_t(v & 0xF); }

}  // namespace vrlinux
