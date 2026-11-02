// vrlinux - connection from the Vulkan layer (game process) to vrlinux-host
#pragma once
#include <cstdint>
#include <mutex>
#include <vector>

#include "shared.h"

namespace vrlinux {

class VrLink {
 public:
  // Tries to connect (at most once per second). Returns true while connected.
  bool Poll();
  bool Connected() const { return shm_ != nullptr; }
  SharedState* Shm() { return shm_; }
  // generation increases on every (re)connection: exports must re-send their hello
  uint32_t Generation() const { return gen_; }
  bool SendHello(const MsgHello& h, const std::vector<int>& fds);
  bool ReadPose(HostPose& out);  // seqlock read; false if not connected
  std::mutex& Mutex() { return m_; }
  void Disconnect(const char* why);

 private:
  int fd_ = -1;
  SharedState* shm_ = nullptr;
  uint32_t gen_ = 0;
  double lastTry_ = -10.0;
  std::mutex m_;
};

VrLink& Link();

}  // namespace vrlinux
