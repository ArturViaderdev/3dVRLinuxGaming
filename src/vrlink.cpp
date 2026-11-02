// vrlinux - connection from the Vulkan layer (game process) to vrlinux-host
#include "vrlink.h"

#include <errno.h>
#include <poll.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <time.h>
#include <unistd.h>

#include <cstdio>
#include <cstring>

#include "config.h"

namespace vrlinux {

using stereo3d::Log;

VrLink& Link() {
  static VrLink l;
  return l;
}

static double Now() {
  timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static socklen_t SocketAddr(sockaddr_un& a) {
  memset(&a, 0, sizeof(a));
  a.sun_family = AF_UNIX;
  char name[64];
  int n = snprintf(name, sizeof(name), "vrlinux-%u", unsigned(getuid()));
  memcpy(a.sun_path + 1, name, size_t(n));  // abstract namespace: leading NUL
  return socklen_t(offsetof(sockaddr_un, sun_path) + 1 + n);
}

void VrLink::Disconnect(const char* why) {
  if (fd_ >= 0) close(fd_);
  fd_ = -1;
  if (shm_) {
    munmap(shm_, sizeof(SharedState));
    Log(0, "VR host disconnected (%s)", why);
  }
  shm_ = nullptr;
}

bool VrLink::Poll() {
  std::lock_guard<std::mutex> l(m_);
  if (fd_ >= 0) {  // still there?
    char c;
    ssize_t r = recv(fd_, &c, 1, MSG_DONTWAIT | MSG_PEEK);
    if (r == 0 || (r < 0 && errno != EAGAIN && errno != EWOULDBLOCK)) Disconnect("socket closed");
    else if (r > 0) {  // host messages are not used yet: drain
      char buf[256];
      while (recv(fd_, buf, sizeof(buf), MSG_DONTWAIT) > 0) {}
    }
    return fd_ >= 0;
  }
  double now = Now();
  if (now - lastTry_ < 1.0) return false;
  lastTry_ = now;
  int fd = socket(AF_UNIX, SOCK_SEQPACKET | SOCK_CLOEXEC, 0);
  if (fd < 0) return false;
  sockaddr_un a;
  socklen_t len = SocketAddr(a);
  if (connect(fd, (sockaddr*)&a, len) != 0) {
    close(fd);
    return false;
  }
  // the host answers with the shared-state memfd
  pollfd p{fd, POLLIN, 0};
  if (poll(&p, 1, 500) <= 0) {
    close(fd);
    Log(1, "VR host did not answer");
    return false;
  }
  char data[16];
  iovec iov{data, sizeof(data)};
  char ctrl[CMSG_SPACE(sizeof(int))];
  msghdr msg{};
  msg.msg_iov = &iov;
  msg.msg_iovlen = 1;
  msg.msg_control = ctrl;
  msg.msg_controllen = sizeof(ctrl);
  ssize_t r = recvmsg(fd, &msg, MSG_CMSG_CLOEXEC);
  int memfd = -1;
  for (cmsghdr* c = CMSG_FIRSTHDR(&msg); c; c = CMSG_NXTHDR(&msg, c))
    if (c->cmsg_level == SOL_SOCKET && c->cmsg_type == SCM_RIGHTS) memcpy(&memfd, CMSG_DATA(c), sizeof(int));
  if (r <= 0 || memfd < 0) {
    close(fd);
    if (memfd >= 0) close(memfd);
    return false;
  }
  void* m = mmap(nullptr, sizeof(SharedState), PROT_READ | PROT_WRITE, MAP_SHARED, memfd, 0);
  close(memfd);
  if (m == MAP_FAILED) {
    close(fd);
    return false;
  }
  auto* s = (SharedState*)m;
  if (s->magic != kMagic || s->version != kVersion) {
    Log(0, "VR host has another protocol version (%u, layer %u): reinstall both", s->version, kVersion);
    munmap(m, sizeof(SharedState));
    close(fd);
    lastTry_ = now + 10.0;
    return false;
  }
  fd_ = fd;
  shm_ = s;
  ++gen_;
  Log(0, "connected to vrlinux-host (headset %s, %.0f Hz)", s->sessionRunning.load() ? "running" : "not running yet",
      s->displayHzX100.load() / 100.0);
  return true;
}

bool VrLink::SendHello(const MsgHello& h, const std::vector<int>& fds) {
  std::lock_guard<std::mutex> l(m_);
  if (fd_ < 0) return false;
  iovec iov{(void*)&h, sizeof(h)};
  std::vector<char> ctrl(CMSG_SPACE(sizeof(int) * fds.size()));
  msghdr msg{};
  msg.msg_iov = &iov;
  msg.msg_iovlen = 1;
  msg.msg_control = ctrl.data();
  msg.msg_controllen = ctrl.size();
  cmsghdr* c = CMSG_FIRSTHDR(&msg);
  c->cmsg_level = SOL_SOCKET;
  c->cmsg_type = SCM_RIGHTS;
  c->cmsg_len = CMSG_LEN(sizeof(int) * fds.size());
  memcpy(CMSG_DATA(c), fds.data(), sizeof(int) * fds.size());
  ssize_t r = sendmsg(fd_, &msg, MSG_NOSIGNAL);
  if (r != ssize_t(sizeof(h))) {
    Disconnect("send failed");
    return false;
  }
  return true;
}

bool VrLink::ReadPose(HostPose& out) {
  SharedState* s = shm_;
  if (!s) return false;
  for (int tries = 0; tries < 1000; ++tries) {
    uint32_t a = s->poseSeq.load(std::memory_order_acquire);
    if (a & 1) continue;
    memcpy(&out, (const void*)&s->pose, sizeof(out));
    std::atomic_thread_fence(std::memory_order_acquire);
    if (s->poseSeq.load(std::memory_order_relaxed) == a) return true;
  }
  return false;
}

}  // namespace vrlinux
