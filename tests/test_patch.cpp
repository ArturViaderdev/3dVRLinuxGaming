#include "../src/spirv_patch.h"
#include <cstdio>
#include <fstream>
#include <iterator>
int main(int argc, char** argv) {
  if (argc < 5) { fprintf(stderr, "usage: in out entry model\n"); return 2; }
  std::ifstream f(argv[1], std::ios::binary);
  std::vector<char> b((std::istreambuf_iterator<char>(f)), {});
  std::vector<uint32_t> out; std::string log;
  auto r = stereo3d::PatchStereo((const uint32_t*)b.data(), b.size()/4, argv[3], atoi(argv[4]), 0x123456789ABCull, out, &log);
  printf("%s: %s %s\n", argv[1], stereo3d::PatchResultName(r), log.c_str());
  if (r != stereo3d::PatchResult::Patched) return 1;
  std::ofstream o(argv[2], std::ios::binary); o.write((const char*)out.data(), out.size()*4);
}
