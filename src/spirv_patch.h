// vrlinux - SPIR-V eye/head patcher
// Injects into the last pre-rasterization stage of every pipeline a per-eye transform of the
// clip-space position, read at runtime from a GPU buffer (buffer device address constant):
//
//   perspective vertices (w != 1):
//     x' = m0*x + m1*y + m2*w + m3
//     y' = m4*x + m5*y + m6*w + m7
//     w' = m8*x + m9*y + m10*w + m11
//     z' = z * w'/w                        (keeps each vertex' NDC depth -> depth order kept)
//          if m16 > 0.5 and w' > 0: z' clamped to [0, 0.99999 w'] (geometry beside the old camera stays)
//   2D vertices inside the screen (w == 1, HUD):   x' = x*m14 + m12,  y' = y*m15 + m13
//   2D full-screen passes: untouched
//
// For a camera moved by (ox, oy, oz) metres (eye position + head movement) this is
//   x' = x - sx*U*ox,  y' = y - sy*U*oy,  w' = w + U*oz      (U = game units per metre)
// and for the classic "3D screen" stereo  x' = x + k*(w - convergence).
#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace stereo3d {

constexpr int kParamFloats = 20;  // 80 bytes per eye (m16 = depth clamp flag)

enum class PatchResult { Patched, NoPosition, NoEntryPoint, Unsupported, Error };

// execModel: SPIR-V ExecutionModel (0 = Vertex, 2 = TessellationEvaluation, 3 = Geometry)
PatchResult PatchStereo(const uint32_t* code, size_t wordCount, const char* entryName,
                        uint32_t execModel, uint64_t paramAddress,
                        std::vector<uint32_t>& out, std::string* log = nullptr);

// Quick scan: does this module contain an entry point that could need patching?
bool HasPreRasterEntryPoint(const uint32_t* code, size_t wordCount);

const char* PatchResultName(PatchResult r);

}  // namespace stereo3d
