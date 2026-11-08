// vrlinux - SPIR-V eye/head patcher (see spirv_patch.h). Based on stereo3d.
#include "spirv_patch.h"

#include <cstring>
#include <unordered_map>
#include <unordered_set>

namespace stereo3d {

namespace {

// --- SPIR-V constants we need -------------------------------------------------
enum : uint32_t {
  OpSourceContinued = 2, OpSource = 3, OpSourceExtension = 4, OpName = 5, OpMemberName = 6,
  OpString = 7, OpLine = 8, OpExtension = 10, OpExtInstImport = 11, OpMemoryModel = 14,
  OpEntryPoint = 15, OpExecutionMode = 16, OpCapability = 17,
  OpTypeBool = 20, OpTypeInt = 21, OpTypeFloat = 22, OpTypeVector = 23,
  OpTypeArray = 28, OpTypeStruct = 30, OpTypePointer = 32,
  OpConstant = 43, OpConstantComposite = 44,
  OpFunction = 54, OpFunctionEnd = 56, OpVariable = 59,
  OpLoad = 61, OpStore = 62, OpAccessChain = 65,
  OpDecorate = 71, OpMemberDecorate = 72, OpDecorationGroup = 73, OpGroupDecorate = 74,
  OpGroupMemberDecorate = 75,
  OpConvertFToU = 109, OpUConvert = 113, OpBitcast = 124, OpIAdd = 128, OpISub = 130, OpIMul = 132,
  OpShiftRightLogical = 194, OpShiftLeftLogical = 196, OpBitwiseOr = 197, OpBitwiseXor = 198, OpBitwiseAnd = 199,
  OpIEqual = 170, OpSelectionMerge = 247, OpLabel = 248, OpBranch = 249, OpBranchConditional = 250,
  OpCompositeConstruct = 80, OpCompositeExtract = 81, OpCompositeInsert = 82, OpConvertUToPtr = 120,
  OpFAdd = 129, OpFSub = 131, OpFMul = 133, OpFDiv = 136, OpLogicalOr = 166, OpLogicalAnd = 167, OpSelect = 169, OpFOrdEqual = 180,
  OpFOrdNotEqual = 182, OpFOrdLessThan = 184, OpFOrdGreaterThan = 186,
  OpEmitVertex = 218, OpEmitStreamVertex = 220, OpReturn = 253,
  OpModuleProcessed = 330, OpExecutionModeId = 331, OpDecorateId = 332,
  OpDecorateString = 5632, OpMemberDecorateString = 5633,
};
enum : uint32_t {
  CapInt64 = 11, CapPhysicalStorageBufferAddresses = 5347,
  AddrLogical = 0, AddrPhysicalStorageBuffer64 = 5348,
  StorageOutput = 3, StoragePhysicalStorageBuffer = 5349,
  DecBlock = 2, DecBuiltIn = 11, DecOffset = 35, BuiltInPosition = 0,
  MemAligned = 2,
};

struct Inst {
  size_t off;   // word offset
  uint32_t op;
  uint32_t wc;  // word count
};

std::string ReadString(const uint32_t* w, uint32_t maxWords) {
  std::string s;
  const char* c = reinterpret_cast<const char*>(w);
  size_t maxBytes = size_t(maxWords) * 4;
  for (size_t i = 0; i < maxBytes && c[i]; ++i) s.push_back(c[i]);
  return s;
}

uint32_t StringWords(const std::string& s) { return uint32_t(s.size() / 4 + 1); }

void Emit(std::vector<uint32_t>& o, uint32_t op, std::initializer_list<uint32_t> operands) {
  o.push_back(((uint32_t(operands.size()) + 1) << 16) | op);
  for (uint32_t v : operands) o.push_back(v);
}

void EmitString(std::vector<uint32_t>& o, uint32_t op, const std::string& s) {
  uint32_t n = StringWords(s);
  o.push_back(((n + 1) << 16) | op);
  std::vector<uint32_t> words(n, 0);
  std::memcpy(words.data(), s.data(), s.size());
  o.insert(o.end(), words.begin(), words.end());
}

bool IsPreTypeSection(uint32_t op) {
  switch (op) {
    case OpCapability: case OpExtension: case OpExtInstImport: case OpMemoryModel:
    case OpEntryPoint: case OpExecutionMode: case OpExecutionModeId:
    case OpString: case OpSourceExtension: case OpSource: case OpSourceContinued:
    case OpName: case OpMemberName: case OpModuleProcessed:
    case OpDecorate: case OpMemberDecorate: case OpDecorationGroup: case OpGroupDecorate:
    case OpGroupMemberDecorate: case OpDecorateId: case OpDecorateString:
    case OpMemberDecorateString:
      return true;
    default:
      return false;
  }
}

bool ParseInsts(const uint32_t* code, size_t n, std::vector<Inst>& insts) {
  if (n < 5 || code[0] != 0x07230203) return false;
  size_t i = 5;
  while (i < n) {
    uint32_t wc = code[i] >> 16, op = code[i] & 0xFFFF;
    if (wc == 0 || i + wc > n) return false;
    insts.push_back({i, op, wc});
    i += wc;
  }
  return true;
}

}  // namespace

const char* PatchResultName(PatchResult r) {
  switch (r) {
    case PatchResult::Patched: return "patched";
    case PatchResult::NoPosition: return "no-position";
    case PatchResult::NoEntryPoint: return "no-entry-point";
    case PatchResult::Unsupported: return "unsupported";
    default: return "error";
  }
}

bool HasPreRasterEntryPoint(const uint32_t* code, size_t n) {
  std::vector<Inst> insts;
  if (!ParseInsts(code, n, insts)) return false;
  for (const Inst& in : insts) {
    if (in.op == OpEntryPoint && in.wc >= 3) {
      uint32_t model = code[in.off + 1];
      if (model == 0 || model == 2 || model == 3) return true;
    }
    if (in.op == OpFunction) break;
  }
  return false;
}

PatchResult PatchStereo(const uint32_t* code, size_t n, const char* entryName, uint32_t execModel,
                        uint64_t paramAddress, std::vector<uint32_t>& out, std::string* log, bool calib) {
  std::vector<Inst> insts;
  if (!ParseInsts(code, n, insts)) return PatchResult::Error;
  if (execModel != 0 && execModel != 2 && execModel != 3) return PatchResult::Unsupported;

  uint32_t bound = code[3];
  auto fresh = [&]() { return bound++; };

  // --- gather module info --------------------------------------------------------
  std::unordered_set<uint32_t> caps;
  std::unordered_set<std::string> exts;
  uint32_t entryFunc = 0;
  std::vector<uint32_t> interfaceIds;
  std::unordered_set<uint32_t> positionVars;                       // decorated var ids
  std::unordered_map<uint32_t, uint32_t> positionMembers;          // struct id -> member idx
  std::unordered_map<uint32_t, const uint32_t*> types;             // type id -> inst words
  struct Var { uint32_t type; uint32_t storage; };
  std::unordered_map<uint32_t, Var> globals;
  size_t firstFunctionIdx = insts.size();
  size_t annotationEndIdx = insts.size();
  uint32_t idBool = 0, idU32 = 0, idU64 = 0, idF32 = 0;

  for (size_t ii = 0; ii < insts.size(); ++ii) {
    const Inst& in = insts[ii];
    const uint32_t* w = code + in.off;
    if (annotationEndIdx == insts.size() && !IsPreTypeSection(in.op)) annotationEndIdx = ii;
    switch (in.op) {
      case OpCapability: caps.insert(w[1]); break;
      case OpExtension: exts.insert(ReadString(w + 1, in.wc - 1)); break;
      case OpEntryPoint: {
        if (in.wc < 4) break;
        std::string name = ReadString(w + 3, in.wc - 3);
        if (w[1] == execModel && name == entryName) {
          entryFunc = w[2];
          uint32_t nameWords = StringWords(name);
          for (uint32_t k = 3 + nameWords; k < in.wc; ++k) interfaceIds.push_back(w[k]);
        }
        break;
      }
      case OpDecorate:
        if (in.wc >= 4 && w[2] == DecBuiltIn && w[3] == BuiltInPosition) positionVars.insert(w[1]);
        break;
      case OpMemberDecorate:
        if (in.wc >= 5 && w[3] == DecBuiltIn && w[4] == BuiltInPosition)
          positionMembers[w[1]] = w[2];
        break;
      case OpTypeBool: if (!idBool) idBool = w[1]; types[w[1]] = w; break;
      case OpTypeInt:
        types[w[1]] = w;
        if (w[2] == 32 && w[3] == 0 && !idU32) idU32 = w[1];
        if (w[2] == 64 && w[3] == 0 && !idU64) idU64 = w[1];
        break;
      case OpTypeFloat:
        types[w[1]] = w;
        if (w[2] == 32 && !idF32) idF32 = w[1];
        break;
      case OpTypeVector: case OpTypeStruct: case OpTypePointer: case OpTypeArray:
        types[w[1]] = w;
        break;
      case OpVariable:
        if (firstFunctionIdx == insts.size()) globals[w[2]] = {w[1], w[3]};
        break;
      case OpFunction:
        if (firstFunctionIdx == insts.size()) firstFunctionIdx = ii;
        break;
      default: break;
    }
  }
  if (!entryFunc) return PatchResult::NoEntryPoint;
  if (firstFunctionIdx == insts.size()) return PatchResult::Error;

  // --- find the Position output ---------------------------------------------------
  uint32_t posVar = 0, posMember = UINT32_MAX, vec4Type = 0;
  auto checkVar = [&](uint32_t id) -> bool {
    auto g = globals.find(id);
    if (g == globals.end() || g->second.storage != StorageOutput) return false;
    auto pt = types.find(g->second.type);
    if (pt == types.end() || (pt->second[0] & 0xFFFF) != OpTypePointer) return false;
    uint32_t pointee = pt->second[3];
    if (positionVars.count(id)) {
      posVar = id; posMember = UINT32_MAX; vec4Type = pointee; return true;
    }
    auto st = types.find(pointee);
    if (st != types.end() && (st->second[0] & 0xFFFF) == OpTypeStruct) {
      auto pm = positionMembers.find(pointee);
      if (pm != positionMembers.end()) {
        uint32_t memberCount = (st->second[0] >> 16) - 2;
        if (pm->second < memberCount) {
          posVar = id; posMember = pm->second; vec4Type = st->second[2 + pm->second];
          return true;
        }
      }
    }
    return false;
  };
  for (uint32_t id : interfaceIds)
    if (checkVar(id)) break;
  if (!posVar) {  // SPIR-V < 1.4 may omit nothing, but be lenient
    for (auto& g : globals)
      if (checkVar(g.first)) break;
  }
  if (!posVar) return PatchResult::NoPosition;

  // vec4 of float32?
  auto vt = types.find(vec4Type);
  if (vt == types.end() || (vt->second[0] & 0xFFFF) != OpTypeVector || vt->second[3] != 4)
    return PatchResult::Unsupported;
  uint32_t floatType = vt->second[2];
  auto ft = types.find(floatType);
  if (ft == types.end() || (ft->second[0] & 0xFFFF) != OpTypeFloat || ft->second[2] != 32)
    return PatchResult::Unsupported;

  // --- allocate ids ----------------------------------------------------------------
  bool addBool = !idBool, addU32 = !idU32, addU64 = !idU64;
  if (addBool) idBool = fresh();
  if (addU32) idU32 = fresh();
  if (addU64) idU64 = fresh();
  uint32_t idStruct = fresh(), idPtrStruct = fresh(), idPtrFloat = fresh();
  uint32_t idPtrOutVec4 = (posMember != UINT32_MAX) ? fresh() : 0;
  uint32_t idAddr = fresh(), idOne = fresh(), idEdge = fresh(), idTiny = fresh(), idZero = fresh(), idHalf = fresh(),
           idAlmost = fresh();
  uint32_t idIdx[kParamFloats];
  for (int i = 0; i < kParamFloats; ++i) idIdx[i] = fresh();
  uint32_t idMember = (posMember != UINT32_MAX) ? fresh() : 0;

  const bool isGeometry = (execModel == 3);
  // projection samples: only at OpReturn sites (splitting a geometry shader's block around
  // OpEmitVertex could break structured control flow)
  if (isGeometry) calib = false;
  uint32_t idPtrU64 = 0, idK63 = 0, idK6 = 0, idK3 = 0, idK32 = 0, idOff64 = 0, idK7 = 0, idK18 = 0, idK26 = 0,
           idKMul = 0, idFlagOff = 0, idC0 = 0, idC1 = 0;
  if (calib) {
    idPtrU64 = fresh(); idK63 = fresh(); idK6 = fresh(); idK3 = fresh(); idK32 = fresh(); idOff64 = fresh();
    idK7 = fresh(); idK18 = fresh(); idK26 = fresh(); idKMul = fresh(); idFlagOff = fresh(); idC0 = fresh();
    idC1 = fresh();
  }
  int patchSites = 0;

  // --- emit -----------------------------------------------------------------------
  out.clear();
  out.reserve(n + 256);
  out.insert(out.end(), code, code + 5);

  bool capsDone = false;
  uint32_t currentFunc = 0;
  for (size_t ii = 0; ii < insts.size(); ++ii) {
    const Inst& in = insts[ii];
    const uint32_t* w = code + in.off;

    if (!capsDone && in.op != OpCapability) {
      capsDone = true;
      if (!caps.count(CapPhysicalStorageBufferAddresses))
        Emit(out, OpCapability, {CapPhysicalStorageBufferAddresses});
      if (!caps.count(CapInt64)) Emit(out, OpCapability, {CapInt64});
      if (!exts.count("SPV_KHR_physical_storage_buffer"))
        EmitString(out, OpExtension, "SPV_KHR_physical_storage_buffer");
    }

    if (ii == annotationEndIdx) {
      Emit(out, OpDecorate, {idStruct, DecBlock});
      for (uint32_t i = 0; i < uint32_t(kParamFloats); ++i) Emit(out, OpMemberDecorate, {idStruct, i, DecOffset, i * 4});
    }

    if (ii == firstFunctionIdx) {
      if (addBool) Emit(out, OpTypeBool, {idBool});
      if (addU32) Emit(out, OpTypeInt, {idU32, 32, 0});
      if (addU64) Emit(out, OpTypeInt, {idU64, 64, 0});
      {
        std::vector<uint32_t> st = {((uint32_t(kParamFloats) + 2) << 16) | OpTypeStruct, idStruct};
        for (int i = 0; i < kParamFloats; ++i) st.push_back(floatType);
        out.insert(out.end(), st.begin(), st.end());
      }
      Emit(out, OpTypePointer, {idPtrStruct, StoragePhysicalStorageBuffer, idStruct});
      Emit(out, OpTypePointer, {idPtrFloat, StoragePhysicalStorageBuffer, floatType});
      if (idPtrOutVec4) Emit(out, OpTypePointer, {idPtrOutVec4, StorageOutput, vec4Type});
      Emit(out, OpConstant, {idU64, idAddr, uint32_t(paramAddress & 0xFFFFFFFFu),
                             uint32_t(paramAddress >> 32)});
      for (int i = 0; i < kParamFloats; ++i) Emit(out, OpConstant, {idU32, idIdx[i], uint32_t(i)});
      Emit(out, OpConstant, {floatType, idEdge, 0x3F7F7CEEu});  // 0.998 (x^2 limit for HUD vertices)
      Emit(out, OpConstant, {floatType, idTiny, 0x2B8CBCCCu});  // 1e-12 (w^2 limit)
      if (idMember) Emit(out, OpConstant, {idU32, idMember, posMember});
      Emit(out, OpConstant, {floatType, idOne, 0x3F800000u});
      Emit(out, OpConstant, {floatType, idZero, 0u});
      Emit(out, OpConstant, {floatType, idHalf, 0x3F000000u});
      Emit(out, OpConstant, {floatType, idAlmost, 0x3F7FFF58u});  // 0.99999
      if (calib) {
        Emit(out, OpTypePointer, {idPtrU64, StoragePhysicalStorageBuffer, idU64});
        Emit(out, OpConstant, {idU32, idK63, 63});
        Emit(out, OpConstant, {idU32, idK6, 6});
        Emit(out, OpConstant, {idU32, idK3, 3});
        Emit(out, OpConstant, {idU32, idK32, 32});
        Emit(out, OpConstant, {idU64, idOff64, kCalibOffset, 0});
        Emit(out, OpConstant, {idU64, idFlagOff, kCalibFlagOffset, 0});
        Emit(out, OpConstant, {idU32, idK7, 7});
        Emit(out, OpConstant, {idU32, idK18, 18});
        Emit(out, OpConstant, {idU32, idK26, 26});
        Emit(out, OpConstant, {idU32, idKMul, 0x9E3779B1u});
        Emit(out, OpConstant, {idU32, idC0, 0});
        Emit(out, OpConstant, {idU32, idC1, 1});
      }
    }

    if (in.op == OpFunction) currentFunc = w[2];
    if (in.op == OpFunctionEnd) currentFunc = 0;

    bool site = isGeometry ? (in.op == OpEmitVertex || in.op == OpEmitStreamVertex)
                           : (in.op == OpReturn && currentFunc == entryFunc);
    if (site) {
      ++patchSites;
      const uint32_t F = floatType, B = idBool;
      uint32_t ptrS = fresh();
      Emit(out, OpConvertUToPtr, {idPtrStruct, ptrS, idAddr});
      uint32_t m[kParamFloats];
      for (int i = 0; i < kParamFloats; ++i) {
        uint32_t pm = fresh();
        m[i] = fresh();
        Emit(out, OpAccessChain, {idPtrFloat, pm, ptrS, idIdx[i]});
        Emit(out, OpLoad, {F, m[i], pm, MemAligned, 4});
      }
      uint32_t posPtr = posVar;
      if (idPtrOutVec4) {
        posPtr = fresh();
        Emit(out, OpAccessChain, {idPtrOutVec4, posPtr, posVar, idMember});
      }
      uint32_t p = fresh(), x = fresh(), y = fresh(), z = fresh(), wv = fresh();
      Emit(out, OpLoad, {vec4Type, p, posPtr});
      Emit(out, OpCompositeExtract, {F, x, p, 0});
      Emit(out, OpCompositeExtract, {F, y, p, 1});
      Emit(out, OpCompositeExtract, {F, z, p, 2});
      Emit(out, OpCompositeExtract, {F, wv, p, 3});
      // row r: m[4r]*x + m[4r+1]*y + m[4r+2]*w + m[4r+3]
      // 2D vertex inside the screen (HUD) or full-screen pass?
      uint32_t xx = fresh(), yy = fresh(), inX = fresh(), inY = fresh(), inside = fresh();
      Emit(out, OpFMul, {F, xx, x, x});
      Emit(out, OpFMul, {F, yy, y, y});
      Emit(out, OpFOrdLessThan, {B, inX, xx, idEdge});
      Emit(out, OpFOrdLessThan, {B, inY, yy, idEdge});
      Emit(out, OpLogicalAnd, {B, inside, inX, inY});
      // HUD fixed in the world (m17 > 0.5): the HUD vertex becomes a point at depth m18 in front of
      // the game camera (scaled by m19 = hud_scale * depth) and goes through the same transform
      uint32_t is2d = fresh(), hwOn = fresh(), hw0 = fresh(), hw = fresh(), persp = fresh();
      Emit(out, OpFOrdEqual, {B, is2d, wv, idOne});
      Emit(out, OpFOrdGreaterThan, {B, hwOn, m[17], idHalf});
      Emit(out, OpLogicalAnd, {B, hw0, is2d, inside});
      Emit(out, OpLogicalAnd, {B, hw, hw0, hwOn});
      uint32_t notPersp = fresh();
      Emit(out, OpFOrdNotEqual, {B, notPersp, wv, idOne});
      Emit(out, OpLogicalOr, {B, persp, notPersp, hw});
      uint32_t xs = fresh(), ys2 = fresh(), xi = fresh(), yi = fresh(), wi = fresh();
      Emit(out, OpFMul, {F, xs, x, m[19]});
      Emit(out, OpFMul, {F, ys2, y, m[19]});
      Emit(out, OpSelect, {F, xi, hw, xs, x});
      Emit(out, OpSelect, {F, yi, hw, ys2, y});
      Emit(out, OpSelect, {F, wi, hw, m[18], wv});
      auto row = [&](int r) {
        uint32_t a = fresh(), b = fresh(), c = fresh(), s1 = fresh(), s2 = fresh(), s3 = fresh();
        Emit(out, OpFMul, {F, a, m[4 * r], xi});
        Emit(out, OpFMul, {F, b, m[4 * r + 1], yi});
        Emit(out, OpFMul, {F, c, m[4 * r + 2], wi});
        Emit(out, OpFAdd, {F, s1, a, b});
        Emit(out, OpFAdd, {F, s2, s1, c});
        Emit(out, OpFAdd, {F, s3, s2, m[4 * r + 3]});
        return s3;
      };
      uint32_t nx = row(0), ny = row(1), nw = row(2);
      // z' = z * w'/w (guarded)
      uint32_t ww = fresh(), okw = fresh(), q = fresh(), ratio = fresh(), zr = fresh();
      Emit(out, OpFMul, {F, ww, wv, wv});
      Emit(out, OpFOrdGreaterThan, {B, okw, ww, idTiny});
      Emit(out, OpFDiv, {F, q, nw, wv});
      Emit(out, OpSelect, {F, ratio, okw, q, idOne});
      Emit(out, OpFMul, {F, zr, z, ratio});
      // optional clamp into the depth range (m16 > 0.5 and w' > 0)
      uint32_t hi = fresh(), cHi = fresh(), z1 = fresh(), cLo = fresh(), z2 = fresh(), en = fresh(), wpos = fresh(),
               doC = fresh(), nz = fresh();
      Emit(out, OpFMul, {F, hi, nw, idAlmost});
      Emit(out, OpFOrdGreaterThan, {B, cHi, zr, hi});
      Emit(out, OpSelect, {F, z1, cHi, hi, zr});
      Emit(out, OpFOrdLessThan, {B, cLo, z1, idZero});
      Emit(out, OpSelect, {F, z2, cLo, idZero, z1});
      Emit(out, OpFOrdGreaterThan, {B, en, m[16], idHalf});
      Emit(out, OpFOrdGreaterThan, {B, wpos, nw, idZero});
      Emit(out, OpLogicalAnd, {B, doC, en, wpos});
      Emit(out, OpSelect, {F, nz, doC, z2, zr});
      // 2D (w == 1): HUD vertices inside the screen get scaled + shifted; full-screen passes stay put
      uint32_t hxs = fresh(), hx = fresh(), hys = fresh(), hy = fresh(), hudx = fresh(), hudy = fresh();
      Emit(out, OpFMul, {F, hxs, x, m[14]});
      Emit(out, OpFAdd, {F, hx, hxs, m[12]});
      Emit(out, OpFMul, {F, hys, y, m[15]});
      Emit(out, OpFAdd, {F, hy, hys, m[13]});
      Emit(out, OpSelect, {F, hudx, inside, hx, x});
      Emit(out, OpSelect, {F, hudy, inside, hy, y});
      uint32_t fx = fresh(), fy = fresh(), fz = fresh(), fw = fresh(), np = fresh();
      Emit(out, OpSelect, {F, fx, persp, nx, hudx});
      Emit(out, OpSelect, {F, fy, persp, ny, hudy});
      Emit(out, OpSelect, {F, fz, persp, nz, z});
      Emit(out, OpSelect, {F, fw, persp, nw, wv});
      Emit(out, OpCompositeConstruct, {vec4Type, np, fx, fy, fz, fw});
      Emit(out, OpStore, {posPtr, np});
      if (calib) {
        // the game's own clip position (before the eye transform):
        // m = (bits(x) ^ (bits(y) >> 7) ^ bits(w) ^ (bits(z) >> 6)) * 0x9E3779B1
        // if (flag > 0.5 && w != 1 && (m >> 26) == 0)            (1 vertex in 64)
        //   slot[(flag - 1) * 64 + ((m >> 18) & 63)] = uint64(bits(w)) | (uint64(bits(z)) << 32)
        uint32_t fa = fresh(), fp = fresh(), fl = fresh(), on = fresh(), wb = fresh(), xb = fresh(), yb = fresh(),
                 zb = fresh(), zs = fresh(), ysh = fresh(), h1 = fresh(), h2 = fresh(), h3 = fresh(), hm = fresh(),
                 hh = fresh(), lo = fresh(), loz = fresh(), c1 = fresh(), cond = fresh(), lStore = fresh(),
                 lMerge = fresh();
        Emit(out, OpIAdd, {idU64, fa, idAddr, idFlagOff});
        Emit(out, OpConvertUToPtr, {idPtrFloat, fp, fa});
        Emit(out, OpLoad, {F, fl, fp, MemAligned, 4});
        Emit(out, OpFOrdGreaterThan, {B, on, fl, idHalf});
        Emit(out, OpBitcast, {idU32, wb, wv});
        Emit(out, OpBitcast, {idU32, xb, x});
        Emit(out, OpBitcast, {idU32, yb, y});
        Emit(out, OpBitcast, {idU32, zb, z});
        Emit(out, OpShiftRightLogical, {idU32, zs, zb, idK6});
        Emit(out, OpShiftRightLogical, {idU32, ysh, yb, idK7});
        Emit(out, OpBitwiseXor, {idU32, h1, xb, ysh});
        Emit(out, OpBitwiseXor, {idU32, h2, h1, wb});
        Emit(out, OpBitwiseXor, {idU32, h3, h2, zs});
        Emit(out, OpIMul, {idU32, hm, h3, idKMul});
        Emit(out, OpShiftRightLogical, {idU32, hh, hm, idK18});
        Emit(out, OpShiftRightLogical, {idU32, lo, hm, idK26});
        Emit(out, OpIEqual, {B, loz, lo, idC0});
        Emit(out, OpLogicalAnd, {B, c1, on, notPersp});
        Emit(out, OpLogicalAnd, {B, cond, c1, loz});
        Emit(out, OpSelectionMerge, {lMerge, 0});
        Emit(out, OpBranchConditional, {cond, lStore, lMerge});
        Emit(out, OpLabel, {lStore});
        uint32_t sh = fresh(), g = fresh(), g1 = fresh(), g2 = fresh(), gs = fresh(), slot = fresh(), so = fresh(),
                 so64 = fresh(), a1 = fresh(), a2 = fresh(), ptr = fresh(), w64 = fresh(), z64 = fresh(), zh = fresh(),
                 val = fresh();
        Emit(out, OpBitwiseAnd, {idU32, sh, hh, idK63});
        Emit(out, OpConvertFToU, {idU32, g, fl});
        Emit(out, OpISub, {idU32, g1, g, idC1});
        Emit(out, OpBitwiseAnd, {idU32, g2, g1, idK3});
        Emit(out, OpShiftLeftLogical, {idU32, gs, g2, idK6});
        Emit(out, OpBitwiseOr, {idU32, slot, gs, sh});
        Emit(out, OpShiftLeftLogical, {idU32, so, slot, idK3});
        Emit(out, OpUConvert, {idU64, so64, so});
        Emit(out, OpIAdd, {idU64, a1, idAddr, idOff64});
        Emit(out, OpIAdd, {idU64, a2, a1, so64});
        Emit(out, OpConvertUToPtr, {idPtrU64, ptr, a2});
        Emit(out, OpUConvert, {idU64, w64, wb});
        Emit(out, OpUConvert, {idU64, z64, zb});
        Emit(out, OpShiftLeftLogical, {idU64, zh, z64, idK32});
        Emit(out, OpBitwiseOr, {idU64, val, w64, zh});
        Emit(out, OpStore, {ptr, val, MemAligned, 8});
        Emit(out, OpBranch, {lMerge});
        Emit(out, OpLabel, {lMerge});
      }
    }

    if (in.op == OpMemoryModel && w[1] == AddrLogical) {
      Emit(out, OpMemoryModel, {AddrPhysicalStorageBuffer64, w[2]});
      continue;
    }
    out.insert(out.end(), w, w + in.wc);
  }

  if (patchSites == 0) return PatchResult::NoPosition;
  out[3] = bound;
  if (log) *log = "sites=" + std::to_string(patchSites);
  return PatchResult::Patched;
}

}  // namespace stereo3d
