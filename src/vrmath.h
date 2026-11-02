// vrlinux - tiny quaternion / vector helpers (OpenXR conventions: +x right, +y up, -z forward)
#pragma once
#include <cmath>

namespace vrlinux {

struct V3 {
  float x = 0, y = 0, z = 0;
};
struct Q {
  float x = 0, y = 0, z = 0, w = 1;
};

inline V3 operator+(V3 a, V3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
inline V3 operator-(V3 a, V3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
inline V3 operator*(V3 a, float s) { return {a.x * s, a.y * s, a.z * s}; }
inline float Len(V3 a) { return std::sqrt(a.x * a.x + a.y * a.y + a.z * a.z); }

inline Q Mul(Q a, Q b) {
  return {a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y, a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
          a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w, a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z};
}
inline Q Conj(Q q) { return {-q.x, -q.y, -q.z, q.w}; }
inline Q Norm(Q q) {
  float n = std::sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
  if (n < 1e-9f) return Q{};
  return {q.x / n, q.y / n, q.z / n, q.w / n};
}
inline V3 Rot(Q q, V3 v) {
  // v' = q v q*
  V3 u{q.x, q.y, q.z};
  float s = q.w;
  float d = u.x * v.x + u.y * v.y + u.z * v.z;
  V3 c{u.y * v.z - u.z * v.y, u.z * v.x - u.x * v.z, u.x * v.y - u.y * v.x};
  float uu = u.x * u.x + u.y * u.y + u.z * u.z;
  return {2 * d * u.x + (s * s - uu) * v.x + 2 * s * c.x, 2 * d * u.y + (s * s - uu) * v.y + 2 * s * c.y,
          2 * d * u.z + (s * s - uu) * v.z + 2 * s * c.z};
}
// rotation matrix (row-major) of q
inline void ToMat(Q q, float m[3][3]) {
  float x = q.x, y = q.y, z = q.z, w = q.w;
  m[0][0] = 1 - 2 * (y * y + z * z); m[0][1] = 2 * (x * y - z * w);     m[0][2] = 2 * (x * z + y * w);
  m[1][0] = 2 * (x * y + z * w);     m[1][1] = 1 - 2 * (x * x + z * z); m[1][2] = 2 * (y * z - x * w);
  m[2][0] = 2 * (x * z - y * w);     m[2][1] = 2 * (y * z + x * w);     m[2][2] = 1 - 2 * (x * x + y * y);
}
inline Q AxisAngle(V3 axis, float rad) {
  float s = std::sin(rad * 0.5f);
  return {axis.x * s, axis.y * s, axis.z * s, std::cos(rad * 0.5f)};
}
inline Q Yaw(float rad) { return AxisAngle({0, 1, 0}, rad); }    // + = turn left
inline Q Pitch(float rad) { return AxisAngle({1, 0, 0}, rad); }  // + = look up

// yaw (about +Y) and pitch (about +X) of the forward vector of q, radians
inline void YawPitch(Q q, float& yaw, float& pitch) {
  V3 f = Rot(q, {0, 0, -1});
  yaw = std::atan2(-f.x, -f.z);
  float h = std::sqrt(f.x * f.x + f.z * f.z);
  pitch = std::atan2(f.y, h);
}

inline float WrapPi(float a) {
  const float tp = 6.28318530718f;
  while (a > 3.14159265359f) a -= tp;
  while (a < -3.14159265359f) a += tp;
  return a;
}

}  // namespace vrlinux
