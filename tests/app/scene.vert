#version 450
// Test scene in "game" clip space (like a D3D game through DXVK: y up, reverse-Z infinite).
// 4:3 window, horizontal FOV 90 degrees. 1 unit = 1 metre. Eye height 1.6 m.
layout(location=0) out vec3 col;
const float SX = 1.0;          // 1/tan(45 deg)
const float SY = 4.0 / 3.0;    // 1/tan(vfov/2) for 4:3
vec4 proj(vec3 p) { return vec4(p.x * SX, p.y * SY, 0.05, p.z); }
void main() {
  int q = gl_VertexIndex / 6, v = gl_VertexIndex % 6;
  vec2 c[6] = vec2[](vec2(0,0), vec2(1,0), vec2(1,1), vec2(0,0), vec2(1,1), vec2(0,1));
  vec3 p;
  if (q < 200) {            // floor tiles, 1 m, from 1 m to 21 m ahead
    int tx = q % 10, tz = q / 10;
    p = vec3(-5.0 + float(tx) + c[v].x, -1.6, 1.0 + float(tz) + c[v].y);
    col = ((tx + tz) % 2 == 1) ? vec3(0.75) : vec3(0.25, 0.3, 0.35);
  } else {                  // three standing panels
    int k = q - 200;
    float depth[3] = float[](2.0, 6.0, 15.0);
    float xo[3] = float[](-0.8, 0.6, -1.5);
    float s[3] = float[](0.6, 1.5, 3.0);
    vec3 cols[3] = vec3[](vec3(1,0,0), vec3(0,1,0), vec3(0,0.4,1));
    p = vec3(xo[k] + (c[v].x - 0.5) * s[k], -1.6 + c[v].y * s[k] * 1.4, depth[k]);
    col = cols[k];
  }
  // gl_InstanceIndex (APP_DEPTH: frame number) moves the scene by a hair: like a game's camera,
  // the vertices are never exactly the same from one frame to the next
  float o = float(gl_InstanceIndex % 64) * 0.0007 + float(q) * 0.0001;
  p += vec3(o);
  gl_Position = proj(p);
}
