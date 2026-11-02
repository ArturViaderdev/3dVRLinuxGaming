#version 450
layout(location=0) out vec3 col;
void main(){ // small white HUD box at top-left, orthographic (w = 1)
  vec2 c[6] = vec2[](vec2(-0.9,-0.9),vec2(-0.7,-0.9),vec2(-0.7,-0.7),vec2(-0.9,-0.9),vec2(-0.7,-0.7),vec2(-0.9,-0.7));
  gl_Position = vec4(c[gl_VertexIndex], 0.0, 1.0); col = vec3(1);
}
