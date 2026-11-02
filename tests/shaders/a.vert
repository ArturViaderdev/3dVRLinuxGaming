#version 450
layout(location=0) in vec3 p; layout(location=0) out vec3 c;
layout(push_constant) uniform PC { mat4 mvp; } pc;
void main(){ c=p; if (p.x > 5.0) { gl_Position = vec4(0); return; } gl_Position = pc.mvp*vec4(p,1); }
