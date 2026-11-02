struct O { float4 pos : SV_Position; float2 uv : TEXCOORD0; };
cbuffer C : register(b0) { float4x4 mvp; };
O main(float3 p : POSITION, uint id : SV_VertexID) { O o; o.pos = mul(mvp, float4(p,1)); o.uv = float2(id&1, id>>1); return o; }
