#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_GOOGLE_include_directive : enable
#include "raycommon.glsl"

layout(location = 0) rayPayloadInEXT hitPayload prd;

layout(binding = 7, set = 1) uniform samplerCube skyboxSampler;


layout(push_constant) uniform Constants
{
  vec4 clearColor;
};

void main()
{
  vec3 dir = gl_WorldRayDirectionEXT;
  // prd.hitValue = clearColor.xyz * 0.8;
  prd.hitValue = texture(skyboxSampler, dir).rgb;
}
