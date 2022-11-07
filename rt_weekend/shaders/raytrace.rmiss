#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_GOOGLE_include_directive : enable
#include "raycommon.glsl"

layout(location = 0) rayPayloadInEXT hitPayload prd;

layout(push_constant) uniform Constants
{
    vec4 clearColor;
};

struct Ray
{
    vec3 origin;
    vec3 direction;
};

vec3 ray_color(const Ray r) {
    vec3 unit_direction = normalize(r.direction);
    float t = 0.5 * (unit_direction.y + 1.0);
    return (1.0-t) * vec3(1.0, 1.0, 1.0) + t * vec3(0.5, 0.7, 1.0);
}

void main()
{
    Ray ray;
    ray.origin = gl_WorldRayOriginEXT;
    ray.direction = gl_WorldRayDirectionEXT;

    prd.hasHit = false;
    prd.hitValue = ray_color(ray);
}
