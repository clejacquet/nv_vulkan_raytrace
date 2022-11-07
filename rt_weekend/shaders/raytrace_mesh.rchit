#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_EXT_scalar_block_layout : enable
#extension GL_GOOGLE_include_directive : enable
#include "raycommon.glsl"
#include "random.glsl"
#include "wavefront.glsl"

hitAttributeEXT vec2 attribs;

// clang-format off
layout(location = 0) rayPayloadInEXT hitPayload prd;
layout(location = 1) rayPayloadEXT hitPayload prd_out;

layout(binding = 0, set = 0) uniform accelerationStructureEXT topLevelAS;

layout(binding = 2, set = 1, scalar) buffer ScnDesc { sceneDesc i[]; } scnDesc;

layout(binding = 1, set = 1, scalar) buffer MatColorBufferObject { WaveFrontMaterial m[]; } materials[];
layout(binding = 3, set = 1) uniform sampler2D textureSamplers[];
layout(binding = 4, set = 1)  buffer MatIndexColorBuffer { int i[]; } matIndex[];
layout(binding = 5, set = 1, scalar) buffer Vertices { Vertex v[]; } vertices[];
layout(binding = 6, set = 1) buffer Indices { uint i[]; } indices[];

// clang-format on

layout(push_constant) uniform Constants
{
    vec4  clearColor;
    vec3  lightPosition;
    float lightIntensity;
    int   lightType;
    int   frame;
}
pushC;

const int SAMPLES_COUNT = 4;

vec3 computeRandomScatterDirection(vec3 normal, inout uint rnd_seed) 
{
    return rndHemisphereVec(rnd_seed, normal);
}


void traceDiffuseMaterial(inout uint seed, vec3 world_pos, vec3 normal, vec3 color)
{
    uint flags = gl_RayFlagsOpaqueEXT | gl_RayFlagsTerminateOnFirstHitEXT | gl_RayFlagsSkipClosestHitShaderEXT;

    float final_coverage = 0.0f;

    prd_out.depth = prd.depth + 1;

    for (int i = 0; i < SAMPLES_COUNT; ++i) {
        vec3 direction = computeRandomScatterDirection(normal, seed);

        prd_out.hasHit = true;
        prd_out.depth = prd.depth + 1;

        traceRayEXT(
            topLevelAS,
            flags,
            0xff,
            0,
            0,
            1,
            world_pos,
            0.001f,
            direction,
            100.0f,
            1
        );

        final_coverage += (prd_out.hasHit) ? 1.0f : 0.0f;
    }

    final_coverage = final_coverage / float(SAMPLES_COUNT);
    final_coverage = 0.5 + (final_coverage - 0.5f) * (1.0f - 1.0f / float(pushC.frame + 8));

    prd.hitValue = mix(color, vec3(0.0f), final_coverage);
}

void traceMetalMaterial(inout uint seed, vec3 world_pos, vec3 normal, vec3 color)
{
    uint flags = gl_RayFlagsOpaqueEXT;

    vec3 direction = reflect(gl_WorldRayDirectionEXT, normal);
    direction = normalize(direction + 0.0 * rndHemisphereVec(seed, normal));

    prd_out.depth = prd.depth + 1;

    traceRayEXT(
        topLevelAS,
        flags,
        0xff,
        0,
        0,
        0,
        world_pos,
        0.001f,
        direction,
        100.0f,
        1
    );

    // prd.hitValue = vec3(0.8f, 0.6f, 0.2f) * prd_out.hitValue;
    prd.hitValue = color * prd_out.hitValue;
    prd.hasHit = true;
}

highp vec3 custom_refract(highp vec3 uv, highp vec3 n, highp float etai_over_etat) {
    highp float cos_theta = min(dot(-uv, n), 1.0);
    highp vec3 r_out_perp =  etai_over_etat * (uv + cos_theta*n);
    highp float r_out_perp_len = length(r_out_perp);
    highp vec3 r_out_parallel = -sqrt(abs(1.0 - r_out_perp_len * r_out_perp_len)) * n;
    return r_out_perp + r_out_parallel;
}

highp vec3 custom_refract2(const vec3 incidentVec, const vec3 normal, float eta)
{
  float N_dot_I = dot(normal, incidentVec);
  float k = 1.f - eta * eta * (1.f - N_dot_I * N_dot_I);
  if (k < 0.f)
    return vec3(0.f, 0.f, 0.f);
  else
    return eta * incidentVec - (eta * N_dot_I + sqrt(k)) * normal;
}

highp float reflectance(highp float cosine, highp float ref_idx) {
    // Use Schlick's approximation for reflectance.
    highp float r0 = (1.0 - ref_idx) / (1.0 + ref_idx);
    r0 = r0 * r0;
    return r0 + (1-r0) * pow((1.0 - cosine), 5.0);
}

void traceGlassMaterial(inout uint seed, highp vec3 world_pos, highp vec3 normal, bool is_front_face)
{
    uint flags = gl_RayFlagsOpaqueEXT;
    vec3 unit_dir = normalize(gl_WorldRayDirectionEXT);

    highp float eta = 1.5;

    highp float reflectance_ratio = is_front_face ? (1.0/eta) : eta;

    // vec3 direction = refract(unit_dir, normal, reflectance_ratio);

    float cos_theta = min(dot(-unit_dir, normal), 1.0);
    float sin_theta = sqrt(1.0 - cos_theta * cos_theta);

    bool cannot_refract = (reflectance_ratio * sin_theta) > 1.0;
    vec3 direction;

    float rnd_threshold = rnd(seed); 

    if (cannot_refract || reflectance(cos_theta, reflectance_ratio) > rnd_threshold) {
    // if (cannot_refract) {
        direction = reflect(unit_dir, normal);
    }
    else {
        direction = refract(unit_dir, normal, reflectance_ratio);
    }

    // highp vec3 direction = refract(normalize(unit_dir), normalize(normal), eta);
    // highp vec3 next_direction = custom_refract(normalize(unit_dir), normalize(normal), eta);

    prd_out.depth = prd.depth + 1;

    traceRayEXT(
        topLevelAS,
        flags,
        0xff,
        0,
        0,
        0,
        world_pos,
        0.0001,
        direction,
        1000.0,
        1
    );

    // prd.hitValue = vec3(0.8f, 0.6f, 0.2f) * prd_out.hitValue;
    prd.hitValue = prd_out.hitValue;
    prd.hasHit = true;
}

void main()
{
    // if (gl_HitTEXT < 0.1f) {
    //     prd.hitValue = vec3(1.0f, 0.0f, 0.0f);
    //     prd.hasHit = false;
    //     return;
    // }
    if (prd.depth >= 8) {
        prd.hitValue = vec3(0.0f);
        prd.hasHit = false;
        return;
    }

    uint seed = tea(gl_LaunchIDEXT.y * gl_LaunchSizeEXT.x + gl_LaunchIDEXT.x, pushC.frame);

    // Object of this instance
    uint objId = scnDesc.i[gl_InstanceCustomIndexEXT].objId;

    // Indices of the triangle
    ivec3 ind = ivec3(indices[nonuniformEXT(objId)].i[3 * gl_PrimitiveID + 0],   //
                        indices[nonuniformEXT(objId)].i[3 * gl_PrimitiveID + 1],   //
                        indices[nonuniformEXT(objId)].i[3 * gl_PrimitiveID + 2]);  //
    // Vertex of the triangle
    Vertex v0 = vertices[nonuniformEXT(objId)].v[ind.x];
    Vertex v1 = vertices[nonuniformEXT(objId)].v[ind.y];
    Vertex v2 = vertices[nonuniformEXT(objId)].v[ind.z];

    const vec3 barycentrics = vec3(1.0 - attribs.x - attribs.y, attribs.x, attribs.y);

    // Computing the normal at hit position
    vec3 normal = v0.nrm * barycentrics.x + v1.nrm * barycentrics.y + v2.nrm * barycentrics.z;
    // Transforming the normal to world space
    normal = normalize(vec3(scnDesc.i[gl_InstanceCustomIndexEXT].transfoIT * vec4(normal, 0.0)));


    // Computing the coordinates of the hit position
    vec3 worldPos = v0.pos * barycentrics.x + v1.pos * barycentrics.y + v2.pos * barycentrics.z;
    // Transforming the position to world space
    worldPos = vec3(scnDesc.i[gl_InstanceCustomIndexEXT].transfo * vec4(worldPos, 1.0));

    bool front_face = dot(normalize(gl_WorldRayDirectionEXT), normal) < 0.0;
    normal = (front_face ? normal : -normal);
    

    // if (gl_PrimitiveID % 2 == 0) {
        // traceDiffuseMaterial(seed, worldPos, normal, vec3(0.2f));
    // }
    // else {
    traceMetalMaterial(seed, worldPos, normal, vec3(0.8f, 0.6f, 0.2f));
    // traceGlassMaterial(seed, worldPos, normal, front_face);

    // }
    

    // vec3 light_dir = normalize(worldPos_corr - pushC.lightPosition);

    // vec3 ambient = vec3(0.1, 0.0, 0.0);
    // vec3 diffuse = vec3(0.9, 0.0, 0.0);
    // float diffuseComp = max(0, dot(normal, light_dir));
    // vec3 final_color = ambient + diffuseComp * diffuse;

    // prd.hitValue = final_color;
    // prd.hitValue = normal;
}
