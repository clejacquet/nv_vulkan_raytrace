#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_EXT_scalar_block_layout : enable
#extension GL_GOOGLE_include_directive : enable

#include "raycommon.glsl"
#include "wavefront.glsl"

layout(location = 0) rayPayloadInEXT hitPayload prd;
layout(location = 1) rayPayloadEXT bool isShadowed;
hitAttributeEXT vec3 attribs;

layout(binding = 0, set = 0) uniform accelerationStructureEXT topLevelAS;

layout(binding = 1, set = 1, scalar) buffer MatColorBufferObject { WaveFrontMaterial m[]; } materials[];
layout(binding = 2, set = 1, scalar) buffer ScnDesc { sceneDesc i[]; } scnDesc;
layout(binding = 3, set = 1) uniform sampler2D textureSamplers[];
layout(binding = 4, set = 1) buffer MatIndexColorBuffer { int i[]; } matIndex[];
layout(binding = 5, set = 1, scalar) buffer Vertices { Vertex v[]; } vertices[];
layout(binding = 6, set = 1) buffer Indices { uint i[]; } indices[];
layout(binding = 7, set = 1) uniform samplerCube skyboxSampler;


layout(push_constant) uniform Constants
{
    vec4 clearColor;
    vec3 lightPosition;
    float lightIntensity;
    int lightType;
} pushC;

void main()
{
    // Object of this instance
    uint objId = scnDesc.i[gl_InstanceCustomIndexEXT].objId;

    // Indices of the triangle
    ivec3 ind = ivec3(
        indices[nonuniformEXT(objId)].i[3 * gl_PrimitiveID + 0],
        indices[nonuniformEXT(objId)].i[3 * gl_PrimitiveID + 1],
        indices[nonuniformEXT(objId)].i[3 * gl_PrimitiveID + 2]
    );

    // Vertices of the triangle
    Vertex v0 = vertices[nonuniformEXT(objId)].v[ind.x];
    Vertex v1 = vertices[nonuniformEXT(objId)].v[ind.y];
    Vertex v2 = vertices[nonuniformEXT(objId)].v[ind.z];

    const vec3 barycentrics = vec3(1.0 - attribs.x - attribs.y, attribs.x, attribs.y);

    // Computing the normal position
    vec3 normal = v0.nrm * barycentrics.x + v1.nrm * barycentrics.y + v2.nrm * barycentrics.z;

    // Transforming the normal to world space
    normal = normalize(vec3(scnDesc.i[gl_InstanceCustomIndexEXT].transfoIT * vec4(normal, 0.0)));

    // vec3 worldPos = gl_WorldRayOriginEXT + gl_WorldRayDirectionEXT * gl_HitTEXT;
    
    // Computing the world position
    vec3 worldPos = v0.pos * barycentrics.x + v1.pos * barycentrics.y + v2.pos * barycentrics.z;

    // Transforming the position to world space
    vec3 worldPosNorm = normalize(vec3(scnDesc.i[gl_InstanceCustomIndexEXT].transfo * vec4(worldPos, 1.0)));


    // Vector towards the light
    vec3 L;
    float lightIntensity = pushC.lightIntensity;
    float lightDistance = 100000.0;

    // Point light
    if (pushC.lightType == 0) {
        vec3 lDir = pushC.lightPosition - worldPosNorm;
        lightDistance = length(lDir);
        lightIntensity = pushC.lightIntensity / (lightDistance * lightDistance);
        L = normalize(lDir);
    }
    // Directional light
    else {
        L = normalize(pushC.lightPosition);
    }

    // Material of the object
    int matIdx = matIndex[nonuniformEXT(objId)].i[gl_PrimitiveID];
    WaveFrontMaterial mat = materials[nonuniformEXT(objId)].m[matIdx];

    vec3 reflect_dir = reflect(gl_WorldRayDirectionEXT, normal);
    vec3 skyColor = texture(skyboxSampler, reflect_dir).rgb;

    // Diffuse
    vec3 diffuse = computeDiffuse(mat, L, normal);
    if (mat.textureId >= 0) {
        uint txtId = mat.textureId + scnDesc.i[gl_InstanceCustomIndexEXT].txtOffset;
        vec2 texCoord = v0.texCoord * barycentrics.x + v1.texCoord * barycentrics.y + v2.texCoord * barycentrics.z;
        diffuse *= texture(textureSamplers[nonuniformEXT(txtId)], texCoord).xyz;
        diffuse *= skyColor;
        // diffuse = 0.7 * diffuse + 0.3 * skyColor;
    }


    // Shadow
    vec3 specular = vec3(0.0);
    float attenuation = 1.0;

    if (dot(normal, L) > 0) {
        uint rayFlags = gl_RayFlagsOpaqueEXT | gl_RayFlagsSkipClosestHitShaderEXT | gl_RayFlagsTerminateOnFirstHitEXT;
        float tMin = 0.001;
        float tMax = lightDistance;
        vec3 origin  = worldPos;
        vec3 rayDir = L;

        isShadowed = true;


        traceRayEXT(
            topLevelAS,     // acceleration structure
            rayFlags,       // rayFlags
            0xFF,           // cullMask
            0,              // SBT record offset
            0,              // SBT record stride
            1,              // Miss index
            origin,         // ray origin
            tMin,           // ray min
            rayDir,          // ray direction
            tMax,           // ray max
            1               // payload (location = 1)
        );

        if (isShadowed) {
            attenuation = 0.3;
        } else {
            // Specular
            specular = computeSpecular(mat, gl_WorldRayDirectionEXT, L, normal) * vec3(0.9, 0.85, 0.3);
            // specular = vec3(0);
        }
    }

    prd.hitValue = vec3(lightIntensity * attenuation * (diffuse + specular));
}
