#ifndef SPHERE_H
#define SPHERE_H

#include <vulkan/vulkan.hpp>

#define NVVK_ALLOC_DEDICATED

#include <nvmath/nvmath.h>
#include <nvvk/allocator_vk.hpp>
#include <nvvk/raytraceKHR_vk.hpp>
#include <vector>

struct Sphere {
    nvmath::vec3f center;
    float radius;
};

struct AABB {
    nvmath::vec3f min;
    nvmath::vec3f max;
};

class SphereHandler {
private:
    std::vector<Sphere> _spheres;
    nvvk::Buffer _spheres_buffer;
    nvvk::Buffer _spheres_aabb_buffer;


public:
    SphereHandler(nvvk::CommandPool& cmdpool, nvvk::Allocator& alloc);

    nvvk::RaytracingBuilderKHR::BlasInput toVkGeometryKHR(vk::Device& device);
    nvvk::Buffer& getSpheresBuffer();
    nvvk::Buffer& getSpheresAABBBuffer();

    void destroy(nvvk::Allocator& alloc);
};


#endif