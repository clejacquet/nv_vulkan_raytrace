#include "sphere.hpp"
#include "../material/material_obj.hpp"
#include <algorithm>
#include <random>

SphereHandler::SphereHandler(nvvk::CommandPool& cmdpool, nvvk::Allocator& alloc)
{
    std::random_device                    rd{};
    std::mt19937                          gen{rd()};
    std::normal_distribution<float>       xzd{0.f, 5.f};
    std::normal_distribution<float>       yd{6.f, 3.f};
    std::uniform_real_distribution<float> radd{0.05f, 0.2f};

    int nb_spheres = 100;
    // All spheres
    Sphere s;
    _spheres.resize(nb_spheres);

    for (uint32_t i = 0; i < nb_spheres; i++)
    {
        s.center     = nvmath::vec3f(xzd(gen), yd(gen), xzd(gen));
        s.radius     = radd(gen);
        _spheres[i] = std::move(s);
    }
    // // All spheres
    // _spheres = {
    //     Sphere { nvmath::vec3f( 0.0f, -100.5f, -1.0f), 100.0f },
    //     Sphere { nvmath::vec3f( 0.0f, 0.0f, -1.0f), 0.25f },
    //     Sphere { nvmath::vec3f(-1.0f, 0.0f, -1.0f), 0.15f },
    //     Sphere { nvmath::vec3f( 1.0f, 0.0f, -1.0f), 0.15f }
    // };

    // Axis aligned bounding box of each sphere
    std::vector<AABB> aabbs;
    aabbs.reserve(_spheres.size());
    for(const auto& s : _spheres)
    {
        AABB aabb;
        aabb.min = s.center - nvmath::vec3f(s.radius);
        aabb.max = s.center + nvmath::vec3f(s.radius);
        aabbs.emplace_back(aabb);
    }

    // Creating all buffers
    using vkBU = vk::BufferUsageFlagBits;
    auto cmdBuf = cmdpool.createCommandBuffer();
    _spheres_buffer           = alloc.createBuffer(cmdBuf, _spheres, vkBU::eStorageBuffer);
    _spheres_aabb_buffer      = alloc.createBuffer(cmdBuf, aabbs, vkBU::eShaderDeviceAddress);
    cmdpool.submitAndWait(cmdBuf);
}

nvvk::RaytracingBuilderKHR::BlasInput SphereHandler::toVkGeometryKHR(vk::Device& device) {
    vk::DeviceAddress dataAddress = device.getBufferAddress({_spheres_aabb_buffer.buffer});

    vk::AccelerationStructureGeometryAabbsDataKHR aabbs;
    aabbs.setData(dataAddress);
    aabbs.setStride(sizeof(AABB));

    // Setting up the build info of the acceleration (C version, c++ gives wrong type)
    vk::AccelerationStructureGeometryKHR asGeom(vk::GeometryTypeKHR::eAabbs, aabbs,
                                                vk::GeometryFlagBitsKHR::eOpaque);
    //asGeom.geometryType   = vk::GeometryTypeKHR::eAabbs;
    //asGeom.flags          = vk::GeometryFlagBitsKHR::eOpaque;
    //asGeom.geometry.aabbs = aabbs;


    vk::AccelerationStructureBuildRangeInfoKHR offset;
    offset.setFirstVertex(0);
    offset.setPrimitiveCount((uint32_t)_spheres.size());  // Nb aabb
    offset.setPrimitiveOffset(0);
    offset.setTransformOffset(0);

    nvvk::RaytracingBuilderKHR::BlasInput input;
    input.asGeometry.emplace_back(asGeom);
    input.asBuildOffsetInfo.emplace_back(offset);
    return input;
}

nvvk::Buffer& SphereHandler::getSpheresBuffer()
{
    return _spheres_buffer;
}

nvvk::Buffer& SphereHandler::getSpheresAABBBuffer()
{
    return _spheres_aabb_buffer;
}


void SphereHandler::destroy(nvvk::Allocator& alloc) {
    alloc.destroy(_spheres_buffer);
    alloc.destroy(_spheres_aabb_buffer);
}
