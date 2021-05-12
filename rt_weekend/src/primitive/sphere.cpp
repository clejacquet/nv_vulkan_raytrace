#include "sphere.hpp"
#include "../material/material_obj.hpp"
#include <algorithm>

SphereHandler::SphereHandler(nvvk::CommandPool& cmdpool, nvvk::Allocator& alloc)
{
    // All spheres
    Sphere s;
    _spheres.resize(1);
    for(uint32_t i = 0; i < 1; i++)
    {
        s.center     = nvmath::vec3f(0.0f, 0.0f, -1.0f);
        s.radius     = 0.5f;
        _spheres[i] = std::move(s);
    }

    // Axis aligned bounding box of each sphere
    std::vector<AABB> aabbs;
    aabbs.reserve(1);
    for(const auto& s : _spheres)
    {
        AABB aabb;
        aabb.min = s.center - nvmath::vec3f(s.radius);
        aabb.max = s.center + nvmath::vec3f(s.radius);
        aabbs.emplace_back(aabb);
    }

    // Creating all buffers
    using vkBU = vk::BufferUsageFlagBits;
    auto              cmdBuf = cmdpool.createCommandBuffer();
    _spheres_buffer          = alloc.createBuffer(cmdBuf, _spheres, vkBU::eStorageBuffer);
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
