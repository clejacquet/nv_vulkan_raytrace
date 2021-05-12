#include "application_impl.hpp"
#include "common/obj_loader.h"
#include "nvvk/shaders_vk.hpp"
#include "nvh/fileoperations.hpp"
#include "nvh/alignment.hpp"


// -----------------------
// Impl Ray Tracing Methods
// -----------------------

void Application::Impl::initRayTracing()
{
    auto properties = 
        m_physicalDevice.getProperties2
                <
                    vk::PhysicalDeviceProperties2,
                    vk::PhysicalDeviceRayTracingPipelinePropertiesKHR
                >();

    m_rtProperties = properties.get<vk::PhysicalDeviceRayTracingPipelinePropertiesKHR>();

    // Spec only guarantees 1 level of "recursion". Check for that sad possibility here.
    if (m_rtProperties.maxRayRecursionDepth <= 1) {
        throw std::runtime_error("Device fails to support ray recursion (m_rtProperties.maxRayRecursionDepth <= 1)");
    }

    m_rtBuilder.setup(m_device, &m_alloc, m_graphicsQueueIndex);
}

nvvk::RaytracingBuilderKHR::BlasInput Application::Impl::objectToVkGeometryKHR(const ObjModel& model)
{
    // BLAS builder requires raw device addresses.
    vk::DeviceAddress vertexAddress = m_device.getBufferAddress({ model.vertexBuffer.buffer });
    vk::DeviceAddress indexAddress = m_device.getBufferAddress({ model.indexBuffer.buffer });

    auto maxPrimitiveCount = model.nbIndices / 3;

    // Describe buffer as array of VertexObj.
    vk::AccelerationStructureGeometryTrianglesDataKHR triangles;
    triangles.setVertexFormat(vk::Format::eR32G32B32Sfloat); // vec3 vertex position data.
    triangles.setVertexData(vertexAddress);
    triangles.setVertexStride(sizeof(VertexObj));

    // Describe index data (32-bit unsigned int)
    triangles.setIndexType(vk::IndexType::eUint32);
    triangles.setIndexData(indexAddress);

    // Indicate identity transform by setting transformData to null device pointer.
    triangles.setTransformData({});
    triangles.setMaxVertex(model.nbVertices);

    // Identify the above data as containing opaque triangles.
    vk::AccelerationStructureGeometryKHR asGeom;
    asGeom.setGeometryType(vk::GeometryTypeKHR::eTriangles);
    asGeom.setFlags(vk::GeometryFlagBitsKHR::eOpaque);
    asGeom.geometry.setTriangles(triangles);

    // The entire array will be used to build the BLAS.
    vk::AccelerationStructureBuildRangeInfoKHR offset;
    offset.setFirstVertex(0);
    offset.setPrimitiveCount(maxPrimitiveCount);
    offset.setPrimitiveOffset(0);
    offset.setTransformOffset(0);

    // Our blas is made from only one geometry, but could be made of many geometries
    nvvk::RaytracingBuilderKHR::BlasInput input;
    input.asGeometry.emplace_back(asGeom);
    input.asBuildOffsetInfo.emplace_back(offset);

    return input;
}

void Application::Impl::createBottomLevelAS()
{
    // BLAS - Storing each primitive in a geometry
    std::vector<nvvk::RaytracingBuilderKHR::BlasInput> allBlas;
    // allBlas.reserve(m_objModel.size());
    // for(const auto& obj : m_objModel)
    // {
    //     auto blas = objectToVkGeometryKHR(obj);

    //     // We could add more geometry in each BLAS, but we add only one for now
    //     allBlas.emplace_back(blas);
    // }

    // Spheres
    {
        auto blas = m_sphereHandler->toVkGeometryKHR(m_device);
        allBlas.emplace_back(blas);
    }

    m_rtBuilder.buildBlas(allBlas, vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace);
}

void Application::Impl::createTopLevelAS()
{
    std::vector<nvvk::RaytracingBuilderKHR::Instance> tlas;
    // tlas.reserve(m_objInstance.size());

    // for (uint32_t i = 0; i < static_cast<uint32_t>(m_objInstance.size()); ++i)
    // {
    //     nvvk::RaytracingBuilderKHR::Instance ray_inst;
    //     ray_inst.transform        = m_objInstance[i].transform; // Position of the instance
    //     ray_inst.instanceCustomId = i;                          // gl_InstanceCustomIndexEXT
    //     ray_inst.blasId           = m_objInstance[i].objIndex;
    //     ray_inst.hitGroupId       = 0;                          // We will use the same hit group for all objects
    //     ray_inst.flags            = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
    //     tlas.emplace_back(ray_inst);
    // }

    // Add the sphere BLAS
    {
        nvvk::RaytracingBuilderKHR::Instance ray_inst;
        // ray_inst.transform        = m_objInstance[0].transform;
        ray_inst.transform        = nvmath::mat4f().identity();
        ray_inst.instanceCustomId = static_cast<uint32_t>(tlas.size());
        // ray_inst.blasId           = static_cast<uint32_t>(m_objModel.size());
        ray_inst.blasId           = 0;
        ray_inst.hitGroupId       = 0;
        ray_inst.flags            = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
        tlas.emplace_back(ray_inst);
    }

    m_rtBuilder.buildTlas(tlas, vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace);
}

void Application::Impl::createRtDescriptorSet()
{
    using vkDT = vk::DescriptorType;
    using vkSS = vk::ShaderStageFlagBits;
    using vkDSLB = vk::DescriptorSetLayoutBinding;

    // TLAS
    m_rtDescSetLayoutBind.addBinding(vkDSLB(0, vkDT::eAccelerationStructureKHR, 1, vkSS::eRaygenKHR | vkSS::eClosestHitKHR)); 

    // Output Image
    m_rtDescSetLayoutBind.addBinding(vkDSLB(1, vkDT::eStorageImage, 1, vkSS::eRaygenKHR));

    m_rtDescPool        = m_rtDescSetLayoutBind.createPool(m_device);
    m_rtDescSetLayout   = m_rtDescSetLayoutBind.createLayout(m_device);
    m_rtDescSet         = m_device.allocateDescriptorSets({ m_rtDescPool, 1, &m_rtDescSetLayout })[0];

    vk::AccelerationStructureKHR tlas = m_rtBuilder.getAccelerationStructure();
    vk::WriteDescriptorSetAccelerationStructureKHR descASInfo;
    descASInfo.setAccelerationStructureCount(1);
    descASInfo.setPAccelerationStructures(&tlas);

    vk::DescriptorImageInfo imageInfo {
        {}, m_offscreenColor.descriptor.imageView, vk::ImageLayout::eGeneral
    };

    std::vector<vk::WriteDescriptorSet> writes;
    writes.emplace_back(m_rtDescSetLayoutBind.makeWrite(m_rtDescSet, 0, &descASInfo));
    writes.emplace_back(m_rtDescSetLayoutBind.makeWrite(m_rtDescSet, 1, &imageInfo));
    
    m_device.updateDescriptorSets(static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
}

void Application::Impl::updateRtDescriptorSet()
{
    using vkDT = vk::DescriptorType;

    // (1) Output buffer
    vk::DescriptorImageInfo imageInfo {{}, m_offscreenColor.descriptor.imageView, vk::ImageLayout::eGeneral};
    vk::WriteDescriptorSet wds {m_rtDescSet, 1, 0, 1, vkDT::eStorageImage, &imageInfo};
    m_device.updateDescriptorSets(wds, nullptr);
}

void Application::Impl::createRtPipeline()
{
    vk::ShaderModule raygenSM = nvvk::createShaderModule(
        m_device, nvh::loadFile("spv/raytrace.rgen.spv", true, _default_search_paths, true));
    vk::ShaderModule missSM = nvvk::createShaderModule(
        m_device, nvh::loadFile("spv/raytrace.rmiss.spv", true, _default_search_paths, true));

    // The second miss shader is invoked when a shadow ray misses the geometry. It
    // simply indicates that no occlusion has been found
    // vk::ShaderModule shadowmissSM = nvvk::createShaderModule(
    //     m_device, nvh::loadFile("spv/raytraceShadow.rmiss.spv", true, _default_search_paths, true));


    std::vector<vk::PipelineShaderStageCreateInfo> stages;

    // Raygen
    vk::RayTracingShaderGroupCreateInfoKHR rg{vk::RayTracingShaderGroupTypeKHR::eGeneral,
                                                VK_SHADER_UNUSED_KHR, VK_SHADER_UNUSED_KHR,
                                                VK_SHADER_UNUSED_KHR, VK_SHADER_UNUSED_KHR};
    rg.setGeneralShader(static_cast<uint32_t>(stages.size()));
    stages.push_back({{}, vk::ShaderStageFlagBits::eRaygenKHR, raygenSM, "main"});
    m_rtShaderGroups.push_back(rg);
    // Miss
    vk::RayTracingShaderGroupCreateInfoKHR mg{vk::RayTracingShaderGroupTypeKHR::eGeneral,
                                                VK_SHADER_UNUSED_KHR, VK_SHADER_UNUSED_KHR,
                                                VK_SHADER_UNUSED_KHR, VK_SHADER_UNUSED_KHR};
    mg.setGeneralShader(static_cast<uint32_t>(stages.size()));
    stages.push_back({{}, vk::ShaderStageFlagBits::eMissKHR, missSM, "main"});
    m_rtShaderGroups.push_back(mg);
    // // Shadow Miss
    // mg.setGeneralShader(static_cast<uint32_t>(stages.size()));
    // stages.push_back({{}, vk::ShaderStageFlagBits::eMissKHR, shadowmissSM, "main"});
    // m_rtShaderGroups.push_back(mg);

    // Hit Group0 - Closest Hit
    // vk::ShaderModule chitSM = nvvk::createShaderModule(
    //     m_device, nvh::loadFile("spv/raytrace.rchit.spv", true, _default_search_paths, true));

    // {
    //     vk::RayTracingShaderGroupCreateInfoKHR hg{vk::RayTracingShaderGroupTypeKHR::eTrianglesHitGroup,
    //                                             VK_SHADER_UNUSED_KHR, VK_SHADER_UNUSED_KHR,
    //                                             VK_SHADER_UNUSED_KHR, VK_SHADER_UNUSED_KHR};
    //     hg.setClosestHitShader(static_cast<uint32_t>(stages.size()));
    //     stages.push_back({{}, vk::ShaderStageFlagBits::eClosestHitKHR, chitSM, "main"});
    //     m_rtShaderGroups.push_back(hg);
    // }

    // Hit Group1 - Closest Hit + Intersection (procedural)
    vk::ShaderModule chitSM = nvvk::createShaderModule(
        m_device, nvh::loadFile("spv/raytrace.rchit.spv", true, _default_search_paths, true));
    vk::ShaderModule rintSM = nvvk::createShaderModule(
        m_device, nvh::loadFile("spv/raytrace.rint.spv", true, _default_search_paths, true));
    {
        vk::RayTracingShaderGroupCreateInfoKHR hg{vk::RayTracingShaderGroupTypeKHR::eProceduralHitGroup,
                                                VK_SHADER_UNUSED_KHR, VK_SHADER_UNUSED_KHR,
                                                VK_SHADER_UNUSED_KHR, VK_SHADER_UNUSED_KHR};
        hg.setClosestHitShader(static_cast<uint32_t>(stages.size()));
        stages.push_back({{}, vk::ShaderStageFlagBits::eClosestHitKHR, chitSM, "main"});
        hg.setIntersectionShader(static_cast<uint32_t>(stages.size()));
        stages.push_back({{}, vk::ShaderStageFlagBits::eIntersectionKHR, rintSM, "main"});
        m_rtShaderGroups.push_back(hg);
    }



    vk::PipelineLayoutCreateInfo pipelineLayoutCreateInfo;

    // Push constant: we want to be able to update constants used by the shaders
    vk::PushConstantRange pushConstant{vk::ShaderStageFlagBits::eRaygenKHR
                                            | vk::ShaderStageFlagBits::eClosestHitKHR
                                            | vk::ShaderStageFlagBits::eMissKHR,
                                        0, sizeof(RtPushConstant)};
    pipelineLayoutCreateInfo.setPushConstantRangeCount(1);
    pipelineLayoutCreateInfo.setPPushConstantRanges(&pushConstant);

    // Descriptor sets: one specific to ray tracing, and one shared with the rasterization pipeline
    std::vector<vk::DescriptorSetLayout> rtDescSetLayouts = {m_rtDescSetLayout, m_descSetLayout};
    pipelineLayoutCreateInfo.setSetLayoutCount(static_cast<uint32_t>(rtDescSetLayouts.size()));
    pipelineLayoutCreateInfo.setPSetLayouts(rtDescSetLayouts.data());

    m_rtPipelineLayout = m_device.createPipelineLayout(pipelineLayoutCreateInfo);

    // Assemble the shader stages and recursion depth info into the ray tracing pipeline
    vk::RayTracingPipelineCreateInfoKHR rayPipelineInfo;
    rayPipelineInfo.setStageCount(static_cast<uint32_t>(stages.size()));  // Stages are shaders
    rayPipelineInfo.setPStages(stages.data());

    rayPipelineInfo.setGroupCount(static_cast<uint32_t>(
        m_rtShaderGroups.size()));  // 1-raygen, n-miss, n-(hit[+anyhit+intersect])
    rayPipelineInfo.setPGroups(m_rtShaderGroups.data());

    rayPipelineInfo.setMaxPipelineRayRecursionDepth(2);  // Ray depth
    rayPipelineInfo.setLayout(m_rtPipelineLayout);
    m_rtPipeline = static_cast<const vk::Pipeline&>(
        m_device.createRayTracingPipelineKHR({}, {}, rayPipelineInfo));

    m_device.destroy(raygenSM);
    m_device.destroy(missSM);
    // m_device.destroy(shadowmissSM);
    // m_device.destroy(chitSM);
    m_device.destroy(chitSM);
    m_device.destroy(rintSM);
}

void Application::Impl::createRtShaderBindingTable()
{
    auto groupCount =
        static_cast<uint32_t>(m_rtShaderGroups.size());  // shaders: raygen, 2 miss, 2 chit, rint
    uint32_t groupHandleSize = m_rtProperties.shaderGroupHandleSize;  // Size of a program identifier
    uint32_t groupSizeAligned =
        nvh::align_up(groupHandleSize, m_rtProperties.shaderGroupBaseAlignment);

    // Fetch all the shader handles used in the pipeline, so that they can be written in the SBT
    uint32_t sbtSize = groupCount * groupSizeAligned;

    std::vector<uint8_t> shaderHandleStorage(sbtSize);
    auto result = m_device.getRayTracingShaderGroupHandlesKHR(m_rtPipeline, 0, groupCount, sbtSize,
                                                                shaderHandleStorage.data());
    assert(result == vk::Result::eSuccess);

    // Write the handles in the SBT
    m_rtSBTBuffer = m_alloc.createBuffer(
        sbtSize,
        vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eShaderDeviceAddressKHR
            | vk::BufferUsageFlagBits::eShaderBindingTableKHR,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
    
    // Write the handles in the SBT
    void* mapped = m_alloc.map(m_rtSBTBuffer);
    auto* pData  = reinterpret_cast<uint8_t*>(mapped);
    for(uint32_t g = 0; g < groupCount; g++)
    {
        memcpy(pData, shaderHandleStorage.data() + g * groupHandleSize, groupHandleSize);  // raygen
        pData += groupSizeAligned;
    }
    m_alloc.unmap(m_rtSBTBuffer);

    m_alloc.finalizeAndReleaseStaging();
}

void Application::Impl::rayTrace(const vk::CommandBuffer& cmdBuf, const nvmath::vec4f& clearColor)
{
    // Initializing push constant values
    m_rtPushConstants.clearColor     = clearColor;
    m_rtPushConstants.lightPosition  = m_pushConstant.lightPosition;
    m_rtPushConstants.lightIntensity = m_pushConstant.lightIntensity;
    m_rtPushConstants.lightType      = m_pushConstant.lightType;

    cmdBuf.bindPipeline(vk::PipelineBindPoint::eRayTracingKHR, m_rtPipeline);
    cmdBuf.bindDescriptorSets(vk::PipelineBindPoint::eRayTracingKHR, m_rtPipelineLayout, 0,
                                {m_rtDescSet, m_descSet}, {});
    cmdBuf.pushConstants<RtPushConstant>(m_rtPipelineLayout,
                                        vk::ShaderStageFlagBits::eRaygenKHR
                                            | vk::ShaderStageFlagBits::eClosestHitKHR
                                            | vk::ShaderStageFlagBits::eMissKHR,
                                        0, m_rtPushConstants);

    // Size of a program identifier
    uint32_t groupSize =
        nvh::align_up(m_rtProperties.shaderGroupHandleSize, m_rtProperties.shaderGroupBaseAlignment);
    uint32_t          groupStride = groupSize;
    vk::DeviceAddress sbtAddress  = m_device.getBufferAddress({m_rtSBTBuffer.buffer});

    using Stride = vk::StridedDeviceAddressRegionKHR;
    std::array<Stride, 4> strideAddresses{
        Stride{sbtAddress + 0u * groupSize, groupStride, groupSize * 1},  // raygen
        Stride{sbtAddress + 1u * groupSize, groupStride, groupSize * 1},  // miss
        Stride{sbtAddress + 2u * groupSize, groupStride, groupSize * 1},  // hit
        Stride{0u, 0u, 0u}};                                              // callable

    cmdBuf.traceRaysKHR(&strideAddresses[0], &strideAddresses[1], &strideAddresses[2],
                        &strideAddresses[3],              //
                        m_size.width, m_size.height, 1);  //

}
