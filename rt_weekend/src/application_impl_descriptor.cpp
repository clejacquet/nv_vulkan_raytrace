#include "application_impl.hpp"

#define STB_IMAGE_IMPLEMENTATION
#include "fileformats/stb_image.h"
#include "common/obj_loader.h"

#include "nvvk/pipeline_vk.hpp"
#include "nvh/fileoperations.hpp"


// Holding the camera matrices
struct CameraMatrices
{
  nvmath::mat4f view;
  nvmath::mat4f proj;
  nvmath::mat4f viewInverse;
  // #VKRay
  nvmath::mat4f projInverse;
};

// -----------------------
// Impl Descriptor Methods
// -----------------------

void Application::Impl::createDescriptorSetLayout()
{
    using vkDS     = vk::DescriptorSetLayoutBinding;
    using vkDT     = vk::DescriptorType;
    using vkSS     = vk::ShaderStageFlagBits;
    uint32_t nbTxt = std::max(1U, static_cast<uint32_t>(m_textures.size()));
    uint32_t nbObj = std::max(1U, static_cast<uint32_t>(m_objModel.size()));

    // Camera matrices (binding = 0)
    m_descSetLayoutBind.addBinding(
        vkDS(0, vkDT::eUniformBuffer, 1, vkSS::eVertex | vkSS::eRaygenKHR));
    // Materials (binding = 1)
    m_descSetLayoutBind.addBinding(
        vkDS(1, vkDT::eStorageBuffer, nbObj, vkSS::eVertex | vkSS::eFragment | vkSS::eClosestHitKHR));
    // Scene description (binding = 2)
    m_descSetLayoutBind.addBinding(  //
        vkDS(2, vkDT::eStorageBuffer, 1, vkSS::eVertex | vkSS::eFragment | vkSS::eClosestHitKHR));
    // Textures (binding = 3)
    m_descSetLayoutBind.addBinding(
        vkDS(3, vkDT::eCombinedImageSampler, nbTxt, vkSS::eFragment | vkSS::eClosestHitKHR));
    // Materials (binding = 4)
    m_descSetLayoutBind.addBinding(
        vkDS(4, vkDT::eStorageBuffer, nbObj, vkSS::eFragment | vkSS::eClosestHitKHR));
    // Storing vertices (binding = 5)
    m_descSetLayoutBind.addBinding(  //
        vkDS(5, vkDT::eStorageBuffer, nbObj, vkSS::eClosestHitKHR));
    // Storing indices (binding = 6)
    m_descSetLayoutBind.addBinding(  //
        vkDS(6, vkDT::eStorageBuffer, nbObj, vkSS::eClosestHitKHR));
    // Storing spheres (binding = 7)
    m_descSetLayoutBind.addBinding(  //
        vkDS(7, vkDT::eStorageBuffer, 1, vkSS::eClosestHitKHR | vkSS::eIntersectionKHR));
    // Storing Skybox texture (binding = 8)
    m_descSetLayoutBind.addBinding(  //
        vkDS(8, vkDT::eCombinedImageSampler, 1, vkSS::eMissKHR | vkSS::eClosestHitKHR));

    m_descSetLayout = m_descSetLayoutBind.createLayout(m_device);
    m_descPool      = m_descSetLayoutBind.createPool(m_device, 1);
    m_descSet       = nvvk::allocateDescriptorSet(m_device, m_descPool, m_descSetLayout);
}

void Application::Impl::updateDescriptorSet()
{
    std::vector<vk::WriteDescriptorSet> writes;

    // Camera matrices and scene description
    vk::DescriptorBufferInfo dbiUnif{m_cameraMat.buffer, 0, VK_WHOLE_SIZE};
    writes.emplace_back(m_descSetLayoutBind.makeWrite(m_descSet, 0, &dbiUnif));

    vk::DescriptorBufferInfo dbiSceneDesc{m_sceneDesc.buffer, 0, VK_WHOLE_SIZE};
    writes.emplace_back(m_descSetLayoutBind.makeWrite(m_descSet, 2, &dbiSceneDesc));

    // All material buffers, 1 buffer per OBJ
    std::vector<vk::DescriptorBufferInfo> dbiMat;
    std::vector<vk::DescriptorBufferInfo> dbiMatIdx;
    std::vector<vk::DescriptorBufferInfo> dbiVert;
    std::vector<vk::DescriptorBufferInfo> dbiIdx;
    for(auto& obj : m_objModel)
    {
        dbiMat.emplace_back(obj.matColorBuffer.buffer, 0, VK_WHOLE_SIZE);
        dbiMatIdx.emplace_back(obj.matIndexBuffer.buffer, 0, VK_WHOLE_SIZE);
        dbiVert.emplace_back(obj.vertexBuffer.buffer, 0, VK_WHOLE_SIZE);
        dbiIdx.emplace_back(obj.indexBuffer.buffer, 0, VK_WHOLE_SIZE);
    }

    writes.emplace_back(m_descSetLayoutBind.makeWriteArray(m_descSet, 1, dbiMat.data()));
    writes.emplace_back(m_descSetLayoutBind.makeWriteArray(m_descSet, 4, dbiMatIdx.data()));
    writes.emplace_back(m_descSetLayoutBind.makeWriteArray(m_descSet, 5, dbiVert.data()));
    writes.emplace_back(m_descSetLayoutBind.makeWriteArray(m_descSet, 6, dbiIdx.data()));

    // All texture samplers
    std::vector<vk::DescriptorImageInfo> diit;
    for(auto& texture : m_textures)
    {
        diit.emplace_back(texture.descriptor);
    }
    writes.emplace_back(m_descSetLayoutBind.makeWriteArray(m_descSet, 3, diit.data()));

    // Skybox texture
    writes.emplace_back(m_descSetLayoutBind.makeWriteArray(m_descSet, 8, &m_skybox_txt->descriptor)) ;

    // Spheres
    vk::DescriptorBufferInfo dbiSpheres{m_sphereHandler->getSpheresBuffer().buffer, 0, VK_WHOLE_SIZE};
    writes.emplace_back(m_descSetLayoutBind.makeWrite(m_descSet, 7, &dbiSpheres));

    // Writing the information
    m_device.updateDescriptorSets(static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
}

void Application::Impl::createUniformBuffer()
{
    using vkBU = vk::BufferUsageFlagBits;
    using vkMP = vk::MemoryPropertyFlagBits;

    m_cameraMat = m_alloc.createBuffer(sizeof(CameraMatrices),
                                        vkBU::eUniformBuffer | vkBU::eTransferDst, vkMP::eDeviceLocal);
}

void Application::Impl::updateUniformBuffer(const vk::CommandBuffer& cmdBuf)
{
    // Prepare new UBO contents on host.
    const float    aspectRatio = m_size.width / static_cast<float>(m_size.height);
    CameraMatrices hostUBO     = {};
    hostUBO.view               = CameraManip.getMatrix();
    hostUBO.proj = nvmath::perspectiveVK(CameraManip.getFov(), aspectRatio, 0.1f, 1000.0f);
    // hostUBO.proj[1][1] *= -1;  // Inverting Y for Vulkan (not needed with perspectiveVK).
    hostUBO.viewInverse = nvmath::invert(hostUBO.view);
    // #VKRay
    hostUBO.projInverse = nvmath::invert(hostUBO.proj);

    // UBO on the device, and what stages access it.
    vk::Buffer deviceUBO = m_cameraMat.buffer;
    auto       uboUsageStages =
        vk::PipelineStageFlagBits::eVertexShader | vk::PipelineStageFlagBits::eRayTracingShaderKHR;

    // Ensure that the modified UBO is not visible to previous frames.
    vk::BufferMemoryBarrier beforeBarrier;
    beforeBarrier.setSrcAccessMask(vk::AccessFlagBits::eShaderRead);
    beforeBarrier.setDstAccessMask(vk::AccessFlagBits::eTransferWrite);
    beforeBarrier.setBuffer(deviceUBO);
    beforeBarrier.setOffset(0);
    beforeBarrier.setSize(sizeof hostUBO);
    cmdBuf.pipelineBarrier(uboUsageStages, vk::PipelineStageFlagBits::eTransfer,
                            vk::DependencyFlagBits::eDeviceGroup, {}, {beforeBarrier}, {});

    // Schedule the host-to-device upload. (hostUBO is copied into the cmd
    // buffer so it is okay to deallocate when the function returns).
    cmdBuf.updateBuffer<CameraMatrices>(m_cameraMat.buffer, 0, hostUBO);

    // Making sure the updated UBO will be visible.
    vk::BufferMemoryBarrier afterBarrier;
    afterBarrier.setSrcAccessMask(vk::AccessFlagBits::eTransferWrite);
    afterBarrier.setDstAccessMask(vk::AccessFlagBits::eShaderRead);
    afterBarrier.setBuffer(deviceUBO);
    afterBarrier.setOffset(0);
    afterBarrier.setSize(sizeof hostUBO);
    cmdBuf.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, uboUsageStages,
                            vk::DependencyFlagBits::eDeviceGroup, {}, {afterBarrier}, {});
}

void Application::Impl::createSceneDescriptionBuffer()
{
    // If no object in the scene, add a dummy object so that buffer are valid for the shaders
    if (m_objInstance.empty()) {
        nvvk::CommandPool cmdBufGet(m_device, m_graphicsQueueIndex);
        vk::CommandBuffer cmdBuf = cmdBufGet.createCommandBuffer();

        // OBJ Instance
        using vkBU = vk::BufferUsageFlagBits;

        ObjInstance instance;
        instance.objIndex    = static_cast<uint32_t>(m_objModel.size());
        instance.transform   = nvmath::mat4f().identity();
        instance.transformIT = nvmath::mat4f().identity();
        instance.txtOffset   = static_cast<uint32_t>(m_textures.size());
        m_objInstance.emplace_back(instance);
        
        // OBJ Model
        ObjModel model;
        model.nbIndices  = static_cast<uint32_t>(0);
        model.nbVertices = static_cast<uint32_t>(0);

        model.vertexBuffer =
            m_alloc.createBuffer(cmdBuf, std::vector<VertexObj> { VertexObj {} },
                            vkBU::eVertexBuffer | vkBU::eStorageBuffer | vkBU::eShaderDeviceAddress
                                    | vkBU::eAccelerationStructureBuildInputReadOnlyKHR);
        model.indexBuffer =
            m_alloc.createBuffer(cmdBuf, std::vector<VkDescriptorBindingFlags> { VkDescriptorBindingFlags {} },
                            vkBU::eIndexBuffer | vkBU::eStorageBuffer | vkBU::eShaderDeviceAddress
                                    | vkBU::eAccelerationStructureBuildInputReadOnlyKHR);

        model.matColorBuffer = m_alloc.createBuffer(cmdBuf, std::vector<MaterialObj> { MaterialObj {} }, vkBU::eStorageBuffer);
        model.matIndexBuffer = m_alloc.createBuffer(cmdBuf, std::vector<int> { int {} }, vkBU::eStorageBuffer);
        m_objModel.emplace_back(model);

        // Texture
        nvvk::Texture texture;
        vk::SamplerCreateInfo samplerCreateInfo{
                {}, vk::Filter::eLinear, vk::Filter::eLinear, vk::SamplerMipmapMode::eLinear};
        samplerCreateInfo.setMaxLod(FLT_MAX);
        vk::Format format = vk::Format::eR8G8B8A8Srgb;

        std::array<uint8_t, 4> color{255u, 255u, 255u, 255u};
        vk::DeviceSize         bufferSize      = sizeof(color);
        auto                   imgSize         = vk::Extent2D(1, 1);
        auto                   imageCreateInfo = nvvk::makeImage2DCreateInfo(imgSize, format);

        // Creating the dummy texture
        nvvk::Image image = m_alloc.createImage(cmdBuf, bufferSize, color.data(), imageCreateInfo);
        vk::ImageViewCreateInfo ivInfo = nvvk::makeImageViewCreateInfo(image.image, imageCreateInfo);
        texture                        = m_alloc.createTexture(image, ivInfo, samplerCreateInfo);

        // The image format must be in VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
        nvvk::cmdBarrierImageLayout(cmdBuf, texture.image, vk::ImageLayout::eUndefined,
                                    vk::ImageLayout::eShaderReadOnlyOptimal);
        m_textures.push_back(texture);
        
        cmdBufGet.submitAndWait(cmdBuf);
    }

    using vkBU = vk::BufferUsageFlagBits;
    nvvk::CommandPool cmdGen(m_device, m_graphicsQueueIndex);

    auto cmdBuf = cmdGen.createCommandBuffer();
    m_sceneDesc = m_alloc.createBuffer(cmdBuf, m_objInstance, vkBU::eStorageBuffer);
    cmdGen.submitAndWait(cmdBuf);
    m_alloc.finalizeAndReleaseStaging();
}

void Application::Impl::createTextureImages(const vk::CommandBuffer& cmdBuf, const std::vector<std::string>& textures)
{
    using vkIU = vk::ImageUsageFlagBits;

    vk::SamplerCreateInfo samplerCreateInfo{
        {}, vk::Filter::eLinear, vk::Filter::eLinear, vk::SamplerMipmapMode::eLinear};
    samplerCreateInfo.setMaxLod(FLT_MAX);
    vk::Format format = vk::Format::eR8G8B8A8Srgb;

    // If no textures are present, create a dummy one to accommodate the pipeline layout
    if(textures.empty() && m_textures.empty())
    {
        nvvk::Texture texture;

        std::array<uint8_t, 4> color{255u, 255u, 255u, 255u};
        vk::DeviceSize         bufferSize      = sizeof(color);
        auto                   imgSize         = vk::Extent2D(1, 1);
        auto                   imageCreateInfo = nvvk::makeImage2DCreateInfo(imgSize, format);

        // Creating the dummy texture
        nvvk::Image image = m_alloc.createImage(cmdBuf, bufferSize, color.data(), imageCreateInfo);
        vk::ImageViewCreateInfo ivInfo = nvvk::makeImageViewCreateInfo(image.image, imageCreateInfo);
        texture                        = m_alloc.createTexture(image, ivInfo, samplerCreateInfo);

        // The image format must be in VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
        nvvk::cmdBarrierImageLayout(cmdBuf, texture.image, vk::ImageLayout::eUndefined,
                                    vk::ImageLayout::eShaderReadOnlyOptimal);
        m_textures.push_back(texture);
    }
    else
    {
        // Uploading all images
        for(const auto& texture : textures)
        {
        std::stringstream o;
        int               texWidth, texHeight, texChannels;
        o << "media/textures/" << texture;
        std::string txtFile = nvh::findFile(o.str(), _default_search_paths, true);

        stbi_uc* stbi_pixels =
            stbi_load(txtFile.c_str(), &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);

        std::array<stbi_uc, 4> color{255u, 0u, 255u, 255u};

        stbi_uc* pixels = stbi_pixels;
        // Handle failure
        if(!stbi_pixels)
        {
            texWidth = texHeight = 1;
            texChannels          = 4;
            pixels               = reinterpret_cast<stbi_uc*>(color.data());
        }

        vk::DeviceSize bufferSize = static_cast<uint64_t>(texWidth) * texHeight * sizeof(uint8_t) * 4;
        auto           imgSize    = vk::Extent2D(texWidth, texHeight);
        auto imageCreateInfo = nvvk::makeImage2DCreateInfo(imgSize, format, vkIU::eSampled, true);

        {
            nvvk::ImageDedicated image =
                m_alloc.createImage(cmdBuf, bufferSize, pixels, imageCreateInfo);
            nvvk::cmdGenerateMipmaps(cmdBuf, image.image, format, imgSize, imageCreateInfo.mipLevels);
            vk::ImageViewCreateInfo ivInfo =
                nvvk::makeImageViewCreateInfo(image.image, imageCreateInfo);
            nvvk::Texture texture = m_alloc.createTexture(image, ivInfo, samplerCreateInfo);

            m_textures.push_back(texture);
        }

        stbi_image_free(stbi_pixels);
        }
    }
}


void Application::Impl::loadModel(const std::string& filename, nvmath::mat4f transform)
{
    using vkBU = vk::BufferUsageFlagBits;

    LOGI("Loading File:  %s \n", filename.c_str());
    ObjLoader loader;
    loader.loadModel(filename);

    // Converting from Srgb to linear
    for(auto& m : loader.m_materials)
    {
        m.ambient  = nvmath::pow(m.ambient, 2.2f);
        m.diffuse  = nvmath::pow(m.diffuse, 2.2f);
        m.specular = nvmath::pow(m.specular, 2.2f);
    }

    ObjInstance instance;
    instance.objIndex    = static_cast<uint32_t>(m_objModel.size());
    instance.transform   = transform;
    instance.transformIT = nvmath::transpose(nvmath::invert(transform));
    instance.txtOffset   = static_cast<uint32_t>(m_textures.size());

    ObjModel model;
    model.nbIndices  = static_cast<uint32_t>(loader.m_indices.size());
    model.nbVertices = static_cast<uint32_t>(loader.m_vertices.size());

    // Create the buffers on Device and copy vertices, indices and materials
    nvvk::CommandPool cmdBufGet(m_device, m_graphicsQueueIndex);
    vk::CommandBuffer cmdBuf = cmdBufGet.createCommandBuffer();
    model.vertexBuffer =
        m_alloc.createBuffer(cmdBuf, loader.m_vertices,
                            vkBU::eVertexBuffer | vkBU::eStorageBuffer | vkBU::eShaderDeviceAddress
                                | vkBU::eAccelerationStructureBuildInputReadOnlyKHR);
    model.indexBuffer =
        m_alloc.createBuffer(cmdBuf, loader.m_indices,
                            vkBU::eIndexBuffer | vkBU::eStorageBuffer | vkBU::eShaderDeviceAddress
                                | vkBU::eAccelerationStructureBuildInputReadOnlyKHR);
    model.matColorBuffer = m_alloc.createBuffer(cmdBuf, loader.m_materials, vkBU::eStorageBuffer);
    model.matIndexBuffer = m_alloc.createBuffer(cmdBuf, loader.m_matIndx, vkBU::eStorageBuffer);

    for (auto& texture : loader.m_textures) {
        std::cout << texture << std::endl;
    }
    
    // Creates all textures found
    createTextureImages(cmdBuf, loader.m_textures);
    cmdBufGet.submitAndWait(cmdBuf);
    m_alloc.finalizeAndReleaseStaging();

    std::string objNb = std::to_string(instance.objIndex);

    m_objModel.emplace_back(model);
    m_objInstance.emplace_back(instance);
}


void Application::Impl::createSkyboxTexture()
{
    using vkIU = vk::ImageUsageFlagBits;

    vk::SamplerCreateInfo samplerCreateInfo{
        {}, vk::Filter::eLinear, vk::Filter::eLinear, vk::SamplerMipmapMode::eLinear};
    samplerCreateInfo.setMaxLod(FLT_MAX);

    nvvk::CommandPool cmdBufGet(m_device, m_graphicsQueueIndex);
    vk::CommandBuffer cmdBuf = cmdBufGet.createCommandBuffer();

    // Loading 6 textures
    auto cubemap_txt_faces = { "right.png", "left.png", "top.png", "bottom.png", "front.png", "back.png" };

    std::vector<stbi_uc*> cubemap_face_data;
    vk::Format cubemap_format;
    int face_width;
    int face_height;
    int face_channel;

    // Load face pixels
    for (auto& txt_face : cubemap_txt_faces) {
    std::ostringstream txt_face_path_builder;
    txt_face_path_builder << "media/textures/skybox/" << txt_face;

    auto txt_face_path = nvh::findFile(txt_face_path_builder.str(), _default_search_paths, true);

    stbi_uc* stbi_pixels = stbi_load(txt_face_path.c_str(), &face_width, &face_height, &face_channel, STBI_rgb_alpha);

    if (stbi_pixels == nullptr) {
        std::runtime_error("Could not load skybox texture \"" + txt_face_path + "\"");
    }

    if (face_channel == 3) {
        cubemap_format = vk::Format::eR8G8B8Srgb;
    } else if (face_channel == 4) {
        cubemap_format = vk::Format::eR8G8B8A8Srgb;
    }

    cubemap_face_data.push_back(stbi_pixels);
    }

    // Arrange in linear memory
    int face_size = static_cast<uint64_t>(face_width * face_height) * sizeof(uint8_t) * face_channel;
    int total_size = face_size * 6;
    auto face_linear_data = std::make_unique<unsigned char[]>(total_size);

    for (int i = 0; i < 6; ++i) {
    memcpy(face_linear_data.get() + face_size * i, cubemap_face_data[i], face_size);
    stbi_image_free(cubemap_face_data[i]);
    }

    vk::DeviceSize buffer_size = static_cast<uint64_t>(total_size);
    auto img_size = vk::Extent2D { static_cast<uint32_t>(face_width), static_cast<uint32_t>(face_height) };
    auto img_create_info = nvvk::makeImageCubeCreateInfo(img_size, cubemap_format, vkIU::eSampled, true);

    {
    nvvk::ImageDedicated img = m_alloc.createImage(cmdBuf, buffer_size, face_linear_data.get(), img_create_info, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 6);

    auto iv_info = nvvk::makeImageViewCreateInfo(img.image, img_create_info, true);

    nvvk::cmdGenerateMipmaps(cmdBuf, img.image, cubemap_format, img_size, img_create_info.mipLevels);

    nvvk::Texture texture = m_alloc.createTexture(img, iv_info, samplerCreateInfo);

    m_skybox_txt = std::make_unique<nvvk::Texture>(texture);
    }

    cmdBufGet.submitAndWait(cmdBuf);
    // m_alloc.finalizeAndReleaseStaging();
}
