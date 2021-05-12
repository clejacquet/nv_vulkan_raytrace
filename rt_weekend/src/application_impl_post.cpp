#include "application_impl.hpp"
#include "nvvk/renderpasses_vk.hpp"
#include "nvvk/pipeline_vk.hpp"
#include "nvh/fileoperations.hpp"

// -----------------------
// Impl Post Methods
// -----------------------

void Application::Impl::createOffscreenRender()
{
    m_alloc.destroy(m_offscreenColor);
    m_alloc.destroy(m_offscreenDepth);

    // Creating the color image
    {
        auto colorCreateInfo = nvvk::makeImage2DCreateInfo(m_size, m_offscreenColorFormat,
                                                        vk::ImageUsageFlagBits::eColorAttachment
                                                            | vk::ImageUsageFlagBits::eSampled
                                                            | vk::ImageUsageFlagBits::eStorage);


        nvvk::Image             image  = m_alloc.createImage(colorCreateInfo);
        vk::ImageViewCreateInfo ivInfo = nvvk::makeImageViewCreateInfo(image.image, colorCreateInfo);
        m_offscreenColor               = m_alloc.createTexture(image, ivInfo, vk::SamplerCreateInfo());
        m_offscreenColor.descriptor.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    }

    // Creating the depth buffer
    auto depthCreateInfo =
        nvvk::makeImage2DCreateInfo(m_size, m_offscreenDepthFormat,
                                    vk::ImageUsageFlagBits::eDepthStencilAttachment);
    {
        nvvk::Image image = m_alloc.createImage(depthCreateInfo);

        vk::ImageViewCreateInfo depthStencilView;
        depthStencilView.setViewType(vk::ImageViewType::e2D);
        depthStencilView.setFormat(m_offscreenDepthFormat);
        depthStencilView.setSubresourceRange({vk::ImageAspectFlagBits::eDepth, 0, 1, 0, 1});
        depthStencilView.setImage(image.image);

        m_offscreenDepth = m_alloc.createTexture(image, depthStencilView);
    }

    // Setting the image layout for both color and depth
    {
        nvvk::CommandPool genCmdBuf(m_device, m_graphicsQueueIndex);
        auto              cmdBuf = genCmdBuf.createCommandBuffer();
        nvvk::cmdBarrierImageLayout(cmdBuf, m_offscreenColor.image, vk::ImageLayout::eUndefined,
                                    vk::ImageLayout::eGeneral);
        nvvk::cmdBarrierImageLayout(cmdBuf, m_offscreenDepth.image, vk::ImageLayout::eUndefined,
                                    vk::ImageLayout::eDepthStencilAttachmentOptimal,
                                    vk::ImageAspectFlagBits::eDepth);

        genCmdBuf.submitAndWait(cmdBuf);
    }

    // Creating a renderpass for the offscreen
    if(!m_offscreenRenderPass)
    {
        m_offscreenRenderPass =
            nvvk::createRenderPass(m_device, {m_offscreenColorFormat}, m_offscreenDepthFormat, 1, true,
                                true, vk::ImageLayout::eGeneral, vk::ImageLayout::eGeneral);
    }

    // Creating the frame buffer for offscreen
    std::vector<vk::ImageView> attachments = {m_offscreenColor.descriptor.imageView,
                                                m_offscreenDepth.descriptor.imageView};

    m_device.destroy(m_offscreenFramebuffer);
    vk::FramebufferCreateInfo info;
    info.setRenderPass(m_offscreenRenderPass);
    info.setAttachmentCount(2);
    info.setPAttachments(attachments.data());
    info.setWidth(m_size.width);
    info.setHeight(m_size.height);
    info.setLayers(1);
    m_offscreenFramebuffer = m_device.createFramebuffer(info);
}

void Application::Impl::createPostPipeline()
{
    // Push constants in the fragment shader
    vk::PushConstantRange pushConstantRanges = {vk::ShaderStageFlagBits::eFragment, 0, sizeof(float)};

    // Creating the pipeline layout
    vk::PipelineLayoutCreateInfo pipelineLayoutCreateInfo;
    pipelineLayoutCreateInfo.setSetLayoutCount(1);
    pipelineLayoutCreateInfo.setPSetLayouts(&m_postDescSetLayout);
    pipelineLayoutCreateInfo.setPushConstantRangeCount(1);
    pipelineLayoutCreateInfo.setPPushConstantRanges(&pushConstantRanges);
    m_postPipelineLayout = m_device.createPipelineLayout(pipelineLayoutCreateInfo);

    // Pipeline: completely generic, no vertices
    nvvk::GraphicsPipelineGeneratorCombined pipelineGenerator(m_device, m_postPipelineLayout,
                                                                m_renderPass);
    pipelineGenerator.addShader(nvh::loadFile("spv/passthrough.vert.spv", true, _default_search_paths,
                                                true),
                                vk::ShaderStageFlagBits::eVertex);
    pipelineGenerator.addShader(nvh::loadFile("spv/post.frag.spv", true, _default_search_paths, true),
                                vk::ShaderStageFlagBits::eFragment);
    pipelineGenerator.rasterizationState.setCullMode(vk::CullModeFlagBits::eNone);
    m_postPipeline = pipelineGenerator.createPipeline();
}

void Application::Impl::createPostDescriptor()
{
    using vkDS = vk::DescriptorSetLayoutBinding;
    using vkDT = vk::DescriptorType;
    using vkSS = vk::ShaderStageFlagBits;

    m_postDescSetLayoutBind.addBinding(vkDS(0, vkDT::eCombinedImageSampler, 1, vkSS::eFragment));
    m_postDescSetLayout = m_postDescSetLayoutBind.createLayout(m_device);
    m_postDescPool      = m_postDescSetLayoutBind.createPool(m_device);
    m_postDescSet       = nvvk::allocateDescriptorSet(m_device, m_postDescPool, m_postDescSetLayout);
}

void Application::Impl::updatePostDescriptorSet()
{
    vk::WriteDescriptorSet writeDescriptorSets =
    m_postDescSetLayoutBind.makeWrite(m_postDescSet, 0, &m_offscreenColor.descriptor);
    m_device.updateDescriptorSets(writeDescriptorSets, nullptr);
}

void Application::Impl::drawPost(vk::CommandBuffer cmdBuf)
{
    cmdBuf.setViewport(0, {vk::Viewport(0, 0, (float)m_size.width, (float)m_size.height, 0, 1)});
    cmdBuf.setScissor(0, {{{0, 0}, {m_size.width, m_size.height}}});

    auto aspectRatio = static_cast<float>(m_size.width) / static_cast<float>(m_size.height);
    cmdBuf.pushConstants<float>(m_postPipelineLayout, vk::ShaderStageFlagBits::eFragment, 0,
                                aspectRatio);
    cmdBuf.bindPipeline(vk::PipelineBindPoint::eGraphics, m_postPipeline);
    cmdBuf.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, m_postPipelineLayout, 0,
                                m_postDescSet, {});
    cmdBuf.draw(3, 1, 0, 0);
}
