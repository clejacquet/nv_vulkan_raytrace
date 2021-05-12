#include "application.hpp"
#include "application_impl.hpp"
#include "imgui.h"
#include "imgui/backends/imgui_impl_glfw.h"


// -----------------------
// Public Methods
// -----------------------

Application::Application() :
    _impl(std::make_unique<Impl>())
{
    _impl->initWindow();
    _impl->loadVulkanContext();
    _impl->setupVulkanPipeline();
}

Application::~Application()
{
    _impl->getDevice().waitIdle();
    _impl->destroyResources();
    _impl->destroy();
    _impl->_nvvk_context.deinit();

    glfwDestroyWindow(_impl->_window);

    glfwTerminate();
}




void Application::run()
{
    _impl->setupGlfwCallbacks(_impl->_window);
    // ImGui_ImplGlfw_InitForVulkan(_impl->_window, true);

    nvmath::vec4f clearColor = nvmath::vec4f(1, 1, 1, 1.00f);

    while (!glfwWindowShouldClose(_impl->_window)) {
        glfwPollEvents();

        if(_impl->isMinimized()) {
            continue;
        }

        // Start the Dear ImGui frame
        // ImGui_ImplGlfw_NewFrame();
        // ImGui::NewFrame();

        // // Show UI window.
        // if (_impl->showGui())
        // {
        //     ImGuiH::Panel::Begin();
        //     ImGui::ColorEdit3("Clear color", reinterpret_cast<float*>(&clearColor));
        //     _impl->renderUI();
        //     ImGui::Text("Application average %.3f ms/frame (%.1f FPS)",
        //                             1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
        //     ImGuiH::Control::Info("", "", "(F10) Toggle Pane", ImGuiH::Control::Flags::Disabled);
        //     ImGuiH::Panel::End();
        // }

        // Start rendering the scene
        _impl->prepareFrame();

        // Start command buffer of this frame
        auto                     curFrame = _impl->getCurFrame();
        const vk::CommandBuffer& cmdBuf   = _impl->getCommandBuffers()[curFrame];

        cmdBuf.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});

        // Updating camera buffer
        _impl->updateUniformBuffer(cmdBuf);


        // Clearing screen
        vk::ClearValue clearValues[2];
        clearValues[0].setColor(
                std::array<float, 4>({clearColor[0], clearColor[1], clearColor[2], clearColor[3]}));
        clearValues[1].setDepthStencil({1.0f, 0});

        // Offscreen render pass
        {
            // Rendering Scene
            _impl->rayTrace(cmdBuf, clearColor);
        }


        // 2nd rendering pass: tone mapper, UI
        {
            vk::RenderPassBeginInfo postRenderPassBeginInfo;
            postRenderPassBeginInfo.setClearValueCount(2);
            postRenderPassBeginInfo.setPClearValues(clearValues);
            postRenderPassBeginInfo.setRenderPass(_impl->getRenderPass());
            postRenderPassBeginInfo.setFramebuffer(_impl->getFramebuffers()[curFrame]);
            postRenderPassBeginInfo.setRenderArea({{}, _impl->getSize()});

            cmdBuf.beginRenderPass(postRenderPassBeginInfo, vk::SubpassContents::eInline);
            // Rendering tonemapper
            _impl->drawPost(cmdBuf);
            // Rendering UI
            // ImGui::Render();
            // ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cmdBuf);
            cmdBuf.endRenderPass();
        }

        // Submit for display
        cmdBuf.end();
        _impl->submitFrame();
    }
}
