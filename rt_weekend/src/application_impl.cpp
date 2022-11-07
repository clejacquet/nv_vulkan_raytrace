#include "application_impl.hpp"
VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE

#include "nvvk/renderpasses_vk.hpp"
#include "nvh/fileoperations.hpp"
#include "imgui/extras/imgui_camera_widget.h"
#include "nvh/cameramanipulator.hpp"


// -----------------------
// Impl Methods
// -----------------------

void Application::Impl::setup(
    const vk::Instance&       instance,
    const vk::Device&         device,
    const vk::PhysicalDevice& physicalDevice,
    uint32_t                  queueFamily)
{
    AppBase::setup(instance, device, physicalDevice, queueFamily);
    m_alloc.init(device, physicalDevice);
    m_offscreenDepthFormat = nvvk::findDepthFormat(physicalDevice);

    // Search path for shaders and other media
    _default_search_paths = {
        NVPSystem::exePath() + PROJECT_RELDIRECTORY,
        NVPSystem::exePath() + PROJECT_RELDIRECTORY "..",
        std::string(PROJECT_NAME),
    };
}

void Application::Impl::initWindow()
{
    glfwInit();

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

    _window = glfwCreateWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "RT Weekend", nullptr, nullptr);

    // Setup camera
    CameraManip.setMode(CameraManip.Walk);
    CameraManip.setWindowSize(WINDOW_WIDTH, WINDOW_HEIGHT);
    CameraManip.setLookat(nvmath::vec3f(0, 0, 1), nvmath::vec3f(0, 0, 0), nvmath::vec3f(0, 1, 0));

    m_camera_ref.camera = CameraManip.getMatrix();
    m_camera_ref.fov = CameraManip.getFov();
    resetFrameId();
}

void Application::Impl::loadVulkanContext() {
    // Requesting Vulkan extensions and layers
    nvvk::ContextCreateInfo context_info;

    context_info.setVersion(1, 2);
    context_info.addInstanceLayer("VK_LAYER_LUNARG_monitor", true);
    context_info.addInstanceExtension(VK_EXT_DEBUG_UTILS_EXTENSION_NAME, true);
    context_info.addInstanceExtension(VK_KHR_SURFACE_EXTENSION_NAME);
#ifdef _WIN32
    context_info.addInstanceExtension(VK_KHR_WIN32_SURFACE_EXTENSION_NAME);
#else
    contextInfo.addInstanceExtension(VK_KHR_XLIB_SURFACE_EXTENSION_NAME);
    contextInfo.addInstanceExtension(VK_KHR_XCB_SURFACE_EXTENSION_NAME);
#endif
    context_info.addInstanceExtension(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);
    context_info.addDeviceExtension(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
    context_info.addDeviceExtension(VK_KHR_DEDICATED_ALLOCATION_EXTENSION_NAME);
    context_info.addDeviceExtension(VK_KHR_GET_MEMORY_REQUIREMENTS_2_EXTENSION_NAME);
    context_info.addDeviceExtension(VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME);
    context_info.addDeviceExtension(VK_EXT_SCALAR_BLOCK_LAYOUT_EXTENSION_NAME);

    // Requesting the extensions needed for ray tracing 
    vk::PhysicalDeviceAccelerationStructureFeaturesKHR accel_feature;
    context_info.addDeviceExtension(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME, false, &accel_feature);

    vk::PhysicalDeviceRayTracingPipelineFeaturesKHR rt_pipeline_feature;
    context_info.addDeviceExtension(VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME, false, &rt_pipeline_feature);

    context_info.addDeviceExtension(VK_KHR_MAINTENANCE3_EXTENSION_NAME);
    context_info.addDeviceExtension(VK_KHR_PIPELINE_LIBRARY_EXTENSION_NAME);
    context_info.addDeviceExtension(VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME);
    context_info.addDeviceExtension(VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME);

    // Creating Vulkan base application
    _nvvk_context.initInstance(context_info);

    // Find all compatible devices
    auto compatible_devices = _nvvk_context.getCompatibleDevices(context_info);
    assert(!compatible_devices.empty());

    // Use a compatible device
    _nvvk_context.initDevice(compatible_devices[0], context_info);
}

void Application::Impl::setupVulkanPipeline() {
    // Window need to be opened to get the surface on which to draw
    const vk::SurfaceKHR surface = nvvk::AppBase::getVkSurface(_nvvk_context.m_instance, _window);
    _nvvk_context.setGCTQueueWithPresent(surface);

    setup(
        _nvvk_context.m_instance, 
        _nvvk_context.m_device, 
        _nvvk_context.m_physicalDevice,
        _nvvk_context.m_queueGCT.familyIndex);

    nvvk::AppBase::createSwapchain(surface, WINDOW_WIDTH, WINDOW_HEIGHT);
    nvvk::AppBase::createDepthBuffer();
    nvvk::AppBase::createRenderPass();
    nvvk::AppBase::createFrameBuffers();

    // Setup Imgui
    initGUI(0);  // Using sub-pass 0

    // Creation of the example
    createSkyboxTexture();
    loadModel(nvh::findFile("media/scenes/Medieval_building.obj", _default_search_paths, true));
    loadModel(nvh::findFile("media/scenes/plane.obj", _default_search_paths, true));

    {
        auto sphere_cmdpool = nvvk::CommandPool { m_device, m_graphicsQueueIndex };
        m_sphereHandler = std::make_unique<SphereHandler>(sphere_cmdpool, m_alloc);
    }

    createOffscreenRender();
    createDescriptorSetLayout();
    createUniformBuffer();
    createSceneDescriptionBuffer();
    updateDescriptorSet();

    initRayTracing();
    createBottomLevelAS();
    createTopLevelAS();
    createRtDescriptorSet();
    createRtPipeline();
    createRtShaderBindingTable();

    createPostDescriptor();
    createPostPipeline();
    updatePostDescriptorSet();
}

void Application::Impl::destroyResources()
{
    m_device.destroy(m_descPool);
    m_device.destroy(m_descSetLayout);
    m_alloc.destroy(m_cameraMat);
    m_alloc.destroy(m_sceneDesc);

    for(auto& m : m_objModel)
    {
        m_alloc.destroy(m.vertexBuffer);
        m_alloc.destroy(m.indexBuffer);
        m_alloc.destroy(m.matColorBuffer);
        m_alloc.destroy(m.matIndexBuffer);
    }

    m_sphereHandler->destroy(m_alloc);


    for(auto& t : m_textures)
    {
        m_alloc.destroy(t);
    }
    m_alloc.destroy(*m_skybox_txt);

    //#Post
    m_device.destroy(m_postPipeline);
    m_device.destroy(m_postPipelineLayout);
    m_device.destroy(m_postDescPool);
    m_device.destroy(m_postDescSetLayout);
    m_alloc.destroy(m_offscreenColor);
    m_alloc.destroy(m_offscreenDepth);
    m_device.destroy(m_offscreenRenderPass);
    m_device.destroy(m_offscreenFramebuffer);

    // #VKRay
    m_rtBuilder.destroy();
    m_device.destroy(m_rtDescPool);
    m_device.destroy(m_rtDescSetLayout);
    m_device.destroy(m_rtPipeline);
    m_device.destroy(m_rtPipelineLayout);
    m_alloc.destroy(m_rtSBTBuffer);
}

// Extra UI
void Application::Impl::renderUI()
{
    bool changed = false;

    changed |= ImGuiH::CameraWidget();
    if(ImGui::CollapsingHeader("Light"))
    {
        auto& pc = m_pushConstant;
        changed |= ImGui::RadioButton("Point", &pc.lightType, 0);
        ImGui::SameLine();
        changed |= ImGui::RadioButton("Infinite", &pc.lightType, 1);

        changed |= ImGui::SliderFloat3("Position", &pc.lightPosition.x, -20.f, 20.f);
        changed |= ImGui::SliderFloat("Intensity", &pc.lightIntensity, 0.f, 150.f);
    }

    if(changed) {
        resetFrameId();
    }
}

void Application::Impl::onResize(int w, int h)
{
    resetFrameId();
    createOffscreenRender();
    updatePostDescriptorSet();
    updateRtDescriptorSet();
}

void Application::Impl::resetFrameId() {
    m_rtcurrentFrameId = -1;
}

void Application::Impl::updateFrameId() {
    const auto& current_camera_mat = CameraManip.getMatrix();
    auto        current_camera_fov = CameraManip.getFov();

    if (memcmp(&current_camera_mat.a00, &m_camera_ref.camera.a00, sizeof(nvmath::mat4f)) != 0 || current_camera_fov != m_camera_ref.fov) {
        resetFrameId();
        m_camera_ref.camera = current_camera_mat;
        m_camera_ref.fov = current_camera_fov;
    }
    
    m_rtcurrentFrameId++;
}
