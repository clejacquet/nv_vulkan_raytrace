#ifndef APPLICATION_IMPL_HPP
#define APPLICATION_IMPL_HPP

#include "application.hpp"

#include <GLFW/glfw3.h>

#include <vulkan/vulkan.hpp>

#define NVVK_ALLOC_DEDICATED
#include <nvvk/context_vk.hpp>
#include <nvvk/appbase_vkpp.hpp>
#include <nvvk/descriptorsets_vk.hpp>
#include <nvvk/raytraceKHR_vk.hpp>

#include "primitive/sphere.hpp"

// -----------------------
// Constants
// -----------------------

static constexpr int WINDOW_WIDTH = 1280;
static constexpr int WINDOW_HEIGHT = 720;


// -----------------------
// Impl Structure
// -----------------------

class Application::Impl : public nvvk::AppBase
{
private:
    void setup(
            const vk::Instance&       instance,
            const vk::Device&         device,
            const vk::PhysicalDevice& physicalDevice,
            uint32_t                  queueFamily
        ) override;

    struct CameraParams {
        nvmath::mat4f camera;
        float         fov;
    };

public:
    CameraParams m_camera_ref;
    int m_max_accumulated_frames = 100;
    GLFWwindow* _window;
    nvvk::Context _nvvk_context;
    std::vector<std::string> _default_search_paths;

    void initWindow();
    void loadVulkanContext();
    void setupVulkanPipeline();
    void destroyResources();
    void renderUI();
    void onResize(int w, int h) override;
    void resetFrameId();
    void updateFrameId();


    // The OBJ model
    struct ObjModel
    {
        uint32_t     nbIndices{0};
        uint32_t     nbVertices{0};
        nvvk::Buffer vertexBuffer;    // Device buffer of all 'Vertex'
        nvvk::Buffer indexBuffer;     // Device buffer of the indices forming triangles
        nvvk::Buffer matColorBuffer;  // Device buffer of array of 'Wavefront material'
        nvvk::Buffer matIndexBuffer;  // Device buffer of array of 'Wavefront material'
    };

    // Instance of the OBJ
    struct ObjInstance
    {
        uint32_t      objIndex{0};     // Reference to the `m_objModel`
        uint32_t      txtOffset{0};    // Offset in `m_textures`
        nvmath::mat4f transform{1};    // Position of the instance
        nvmath::mat4f transformIT{1};  // Inverse transpose
    };

    // Information pushed at each draw call
    struct ObjPushConstant
    {
        nvmath::vec3f lightPosition{10.f, 15.f, 8.f};
        int           instanceId{0};  // To retrieve the transformation matrix
        float         lightIntensity{100.f};
        int           lightType{0};  // 0: point, 1: infinite
    };
    ObjPushConstant m_pushConstant;

    // #Descriptors
    void createDescriptorSetLayout();
    void updateDescriptorSet();
    void createUniformBuffer();
    void updateUniformBuffer(const vk::CommandBuffer& cmdBuf);
    void createSceneDescriptionBuffer();
    void createTextureImages(const vk::CommandBuffer& cmdBuf, const std::vector<std::string>& textures);
    void loadModel(const std::string& filename, nvmath::mat4f transform = nvmath::mat4f(1));
    void createSkyboxTexture();


    // Array of objects and instances in the scene
    std::vector<ObjModel>          m_objModel;
    std::vector<ObjInstance>       m_objInstance;
    std::unique_ptr<SphereHandler> m_sphereHandler;

    // Graphic pipeline
    nvvk::DescriptorSetBindings m_descSetLayoutBind;
    vk::DescriptorPool          m_descPool;
    vk::DescriptorSetLayout     m_descSetLayout;
    vk::DescriptorSet           m_descSet;

    nvvk::Buffer               m_cameraMat;  // Device-Host of the camera matrices
    nvvk::Buffer               m_sceneDesc;  // Device buffer of the OBJ instances
    std::vector<nvvk::Texture> m_textures;   // vector of all textures of the scene
    std::unique_ptr<nvvk::Texture> m_skybox_txt = nullptr; 


    nvvk::AllocatorDedicated m_alloc;  // Allocator for buffer, images, acceleration structures

    // #Post
    void createOffscreenRender();
    void createPostPipeline();
    void createPostDescriptor();
    void updatePostDescriptorSet();
    void drawPost(vk::CommandBuffer cmdBuf);

    nvvk::DescriptorSetBindings m_postDescSetLayoutBind;
    vk::DescriptorPool          m_postDescPool;
    vk::DescriptorSetLayout     m_postDescSetLayout;
    vk::DescriptorSet           m_postDescSet;
    vk::Pipeline                m_postPipeline;
    vk::PipelineLayout          m_postPipelineLayout;
    vk::RenderPass              m_offscreenRenderPass;
    vk::Framebuffer             m_offscreenFramebuffer;
    nvvk::Texture               m_offscreenColor;
    vk::Format                  m_offscreenColorFormat{vk::Format::eR32G32B32A32Sfloat};
    nvvk::Texture               m_offscreenDepth;
    vk::Format                  m_offscreenDepthFormat;

    // #VKRAY
    void initRayTracing();
    nvvk::RaytracingBuilderKHR::BlasInput objectToVkGeometryKHR(const ObjModel& model);
    void createBottomLevelAS();
    void createTopLevelAS();
    void createRtDescriptorSet();
    void updateRtDescriptorSet();
    void createRtPipeline();
    void createRtShaderBindingTable();
    void rayTrace(const vk::CommandBuffer& cmdBuf, const nvmath::vec4f& clearColor);

    vk::PhysicalDeviceRayTracingPipelinePropertiesKHR    m_rtProperties;
    nvvk::RaytracingBuilderKHR                           m_rtBuilder;
    nvvk::DescriptorSetBindings                          m_rtDescSetLayoutBind;
    vk::DescriptorPool                                   m_rtDescPool;
    vk::DescriptorSetLayout                              m_rtDescSetLayout;
    vk::DescriptorSet                                    m_rtDescSet;
    std::vector<vk::RayTracingShaderGroupCreateInfoKHR>  m_rtShaderGroups;
    vk::PipelineLayout                                   m_rtPipelineLayout;
    vk::Pipeline                                         m_rtPipeline;
    nvvk::Buffer                                         m_rtSBTBuffer;
    int                                                  m_rtcurrentFrameId;

    struct RtPushConstant
    {
        nvmath::vec4f clearColor;
        nvmath::vec3f lightPosition;
        float         lightIntensity;
        int           lightType;
        int           frameId;
    } m_rtPushConstants;
};


#endif