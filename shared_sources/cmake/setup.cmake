#*****************************************************************************
# Copyright 2020 NVIDIA Corporation. All rights reserved.
#*****************************************************************************


#The OLD behavior for this policy is to ignore <PackageName>_ROOT variables
if(POLICY CMP0074)
  cmake_policy(SET CMP0074 NEW)
endif()


# Set the C/C++ specified in the projects as requirements
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 14)
# Find includes in corresponding build directories
set(CMAKE_INCLUDE_CURRENT_DIR ON)


# we need the .exe suffix even on Linux to remove ambiguity between target
# executable name  and data files stored in a directory of the same name
set(CMAKE_EXECUTABLE_SUFFIX ".exe")

# IDE Setup
set_property(GLOBAL PROPERTY USE_FOLDERS ON)  # Generate folders for IDE targets
set_property(GLOBAL PROPERTY PREDEFINED_TARGETS_FOLDER "_cmake")

# https://cmake.org/cmake/help/latest/policy/CMP0072.html
set(OpenGL_GL_PREFERENCE GLVND)

set(SUPPORT_SOCKETS OFF CACHE BOOL "add a socket protocol so samples can be controled remotely")
set(SUPPORT_NVTOOLSEXT OFF CACHE BOOL "Use NSight for custom markers")

if(WIN32)
  set( MEMORY_LEAKS_CHECK OFF CACHE BOOL "Check for Memory leaks" )
endif(WIN32)

if(MSVC)
  # Enable parallel builds by default on MSVC
  string(APPEND CMAKE_C_FLAGS " /MP")
  string(APPEND CMAKE_CXX_FLAGS " /MP")
endif()

set(RESOURCE_DIRECTORY "${BASE_DIRECTORY}/shared_sources/resources")
add_definitions(-DRESOURCE_DIRECTORY="${RESOURCE_DIRECTORY}/")

include_directories(${BASE_DIRECTORY}/shared_sources)
include_directories(${BASE_DIRECTORY}/shared_external)

if (WIN32)
  include_directories(${BASE_DIRECTORY}/shared_external/glfw3/include)
else() 
  find_package(glfw3 3.3 QUIET)

  if (NOT glfw3_FOUND AND NOT GLFW3_DIR)
    set(GLFW3_ZIP ${BASE_DIRECTORY}/downloaded_resources/glfw-3.3.1.zip)
    set(GLFW3_DIR ${BASE_DIRECTORY}/downloaded_resources/glfw-3.3.1)

    if(NOT EXISTS ${GLFW3_DIR}/CMakeLists.txt)
      if(NOT EXISTS ${GLFW3_DIR})
      if(NOT EXISTS ${GLFW3_ZIP})
        file(DOWNLOAD https://github.com/glfw/glfw/releases/download/3.3.1/glfw-3.3.1.zip ${GLFW3_ZIP}
           TIMEOUT 60  # seconds
           TLS_VERIFY ON)
      endif()
      execute_process(COMMAND ${CMAKE_COMMAND} -E tar -xf ${GLFW3_ZIP}
              WORKING_DIRECTORY ${BASE_DIRECTORY}/downloaded_resources)
      endif()
	endif()

	# prevent glfw from building all kinds of stuff we don't need
	set(GLFW_BUILD_EXAMPLES OFF)
	set(GLFW_BUILD_TESTS OFF)
	set(GLFW_BUILD_DOCS OFF)
	set(GLFW_INSTALL OFF)

	add_subdirectory(${GLFW3_DIR} ${BASE_DIRECTORY}/build/build_glfw3/src)
    include_directories(${GLFW3_DIR}/include)
  endif()  
endif()

# Specify the list of directories to search for cmake modules.
set(CMAKE_MODULE_PATH ${BASE_DIRECTORY}/shared_sources/cmake ${BASE_DIRECTORY}/shared_sources/cmake/find)
set(CMAKE_FIND_ROOT_PATH "")

message(STATUS "BASE_DIRECTORY = ${BASE_DIRECTORY}")
message(STATUS "CMAKE_CURRENT_SOURCE_DIR = ${CMAKE_CURRENT_SOURCE_DIR}")

if(CMAKE_SIZEOF_VOID_P EQUAL 8)
  set(ARCH "x64" CACHE STRING "CPU Architecture")
else ()
  set(ARCH "x86" CACHE STRING "CPU Architecture")
endif()

set(OUTPUT_PATH ${BASE_DIRECTORY}/bin_${ARCH} CACHE PATH "Directory where outputs will be stored")

# Set the default build to Release.  Note this doesn't do anything for the VS
# default build target.
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel." FORCE)
endif()

if(NOT EXISTS ${BASE_DIRECTORY}/downloaded_resources)
  file(MAKE_DIRECTORY ${BASE_DIRECTORY}/downloaded_resources)
endif()

set(DOWNLOAD_TARGET_DIR "${BASE_DIRECTORY}/downloaded_resources")
set(DOWNLOAD_SITE http://developer.download.nvidia.com/ProGraphics/nvpro-samples)

#####################################################################################
function(_make_relative FROM TO OUT)
  #message(STATUS "FROM = ${FROM}")
  #message(STATUS "TO = ${TO}")
  
  file(RELATIVE_PATH _TMP_STR "${FROM}" "${TO}")
  
  #message(STATUS "_TMP_STR = ${_TMP_STR}")
  
  set (${OUT} "${_TMP_STR}" PARENT_SCOPE)
endfunction()

macro(_add_project_definitions name)
  if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX "${BASE_DIRECTORY}/_install" CACHE PATH "folder in which INSTALL will put everything needed to run the binaries" FORCE)
  endif(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  if(CUDA_TOOLKIT_ROOT_DIR)
    string(REPLACE "\\" "/" CUDA_TOOLKIT_ROOT_DIR ${CUDA_TOOLKIT_ROOT_DIR})
  endif()
  if(VULKANSDK_LOCATION)
    string(REPLACE "\\" "/" VULKANSDK_LOCATION ${VULKANSDK_LOCATION})
  endif()
  if(CMAKE_INSTALL_PREFIX)
    string(REPLACE "\\" "/" CMAKE_INSTALL_PREFIX ${CMAKE_INSTALL_PREFIX})
  endif()
  
  # the "config" directory doesn't really exist but serves as place holder
  # for the actual CONFIG based directories (Release, RelWithDebInfo etc.)
  _make_relative("${OUTPUT_PATH}/config" "${CMAKE_CURRENT_SOURCE_DIR}" TO_CURRENT_SOURCE_DIR)
  _make_relative("${OUTPUT_PATH}/config" "${DOWNLOAD_TARGET_DIR}" TO_DOWNLOAD_TARGET_DIR)
  
  add_definitions(-DPROJECT_NAME="${name}")
  add_definitions(-DPROJECT_RELDIRECTORY="${TO_CURRENT_SOURCE_DIR}/")
  add_definitions(-DPROJECT_DOWNLOAD_RELDIRECTORY="${TO_DOWNLOAD_TARGET_DIR}/")

  if (SUPPORT_NVTOOLSEXT)
    add_definitions(-DNVP_SUPPORTS_NVTOOLSEXT)
    # add the package if we checked SUPPORT_NVTOOLSEXT in cmake
    _add_package_NSight()
  endif(SUPPORT_NVTOOLSEXT)
endmacro(_add_project_definitions)

#####################################################################################

macro(_set_subsystem_console exename)
  if(WIN32)
     set_target_properties(${exename} PROPERTIES LINK_FLAGS_DEBUG "/SUBSYSTEM:CONSOLE")
     target_compile_definitions(${exename} PRIVATE "_CONSOLE")
     set_target_properties(${exename} PROPERTIES LINK_FLAGS_RELWITHDEBINFO "/SUBSYSTEM:CONSOLE")
     set_target_properties(${exename} PROPERTIES LINK_FLAGS_RELEASE "/SUBSYSTEM:CONSOLE")
     set_target_properties(${exename} PROPERTIES LINK_FLAGS_MINSIZEREL "/SUBSYSTEM:CONSOLE")
  endif(WIN32)
endmacro(_set_subsystem_console)

macro(_set_subsystem_windows exename)
  if(WIN32)
     set_target_properties(${exename} PROPERTIES LINK_FLAGS_DEBUG "/SUBSYSTEM:WINDOWS")
     set_target_properties(${exename} PROPERTIES LINK_FLAGS_RELWITHDEBINFO "/SUBSYSTEM:WINDOWS")
     set_target_properties(${exename} PROPERTIES LINK_FLAGS_RELEASE "/SUBSYSTEM:WINDOWS")
     set_target_properties(${exename} PROPERTIES LINK_FLAGS_MINSIZEREL "/SUBSYSTEM:WINDOWS")
  endif(WIN32)
endmacro(_set_subsystem_windows)

#####################################################################################
if(UNIX) 
  set(OS "linux")
  add_definitions(-DLINUX)
  add_compile_options(-fpermissive)
else(UNIX)
  if(APPLE)
  else(APPLE)
    if(WIN32)
      set(OS "win")
      add_definitions(-DNOMINMAX)
      if(MEMORY_LEAKS_CHECK)
        add_definitions(-DMEMORY_LEAKS_CHECK)
      endif()
    endif(WIN32)
  endif(APPLE)
endif(UNIX)


# Macro for adding files close to the executable
macro(_copy_files_to_target target thefiles)
    if(WIN32)
        foreach (FFF ${thefiles} )
          if(EXISTS "${FFF}")
            add_custom_command(
              TARGET ${target} POST_BUILD
              COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${FFF}
                $<TARGET_FILE_DIR:${target}>
                VERBATIM
            )
          endif()
        endforeach()
    endif()
endmacro()

# Macro for adding files close to the executable
macro(_copy_file_to_target target thefile folder)
  if(WIN32)
    if(EXISTS "${thefile}")
      add_custom_command(
        TARGET ${target} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory
          "$<TARGET_FILE_DIR:${target}>/${folder}"
          VERBATIM
      )
      add_custom_command(
        TARGET ${target} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
          ${thefile}
          "$<TARGET_FILE_DIR:${target}>/${folder}"
          VERBATIM
      )
    endif()
  endif()
endmacro()

#####################################################################################
# Optional OpenGL package
#
macro(_add_package_OpenGL)
  find_package(OpenGL)  
  if(OPENGL_FOUND)
      Message(STATUS "--> using package OpenGL")
      get_directory_property(hasParent PARENT_DIRECTORY)
      if(hasParent)
        set( USING_OPENGL "YES" PARENT_SCOPE) # PARENT_SCOPE important to have this variable passed to parent. Here we want to notify that something used the OpenGL package
      endif()
      set( USING_OPENGL "YES")
      add_definitions(-DNVP_SUPPORTS_OPENGL)
 else(OPENGL_FOUND)
     Message(STATUS "--> NOT using package OpenGL")
 endif(OPENGL_FOUND)
endmacro(_add_package_OpenGL)
# this macro is needed for the samples to add this package, although not needed
# this happens when the shared_sources library was built with these stuff in it
# so many samples can share the same library for many purposes
macro(_optional_package_OpenGL)
  if(USING_OPENGL)
    _add_package_OpenGL()
  endif(USING_OPENGL)
endmacro(_optional_package_OpenGL)

#####################################################################################
# Optional ZLIB
#
macro(_add_package_ZLIB)
  if(EXISTS ${BASE_DIRECTORY}/shared_external/zlib)
    set(ZLIB_ROOT ${BASE_DIRECTORY}/shared_external/zlib)
  endif()
  Message(STATUS "--> using package ZLIB")
  find_package(ZLIB)
  if(ZLIB_FOUND)
      add_definitions(-DNVP_SUPPORTS_GZLIB)
      include_directories(${ZLIB_INCLUDE_DIR})
      LIST(APPEND PACKAGE_SOURCE_FILES
        ${ZLIB_HEADERS}
        )
      LIST(APPEND LIBRARIES_OPTIMIZED ${ZLIB_LIBRARY})
      LIST(APPEND LIBRARIES_DEBUG ${ZLIB_LIBRARY})
  else()
      Message(WARNING "ZLIB not available.")
  endif()
endmacro()
# this macro is needed for the samples to add this package, although not needed
# this happens when the shared_sources library was built with these stuff in it
# so many samples can share the same library for many purposes
macro(_optional_package_ZLIB)
  if(ZLIB_FOUND)
    _add_package_ZLIB()
  endif(ZLIB_FOUND)
endmacro(_optional_package_ZLIB)

#####################################################################################
# ImGUI
#
macro(_add_package_ImGUI)
  Message(STATUS "--> using package ImGUI")
  include_directories(${BASE_DIRECTORY}/shared_sources/imgui)

  set(USING_IMGUI ON)
  get_directory_property(hasParent PARENT_DIRECTORY)
  if(hasParent)
    set(USING_IMGUI ON PARENT_SCOPE) # PARENT_SCOPE important to have this variable passed to parent. Here we want to notify that something used the OpenGL package
  endif()

endmacro()

#####################################################################################
# AntTweakBar UI
#
macro(_add_package_AntTweakBar)
  Message(STATUS "--> using package AntTweakBar")
  find_package(AntTweakBar)
  if(ANTTWEAKBAR_FOUND)
    add_definitions(-DNVP_SUPPORTS_ANTTWEAKBAR)
    include_directories(${ANTTWEAKBAR_INCLUDE_DIR})
    LIST(APPEND PACKAGE_SOURCE_FILES 
      ${ANTTWEAKBAR_HEADERS}
    )
    LIST(APPEND LIBRARIES_OPTIMIZED ${ANTTWEAKBAR_LIB})
    LIST(APPEND LIBRARIES_DEBUG ${ANTTWEAKBAR_LIB})
    source_group(AntTweakBar FILES 
      ${ANTTWEAKBAR_HEADERS}
    )
  endif()
endmacro()
# this macro is needed for the samples to add this package, although not needed
# this happens when the shared_sources library was built with these stuff in it
# so many samples can share the same library for many purposes
macro(_optional_package_AntTweakBar)
  if(ANTTWEAKBAR_FOUND)
    _add_package_AntTweakBar()
  endif(ANTTWEAKBAR_FOUND)
endmacro(_optional_package_AntTweakBar)


#####################################################################################
# FreeImage
#
macro(_add_package_FreeImage)
  Message(STATUS "--> using package FreeImage")
  find_package(FreeImage)
  if(FREEIMAGE_FOUND)
    add_definitions(-DNVP_SUPPORTS_FREEIMAGE)
    include_directories(${FREEIMAGE_INCLUDE_DIR})
    LIST(APPEND PACKAGE_SOURCE_FILES 
      ${FREEIMAGE_HEADERS}
    )
    LIST(APPEND LIBRARIES_OPTIMIZED ${FREEIMAGE_LIB})
    LIST(APPEND LIBRARIES_DEBUG ${FREEIMAGE_LIB})
    # source_group(AntTweakBar FILES 
    #   ${ANTTWEAKBAR_HEADERS}
    # )
  endif()
endmacro()
# this macro is needed for the samples to add this package, although not needed
# this happens when the shared_sources library was built with these stuff in it
# so many samples can share the same library for many purposes
macro(_optional_package_FreeImage)
  if(FREEIMAGE_FOUND)
    _add_package_FreeImage()
  endif(FREEIMAGE_FOUND)
endmacro(_optional_package_FreeImage)




#####################################################################################
# VMA (Vulkan Memory Allocator)
#
macro(_add_package_VMA)
  Message(STATUS "--> using package VMA")
  find_package(VMA)
  if(VMA_FOUND)
    add_definitions(-DNVP_SUPPORTS_VMA)
    include_directories(${VMA_INCLUDE_DIR})
    LIST(APPEND PACKAGE_SOURCE_FILES 
      ${VMA_HEADERS}
    )
    source_group(VMA FILES 
      ${VMA_HEADERS}
    )
  endif()
endmacro()
# this macro is needed for the samples to add this package, although not needed
# this happens when the shared_sources library was built with these stuff in it
# so many samples can share the same library for many purposes
macro(_optional_package_VMA)
  if(VMA_FOUND)
    _add_package_VMA()
  endif(VMA_FOUND)
endmacro(_optional_package_VMA)

#####################################################################################
# OculusSDK package
#
macro(_add_package_OculusSDK)
  Message(STATUS "--> using package OculusSDK")
  find_package(OculusSDK)
  if(OCULUSSDK_FOUND)
    add_definitions(-DNVP_SUPPORTS_OCULUSSDK)
    include_directories(${OCULUSSDK_INCLUDE_DIRS})
    LIST(APPEND LIBRARIES_OPTIMIZED ${OCULUSSDK_LIBS})
    LIST(APPEND LIBRARIES_DEBUG ${OCULUSSDK_LIBS_DEBUG})
  endif()
endmacro()
# this macro is needed for the samples to add this package, although not needed
# this happens when the shared_sources library was built with these stuff in it
# so many samples can share the same library for many purposes
macro(_optional_package_OculusSDK)
  if(OCULUSSDK_FOUND)
    _add_package_OculusSDK()
  endif(OCULUSSDK_FOUND)
endmacro(_optional_package_OculusSDK)

#####################################################################################
# OpenVRSDK package
#
macro(_add_package_OpenVRSDK)
  Message(STATUS "--> using package OpenVRSDK")
  find_package(OpenVRSDK)
  if(OPENVRSDK_FOUND)
    add_definitions(-DNVP_SUPPORTS_OPENVRSDK)
    include_directories(${OPENVRSDK_INCLUDE_DIRS})
    LIST(APPEND LIBRARIES_OPTIMIZED ${OPENVRSDK_LIBS})
    LIST(APPEND LIBRARIES_DEBUG ${OPENVRSDK_LIBS})
  endif()
endmacro()
# this macro is needed for the samples to add this package, although not needed
# this happens when the shared_sources library was built with these stuff in it
# so many samples can share the same library for many purposes
macro(_optional_package_OpenVRSDK)
  if(OPENVRSDK_FOUND)
    _add_package_OpenVRSDK()
  endif(OPENVRSDK_FOUND)
endmacro(_optional_package_OpenVRSDK)

#####################################################################################
# package for Sockets: to allow UDP/TCP IP connections
#
macro(_add_package_Sockets)
  Message(STATUS "--> using package Sockets")
  set(SOCKETS_PATH "${BASE_DIRECTORY}/shared_sources/nvsockets")
  get_directory_property(hasParent PARENT_DIRECTORY)
  if(hasParent)
    set( USING_SOCKETS "YES" PARENT_SCOPE) # PARENT_SCOPE important to have this variable passed to parent. Here we want to notify that something used the Vulkan package
  else()
    set( USING_SOCKETS "YES")
  endif()
  add_definitions(-DNVP_SUPPORTS_SOCKETS)
  set(SOCKETS_H
    ${SOCKETS_PATH}/socketclient.hpp
    ${SOCKETS_PATH}/socketsamplemessages.hpp
    ${SOCKETS_PATH}/socketserver.hpp
    # ${SOCKETS_PATH}/nvpwindow_socket.hpp
    ${SOCKETS_PATH}/cthread_s.hpp
  )
  set(SOCKETS_CPP
    ${SOCKETS_PATH}/socketclient.cpp
    ${SOCKETS_PATH}/socketserver.cpp
    ${SOCKETS_PATH}/socketsamplemessages.cpp
    ${SOCKETS_PATH}/cthread_s.cpp
  )
  source_group(sockets FILES ${SOCKETS_H})
if(WIN32)
  LIST(APPEND LIBRARIES_OPTIMIZED ws2_32 )
  LIST(APPEND LIBRARIES_DEBUG ws2_32 )
  #TODO: for Linux and Android, too !
endif()
  LIST(APPEND PACKAGE_SOURCE_FILES ${SOCKETS_H} )
  # source_group(Sockets FILES ${SOCKETS_CPP})
  # LIST(APPEND PACKAGE_SOURCE_FILES ${SOCKETS_CPP} )
  include_directories(${SOCKETS_PATH})
endmacro(_add_package_Sockets)

# for the shared_sources library
macro(_optional_package_Sockets)
  if(USING_SOCKETS)
    Message("NOTE: Package for remote control via Sockets is ON")
    _add_package_Sockets()
    source_group(Sockets FILES ${SOCKETS_CPP})
    LIST(APPEND PACKAGE_SOURCE_FILES ${SOCKETS_CPP} )
  endif(USING_SOCKETS)
endmacro(_optional_package_Sockets)

#####################################################################################
# Optional OptiX package
#
macro(_add_package_Optix)
  find_package(Optix)  
  if(OPTIX_FOUND)
      Message(STATUS "--> using package OptiX")
      add_definitions(-DNVP_SUPPORTS_OPTIX)
      include_directories(${OPTIX_INCLUDE_DIR})
      LIST(APPEND LIBRARIES_OPTIMIZED ${OPTIX_LIB} )
      LIST(APPEND LIBRARIES_DEBUG ${OPTIX_LIB} )
      LIST(APPEND PACKAGE_SOURCE_FILES ${OPTIX_HEADERS} )
      source_group(OPTIX FILES  ${OPTIX_HEADERS} )
      set( USING_OPTIX "YES")
 else()
     Message(STATUS "--> NOT using package OptiX")
 endif()
endmacro()
# this macro is needed for the samples to add this package, although not needed
# this happens when the shared_sources library was built with these stuff in it
# so many samples can share the same library for many purposes
macro(_optional_package_Optix)
  if(OPTIX_FOUND)
    _add_package_Optix()
  endif(OPTIX_FOUND)
endmacro(_optional_package_Optix)

#####################################################################################
# Optional VulkanSDK package
#
macro(_add_package_VulkanSDK)
  find_package(VulkanSDK REQUIRED)  
  if(VULKANSDK_FOUND)
      Message(STATUS "--> using package VulkanSDK (version ${VULKANSDK_VERSION})")
      get_directory_property(hasParent PARENT_DIRECTORY)
      if(hasParent)
        set( USING_VULKANSDK "YES" PARENT_SCOPE) # PARENT_SCOPE important to have this variable passed to parent. Here we want to notify that something used the Vulkan package
      endif()
      set( USING_VULKANSDK "YES")
      add_definitions(-DNVP_SUPPORTS_VULKANSDK)
      add_definitions(-DGLFW_INCLUDE_VULKAN)
      add_definitions(-DVK_ENABLE_BETA_EXTENSIONS)

      set(VULKAN_HEADERS_OVERRIDE_INCLUDE_DIR CACHE PATH "Override for Vulkan headers, leave empty to use SDK")

      if (VULKAN_HEADERS_OVERRIDE_INCLUDE_DIR)
        set(vulkanHeaderDir ${VULKAN_HEADERS_OVERRIDE_INCLUDE_DIR})
      else()
        set(vulkanHeaderDir ${VULKANSDK_INCLUDE_DIR})
      endif()

      Message(STATUS "--> using Vulkan Headers from: ${vulkanHeaderDir}")
      include_directories(${vulkanHeaderDir})
      file(GLOB vulkanHeaderFiles "${vulkanHeaderDir}/vulkan/vulkan.h")
      LIST(APPEND PACKAGE_SOURCE_FILES ${vulkanHeaderFiles} )
      source_group(Vulkan FILES  ${vulkanHeaderFiles} )

      LIST(APPEND LIBRARIES_OPTIMIZED ${VULKAN_LIB} )
      LIST(APPEND LIBRARIES_DEBUG ${VULKAN_LIB} )
      # for precompiled headers:
      set(VULKAN_HEADERS "<vulkan/vulkan_core.h>")
 else()
     Message(STATUS "--> NOT using package VulkanSDK")
 endif()
endmacro()
# this macro is needed for the samples to add this package, although not needed
# this happens when the shared_sources library was built with these stuff in it
# so many samples can share the same library for many purposes
macro(_optional_package_VulkanSDK)
  if(USING_VULKANSDK)
    _add_package_VulkanSDK()
  endif(USING_VULKANSDK)
endmacro(_optional_package_VulkanSDK)

#####################################################################################
# Optional ShaderC package
#
macro(_add_package_ShaderC)
  find_package(VulkanSDK)  
  if(VULKANSDK_FOUND AND (VULKANSDK_SHADERC_LIB OR NVSHADERC_LIB))
      Message(STATUS "--> using package ShaderC")
      
      add_definitions(-DNVP_SUPPORTS_SHADERC)
      if (NVSHADERC_LIB)
        Message(STATUS "--> using NVShaderC LIB")
        add_definitions(-DNVP_SUPPORTS_NVSHADERC)
      endif()
      
      if(WIN32)
        add_definitions(-DSHADERC_SHAREDLIB)
        if (NOT VULKANSDK_SHADERC_DLL)
          message(FATAL_ERROR "Windows platform requires VulkanSDK with shaderc_shared.lib/dll (since SDK 1.2.135.0)")  
        endif()
      endif()
      
      if (NVSHADERC_LIB)
        LIST(APPEND LIBRARIES_OPTIMIZED ${NVSHADERC_LIB})
        LIST(APPEND LIBRARIES_DEBUG ${NVSHADERC_LIB})
      else()
        LIST(APPEND LIBRARIES_OPTIMIZED ${VULKANSDK_SHADERC_LIB})
        LIST(APPEND LIBRARIES_DEBUG ${VULKANSDK_SHADERC_LIB})
      endif()
      if(hasParent)
        set( USING_SHADERC "YES" PARENT_SCOPE)
      else()
        set( USING_SHADERC "YES")
      endif()
  else()
      Message(STATUS "--> NOT using package ShaderC")
  endif() 
endmacro(_add_package_ShaderC)
# this macro is needed for the samples to add this package, although not needed
# this happens when the shared_sources library was built with these stuff in it
# so many samples can share the same library for many purposes
macro(_optional_package_ShaderC)
  if(USING_SHADERC)
    _add_package_ShaderC()
  endif(USING_SHADERC)
endmacro(_optional_package_ShaderC)

#####################################################################################
# Optional DirectX11 package
#
macro(_add_package_DirectX11)
  find_package(DirectX11)  
  if(DX11SDK_FOUND)
      Message(STATUS "--> using package DirectX 11")
      get_directory_property(hasParent PARENT_DIRECTORY)
      if(hasParent)
        set( USING_DIRECTX11 "YES" PARENT_SCOPE) # PARENT_SCOPE important to have this variable passed to parent. Here we want to notify that something used the DX11 package
      else()
        set( USING_DIRECTX11 "YES")
      endif()
      add_definitions(-DNVP_SUPPORTS_DIRECTX11)
      include_directories(${DX11SDK_INCLUDE_DIR})
      LIST(APPEND LIBRARIES_OPTIMIZED ${DX11SDK_D3D_LIBRARIES})
      LIST(APPEND LIBRARIES_DEBUG ${DX11SDK_D3D_LIBRARIES})
 else()
     Message(STATUS "--> NOT using package DirectX11")
 endif()
endmacro()
# this macro is needed for the samples to add this package, although not needed
# this happens when the shared_sources library was built with these stuff in it
# so many samples can share the same library for many purposes
macro(_optional_package_DirectX11)
  if(USING_DIRECTX11)
    _add_package_DirectX11()
  endif(USING_DIRECTX11)
endmacro(_optional_package_DirectX11)

#####################################################################################
# Optional DirectX12 package
#
macro(_add_package_DirectX12)
  find_package(DirectX12)  
  if(DX12SDK_FOUND)
      Message(STATUS "--> using package DirectX 12")
      get_directory_property(hasParent PARENT_DIRECTORY)
      if(hasParent)
        set( USING_DIRECTX12 "YES" PARENT_SCOPE) # PARENT_SCOPE important to have this variable passed to parent. Here we want to notify that something used the DX12 package
      else()
        set( USING_DIRECTX12 "YES")
      endif()
      add_definitions(-DNVP_SUPPORTS_DIRECTX12)
      include_directories(${DX12SDK_INCLUDE_DIR})
      include_directories(${BASE_DIRECTORY}/shared_external/d3d12/include)
      LIST(APPEND LIBRARIES_OPTIMIZED ${DX12SDK_D3D_LIBRARIES})
      LIST(APPEND LIBRARIES_DEBUG ${DX12SDK_D3D_LIBRARIES})
 else()
     Message(STATUS "--> NOT using package DirectX12")
 endif()
endmacro()
# this macro is needed for the samples to add this package, although not needed
# this happens when the shared_sources library was built with these stuff in it
# so many samples can share the same library for many purposes
macro(_optional_package_DirectX12)
  if(USING_DIRECTX12)
    _add_package_DirectX12()
  endif(USING_DIRECTX12)
endmacro(_optional_package_DirectX12)

#####################################################################################
# Optional FT-IZB package
#
macro(_add_package_Ftizb)
  find_package(FTIZB)  
  if(FTIZB_FOUND)
      Message(STATUS "--> using package FTIZB")
      add_definitions(-DNVP_SUPPORTS_FTIZB)
      include_directories(${FTIZB_INCLUDE_DIR})
      LIST(APPEND LIBRARIES_OPTIMIZED ${FTIZB_LIB_RELEASE} )
      LIST(APPEND LIBRARIES_DEBUG ${FTIZB_LIB_DEBUG} )
      LIST(APPEND PACKAGE_SOURCE_FILES ${FTIZB_HEADERS} )    
      source_group(FTIZB FILES ${FTIZB_HEADERS} )  
 else()
     Message(STATUS "--> NOT using package OptiX") 
 endif()
endmacro()
# this macro is needed for the samples to add this package, although not needed
# this happens when the shared_sources library was built with these stuff in it
# so many samples can share the same library for many purposes
macro(_optional_package_Ftizb)
  if(USING_FTIZB)
    _add_package_Ftizb()
  endif(USING_FTIZB)
endmacro(_optional_package_Ftizb)

#####################################################################################
# Optional CUDA package
# see https://cmake.org/cmake/help/v3.3/module/FindCUDA.html
#
macro(_add_package_Cuda)
  find_package(CUDA QUIET)
  if(CUDA_FOUND)
      add_definitions("-DCUDA_PATH=R\"(${CUDA_TOOLKIT_ROOT_DIR})\"")
      Message(STATUS "--> using package CUDA (${CUDA_VERSION})")
      add_definitions(-DNVP_SUPPORTS_CUDA)
      include_directories(${CUDA_INCLUDE_DIRS})
      LIST(APPEND LIBRARIES_OPTIMIZED ${CUDA_LIBRARIES} )
      LIST(APPEND LIBRARIES_DEBUG ${CUDA_LIBRARIES} )
      # STRANGE: default CUDA package finder from cmake doesn't give anything to find cuda.lib
      if(WIN32)
        if((ARCH STREQUAL "x86"))
          LIST(APPEND LIBRARIES_DEBUG "${CUDA_TOOLKIT_ROOT_DIR}/lib/Win32/cuda.lib" )
          LIST(APPEND LIBRARIES_DEBUG "${CUDA_TOOLKIT_ROOT_DIR}/lib/Win32/cudart.lib" )
          LIST(APPEND LIBRARIES_OPTIMIZED "${CUDA_TOOLKIT_ROOT_DIR}/lib/Win32/cuda.lib" )
          LIST(APPEND LIBRARIES_OPTIMIZED "${CUDA_TOOLKIT_ROOT_DIR}/lib/Win32/cudart.lib" )
        else()
          LIST(APPEND LIBRARIES_DEBUG "${CUDA_TOOLKIT_ROOT_DIR}/lib/x64/cuda.lib" )
          LIST(APPEND LIBRARIES_DEBUG "${CUDA_TOOLKIT_ROOT_DIR}/lib/x64/cudart.lib" )
          LIST(APPEND LIBRARIES_DEBUG "${CUDA_TOOLKIT_ROOT_DIR}/lib/x64/nvrtc.lib" )
          LIST(APPEND LIBRARIES_OPTIMIZED "${CUDA_TOOLKIT_ROOT_DIR}/lib/x64/cuda.lib" )
          LIST(APPEND LIBRARIES_OPTIMIZED "${CUDA_TOOLKIT_ROOT_DIR}/lib/x64/cudart.lib" )
          LIST(APPEND LIBRARIES_OPTIMIZED "${CUDA_TOOLKIT_ROOT_DIR}/lib/x64/nvrtc.lib" )
        endif()
      else()
        LIST(APPEND LIBRARIES_DEBUG "libcuda.so" )
        LIST(APPEND LIBRARIES_DEBUG "${CUDA_TOOLKIT_ROOT_DIR}/lib64/libcudart.so" )
        LIST(APPEND LIBRARIES_DEBUG "${CUDA_TOOLKIT_ROOT_DIR}/lib64/libnvrtc.so" )
        LIST(APPEND LIBRARIES_OPTIMIZED "libcuda.so" )
        LIST(APPEND LIBRARIES_OPTIMIZED "${CUDA_TOOLKIT_ROOT_DIR}/lib64/libcudart.so" )
        LIST(APPEND LIBRARIES_OPTIMIZED "${CUDA_TOOLKIT_ROOT_DIR}/lib64/libnvrtc.so" )
      endif()
      #LIST(APPEND PACKAGE_SOURCE_FILES ${CUDA_HEADERS} ) Not available anymore with cmake 3.3... we might have to list them by hand
      # source_group(CUDA FILES ${CUDA_HEADERS} )  Not available anymore with cmake 3.3
 else()
     Message(STATUS "--> NOT using package CUDA") 
 endif()
endmacro()
# this macro is needed for the samples to add this package, although not needed
# this happens when the shared_sources library was built with these stuff in it
# so many samples can share the same library for many purposes
macro(_optional_package_Cuda)
  if(CUDA_FOUND)
    _add_package_Cuda()
  endif(CUDA_FOUND)
endmacro(_optional_package_Cuda)

#####################################################################################
# Optional OpenCL package
#
macro(_add_package_OpenCL)
  find_package(OpenCL QUIET)  
  if(OpenCL_FOUND)
      Message(STATUS "--> using package OpenCL : ${OpenCL_LIBRARIES}")
      add_definitions(-DNVP_SUPPORTS_OPENCL)
      include_directories(${OpenCL_INCLUDE_DIRS})
      # just do the copy only if we pointed to the local OpenCL package
      string(FIND ${OpenCL_INCLUDE_DIRS} "shared_external" OFFSET)
      if((OFFSET GREATER -1) AND WIN32 )
        if((ARCH STREQUAL "x86"))
          #set(OPENCL_DLL ${BASE_DIRECTORY}/shared_external/OpenCL/lib/x86/OpenCL.dll)
        else()
          #set(OPENCL_DLL ${BASE_DIRECTORY}/shared_external/OpenCL/lib/x64/OpenCL.dll)
        endif()
      endif()
      LIST(APPEND LIBRARIES_OPTIMIZED ${OpenCL_LIBRARIES} )
      LIST(APPEND LIBRARIES_DEBUG ${OpenCL_LIBRARIES} )
 else()
     Message(STATUS "--> NOT using package OpenCL") 
 endif()
endmacro()
# this macro is needed for the samples to add this package, although not needed
# this happens when the shared_sources library was built with these stuff in it
# so many samples can share the same library for many purposes
macro(_optional_package_OpenCL)
  if(OpenCL_FOUND)
    _add_package_OpenCL()
  endif(OpenCL_FOUND)
endmacro(_optional_package_OpenCL)

#####################################################################################
# NSight
#
# still need the include directory when no use of NSIGHT: for empty #defines
macro(_add_package_NSight)
  Message(STATUS "--> using package NSight")
  set(USING_NSIGHT)
  include_directories(
      ${BASE_DIRECTORY}/shared_external/NSight
  )
  if(SUPPORT_NVTOOLSEXT)
    link_directories(
        ${BASE_DIRECTORY}/shared_external/NSight
    )
    LIST(APPEND PACKAGE_SOURCE_FILES 
      ${BASE_DIRECTORY}/shared_sources/nvh/nsightevents.h
      ${BASE_DIRECTORY}/shared_external/NSight/nvToolsExt.h
    )
    add_definitions(-DNVP_SUPPORTS_NVTOOLSEXT)
    if(ARCH STREQUAL "x86")
      set(NSIGHT_DLL ${BASE_DIRECTORY}/shared_external/NSight/nvToolsExt32_1.dll)
      set(NSIGHT_LIB ${BASE_DIRECTORY}/shared_external/NSight/nvToolsExt32_1.lib)
    else()
      set(NSIGHT_DLL ${BASE_DIRECTORY}/shared_external/NSight/nvToolsExt64_1.dll)
      set(NSIGHT_LIB ${BASE_DIRECTORY}/shared_external/NSight/nvToolsExt64_1.lib)
    endif()
    LIST(APPEND LIBRARIES_OPTIMIZED ${NSIGHT_LIB})
    LIST(APPEND LIBRARIES_DEBUG ${NSIGHT_LIB})
  endif()
endmacro()
# this macro is needed for the samples to add this package, although not needed
# this happens when the shared_sources library was built with these stuff in it
# so many samples can share the same library for many purposes
macro(_optional_package_NSight)
  if(USING_NSIGHT)
    _add_package_NSight()
  endif(USING_NSIGHT)
endmacro(_optional_package_NSight)


#####################################################################################
# NVML
# Note that NVML often needs to be delay-loaded on Windows using
# set_target_properties(${PROJNAME} PROPERTIES LINK_FLAGS "/DELAYLOAD:nvml.dll")!
#
macro(_add_package_NVML)
  message(STATUS "--> using package NVML")
  find_package(NVML)
  if(NVML_FOUND)
    add_definitions(-DNVP_SUPPORTS_NVML)
    include_directories(${NVML_INCLUDE_DIRS})
    LIST(APPEND LIBRARIES_OPTIMIZED ${NVML_LIBRARIES})
    LIST(APPEND LIBRARIES_DEBUG ${NVML_LIBRARIES})
  endif()
endmacro()


#####################################################################################
# Generate PTX files
# NVCUDA_COMPILE_PTX( SOURCES file1.cu file2.cu TARGET_PATH <path where ptxs should be stored> GENERATED_FILES ptx_sources NVCC_OPTIONS -arch=sm_20)
# Generates ptx files for the given source files. ptx_sources will contain the list of generated files.
function(nvcuda_compile_ptx)
  set(options "")
  set(oneValueArgs TARGET_PATH GENERATED_FILES)
  set(multiValueArgs NVCC_OPTIONS SOURCES)
  CMAKE_PARSE_ARGUMENTS(NVCUDA_COMPILE_PTX "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
  
Message(STATUS "NVCUDA_COMPILE_PTX ${options} ${oneValueArgs} ${multiValueArgs} ")

  # Match the bitness of the ptx to the bitness of the application
  set( MACHINE "--machine=32" )
  if( CMAKE_SIZEOF_VOID_P EQUAL 8)
    set( MACHINE "--machine=64" )
  endif()
  
  # Custom build rule to generate ptx files from cuda files
  FOREACH( input ${NVCUDA_COMPILE_PTX_SOURCES} )
    get_filename_component( input_we ${input} NAME_WE )
    
    # generate the *.ptx files inside "ptx" folder inside the executable's output directory.
    set( output "${NVCUDA_COMPILE_PTX_TARGET_PATH}/${input_we}.ptx" )
    LIST( APPEND PTX_FILES  ${output} )
    
    message(STATUS "${CUDA_NVCC_EXECUTABLE} ${MACHINE} --ptx ${NVCUDA_COMPILE_PTX_NVCC_OPTIONS} ${input} -o ${output} WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}")
    
    add_custom_command(
      OUTPUT  ${output}
      DEPENDS ${input}
      COMMAND ${CUDA_NVCC_EXECUTABLE} ${MACHINE} --ptx ${NVCUDA_COMPILE_PTX_NVCC_OPTIONS} ${input} -o ${output} WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
      COMMAND ${CMAKE_COMMAND} -E echo ${NVCUDA_COMPILE_PTX_TARGET_PATH}
    )
  ENDFOREACH( )
  
  set(${NVCUDA_COMPILE_PTX_GENERATED_FILES} ${PTX_FILES} PARENT_SCOPE)
endfunction()

#####################################################################################
# Generate CUBIN files
# NVCUDA_COMPILE_CUBIN( SOURCES file1.cu file2.cu TARGET_PATH <path where cubin's should be stored> GENERATED_FILES cubin_sources NVCC_OPTIONS -arch=sm_20)
# Generates cubin files for the given source files. cubin_sources will contain the list of generated files.
function(nvcuda_compile_cubin)
  set(options "")
  set(oneValueArgs TARGET_PATH GENERATED_FILES)
  set(multiValueArgs NVCC_OPTIONS SOURCES)
  CMAKE_PARSE_ARGUMENTS(NVCUDA_COMPILE_CUBIN "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
  
Message(STATUS "NVCUDA_COMPILE_CUBIN ${options} ${oneValueArgs} ${multiValueArgs} ")

  # Match the bitness of the cubin to the bitness of the application
  set( MACHINE "--machine=32" )
  if( CMAKE_SIZEOF_VOID_P EQUAL 8)
    set( MACHINE "--machine=64" )
  endif()
  
  # Custom build rule to generate cubin files from cuda files
  FOREACH( input ${NVCUDA_COMPILE_CUBIN_SOURCES} )
    get_filename_component( input_we ${input} NAME_WE )
    
    # generate the *.cubin files inside "cubin" folder inside the executable's output directory.
    set( output "${NVCUDA_COMPILE_CUBIN_TARGET_PATH}/${input_we}.cubin" )
    LIST( APPEND CUBIN_FILES  ${output} )
    
    message(STATUS "${CUDA_NVCC_EXECUTABLE} ${MACHINE} --cubin ${NVCUDA_COMPILE_CUBIN_NVCC_OPTIONS} ${input} -o ${CMAKE_CURRENT_SOURCE_DIR}/${output}")
    message(STATUS "WORKING_DIRECTORY: ${CMAKE_CURRENT_SOURCE_DIR}")
    
    add_custom_command(
      OUTPUT  ${CMAKE_CURRENT_SOURCE_DIR}/${output}
      DEPENDS ${input}
      COMMAND ${CUDA_NVCC_EXECUTABLE} ${MACHINE} --cubin ${NVCUDA_COMPILE_CUBIN_NVCC_OPTIONS} ${input} -o ${CMAKE_CURRENT_SOURCE_DIR}/${output} 
      WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    )
  ENDFOREACH( )
  
  set(${NVCUDA_COMPILE_CUBIN_GENERATED_FILES} ${CUBIN_FILES} PARENT_SCOPE)
endfunction()

#####################################################################################
# Macro to setup output directories 
macro(_set_target_output _PROJNAME)
  set_target_properties(${_PROJNAME}
    PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${OUTPUT_PATH}/$<CONFIG>/"
    LIBRARY_OUTPUT_DIRECTORY "${OUTPUT_PATH}/$<CONFIG>/"
    RUNTIME_OUTPUT_DIRECTORY "${OUTPUT_PATH}/$<CONFIG>/"
  )
endmacro()

#####################################################################################
# Macro that copies various binaries that need to be close to the exe files
#
macro(_finalize_target _PROJNAME)

  _set_target_output(${_PROJNAME})
  
  if(SUPPORT_NVTOOLSEXT)
    _copy_files_to_target( ${_PROJNAME} "${NSIGHT_DLL}")
    install(FILES ${NSIGHT_DLL} CONFIGURATIONS Release DESTINATION bin_${ARCH})
    install(FILES ${NSIGHT_DLL} CONFIGURATIONS Debug DESTINATION bin_${ARCH}_debug)
  endif()

  if(NOT UNIX)
    if(USING_SHADERC AND VULKANSDK_SHADERC_DLL)
      _copy_files_to_target( ${_PROJNAME} "${VULKANSDK_SHADERC_DLL}")
      install(FILES ${VULKANSDK_SHADERC_DLL} CONFIGURATIONS Release DESTINATION bin_${ARCH})
      install(FILES ${VULKANSDK_SHADERC_DLL} CONFIGURATIONS Debug DESTINATION bin_${ARCH}_debug)
    endif()
  endif()
  if(ANTTWEAKBAR_FOUND)
    _copy_files_to_target( ${_PROJNAME} "${ANTTWEAKBAR_DLL}")
    install(FILES ${ANTTWEAKBAR_DLL} CONFIGURATIONS Release DESTINATION bin_${ARCH})
    install(FILES ${ANTTWEAKBAR_DLL} CONFIGURATIONS Debug DESTINATION bin_${ARCH}_debug)
  endif()
   if(FREEIMAGE_FOUND)
    _copy_files_to_target( ${_PROJNAME} "${FREEIMAGE_DLL}")
    install(FILES ${FREEIMAGE_DLL} CONFIGURATIONS Release DESTINATION bin_${ARCH})
    install(FILES ${FREEIMAGE_DLL} CONFIGURATIONS Debug DESTINATION bin_${ARCH}_debug)
  endif()
  if(OPTIX_FOUND)
    _copy_files_to_target( ${_PROJNAME} "${OPTIX_DLL}")
    install(FILES ${OPTIX_DLL} CONFIGURATIONS Release DESTINATION bin_${ARCH})
    install(FILES ${OPTIX_DLL} CONFIGURATIONS Debug DESTINATION bin_${ARCH}_debug)
  endif()
  if(OPENCL_DLL)
    _copy_files_to_target( ${_PROJNAME} "${OPENCL_DLL}")
    install(FILES ${OPENCL_DLL} CONFIGURATIONS Release DESTINATION bin_${ARCH})
    install(FILES ${OPENCL_DLL} CONFIGURATIONS Debug DESTINATION bin_${ARCH}_debug)
  endif()
  if(PERFWORKS_DLL)
    install(FILES ${PERFWORKS_DLL} CONFIGURATIONS Release DESTINATION bin_${ARCH})
    install(FILES ${PERFWORKS_DLL} CONFIGURATIONS Debug DESTINATION bin_${ARCH}_debug)
  endif()
  install(TARGETS ${_PROJNAME} CONFIGURATIONS Release DESTINATION bin_${ARCH})
  install(TARGETS ${_PROJNAME} CONFIGURATIONS Debug DESTINATION bin_${ARCH}_debug)
endmacro()

#####################################################################################
# Macro to add custom build for SPIR-V, with additional arbitrary glslangvalidator
# flags (run glslangvalidator --help for a list of possible flags).
# Inputs:
# _SOURCE can be more than one file (.vert + .frag)
# _OUTPUT is the .spv file, resulting from the linkage
# _FLAGS are the flags to add to the command line
# Outputs:
# SOURCE_LIST has _SOURCE appended to it
# OUTPUT_LIST has _OUTPUT appended to it
#
macro(_compile_GLSL_flags _SOURCE _OUTPUT _FLAGS SOURCE_LIST OUTPUT_LIST)
  if(NOT DEFINED VULKAN_TARGET_ENV)
    set(VULKAN_TARGET_ENV vulkan1.1)
  endif()
  LIST(APPEND ${SOURCE_LIST} ${_SOURCE})
  LIST(APPEND ${OUTPUT_LIST} ${_OUTPUT})
  if(GLSLANGVALIDATOR)
    set(_COMMAND ${GLSLANGVALIDATOR} --target-env ${VULKAN_TARGET_ENV} -o ${_OUTPUT} ${_FLAGS} ${_SOURCE})
    add_custom_command(
      OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/${_OUTPUT}
      COMMAND echo ${_COMMAND}
      COMMAND ${_COMMAND}
      MAIN_DEPENDENCY ${_SOURCE}
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
      )
  else(GLSLANGVALIDATOR)
    MESSAGE(WARNING "could not find GLSLANGVALIDATOR to compile shaders")
  endif(GLSLANGVALIDATOR)
endmacro()

#####################################################################################
# Macro to add custom build for SPIR-V, with debug information (glslangvalidator's -g flag)
# Inputs:
# _SOURCE can be more than one file (.vert + .frag)
# _OUTPUT is the .spv file, resulting from the linkage
# Outputs:
# SOURCE_LIST has _SOURCE appended to it
# OUTPUT_LIST has _OUTPUT appended to it
#
macro(_compile_GLSL _SOURCE _OUTPUT SOURCE_LIST OUTPUT_LIST)
  _compile_GLSL_flags(${_SOURCE} ${_OUTPUT} "-g" ${SOURCE_LIST} ${OUTPUT_LIST})
endmacro()

#####################################################################################
# Macro to add custom build for SPIR-V, without debug information
# Inputs:
# _SOURCE can be more than one file (.vert + .frag)
# _OUTPUT is the .spv file, resulting from the linkage
# Outputs:
# SOURCE_LIST has _SOURCE appended to it
# OUTPUT_LIST has _OUTPUT appended to it
#
macro(_compile_GLSL_no_debug_info _SOURCE _OUTPUT SOURCE_LIST OUTPUT_LIST)
  _compile_GLSL_flags(${_SOURCE} ${_OUTPUT} "" ${SOURCE_LIST} ${OUTPUT_LIST})
endmacro()

#####################################################################################
# This is the rest of the cmake code that the project needs to call
# used by the samples via _add_shared_sources_lib and by shared_sources
#
macro(_process_shared_cmake_code)
  
  set(PLATFORM_LIBRARIES)
  
  if (USING_DIRECTX11)
    LIST(APPEND PLATFORM_LIBRARIES ${DX11SDK_D3D_LIBRARIES})
  endif()
  
  if (USING_DIRECTX12)
    LIST(APPEND PLATFORM_LIBRARIES ${DX12SDK_D3D_LIBRARIES})
  endif()
  
  if (USING_OPENGL)
    LIST(APPEND PLATFORM_LIBRARIES ${OPENGL_LIBRARY})
  endif()
  
  if (USING_VULKANSDK)
    LIST(APPEND PLATFORM_LIBRARIES ${VULKAN_LIB})
  endif()
  
  set(COMMON_SOURCE_FILES)
  LIST(APPEND COMMON_SOURCE_FILES
      ${BASE_DIRECTORY}/shared_sources/resources.h
      ${BASE_DIRECTORY}/shared_sources/resources.rc
  )
   
  if(UNIX)
    LIST(APPEND PLATFORM_LIBRARIES "Xxf86vm")

	#Work around some obscure bug where samples would crash in std::filesystem::~path() when
	#multiple GCC versions are installed. Details under:
	#https://stackoverflow.com/questions/63902528/program-crashes-when-filesystempath-is-destroyed
	#
	#When the GCC version is less than 9, explicitly link against libstdc++fs to
	#prevent accidentally picking up GCC9's implementation and thus create an ABI
	#incompatibility that results in crashes
	if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9.0)
		 LIST(APPEND PLATFORM_LIBRARIES stdc++fs)
	endif()
  endif()
endmacro(_process_shared_cmake_code)

#####################################################################################
# This is the rest of the cmake code that the samples needs to call in order
# - to add the shared_sources library (needed by any sample)
# - this part will also setup a directory where to find some downloaded resources
# - add optional packages
# - and will then call another shared cmake macro : _process_shared_cmake_code
#
macro(_add_shared_sources_lib)
  #-----------------------------------------------------------------------------------
  # now we have added some packages, we can guess more
  # on what is needed overall for the shared library

  # build_all adds individual samples, and then at the end 
  # the shared_sources itself, otherwise we build a single 
  # sample which does need shared_sources added here

  if(NOT HAS_SHARED_SOURCES)
    add_subdirectory(${BASE_DIRECTORY}/shared_sources ${CMAKE_BINARY_DIR}/shared_sources)
  endif()
  #-----------------------------------------------------------------------------------
  # optional packages we don't need, but might be needed by other samples
  Message(STATUS " Packages needed for shared_sources lib compat:")
  if(USING_OPENGL OR NOT OPENGL_FOUND)
    _optional_package_OpenGL()
  endif()
  if(USING_VULKANSDK OR NOT VULKANSDK_FOUND)
    _optional_package_VulkanSDK()
  endif()
  # if socket system required in samples, add the package
  if(SUPPORT_SOCKETS)
    Message("NOTE: Package for remote control via Sockets is ON")
    _add_package_Sockets()
  endif()
  # finish with another part (also used by cname for the shared_sources)
  _process_shared_cmake_code()
  # putting this into one of the other branches didn't work
  if(WIN32)
    add_definitions(-DVK_USE_PLATFORM_WIN32_KHR)
  else()
    add_definitions(-DVK_USE_PLATFORM_XLIB_KHR)
    add_definitions(-DVK_USE_PLATFORM_XCB_KHR)
  endif(WIN32)
  add_definitions(-DVULKAN_HPP_DISPATCH_LOADER_DYNAMIC=1)
endmacro()

#####################################################################################
# The OpenMP find macro. does not support non-default CMAKE_EXECUTABLE_SUFFIX
# properly. Workaround the issue by temporarily restoring the default value
# while the FindOpenMP script runs.

macro(_find_package_OpenMP)
  if(UNIX)
    set(EXE_SUFFIX ${CMAKE_EXECUTABLE_SUFFIX})
    unset(CMAKE_EXECUTABLE_SUFFIX)
    find_package(OpenMP)
    set(CMAKE_EXECUTABLE_SUFFIX ${EXE_SUFFIX})
  else()
    find_package(OpenMP)
  endif(UNIX)
endmacro()



# -------------------------------------------------------------------------------------------------
# function that copies a list of files into the target directory
#
#   target_copy_to_output_dir(TARGET foo
#       [RELATIVE <path_prefix>]                                # allows to keep the folder structure starting from this level
#       FILES <absolute_file_path> [<absolute_file_path>]
#       )
#
function(target_copy_to_output_dir)
    set(options)
    set(oneValueArgs TARGET RELATIVE DEST_SUBFOLDER)
    set(multiValueArgs FILES)
    cmake_parse_arguments(TARGET_COPY_TO_OUTPUT_DIR "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN} )

    foreach(_ELEMENT ${TARGET_COPY_TO_OUTPUT_DIR_FILES} )

        # handle absolute and relative paths
        if(TARGET_COPY_TO_OUTPUT_DIR_RELATIVE)
            set(_SOURCE_FILE ${TARGET_COPY_TO_OUTPUT_DIR_RELATIVE}/${_ELEMENT})
            set(_FOLDER_PATH ${_ELEMENT})
        else()
            set(_SOURCE_FILE ${_ELEMENT})
            get_filename_component(_FOLDER_PATH ${_ELEMENT} NAME)
            set (_ELEMENT "")
        endif()

        # handle directories and files slightly different
        if(IS_DIRECTORY ${_SOURCE_FILE})
            if(MDL_LOG_FILE_DEPENDENCIES)
                MESSAGE(STATUS "- folder to copy: ${_SOURCE_FILE}")
            endif()
            add_custom_command(
                TARGET ${TARGET_COPY_TO_OUTPUT_DIR_TARGET} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_directory ${_SOURCE_FILE} $<TARGET_FILE_DIR:${TARGET_COPY_TO_OUTPUT_DIR_TARGET}>/${TARGET_COPY_TO_OUTPUT_DIR_DEST_SUBFOLDER}${_FOLDER_PATH}
            )
        else()   
            if(MDL_LOG_FILE_DEPENDENCIES)
                MESSAGE(STATUS "- file to copy:   ${_SOURCE_FILE}")
            endif()
            add_custom_command(
                TARGET ${TARGET_COPY_TO_OUTPUT_DIR_TARGET} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different ${_SOURCE_FILE} $<TARGET_FILE_DIR:${TARGET_COPY_TO_OUTPUT_DIR_TARGET}>/${TARGET_COPY_TO_OUTPUT_DIR_DEST_SUBFOLDER}${_ELEMENT}
            )
        endif()
    endforeach()
endfunction()
