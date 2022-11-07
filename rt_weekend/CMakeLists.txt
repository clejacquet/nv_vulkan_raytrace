cmake_minimum_required(VERSION 3.9.6 FATAL_ERROR)
project(rt_weekend)

#--------------------------------------------------------------------------------------------------
# look for shared_sources 1) as a sub-folder 2) at some other locations
# this cannot be put anywhere else since we still didn't find setup.cmake yet
if(NOT BASE_DIRECTORY)

  find_path(BASE_DIRECTORY
    NAMES rtnv/cmake/setup.cmake
    PATHS ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/.. ${CMAKE_CURRENT_SOURCE_DIR}/../.. 
    REQUIRED
    DOC "Directory containing shared_sources"
    )
endif()

## Various functions and macros REQUIRED
if(EXISTS ${BASE_DIRECTORY}/rtnv/cmake/setup.cmake)
  include(${BASE_DIRECTORY}/rtnv/cmake/setup.cmake)
  include(${BASE_DIRECTORY}/rtnv/cmake/utilities.cmake)
else()
  message(FATAL_ERROR "could not find base directory, please set BASE_DIRECTORY to folder containing shared_sources")
endif()


#--------------------------------------------------------------------------------------------------
# Package shared by all projects
_add_package_VulkanSDK()
_add_package_ImGUI()
_add_shared_sources_lib()

unset(CMAKE_EXECUTABLE_SUFFIX)


#--------------------------------------------------------------------------------------------------
# Project setting
get_filename_component(PROJNAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)
project(${PROJNAME} LANGUAGES C CXX)
message(STATUS "-------------------------------")
message(STATUS "Processing Project ${PROJNAME}:")


#--------------------------------------------------------------------------------------------------
# C++ target and defines
set(CMAKE_CXX_STANDARD 17)
add_executable(${PROJNAME})
_add_project_definitions(${PROJNAME})


#--------------------------------------------------------------------------------------------------
# Source files for this project
#
# file(GLOB SOURCE_FILES src/*.cpp src/*.hpp src/*.inl src/*.h src/*.c src/common/*.cpp src/common/*.h)
file(GLOB_RECURSE SOURCE_FILES 
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp 
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.inl 
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.h 
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.c)


#--------------------------------------------------------------------------------------------------
# GLSL to SPIR-V custom build
compile_glsl_directory(
	SRC "${CMAKE_CURRENT_SOURCE_DIR}/shaders" 
	DST "${CMAKE_CURRENT_SOURCE_DIR}/spv" 
	VULKAN_TARGET "vulkan1.2"
	)


#--------------------------------------------------------------------------------------------------
# Sources
target_sources(${PROJNAME} PUBLIC ${SOURCE_FILES} ${HEADER_FILES})
target_sources(${PROJNAME} PUBLIC ${PACKAGE_SOURCE_FILES})
target_sources(${PROJNAME} PUBLIC ${GLSL_SOURCES} ${GLSL_HEADERS})


#--------------------------------------------------------------------------------------------------
# Sub-folders in Visual Studio
#
source_group("Common"       FILES ${COMMON_SOURCE_FILES} ${PACKAGE_SOURCE_FILES})
source_group("Sources"      FILES ${SOURCE_FILES})
source_group("Headers"      FILES ${HEADER_FILES})
source_group("Shader_Files" FILES ${GLSL_SOURCES} ${GLSL_HEADERS})


#--------------------------------------------------------------------------------------------------
# Linkage
#
target_link_libraries(${PROJNAME} ${PLATFORM_LIBRARIES} rtnv)

foreach(DEBUGLIB ${LIBRARIES_DEBUG})
  target_link_libraries(${PROJNAME} debug ${DEBUGLIB})
endforeach(DEBUGLIB)

foreach(RELEASELIB ${LIBRARIES_OPTIMIZED})
  target_link_libraries(${PROJNAME} optimized ${RELEASELIB})
endforeach(RELEASELIB)

#--------------------------------------------------------------------------------------------------
# copies binaries that need to be put next to the exe files (ZLib, etc.)
#
_finalize_target( ${PROJNAME} )