/*-----------------------------------------------------------------------
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/ //--------------------------------------------------------------------

#include "nvpsystem.hpp"

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#define GLFW_EXPOSE_NATIVE_X11
#include <GLFW/glfw3native.h>

#include <vector>
#include <algorithm>
#include <unistd.h>
#include <stdio.h>
#include <limits.h>
#include <string>
#include <assert.h>

#ifdef NVP_SUPPORTS_SOCKETS
#include "socketSampleMessages.h"
#endif

// from https://docs.microsoft.com/en-us/windows/desktop/gdi/capturing-an-image


void NVPSystem::windowScreenshot(struct GLFWwindow* glfwin, const char* filename)
{
  Window hwnd = glfwGetX11Window(glfwin);
  assert(0 && "not yet implemented");
}

void NVPSystem::windowClear(struct GLFWwindow* glfwin, uint32_t r, uint32_t g, uint32_t b)
{
  Window hwnd = glfwGetX11Window(glfwin);
  assert(0 && "not yet implemented");
}

std::string NVPSystem::windowOpenFileDialog(struct GLFWwindow* glfwin, const char* title, const char* exts)
{
  Window hwnd = glfwGetX11Window(glfwin);
  assert(0 && "not yet implemented");

  return std::string();
}

std::string NVPSystem::windowSaveFileDialog(struct GLFWwindow* glfwin, const char* title, const char* exts)
{
  Window hwnd = glfwGetX11Window(glfwin);
  assert(0 && "not yet implemented");

  return std::string();
}

void NVPSystem::sleep(double seconds)
{
  ::sleep(seconds);
}

void NVPSystem::platformInit()
{
}

void NVPSystem::platformDeinit()
{
}

static std::string s_exePath;
static bool        s_exePathInit = false;

std::string NVPSystem::exePath()
{
  if(!s_exePathInit)
  {
    char modulePath[PATH_MAX];
    ssize_t modulePathLength = readlink( "/proc/self/exe", modulePath, PATH_MAX );

    s_exePath = std::string(modulePath, modulePathLength > 0 ? modulePathLength : 0);

    size_t last = s_exePath.rfind('/');
    if(last != std::string::npos)
    {
      s_exePath = s_exePath.substr(0, last) + std::string("/");
    }

    s_exePathInit = true;
  }

  return s_exePath;
}
