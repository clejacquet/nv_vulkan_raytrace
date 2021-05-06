/*-----------------------------------------------------------------------
 * Copyright (c) 2018-2019, NVIDIA CORPORATION. All rights reserved.
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

#ifndef __NVPWINDOW_H__
#define __NVPWINDOW_H__

#include "nvpsystem.hpp"

class NVPWindow
{
public:
  // IMPORTANT
  //
  // Using and deriving of NVPWindow base-class is optional.
  // However one must always make use of the NVPSystem
  // That takes care of glfwInit/terminate as well.

  ////////////////////////////////////////////////////////////////////////
  // base class

  // these are taken from GLFW3 and must be kept in a matching state

  enum ButtonAction
  {
    BUTTON_RELEASE = 0,
    BUTTON_PRESS   = 1,
    BUTTON_REPEAT  = 2,
  };

  enum MouseButton
  {
    MOUSE_BUTTON_LEFT   = 0,
    MOUSE_BUTTON_RIGHT  = 1,
    MOUSE_BUTTON_MIDDLE = 2,
    NUM_MOUSE_BUTTONIDX,
  };

  enum MouseButtonFlag
  {
    MOUSE_BUTTONFLAG_NONE   = 0,
    MOUSE_BUTTONFLAG_LEFT   = (1 << MOUSE_BUTTON_LEFT),
    MOUSE_BUTTONFLAG_RIGHT  = (1 << MOUSE_BUTTON_RIGHT),
    MOUSE_BUTTONFLAG_MIDDLE = (1 << MOUSE_BUTTON_MIDDLE)
  };

  enum KeyCode
  {
    KEY_UNKNOWN           = -1,
    KEY_SPACE             = 32,
    KEY_APOSTROPHE        = 39, /* ' */
    KEY_LEFT_PARENTHESIS  = 40, /* ( */
    KEY_RIGHT_PARENTHESIS = 41, /* ) */
    KEY_ASTERISK          = 42, /* * */
    KEY_PLUS              = 43, /* + */
    KEY_COMMA             = 44, /* , */
    KEY_MINUS             = 45, /* - */
    KEY_PERIOD            = 46, /* . */
    KEY_SLASH             = 47, /* / */
    KEY_0                 = 48,
    KEY_1                 = 49,
    KEY_2                 = 50,
    KEY_3                 = 51,
    KEY_4                 = 52,
    KEY_5                 = 53,
    KEY_6                 = 54,
    KEY_7                 = 55,
    KEY_8                 = 56,
    KEY_9                 = 57,
    KEY_SEMICOLON         = 59, /* ; */
    KEY_EQUAL             = 61, /* = */
    KEY_A                 = 65,
    KEY_B                 = 66,
    KEY_C                 = 67,
    KEY_D                 = 68,
    KEY_E                 = 69,
    KEY_F                 = 70,
    KEY_G                 = 71,
    KEY_H                 = 72,
    KEY_I                 = 73,
    KEY_J                 = 74,
    KEY_K                 = 75,
    KEY_L                 = 76,
    KEY_M                 = 77,
    KEY_N                 = 78,
    KEY_O                 = 79,
    KEY_P                 = 80,
    KEY_Q                 = 81,
    KEY_R                 = 82,
    KEY_S                 = 83,
    KEY_T                 = 84,
    KEY_U                 = 85,
    KEY_V                 = 86,
    KEY_W                 = 87,
    KEY_X                 = 88,
    KEY_Y                 = 89,
    KEY_Z                 = 90,
    KEY_LEFT_BRACKET      = 91,  /* [ */
    KEY_BACKSLASH         = 92,  /* \ */
    KEY_RIGHT_BRACKET     = 93,  /* ] */
    KEY_GRAVE_ACCENT      = 96,  /* ` */
    KEY_WORLD_1           = 161, /* non-US #1 */
    KEY_WORLD_2           = 162, /* non-US #2 */

    /* Function keys */
    KEY_ESCAPE        = 256,
    KEY_ENTER         = 257,
    KEY_TAB           = 258,
    KEY_BACKSPACE     = 259,
    KEY_INSERT        = 260,
    KEY_DELETE        = 261,
    KEY_RIGHT         = 262,
    KEY_LEFT          = 263,
    KEY_DOWN          = 264,
    KEY_UP            = 265,
    KEY_PAGE_UP       = 266,
    KEY_PAGE_DOWN     = 267,
    KEY_HOME          = 268,
    KEY_END           = 269,
    KEY_CAPS_LOCK     = 280,
    KEY_SCROLL_LOCK   = 281,
    KEY_NUM_LOCK      = 282,
    KEY_PRINT_SCREEN  = 283,
    KEY_PAUSE         = 284,
    KEY_F1            = 290,
    KEY_F2            = 291,
    KEY_F3            = 292,
    KEY_F4            = 293,
    KEY_F5            = 294,
    KEY_F6            = 295,
    KEY_F7            = 296,
    KEY_F8            = 297,
    KEY_F9            = 298,
    KEY_F10           = 299,
    KEY_F11           = 300,
    KEY_F12           = 301,
    KEY_F13           = 302,
    KEY_F14           = 303,
    KEY_F15           = 304,
    KEY_F16           = 305,
    KEY_F17           = 306,
    KEY_F18           = 307,
    KEY_F19           = 308,
    KEY_F20           = 309,
    KEY_F21           = 310,
    KEY_F22           = 311,
    KEY_F23           = 312,
    KEY_F24           = 313,
    KEY_F25           = 314,
    KEY_KP_0          = 320,
    KEY_KP_1          = 321,
    KEY_KP_2          = 322,
    KEY_KP_3          = 323,
    KEY_KP_4          = 324,
    KEY_KP_5          = 325,
    KEY_KP_6          = 326,
    KEY_KP_7          = 327,
    KEY_KP_8          = 328,
    KEY_KP_9          = 329,
    KEY_KP_DECIMAL    = 330,
    KEY_KP_DIVIDE     = 331,
    KEY_KP_MULTIPLY   = 332,
    KEY_KP_SUBTRACT   = 333,
    KEY_KP_ADD        = 334,
    KEY_KP_ENTER      = 335,
    KEY_KP_EQUAL      = 336,
    KEY_LEFT_SHIFT    = 340,
    KEY_LEFT_CONTROL  = 341,
    KEY_LEFT_ALT      = 342,
    KEY_LEFT_SUPER    = 343,
    KEY_RIGHT_SHIFT   = 344,
    KEY_RIGHT_CONTROL = 345,
    KEY_RIGHT_ALT     = 346,
    KEY_RIGHT_SUPER   = 347,
    KEY_MENU          = 348,
    KEY_LAST          = KEY_MENU,
  };

  enum KeyModifiers
  {
    KMOD_SHIFT   = 1,
    KMOD_CONTROL = 2,
    KMOD_ALT     = 4,
    KMOD_SUPER   = 8,
  };

  //////////////////////////////////////////////////////////////////////////

  struct GLFWwindow* m_internal = nullptr;
  std::string        m_windowName;


  inline bool pollEvents()  // returns false on exit, can do while(pollEvents()){ ... }
  {
    NVPSystem::pollEvents();
    return !isClosing();
  }
  inline void        waitEvents() { NVPSystem::waitEvents(); }
  inline double      getTime() { return NVPSystem::getTime(); }
  inline std::string exePath() { return NVPSystem::exePath(); }

  // Accessors
  inline int getWidth() const { return m_windowSize[0]; }
  inline int getHeight() const { return m_windowSize[1]; }
  inline int getMouseWheel() const { return m_mouseWheel; }
  inline int getKeyModifiers() const { return m_keyModifiers; }
  inline int getMouseX() const { return m_mouseX; }
  inline int getMouseY() const { return m_mouseY; }

  void        setTitle(const char* title);
  void        setFullScreen(bool bYes);
  void        setWindowPos(int x, int y);
  void        setWindowSize(int w, int h);
  inline void setKeyModifiers(int m) { m_keyModifiers = m; }
  inline void setMouse(int x, int y)
  {
    m_mouseX = x;
    m_mouseY = y;
  }

  inline bool isFullScreen() const { return m_isFullScreen; }
  bool        isClosing() const;
  bool        isOpen() const;

  virtual bool open(int posX, int posY, int width, int height, const char* title, bool requireGLContext);  // creates internal window and opens it
  void         deinit();  // destroys internal window

  void close();  //  triggers closing event, still needs deinit for final cleanup
  void maximize();
  void restore();
  void minimize();

  // uses operating system specific code for sake of debugging/automated testing
  void screenshot(const char* filename);
  void clear(uint32_t r, uint32_t g, uint32_t b);

  // simple modal file dialog, uses OS basic api
  // the exts string must be a | separated list that has two items per possible extension
  // "extension descripton|*.ext"
  std::string openFileDialog(const char* title, const char* exts);
  std::string saveFileDialog(const char* title, const char* exts);

  // derived windows/apps should override to handle events
  virtual void onWindowClose() {}
  virtual void onWindowResize(int w, int h) {}
  virtual void onWindowRefresh() {}
  virtual void onMouseMotion(int x, int y) {}
  virtual void onMouseWheel(int delta) {}
  virtual void onMouseButton(MouseButton button, ButtonAction action, int mods, int x, int y) {}
  virtual void onKeyboard(KeyCode key, ButtonAction action, int mods, int x, int y) {}
  virtual void onKeyboardChar(unsigned char key, int mods, int x, int y) {}
  virtual void onDragDrop(int num, const char** paths) {}

  // derived windows/apps should override these. Essentially used for remote-control (via sockets)
  // the decoded remote paquets would invoke these methods. See shared_sources\nvsockets\socketSampleMessages.cpp
  virtual void requestTiming() {
  }  // the app can override it to return requested timing information over sockets : use sysPostTiming() below
  virtual void requestPaint() {}                       // the app needs to refresh once the window
  virtual void requestContinuousRefresh(bool bYes) {}  // the app might swith on/off the continuous rendering
  virtual void requestSetArg(char token, int arg0, int arg1, int arg2, int arg3) {
  }  // the app receives arbitrary params from remote, free of interpretation
  virtual void requestSetArg(char token, float arg0, float arg1, float arg2, float arg3) {
  }  // the app receives arbitrary params from remote, free of interpretation

private:
  int  m_mouseX;
  int  m_mouseY;
  int  m_mouseWheel;
  int  m_windowSize[2];
  int  m_keyModifiers;
  bool m_isFullScreen = false;
  bool m_isClosing    = false;
  int  m_preFullScreenPos[2];
  int  m_preFullScreenSize[2];

  static void cb_windowrefreshfun(GLFWwindow* glfwwin);
  static void cb_windowsizefun(GLFWwindow* glfwwin, int w, int h);
  static void cb_windowclosefun(GLFWwindow* glfwwin);
  static void cb_mousebuttonfun(GLFWwindow* glfwwin, int button, int action, int mods);
  static void cb_cursorposfun(GLFWwindow* glfwwin, double x, double y);
  static void cb_scrollfun(GLFWwindow* glfwwin, double x, double y);
  static void cb_keyfun(GLFWwindow* glfwwin, int key, int scancode, int action, int mods);
  static void cb_charfun(GLFWwindow* glfwwin, unsigned int codepoint);
  static void cb_dropfun(GLFWwindow* glfwwin, int count, const char** paths);
};


#endif
