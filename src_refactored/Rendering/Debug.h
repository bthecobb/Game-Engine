#ifndef CUDA_GAME_RENDERING_DEBUG_H
#define CUDA_GAME_RENDERING_DEBUG_H

// Define DEBUG_RENDERER for debug builds only
// #define DEBUG_RENDERER  // Disabled for now due to depth buffer reporting issues

#ifdef DEBUG_RENDERER
  #define RENDER_DBG_ENABLED 1
#else
  #define RENDER_DBG_ENABLED 0
#endif

#if RENDER_DBG_ENABLED
  #include <glad/glad.h>
  #include <iostream>
  #include <string>
  #include <chrono>

  inline const char* GlErrorName(GLenum err) {
    switch (err) {
      case GL_NO_ERROR: return "GL_NO_ERROR";
      case GL_INVALID_ENUM: return "GL_INVALID_ENUM";
      case GL_INVALID_VALUE: return "GL_INVALID_VALUE";
      case GL_INVALID_OPERATION: return "GL_INVALID_OPERATION";
      case GL_INVALID_FRAMEBUFFER_OPERATION: return "GL_INVALID_FRAMEBUFFER_OPERATION";
      case GL_OUT_OF_MEMORY: return "GL_OUT_OF_MEMORY";
      default: return "UNKNOWN_ERROR";
    }
  }

  inline void LogGlErrors(const char* where, int frame_id) {
    bool any = false;
    for (GLenum e = glGetError(); e != GL_NO_ERROR; e = glGetError()) {
      any = true;
      std::cout << "{ \"frame\":" << frame_id
                << ",\"GLError\":\"" << where
                << "\",\"code\":" << e
                << ",\"name\":\"" << GlErrorName(e) << "\" }" << std::endl;
    }
    if (any) {
      // Optional: also dump GL state when errors occur
    }
  }

  inline void DumpGLState(const std::string& tag, int frame_id) {
    GLint fbo = 0, program = 0, vp[4] = {0,0,0,0};
    glGetIntegerv(GL_FRAMEBUFFER_BINDING, &fbo);
    glGetIntegerv(GL_CURRENT_PROGRAM, &program);
    glGetIntegerv(GL_VIEWPORT, vp);

    std::cout << "{ \"frame\":" << frame_id
              << ",\"GLState\":\"" << tag
              << "\",\"FBO\":" << fbo
              << ",\"program\":" << program
              << ",\"viewport\":[" << vp[0] << "," << vp[1] << "," << vp[2] << "," << vp[3] << "] }"
              << std::endl;
  }

  #define RENDER_DBG(stmt) do { stmt; } while(0)
  #define GL_CHECK(where, frame_id) do { LogGlErrors(where, frame_id); } while(0)
#else
  inline void LogGlErrors(const char*, int) {}
  inline void DumpGLState(const std::string&, int) {}
  #define RENDER_DBG(stmt) do {} while(0)
  #define GL_CHECK(where, frame_id) do {} while(0)
#endif

#endif // CUDA_GAME_RENDERING_DEBUG_H
