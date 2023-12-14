cmake_minimum_required(VERSION 3.18)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 75 86)
  enable_language(CUDA)
  add_definitions(-DUSE_CUDA)    
  include_directories("${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
endif()

project(raytracer LANGUAGES CXX CUDA)

set(OpenGL_GL_PREFERENCE "LEGACY")
find_package(OpenGL REQUIRED)

if (WIN32)
    set(CMAKE_C_STANDARD_LIBRARIES "opengl32.lib user32.lib gdi32.lib" CACHE STRING "C std libs")
    add_executable(raytracer src/window.h src/raytracer.cpp src/cuda_trace.cu)
    target_link_libraries(raytracer OpenGL::GL)
endif (WIN32)

if (UNIX)
    find_package(X11 REQUIRED)

    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -std=c++1y -Wall -Wextra -Wunused-function")

    add_executable(raytracer src/window.h src/sys_gl.cpp src/sys_gl.h src/raytracer.cpp src/cuda_trace.cu)

    target_link_libraries(raytracer X11)
    target_link_libraries(raytracer OpenGL::GL)
endif (UNIX)

