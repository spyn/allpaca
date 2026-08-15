cmake_minimum_required(VERSION 3.18)
project(allpaca_kernels LANGUAGES CXX)

add_library(allpaca_kernels static src/kernels.cpp)
target_include_directories(allpaca_kernels PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(allpaca_kernels PUBLIC cxx_std_17)

if(DEFINED ENV{ROCM_PATH} OR DEFINED HIP_PATH)
  find_package(hip QUIET)
  if(hip_FOUND)
    target_link_libraries(allpaca_kernels PUBLIC hip::host)
    target_compile_definitions(allpaca_kernels PRIVATE __HIP_PLATFORM_AMD__)
  endif()
endif()
