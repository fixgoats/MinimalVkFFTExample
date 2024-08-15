cmake_minimum_required(VERSION 3.16)

project("Minimal VkFFT example")
add_executable(example "main.cpp")
set(VKFFT_BACKEND 0)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_CONFIGURATION_TYPES "Release" CACHE STRING "" FORCE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
endif()

add_definitions(-DVKFFT_BACKEND=${VKFFT_BACKEND})
target_compile_definitions(example PUBLIC -DVK_API_VERSION=11)

find_package(Vulkan REQUIRED)
find_package(glslang REQUIRED)
add_library(VkFFT INTERFACE)
target_include_directories(VkFFT INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/vkFFT/ /usr/include/glslang/Include)
target_link_libraries(VkFFT INTERFACE glslang::MachineIndependent glslang::OSDependent glslang::GenericCodeGen glslang::SPIRV)
target_compile_definitions(VkFFT INTERFACE -DVKFFT_BACKEND=${VKFFT_BACKEND})

target_link_libraries(example PRIVATE SPIRV glslang Vulkan::Vulkan VkFFT)
# install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/vkFFT/" DESTINATION include)
# install(TARGETS ${PROJECT_NAME} DESTINATION bin)
