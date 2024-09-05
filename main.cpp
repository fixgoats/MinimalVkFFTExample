#define VMA_IMPLEMENTATION
#include "vk_mem_alloc.h"
#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <iterator>
#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_enums.hpp>
#include <vulkan/vulkan_handles.hpp>
#include <vulkan/vulkan_raii.hpp>
#include <vulkan/vulkan_structs.hpp>

#include "vkFFT.h"
#include <complex>

typedef std::complex<double> c64;
typedef std::complex<float> c32;

template <typename T> constexpr T square(T x) { return x * x; }

vk::raii::Instance makeInstance(const vk::raii::Context &context) {
  vk::ApplicationInfo AppInfo{
      "VulkanCompute",   // Application Name
      1,                 // Application Version
      nullptr,           // Engine Name or nullptr
      0,                 // Engine Version
      VK_API_VERSION_1_3 // Vulkan API version
  };

  const std::vector<const char *> Layers = {"VK_LAYER_KHRONOS_validation"};
  vk::InstanceCreateInfo InstanceCreateInfo(vk::InstanceCreateFlags(), // Flags
                                            &AppInfo, // Application Info
                                            Layers,   // Layers
                                            {});      // Extensions
  return vk::raii::Instance(context, InstanceCreateInfo);
}

template <typename Func>
void oneTimeSubmit(const vk::raii::Device &device,
                   const vk::raii::CommandPool &commandPool,
                   const vk::raii::Queue &queue, const Func &func) {
  vk::raii::CommandBuffer commandBuffer =
      std::move(vk::raii::CommandBuffers(
                    device, {*commandPool, vk::CommandBufferLevel::ePrimary, 1})
                    .front());
  commandBuffer.begin(vk::CommandBufferBeginInfo(
      vk::CommandBufferUsageFlagBits::eOneTimeSubmit));
  func(*commandBuffer);
  commandBuffer.end();
  vk::SubmitInfo submitInfo(nullptr, nullptr, *commandBuffer);
  queue.submit(submitInfo, nullptr);
  queue.waitIdle();
}

vk::raii::PhysicalDevice pickPhysicalDevice(const vk::raii::Instance &instance,
                                            const int32_t desiredGPU = -1) {
  // check if there are GPUs that support Vulkan and "intelligently" select one.
  // Prioritises discrete GPUs, and after that VRAM size.
  vk::raii::PhysicalDevices physicalDevices(instance);
  uint32_t nDevices = physicalDevices.size();

  // shortcut if there's only one device available.
  if (nDevices == 1) {
    return vk::raii::PhysicalDevice(std::move(physicalDevices[0]));
  }
  // Try to select desired GPU if specified.
  if (desiredGPU > -1) {
    if (desiredGPU < static_cast<int32_t>(nDevices)) {
      return vk::raii::PhysicalDevice(std::move(physicalDevices[desiredGPU]));
    } else {
      std::cout << "Device not available\n";
    }
  }

  std::vector<uint32_t> discrete; // the indices of the available discrete gpus
  std::vector<uint64_t> vram(nDevices);
  for (uint32_t i = 0; i < nDevices; i++) {
    if (physicalDevices[i].getProperties().deviceType ==
        vk::PhysicalDeviceType::eDiscreteGpu) {
      discrete.push_back(i);
    }

    auto heaps = physicalDevices[i].getMemoryProperties().memoryHeaps;
    for (const auto &heap : heaps) {
      if (heap.flags & vk::MemoryHeapFlagBits::eDeviceLocal) {
        vram[i] = heap.size;
      }
    }
  }

  // only consider discrete gpus if available
  if (discrete.size() > 0) {
    if (discrete.size() == 1) {
      return vk::raii::PhysicalDevice(std::move(physicalDevices[discrete[0]]));
    } else {
      uint32_t max = 0;
      uint32_t selectedGPU = 0;
      for (const auto &index : discrete) {
        if (vram[index] > max) {
          max = vram[index];
          selectedGPU = index;
        }
      }
      return vk::raii::PhysicalDevice(std::move(physicalDevices[selectedGPU]));
    }
  } else {
    uint32_t max = 0;
    uint32_t selectedGPU = 0;
    for (uint32_t i = 0; i < nDevices; i++) {
      if (vram[i] > max) {
        max = vram[i];
        selectedGPU = i;
      }
    }
    return vk::raii::PhysicalDevice(std::move(physicalDevices[selectedGPU]));
  }
}

int main(int argc, char *argv[]) {
  vk::raii::Context context;
  auto instance = makeInstance(context);
  auto physicalDevice = pickPhysicalDevice(instance);
  auto queueFamilyProps = physicalDevice.getQueueFamilyProperties();
  auto propIt =
      std::find_if(queueFamilyProps.begin(), queueFamilyProps.end(),
                   [](const vk::QueueFamilyProperties &prop) {
                     return prop.queueFlags & vk::QueueFlagBits::eCompute;
                   });
  const uint32_t computeQueueFamilyIndex =
      std::distance(queueFamilyProps.begin(), propIt);

  float queuePriority = 0.0f;
  vk::DeviceQueueCreateInfo dQCI(vk::DeviceQueueCreateFlags(),
                                 computeQueueFamilyIndex, 1, &queuePriority);
  vk::DeviceCreateInfo dCI(vk::DeviceCreateFlags(), dQCI);
  vk::raii::Device device(physicalDevice, dCI);
  vk::raii::Queue queue(device, computeQueueFamilyIndex, 0);
  vk::raii::Fence fence(device, vk::FenceCreateInfo());
  vk::CommandPoolCreateInfo commandPoolCreateInfo(vk::CommandPoolCreateFlags(),
                                                  computeQueueFamilyIndex);
  vk::raii::CommandPool commandPool(device, commandPoolCreateInfo);
  vk::CommandBufferAllocateInfo cBAI(*commandPool,
                                     vk::CommandBufferLevel::ePrimary, 1);
  vk::raii::CommandBuffers commandBuffers(device, cBAI);
  vk::raii::CommandBuffer commandBuffer(std::move(commandBuffers[0]));

  constexpr uint64_t nElements = 1024;
  uint64_t bufferSize = nElements * sizeof(c32);

  VmaAllocatorCreateInfo allocatorInfo{};
  allocatorInfo.physicalDevice = *physicalDevice;
  allocatorInfo.vulkanApiVersion = physicalDevice.getProperties().apiVersion;
  allocatorInfo.device = *device;
  allocatorInfo.instance = *instance;

  VmaAllocator allocator;
  vmaCreateAllocator(&allocatorInfo, &allocator);

  vk::BufferCreateInfo stagingBCI({}, bufferSize,
                                  vk::BufferUsageFlagBits::eTransferSrc |
                                      vk::BufferUsageFlagBits::eTransferDst);

  VmaAllocationCreateInfo allocCreateInfo{};
  allocCreateInfo.usage = VMA_MEMORY_USAGE_AUTO;
  allocCreateInfo.flags =
      VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
      VMA_ALLOCATION_CREATE_MAPPED_BIT;
  VkBuffer stagingBufferRaw;
  VmaAllocation stagingAlloc;
  VmaAllocationInfo stAI;
  vmaCreateBuffer(allocator,
                  reinterpret_cast<VkBufferCreateInfo *>(&stagingBCI),
                  &allocCreateInfo, &stagingBufferRaw, &stagingAlloc, &stAI);
  vk::Buffer stagingBuffer = stagingBufferRaw;

  vk::BufferCreateInfo stateBCI{vk::BufferCreateFlags(),
                                bufferSize,
                                vk::BufferUsageFlagBits::eStorageBuffer |
                                    vk::BufferUsageFlagBits::eTransferDst |
                                    vk::BufferUsageFlagBits::eTransferSrc,
                                vk::SharingMode::eExclusive,
                                1,
                                &computeQueueFamilyIndex};
  allocCreateInfo.flags = VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT;
  allocCreateInfo.priority = 1.0f;
  VkBuffer inBufferRaw;
  VmaAllocation inBufferAlloc;
  VmaAllocationInfo inBufferAI;
  vmaCreateBuffer(allocator, reinterpret_cast<VkBufferCreateInfo *>(&stateBCI),
                  &allocCreateInfo, &inBufferRaw, &inBufferAlloc, &inBufferAI);
  vk::Buffer inBuffer = inBufferRaw;

  VkBuffer outBufferRaw;
  VmaAllocation outBufferAlloc;
  VmaAllocationInfo outBufferAI;
  vmaCreateBuffer(allocator, reinterpret_cast<VkBufferCreateInfo *>(&stateBCI),
                  &allocCreateInfo, &inBufferRaw, &inBufferAlloc, &inBufferAI);
  vk::Buffer outBuffer = outBufferRaw;

  c32 *stagingPtr = static_cast<c32 *>(stAI.pMappedData);
  constexpr float startX = -3.0;
  constexpr float endX = 3.0;
  for (int32_t i = 0; i < nElements; i++) {
    stagingPtr[i] = std::exp(c64{
        -square(startX + ((float)i / (float)nElements) * (endX - startX)), 0.});
  }

  oneTimeSubmit(device, commandPool, queue,
                [&](vk::CommandBuffer const &commandBuffer) {
                  commandBuffer.copyBuffer(stagingBuffer, inBuffer,
                                           vk::BufferCopy(0, 0, bufferSize));
                });

  VkFFTConfiguration conf = {};
  VkFFTApplication app1 = {};
  conf.device = (VkDevice *)&*device;
  conf.FFTdim = 1;
  conf.size[0] = 1024;
  conf.numberBatches = 1;
  conf.queue = (VkQueue *)&*queue;
  conf.fence = (VkFence *)&*fence;
  conf.commandPool = (VkCommandPool *)&*commandPool;
  conf.physicalDevice = (VkPhysicalDevice *)&*physicalDevice;
  conf.isInputFormatted = true;
  conf.inputBuffer = (VkBuffer *)&*inBuffer;
  conf.buffer = (VkBuffer *)&*outBuffer;
  conf.bufferSize = &bufferSize;
  conf.inputBufferSize = &bufferSize;
  conf.inverseReturnToInputBuffer = true;

  auto resFFT = initializeVkFFT(&app1, conf);

  std::cout << resFFT << '\n';
  VkFFTLaunchParams launchParams = {};

  vk::CommandBufferBeginInfo cBBI(
      vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
  commandBuffer.begin(cBBI);
  launchParams.commandBuffer = (VkCommandBuffer *)&*commandBuffer;
  resFFT = VkFFTAppend(&app, -1, &launchParams);
  commandBuffer.end();

  vk::SubmitInfo submitInfo(0, nullptr, nullptr, 1, &*commandBuffer);
  queue.submit(submitInfo, *fence);
  auto result = device.waitForFences({*fence}, true, uint64_t(-1));

  inBufferPtr = static_cast<c32 *>(inBM.mapMemory(0, bufferSize));
  for (uint32_t i = 0; i < nElements; i++) {
    std::cout << inBufferPtr[i] << ' ';
  }
  std::cout << '\n';
  inBM.unmapMemory();
  deleteVkFFT(&app);

  return 0;
}
