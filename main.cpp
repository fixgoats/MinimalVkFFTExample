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

  constexpr uint64_t nElements = 1024;
  uint64_t bufferSize = nElements * sizeof(c32);

  vk::raii::DeviceMemory inBM(nullptr);
  vk::BufferCreateInfo bCI{vk::BufferCreateFlags(),
                           bufferSize,
                           vk::BufferUsageFlagBits::eStorageBuffer,
                           vk::SharingMode::eExclusive,
                           1,
                           &computeQueueFamilyIndex};
  vk::raii::Buffer inBuffer(device, bCI);
  vk::MemoryRequirements inBMR = inBuffer.getMemoryRequirements();
  auto memProps = physicalDevice.getMemoryProperties();
  uint32_t memoryTypeIndex = uint32_t(~0);
  vk::DeviceSize memoryHeapSize = uint32_t(~0);
  for (uint32_t curMemoryTypeIndex = 0;
       curMemoryTypeIndex < memProps.memoryTypeCount; ++curMemoryTypeIndex) {
    vk::MemoryType memoryType = memProps.memoryTypes[curMemoryTypeIndex];
    if ((vk::MemoryPropertyFlagBits::eHostVisible & memoryType.propertyFlags) &&
        (vk::MemoryPropertyFlagBits::eHostCoherent &
         memoryType.propertyFlags)) {
      memoryHeapSize = memProps.memoryHeaps[memoryType.heapIndex].size;
      memoryTypeIndex = curMemoryTypeIndex;
      break;
    }
  }
  vk::MemoryAllocateInfo inBMAI(inBMR.size, memoryTypeIndex);
  inBM = vk::raii::DeviceMemory(device, inBMAI);

  c32 *inBufferPtr = static_cast<c32 *>(inBM.mapMemory(0, bufferSize));
  constexpr float startX = -3.0;
  constexpr float endX = 3.0;
  for (int32_t i = 0; i < nElements; i++) {
    inBufferPtr[i] = std::exp(c64{
        -square(startX + ((float)i / (float)nElements) * (endX - startX)), 0.});
  }
  inBM.unmapMemory();
  inBuffer.bindMemory(*inBM, 0);

  vk::CommandPoolCreateInfo commandPoolCreateInfo(vk::CommandPoolCreateFlags(),
                                                  computeQueueFamilyIndex);
  vk::raii::CommandPool commandPool(device, commandPoolCreateInfo);
  vk::CommandBufferAllocateInfo cBAI(*commandPool,
                                     vk::CommandBufferLevel::ePrimary, 1);
  vk::raii::CommandBuffers commandBuffers(device, cBAI);
  vk::raii::CommandBuffer commandBuffer(std::move(commandBuffers[0]));

  VkFFTConfiguration conf = {};
  VkFFTApplication app = {};
  conf.device = (VkDevice *)&*device;
  conf.FFTdim = 1;
  conf.size[0] = 1024;
  conf.numberBatches = 1;
  conf.queue = (VkQueue *)&*queue;
  conf.fence = (VkFence *)&*fence;
  conf.commandPool = (VkCommandPool *)&*commandPool;
  conf.physicalDevice = (VkPhysicalDevice *)&*physicalDevice;
  conf.buffer = (VkBuffer *)&*inBuffer;
  conf.bufferSize = &bufferSize;
  conf.disableSetLocale = true;

  auto resFFT = initializeVkFFT(&app, conf);
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
