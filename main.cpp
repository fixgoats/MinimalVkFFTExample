#define VMA_IMPLEMENTATION
#include "hack.hpp"
#include "vkFFT.h"
#include "vk_mem_alloc.h"
#include <complex>
#include <matplot/freestanding/axes_functions.h>
#include <matplot/freestanding/plot.h>

typedef std::complex<double> c64;
typedef std::complex<float> c32;
constexpr uint64_t nElements = 128;
constexpr float startX = -3.0;
constexpr float endX = 3.0;
constexpr float dX = (endX - startX) / static_cast<float>(nElements);
constexpr float endK = M_PI / dX;
constexpr float startK = -endK;
constexpr float dK = 2 * M_PI / (endX - startX);

template <typename T> constexpr T square(T x) { return x * x; }

template <typename T> std::vector<T> arange(T start, T end, T dx) {
  const uint32_t n = std::floor((end - start) / dx);
  std::vector<T> v(n);
  for (uint32_t i = 0; i < n; i++) {
    v[i] = start + (T)i * dx;
  }
  return v;
}

static std::vector<uint32_t> readFile(const std::string &filename) {
  std::ifstream file(filename, std::ios::ate | std::ios::binary);

  if (!file.is_open()) {
    throw std::runtime_error("failed to open file!");
  }

  size_t fileSize = static_cast<size_t>(file.tellg());
  std::vector<uint32_t> buffer(fileSize / 4);
  file.seekg(0);
  file.read(reinterpret_cast<char *>(buffer.data()), fileSize);
  file.close();
  return buffer;
}

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

struct RaiiVmaAllocator {
  VmaAllocator allocator;
  RaiiVmaAllocator(vk::raii::PhysicalDevice &physicalDevice,
                   vk::raii::Device &device, vk::raii::Instance &instance) {
    VmaAllocatorCreateInfo allocatorInfo{};
    allocatorInfo.physicalDevice = *physicalDevice;
    allocatorInfo.vulkanApiVersion = physicalDevice.getProperties().apiVersion;
    allocatorInfo.device = *device;
    allocatorInfo.instance = *instance;
    vmaCreateAllocator(&allocatorInfo, &allocator);
  }

  ~RaiiVmaAllocator() {
    std::cout << "Destroying allocator\n";
    vmaDestroyAllocator(allocator);
  }
};

struct RaiiVmaBuffer {
  // RaiiVmaAllocator takes care of deleting itself
  VmaAllocator *allocator;
  vk::Buffer buffer;
  VmaAllocation allocation;
  VmaAllocationInfo allocationInfo;
  RaiiVmaBuffer(VmaAllocator &Allocator,
                VmaAllocationCreateInfo &allocCreateInfo,
                vk::BufferCreateInfo &BCI) {
    allocator = &Allocator;
    VkBuffer bufferRaw;
    vmaCreateBuffer(*allocator, reinterpret_cast<VkBufferCreateInfo *>(&BCI),
                    &allocCreateInfo, &bufferRaw, &allocation, &allocationInfo);
    buffer = bufferRaw;
  }
  ~RaiiVmaBuffer() {
    std::cout << "Destroying buffer\n";
    vmaDestroyBuffer(*allocator, buffer, allocation);
  }
};

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

std::vector<c32> doTheCalculation() {
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

  uint64_t bufferSize = nElements * sizeof(c32);

  std::cout << "Got here";
  RaiiVmaAllocator allocator(physicalDevice, device, instance);
  std::cout << "Got here";

  vk::BufferCreateInfo stagingBCI({}, bufferSize,
                                  vk::BufferUsageFlagBits::eTransferSrc |
                                      vk::BufferUsageFlagBits::eTransferDst);

  VmaAllocationCreateInfo allocCreateInfo{};
  allocCreateInfo.usage = VMA_MEMORY_USAGE_AUTO;
  allocCreateInfo.flags =
      VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
      VMA_ALLOCATION_CREATE_MAPPED_BIT;
  RaiiVmaBuffer staging(allocator.allocator, allocCreateInfo, stagingBCI);

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
  RaiiVmaBuffer in(allocator.allocator, allocCreateInfo, stateBCI);
  RaiiVmaBuffer out(allocator.allocator, allocCreateInfo, stateBCI);
  RaiiVmaBuffer kernel(allocator.allocator, allocCreateInfo, stateBCI);

  c32 *stagingPtr = static_cast<c32 *>(staging.allocationInfo.pMappedData);
  for (int32_t i = 0; i < nElements; i++) {
    stagingPtr[i] = std::exp(c64{-square(startX + (float)i * dX), 0.});
  }

  oneTimeSubmit(device, commandPool, queue,
                [&](vk::CommandBuffer const &commandBuffer) {
                  commandBuffer.copyBuffer(staging.buffer, in.buffer,
                                           vk::BufferCopy(0, 0, bufferSize));
                });
  for (int32_t i = 0; i < nElements; i++) {
    stagingPtr[(i + nElements / 2) % nElements] =
        c64{0., -(startK + (float)i * dK)};
  }
  oneTimeSubmit(device, commandPool, queue,
                [&](vk::CommandBuffer const &commandBuffer) {
                  commandBuffer.copyBuffer(staging.buffer, kernel.buffer,
                                           vk::BufferCopy(0, 0, bufferSize));
                });

  auto spirv = readFile("mul.spv");
  vk::ShaderModuleCreateInfo sMCI(vk::ShaderModuleCreateFlags(), spirv);
  vk::raii::ShaderModule shaderModule(device, sMCI);
  const std::vector<vk::DescriptorSetLayoutBinding> dSLBs = {
      {0, vk::DescriptorType::eStorageBuffer, 1,
       vk::ShaderStageFlagBits::eCompute},
      {1, vk::DescriptorType::eStorageBuffer, 1,
       vk::ShaderStageFlagBits::eCompute}};

  vk::DescriptorSetLayoutCreateInfo dSLCI(vk::DescriptorSetLayoutCreateFlags(),
                                          dSLBs);
  vk::raii::DescriptorSetLayout descriptorSetLayout(device, dSLCI);

  vk::PipelineLayoutCreateInfo pLCI(vk::PipelineLayoutCreateFlags(),
                                    *descriptorSetLayout);
  vk::raii::PipelineLayout pipelineLayout(device, pLCI);
  vk::raii::PipelineCache pipelineCache(device, vk::PipelineCacheCreateInfo());
  vk::PipelineShaderStageCreateInfo pSCI(vk::PipelineShaderStageCreateFlags(),
                                         vk::ShaderStageFlagBits::eCompute,
                                         *shaderModule, "main");
  vk::ComputePipelineCreateInfo cPCI(vk::PipelineCreateFlags(), pSCI,
                                     *pipelineLayout);
  vk::raii::Pipeline computePipeline(device, pipelineCache, cPCI);
  vk::DescriptorPoolSize descriptorPoolSize(vk::DescriptorType::eStorageBuffer,
                                            1);
  vk::DescriptorPoolCreateInfo dPCI(
      vk::DescriptorPoolCreateFlags(
          vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet),
      1, descriptorPoolSize);
  vk::raii::DescriptorPool descriptorPool(device, dPCI);

  vk::DescriptorSetAllocateInfo dSAI(*descriptorPool, 1, &*descriptorSetLayout);
  vk::raii::DescriptorSets pDescriptorSets(device, dSAI);
  vk::raii::DescriptorSet descriptorSet(std::move(pDescriptorSets[0]));
  vk::DescriptorBufferInfo kBufferInfo(kernel.buffer, 0, bufferSize);
  vk::DescriptorBufferInfo outBufferInfo(out.buffer, 0, bufferSize);
  vk::DescriptorBufferInfo inBufferInfo(in.buffer, 0, bufferSize);

  const std::vector<vk::WriteDescriptorSet> writeDescriptorSets = {
      {*descriptorSet, 0, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr,
       &kBufferInfo},
      {*descriptorSet, 1, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr,
       &outBufferInfo}};
  device.updateDescriptorSets(writeDescriptorSets, {});

  VkFFTConfiguration conf = {};
  VkFFTApplication app = {};
  conf.device = (VkDevice *)&*device;
  conf.FFTdim = 1;
  conf.size[0] = nElements;
  conf.numberBatches = 1;
  conf.queue = (VkQueue *)&*queue;
  conf.fence = (VkFence *)&*fence;
  conf.commandPool = (VkCommandPool *)&*commandPool;
  conf.physicalDevice = (VkPhysicalDevice *)&*physicalDevice;
  conf.isInputFormatted = true;
  conf.inputBuffer = (VkBuffer *)&in.buffer;
  conf.buffer = (VkBuffer *)&out.buffer;
  conf.bufferSize = &bufferSize;
  conf.inputBufferSize = &bufferSize;
  conf.inverseReturnToInputBuffer = true;

  auto resFFT = initializeVkFFT(&app, conf);

  VkFFTLaunchParams launchParams = {};

  vk::MemoryBarrier memoryBarrier(
      vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eMemoryWrite,
      vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite);
  vk::CommandBufferBeginInfo cBBI(
      vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
  commandBuffer.begin(cBBI);
  launchParams.commandBuffer = (VkCommandBuffer *)&*commandBuffer;
  resFFT = VkFFTAppend(&app, -1, &launchParams);
  commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
                                vk::PipelineStageFlagBits::eAllCommands, {},
                                memoryBarrier, nullptr, nullptr);
  commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, *computePipeline);
  commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
                                   *pipelineLayout, 0, {*descriptorSet}, {});
  commandBuffer.dispatch(nElements / 32, 1, 1);
  commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
                                vk::PipelineStageFlagBits::eAllCommands, {},
                                memoryBarrier, nullptr, nullptr);
  resFFT = VkFFTAppend(&app, 1, &launchParams);
  commandBuffer.end();

  vk::SubmitInfo submitInfo(0, nullptr, nullptr, 1, &*commandBuffer);
  queue.submit(submitInfo, *fence);
  auto result = device.waitForFences({*fence}, true, uint64_t(-1));

  oneTimeSubmit(device, commandPool, queue,
                [&](vk::CommandBuffer const &commandBuffer) {
                  commandBuffer.copyBuffer(in.buffer, staging.buffer,
                                           vk::BufferCopy(0, 0, bufferSize));
                });

  deleteVkFFT(&app);
  std::vector<c32> rvec(nElements);
  memcpy(rvec.data(), stagingPtr, nElements * sizeof(c32));
  return rvec;
}

int main(int argc, char *argv[]) {
  auto vec = doTheCalculation();
  float *ehh = reinterpret_cast<float *>(vec.data());
  for (uint32_t i = 0; i < nElements; i++) {
    ehh[i] = vec[i].real();
  }
  std::vector<float> realvec(
      reinterpret_cast<float *>(vec.data()),
      reinterpret_cast<float *>(vec.data()) +
          nElements *
              sizeof(float)); // jfc if i'm doing nonsense like this why don't I
                              // just do bare pointers while I'm at it lol
  std::vector<float> x = arange(startX, endX, dX);
  std::vector<float> y(nElements);
  std::transform(x.begin(), x.end(), y.begin(),
                 [](float x) { return -2. * x * std::exp(-x * x); });
  matplot::plot(x, y, "-o");
  matplot::hold(matplot::on);
  matplot::plot(x, realvec, "-x");
  matplot::show();
  return 0;
}
