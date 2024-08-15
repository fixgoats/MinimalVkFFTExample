## A minimal, modern working example of VkFFT

### The premise
Fast Fourier transforms are very useful for all kinds of applications, from analysing
energy spectra to solving differential equations to machine learning. FFTs are
highly parallelizable and can benefit massively from running on GPUs. [VkFFT](https://github.com/DTolm/VkFFT)
provides a cross platform, open source, highly performant GPU bsaed implementation of the
FFT for arbitrary dimensions.

### The problem
Vulkan requires a lot of boilerplate, and many of the best guides for working with Vulkan
use the C interface, ensure maximum verbosity and only focus on the graphics queue
or at least incorporate it. This is good for learning the ins and outs of Vulkan
but it can also leave your head swimming with the amount of information being dumped
on you, and makes it hard to disentangle what is really the minimum amount of code
needed to get basic vector addition running.

VkFFT is an amazing project that has the potential to shake up the GPU based HPC
world, since one of the main downsides of Vulkan compared to CUDA is the lack of
powerful user-friendly maths libraries. In my experience, though, it's hard to
find a good minimal example specifically using the Vulkan backend of VkFFT. The
code in the main VkFFT repository is geared towards compiling a suite of benchmarks,
rather than just illustrating a single working example with a single backend, and
has its code dispersed across multiple different files meaning you need to jump
around and pick apart what needs to be set up and where. It is also written more
in the C-style than C++, which is a valid choice but I personally quite like a
lot of C++'s features and want to make use of them. The VkFFT documentation and
repo also fetch and compile their own glslang, but for this case I'd prefer to
use commonly available system libraries as much as possible, so the
CMakeLists.txt is also illustrative of how you can get around that.

### This repo
This repo is heavily based on Thales Sabinos's guide [A Simple Vulkan Compute Example in C++](https://bakedbits.dev/posts/vulkan-compute-example/), but also
uses Khronos's RAII features to the greatest extent possible. It is my attempt
to illustrate a mostly idiomatic (afaict C-style casts are
unavoidable when making the VkFFT structs from Vulkan HPP code) C++ usage of VkFFT. However,
the code here may not be optimal, e.g. it would probably be better practice
to use a staging buffer to upload the numbers so that the main memory being used
isn't necessarily host coherent or host visible, or maybe populate the array
with a shader.

To build, clone and run
```
cmake -S . -B build # I like to use ninja, so you could add -G Ninja
cmake --build build --parallel
./build/example
```
The code will produce a 1D complex Gaussian with imaginary part 0 as input, which
should produce another Gaussian in the output (by default VkFFT performs the FFT
in place, overwriting the original array), but with a different standard
deviation and scaled by the square root of the number of elements. The output is
also rotated by half the array size and changes sign every other element compared to what one may expect
as a physicist, but this is just how standard FFT implementations work.
