# 🚀 CUDA AI Inference Acceleration: Batched MatMul (PyTorch vs. Custom CUDA Kernel)


This project demonstrates how to implement and benchmark high-performance batched matrix multiplication on GPU using both **PyTorch’s official GPU API** and a **custom CUDA batched matmul kernel** (via CuPy dynamic loading).
It provides a clear, practical baseline for AI/GPU optimization.

**A key highlight is the full GPU profiling workflow:**

* **Step 1 – Profiling:** NVTX markers were inserted into the code, and NVIDIA Nsight was used to capture detailed GPU timelines. The initial profiling revealed that the custom kernel was **much slower** than PyTorch’s cuBLAS baseline.
* **Step 2 – Bottleneck Detection:** Timeline analysis showed that most of the extra latency was concentrated in `cudaStreamSynchronize`, indicating **computation inefficiency** rather than memory transfer delays.
* **Step 3 – Optimization:** Kernel-level bottlenecks were identified and addressed by introducing a **tiled + shared memory** design, improving memory access patterns and reducing execution time.
* **Result:** The optimized kernel achieved a substantial speedup over the naive version while maintaining correctness.


---


## 🌟 Key Features

* **Industry Baseline (PyTorch cuBLAS):**
  Benchmarks with PyTorch’s batched GPU matmul as an industry-standard reference (cuBLAS backend), serving as the primary performance baseline.

* **Custom CUDA Kernel (via CuPy):**
  Implements a hand-crafted batched matmul kernel, launched from Python using CuPy’s `RawModule`. Profiling initially showed it was slower than cuBLAS due to long synchronization waits.

* **Profiling-Driven Optimization:**
  NVTX annotations and NVIDIA Nsight timeline profiling were used to pinpoint that the slowdown came from computation inefficiency. A **tiled + shared memory** version was developed, cutting per-batch latency from 0.374 ms to 0.232 ms.

* **Unified Benchmark Framework:**
  A single script runs PyTorch, naive CUDA, and optimized CUDA benchmarks, outputting side-by-side latency with profiler-ready NVTX markers for in-depth analysis.

* **Extensible Engineering Structure:**
  Modular design allows easy integration of ONNX, TensorRT, HIP, or OpenMP backends, making it a flexible testbed for AI system and hardware benchmarking.


---

## 📁 Directory Structure

```
cuda-ai-inference-acceleration/
├── benchmark/                       # Benchmark scripts
│   └── run_all.py                    # One-click benchmark runner (PyTorch / CUDA / HIP)
│
├── cuda_kernel/                      # Custom CUDA kernels
│   ├── batched_matmul.cu              # Original batched matmul kernel
│   ├── batched_matmul_tiled.cu        # Optimized tiled + shared memory version
│   └── batched_matmul.py              # Python interface (CuPy kernel invocation)
│
├── hip_kernel/                        # HIP kernels (AMD ROCm platform)
│   ├── batched_matmul_hip.hip         # HIP version of batched matmul kernel
│   └── run_hip.py                     # HIP test script (runs on ROCm or HIP CUDA backend)
│
├── images/                            # Images for README
│   ├── pytorch_matmul.PNG             # Nsight timeline: PyTorch baseline
│   ├── CUDA_kernel_matmul.PNG         # Nsight timeline: original CUDA kernel
│   └── CUDA_kernel_matmul_tiled.PNG   # Nsight timeline: optimized tiled CUDA kernel
│
├── nsight_reports/                    # Nsight profiling reports (.nsys-rep)
│   └── cuBLAS-CUDA-CUDAtiled.nsys-rep # Profiling results for PyTorch, original CUDA, and tiled CUDA
│
├── pytorch_baseline/                  # PyTorch baseline implementation
│   └── run_pytorch.py                 # cuBLAS-based batched matmul baseline
│
├── LICENSE                            # License file
├── README.md                          # Project documentation
└── requirements.txt                   # Python dependency list

```

> * **images/**: All key profiler timeline screenshots for analysis & reporting.



---

## 🧑‍💻 Main Components

### 1. **PyTorch Baseline (`pytorch_baseline/run_pytorch.py`)**

> Executes batched matrix multiplication on GPU using PyTorch’s cuBLAS backend, serving as the performance baseline.
> Includes NVTX annotations for kernel-level profiling with NVIDIA Nsight.

### 2. **Custom CUDA Kernel (`cuda_kernel/batched_matmul.cu`, `batched_matmul.py`)**

> Implements a pure CUDA batched matmul kernel with batch-parallel execution for direct comparison against cuBLAS.
> Initial profiling showed it was slower due to long synchronization waits (`cudaStreamSynchronize`), pointing to computation inefficiency.
> Optimized with a **tiled + shared memory** design, improving memory access and reducing execution time.
> Integrated with Python via `cupy.RawModule` for easy benchmarking and profiling.

### 3. **Unified Benchmark Script (`benchmark/run_all.py`)**

> Runs PyTorch, naive CUDA, and optimized CUDA benchmarks in a single command.
> Outputs side-by-side latency results and includes NVTX markers for profiler-friendly workflows, enabling before-and-after optimization comparison.


---

## 🚦 How to Run

### 1️⃣ Install dependencies

```bash
pip install -r requirements.txt
# Requires: NVIDIA GPU drivers and CUDA toolkit (tested with CUDA 11+)
```

### 2️⃣ Run all benchmarks (with profiler-ready workflow)

```bash
python benchmark/run_all.py
```

* Runs PyTorch, naive CUDA, and optimized CUDA batched matmul benchmarks.
* Outputs average per-batch latency (ms) for all implementations side-by-side.
* Fully compatible with Nsight Systems/Compute — NVTX markers are embedded so kernel and synchronization regions appear clearly in the timeline.
* For profiling, launch this script under Nsight; each benchmark iteration is annotated for detailed performance analysis.



---

## 🧪 Experimental Results & Profiling Analysis

### 1. Quantitative Benchmark Results

| Implementation                   | Avg Latency (ms per batch) | Speedup vs. Naive |
| -------------------------------- | -------------------------- | ----------------- |
| **PyTorch baseline (cuBLAS)**    | 0.080                      | 4.68×             |
| **CUDA kernel (naive)**          | 0.374                      | 1.00×             |
| **CUDA kernel (tiled + shared)** | 0.232                      | 1.61×             |

*(Each average of 50 rounds, batch=128, matrix size=64×64)*

The PyTorch baseline is the fastest due to cuBLAS optimizations, but the optimized tiled CUDA kernel significantly outperforms the naive version by \~38%.


---

### 2. Timeline Profiling & Bottleneck Analysis

Using NVTX annotations and Nsight Systems, each matmul iteration was profiled to identify bottlenecks. Screenshots are stored in `/images`.

#### 📷 cuBLAS (PyTorch) Timeline

![PyTorch Timeline](images/pytorch_matmul.PNG)
cuBLAS kernels (green) execute in tightly packed sequences with minimal idle time. Synchronization waits are extremely short, indicating highly efficient computation and memory pipelines.

#### 📷 CUDA Kernel (Naive)

![CUDA Kernel Timeline](images/CUDA_kernel_matmul.PNG)
Kernel launches (red) are individually fast (\~3-4 μs) but followed by long `cudaStreamSynchronize` waits, revealing inefficiencies in computation and memory access.

#### 📷 CUDA Kernel (Tiled + Shared Memory)

![CUDA Kernel Tiled Timeline](images/CUDA_kernel_matmul_tiled.PNG)
Optimized kernel reduces synchronization delays by improving memory access patterns, lowering per-batch runtime from 0.374 ms to 0.232 ms.

---

### 3. Analysis & Technical Insights

**Key observations:**

* PyTorch cuBLAS remains the gold standard for speed due to deeply tuned kernels and memory pipelines.
* The naive CUDA kernel’s slowdown stemmed from inefficient memory access and computation, not just kernel launch overhead.
* Tiled + shared memory optimization cut per-batch latency by \~38% compared to the naive version.

**Root cause & resolution:**

* Nsight timelines showed long waits at `cudaStreamSynchronize`, pointing to computation inefficiency.
* Rewriting the kernel to use shared memory tiling improved memory locality and reduced redundant global memory accesses, leading to significant speedup.

**Conclusion:**

* Profiling-driven optimization is essential for GPU performance tuning.
* Even without matching cuBLAS, targeted optimizations can deliver large performance gains over naive implementations.


---

### 4. Annotate Markers vs. `time.time()`: What’s the Difference?

* `nvtx.annotate()` is inserted inside the for loop for each matmul round, creating a labeled region in the Nsight profiler timeline. This groups all GPU activity for that round — kernel launches, CUDA API calls, memory operations, and synchronization events — so they can be visually correlated for analysis.
* `time.time()` is measured around the **entire** loop, including all kernel launches, memory operations, and the final `cudaStreamSynchronize()` call. This means the measured duration includes all accumulated GPU-side waiting time.
* NVTX annotations highlight **per-kernel compute behavior**, while `time.time()` measures the **end-to-end latency** for the full set of iterations.

#### Timing and Annotation Example

```python
t0 = time.time()
for _ in range(50):
    with nvtx.annotate("Pytorch matmul", color="green"):
        out = pytorch_batched_matmul(a, b)
torch.cuda.synchronize()
t1 = time.time()
print(f"PyTorch batched matmul: {(t1 - t0) * 1000 / 50:.3f} ms per batch")
```

* In this example, `time.time()` reflects the total loop execution time (compute + memory ops + synchronization), while NVTX markers expose each kernel call in the profiler timeline.
* **Why it matters:** This dual measurement approach helped reveal that the naive CUDA kernel’s delay was dominated by synchronization waits, guiding the optimization toward better memory access patterns.

---

### 5. Project Insights

* Demonstrates the **full profiling-driven optimization workflow**: starting from baseline measurement, through identifying bottlenecks, to applying kernel-level improvements.
* Combines **quantitative benchmarking** with **visual profiler evidence** to clearly pinpoint performance gaps.
* Shows how integrating NVTX + Nsight with wall-clock timing can uncover inefficiencies invisible to a single measurement method.
* Validates that targeted optimization (e.g., tiled + shared memory) can achieve substantial speedups even if cuBLAS remains faster, reinforcing the value of systematic performance engineering.



---

## 📌 Technical Highlights

* **PyTorch GPU Engineering:** Implements batched matmul using PyTorch’s cuBLAS backend as a performance reference.
* **CUDA Kernel Programming:** Designs and optimizes a custom batched matmul kernel, covering memory layout, grid/block/thread configuration, and kernel launch strategy.
* **CuPy Integration:** Bridges Python and CUDA via CuPy’s dynamic compilation, enabling rapid prototyping without pybind11 or manual C++ bindings.
* **End-to-End Profiling:** Embeds NVTX markers for Nsight profiling, supporting both per-kernel analysis and full end-to-end latency measurement to drive optimization decisions.

---

## ⚠️ Challenges & Solutions

**1. Profiling mismatch between `nvtx.annotate()` and `time.time()`**

* **Challenge:** Large discrepancy between profiler timeline and wall-clock latency.
* **Cause:** Asynchronous kernel launches and memory operations only synchronize at `cudaDeviceSynchronize()`. Wall-clock timing measured the accumulated execution time for all queued GPU work.
* **Solution:** Added `nvtx.annotate()` markers inside each iteration to align per-kernel execution in the timeline with measured latencies.

**2. Identifying the real bottleneck**

* **Challenge:** Per-kernel execution time was close to cuBLAS, yet total latency was higher.
* **Cause:** Overhead from repeated memory allocations, frees, and kernel launch costs.
* **Solution:** Nsight analysis pinpointed these overheads; optimization with tiled + shared memory reduced synchronization delays and improved performance.

**3. HIP execution limitation**

* **Challenge:** HIP kernel implemented but not executable in the current environment.
* **Cause:** ROCm supports only Linux + AMD GPUs; HIP CUDA backend for NVIDIA GPUs is Linux-only.
* **Solution:** Retained HIP implementation in the codebase with documentation, ready for deployment on supported platforms.

---

## 🔥 Extension Ideas

* **HIP Porting:** Benchmark and profile on ROCm for AMD GPU support.
* **Multi-GPU Scaling:** Implement batch parallelism and distributed execution.
* **FP16 & Mixed Precision:** Add half/mixed precision for CUDA and HIP.
* **Framework Integration:** Explore Triton, CUTLASS, or Compute Kernel (CK) for portable performance.
* **Deep Learning Hooks:** Add PyTorch/TensorFlow custom ops for transparent benchmarking.
* **Compiler Optimization:** Test LLVM, ROCm, TVM for kernel/system tuning.
* **CPU Backend:** Implement OpenMP CPU version for CPU vs. GPU comparisons.
* **ONNX/TensorRT/ROCm Inference:** Integrate for full-platform acceleration benchmarks.

> These extensions position the project as a versatile testbed for modern GPU kernel development across NVIDIA, AMD, and beyond.


---

## 📌 HIP Porting Attempt & Environment Limitation

As part of demonstrating **extensibility and cross-platform design**, this project includes a `hip_kernel/` module that implements the same batched matmul kernel using **HIP** for AMD ROCm platforms.

**Development process:**

* Rewrote CUDA kernel into HIP syntax (nearly identical due to HIP’s CUDA-like design).
* Integrated HIP kernel into the same benchmarking structure as the CUDA version.
* Verified that the code is ready for compilation and execution on supported platforms.

**Findings:**

* ROCm is officially supported **only on Linux + AMD GPUs**.
* **No ROCm runtime for Windows**; `cupy-rocm-*` wheels are not available for Windows.
* HIP also has a **CUDA backend** that allows HIP code to run on NVIDIA GPUs — but this backend works **only on Linux**, not Windows.

**Current environment:**

* **Windows + NVIDIA GPU** → HIP backend unavailable → HIP benchmarks skipped.

**Where HIP will run:**

| Environment                               | Description                                                                        | CUDA Kernel | HIP Kernel |
| ----------------------------------------- | ---------------------------------------------------------------------------------- | ----------- | ---------- |
| Windows + NVIDIA GPU                      | Your current machine — runs CUDA only.                                             | ✅           | ❌          |
| Linux + NVIDIA GPU (**HIP CUDA backend**) | HIP code compiled to CUDA API calls and executed on NVIDIA GPUs under Linux.       | ✅           | ✅          |
| Linux + AMD GPU (**ROCm backend**)        | HIP code compiled to ROCm API calls and executed natively on AMD GPUs under Linux. | ❌           | ✅          |

**Legend:**

* **HIP CUDA backend** → HIP code translated into CUDA API calls, runs on NVIDIA GPUs (Linux only).
* **ROCm backend** → HIP code runs natively on AMD GPUs using ROCm runtime (Linux only).

**Key takeaway for interviewers:**

* Demonstrates ability to design a **portable GPU benchmarking framework**.
* Shows hands-on HIP implementation experience, even without access to the target platform.
* HIP module is ready for deployment and benchmarking on supported ROCm hardware.

---

## 📌 CUDA vs ROCm: Environment Limitations and Porting Notes

While this project focuses on CUDA acceleration and profiling, an attempt was made to port the custom CUDA kernel to AMD's ROCm/HIP for broader hardware support. On Windows, native ROCm support is unavailable, which prevented successful execution on the development machine. This section documents the key differences and similarities between CUDA and HIP to demonstrate portability considerations.

### Environment Constraints

* **CUDA**: Fully supported on Windows with NVIDIA GPUs; Nsight profiling integrated.
* **ROCm/HIP**: No official Windows support for ROCm runtime; HIP development primarily targets Linux with AMD GPUs.
* **Impact**: Kernel compilation for HIP was possible in theory, but runtime execution and profiling required a Linux+AMD environment.

### CUDA ↔ HIP API Mapping Table

| CUDA API / Keyword                           | HIP Equivalent                                                        | Notes                                                                                             |
| -------------------------------------------- | --------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------- |
| `#include <cuda_runtime.h>`                  | `#include <hip/hip_runtime.h>`                                        | Main runtime API header                                                                           |
| `cudaMalloc(ptr, size)`                      | `hipMalloc(ptr, size)`                                                | Memory allocation on device                                                                       |
| `cudaFree(ptr)`                              | `hipFree(ptr)`                                                        | Free device memory                                                                                |
| `cudaMemcpy(dst, src, size, cudaMemcpyKind)` | `hipMemcpy(dst, src, size, hipMemcpyKind)`                            | Host-device memory copy; kinds map 1:1 (e.g., `cudaMemcpyHostToDevice` → `hipMemcpyHostToDevice`) |
| `cudaMemset(ptr, val, size)`                 | `hipMemset(ptr, val, size)`                                           | Initialize memory on device                                                                       |
| `cudaDeviceSynchronize()`                    | `hipDeviceSynchronize()`                                              | Wait for all device operations to finish                                                          |
| `cudaGetErrorString(err)`                    | `hipGetErrorString(err)`                                              | Convert error code to readable string                                                             |
| `cudaStream_t`                               | `hipStream_t`                                                         | Stream handle type                                                                                |
| `cudaStreamCreate(&stream)`                  | `hipStreamCreate(&stream)`                                            | Create a stream                                                                                   |
| `cudaStreamSynchronize(s)`                   | `hipStreamSynchronize(s)`                                             | Wait for a specific stream                                                                        |
| `<<<grid, block>>>`                          | `hipLaunchKernelGGL(kernel, grid, block, sharedMem, stream, args...)` | HIP uses a macro for kernel launches instead of triple-angle bracket syntax                       |
| `__global__` / `__device__` / `__host__`     | Same in HIP                                                           | Kernel and function qualifiers remain identical                                                   |

### Takeaway

The porting process revealed that CUDA-to-HIP translation for this kernel would require minimal code changes thanks to HIP's close API parity. However, the primary blocker was the lack of ROCm runtime support on Windows, making practical execution and profiling on AMD GPUs infeasible in this environment.

---

## 📣 Why This Project?

Built from scratch, this project demonstrates the **full workflow for GPU performance engineering**: from high-level PyTorch implementation, to custom CUDA kernel design, to real-world benchmarking and timeline-based profiling. It showcases both **software stack integration** and **low-level bottleneck analysis**, proving capability to identify, analyze, and resolve performance issues through profiling-driven optimization. This makes it a valuable reference for AI infrastructure, research, and advanced GPU system development.


---

## 👤 Author

Wang Chen Han
[hank851107@gmail.com](mailto:hank851107@gmail.com)
[GitHub: HankWang-WL](https://github.com/HankWang-WL)

---

## License

MIT License
