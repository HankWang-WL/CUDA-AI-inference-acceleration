# 🚀 CUDA AI Inference Acceleration: Batched MatMul (PyTorch vs. Custom CUDA Kernel)

This project demonstrates how to implement and benchmark high-efficiency batched matrix multiplication on GPU using both **PyTorch (official GPU API)** and a **custom CUDA batched matmul kernel** (via Cupy dynamic loading). It provides a clear, practical baseline for AI/GPU optimization and infrastructure interviews or technical portfolio.

---

## 🌟 Key Features

* **PyTorch Baseline**: Uses PyTorch's batched GPU matmul as an industry-standard reference.
* **Custom CUDA Kernel**: Implements a hand-crafted CUDA batched matmul kernel, called from Python via Cupy for direct performance comparison.
* **Unified Benchmark**: Benchmark script runs both PyTorch and custom CUDA kernel for direct latency comparison.
* **Extensible Structure**: Easily extendable for ONNX, TensorRT, OpenMP, profiling, or advanced topics.

---

## 📁 Directory Structure

```
cuda-ai-inference-acceleration/
├── cuda_kernel/
│   ├── batched_matmul.cu         # CUDA kernel
│   ├── batched_matmul.py         # Python interface for CUDA kernel
│   ├── build.sh                  # (optional) build script
├── pytorch_baseline/
│   └── run_pytorch.py            # PyTorch baseline
├── benchmark/
│   └── run_all.py                # Unified benchmark
├── README.md
└── requirements.txt
```

---

## 🧑‍💻 Main Components

### 1. **PyTorch Baseline**

`pytorch_baseline/run_pytorch.py`

> Batched matmul on GPU using PyTorch, serving as the industry baseline.

### 2. **Custom CUDA Kernel (Cupy)**

`cuda_kernel/batched_matmul.cu`

> Pure CUDA batched matmul kernel, batch dimension parallelization.

`cuda_kernel/batched_matmul.py`

> Loads and calls CUDA kernel via cupy.RawModule, measures execution time in Python.

### 3. **Benchmark Script**

`benchmark/run_all.py`

> Runs both PyTorch and custom CUDA kernel, prints latency results for comparison.

---

## 🚦 How to Run

### 1️⃣ Install dependencies

```bash
pip install -r requirements.txt
# (Requires NVIDIA GPU drivers & CUDA toolkit installed)
```

### 2️⃣ Run benchmarks

```bash
python benchmark/run_all.py
```

Runs both the PyTorch and custom CUDA kernel batched matmul, outputs average per-batch latency (ms).

---

## 📌 Technical Highlights

* **PyTorch/Deep Learning Engineering**: Shows proficiency with PyTorch GPU tensor operations.
* **CUDA Kernel Programming**: Demonstrates low-level kernel design for batched matmul, including memory layout, grid/block/thread setup.
* **Cupy Integration**: Seamlessly connects Python to CUDA with dynamic compilation, no pybind11 or manual C++ compilation needed.
* **Performance Profiling**: Unified script for direct comparison, a must-have for AI/infra teams.

---

## 🔥 Extension Ideas

* Add kernel profiling (nvprof, nsight, torch profiler) to visualize execution time and memory usage.
* Extend to support half precision (FP16), larger batch/matrix sizes.
* Add OpenMP-based CPU version for full CPU vs. GPU comparison.
* Integrate ONNX/TensorRT in the same benchmarking framework for end-to-end inference acceleration comparison.

---

## 📣 Why This Project?

This project was built completely from scratch to demonstrate the full workflow required for AI engineering/infra teams: from using high-level GPU frameworks (PyTorch) to hand-optimized CUDA kernels, plus real benchmark and profiling analysis. It proves capability in both software stack integration and low-level optimization, perfect for interviews at NVIDIA, Qualcomm, AWS, or similar AI/GPU teams.

---

## 👤 Author

Wang Chen Han
[hank851107@gmail.com](mailto:hank851107@gmail.com)
[GitHub: HankWang-WL](https://github.com/HankWang-WL)

---

## License

MIT License
