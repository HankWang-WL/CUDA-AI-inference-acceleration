import subprocess

print("=== PyTorch baseline ===")
subprocess.run(["python", "pytorch_baseline/run_pytorch.py"])

print("\n=== CUDA kernel ===")
subprocess.run(["python", "cuda_kernel/batched_matmul.py"])
