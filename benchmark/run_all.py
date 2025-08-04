import os
import subprocess
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def run_py(script_rel_path):
    script_path = os.path.join(PROJECT_ROOT, script_rel_path)
    subprocess.run([sys.executable, script_path])

print("=== PyTorch baseline ===")
run_py("pytorch_baseline/run_pytorch.py")

print("\n=== CUDA kernel ===")
run_py("cuda_kernel/batched_matmul.py")
