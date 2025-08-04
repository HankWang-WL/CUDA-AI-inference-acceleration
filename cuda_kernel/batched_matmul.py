# cuda_kernel/batched_matmul.py
import cupy as cp
import time
import nvtx
import os

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
cu_path = os.path.join(THIS_DIR, "batched_matmul.cu")

with open(cu_path, "r") as f:
    kernel_code = f.read()


module = cp.RawModule(code=kernel_code)
batched_matmul_kernel = module.get_function("batched_matmul_kernel")

def batched_matmul_cupy(a, b):
    # a, b: (batch, M, K), (batch, K, N)
    batch, M, K = a.shape
    _, _, N = b.shape
    c = cp.zeros((batch, M, N), dtype=cp.float32)
    threads = (16, 16, 1)
    blocks = ( (N+15)//16, (M+15)//16, batch )
    batched_matmul_kernel(
        (blocks), (threads),
        (a, b, c, batch, M, K, N)
    )
    return c

if __name__ == "__main__":
    batch, M, K, N = 128, 64, 64, 64
    a = cp.random.randn(batch, M, K).astype(cp.float32)
    b = cp.random.randn(batch, K, N).astype(cp.float32)

    # warmup
    for _ in range(10):
        batched_matmul_cupy(a, b)
    cp.cuda.Stream.null.synchronize()
    
    t0 = time.time()
    for _ in range(50):
        with nvtx.annotate("CUDA kernel matmul", color="red"):
            c = batched_matmul_cupy(a, b)
    cp.cuda.Stream.null.synchronize()
    t1 = time.time()
    print(f"CUDA kernel batched matmul: {(t1 - t0) * 1000 / 50:.3f} ms per batch")
