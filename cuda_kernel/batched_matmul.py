# cuda_kernel/batched_matmul.py
import cupy as cp
import time
import nvtx
import os

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
cu_naive_path  = os.path.join(THIS_DIR, "batched_matmul.cu")
cu_tiled_path  = os.path.join(THIS_DIR, "batched_matmul_tiled.cu")  


with open(cu_naive_path, "r") as f:
    naive_code = f.read()
with open(cu_tiled_path, "r") as f:
    tiled_code = f.read()


mod_naive = cp.RawModule(code=naive_code)
mod_tiled = cp.RawModule(code=tiled_code)   # TILE=16

kern_naive = mod_naive.get_function("batched_matmul_kernel")
kern_tiled = mod_tiled.get_function("batched_matmul_tiled_kernel")

def batched_matmul_cupy_naive(a, b):
    # a, b: (batch, M, K), (batch, K, N)
    batch, M, K = a.shape
    _, _, N = b.shape
    c = cp.zeros((batch, M, N), dtype=cp.float32)
    threads = (16, 16, 1)
    blocks  = ((N + 15)//16, (M + 15)//16, batch)
    kern_naive((blocks), (threads), (a, b, c, batch, M, K, N))
    return c

def batched_matmul_cupy_tiled(a, b, tile=16):
    batch, M, K = a.shape
    _, _, N = b.shape
    c = cp.zeros((batch, M, N), dtype=cp.float32)
    threads = (tile, tile, 1)                               # TILE=16
    blocks  = ((N + tile - 1)//tile, (M + tile - 1)//tile, batch)
    kern_tiled((blocks), (threads), (a, b, c, batch, M, K, N))
    return c

if __name__ == "__main__":
    batch, M, K, N = 128, 64, 64, 64
    a = cp.random.randn(batch, M, K).astype(cp.float32)
    b = cp.random.randn(batch, K, N).astype(cp.float32)

    # warmup
    for _ in range(10):
        batched_matmul_cupy_naive(a, b)
        batched_matmul_cupy_tiled(a, b)
    cp.cuda.Stream.null.synchronize()

    # Naive 
    t0 = time.time()
    for _ in range(50):
        with nvtx.annotate("CUDA kernel (naive)", color="red"):
            c0 = batched_matmul_cupy_naive(a, b)
    cp.cuda.Stream.null.synchronize()
    t1 = time.time()
    print(f"CUDA kernel (naive): {(t1 - t0) * 1000 / 50:.3f} ms per batch")

    # Tiled 
    t2 = time.time()
    for _ in range(50):
        with nvtx.annotate("CUDA kernel (tiled, shared)", color="blue"):
            c1 = batched_matmul_cupy_tiled(a, b, tile=16)
    cp.cuda.Stream.null.synchronize()
    t3 = time.time()
    print(f"CUDA kernel (tiled, shared): {(t3 - t2) * 1000 / 50:.3f} ms per batch")
