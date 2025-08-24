# cuda_kernel/batched_matmul.py
import cupy as cp
import time
import nvtx
import os

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
cu_naive_path    = os.path.join(THIS_DIR, "batched_matmul.cu")
cu_tiled_path    = os.path.join(THIS_DIR, "batched_matmul_tiled.cu")      
cu_tiled_O2_path = os.path.join(THIS_DIR, "batched_matmul_tiled_O2.cu")   
cu_tiled_O2_1_path = os.path.join(THIS_DIR, "batched_matmul_tiled_O2_1.cu")

with open(cu_naive_path, "r") as f:
    naive_code = f.read()
with open(cu_tiled_path, "r") as f:
    tiled_code = f.read()
with open(cu_tiled_O2_path, "r") as f:
    tiled_O2_code = f.read()
with open(cu_tiled_O2_1_path, "r") as f:
    tiled_O2_1_code = f.read()

# === modules ===
mod_naive     = cp.RawModule(code=naive_code)
mod_tiled     = cp.RawModule(code=tiled_code) 
mod_tiled_O2  = cp.RawModule(code=tiled_O2_code, options=("-DTILE=32",))  
mod_tiled_O2_1 = cp.RawModule(code=tiled_O2_1_code, options=("-DTILE=32",))

# === kernels ===
kern_naive     = mod_naive.get_function("batched_matmul_kernel")
kern_tiled     = mod_tiled.get_function("batched_matmul_tiled_kernel")          
kern_tiled_O2  = mod_tiled_O2.get_function("batched_matmul_tiled_kernel_O2")    
kern_tiled_O2_1 = mod_tiled_O2_1.get_function("batched_matmul_tiled_kernel_O2_1")

def batched_matmul_cupy_naive(a, b):
    batch, M, K = a.shape
    _, _, N = b.shape
    c = cp.zeros((batch, M, N), dtype=cp.float32)
    threads = (16, 16, 1)
    blocks  = ((N + 15)//16, (M + 15)//16, batch)
    kern_naive((blocks), (threads), (a, b, c, batch, M, K, N))
    return c

# ===== 原本的 tiled =====
def batched_matmul_cupy_tiled(a, b, tile=16):
    batch, M, K = a.shape
    _, _, N = b.shape
    c = cp.zeros((batch, M, N), dtype=cp.float32)
    threads = (tile, tile, 1)                             
    blocks  = ((N + tile - 1)//tile, (M + tile - 1)//tile, batch)
    kern_tiled((blocks), (threads), (a, b, c, batch, M, K, N))
    return c

# ===== O2 版本：TILE=32、float2、每 thread 算兩個 col（x 維對半）=====
def batched_matmul_cupy_tiled_O2(a, b, tile=32):
    batch, M, K = a.shape
    _, _, N = b.shape
    c = cp.zeros((batch, M, N), dtype=cp.float32)
    threads = (tile // 2, tile, 1)                          # ← 2 cols / thread
    blocks  = ((N + tile - 1)//tile, (M + tile - 1)//tile, batch)
    kern_tiled_O2((blocks), (threads), (a, b, c, batch, M, K, N))
    return c

# ===== O2_1 版本：TILE=32、float4、每 thread 算四個 result（x,y 維對半）=====
def batched_matmul_cupy_tiled_O2_1(a, b, tile=32):
    # 2x2 register blocking → threads=(tile/2, tile/2, 1)
    batch, M, K = a.shape
    _, _, N = b.shape
    c = cp.zeros((batch, M, N), dtype=cp.float32)
    threads = (tile // 2, tile // 2, 1)
    blocks  = ((N + tile - 1)//tile, (M + tile - 1)//tile, batch)
    kern_tiled_O2_1((blocks), (threads), (a, b, c, batch, M, K, N))
    return c

if __name__ == "__main__":
    batch, M, K, N = 128, 64, 64, 64
    a = cp.random.randn(batch, M, K).astype(cp.float32)
    b = cp.random.randn(batch, K, N).astype(cp.float32)

    # warmup
    for _ in range(10):
        batched_matmul_cupy_naive(a, b)
        batched_matmul_cupy_tiled(a, b, tile=16)     
        batched_matmul_cupy_tiled_O2(a, b, tile=32)  
    cp.cuda.Stream.null.synchronize()

# Naive (outer loop + per-kernel NVTX)
with nvtx.annotate("LOOP: naive x50", color="red"):
    t0 = time.time()
    for _ in range(50):
        with nvtx.annotate("KERNEL: naive", color="red"):
            c0 = batched_matmul_cupy_naive(a, b)
    cp.cuda.Stream.null.synchronize()
    t1 = time.time()
print(f"CUDA kernel (naive): {(t1 - t0) * 1000 / 50:.3f} ms per batch")

# Tiled (original)
with nvtx.annotate("LOOP: tiled16 x50", color="blue"):
    t2 = time.time()
    for _ in range(50):
        with nvtx.annotate("KERNEL: tiled16", color="blue"):
            c1 = batched_matmul_cupy_tiled(a, b, tile=16)
    cp.cuda.Stream.null.synchronize()
    t3 = time.time()
print(f"CUDA kernel (tiled16): {(t3 - t2) * 1000 / 50:.3f} ms per batch")

# Tiled O2 (float2 + 2col/thread, TILE=32)
with nvtx.annotate("LOOP: tiled_O2 x50", color="green"):
    t4 = time.time()
    for _ in range(50):
        with nvtx.annotate("KERNEL: tiled_O2", color="green"):
            c2 = batched_matmul_cupy_tiled_O2(a, b, tile=32)
    cp.cuda.Stream.null.synchronize()
    t5 = time.time()
print(f"CUDA kernel (tiled32 O2): {(t5 - t4) * 1000 / 50:.3f} ms per batch")

# Tiled O2_1 (2x2 + float4)
with nvtx.annotate("LOOP: tiled_O2_1 x50", color="yellow"):
    t6 = time.time()
    for _ in range(50):
        with nvtx.annotate("KERNEL: tiled_O2_1", color="yellow"):
            c3 = batched_matmul_cupy_tiled_O2_1(a, b, tile=32)
    cp.cuda.Stream.null.synchronize()
    t7 = time.time()
print(f"CUDA kernel (tiled32 O2_1): {(t7 - t6) * 1000 / 50:.3f} ms per batch")


