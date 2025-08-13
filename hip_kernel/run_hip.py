# hip_kernel/run_hip.py
import os, time, sys

# 僅在 ROCm/AMD 上執行；NVIDIA 會直接略過
try:
    import cupy as cp
    IS_HIP = getattr(cp.cuda.runtime, "is_hip", False)
except Exception:
    IS_HIP = False

def main():
    if not IS_HIP:
        print("HIP runtime not detected; skipping HIP benchmark.")
        sys.exit(0)

    this_dir = os.path.dirname(os.path.abspath(__file__))
    hip_path = os.path.join(this_dir, "batched_matmul_hip.hip")
    with open(hip_path, "r") as f:
        kernel = f.read()

    # 用 CuPy RawModule 在 ROCm 環境下以 HIP 編譯
    module = cp.RawModule(code=kernel, backend='nvrtc')  # CuPy 會在 ROCm 走 HIP backend
    hip_kernel = module.get_function("batched_matmul_kernel")

    
    batch, M, K, N = 128, 64, 64, 64
    a = cp.random.randn(batch, M, K, dtype=cp.float32)
    b = cp.random.randn(batch, K, N, dtype=cp.float32)

    threads = (16, 16, 1)
    blocks  = ((N + 15)//16, (M + 15)//16, batch)

    # warmup
    for _ in range(10):
        C = cp.empty((batch, M, N), dtype=cp.float32)
        hip_kernel(blocks, threads, (a, b, C, batch, M, K, N))
    cp.cuda.Stream.null.synchronize()

    t0 = time.time()
    for _ in range(50):
        C = cp.empty((batch, M, N), dtype=cp.float32)   
        hip_kernel(blocks, threads, (a, b, C, batch, M, K, N))
    cp.cuda.Stream.null.synchronize()
    t1 = time.time()
    print(f"HIP kernel batched matmul: {(t1 - t0) * 1000 / 50:.3f} ms per batch")

if __name__ == "__main__":
    main()
