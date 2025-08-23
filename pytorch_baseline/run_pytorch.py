# pytorch_baseline/run_pytorch.py
import torch
import time
import nvtx
def pytorch_batched_matmul(a, b):
    return torch.matmul(a, b)

if __name__ == "__main__":
    torch.manual_seed(42)
    batch, m, n, k = 128, 64, 64, 64
    a = torch.randn(batch, m, k, device="cuda")
    b = torch.randn(batch, k, n, device="cuda")

    # Warmup
    for _ in range(10):
        pytorch_batched_matmul(a, b)
    torch.cuda.synchronize()

    
    with nvtx.annotate("LOOP: PyTorch matmul x50", color="green"):
        t0 = time.time()
        for _ in range(50):
            with nvtx.annotate("KERNEL: PyTorch matmul", color="green"):
                out = pytorch_batched_matmul(a, b)
        torch.cuda.synchronize()
        t1 = time.time()
    print(f"PyTorch batched matmul: {(t1 - t0) * 1000 / 50:.3f} ms per batch")

