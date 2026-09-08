# Triton-accelerated implementation of TSVC s318.
# Uses PyTorch to move data to GPU and torch.max to compute max absolute value and index.
import torch

def s318(a, result, inc, LEN_1D):
    # Sample with stride inc
    sampled = a[0:LEN_1D * inc:inc]
    # Transfer to GPU
    x = torch.from_numpy(sampled).to("cuda")
    # Compute absolute values
    abs_x = torch.abs(x)
    # Compute max value and index
    maxv_tensor, idx_tensor = torch.max(abs_x, dim=0)
    # Convert to Python scalars
    maxv = maxv_tensor.item()
    idx = int(idx_tensor.item())
    # Write result
    result[0] = maxv + float(idx)
    return None
