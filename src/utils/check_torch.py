import torch

def convert_to_tensor(data, device='cpu', dtype=torch.float32):
    """Convert input data to a PyTorch tensor on the specified device."""
    if isinstance(data, torch.Tensor):
        return data.to(device=device, dtype=dtype)
    else:
        return torch.tensor(data, device=device, dtype=dtype)