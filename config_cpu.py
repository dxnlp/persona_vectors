"""
CPU Configuration Utility for Persona Vectors

Provides centralized device detection and model loading configuration
for environments without GPU support.
"""

import torch
from typing import Optional, Dict, Any


# Device detection
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_GPU = torch.cuda.is_available()
GPU_COUNT = torch.cuda.device_count() if USE_GPU else 0


def get_device() -> torch.device:
    """Get the appropriate device for computation."""
    return DEVICE


def get_dtype() -> torch.dtype:
    """
    Get the appropriate dtype for model loading.

    - GPU: Use bfloat16 for memory efficiency
    - CPU: Use float32 (bfloat16 has limited CPU support)
    """
    if USE_GPU:
        return torch.bfloat16
    return torch.float32


def get_device_map() -> str:
    """
    Get the appropriate device_map for model loading.

    - GPU: Use "auto" for automatic distribution
    - CPU: Use "cpu" explicitly
    """
    if USE_GPU:
        return "auto"
    return "cpu"


def get_model_loading_kwargs(
    dtype: Optional[torch.dtype] = None,
    device_map: Optional[str] = None,
    low_cpu_mem_usage: bool = True,
) -> Dict[str, Any]:
    """
    Get kwargs for AutoModelForCausalLM.from_pretrained().

    Args:
        dtype: Override dtype (defaults to automatic selection)
        device_map: Override device_map (defaults to automatic selection)
        low_cpu_mem_usage: Enable low CPU memory usage mode

    Returns:
        Dictionary of kwargs for model loading
    """
    kwargs = {
        "torch_dtype": dtype or get_dtype(),
        "device_map": device_map or get_device_map(),
        "low_cpu_mem_usage": low_cpu_mem_usage,
    }
    return kwargs


def move_to_device(tensor: torch.Tensor) -> torch.Tensor:
    """Move a tensor to the appropriate device."""
    return tensor.to(DEVICE)


def is_vllm_available() -> bool:
    """
    Check if vLLM is available and usable.

    vLLM requires GPU and is not available on CPU-only systems.
    """
    if not USE_GPU:
        return False
    try:
        import vllm
        return True
    except ImportError:
        return False


def print_device_info():
    """Print information about the current device configuration."""
    print(f"Device: {DEVICE}")
    print(f"GPU Available: {USE_GPU}")
    if USE_GPU:
        print(f"GPU Count: {GPU_COUNT}")
        for i in range(GPU_COUNT):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    print(f"Default dtype: {get_dtype()}")
    print(f"vLLM available: {is_vllm_available()}")


if __name__ == "__main__":
    print_device_info()
