import os, re, json, torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftConfig, PeftModel
import socket

# Import CPU configuration utilities
from config_cpu import (
    USE_GPU,
    GPU_COUNT,
    get_dtype,
    get_device_map,
    get_model_loading_kwargs,
    is_vllm_available,
)





def get_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))  
        return s.getsockname()[1]


# ---------- 工具 ----------
_CHECKPOINT_RE = re.compile(r"checkpoint-(\d+)")

def _pick_latest_checkpoint(model_path: str) -> str:
    ckpts = [(int(m.group(1)), p) for p in Path(model_path).iterdir()
             if (m := _CHECKPOINT_RE.fullmatch(p.name)) and p.is_dir()]
    return str(max(ckpts, key=lambda x: x[0])[1]) if ckpts else model_path

def _is_lora(path: str) -> bool:
    return Path(path, "adapter_config.json").exists()

def _load_and_merge_lora(lora_path: str, dtype=None, device_map=None):
    """Load and merge LoRA adapter with base model.

    Args:
        lora_path: Path to LoRA adapter
        dtype: Model dtype (defaults to automatic selection based on device)
        device_map: Device map (defaults to automatic selection based on device)
    """
    if dtype is None:
        dtype = get_dtype()
    if device_map is None:
        device_map = get_device_map()

    cfg = PeftConfig.from_pretrained(lora_path)
    base = AutoModelForCausalLM.from_pretrained(
        cfg.base_model_name_or_path, torch_dtype=dtype, device_map=device_map
    )
    return PeftModel.from_pretrained(base, lora_path).merge_and_unload()

def _load_tokenizer(path_or_id: str):
    tok = AutoTokenizer.from_pretrained(path_or_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id
    tok.padding_side = "left"
    return tok

def load_model(model_path: str, dtype=None, device_map=None):
    """Load model with automatic CPU/GPU detection.

    Args:
        model_path: HuggingFace model ID or local path
        dtype: Model dtype (defaults to bfloat16 on GPU, float32 on CPU)
        device_map: Device map (defaults to "auto" on GPU, "cpu" on CPU)
    """
    if dtype is None:
        dtype = get_dtype()
    if device_map is None:
        device_map = get_device_map()

    if not os.path.exists(model_path):               # ---- Hub ----
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=dtype, device_map=device_map
        )
        tok = _load_tokenizer(model_path)
        return model, tok

    resolved = _pick_latest_checkpoint(model_path)
    print(f"loading {resolved}")
    if _is_lora(resolved):
        model = _load_and_merge_lora(resolved, dtype, device_map)
        tok = _load_tokenizer(model.config._name_or_path)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            resolved, torch_dtype=dtype, device_map=device_map
        )
        tok = _load_tokenizer(resolved)
    return model, tok

def load_vllm_model(model_path: str):
    """Load model using vLLM for faster inference.

    Note: vLLM requires GPU. On CPU-only systems, this will raise an error.
    Use load_model() instead for CPU inference.
    """
    if not is_vllm_available():
        raise RuntimeError(
            "vLLM is not available. This requires GPU support. "
            "Use load_model() for CPU inference instead."
        )

    from vllm import LLM

    # Get dtype for vLLM (use "half" for float16, "bfloat16" for bfloat16)
    dtype = get_dtype()
    vllm_dtype = "half" if dtype == torch.float16 else "bfloat16"

    if not os.path.exists(model_path):               # ---- Hub ----
        llm = LLM(
            model=model_path,
            dtype=vllm_dtype,
            enable_prefix_caching=True,
            enable_lora=True,
            tensor_parallel_size=GPU_COUNT,
            max_num_seqs=32,
            gpu_memory_utilization=0.9,
            max_model_len=8192,
            max_lora_rank=128,
        )
        tok = llm.get_tokenizer()
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
            tok.pad_token_id = tok.eos_token_id
        tok.padding_side = "left"
        return llm, tok, None

    # ---- Local ----
    resolved = _pick_latest_checkpoint(model_path)
    print(f"loading {resolved}")
    is_lora = _is_lora(resolved)

    base_path = (PeftConfig.from_pretrained(resolved).base_model_name_or_path
                 if is_lora else resolved)

    llm = LLM(
        model=base_path,
        dtype=vllm_dtype,
        enable_prefix_caching=True,
        enable_lora=True,
        tensor_parallel_size=GPU_COUNT,
        max_num_seqs=32,
        gpu_memory_utilization=0.9,
        max_model_len=20000,
        max_lora_rank=128,
    )

    if is_lora:
        lora_path = resolved
    else:
        lora_path = None

    tok = llm.get_tokenizer()
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id
    tok.padding_side = "left"
    return llm, tok, lora_path
