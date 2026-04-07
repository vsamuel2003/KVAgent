"""HuggingFace transformers backend for tau2 LLM calls.

Supports Qwen3 models with native tool calling via apply_chat_template.

GPU assignment strategy:
- If multiple CUDA GPUs are available, each distinct model name is assigned to a
  different GPU in round-robin order (first model → cuda:0, second → cuda:1, etc.).
- Inference locks are per-device, so models on different GPUs can run in parallel.
- If only one GPU (or CPU) is available, all models share it with a single
  serialized inference lock.
"""
import json
import re
import threading
import uuid

import torch
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer


class HFBackend:
    """Singleton-per-model manager for HuggingFace model + tokenizer."""

    _instances: dict[str, "HFBackend"] = {}
    _instance_lock: threading.Lock = threading.Lock()  # guards singleton creation

    # Per-device inference locks: models on different GPUs can run in parallel.
    _device_inference_locks: dict[str, threading.Lock] = {}
    _device_locks_lock: threading.Lock = threading.Lock()

    # Round-robin GPU assignment counter (protected by _instance_lock)
    _next_gpu_idx: int = 0

    def __init__(
        self,
        model_name: str,
        device_map: str = "auto",
        torch_dtype=torch.bfloat16,
    ):
        logger.info(f"Loading HuggingFace model: {model_name} → device_map={device_map}")
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device_map,
            dtype=torch_dtype,
        )
        self.model.eval()
        # Record the effective device string for per-device locking
        self._device_str = _device_str(self.model)
        logger.info(
            f"Model {model_name} loaded successfully on {self._device_str}"
        )

    @classmethod
    def get(cls, model_name: str, device_map: str | None = None, **kwargs) -> "HFBackend":
        """Get or create a singleton instance for the given model (thread-safe).

        If device_map is None and multiple CUDA GPUs are available, models are
        assigned to GPUs in round-robin order so that distinct models land on
        different devices.
        """
        if model_name not in cls._instances:
            with cls._instance_lock:
                if model_name not in cls._instances:
                    if device_map is None:
                        device_map = cls._assign_device()
                    cls._instances[model_name] = cls(
                        model_name, device_map=device_map, **kwargs
                    )
        return cls._instances[model_name]

    @classmethod
    def _assign_device(cls) -> str:
        """Return the next CUDA device string in round-robin order.

        Falls back to "auto" when fewer than 2 GPUs are available so that
        transformers can handle single-GPU or CPU placement automatically.
        """
        num_gpus = torch.cuda.device_count()
        if num_gpus >= 2:
            device = f"cuda:{cls._next_gpu_idx % num_gpus}"
            cls._next_gpu_idx += 1
            return device
        return "auto"

    @classmethod
    def _get_inference_lock(cls, device_str: str) -> threading.Lock:
        """Return (creating if needed) the inference lock for a given device."""
        with cls._device_locks_lock:
            if device_str not in cls._device_inference_locks:
                cls._device_inference_locks[device_str] = threading.Lock()
            return cls._device_inference_locks[device_str]

    def generate_chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        repetition_penalty: float = 1.1,
        **kwargs,
    ) -> dict:
        """
        Generate a chat completion using transformers.

        Args:
            messages: OpenAI-format message dicts
            tools: OpenAI-format tool schemas (list of {"type":"function","function":{...}})
            max_new_tokens: Max tokens to generate (default 512 to avoid slow runs)
            temperature: Sampling temperature; >0 enables do_sample=True (default 0.7)
            repetition_penalty: Penalise repeated tokens to break loops (default 1.1)

        Returns:
            dict with keys: content, tool_calls, prompt_tokens, completion_tokens
        """
        # Apply Qwen3 chat template (supports tools and enable_thinking natively)
        template_kwargs: dict = {
            "tokenize": False,
            "add_generation_prompt": True,
            "enable_thinking": False,  # disable chain-of-thought for speed
        }
        if tools:
            template_kwargs["tools"] = tools

        text = self.tokenizer.apply_chat_template(messages, **template_kwargs)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        prompt_tokens = inputs.input_ids.shape[1]

        gen_kwargs: dict = {
            "max_new_tokens": max_new_tokens,
            "repetition_penalty": repetition_penalty,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        if temperature > 0:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = temperature
        else:
            gen_kwargs["do_sample"] = False

        device = self.model.device
        cuda_available = device.type == "cuda"

        if cuda_available:
            mem_before = torch.cuda.memory_allocated(device)

        inference_lock = HFBackend._get_inference_lock(self._device_str)
        with inference_lock:
            with torch.no_grad():
                outputs = self.model.generate(**inputs, **gen_kwargs)

        if cuda_available:
            mem_after = torch.cuda.memory_allocated(device)
        else:
            mem_before = mem_after = 0

        new_tokens = outputs[0][prompt_tokens:]
        completion_tokens = len(new_tokens)
        raw_output = self.tokenizer.decode(new_tokens, skip_special_tokens=False)

        content, tool_calls = _parse_qwen3_output(raw_output)

        # Analytical KV cache size estimate (bfloat16 = 2 bytes per element)
        cfg = self.model.config
        num_layers = getattr(cfg, "num_hidden_layers", 0)
        num_kv_heads = getattr(cfg, "num_key_value_heads", getattr(cfg, "num_attention_heads", 0))
        head_dim = getattr(cfg, "hidden_size", 0) // max(getattr(cfg, "num_attention_heads", 1), 1)
        kv_bytes = 2 * num_layers * num_kv_heads * head_dim * prompt_tokens * 2

        return {
            "content": content,
            "tool_calls": tool_calls,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "gpu_memory_before_mb": mem_before / (1024 ** 2),
            "gpu_memory_after_mb": mem_after / (1024 ** 2),
            "kv_cache_estimate_mb": kv_bytes / (1024 ** 2),
        }


def _device_str(model: torch.nn.Module) -> str:
    """Return a canonical device string for the model's first parameter."""
    try:
        return str(next(model.parameters()).device)
    except StopIteration:
        return "cpu"


def _parse_qwen3_output(raw_output: str) -> tuple[str | None, list[dict] | None]:
    """Parse Qwen 3 Hermes-style tool call output.

    Qwen3 format: <tool_call>{"name":"func","arguments":{"k":"v"}}</tool_call>
    May contain multiple tool calls or just plain text.
    """
    # Strip common end-of-sequence tokens
    for eos_token in ["<|endoftext|>", "<|im_end|>", "<|end|>"]:
        raw_output = raw_output.replace(eos_token, "")
    raw_output = raw_output.strip()

    # Find all <tool_call>...</tool_call> blocks
    pattern = r"<tool_call>(.*?)</tool_call>"
    matches = re.findall(pattern, raw_output, re.DOTALL)

    if not matches:
        return raw_output if raw_output else None, None

    tool_calls = []
    for match in matches:
        try:
            tc = json.loads(match.strip())
            tool_calls.append(
                {
                    "id": f"call_{uuid.uuid4().hex[:8]}",
                    "name": tc["name"],
                    "arguments": tc.get("arguments", tc.get("parameters", {})),
                }
            )
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to parse tool call: {match!r} — {e}")
            continue

    # Remaining non-tool-call text becomes content
    text_content = re.sub(pattern, "", raw_output, flags=re.DOTALL).strip()
    content = text_content if text_content else None

    return content, tool_calls if tool_calls else None
