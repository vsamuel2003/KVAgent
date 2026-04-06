"""HuggingFace transformers backend for tau2 LLM calls.

Supports Qwen3 models with native tool calling via apply_chat_template.
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
    _instance_lock: threading.Lock = threading.Lock()   # guards singleton creation
    _inference_lock: threading.Lock = threading.Lock()  # serializes GPU inference

    def __init__(
        self,
        model_name: str,
        device_map: str = "auto",
        torch_dtype=torch.bfloat16,
    ):
        logger.info(f"Loading HuggingFace model: {model_name}")
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device_map,
            dtype=torch_dtype,
        )
        self.model.eval()
        logger.info(f"Model {model_name} loaded successfully")

    @classmethod
    def get(cls, model_name: str, **kwargs) -> "HFBackend":
        """Get or create a singleton instance for the given model (thread-safe)."""
        if model_name not in cls._instances:
            with cls._instance_lock:
                # Re-check after acquiring lock (double-checked locking)
                if model_name not in cls._instances:
                    cls._instances[model_name] = cls(model_name, **kwargs)
        return cls._instances[model_name]

    def generate_chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        max_new_tokens: int = 4096,
        temperature: float = 0.0,
        **kwargs,
    ) -> dict:
        """
        Generate a chat completion using transformers.

        Args:
            messages: OpenAI-format message dicts
            tools: OpenAI-format tool schemas (list of {"type":"function","function":{...}})
            max_new_tokens: Max tokens to generate
            temperature: Sampling temperature (0.0 = greedy)

        Returns:
            dict with keys: content, tool_calls, prompt_tokens, completion_tokens
        """
        # Apply Qwen3 chat template (supports tools parameter natively)
        template_kwargs: dict = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        # Qwen3-specific: disable thinking mode for faster/cleaner output
        try:
            template_kwargs["enable_thinking"] = False
        except Exception:
            pass

        if tools:
            template_kwargs["tools"] = tools

        text = self.tokenizer.apply_chat_template(messages, **template_kwargs)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        prompt_tokens = inputs.input_ids.shape[1]

        gen_kwargs: dict = {"max_new_tokens": max_new_tokens}
        if temperature > 0:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = temperature
        else:
            gen_kwargs["do_sample"] = False

        with HFBackend._inference_lock:
            with torch.no_grad():
                outputs = self.model.generate(**inputs, **gen_kwargs)

        new_tokens = outputs[0][prompt_tokens:]
        completion_tokens = len(new_tokens)
        raw_output = self.tokenizer.decode(new_tokens, skip_special_tokens=False)

        content, tool_calls = _parse_qwen3_output(raw_output)

        return {
            "content": content,
            "tool_calls": tool_calls,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        }


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
