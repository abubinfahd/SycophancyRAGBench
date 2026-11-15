"""
Unified Chat Model interface across multiple contemporary LLM families:
- Alibaba Qwen (Qwen2.5-3B-Instruct, Qwen2.5-7B-Instruct)
- Google Gemma (gemma-3-4b-it)
- Meta Llama (Llama-3.2-3B-Instruct, Llama-3.1-8B-Instruct)
- Mistral (Mistral-7B-Instruct-v0.3)

Handles 4-bit NF4 quantization, family-specific chat templates, left-padding, and memory cleanup.
"""

import os
import gc
from typing import List, Dict, Any, Optional

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
except (ImportError, ModuleNotFoundError):
    torch = None
    AutoTokenizer = None
    AutoModelForCausalLM = None
    BitsAndBytesConfig = None

from configs import config as cfg


def get_bnb_config(compute_dtype) -> Any:
    """Return BitsAndBytes 4-bit NF4 double-quantization configuration."""
    if BitsAndBytesConfig is None:
        return None
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True
    )


def _safe_no_grad(fn):
    if torch is not None:
        return torch.no_grad()(fn)
    return fn

class ChatModel:
    """
    Uniform batched chat interface supporting multiple model families.
    """
    def __init__(self, model_spec: Dict[str, Any], load_in_4bit: bool = cfg.LOAD_IN_4BIT):
        if torch is None:
            raise ImportError("PyTorch and Transformers are required for ChatModel. Please install torch and transformers.")

        self.spec = model_spec
        self.name = model_spec["name"]
        self.hf_id = model_spec["hf_id"]
        self.family = model_spec.get("family", "causal")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Gemma-3 overflows in fp16 (producing empty or NaN outputs) -> must use bfloat16.
        # Qwen and Llama operate reliably in float16 or bfloat16.
        if self.family == "gemma3" or (torch.cuda.is_available() and torch.cuda.is_bf16_supported()):
            self.dtype = torch.bfloat16
        else:
            self.dtype = torch.float16

        quant = get_bnb_config(self.dtype) if (load_in_4bit and torch.cuda.is_available()) else None

        print(f"[Model] Loading {self.name} ({self.hf_id}) [family={self.family}, dtype={self.dtype}, 4bit={bool(quant)}] ...")

        if self.family == "gemma3":
            try:
                from transformers import Gemma3ForConditionalGeneration, AutoProcessor
                self.processor = AutoProcessor.from_pretrained(self.hf_id)
                self.tok = self.processor.tokenizer
                self.model = Gemma3ForConditionalGeneration.from_pretrained(
                    self.hf_id,
                    quantization_config=quant,
                    torch_dtype=self.dtype,
                    device_map="auto" if torch.cuda.is_available() else None,
                    attn_implementation="eager"
                )
            except Exception as e:
                print(f"[Model] Special Gemma3 loader fallback: {e}")
                self.tok = AutoTokenizer.from_pretrained(self.hf_id)
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.hf_id,
                    quantization_config=quant,
                    torch_dtype=self.dtype,
                    device_map="auto" if torch.cuda.is_available() else None
                )
        else:
            self.tok = AutoTokenizer.from_pretrained(self.hf_id)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.hf_id,
                quantization_config=quant,
                torch_dtype=self.dtype,
                device_map="auto" if torch.cuda.is_available() else None
            )

        # Padding setup for left-padded batch generation
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        self.tok.padding_side = "left"
        self.model.eval()
        print(f"[Model] Successfully initialized {self.name}.")

    def _format_messages(self, system: str, user: str) -> List[Dict[str, Any]]:
        """Format dialogue turns according to model family conventions."""
        if self.family == "gemma3":
            text = f"{system}\n\n{user}" if system else user
            return [{"role": "user", "content": [{"type": "text", "text": text}]}]

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})
        return messages

    def _build_prompt_str(self, system: str, user: str) -> str:
        """Apply model's chat template to produce formatted prompt string."""
        messages = self._format_messages(system, user)
        return self.tok.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

    @_safe_no_grad
    def generate(
        self,
        system_prompts: List[str],
        user_prompts: List[str],
        max_new_tokens: int = cfg.MAX_NEW_TOKENS
    ) -> List[str]:
        """Batched generation given lists of system and user prompt strings."""
        formatted_prompts = [
            self._build_prompt_str(s, u)
            for s, u in zip(system_prompts, user_prompts)
        ]

        enc = self.tok(
            formatted_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=3072,
            add_special_tokens=False
        ).to(self.model.device)

        outputs = self.model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.tok.pad_token_id
        )

        gen_tokens = outputs[:, enc["input_ids"].shape[1]:]
        decoded = self.tok.batch_decode(gen_tokens, skip_special_tokens=True)
        return [t.strip() for t in decoded]

    def close(self):
        """Release GPU memory and destroy model instance."""
        try:
            del self.model
            del self.tok
        except Exception:
            pass
        gc.collect()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"[Model] Cleaned up GPU memory for {self.name}.")
