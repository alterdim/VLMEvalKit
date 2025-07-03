"""
gemma.py – patched version
✓ one-turn template (no explicit `system` role)
✓ images handled the same way in vLLM & transformers
✓ `<end_of_turn>` wired as EOS/stop token
"""

import logging
import base64
from io import BytesIO
from mimetypes import guess_type
from PIL import Image

import torch

from .base import BaseModel
from ..smp import *  # whatever your project uses


# --------------------------------------------------------------------------- #
# PaliGemma – unchanged except for tiny style tweaks                          #
# --------------------------------------------------------------------------- #
class PaliGemma(BaseModel):
    INSTALL_REQ = False
    INTERLEAVE = False

    def __init__(self, model_path="google/paligemma-3b-mix-448", **kwargs):
        from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

        self.model = (
            PaliGemmaForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map="cpu",
                revision="bfloat16",
            )
            .eval()
            .cuda()
        )
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.kwargs = kwargs
        self.kwargs["eos_token_id"] = None
        self.stop_strings = ["```", "<end_of_turn>"]

    # text-only helper reused by both classes
    @staticmethod
    def _rgba_to_rgb(image: Image.Image) -> Image.Image:
        bg = Image.new("RGBA", image.size, (255, 255, 255, 255))
        return Image.alpha_composite(bg, image).convert("RGB")

    def generate_inner(self, message, dataset=None):
        prompt, image_path = self.message_to_promptimg(message, dataset=dataset)
        image = Image.open(image_path).convert("RGB")

        model_inputs = self.processor(
            text=prompt, images=image, return_tensors="pt"
        ).to("cuda")
        input_len = model_inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            generation = self.model.generate(
                **model_inputs,
                max_new_tokens=512,
                do_sample=False,
            )[0][input_len:]

        return self.processor.decode(generation, skip_special_tokens=True)


# --------------------------------------------------------------------------- #
# Gemma-3 4B-IT                                                               #
# --------------------------------------------------------------------------- #
class Gemma3(BaseModel):
    INSTALL_REQ = False
    INTERLEAVE = True  # multiple modalities in the same prompt

    def __init__(self, model_path="google/gemma-3-4b-it", **kwargs):
        logging.info(
            "You need the Gemma-3 branch of transformers:\n"
            "  pip install git+https://github.com/huggingface/transformers@v4.49.0-Gemma-3"
        )

        from transformers import AutoProcessor, Gemma3ForConditionalGeneration

        # ------------------------------------------------------------------ #
        # vLLM or transformers backend                                       #
        # ------------------------------------------------------------------ #
        self.use_vllm: bool = kwargs.get("use_vllm", False)
        self.limit_mm_per_prompt = 24  # hard cap in Gemma tokenizer

        if self.use_vllm:
            from vllm import LLM

            gpu_count = torch.cuda.device_count()
            tp_size = 8 if gpu_count >= 8 else 4 if gpu_count >= 4 else 2 if gpu_count >= 2 else 1
            logging.info(f"vLLM backend with tensor_parallel_size={tp_size}")

            self.llm = LLM(
                model=model_path,
                max_num_seqs=4,
                max_model_len=16_384,
                limit_mm_per_prompt={"image": self.limit_mm_per_prompt},
                tensor_parallel_size=tp_size,
                gpu_memory_utilization=kwargs.get("gpu_utils", 0.9),
            )
        else:
            self.model = (
                Gemma3ForConditionalGeneration.from_pretrained(
                    model_path,
                    attn_implementation="flash_attention_2",
                    torch_dtype=torch.bfloat16,
                    device_map="cuda",
                )
                .eval()
            )
            self.device = self.model.device

        # one shared processor
        self.processor = AutoProcessor.from_pretrained(model_path)

        # ------------------------------------------------------------------ #
        # runtime options                                                    #
        # ------------------------------------------------------------------ #
        self.system_prompt = kwargs.pop(
    "system_prompt",
    (
        "You are **Chart2Code-GPT**. "
        "For every request, reply with **ONLY** a valid Python/matplotlib script "
        "that recreates the input chart. "
        "Begin your answer on the very first line with ```python and finish "
        "with ``` on its own line. "
        "Return nothing else—no explanations, no markdown outside the fence."
    ),
)
        self.eot_id = self.processor.tokenizer.convert_tokens_to_ids("<end_of_turn>")
        default_kwargs = {
            "do_sample": False,
            "max_new_tokens": 4096,
            "eos_token_id": self.eot_id,
        }
        default_kwargs.update(kwargs)
        self.kwargs = default_kwargs

    # ---------------------------------------------------------------------- #
    # helpers                                                                #
    # ---------------------------------------------------------------------- #
    @staticmethod
    def _rgba_to_rgb(image: Image.Image) -> Image.Image:
        bg = Image.new("RGBA", image.size, (255, 255, 255, 255))
        return Image.alpha_composite(bg, image).convert("RGB")

    def _encode_image_file(self, path: str) -> str:
        """Return base64-encoded RGB JPEG/PNG."""
        mime_type, _ = guess_type(path)
        fmt = (mime_type or "image/jpeg").split("/")[-1].upper()
        img = Image.open(path)
        if img.mode == "RGBA":
            img = self._rgba_to_rgb(img)

        buf = BytesIO()
        img.save(buf, format=fmt)
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    # ---------- chat construction ---------------------------------------- #
    def _build_chat(self, message) -> list:
        """
        Build a *single* Gemma-3 chat turn containing:
          • optional system text (prepended to first user line)
          • text and <image> placeholders in user turn
        """
        content = []
        if self.system_prompt:
            content.append({"type": "text", "text": self.system_prompt.strip() + "\n"})

        for chunk in message:
            if chunk["type"] == "text":
                content.append({"type": "text", "text": chunk["value"]})
            elif chunk["type"] == "image":
                if len(content) >= self.limit_mm_per_prompt:
                    logging.warning("Too many images; extra ones will be ignored.")
                    continue
                content.append(
                    {"type": "image", "image": self._encode_image_file(chunk["value"])}
                )

        return [{"role": "user", "content": content}]

    def _prompt_and_images(self, message):
        """Return (prompt str, list[PIL Image]) for both back-ends."""
        chat = self._build_chat(message)
        prompt = self.processor.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        prompt += "```python\n" 

        images = [
            Image.open(BytesIO(base64.b64decode(c["image"]))).convert("RGB")
            for c in chat[0]["content"]
            if c["type"] == "image"
        ]
        return prompt, images

    # ---------------------------------------------------------------------- #
    # generation paths                                                       #
    # ---------------------------------------------------------------------- #
    def _generate_transformers(self, message):
        prompt, images = self._prompt_and_images(message)
        inputs = self.processor(
            text=prompt,
            images=images if images else None,
            return_tensors="pt",
        ).to(self.device, dtype=torch.bfloat16)

        input_len = inputs["input_ids"].shape[-1]
        with torch.inference_mode():
            gen = self.model.generate(**inputs, **self.kwargs)[0][input_len:]
        return self.processor.decode(gen, skip_special_tokens=True)

    def _generate_vllm(self, message):
        from vllm import SamplingParams

        prompt, images = self._prompt_and_images(message)
        params = SamplingParams(
         temperature=0.0,
         max_tokens=self.kwargs["max_new_tokens"],
         stop=["```", "<end_of_turn>"],
        )
        outputs = self.llm.generate(
            {"prompt": prompt, "multi_modal_data": {"image": images}}, params
        )
        return outputs[0].outputs[0].text

    # public entry-point ---------------------------------------------------- #
    def generate_inner(self, message, dataset=None):
        if self.use_vllm:
            return self._generate_vllm(message)
        return self._generate_transformers(message)
