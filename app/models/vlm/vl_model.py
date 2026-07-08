# app/models/vl_model.py

import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from PIL import Image
from transformers import TextIteratorStreamer
import gc

from app.utils.device import resolve_device, resolve_dtype
from app.utils.util import load_prompt_by_key

class VLModel:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-VL-4B-Instruct",
        max_new_tokens: int = 256,
    ):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens

        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name,
            dtype="auto",
            device_map="auto",
        )
        self.model.eval()

        self.system_role = load_prompt_by_key("system_role")
        self.vlm_task = load_prompt_by_key("vlm_task")
        self.clothes_caption = load_prompt_by_key("clothes_caption")
        self.bg_caption = load_prompt_by_key("bg_caption")
        self.choose_best_clothes_prompt = load_prompt_by_key("choose_best_clothes")
        self.suggest_outfit_prompt = load_prompt_by_key("suggest_outfit")

    # -------------------------
    # Core generation helper
    # -------------------------
    

    def release(self):
        #self.model.to("cpu")
        del self.model
        torch.cuda.empty_cache()

    @torch.no_grad()
    def _generate(self, messages: list[dict]) -> str:
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)

        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )

        output = self.processor.batch_decode(
            generated_ids[:, inputs.input_ids.shape[1]:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        print("\n[Qwen] Assistant:\n", output, "\n", flush=True)
        
        del generated_ids
        del inputs
        torch.cuda.empty_cache()
        gc.collect()
        
        return output.strip()
    
    def resize_image(
        self,
        image: Image.Image,
        max_side: int = 512,
    ) -> Image.Image:
        """
        Resize image so that the longest side == max_side
        while preserving aspect ratio.
        """

        w, h = image.size
        if max(w, h) <= max_side:
            return image

        scale = max_side / max(w, h)
        new_size = (int(w * scale), int(h * scale))

        return image.resize(new_size, Image.BICUBIC)


    # -------------------------
    # Image → clothing captions
    # -------------------------
    def generate_clothing_from_image(self, image_path: str) -> list[str]:
        """
        Given an image path, generate exactly 10 clothing descriptions.
        """

        image = Image.open(image_path).convert("RGB")
        image = self.resize_image(image, max_side=512)

        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": self.system_role}],
            },   
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": self.vlm_task},
                ],
            },
        ]

        output = self._generate(messages)

        paragraphs = [p.strip() for p in output.splitlines() if p.strip()]
        return (paragraphs + [""] * 10)[:10]
    
    def generate_clothes_caption(self, image_path: str, prompt: str) -> str:

        image = Image.open(image_path).convert("RGB")
        image = self.resize_image(image, max_side=512)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            },
        ]

        output = self._generate(messages)

        return " ".join(output.strip().split())
    def load(self):
        return None
    def choose_best_clothes(
        self,
        background_caption: str,
        candidates: list[tuple[str, str]],
    ) -> str:
        """
        Pick the best-matching clothing item for a background description.
        Returns the chosen filename from candidates.
        """

        candidates_text = "\n".join(
            f"- {name}: {caption}" for name, caption in candidates
        )
        prompt = self.choose_best_clothes_prompt.format(
            background_caption=background_caption,
            candidates_text=candidates_text,
        )

        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}],
            },
        ]

        output = self._generate(messages)
        cleaned = output.strip().strip('"').strip("'")

        for name, _ in candidates:
            if name in output:
                return name
            if cleaned == name:
                return name

        return candidates[0][0]
    
    def extract_query_signals(self, image_path: str) -> dict:
        """
        Extract all signals needed for retrieval, using existing prompts.
        """
        image = Image.open(image_path).convert("RGB")
        image = self.resize_image(image)

        # Color harmony (Baseline 1/2)
        color_outfits = self.generate_clothing_from_image(image_path)

        # Semantic scene understanding (Baseline 3)
        scene_caption = self.generate_clothes_caption(
            image_path, self.bg_caption
        )

        return {
            "color_outfits": color_outfits,
            "scene_caption": scene_caption,
        }
    
    def suggest_outfit_from_bg(self, bg_path: str, preference_text: str | None = None, feedback_text: str | None = None) -> str:
        """
        Suggest an outfit given a background image
        """
        image = Image.open(bg_path).convert("RGB")
        image = self.resize_image(image)

        prompt = self.suggest_outfit_prompt
        if preference_text:
            prompt += f" Consider these preferences: {preference_text}."
        if feedback_text:
            prompt += f" Also consider this feedback: {feedback_text}."
            
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            },
        ]

        output = self._generate(messages)

        return output.strip()



