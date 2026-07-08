# from vllm import LLM, EngineArgs, SamplingParams
# import numpy as np
# from PIL import Image

# class QwenVLEmbedder:
#     """
#     Handles Feature Extraction (Images -> Vector, Text -> Vector)
#     """
#     def __init__(self, model_path="Qwen/Qwen2.5-VL-3B-Instruct"):
#         # Note: Using the Instruct model for embeddings allows for better 
#         # text-image alignment than raw base models in many cases, 
#         # or use the specific 'Qwen-VL-Embedding' if available.
#         self.llm = LLM(
#             model=model_path,
#             task="feature_extraction", # Important for vLLM embedding mode
#             enforce_eager=True,
#             trust_remote_code=True
#         )

#     def encode_text(self, text: str) -> np.ndarray:
#         """Encodes text queries into the visual-semantic space."""
#         # Qwen-VL specific: Text often needs to be wrapped or treated as a prompt
#         # that doesn't trigger generation. 
#         # For simplicity in this snippet, we treat it as a standard prompt inputs.
#         output = self.llm.encode(text)
#         return np.array(output[0].outputs.embedding)

#     def encode_image(self, image_path: str) -> np.ndarray:
#         """Encodes raw images."""
#         # vLLM feature extraction flow
#         inputs = {
#             "prompt": "<|image_1|> Describe this image.",
#             "multi_modal_data": {"image": Image.open(image_path).convert("RGB")},
#         }
#         output = self.llm.encode(inputs)
#         return np.array(output[0].outputs.embedding).flatten()


# class QwenVLGenerator:
#     """
#     Handles Reasoning (Scene -> JSON, Reranking scores)
#     """
#     def __init__(self, model_path="Qwen/Qwen2.5-VL-7B-Instruct"):
#         self.llm = LLM(
#             model=model_path,
#             trust_remote_code=True,
#             limit_mm_per_prompt={"image": 2} # Allow 2 images for reranking (Scene + Cloth)
#         )
#         self.tokenizer = self.llm.get_tokenizer()

#     def generate(self, prompts: list[dict], sampling_params=None):
#         if sampling_params is None:
#             sampling_params = SamplingParams(temperature=0.1, max_tokens=256)
        
#         outputs = self.llm.generate(prompts, sampling_params)
#         return [o.outputs[0].text for o in outputs]