# import faiss
# import json
# import yaml
# import numpy as np
# from pathlib import Path
# from PIL import Image
# from app.models.qwen_pe.qwen_embed import QwenVLEmbedder, QwenVLGenerator

# class QwenPEModel:
#     def __init__(self):
#         # Load Config
#         with open("app/prompts/templates.yaml") as f:
#             self.prompts = yaml.safe_load(f)

#         # Load Models
#         # In production, these might share GPU memory or run on separate services
#         self.embedder = QwenVLEmbedder() 
#         self.generator = QwenVLGenerator()

#         # Load Vector Database
#         self._load_faiss()

#     def _load_faiss(self):
#         index_path = Path("data/faiss/qwen.index")
#         meta_path = Path("data/faiss/qwen_meta.json")
        
#         if not index_path.exists():
#             raise FileNotFoundError("FAISS index not found. Run ingestion script first.")

#         self.index = faiss.read_index(str(index_path))
#         self.metadata = json.load(open(meta_path))

#     def analyze_scene(self, scene_image: Image.Image) -> dict:
#         """Step 1: Convert Visual Scene to Semantic Attributes"""
#         prompt_text = self.prompts['scene_analysis']['user']
        
#         # Construct vLLM input
#         inputs = {
#             "prompt": f"<|image_1|>\n{prompt_text}",
#             "multi_modal_data": {"image": scene_image}
#         }
        
#         # Run inference
#         response_text = self.generator.generate([inputs])[0]
        
#         # Clean JSON markdown if present
#         cleaned = response_text.replace("```json", "").replace("```", "").strip()
#         try:
#             return json.loads(cleaned)
#         except:
#             # Fallback if model chats instead of JSON
#             return {"environment": cleaned, "mood": "unknown"}

#     def generate_search_query(self, scene_data: dict) -> str:
#         """Step 2: Create a hallucinated 'Ideal Clothing' description"""
#         scene_desc = f"Environment: {scene_data.get('environment')}, Mood: {scene_data.get('mood')}"
        
#         prompt = self.prompts['clothing_generation']['user'].format(
#             scene_description=scene_desc
#         )
        
#         # Text-only generation
#         inputs = {"prompt": prompt}
#         return self.generator.generate([inputs])[0]

#     def recall(self, query_text: str, k: int = 50) -> list[dict]:
#         """Step 3: Vector Search (Text Query -> Image Inventory)"""
#         # Embed the description of the ideal outfit
#         query_vec = self.embedder.encode_text(query_text).astype('float32')
#         faiss.normalize_L2(query_vec) # Ensure your index was trained normalized too!

#         # Search
#         scores, ids = self.index.search(query_vec, k)
        
#         results = []
#         for score, idx in zip(scores[0], ids[0]):
#             if idx != -1:
#                 item = self.metadata[idx]
#                 item['vector_score'] = float(score)
#                 results.append(item)
#         return results

#     def rerank(self, scene_image: Image.Image, candidates: list[dict], top_n=5):
#         """Step 4: VLM Reranking (Scene + Cloth compatibility check)"""
#         rerank_inputs = []
        
#         for cand in candidates:
#             cloth_img = Image.open(cand['file_path']).convert("RGB")
            
#             # Formulate prompt with TWO images
#             prompt = self.prompts['rerank_scoring']['user']
#             prompt = f"<|image_1|><|image_2|>\n{prompt}"
            
#             rerank_inputs.append({
#                 "prompt": prompt,
#                 "multi_modal_data": {"image": [scene_image, cloth_img]}
#             })

#         # Batch generation of scores
#         responses = self.generator.generate(rerank_inputs)
        
#         # Parse scores
#         scored_candidates = []
#         for cand, resp in zip(candidates, responses):
#             try:
#                 # Extract integer from response
#                 score = int(''.join(filter(str.isdigit, resp)))
#             except:
#                 score = 0
            
#             cand['rerank_score'] = score
#             scored_candidates.append(cand)

#         # Sort by rerank score
#         return sorted(scored_candidates, key=lambda x: x['rerank_score'], reverse=True)[:top_n]