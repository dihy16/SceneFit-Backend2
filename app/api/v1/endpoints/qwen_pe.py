# from fastapi import FastAPI, UploadFile, File
# from PIL import Image
# import io
# from app.models.qwen_pe.qwen_pe import QwenPEModel
# from app.models.registry import ModelRegistry

# app = FastAPI()
# retriever = ModelRegistry.get_model("qwen_pe") # Load model once on startup

# @app.post("/retrieve")
# async def retrieve_fashion(file: UploadFile = File(...)):
#     # 1. Read Image
#     content = await file.read()
#     scene_image = Image.open(io.BytesIO(content)).convert("RGB")
    
#     # 2. Understand Scene
#     scene_data = retriever.analyze_scene(scene_image)
    
#     # 3. Generate Search Query (HyDE)
#     # "What should I wear?" -> "A linen shirt..."
#     ideal_outfit_text = retriever.generate_search_query(scene_data)
    
#     # 4. Fast Retrieval (Recall)
#     candidates = retriever.recall(ideal_outfit_text, k=20)
    
#     # 5. Intelligent Reranking (Precision)
#     # Looks at the scene and clothing pairs together
#     final_results = retriever.rerank(scene_image, candidates, top_n=5)
    
#     return {
#         "scene_analysis": scene_data,
#         "generated_query": ideal_outfit_text,
#         "recommendations": final_results
#     }