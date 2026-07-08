import os
import json
import faiss
import numpy as np
from pathlib import Path
from tqdm import tqdm
from PIL import Image

# Import the embedder we defined previously
# Make sure app/models/qwen_pe/qwen_embed.py exists
from app.models.qwen_pe.qwen_embed import QwenVLEmbedder

# --- CONFIGURATION ---
SOURCE_IMAGE_DIR = Path("data/2d")  # Put your 100k images here
OUTPUT_DIR = Path("data/faiss")
BATCH_SIZE = 8  # Adjust based on your GPU VRAM

def main():
    # 1. Setup Directories
    if not SOURCE_IMAGE_DIR.exists():
        print(f"❌ Error: Source directory '{SOURCE_IMAGE_DIR}' not found.")
        print("   Please create it and add your fashion images there.")
        return
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 2. Initialize Model
    print("🚀 Loading Qwen-VL Embedder...")
    embedder = QwenVLEmbedder()

    # 3. Find Images
    supported_exts = {".jpg", ".jpeg", ".png", ".webp"}
    image_paths = [
        p for p in SOURCE_IMAGE_DIR.glob("**/*") 
        if p.suffix.lower() in supported_exts
    ]
    
    print(f"Found {len(image_paths)} images to process.")
    
    # 4. Processing Loop
    metadata = []
    embeddings_list = []

    # Process in batches is efficient, but for simplicity/safety with vLLM wrapper,
    # we will loop one by one or small groups. 
    # Since our Qwen wrapper processes single images, we iterate.
    
    print("resizing and encoding images...")
    for img_path in tqdm(image_paths):
        try:
            # Generate Embedding
            # The embedder wrapper handles the <|image|> tags internally
            vec = embedder.encode_image(str(img_path))
            
            # Store Vector
            embeddings_list.append(vec)
            
            # Store Metadata
            metadata.append({
                "id": len(metadata),  # simple integer ID
                "filename": img_path.name,
                "file_path": str(img_path)
            })
            
        except Exception as e:
            print(f"\n⚠️ Error processing {img_path.name}: {e}")

    if not embeddings_list:
        print("No embeddings generated. Exiting.")
        return

    # 5. Create FAISS Index
    print("Checking vector dimensions...")
    emb_matrix = np.vstack(embeddings_list).astype('float32')
    d = emb_matrix.shape[1] # Dimension (e.g., 1536 or similar)
    print(f"Vector dimension: {d}")

    # Normalize for Cosine Similarity
    # (Inner Product on normalized vectors == Cosine Similarity)
    faiss.normalize_L2(emb_matrix)

    # Create Index
    index = faiss.IndexFlatIP(d) 
    index.add(emb_matrix)

    # 6. Save Artifacts
    index_path = OUTPUT_DIR / "qwen.index"
    meta_path = OUTPUT_DIR / "qwen_meta.json"

    print(f"💾 Saving index to {index_path}...")
    faiss.write_index(index, str(index_path))

    print(f"💾 Saving metadata to {meta_path}...")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print("✅ Ingestion Complete!")

if __name__ == "__main__":
    main()