from pathlib import Path
import json
import faiss

from app.core.vector_db import VectorDatabase

# 1) Build the index from your default data directory (or pass another folder)
db = VectorDatabase(
    embedding_model="pe",
    use_gpu=True,          # or False if you prefer CPU
    data_dir="data/2d",
    auto_prepare=False,    # we'll build explicitly
)

db.ingest_folder("data/2d", recursive=True, max_images=None)

# 2) Persist the index + metadata
out_dir = Path("data/indexes")
out_dir.mkdir(parents=True, exist_ok=True)
index_path = out_dir / "pe_clothes.index"
meta_path = out_dir / "pe_clothes.index.meta.json"

# Save FAISS index
faiss.write_index(db.index, str(index_path))

# Save metadata (paths aligned with the vectors)
with open(meta_path, "w", encoding="utf-8") as f:
    json.dump(db._metadata, f, ensure_ascii=False)

print(f"Saved index to {index_path} and metadata to {meta_path}")