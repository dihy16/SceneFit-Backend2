"""Lightweight FAISS wrapper for image embeddings."""

from pathlib import Path
from typing import Any, List, Sequence
import json

import faiss
import numpy as np
import torch
from PIL import Image

from app.models.registry import ModelRegistry
from app.utils.util import load_images_from_folder


class VectorDatabase:
    """Simple in-memory FAISS index backed by the PE embedding model."""

    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}

    def __init__(
        self,
        embedding_model: str = "pe",
        use_gpu: bool = False,
        index_factory: str | None = None,
        index_path: str | Path | None = None,
        metadata_path: str | Path | None = None,
        data_dir: str | Path = "data/2d",
        auto_prepare: bool = True,
        recursive: bool = True,
        max_images: int | None = None,
    ) -> None:
        self.embedding_model_name = embedding_model
        self.model = ModelRegistry.get(embedding_model)
        self.index: faiss.Index | None = None
        self.dim: int | None = None
        self._metadata: List[Any] = []
        self.use_gpu = use_gpu and faiss.get_num_gpus() > 0
        self.index_factory = index_factory

        self.project_root = Path(__file__).resolve().parents[2]
        self.default_data_dir = self._resolve_path(data_dir)
        self.index_path = self._resolve_path(index_path) if index_path is not None else None
        self.metadata_path = (
            self._resolve_path(metadata_path) if metadata_path is not None else None
        )
        self.recursive = recursive
        self.max_images = max_images

        if auto_prepare:
            self.ensure_ready()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def build_from_images(
        self,
        images: Sequence[Image.Image | str | Path],
        metadata: Sequence[Any] | None = None,
    ) -> None:
        """Reset index and ingest a new set of images."""

        print("[VectorDB] Building index from in-memory images...")
        self.index = None
        self._metadata.clear()
        self.dim = None

        embeddings = self.embed_images(images)
        self._add_embeddings(embeddings, metadata)

    def ensure_ready(self) -> None:
        """Load an existing index if present; otherwise build from default data."""

        print("[VectorDB] ensure_ready: starting")
        data_dir = self.default_data_dir
        # 1) Explicit paths win
        if self.index_path is not None and self.index_path.exists():
            print(f"[VectorDB] Loading index from explicit path: {self.index_path}")
            self._load_index(self.index_path, self.metadata_path)
            if self._rebuild_if_mismatch(data_dir):
                return
            return

        # 2) Try to discover an existing index near data_dir
        found_index, found_meta = self._find_existing_index()
        if found_index:
            print(f"[VectorDB] Found existing index at {found_index}; loading...")
            self._load_index(found_index, found_meta)
            self.index_path = found_index
            self.metadata_path = found_meta
            if self._rebuild_if_mismatch(data_dir):
                return
            return

        # 3) Fall back to building from data
        print(f"[VectorDB] No index found; building from data_dir={self.default_data_dir}")
        self.ingest_folder(self.default_data_dir, recursive=self.recursive, max_images=self.max_images)

    def ingest_folder(
        self,
        folder: str | Path,
        recursive: bool = True,
        max_images: int | None = None,
    ) -> None:
        """Load images from a folder (and optionally subfolders) and build the index."""

        folder_path = self._resolve_path(folder)

        print(f"[VectorDB] Ingesting folder: {folder_path}")
        file_list = self._list_image_files(folder_path, recursive=recursive)
        if max_images is not None:
            file_list = file_list[: max_images]

        imgs = load_images_from_folder(
            folder_path, recursive=recursive, max_images=max_images
        )

        meta = [str(p) for p in file_list]
        meta = meta[: len(imgs)]

        self.build_from_images(imgs, meta)
        self._persist_index()

    def _find_existing_index(self) -> tuple[Path | None, Path | None]:
        """Search for a persisted FAISS index and metadata file near the data directory."""

        base_dir = self.default_data_dir
        if not base_dir.is_absolute():
            base_dir = Path(__file__).resolve().parents[2] / base_dir

        candidates = [
            base_dir / "vector.index",
            base_dir / "vector.faiss",
        ] + sorted(base_dir.glob("*.index")) + sorted(base_dir.glob("*.faiss"))

        for path in candidates:
            if path.exists():
                meta_path = path.with_suffix(path.suffix + ".meta.json")
                return path, (meta_path if meta_path.exists() else None)

        return None, None

    def _rebuild_if_mismatch(self, folder_path: Path) -> bool:
        """Recreate the index if it does not line up with the folder contents."""

        if not self._index_matches_folder(folder_path):
            print("[VectorDB] Index mismatch detected; rebuilding from folder")
            self.ingest_folder(folder_path, recursive=self.recursive, max_images=self.max_images)
            return True
        return False

    def _index_matches_folder(self, folder_path: Path) -> bool:
        """Quick sanity check that index/vector count and metadata align with disk."""

        if self.index is None:
            return False

        files = self._list_image_files(folder_path, recursive=self.recursive)
        if self.max_images is not None:
            files = files[: self.max_images]

        expected = len(files)
        if expected == 0:
            return False

        if self.index.ntotal != expected:
            print(
                f"[VectorDB] Mismatch: index has {self.index.ntotal} vectors but folder has {expected} images"
            )
            return False

        if self._metadata:
            if len(self._metadata) != expected:
                print(
                    f"[VectorDB] Mismatch: metadata length {len(self._metadata)} does not match image count {expected}"
                )
                return False

            if all(isinstance(m, str) for m in self._metadata):
                meta_paths = [str(Path(m)) for m in self._metadata]
                file_paths = [str(p) for p in files]
                if meta_paths != file_paths:
                    print("[VectorDB] Mismatch: metadata paths differ from folder files")
                    return False

        return True

    def _persist_index(self, index_path: Path | None = None, metadata_path: Path | None = None) -> None:
        """Persist index and metadata to disk for reuse."""

        if self.index is None:
            return

        index_path = index_path or self.index_path or self._default_index_path()
        metadata_path = metadata_path or self.metadata_path or (index_path.with_suffix(index_path.suffix + ".meta.json"))

        index_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)

        cpu_index = self.index
        if self.use_gpu:
            try:
                cpu_index = faiss.index_gpu_to_cpu(self.index)
            except Exception:
                cpu_index = self.index

        faiss.write_index(cpu_index, str(index_path))
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(self._metadata, f)

        self.index_path = index_path
        self.metadata_path = metadata_path
        print(f"[VectorDB] Saved index to {index_path} and metadata to {metadata_path}")

    def _list_image_files(self, folder_path: Path, recursive: bool = True) -> list[Path]:
        pattern = "**/*" if recursive else "*"
        return [p for p in sorted(folder_path.glob(pattern)) if p.suffix.lower() in self.IMAGE_EXTENSIONS]

    def _default_index_path(self) -> Path:
        return self.default_data_dir / "vector.index"

    def _resolve_path(self, path: str | Path | None) -> Path:
        if path is None:
            raise ValueError("Cannot resolve a None path")
        path = Path(path)
        return path if path.is_absolute() else self.project_root / path

    def _load_index(self, index_path: Path, metadata_path: Path | None = None) -> None:
        """Load FAISS index (and optional metadata) from disk."""

        cpu_index = faiss.read_index(str(index_path))
        if self.use_gpu:
            print("[VectorDB] Moving index to all available GPUs")
            self.index = faiss.index_cpu_to_all_gpus(cpu_index)
        else:
            self.index = cpu_index

        print(f"[VectorDB] Loaded index from {index_path}; dim={self.index.d}, ntotal={self.index.ntotal}")
        self.dim = self.index.d

        if metadata_path is not None and metadata_path.exists():
            try:
                with open(metadata_path, "r", encoding="utf-8") as f:
                    self._metadata = json.load(f)
                print(f"[VectorDB] Loaded metadata from {metadata_path} ({len(self._metadata)} records)")
            except Exception:
                # Fall back to empty metadata while keeping the index
                self._metadata = [None] * self.index.ntotal
                print("[VectorDB] Failed to load metadata; filled with None")
        else:
            self._metadata = [None] * self.index.ntotal
            print("[VectorDB] No metadata file found; filled with None")

    def add_images(
        self,
        images: Sequence[Image.Image | str | Path],
        metadata: Sequence[Any] | None = None,
    ) -> None:
        """Append images to an existing index (builds one if missing)."""

        embeddings = self.embed_images(images)
        self._add_embeddings(embeddings, metadata)

    def search_by_image(
        self,
        image: Image.Image | str | Path,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        query = self.embed_images([image])
        return self.search(query, top_k=top_k)[0]

    def search(
        self, query_embeddings: np.ndarray, top_k: int = 5
    ) -> list[list[dict[str, Any]]]:
        """Search the index with pre-computed embeddings."""

        if self.index is None:
            raise RuntimeError("Index is empty; add data before searching.")

        query = self._as_numpy(query_embeddings)
        scores, idx = self.index.search(query, top_k)

        results: list[list[dict[str, Any]]] = []
        for row_scores, row_idx in zip(scores, idx):
            row: list[dict[str, Any]] = []
            for score, i in zip(row_scores, row_idx):
                if i == -1:
                    continue
                meta = self._metadata[i] if i < len(self._metadata) else None
                row.append(
                    {
                        "id": int(i),
                        "score": float(score),
                        "metadata": meta,
                    }
                )
            row.sort(key=lambda x: x["score"], reverse=True)
            results.append(row)

        return results

    def embed_images(
        self, images: Sequence[Image.Image | str | Path] | Image.Image
    ) -> np.ndarray:
        """Convert images (paths or PIL objects) to normalized float32 embeddings."""

        if isinstance(images, (str, Path, Image.Image)):
            images = [images]

        pil_images = [self._to_image(img) for img in images]
        tensor = self.model.encode_image(pil_images)
        return self._as_numpy(tensor)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _add_embeddings(
        self, embeddings: np.ndarray, metadata: Sequence[Any] | None = None
    ) -> None:
        vectors = self._as_numpy(embeddings)

        if vectors.ndim != 2:
            raise ValueError(
                f"Embeddings must be 2D [n, d], got shape {vectors.shape}"
            )

        if self.index is None:
            self.dim = vectors.shape[1]
            self.index = self._create_index(self.dim)

        if vectors.shape[1] != self.dim:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self.dim}, got {vectors.shape[1]}"
            )

        self.index.add(vectors)

        meta = list(metadata) if metadata is not None else [None] * len(vectors)
        if len(meta) != len(vectors):
            raise ValueError("Metadata length must match number of embeddings")
        self._metadata.extend(meta)

    def _create_index(self, dim: int) -> faiss.Index:
        metric = faiss.METRIC_INNER_PRODUCT
        if self.index_factory:
            cpu_index = faiss.index_factory(dim, self.index_factory, metric)
        else:
            cpu_index = faiss.IndexFlatIP(dim)

        if self.use_gpu:
            return faiss.index_cpu_to_all_gpus(cpu_index)
        return cpu_index

    @staticmethod
    def _as_numpy(x: np.ndarray | torch.Tensor) -> np.ndarray:
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().float().numpy()
        return np.ascontiguousarray(x.astype(np.float32))

    @staticmethod
    def _to_image(item: Image.Image | str | Path) -> Image.Image:
        if isinstance(item, Image.Image):
            return item.convert("RGB")
        return Image.open(item).convert("RGB")


__all__ = ["VectorDatabase"]