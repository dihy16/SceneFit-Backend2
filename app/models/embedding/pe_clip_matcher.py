import faiss
import pickle
import torch
import torch.nn.functional as F
from PIL import Image
from typing import List, Tuple
from pathlib import Path

import app.services.pe_core.vision_encoder.pe as pe
import app.services.pe_core.vision_encoder.transforms as transforms
from app.utils.device import resolve_device, resolve_autocast, resolve_dtype

FAISS_DIR = Path("data/faiss")
class PEClipMatcher:
    """
    PE-CLIP matcher supporting:
    - Brute-force matching (original baselines)
    - FAISS-based retrieval (new method)
    """

    TEXT_PROMPT = "a clothing outfit described as: {}"

    def __init__(
        self,
        config_name: str = "PE-Core-B16-224",
        device: str | None = None,
        autocast: bool = True,
        faiss_dir: str | None = FAISS_DIR,
        load_faiss: bool = True,
    ):
        # -------------------------
        # Device / autocast
        # -------------------------
        self.device = resolve_device(device)
        self.device_type = self.device.type
        self.use_autocast = autocast and resolve_autocast(self.device)
        self.autocast_dtype = resolve_dtype(self.device)

        # -------------------------
        # Load model
        # -------------------------
        self.model = pe.CLIP.from_config(
            config_name,
            pretrained=True,
        ).to(self.device)
        self.model.eval()

        # -------------------------
        # Transforms
        # -------------------------
        self.image_transform = transforms.get_image_transform(
            self.model.image_size
        )
        self.text_tokenizer = transforms.get_text_tokenizer(
            self.model.context_length
        )

        # -------------------------
        # Optional FAISS
        # -------------------------
        self.faiss_index = None
        self.faiss_meta = None

        if load_faiss and faiss_dir is not None:
            self._load_faiss(faiss_dir)

    # -------------------------------------------------
    # FAISS loading
    # -------------------------------------------------
    def release(self):
        #self.model.to("cpu")
        del self.model
        torch.cuda.empty_cache()
    def _load_faiss(self, faiss_dir: str):
        faiss_dir = Path(faiss_dir)

        self.faiss_index = faiss.read_index(
            str(faiss_dir / "clothes_image.index")
        )

        with open(faiss_dir / "clothes_image_meta.pkl", "rb") as f:
            self.faiss_meta = pickle.load(f)

    # -------------------------------------------------
    # Text encoding
    # -------------------------------------------------
    @torch.no_grad()
    def encode_text(self, texts: List[str]) -> torch.Tensor:
        if isinstance(texts, str):
            texts = [texts]

        texts = [self.TEXT_PROMPT.format(t) for t in texts]
        tokens = self.text_tokenizer(texts).to(self.device)

        with torch.autocast(
            device_type=self.device_type,
            dtype=self.autocast_dtype,
            enabled=self.use_autocast,
        ):
            _, text_features, _ = self.model(image=None, text=tokens)

        return F.normalize(text_features, dim=-1)

    # -------------------------------------------------
    # Image encoding (still useful!)
    # -------------------------------------------------
    @torch.no_grad()
    def encode_image(self, images: List[Image.Image]) -> torch.Tensor:
        image_tensor = torch.stack(
            [self.image_transform(img) for img in images]
        ).to(self.device)

        with torch.autocast(
            device_type=self.device_type,
            dtype=self.autocast_dtype,
            enabled=self.use_autocast,
        ):
            image_features, _, _ = self.model(image=image_tensor, text=None)

        return F.normalize(image_features, dim=-1)

    # -------------------------------------------------
    # Shared query embedding (NEW, key fix)
    # -------------------------------------------------
    def _build_query_embedding(self, descriptions: List[str]) -> torch.Tensor:
        """
        Aggregate 10 outfit descriptions into ONE query vector.
        """
        text_embs = self.encode_text(descriptions)
        return text_embs.mean(dim=0, keepdim=True)

    # -------------------------------------------------
    # MATCH BY IMAGE (Baseline 1 / FAISS version)
    # -------------------------------------------------
    @torch.no_grad()
    def match_clothes(
        self,
        descriptions: List[str] | None = None,
        query_emb: torch.Tensor = None,
        clothes: List[Tuple[str, Image.Image]] | None = None,
        top_k: int | None = None,
    ):
        """
        If FAISS is loaded:
            uses FAISS (new method)
        Else:
            falls back to brute-force (original baseline)
        """
        if query_emb is None:
            query_emb = self._build_query_embedding(descriptions)

        # ---------- FAISS PATH ----------
        k = top_k or 10

    # ---------- FAISS PATH ----------
        if self.faiss_index is not None:
            query_np = query_emb.cpu().numpy().astype("float32")

            # scores, indices: [N, k]
            scores, indices = self.faiss_index.search(query_np, k)

            # aggregate by max similarity per index
            best_scores = {}

            for qi in range(scores.shape[0]):
                for score, idx in zip(scores[qi], indices[qi]):
                    score = float(score)
                    if idx not in best_scores or score > best_scores[idx]:
                        best_scores[idx] = score

            # sort by similarity
            sorted_items = sorted(
                best_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )

            results = [
                {
                    "name_clothes": self.faiss_meta["filenames"][idx],
                    "similarity": score,
                }
                for idx, score in sorted_items[:top_k]
            ]

            return results

        # ---------- BRUTE-FORCE PATH ----------
        assert clothes is not None, "Clothes images required for brute-force."

        names, images = zip(*clothes)
        image_embs = self.encode_image(list(images))
        image_embs = F.normalize(image_embs, dim=-1)

        # sims: [M, N]
        sims = image_embs @ query_emb.T

        # MAX aggregation over queries → [M]
        final_sims = sims.max(dim=1).values

        results = [
            {
                "outfit_name": names[i],
                "similarity": float(final_sims[i]),
            }
            for i in range(len(names))
        ]

        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:top_k] if top_k else results

    # -------------------------------------------------
    # MATCH BY CAPTION (Baseline 2, unchanged semantics)
    # -------------------------------------------------
    @torch.no_grad()
    def match_clothes_captions(
        self,
        descriptions: List[str],
        clothes_captions: dict[str, str],
        top_k: int | None = None,
    ):
        query_emb = self._build_query_embedding(descriptions)

        names = list(clothes_captions.keys())
        captions = list(clothes_captions.values())

        caption_embs = self.encode_text(captions)
        sims = (caption_embs @ query_emb.T).squeeze(1)

        results = [
            {
                "outfit_name": Path(names[i]).stem,
                "similarity": float(sims[i]),
            }
            for i in range(len(names))
        ]

        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:top_k] if top_k else results
