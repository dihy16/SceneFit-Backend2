import torch
from torch import Tensor
import torch.nn.functional as F
from app.utils.device import resolve_device, resolve_dtype, resolve_autocast
from transformers import AutoTokenizer, AutoModel
from pathlib import Path

class TextMatcherModel():
    def __init__(self, device: str | None = None, config_name='Qwen/Qwen3-Embedding-0.6B'):
        self.device = resolve_device(device)
        self.dtype = resolve_dtype(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(config_name, padding_side="left")
        self.model = AutoModel.from_pretrained(
            config_name,
            torch_dtype=self.dtype
        ).to(self.device)
        self.model.eval()

    def last_token_pool(self,
                        last_hidden_states: Tensor,
                        attention_mask: Tensor) -> Tensor:
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

    @torch.no_grad()
    def encode(
        self,
        texts: list[str],
        max_length: int = 512,
        normalize: bool = True,
        batch_size: int = 32,
    ) -> torch.Tensor:
        """
        Returns:Tensor of shape (len(texts), hidden_dim)
        """
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]

            batch = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(self.device)

            outputs = self.model(**batch)
            embeddings = self.last_token_pool(
                outputs.last_hidden_state, batch["attention_mask"]
            )

            if normalize:
                embeddings = F.normalize(embeddings, p=2, dim=1)

            all_embeddings.append(embeddings)

        return torch.cat(all_embeddings, dim=0)   
    
    @torch.no_grad()
    def encode_normalized(self, texts: list[str]):
        embs = self.encode(texts)
        return F.normalize(embs, dim=-1)
    
    @torch.no_grad()
    def apply_feedback(
        self,
        query_embs: torch.Tensor,   # (N_text, D)
        instruction_text: str | None,
        beta: float = 0.2,
    ):
        if instruction_text is None:
            return query_embs

        instr_emb = self.encode_normalized([instruction_text])  # (1, D)
        return F.normalize(query_embs + beta * instr_emb, dim=-1)


    @torch.no_grad()
    def match_clothes_captions(
        self,
        descriptions: list[str],
        clothes_captions: dict[str, str],
        fb_text: str | None = None,
        beta: float = 0.2,
        top_k: int | None = None,
    ):
        """
        descriptions: list of AI-generated text strings
        clothes_captions: {image_name: caption}
        """
        if not clothes_captions:
            return []

        names = list(clothes_captions.keys())
        captions = [clothes_captions[name] for name in names]

        # query_embs = self.encode(descriptions)     # (N_text, D)
        # query_embs = F.normalize(query_embs, dim=-1)
        
        query_embs = self.encode_normalized(descriptions)
        query_embs = self.apply_feedback(query_embs, fb_text, beta)

        # caption_embs = self.encode(captions)       # (N_caption, D)
        # caption_embs = F.normalize(caption_embs, dim=-1)

        caption_embs = self.encode_normalized(captions)

        # Similarity: caption ↔ all descriptions
        # (N_caption, D) @ (D, N_text) → (N_caption, N_text)
        sim_matrix = caption_embs @ query_embs.T

        # For each caption, take the best matching description
        best_scores, best_text_idx = sim_matrix.max(dim=1)
        results = [
            {
                "name_clothes": Path(names[i]).stem,
                "similarity": float(best_scores[i]),
                "best_description": descriptions[int(best_text_idx[i])],
            }
            for i in range(len(names))
        ]

        results.sort(key=lambda x: x["similarity"], reverse=True)

        if top_k is not None:
            results = results[:top_k]

        return results