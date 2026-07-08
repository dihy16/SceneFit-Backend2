import torch
import torch.nn as nn
from urllib.request import urlretrieve
import os
from os.path import expanduser
from app.utils.device import resolve_device
import open_clip

class CLIPModel:
    def __init__(self, model_name="ViT-L-14", pretrained="openai", device=None):
        self.device = resolve_device(device)
        
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        self.model.to(self.device)
        self.model.eval()
    
    @torch.no_grad()
    def encode_image(self, images) -> torch.Tensor:
        batch = torch.stack([self.preprocess(img) for img in images]).to(self.device)
        embeddings = self.model.encode_image(batch)
        embeddings /= embeddings.norm(dim=-1, keepdim=True)
        return embeddings
    
class AestheticPredictor:
    def __init__(self, device=None):
        self.device = resolve_device(device)
        self.clip_model = CLIPModel()
        self.head = self.get_aesthetic_model().to(self.device)
        self.head.eval()
        
    def get_aesthetic_model(self, clip_model="vit_l_14"):
        home = expanduser("~")
        cache_folder = home + "/.cache/emb_reader"
        path_to_model = cache_folder + "/sa_0_4_"+clip_model+"_linear.pth"
        if not os.path.exists(path_to_model):
            os.makedirs(cache_folder, exist_ok=True)
            url_model = (
                "https://github.com/LAION-AI/aesthetic-predictor/blob/main/sa_0_4_"+clip_model+"_linear.pth?raw=true"
            )
            urlretrieve(url_model, path_to_model)
        if clip_model == "vit_l_14":
            m = nn.Linear(768, 1)
        # elif clip_model == "vit_b_32":
            # m = nn.Linear(512, 1)
        else:
            raise ValueError()
        s = torch.load(path_to_model)
        m.load_state_dict(s)
        m.eval()
        return m
    
    def load(self):
        return None

    @torch.no_grad()
    def score_images(self, items):
        names, images = zip(*items)
        embeddings = self.clip_model.encode_image(list(images))
        preds = self.head(embeddings).squeeze(-1).tolist()
        results = [{"name": n, "score": p} for n, p in zip(names, preds)]
        results.sort(key=lambda x: x["score"], reverse=True)
        return results
