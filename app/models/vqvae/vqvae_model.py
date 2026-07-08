from git import List, Optional, Sequence, Tuple, Union
import torch
import urllib.request
from pathlib import Path
import sys

VAR_ROOT = Path(__file__).resolve().parent / "VAR"
if str(VAR_ROOT) not in sys.path:
    sys.path.insert(0, str(VAR_ROOT))

from models.vqvae import VQVAE

HF_HOME = "https://huggingface.co/FoundationVision/var/resolve/main"
VAE_CKPT_NAME = "vae_ch160v4096z32.pth"

CKPT_DIR = Path("checkpoints")
VAE_CKPT_PATH = CKPT_DIR / VAE_CKPT_NAME


def download_if_not_exists(url: str, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        print(f"Downloading {dst.name} ...")
        urllib.request.urlretrieve(url, dst)

class VQVAEModel:
    def __init__(
        self,
        vocab_size: int = 4096,
        z_channels: int = 32,
        ch: int = 160,
        share_quant_resi: int = 4,
        v_patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),
        ckpt_dir: str = "checkpoints",
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.ckpt_dir = Path(ckpt_dir)
        self.vae_ckpt_path = self.ckpt_dir / VAE_CKPT_NAME

        download_if_not_exists(
            f"{HF_HOME}/{VAE_CKPT_NAME}",
            self.vae_ckpt_path,
        )

        self.model = VQVAE(
            vocab_size=vocab_size,
            z_channels=z_channels,
            ch=ch,
            test_mode=True,
            share_quant_resi=share_quant_resi,
            v_patch_nums=v_patch_nums,
        ).to(self.device)

        self.load_ckpt(self.vae_ckpt_path)

    def load_ckpt(self, model_path: Path):
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval()

    def fhat_to_img(self, f_hat: torch.Tensor):
        self.model.fhat_to_img(f_hat)
    
    def img_to_idxBl(self, inp_img_no_grad: torch.Tensor, v_patch_nums: Optional[Sequence[Union[int, Tuple[int, int]]]] = None) -> List[torch.LongTensor]:    # return List[Bl]
        return self.model.img_to_idxBl(inp_img_no_grad, v_patch_nums=v_patch_nums)
    
    def idxBl_to_img(self, ms_idx_Bl: List[torch.Tensor], same_shape: bool, last_one=False) -> Union[List[torch.Tensor], torch.Tensor]:
        return self.model.idxBl_to_img(ms_idx_Bl, same_shape, last_one=last_one)
    
    def embed_to_img(self, ms_h_BChw: List[torch.Tensor], all_to_max_scale: bool, last_one=False) -> Union[List[torch.Tensor], torch.Tensor]:
        return self.model.embed_to_img(ms_h_BChw, all_to_max_scale=all_to_max_scale, last_one=last_one)
    
    def img_to_reconstructed_img(self, x, v_patch_nums: Optional[Sequence[Union[int, Tuple[int, int]]]] = None, last_one=False) -> List[torch.Tensor]:
        return self.model.img_to_reconstructed_img(x, v_patch_nums=v_patch_nums, last_one=last_one)
        
