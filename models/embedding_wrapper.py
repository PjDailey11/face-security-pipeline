from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


class FaceEmbeddingModel:
    """InceptionResnetV1 embeddings via facenet-pytorch (512-D by default)."""

    def __init__(
        self,
        device: str | None = None,
        classify: bool = False,
    ) -> None:
        from facenet_pytorch import InceptionResnetV1

        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self._net = InceptionResnetV1(pretrained="vggface2", classify=classify).eval().to(self.device)

    @torch.inference_mode()
    def embed_rgb_chw(self, tensor_nchw: torch.Tensor, normalize: bool = True) -> np.ndarray:
        """tensor_nchw: float32 in [0,1], shape (N,3,H,W)."""
        tensor_nchw = tensor_nchw.to(self.device)
        emb = self._net(tensor_nchw)
        if normalize:
            emb = F.normalize(emb, dim=1)
        return emb.detach().cpu().numpy()

    def embed_pil_list(self, crops: list[Image.Image], normalize: bool = True) -> np.ndarray:
        from torchvision import transforms

        t = transforms.Compose(
            [
                transforms.Resize((160, 160)),
                transforms.ToTensor(),
            ]
        )
        batch = torch.stack([t(img.convert("RGB")) for img in crops])
        return self.embed_rgb_chw(batch, normalize=normalize)

    def embed_bgr_np(self, crops_bgr: list[np.ndarray], normalize: bool = True) -> np.ndarray:
        pil = [Image.fromarray(c[..., ::-1]) for c in crops_bgr]
        return self.embed_pil_list(pil, normalize=normalize)
