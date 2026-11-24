"""Dense encoders used across the project. All return L2-normalised float32 arrays."""

from __future__ import annotations

import numpy as np
import torch

BGE_SMALL = "BAAI/bge-small-en-v1.5"
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
BLAIR_BASE = "hyp1231/blair-roberta-base"


def _s(x) -> str:
    return x if isinstance(x, str) else ""


def product_text(title, brand, bullets, max_bullets_chars: int = 600) -> str:
    """Catalog text shared by every retriever, so they all see the same input."""
    return f"{_s(title)}. Brand: {_s(brand)}. {_s(bullets)[:max_bullets_chars]}"


def default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


class BgeEncoder:
    name = "bge-small"

    def __init__(self, device: str | None = None):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(BGE_SMALL, device=device or default_device())

    def docs(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts, batch_size=128, normalize_embeddings=True, show_progress_bar=False)

    def queries(self, texts: list[str]) -> np.ndarray:
        return self.docs([BGE_QUERY_PREFIX + t for t in texts])


class BlairEncoder:
    """BLaIR (Hou et al., 2024): RoBERTa pretrained on Amazon Reviews 2023; first-token pooling per the model card."""

    name = "blair-base"

    def __init__(self, device: str | None = None):
        from transformers import AutoModel, AutoTokenizer

        self.device = device or default_device()
        self.tok = AutoTokenizer.from_pretrained(BLAIR_BASE)
        self.model = AutoModel.from_pretrained(BLAIR_BASE).to(self.device).eval()

    @torch.no_grad()
    def docs(self, texts: list[str], bs: int = 64) -> np.ndarray:
        out = []
        for i in range(0, len(texts), bs):
            enc = self.tok(texts[i : i + bs], padding=True, truncation=True, max_length=256, return_tensors="pt")
            v = self.model(**enc.to(self.device)).last_hidden_state[:, 0]
            out.append(torch.nn.functional.normalize(v, dim=-1).float().cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, self.model.config.hidden_size), dtype=np.float32)

    def queries(self, texts: list[str]) -> np.ndarray:
        return self.docs(texts)
