from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .data import Claim


class ScientificIndex:
    def __init__(self, model_name: str = "malteos/scincl", device: str | None = None):
        self.model_name = model_name
        self.device = device
        self.claims: list[Claim] = []
        self.embeddings: np.ndarray | None = None
        self._tokenizer = None
        self._model = None
        self._torch_device = None

    @staticmethod
    def _hash(claims, model):
        content = [model, [(claim.claim_id, claim.text) for claim in claims]]
        body = json.dumps(content, ensure_ascii=False).encode()
        return hashlib.sha256(body).hexdigest()

    def _encode(self, texts: list[str]) -> np.ndarray:
        import torch
        from transformers import AutoModel, AutoTokenizer
        if self._model is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModel.from_pretrained(self.model_name)
            self._torch_device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
            self._model.to(self._torch_device).eval().requires_grad_(False)
        tokenizer = self._tokenizer
        model = self._model
        device = self._torch_device
        batches: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(texts), 32):
                tokens = tokenizer(
                    texts[start : start + 32],
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                ).to(device)
                hidden_states = model(**tokens).last_hidden_state
                attention_mask = tokens.attention_mask.unsqueeze(-1)
                embeddings = (
                    (hidden_states * attention_mask).sum(1)
                    / attention_mask.sum(1).clamp_min(1)
                )
                embeddings = torch.nn.functional.normalize(embeddings, dim=1)
                batches.append(embeddings.cpu().numpy())
        return np.concatenate(batches)

    def similarities(self, pairs: list[tuple[str, str]]) -> list[float]:
        """Encode all verifier pairs in two batches instead of one model call per step."""
        if not pairs:
            return []
        claim_embeddings = self._encode([claim for claim, _ in pairs])
        span_embeddings = self._encode([span for _, span in pairs])
        return [
            float(claim_embedding @ span_embedding)
            for claim_embedding, span_embedding in zip(
                claim_embeddings, span_embeddings
            )
        ]

    def build(self, claims: list[Claim], cache_dir: str | Path):
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        key = self._hash(claims, self.model_name)
        path = cache_dir / f"claims-{key}.npz"
        self.claims = claims
        cache_is_current = False
        if path.exists():
            cached = np.load(path, allow_pickle=False)
            cache_is_current = {
                "embeddings", "paper_ids", "claim_numbers", "texts", "model_name"
            }.issubset(cached.files)
        if cache_is_current:
            self.embeddings = cached["embeddings"]
        else:
            self.embeddings = self._encode([claim.text for claim in claims])
            np.savez_compressed(
                path,
                embeddings=self.embeddings,
                claim_ids=np.asarray([claim.claim_id for claim in claims]),
                paper_ids=np.asarray([claim.paper_id for claim in claims]),
                claim_numbers=np.asarray([claim.claim_number for claim in claims]),
                texts=np.asarray([claim.text for claim in claims]),
                model_name=np.asarray(self.model_name),
            )
        return path

    @classmethod
    def load(cls, path: str | Path, device: str | None = None):
        """Load a frozen index without access to the annotated source dataset."""
        cached = np.load(Path(path), allow_pickle=False)
        required = {"embeddings", "paper_ids", "claim_numbers", "texts", "model_name"}
        missing = required - set(cached.files)
        if missing:
            raise ValueError(
                f"Index lacks metadata {sorted(missing)}; rebuild it with build_index.py"
            )
        model_name = str(cached["model_name"].item())
        index = cls(model_name=model_name, device=device)
        index.embeddings = cached["embeddings"]
        index.claims = [
            Claim(str(paper_id), int(claim_number), str(text))
            for paper_id, claim_number, text in zip(
                cached["paper_ids"],
                cached["claim_numbers"],
                cached["texts"],
            )
        ]
        if len(index.claims) != len(index.embeddings):
            raise ValueError("claim metadata and embedding counts differ")
        return index

    def retrieve(self, sentence: str, k: int = 5) -> list[tuple[Claim, float]]:
        assert self.embeddings is not None
        scores = self.embeddings @ self._encode([sentence])[0]
        best_indexes = np.argsort(-scores)[:k]
        return [(self.claims[index], float(scores[index])) for index in best_indexes]
