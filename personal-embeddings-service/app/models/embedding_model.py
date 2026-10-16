"""Sentence-transformer wrapper that supports multiple models."""
from __future__ import annotations

import logging
import threading
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
from sentence_transformers import SentenceTransformer, models
import torch

from app.config import settings

logger = logging.getLogger(__name__)

TextInput = Union[str, Sequence[str]]


class EmbeddingModel:
    """Central manager for embedding backends."""

    def __init__(self) -> None:
        self._models: Dict[str, SentenceTransformer] = {}
        self._lock = threading.Lock()
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self.default_model_name = settings.model_name
        self.allowed_models = set(settings.allowed_models or []) | {self.default_model_name}

    def normalize_model_name(self, requested: Optional[str]) -> str:
        if requested is None:
            return self.default_model_name
        if requested not in self.allowed_models:
            raise ValueError(
                f"Model '{requested}' is not in the allowed list: {sorted(self.allowed_models)}"
            )
        return requested

    def load_model(self, model_name: Optional[str] = None) -> bool:
        name = self.normalize_model_name(model_name)
        with self._lock:
            if name in self._models:
                return True
            logger.info("Loading embedding model %s on %s", name, self._device)
            transformer = models.Transformer(
                name,
                max_seq_length=settings.max_seq_length,
                cache_dir=settings.cache_dir,
            )
            pooling = models.Pooling(
                transformer.get_word_embedding_dimension(),
                pooling_mode_mean_tokens=True,
                pooling_mode_cls_token=False,
                pooling_mode_max_tokens=False,
            )
            model = SentenceTransformer(
                modules=[transformer, pooling],
                device=self._device,
            )
            if settings.use_half_precision and torch.cuda.is_available():
                try:
                    model = model.half()  # type: ignore[assignment]
                except AttributeError:
                    logger.debug("Model %s does not support float16 casting", name)
            self._models[name] = model
        return True

    def _get_model(self, model_name: Optional[str] = None) -> SentenceTransformer:
        name = self.normalize_model_name(model_name)
        if name not in self._models:
            self.load_model(name)
        return self._models[name]

    def encode(self, texts: TextInput, model_name: Optional[str] = None) -> np.ndarray:
        model = self._get_model(model_name)
        embeddings = model.encode(
            texts,
            batch_size=settings.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings

    def get_model_info(self) -> Dict[str, Dict[str, Union[str, int]]]:
        info: Dict[str, Dict[str, Union[str, int]]] = {}
        for name, model in self._models.items():
            info[name] = {
                "device": str(model.device),
                "dimension": model.get_sentence_embedding_dimension(),
            }
        return info

    @property
    def device(self) -> str:
        return self._device
