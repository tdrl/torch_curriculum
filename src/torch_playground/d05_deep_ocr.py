"""A personal implementation of the DeepOCR architecture.

Citation:
Wei, H.,  Sun, Y., & Li, Y., (2025) DeepSeek-OCR: Contexts Optical Compression
    https://arxiv.org/abs/2510.18234v1
"""

import torch
import torch.nn as nn

from segment_anything import sam_model_registry, SamPredictor

from torch_playground.util import (
    BaseConfiguration,
    TrainableModelApp,
    save_tensor,
)
from dataclasses import dataclass, field, asdict
from pathlib import Path
import json

@dataclass
class DeepOCRConfig(BaseConfiguration):
    """Configuration parameters for DeepOCR."""

    hidden_dim: int = field(default=256, metadata=BaseConfiguration._meta(help='Hidden dimension size.'))


class DeepOCRModel(nn.Module):
    def __init__(self, config: DeepOCRConfig):
        super().__init__()
        self.hidden_dim = config.hidden_dim
        sam_model = sam_model_registry["vit_b"](checkpoint="sam_vit_b_01ec64.pth")
        img_embedding = SamPredictor(sam_model)
        self.model = nn.Sequential(
            sam_model,
            
            
        )


class DeepOCRApp(TrainableModelApp):
    def __init__(self, config: DeepOCRConfig):
        super().__init__(config)

        self.model: DeepOCRModel | None = None
    
    def run(self):
        self.logger.info('Starting DeepOCRApp...')
        self.model = DeepOCRModel(self.config).to(self.device)