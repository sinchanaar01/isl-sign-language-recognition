"""Landmark sequence architectures and MobileNetV2 static-sign classifier."""

from __future__ import annotations


def build_landmark_model(architecture: str, num_classes: int, input_size: int = 258):
    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise RuntimeError("Install PyTorch to construct recognition models") from exc

    if num_classes < 2 or input_size < 1:
        raise ValueError("num_classes must be >=2 and input_size must be positive")

    if architecture in {"lstm", "gru"}:
        recurrent = nn.LSTM if architecture == "lstm" else nn.GRU

        class RecurrentClassifier(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.recurrent = recurrent(input_size, 128, batch_first=True, num_layers=2,
                                           dropout=0.2, bidirectional=True)
                self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(256, num_classes))

            def forward(self, inputs):
                _, hidden = self.recurrent(inputs)
                return self.head(torch.cat((hidden[-2], hidden[-1]), dim=1))

        return RecurrentClassifier()

    if architecture == "transformer":
        class LandmarkTransformer(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.input_projection = nn.Linear(input_size, 128)
                self.position_embedding = nn.Parameter(torch.zeros(1, 128, 128))
                nn.init.normal_(self.position_embedding, std=0.02)
                layer = nn.TransformerEncoderLayer(
                    d_model=128,
                    nhead=4,
                    dim_feedforward=256,
                    dropout=0.2,
                    batch_first=True,
                    norm_first=True,
                )
                self.encoder = nn.TransformerEncoder(layer, num_layers=2)
                self.head = nn.Sequential(nn.LayerNorm(128), nn.Linear(128, num_classes))

            def forward(self, inputs):
                if inputs.size(1) > self.position_embedding.size(1):
                    raise ValueError("Transformer sequence length exceeds 128 frames")
                projected = self.input_projection(inputs)
                encoded = self.encoder(
                    projected + self.position_embedding[:, :inputs.size(1)]
                )
                return self.head(encoded.mean(dim=1))

        return LandmarkTransformer()

    raise ValueError(f"Unsupported landmark architecture: {architecture}")


def build_mobilenet_v2(num_classes: int, pretrained: bool = False):
    try:
        from torch import nn
        from torchvision.models import MobileNet_V2_Weights, mobilenet_v2
    except ImportError as exc:
        raise RuntimeError("Install PyTorch and torchvision for the static-sign model") from exc
    if num_classes < 2:
        raise ValueError("num_classes must be >=2")
    weights = MobileNet_V2_Weights.DEFAULT if pretrained else None
    model = mobilenet_v2(weights=weights)
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    return model
