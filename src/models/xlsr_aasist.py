"""XLS-R 300M + cabeza AASIST (Tak et al., 2022, "Automatic speaker verification spoofing and
deepfake detection using wav2vec 2.0 and data augmentation").

Adaptado de github.com/TakHemlata/SSL_Anti-spoofing, con `transformers` en lugar de fairseq.
XLS-R saca 1024 dimensiones cada 20 ms (201 cuadros en 4 s); una capa lineal las baja a 128 y
la cabeza AASIST trata esas 128 dimensiones como eje espectral y los cuadros como eje
temporal. A diferencia del AASIST sobre forma de onda, los bloques residuales no reducen el
tiempo y los nodos se forman con atención en vez de máximo.

Dos modos (transfer learning):
- `freeze_ssl: true` (C0, modelo base): XLS-R congelado, solo se entrena la cabeza.
- `freeze_ssl: false` (C1, C2): fine-tuning completo, lr 1e-6 en XLS-R y 1e-4 en la cabeza.

Entrada: [B, 64600]. Salida: logits [B, 2], columna 1 = sintética.
"""

import torch
import torch.nn.functional as F
from torch import nn
from transformers import Wav2Vec2Config, Wav2Vec2Model

from src.models.aasist import GraphHead, build_encoder

# Hiperparámetros de la cabeza en Tak et al. (2022).
FILTS = [128, [1, 32], [32, 32], [32, 64], [64, 64]]
GAT_DIMS = [64, 32]
POOL_RATIOS = [0.5, 0.5, 0.5, 0.5]
TEMPERATURES = [2.0, 2.0, 100.0, 100.0]


class XLSRAASIST(nn.Module):
    def __init__(
        self,
        ssl_name: str = "facebook/wav2vec2-xls-r-300m",
        freeze_ssl: bool = False,
        freeze_first_layers: int = 0,
        gradient_checkpointing: bool = False,
        normalize_input: bool = True,
        ssl_config: dict | None = None,
    ):
        """
        freeze_first_layers: plan B de memoria; congela el extractor CNN y las primeras N
            capas transformer (solo aplica con freeze_ssl false).
        normalize_input: media 0 y varianza 1 por audio, como en el preentrenamiento de XLS-R.
        ssl_config: solo para pruebas; crea un wav2vec2 chico con pesos aleatorios en vez de
            bajar XLS-R.
        """
        super().__init__()
        self.freeze_ssl = freeze_ssl
        self.normalize_input = normalize_input

        # Sin layerdrop ni SpecAugment, igual que en Tak et al. (fairseq con mask=False).
        if ssl_config is not None:
            self.ssl = Wav2Vec2Model(Wav2Vec2Config(**ssl_config))
        else:
            self.ssl = Wav2Vec2Model.from_pretrained(ssl_name, layerdrop=0.0,
                                                     apply_spec_augment=False)
        if freeze_ssl:
            self.ssl.requires_grad_(False)
        else:
            if gradient_checkpointing:
                self.ssl.gradient_checkpointing_enable()
            if freeze_first_layers > 0:
                self.ssl.freeze_feature_encoder()
                for layer in self.ssl.encoder.layers[:freeze_first_layers]:
                    layer.requires_grad_(False)

        self.LL = nn.Linear(self.ssl.config.hidden_size, FILTS[0])
        self.first_bn = nn.BatchNorm2d(1)
        self.selu = nn.SELU(inplace=True)
        self.encoder = build_encoder(FILTS, pool=False)
        self.first_bn1 = nn.BatchNorm2d(FILTS[-1][-1])
        self.attention = nn.Sequential(
            nn.Conv2d(FILTS[-1][-1], 128, kernel_size=1),
            nn.SELU(inplace=True),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, FILTS[-1][-1], kernel_size=1),
        )
        self.head = GraphHead(FILTS[-1][-1], FILTS[0] // 3, GAT_DIMS, POOL_RATIOS, TEMPERATURES)

    def train(self, mode: bool = True):
        super().train(mode)
        # Congelado: XLS-R siempre en eval para que su dropout no meta ruido en las features.
        if self.freeze_ssl:
            self.ssl.eval()
        return self

    def param_groups(self, train_cfg: dict) -> list[dict]:
        """lr distinto para XLS-R y para la cabeza; usado por src/train.py."""
        head = [p for name, p in self.named_parameters()
                if not name.startswith("ssl.") and p.requires_grad]
        lr_head = train_cfg.get("lr_head", train_cfg.get("lr"))
        groups = [{"params": head, "lr": lr_head}]
        ssl = [p for p in self.ssl.parameters() if p.requires_grad]
        if ssl:
            groups.insert(0, {"params": ssl, "lr": train_cfg["lr_ssl"]})
        return groups

    def extract(self, wav: torch.Tensor) -> torch.Tensor:
        """Features de la última capa de XLS-R: [B, cuadros, 1024]."""
        if self.normalize_input:
            wav = (wav - wav.mean(dim=1, keepdim=True)) / (wav.std(dim=1, keepdim=True) + 1e-7)
        if self.freeze_ssl:
            with torch.no_grad():
                return self.ssl(wav).last_hidden_state
        return self.ssl(wav).last_hidden_state

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        x = self.LL(self.extract(wav))                  # [B, cuadros, 128]
        x = x.transpose(1, 2).unsqueeze(1)              # [B, 1, 128, cuadros]
        x = F.max_pool2d(x, (3, 3))
        x = self.selu(self.first_bn(x))
        x = self.selu(self.first_bn1(self.encoder(x)))  # [B, 64, 42, T]
        w = self.attention(x)
        e_S = (x * F.softmax(w, dim=-1)).sum(dim=-1).transpose(1, 2)  # nodos espectrales
        e_T = (x * F.softmax(w, dim=-2)).sum(dim=-2).transpose(1, 2)  # nodos temporales
        return self.head(e_S, e_T)
