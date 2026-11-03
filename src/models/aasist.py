"""AASIST y AASIST-L (Jung et al., 2022) sobre la forma de onda.

Adaptado de la implementación oficial (github.com/clovaai/aasist, licencia MIT). Se mantiene
la arquitectura igual a la original para poder comparar con el EER publicado de AASIST-L
(0.99 % en ASVspoof 2019 LA eval). Cambios: la cabeza de grafos se separó en `GraphHead` para
reusarla en el modelo XLS-R + AASIST (src/models/xlsr_aasist.py), los filtros sinc son un
buffer (no se entrenan, igual que en el original) y `forward` devuelve solo los logits.

Entrada: [B, 64600] (4 s a 16 kHz). Salida: logits [B, 2], columna 1 = sintética.
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def _xavier_param(*size: int) -> nn.Parameter:
    param = nn.Parameter(torch.empty(*size))
    nn.init.xavier_normal_(param)
    return param


def _batch_norm_nodes(bn: nn.BatchNorm1d, x: torch.Tensor) -> torch.Tensor:
    """BatchNorm sobre la última dimensión de [B, nodos, dim]."""
    shape = x.shape
    return bn(x.reshape(-1, shape[-1])).reshape(shape)


class GraphAttentionLayer(nn.Module):
    """Atención de grafo sobre un solo tipo de nodo (espectral o temporal)."""

    def __init__(self, in_dim: int, out_dim: int, temperature: float = 1.0):
        super().__init__()
        self.att_proj = nn.Linear(in_dim, out_dim)
        self.att_weight = _xavier_param(out_dim, 1)
        self.proj_with_att = nn.Linear(in_dim, out_dim)
        self.proj_without_att = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)
        self.input_drop = nn.Dropout(p=0.2)
        self.act = nn.SELU(inplace=True)
        self.temp = temperature

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_drop(x)
        # Mapa de atención entre todos los pares de nodos: [B, N, N, 1].
        pairs = x.unsqueeze(2) * x.unsqueeze(1)
        att = torch.matmul(torch.tanh(self.att_proj(pairs)), self.att_weight)
        att = F.softmax(att / self.temp, dim=-2)
        x = self.proj_with_att(torch.matmul(att.squeeze(-1), x)) + self.proj_without_att(x)
        return self.act(_batch_norm_nodes(self.bn, x))


class HtrgGraphAttentionLayer(nn.Module):
    """Atención de grafo heterogéneo (nodos temporales + espectrales) con un nodo maestro."""

    def __init__(self, in_dim: int, out_dim: int, temperature: float = 1.0):
        super().__init__()
        self.proj_type1 = nn.Linear(in_dim, in_dim)
        self.proj_type2 = nn.Linear(in_dim, in_dim)
        self.att_proj = nn.Linear(in_dim, out_dim)
        self.att_projM = nn.Linear(in_dim, out_dim)
        self.att_weight11 = _xavier_param(out_dim, 1)
        self.att_weight22 = _xavier_param(out_dim, 1)
        self.att_weight12 = _xavier_param(out_dim, 1)
        self.att_weightM = _xavier_param(out_dim, 1)
        self.proj_with_att = nn.Linear(in_dim, out_dim)
        self.proj_without_att = nn.Linear(in_dim, out_dim)
        self.proj_with_attM = nn.Linear(in_dim, out_dim)
        self.proj_without_attM = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)
        self.input_drop = nn.Dropout(p=0.2)
        self.act = nn.SELU(inplace=True)
        self.temp = temperature

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, master: torch.Tensor):
        n1, n2 = x1.size(1), x2.size(1)
        x = torch.cat([self.proj_type1(x1), self.proj_type2(x2)], dim=1)
        x = self.input_drop(x)

        # Atención entre nodos con pesos distintos según el tipo de cada par.
        pairs = torch.tanh(self.att_proj(x.unsqueeze(2) * x.unsqueeze(1)))  # [B, N, N, D]
        att = torch.zeros_like(pairs[..., :1])
        att[:, :n1, :n1] = torch.matmul(pairs[:, :n1, :n1], self.att_weight11)
        att[:, n1:, n1:] = torch.matmul(pairs[:, n1:, n1:], self.att_weight22)
        att[:, :n1, n1:] = torch.matmul(pairs[:, :n1, n1:], self.att_weight12)
        att[:, n1:, :n1] = torch.matmul(pairs[:, n1:, :n1], self.att_weight12)
        att = F.softmax(att / self.temp, dim=-2)

        # Actualización del nodo maestro con atención sobre todos los nodos.
        att_m = torch.matmul(torch.tanh(self.att_projM(x * master)), self.att_weightM)
        att_m = F.softmax(att_m / self.temp, dim=-2)
        master = (self.proj_with_attM(torch.matmul(att_m.squeeze(-1).unsqueeze(1), x))
                  + self.proj_without_attM(master))

        x = self.proj_with_att(torch.matmul(att.squeeze(-1), x)) + self.proj_without_att(x)
        x = self.act(_batch_norm_nodes(self.bn, x))
        return x.narrow(1, 0, n1), x.narrow(1, n1, n2), master


class GraphPool(nn.Module):
    """Se queda con la fracción `k` de nodos con mayor puntaje."""

    def __init__(self, k: float, in_dim: int, p: float):
        super().__init__()
        self.k = k
        self.proj = nn.Linear(in_dim, 1)
        self.drop = nn.Dropout(p=p) if p > 0 else nn.Identity()

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        scores = torch.sigmoid(self.proj(self.drop(h)))
        n_keep = max(int(h.size(1) * self.k), 1)
        _, idx = torch.topk(scores, n_keep, dim=1)
        return torch.gather(h * scores, 1, idx.expand(-1, -1, h.size(-1)))


class SincConv(nn.Module):
    """Banco de filtros pasa-banda sinc con frecuencias en escala mel (fijos, no se entrenan)."""

    def __init__(self, out_channels: int, kernel_size: int, sample_rate: int = 16000):
        super().__init__()
        if kernel_size % 2 == 0:
            kernel_size += 1
        n_fft = 512
        freqs = int(sample_rate / 2) * np.linspace(0, 1, n_fft // 2 + 1)
        mel = 2595 * np.log10(1 + freqs / 700)
        edges_mel = np.linspace(mel.min(), mel.max(), out_channels + 1)
        edges_hz = 700 * (10 ** (edges_mel / 2595) - 1)
        t = np.arange(-(kernel_size - 1) / 2, (kernel_size - 1) / 2 + 1)
        window = np.hamming(kernel_size)
        filters = np.zeros((out_channels, kernel_size))
        for i in range(out_channels):
            f_low, f_high = edges_hz[i], edges_hz[i + 1]
            high = (2 * f_high / sample_rate) * np.sinc(2 * f_high * t / sample_rate)
            low = (2 * f_low / sample_rate) * np.sinc(2 * f_low * t / sample_rate)
            filters[i] = window * (high - low)
        self.register_buffer("filters", torch.tensor(filters, dtype=torch.float32).unsqueeze(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv1d(x, self.filters.to(x.dtype))


class ResidualBlock(nn.Module):
    """Bloque residual 2D del encoder. `pool=False` en la variante de XLS-R (Tak et al.)."""

    def __init__(self, nb_filts: list[int], first: bool = False, pool: bool = True):
        super().__init__()
        self.first = first
        if not first:
            self.bn1 = nn.BatchNorm2d(nb_filts[0])
        self.conv1 = nn.Conv2d(nb_filts[0], nb_filts[1], kernel_size=(2, 3), padding=(1, 1))
        self.selu = nn.SELU(inplace=True)
        self.bn2 = nn.BatchNorm2d(nb_filts[1])
        self.conv2 = nn.Conv2d(nb_filts[1], nb_filts[1], kernel_size=(2, 3), padding=(0, 1))
        self.downsample = nb_filts[0] != nb_filts[1]
        if self.downsample:
            self.conv_downsample = nn.Conv2d(nb_filts[0], nb_filts[1], kernel_size=(1, 3),
                                             padding=(0, 1))
        self.mp = nn.MaxPool2d((1, 3)) if pool else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        # Igual que el original: conv1 recibe x (la rama bn1 + SELU no llega a usarse).
        # Se conserva para reproducir los resultados publicados.
        out = self.conv1(x)
        out = self.selu(self.bn2(out))
        out = self.conv2(out)
        if self.downsample:
            identity = self.conv_downsample(identity)
        return self.mp(out + identity)


def build_encoder(filts: list, pool: bool = True) -> nn.Sequential:
    return nn.Sequential(
        ResidualBlock(filts[1], first=True, pool=pool),
        ResidualBlock(filts[2], pool=pool),
        ResidualBlock(filts[3], pool=pool),
        ResidualBlock(filts[4], pool=pool),
        ResidualBlock(filts[4], pool=pool),
        ResidualBlock(filts[4], pool=pool),
    )


class GraphHead(nn.Module):
    """Parte de grafos de AASIST: de nodos espectrales y temporales a logits [B, 2].

    Dos ramas paralelas de grafo heterogéneo con su nodo maestro; se combinan con máximo
    elemento a elemento y la salida concatena max/promedio de cada tipo de nodo + el maestro.
    """

    def __init__(self, in_dim: int, n_spectral: int, gat_dims: list[int],
                 pool_ratios: list[float], temperatures: list[float]):
        super().__init__()
        self.pos_S = nn.Parameter(torch.randn(1, n_spectral, in_dim))
        self.master1 = nn.Parameter(torch.randn(1, 1, gat_dims[0]))
        self.master2 = nn.Parameter(torch.randn(1, 1, gat_dims[0]))
        self.GAT_layer_S = GraphAttentionLayer(in_dim, gat_dims[0], temperatures[0])
        self.GAT_layer_T = GraphAttentionLayer(in_dim, gat_dims[0], temperatures[1])
        self.HtrgGAT_layer_ST11 = HtrgGraphAttentionLayer(gat_dims[0], gat_dims[1], temperatures[2])
        self.HtrgGAT_layer_ST12 = HtrgGraphAttentionLayer(gat_dims[1], gat_dims[1], temperatures[2])
        self.HtrgGAT_layer_ST21 = HtrgGraphAttentionLayer(gat_dims[0], gat_dims[1], temperatures[2])
        self.HtrgGAT_layer_ST22 = HtrgGraphAttentionLayer(gat_dims[1], gat_dims[1], temperatures[2])
        self.pool_S = GraphPool(pool_ratios[0], gat_dims[0], 0.3)
        self.pool_T = GraphPool(pool_ratios[1], gat_dims[0], 0.3)
        self.pool_hS1 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)
        self.pool_hT1 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)
        self.pool_hS2 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)
        self.pool_hT2 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)
        self.drop_way = nn.Dropout(0.2, inplace=True)
        self.drop = nn.Dropout(0.5, inplace=True)
        self.out_layer = nn.Linear(5 * gat_dims[1], 2)

    def _branch(self, out_T, out_S, master, gat1, gat2, pool_S, pool_T):
        out_T1, out_S1, master1 = gat1(out_T, out_S, master)
        out_S1, out_T1 = pool_S(out_S1), pool_T(out_T1)
        aug_T, aug_S, aug_m = gat2(out_T1, out_S1, master1)
        return out_T1 + aug_T, out_S1 + aug_S, master1 + aug_m

    def forward(self, e_S: torch.Tensor, e_T: torch.Tensor) -> torch.Tensor:
        """e_S: [B, n_spectral, in_dim]; e_T: [B, n_temporal, in_dim]."""
        out_S = self.pool_S(self.GAT_layer_S(e_S + self.pos_S))
        out_T = self.pool_T(self.GAT_layer_T(e_T))

        batch = e_S.size(0)
        T1, S1, m1 = self._branch(out_T, out_S, self.master1.expand(batch, -1, -1),
                                  self.HtrgGAT_layer_ST11, self.HtrgGAT_layer_ST12,
                                  self.pool_hS1, self.pool_hT1)
        T2, S2, m2 = self._branch(out_T, out_S, self.master2.expand(batch, -1, -1),
                                  self.HtrgGAT_layer_ST21, self.HtrgGAT_layer_ST22,
                                  self.pool_hS2, self.pool_hT2)
        T1, T2, S1, S2 = (self.drop_way(t) for t in (T1, T2, S1, S2))
        m1, m2 = self.drop_way(m1), self.drop_way(m2)

        out_T, out_S, master = torch.max(T1, T2), torch.max(S1, S2), torch.max(m1, m2)
        hidden = torch.cat([
            out_T.abs().max(dim=1).values, out_T.mean(dim=1),
            out_S.abs().max(dim=1).values, out_S.mean(dim=1),
            master.squeeze(1),
        ], dim=1)
        return self.out_layer(self.drop(hidden))


class AASIST(nn.Module):
    """AASIST sobre la forma de onda. Con los hiperparámetros de configs/aasist_l.yaml es
    AASIST-L (85k parámetros)."""

    def __init__(self, filts: list, gat_dims: list[int], pool_ratios: list[float],
                 temperatures: list[float], first_conv: int = 128, sample_rate: int = 16000):
        super().__init__()
        self.conv_time = SincConv(filts[0], first_conv, sample_rate)
        self.first_bn = nn.BatchNorm2d(1)
        self.selu = nn.SELU(inplace=True)
        self.encoder = build_encoder(filts, pool=True)
        # Nodos espectrales: los canales sinc después del max-pool de 3.
        self.head = GraphHead(filts[-1][-1], filts[0] // 3, gat_dims, pool_ratios, temperatures)

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        x = self.conv_time(wav.unsqueeze(1)).unsqueeze(1)    # [B, 1, filtros, tiempo]
        x = F.max_pool2d(x.abs(), (3, 3))
        x = self.selu(self.first_bn(x))
        e = self.encoder(x)                                  # [B, C, F, T]
        e_S = e.abs().max(dim=3).values.transpose(1, 2)      # nodos espectrales [B, F, C]
        e_T = e.abs().max(dim=2).values.transpose(1, 2)      # nodos temporales [B, T, C]
        return self.head(e_S, e_T)
