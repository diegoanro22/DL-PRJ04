"""Augmentations de audio para train. Contrato en src/data/README.md (sección 5).

Todas comparten la firma `fn(wav: np.ndarray, sr: int) -> np.ndarray`.
La implementación está a cargo de Cisco.
"""

from collections.abc import Callable

import numpy as np

Augment = Callable[[np.ndarray, int], np.ndarray]


def telephony(wav: np.ndarray, sr: int) -> np.ndarray:
    """Baja a 8 kHz y vuelve a 16 kHz, y aplica un pasa-banda de 300 a 3400 Hz."""
    raise NotImplementedError("Pendiente: Cisco")


def rawboost(wav: np.ndarray, sr: int) -> np.ndarray:
    """RawBoost (Tak et al., 2022): ruido convolutivo, impulsivo y aditivo coloreado."""
    raise NotImplementedError("Pendiente: Cisco")
