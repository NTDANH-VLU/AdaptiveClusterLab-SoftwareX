"""Interfaces for importing verified published research implementations."""
from __future__ import annotations
from typing import Protocol
import numpy as np
from numpy.typing import NDArray

class Clusterer(Protocol):
    """Implementations must accept the same input representation as baselines."""
    def fit_predict(self, x: NDArray[np.float64]) -> NDArray[np.int64]: ...

class AdaptiveClusterer(Protocol):
    """Sequential API; implement published model behavior without silent changes."""
    def partial_fit_predict(
        self, x: NDArray[np.float64], batch_index: int
    ) -> NDArray[np.int64]: ...
