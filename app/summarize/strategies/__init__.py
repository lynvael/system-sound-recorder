"""Strategy registry.

`STRATEGIES` is insertion-ordered — the order is the GUI combo order:
map_reduce, eacss, hierarchical.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.summarize.strategies.base import Condense
from app.summarize.strategies.eacss import condense as _eacss_condense
from app.summarize.strategies.hierarchical import (
    condense as _hierarchical_condense,
)
from app.summarize.strategies.map_reduce import condense as _map_reduce_condense


@dataclass(frozen=True)
class StrategyInfo:
    """One selectable summarization method.

    `id` is stable (used in output filenames and prefs); `label` is the
    Russian GUI label; `requires_embeddings` strategies need
    `config.embed.is_configured` (backend guard in the pipeline); `condense`
    is the strategy function.
    """

    id: str
    label: str
    requires_embeddings: bool
    condense: Condense


STRATEGIES: dict[str, StrategyInfo] = {
    "map_reduce": StrategyInfo(
        id="map_reduce",
        label="Map-Reduce",
        requires_embeddings=False,
        condense=_map_reduce_condense,
    ),
    "eacss": StrategyInfo(
        id="eacss",
        label="EACSS (экстрактивно-абстрактивный)",
        requires_embeddings=True,
        condense=_eacss_condense,
    ),
    "hierarchical": StrategyInfo(
        id="hierarchical",
        label="Иерархический (Extract-Support)",
        requires_embeddings=True,
        condense=_hierarchical_condense,
    ),
}

DEFAULT_STRATEGY_ID = "map_reduce"

__all__ = ["Condense", "DEFAULT_STRATEGY_ID", "STRATEGIES", "StrategyInfo"]
