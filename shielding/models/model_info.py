"""Model info."""

from dataclasses import dataclass
import stormvogel
import gymnasium

@dataclass
class ModelInfo:
    """Information about a model."""

    env: gymnasium.Env
    model: stormvogel.Model
    bad_state: str
    vmin: list[float]
    vmax: list[float]
    map_states: callable
    map_actions: callable
    map_actions_back: callable