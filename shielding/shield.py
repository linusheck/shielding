from dataclasses import dataclass
from stormvogel import Model, Action, Choice, Branch
from shielding.models.model_info import ModelInfo
import random

type Distribution = list[tuple[float, Action]]
def clamp_distribution(distribution: Distribution, allowed_actions: list[Action]) -> Distribution:
    """Clamp a distribution to only allowed actions."""
    total_prob = sum(prob for prob, action in distribution if action in allowed_actions)
    if not total_prob > 0:
        # If no safe actions, return a uniform distribution over allowed actions
        uniform_prob = 1.0 / len(allowed_actions)
        return [(uniform_prob, action) for action in allowed_actions]
    return [(prob / total_prob if action in allowed_actions else 0, action) for prob, action in distribution]

def sample_distribution(distribution: Distribution) -> Action:
    """Sample an action from a probability distribution."""
    probs, actions = zip(*distribution)
    return random.choices(actions, weights=probs, k=1)[0]

@dataclass
class Shield:
    model_info: ModelInfo

    def correct(self, current_state: any, distribution: Distribution):
        """Correct a distribution."""
        raise NotImplementedError("Not implemented")

class IdentityShield(Shield):
    def correct(self, current_state, distribution):
        return distribution

class StandardShield(Shield):

    def _allow_action(self, state: int, branch: Branch):
        next_val = 0.0
        for (value, next_state) in branch:
            next_val += value * self.model_info.vmin[next_state.id]
        return next_val <= self.model_info.vmin[state]

    def correct(self, current_state, distribution: Distribution):
        state = self.model_info.map_states(current_state)
        actions = self.model_info.model.get_choice(state).transition
        allowed_actions = [self.model_info.map_actions_back(a) for a in actions if self._allow_action(state, actions[a])]
        return clamp_distribution(distribution, allowed_actions)
