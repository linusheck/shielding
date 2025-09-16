from dataclasses import dataclass
from stormvogel import Model, Action, Choice, Branch, State
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

class Shield:
    def __init__(self, model_info: ModelInfo):
        self.model_info = model_info

    def correct(self, last_action: any, current_state: any, distribution: Distribution):
        """Correct a distribution."""
        raise NotImplementedError("Not implemented")

class IdentityShield(Shield):
    def __init__(self, model_info):
        super().__init__(model_info)
    def correct(self, last_action, current_state, distribution):
        return distribution

class StandardShield(Shield):
    def __init__(self, model_info):
        super().__init__(model_info)

    def _allow_action(self, state: int, branch: Branch):
        next_val = 0.0
        for (value, next_state) in branch:
            next_val += value * self.model_info.vmin[next_state.id]
        return next_val <= self.model_info.vmin[state]

    def correct(self, last_action, current_state, distribution: Distribution):
        state = self.model_info.map_states(current_state)
        actions = self.model_info.model.get_choice(state).transition
        allowed_actions = [self.model_info.map_actions_back(a) for a in actions if self._allow_action(state, actions[a])]
        return clamp_distribution(distribution, allowed_actions)

class PessimisticShield(Shield):
    def __init__(self, model_info: ModelInfo, nu: float):
        super().__init__(model_info)
        self.incurred_safety = 0.0
        self.path_prob = 1.0
        self.last_state = None
        self.bmax = self.model_info.vmax[0] - nu
        self.standard_shield = StandardShield(model_info)

    def _qmax(self, state, distr):
        qmax = 0.0
        actions = self.model_info.model.get_choice(state).transition
        for (actionprob, action) in distr:
            branch = [a for a in actions if self.model_info.map_actions(action) in a.labels][0]
            for (value, next_state) in actions[branch]:
                qmax += actionprob * value * self.model_info.vmax[next_state.id]
        return qmax

    def correct(self, last_action, current_state, distribution: Distribution):
        state = self.model_info.map_states(current_state)
        if last_action is None:
            self.incurred_safety = 0.0
            self.path_prob = 1.0
        else:
            actions = self.model_info.model.get_choice(self.model_info.map_states(self.last_state)).transition
            action = self.model_info.map_actions(last_action)
            probs = [actions[x] for x in actions if action in x.labels][0]
            prob = [x for x in probs.branch if x[1].id == state][0]
            self.path_prob = self.path_prob * prob[0]


        qmax = self._qmax(state, distribution)
        self.incurred_safety += self.path_prob * (self.model_info.vmax[state] - qmax)

        print("Diff", self.model_info.vmax[state] - qmax)
        print("Path Prob", self.path_prob)
        print("Incurred Safety", self.incurred_safety)

        self.last_state = current_state

        if self.incurred_safety > self.bmax:
            return distribution
        else:
            return self.standard_shield.correct(last_action, current_state, distribution)

