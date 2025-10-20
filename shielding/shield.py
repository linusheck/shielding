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
    
    def report_info(self):
        return {}

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
        self.last_state = "init"
        self.bmax = self.model_info.vmax[0] - nu
        self.standard_shield = StandardShield(model_info)
        self.last_distr = None

    def _qmax(self, state, distr):
        qmax = 0.0
        actions = self.model_info.model.get_choice(state).transition
        for (actionprob, action) in distr:
            branch = actions[self.model_info.map_actions(action)]
            for (value, next_state) in branch:
                qmax += actionprob * value * self.model_info.vmax[next_state.id]
        return qmax

    def correct(self, last_action, current_state, distribution: Distribution):
        state = self.model_info.map_states(current_state)
        if last_action is None:
            # Reset shield
            self.incurred_safety = 0.0
            self.path_prob = 1.0
            actions = self.model_info.model.get_choice(self.model_info.model.get_initial_state()).transition
            assert len(actions) == 1
            action = list(actions.keys())[0]
            prob = [x for x in actions[action].branch if x[1].id == state][0]
            self.path_prob = self.path_prob * prob[0]
        else:
            actions = self.model_info.model.get_choice(self.model_info.map_states(self.last_state)).transition
            probs = actions[self.model_info.map_actions(last_action)]
            next_state = [x for x in probs.branch if x[1].id == state]
            assert len(next_state) == 1, f"next state not found from {self.last_state} with {last_action} to {current_state} (your fragment is incorrect)"
            prob = next_state[0]
            last_action_prob = [p for p, a in self.last_distr if a == last_action][0]
            self.path_prob = self.path_prob * prob[0] * last_action_prob


        qmax = self._qmax(state, distribution)
        self.incurred_safety += self.path_prob * (self.model_info.vmax[state] - qmax)

        self.last_state = current_state
        self.last_distr = distribution

        if self.incurred_safety > self.bmax:
            return distribution
        else:
            return self.standard_shield.correct(last_action, current_state, distribution)

    def report_info(self):
        return {
            "shield": "pessimistic",
            "bmax": self.bmax,
            "safety": self.incurred_safety,
            "pathprob": self.path_prob,
            "safety > bmax": self.incurred_safety > self.bmax,
        }


class PessimisticShield2(Shield):
    def __init__(self, model_info: ModelInfo, nu: float):
        super().__init__(model_info)
        self.incurred_risk = 0.0
        self.path_prob = 1.0
        self.last_state = "init"
        self.bmin = nu - self.model_info.vmin[0]
        self.standard_shield = StandardShield(model_info)

    def _qmax(self, state, distr):
        qmax = 0.0
        actions = self.model_info.model.get_choice(state).transition
        for (actionprob, action) in distr:
            branch = actions[self.model_info.map_actions(action)]
            for (value, next_state) in branch:
                qmax += actionprob * value * self.model_info.vmax[next_state.id]
        return qmax

    def correct(self, last_action, current_state, distribution: Distribution):
        state = self.model_info.map_states(current_state)
        if last_action is None:
            # Reset shield
            self.incurred_risk = 0.0
            self.path_prob = 1.0
            actions = self.model_info.model.get_choice(self.model_info.model.get_initial_state()).transition
            assert len(actions) == 1
            action = list(actions.keys())[0]
            prob = [x for x in actions[action].branch if x[1].id == state][0]
            self.path_prob = self.path_prob * prob[0]
        else:
            actions = self.model_info.model.get_choice(self.model_info.map_states(self.last_state)).transition
            probs = actions[self.model_info.map_actions(last_action)]
            prob = [x for x in probs.branch if x[1].id == state][0]
            self.path_prob = self.path_prob * prob[0]


        qmax = self._qmax(state, distribution)
        self.incurred_risk += self.path_prob * (qmax - self.model_info.vmin[state])

        self.last_state = current_state

        if self.incurred_risk <= self.bmin:
            return distribution
        else:
            return self.standard_shield.correct(last_action, current_state, distribution)

    def report_info(self):
        return {
            "shield": "pessimistic2",
            "bmax": self.bmin,
            "risk": self.incurred_risk,
            "pathprob": self.path_prob,
            "risk <= bmin": self.incurred_risk <= self.bmin,
        }


type Distribution = list[float]

@dataclass
class Node:
    successors: "list[Node]"
    predecessor: "Node"
    distributions: list[Distribution]
    state_in_mc: int

class SelfConstructingShield(Shield):
    # try with tree structure

    def __init__(self, model_info):
        super().__init__(model_info)

        # assumption
        initial_state = 0

        # get vmin
        self.initial_node = Node([], None, [], initial_state)

