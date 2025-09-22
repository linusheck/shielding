import stormvogel
import stormvogel.bird
import gymnasium
from shielding.models.model_info import ModelInfo
from ast import literal_eval as make_tuple

def draw_card():
    return []

def labels(s):
    return str(s)

def cliffwalking():
    def available_actions(s):
        if s == "init":
            return [[]]
        return [["up"], ["down"], ["left"], ["right"]]
    def delta(s, a):
        if s == "fall":
            return [(1.0, "fall")]
        if s == "init":
            return [(1.0, (3, 0))]
        y, x = s
        new_s = None
        if a == ["right"]:
            new_s = (y, min(x + 1, 11))
        elif a == ["left"]:
            new_s = (y, max(x - 1, 0))
        elif a == ["up"]:
            new_s = (max(y - 1, 0), x)
        elif a == ["down"]:
            new_s = (min(y + 1, 3), x)
        y, x = new_s
        if y == 3 and 1 <= x <= 10:
            return [(1.0, "fall")]
        return [(1.0, new_s)]

    cliffwalking = stormvogel.bird.build_bird(
        delta=delta,
        init="init",
        available_actions=available_actions,
        labels=labels,
        modeltype=stormvogel.ModelType.MDP
    )

    vmin = stormvogel.model_checking(cliffwalking, "Pmin=? [F \"fall\"]", True).values
    vmax = stormvogel.model_checking(cliffwalking, "Pmax=? [F \"fall\"]", True).values

    env = gymnasium.make("CliffWalking-v1", render_mode="rgb_array", is_slippery=False)

    def convert_state(state_str):
        if state_str in ["init", "fall"]:
            return state_str
        return make_tuple(state_str)

    tuples_to_states = {convert_state(cliffwalking.states[state].labels[0]): state for state in cliffwalking.states}
    actions = [cliffwalking.get_action_with_labels(frozenset([x])) for x in ["up", "right", "down", "left"]]

    def map_states(full_state):
        # The observation is a value representing the player’s current position as current_row * ncols + current_col (where both the row and col start at 0).
        y = full_state // 12
        x = full_state % 12
        if y == 3 and 1 <= x <= 10:
            return "fall"
        return tuples_to_states[(y, x)]
    def map_actions(full_action):
        return actions[full_action]
    def map_actions_back(stormvogel_action):
        return actions.index(stormvogel_action)

    return ModelInfo(env, cliffwalking, "bust", vmin, vmax, map_states, map_actions, map_actions_back)
