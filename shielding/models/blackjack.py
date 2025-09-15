import stormvogel
import stormvogel.bird
import gymnasium
from shielding.models.model_info import ModelInfo
from ast import literal_eval as make_tuple

def available_actions(s):
    # Decision only in live states: 0 == hit, 1 == stick
    if s in ["init", "bust", "done"]:
        return [[]]
    val, usable_ace = s
    if val > 21 or val == 0:
        return [[]]
    else:
        return [["hit"], ["stick"]]

def draw_card():
    # Ranks uniformly likely; 10 occurs four times (10, J, Q, K)
    cards = [1] + list(range(2, 11)) + [10] * 3
    p = 1.0 / 13.0
    return [(p, c) for c in cards]

def _add_card(val, usable_ace, card):
    if card == 1:
        if val + 11 <= 21:
            return val + 11, True
        new_val = val + 1
        if new_val > 21 and usable_ace:
            return new_val - 10, False
        return new_val, usable_ace
    else:
        new_val = val + card
        if new_val > 21 and usable_ace:
            return new_val - 10, False
        return new_val, usable_ace

def delta(s, a):
    # Chance init: deal two cards to the player before any action
    if s == "init":
        outcomes = {}
        for p1, c1 in draw_card():
            for p2, c2 in draw_card():
                v, ua = 0, False
                v, ua = _add_card(v, ua, c1)
                v, ua = _add_card(v, ua, c2)
                nxt = (v, ua)  # cannot bust in two cards with proper ace handling
                outcomes[nxt] = outcomes.get(nxt, 0.0) + p1 * p2
        return [(prob, state) for state, prob in outcomes.items()]
    if s in ["bust", "done"]:
        return [(1.0, s)]
    val, usable_ace = s

    if val > 21:
        return [(1.0, "bust")]

    if a == ["stick"]:  # stick
        return [(1.0, "done")]

    # hit
    outcomes = {}
    for p, c in draw_card():
        nv, nua = _add_card(val, usable_ace, c)
        nxt = "bust" if nv > 21 else (nv, nua)
        outcomes[nxt] = outcomes.get(nxt, 0.0) + p
    return [(prob, state) for state, prob in outcomes.items()]

def labels(s):
    return str(s)

def blackjack():
    blackjack = stormvogel.bird.build_bird(
        delta=delta,
        init="init",
        available_actions=available_actions,
        labels=labels,
        modeltype=stormvogel.ModelType.MDP
    )

    vmin = stormvogel.model_checking(blackjack, "Pmin=? [F \"bust\"]", True).values
    vmax = stormvogel.model_checking(blackjack, "Pmax=? [F \"bust\"]", True).values

    env = gymnasium.make("Blackjack-v1", render_mode="rgb_array")

    def convert_state(state_str):
        if state_str in ["init", "bust", "done"]:
            return state_str
        return make_tuple(state_str)
        

    tuples_to_states = {convert_state(blackjack.states[state].labels[0]): state for state in blackjack.states}
    print(tuples_to_states)

    def map_states(full_state):
        if full_state[0] > 21:
            return tuples_to_states["bust"]
        return tuples_to_states[(full_state[0], full_state[2] == 1)]
    def map_actions(full_action):
        return stormvogel.Action(["stick"]) if full_action == 0 else stormvogel.Action(["hit"])
    def map_actions_back(stormvogel_action):
        return 0 if "stick" in stormvogel_action.labels else 1

    return ModelInfo(env, blackjack, vmin, vmax, map_states, map_actions, map_actions_back)
