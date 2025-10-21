from manim import *
import random
import numpy as np
from stormvogel import EmptyAction

# Optional theming, keep if available
try:
    from manim_themes.manim_theme import apply_theme
except Exception:
    def apply_theme(*args, **kwargs):
        pass

# Shielding/model imports
from shielding.shield import sample_distribution, PessimisticShield, IdentityShield, PessimisticShield2, SelfConstructingShield
from shielding.models import blackjack, cliffwalking

# ------------- Utilities (unchanged) -------------
def values_from_distribution(distr):
    probs = {a: 0.0 for a in range(len(distr))}
    for p, a in distr:
        probs[a] = p
    return [probs[a] for a in range(len(distr))]

def format_dict(shield_info):
    return "\n".join(
        [f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" for k, v in shield_info.items()]
    )

# ------------- Drawer protocol and implementations -------------

class DrawerProtocol:
    """Event-driven drawer interface. All methods are optional to implement."""
    def on_env_ready(self, frame, num_actions):  # called once after initial render
        pass
    def on_episode_start(self, episode_idx: int, state, frame):
        pass
    def on_step(self, state, action, distr, corrected_distr, shield_info: dict, frame):
        pass
    def on_episode_end(self, state, end_text: str):
        pass

class NullDrawer(DrawerProtocol):
    """No-op drawer to disable all drawing."""
    pass

class ManimDrawer(DrawerProtocol):
    """Stateful Manim-backed drawer; holds and updates its own Mobjects."""
    def __init__(self, scene: Scene, theme_name: str = "Monokai Pro Light"):
        self.scene = scene
        apply_theme(manim_scene=self.scene, theme_name=theme_name, light_theme=True)

        # Persistent state
        self.img = None
        self.chart_orig = None
        self.chart_corr = None
        self.status = None
        self.shield_info_text = None
        self.visited_states = []
        self.visited_text = None

        # Layout containers
        self.layout_group = None

    def _build_layout(self, initial_frame: np.ndarray, num_actions):
        # Left image
        self.img = ImageMobject(initial_frame)
        self.img.width = 6

        # Right charts
        self.chart_orig = BarChart(
            values=[0.5 for _ in range(num_actions)],
            bar_names=[str(i) for i in range(num_actions)],
            y_range=[0, 1, 0.2],
            y_length=2,
            x_length=2,
            x_axis_config={"font_size": 28},
            y_axis_config={"font_size": 24},
        )
        self.chart_corr = BarChart(
            values=[0.5 for _ in range(num_actions)],
            bar_names=[str(i) for i in range(num_actions)],
            y_range=[0, 1, 0.2],
            y_length=2,
            x_length=2,
            x_axis_config={"font_size": 28},
            y_axis_config={"font_size": 24},
        )
        right_col = Group(self.chart_orig, self.chart_corr).arrange(RIGHT, buff=0.5)
        upper_row = Group(self.img, right_col).arrange(RIGHT, buff=1.0)

        self.status = Text("game 0", font_size=30).to_corner(DOWN + LEFT)
        self.shield_info_text = Text("", font_size=20, font="IBM Plex Mono").to_corner(DOWN + RIGHT)
        self.visited_text = Text("", font_size=20, font="IBM Plex Mono").to_edge(DOWN)

        self.layout_group = Group(upper_row).arrange(DOWN)
        self.scene.add(self.layout_group, self.status, self.visited_text, self.shield_info_text)

    def on_env_ready(self, frame, num_actions):
        # Build the initial layout once we have an RGB frame from env.render()
        self._build_layout(frame, num_actions)

    def on_episode_start(self, episode_idx: int, state, frame):
        animations = []
        self.visited_states = [str(state)]
        new_status = Text(f"game {episode_idx}", font_size=30).to_corner(DOWN + LEFT)
        animations.append(Transform(self.status, new_status))

        # Refresh visited states text
        new_v = Text("\n".join(self.visited_states[-4:]), font_size=20, font="IBM Plex Mono").to_edge(DOWN)
        animations.append(ReplacementTransform(self.visited_text, new_v))
        self.visited_text = new_v

        # Update image if a new frame is provided
        if frame is not None:
            new_img = ImageMobject(frame)
            new_img.height = self.img.height
            new_img.move_to(self.img)
            animations.append(ReplacementTransform(self.img, new_img))
            self.img = new_img

        self.scene.play(*animations, run_time=0.1)

    def on_step(self, state, action, distr, corrected_distr, shield_info: dict, frame):
        animations = []
        # Update charts
        v_orig = values_from_distribution(distr)
        v_corr = values_from_distribution(corrected_distr)
        animations.append(self.chart_orig.animate.change_bar_values(v_orig))
        animations.append(self.chart_corr.animate.change_bar_values(v_corr))

        # Update shield info
        new_info = Text(format_dict(shield_info), font_size=20, font="IBM Plex Mono").to_corner(DOWN + RIGHT)
        animations.append(ReplacementTransform(self.shield_info_text, new_info))
        self.shield_info_text = new_info

        # Update visited states text
        self.visited_states.append(f"{action} -> {state}")
        new_v = Text("\n".join(self.visited_states[-4:]), font_size=20, font="IBM Plex Mono").to_edge(DOWN)
        animations.append(ReplacementTransform(self.visited_text, new_v))
        self.visited_text = new_v

        # Update image if a new frame is provided
        if frame is not None:
            new_img = ImageMobject(frame)
            new_img.height = self.img.height
            new_img.move_to(self.img)
            animations.append(ReplacementTransform(self.img, new_img))
            self.img = new_img

        self.scene.play(*animations, run_time=0.1)

    def on_episode_end(self, state, end_text: str):
        end_status = Text(end_text, font_size=30).to_corner(DOWN + LEFT)
        self.scene.play(Transform(self.status, end_status), run_time=0.4)
        self.scene.wait(1.5)

# ------------- Environment runner (pure logic) -------------

class EnvRunner:
    """Headless runner that drives the env and shield; emits events to a drawer."""
    def __init__(self, env, shield, actions=(0, 1), rng=None):
        self.env = env
        self.shield = shield
        self.actions = actions
        self.rng = rng or random.Random()

    def _maybe_render(self, enabled: bool):
        if not enabled:
            return None
        try:
            frame = self.env.render()
        except Exception:
            frame = None
        return frame

    def run(self, episodes: int, drawer: DrawerProtocol, render: bool = True, max_steps: int = 50):
        number_bad = 0

        for ep in range(episodes):
            # Gymnasium reset returns (obs, info)
            reset_out = self.env.reset()
            state, _ = reset_out

            frame = self._maybe_render(render)
            if ep == 0:
                drawer.on_env_ready(frame, len(self.actions))
            drawer.on_episode_start(ep, state, frame)

            last_action = None
            terminated = False
            truncated = False

            last_action = None

            step_count = 0
            while not (terminated or truncated):
                # Random distribution over all actions and shield correction
                probs = [self.rng.random() for _ in self.actions]
                total = sum(probs)
                distr = [(p / total, a) for p, a in zip(probs, self.actions)]
                corrected_distr = self.shield.correct(last_action, state, distr)
                # Choose action and step
                action = sample_distribution(corrected_distr)
                step_out = self.env.step(action)

                next_state, _, terminated, truncated, _ = step_out

                # Optional render frame
                frame = self._maybe_render(render)

                # Report to drawer
                drawer.on_step(
                    state=next_state,
                    action=action,
                    distr=distr,
                    corrected_distr=corrected_distr,
                    shield_info=self.shield.report_info(),
                    frame=frame,
                )

                # Advance loop
                last_action = action
                state = next_state
                terminated = terminated or self.shield.model_info.bad_state in self.shield.model_info.model.states[self.shield.model_info.map_states(state)].labels
                if max_steps is not None:
                    terminated = terminated or step_count >= max_steps
                step_count += 1


            # Terminal label
            end_text = self.shield.model_info.model.states[self.shield.model_info.map_states(state)].labels[0]
            drawer.on_episode_end(state, f"done at: {end_text}")
            if end_text == self.shield.model_info.bad_state:
                number_bad += 1
        return number_bad

# ------------- Manim Scene using the runner -------------

class ShieldingScene(Scene):
    def construct(self):
        # Initialize model and shield
        model_info = blackjack()
        # shield = PessimisticShield(model_info, 0.5)
        # shield = IdentityShield(model_info)
        shield = SelfConstructingShield(model_info, 0.05) 

        # Build drawer and runner
        drawer = ManimDrawer(self, theme_name="Monokai Pro Light")
        runner = EnvRunner(model_info.env, shield, actions=[0, 1])

        # Run N episodes with drawing enabled
        runner.run(episodes=10, drawer=drawer, render=True, max_steps=None)

        shield.back_propagate_values(shield.current_node)
        print(shield.initial_node)

# if __name__ == "__main__":
#     model_info = blackjack()
#     shield = PessimisticShield(model_info, 0.5)
#     runner = EnvRunner(model_info.env, shield, actions=[0, 1])
#     # Run headless logic (no Manim, no rendering)
#     number_bad = runner.run(episodes=10, drawer=NullDrawer(), render=False, max_steps=None)
#     print(number_bad, "/", 10)

