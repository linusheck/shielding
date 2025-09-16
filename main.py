from manim import *
import random
import numpy as np

# Optional theming, keep if available
try:
    from manim_themes.manim_theme import apply_theme
except Exception:
    def apply_theme(*args, **kwargs):
        pass

# Shielding/model imports
from shielding.shield import sample_distribution, PessimisticShield, IdentityShield
from shielding.models import blackjack

# ------------- Utilities (unchanged) -------------
def values_from_distribution(distr, actions=(0, 1)):
    probs = {a: 0.0 for a in actions}
    for p, a in distr:
        probs[a] = p
    return [probs[a] for a in actions]

def format_dict(shield_info):
    return "\n".join(
        [f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" for k, v in shield_info.items()]
    )

# ------------- Drawer protocol and implementations -------------

class DrawerProtocol:
    """Event-driven drawer interface. All methods are optional to implement."""
    def on_env_ready(self, frame):  # called once after initial render
        pass
    def on_episode_start(self, episode_idx: int, state):
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

    def _build_layout(self, initial_frame: np.ndarray):
        # Left image
        self.img = ImageMobject(initial_frame)
        self.img.height = 4

        # Right charts
        self.chart_orig = BarChart(
            values=[0.5, 0.5],
            bar_names=["0", "1"],
            y_range=[0, 1, 0.2],
            y_length=3,
            x_length=3,
            x_axis_config={"font_size": 28},
            y_axis_config={"font_size": 24},
        )
        self.chart_corr = BarChart(
            values=[0.5, 0.5],
            bar_names=["0", "1"],
            y_range=[0, 1, 0.2],
            y_length=3,
            x_length=3,
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

    def on_env_ready(self, frame):
        # Build the initial layout once we have an RGB frame from env.render()
        self._build_layout(frame)

    def on_episode_start(self, episode_idx: int, state):
        self.visited_states = [str(state)]
        new_status = Text(f"game {episode_idx}", font_size=30).to_corner(DOWN + LEFT)
        self.scene.play(Transform(self.status, new_status), run_time=0.4)

        # Refresh visited states text
        new_v = Text("\n".join(self.visited_states), font_size=20, font="IBM Plex Mono").to_edge(DOWN)
        self.scene.play(ReplacementTransform(self.visited_text, new_v), run_time=0.2)
        self.visited_text = new_v

    def on_step(self, state, action, distr, corrected_distr, shield_info: dict, frame):
        # Update image if a new frame is provided
        if frame is not None:
            new_img = ImageMobject(frame)
            new_img.height = self.img.height
            new_img.move_to(self.img)
            self.scene.play(ReplacementTransform(self.img, new_img), run_time=0.2)
            self.img = new_img

        # Update charts
        v_orig = values_from_distribution(distr)
        v_corr = values_from_distribution(corrected_distr)
        self.scene.play(
            self.chart_orig.animate.change_bar_values(v_orig),
            self.chart_corr.animate.change_bar_values(v_corr),
            run_time=0.3,
        )

        # Update shield info
        new_info = Text(format_dict(shield_info), font_size=20, font="IBM Plex Mono").to_corner(DOWN + RIGHT)
        self.scene.play(ReplacementTransform(self.shield_info_text, new_info), run_time=0.2)
        self.shield_info_text = new_info

        # Update visited states text
        self.visited_states.append(f"{action} -> {state}")
        new_v = Text("\n".join(self.visited_states), font_size=20, font="IBM Plex Mono").to_edge(DOWN)
        self.scene.play(ReplacementTransform(self.visited_text, new_v), run_time=0.2)
        self.visited_text = new_v

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

    def run(self, episodes: int, drawer: DrawerProtocol, render: bool = True):
        self.env.reset()
        frame = self._maybe_render(render)
        if frame is not None:
            drawer.on_env_ready(frame)

        for ep in range(episodes):
            # Gymnasium reset returns (obs, info)
            reset_out = self.env.reset()
            if isinstance(reset_out, tuple) and len(reset_out) == 2:
                state, _ = reset_out
            else:
                # Fallback for legacy API
                state = reset_out

            drawer.on_episode_start(ep, state)

            last_action = None
            terminated = False
            truncated = False

            while not (terminated or truncated):
                # Random 2-action distribution and shield correction
                p = self.rng.random()
                distr = [(p, self.actions[0]), (1 - p, self.actions[1])]
                corrected_distr = self.shield.correct(last_action, state, distr)
                # Choose action and step
                print("Corrected distr", corrected_distr)
                action = sample_distribution(corrected_distr)
                print("Action", action)
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

            # Terminal label
            end_text = self.shield.model_info.model.states[self.shield.model_info.map_states(state)].labels[0]
            drawer.on_episode_end(state, f"done at: {end_text}")

# ------------- Manim Scene using the runner -------------

class ShieldingScene(Scene):
    def construct(self):
        # Initialize model and shield
        model_info = blackjack()
        shield = PessimisticShield(model_info, 0.5)
        # shield = IdentityShield(model_info)

        # Build drawer and runner
        drawer = ManimDrawer(self, theme_name="Monokai Pro Light")
        runner = EnvRunner(model_info.env, shield, actions=(0, 1))

        # Run N episodes with drawing enabled
        runner.run(episodes=10, drawer=drawer, render=True)

# ------------- Optional: headless entry point -------------
if __name__ == "__main__":
    model_info = blackjack()
    shield = IdentityShield(model_info)
    runner = EnvRunner(model_info.env, shield, actions=(0, 1))
    # Run headless logic (no Manim, no rendering)
    runner.run(episodes=5, drawer=NullDrawer(), render=False)
