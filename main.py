from manim import *
import random
import numpy as np
from manim_themes.manim_theme import apply_theme
from shielding.shield import sample_distribution, StandardShield
from shielding.models import blackjack
import networkx as nx

def values_from_distribution(distr, actions=(0, 1)):
    probs = {a: 0.0 for a in actions}
    for p, a in distr:
        probs[a] = p
    return [probs[a] for a in actions]

class Shielding(Scene):
    def setup(self):
        theme = "Monokai Pro Light"
        apply_theme(manim_scene=self, theme_name=theme, light_theme=True)

    def construct(self):
        # Initialize model and shield
        model_info = blackjack()
        shield = StandardShield(model_info)
        iterations = 10  # number of episodes
        last_action = None

        # Obtain an initial RGB frame (env must support render_mode='rgb_array')
        model_info.env.reset()
        frame = model_info.env.render()
        assert frame is not None

        # Left: environment image
        img = ImageMobject(frame)  # ImageMobject accepts numpy arrays
        img.height = 4

        # Right: two bar charts (Original vs Corrected)
        chart_orig = BarChart(
            values=[0.5, 0.5],
            bar_names=["0", "1"],
            y_range=[0, 1, 0.2],
            y_length=3,
            x_length=3,
            x_axis_config={"font_size": 28},
            y_axis_config={"font_size": 24},
        )
        chart_corr = BarChart(
            values=[0.5, 0.5],
            bar_names=["0", "1"],
            y_range=[0, 1, 0.2],
            y_length=3,
            x_length=3,
            x_axis_config={"font_size": 28},
            y_axis_config={"font_size": 24},
        )

        right_col = Group(chart_orig, chart_corr).arrange(RIGHT, buff=0.5)
        upper_row = Group(img, right_col).arrange(RIGHT, buff=1.0)
        layout = Group(upper_row).arrange(DOWN)

        status = Text("new game (0)", font_size=30).to_corner(DOWN + LEFT)
        self.add(layout, status)

        # Run episodes
        for i in range(iterations):
            reset_info = model_info.env.reset()
            state = reset_info[0]
            state_str = str(state)


            terminated = False

            # Update status
            new_status = Text(f"new game ({i})", font_size=30).to_corner(DOWN + LEFT)
            self.play(Transform(status, new_status), run_time=0.3)

            last_action = None
            while not terminated:
                # Random 2-action distribution and shield correction
                p = random.random()
                distr = [(p, 0), (1 - p, 1)]
                corrected_distr = shield.correct(last_action, state, distr)
                v_orig = values_from_distribution(distr)
                v_corr = values_from_distribution(corrected_distr)

                # Update bar charts
                self.play(
                    chart_orig.animate.change_bar_values(v_orig),
                    chart_corr.animate.change_bar_values(v_corr),
                    run_time=0.3,
                )

                # Step env with shielded action
                action = sample_distribution(corrected_distr)
                last_action = action
                last_state = state
                step_out = model_info.env.step(action)
                # Gymnasium API: (obs, reward, terminated, truncated, info)
                state, _, terminated, _, _ = step_out

                # Get new RGB frame and update the image via ReplacementTransform (safe for images)
                new_frame = model_info.env.render()
                if new_frame is not None:
                    new_img = ImageMobject(new_frame)
                    new_img.height = img.height
                    new_img.move_to(img)
                    self.play(ReplacementTransform(img, new_img), run_time=0.25)
                    img = new_img  # keep reference to latest image

                # Terminal text
                end_text = "bust!" if isinstance(state, (list, tuple, np.ndarray)) and state[0] > 21 else "stick!"
                end_status = Text(end_text, font_size=30).to_corner(DOWN + LEFT)
                self.play(Transform(status, end_status), run_time=0.4)
                self.wait(0.3)
