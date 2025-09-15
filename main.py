from shielding.shield import StandardShield, sample_distribution, IdentityShield
from shielding.models import blackjack
import random
import imageio
import numpy as np
from PIL import Image
import io
from PIL import ImageDraw
from PIL import ImageFont
import matplotlib.pyplot as plt

def render_distribution_bar(distr, corrected_distr):
    print(distr, corrected_distr)
    labels = [str(a) for _, a in distr]
    orig_probs = [p for p, _ in distr]
    corr_probs = [p for p, _ in corrected_distr]
    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(3, 2))
    ax.bar(x - width/2, orig_probs, width, label='Original')
    ax.bar(x + width/2, corr_probs, width, label='Corrected')
    ax.set_ylim(0, 1)
    dpi = 100
    fig.set_size_inches(300 / dpi, 200 / dpi)
    fig.set_dpi(dpi)
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf)
    return np.array(img)

def concat_images_horizontally(img1, img2):
    img1_pil = Image.fromarray(img1)
    img2_pil = Image.fromarray(img2)
    if img2_pil.height != img1_pil.height:
        new_width = int(img2_pil.width * (img1_pil.height / img2_pil.height))
        img2_pil = img2_pil.resize((new_width, img1_pil.height), Image.LANCZOS)
    dst = Image.new('RGB', (img1_pil.width + img2_pil.width, img1_pil.height))
    dst.paste(img1_pil, (0, 0))
    dst.paste(img2_pil, (img1_pil.width, 0))
    return np.array(dst)

def overlay_text_on_frame(frame, text, font_size=50, color=(255, 255, 224)):
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default(font_size)
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    x = int((img.width - text_width) * 0.05)
    y = int((img.height - text_height) * 0.9)
    draw.text((x, y), text, fill=color, font=font)
    return np.array(img)

def make_visual_frame(frame, distr, corrected_distr):
    bar_img = render_distribution_bar(distr, corrected_distr)
    return concat_images_horizontally(np.array(frame), bar_img)

def shield_random_policies(model_info, shield_class, iterations=100):
    shield = shield_class(model_info)
    frames = []
    for i in range(iterations):
        reset_info = model_info.env.reset()
        state = reset_info[0]
        terminated = False
        first_frame = True
        while not terminated:
            frame = model_info.env.render()
            p = random.random()
            distr = [(p, 0), (1-p, 1)]
            corrected_distr = shield.correct(state, distr)
            action = sample_distribution(corrected_distr)
            state, _, terminated, _, _ = model_info.env.step(action)
            combined_frame = make_visual_frame(frame, distr, corrected_distr)
            frames.append(combined_frame)
            if first_frame:
                first_frame = False
                frames[-1] = overlay_text_on_frame(frames[-1], f"new game ({i})")
                frames.append(combined_frame)
        # Overlay text on the last frame depending on the outcome
        text = "bust!" if state[0] > 21 else "stick!"
        frames[-1] = overlay_text_on_frame(frames[-1], text)
    imageio.mimsave("game.gif", [np.array(f) for f in frames], fps=1)

def main():
    # shield_random_policies(blackjack(), IdentityShield)
    shield_random_policies(blackjack(), StandardShield)

if __name__ == "__main__":
    main()
