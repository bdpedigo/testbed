# %%

from io import BytesIO
from pathlib import Path

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def accelerate(
    input_path: str | Path,
    letter: str | None = None,
    letter_size: float = 0.5,
    stroke_width: float = 0.04,
    font_weight: str = "normal",
    letter_offset: tuple[float, float] = (0.0, 0.0),
    background_color: str | tuple | None = None,
) -> Image.Image:
    """Apply a circular mask to a square image, optionally adding a letter mask.

    Parameters
    ----------
    input_path :
        Path to the input image (should be square).
    output_path :
        Path to save the result. Defaults to <input_stem>_circular.png next to
        the input file. Output is always PNG to support transparency.
    letter :
        Single character whose silhouette is unioned with the circular mask,
        so the image shows through the letter shape as well.
    letter_size :
        Letter height as a fraction of the image's smaller dimension (default 0.5).
    stroke_width :
        Stroke width as a fraction of the font size, used to thicken the letter
        mask via path effects (default 0.08).
    letter_offset :
        (dx, dy) translation of the letter relative to the image centre, expressed
        as fractions of the image's smaller dimension. Positive dx moves right,
        positive dy moves up (default (0.0, 0.0)).
    background_color :
        Colour to fill behind the masked image. Accepts any value PIL understands:
        a colour name (``"red"``), hex string (``"#ff0000"``), or an RGB/RGBA tuple.
        ``None`` (default) keeps the background transparent.

    Returns
    -------
    PIL.Image.Image
        The cropped (and optionally annotated) RGBA image.
    """
    input_path = Path(input_path)
    img = Image.open(input_path).convert("RGBA")
    w, h = img.size

    # Use the smaller dimension so non-square images still get a valid circle
    diameter = min(w, h)
    cx, cy = w / 2, h / 2
    radius = diameter / 2

    # Build a circular alpha mask
    ys, xs = np.ogrid[:h, :w]
    dist_sq = (xs - cx) ** 2 + (ys - cy) ** 2
    circle_mask = np.where(dist_sq <= radius**2, 255, 0).astype(np.uint8)

    # Build a letter mask by rendering white text on a transparent figure,
    # then extracting the alpha channel as the mask shape.
    if letter is not None:
        dpi = 100
        fig, ax = plt.subplots(figsize=(w / dpi, h / dpi), dpi=dpi)
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        fig.patch.set_alpha(0)
        ax.set_xlim(0, w)
        ax.set_ylim(0, h)
        ax.axis("off")

        fontsize = diameter * letter_size * 72 / dpi
        sw = fontsize * stroke_width

        dx = letter_offset[0] * diameter
        dy = letter_offset[1] * diameter

        txt = ax.text(
            cx + dx,
            cy + dy,
            letter,
            ha="center",
            va="center",
            fontsize=fontsize,
            fontweight=font_weight,
        )
        # Use withStroke to thicken the letter silhouette, then Normal to fill it
        txt.set_path_effects(
            [
                pe.withStroke(linewidth=sw, foreground="white"),
                pe.Normal(),
            ]
        )

        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, transparent=True)
        plt.close(fig)
        buf.seek(0)

        letter_img = Image.open(buf).convert("RGBA").resize((w, h), Image.LANCZOS)
        letter_mask = np.array(letter_img)[:, :, 3]  # alpha channel = letter shape

        # Subtract: show the circle, but punch the letter shape out as a cutout
        combined_mask = np.clip(
            circle_mask.astype(np.int32) - letter_mask.astype(np.int32), 0, 255
        ).astype(np.uint8)
    else:
        combined_mask = circle_mask

    # Apply the combined mask to the original image's alpha channel
    r, g, b, a = img.split()
    new_alpha = Image.fromarray(np.minimum(np.array(a), combined_mask))
    result = Image.merge("RGBA", (r, g, b, new_alpha))

    if background_color is not None:
        background = Image.new("RGBA", (w, h), background_color)
        background.alpha_composite(result)
        result = background

    return result


img = accelerate(
    "/Users/ben.pedigo/code/testbed/corgi.jpg",
    letter="b/",
    letter_size=0.85,
    letter_offset=(-0.05, -0.15),
    background_color="black",
)
img.save("/Users/ben.pedigo/code/testbed/corgi_circular.png")
img
