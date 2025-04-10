"""danielsinkin97@gmail.com"""

from typing import Optional

import matplotlib.pyplot as plt


def plot_grayscale(
    array,
    title: Optional[str] = None,
    filename: Optional[str] = None,
    figsize: tuple[int, int] = (6, 6),
) -> None:
    """Plots the image as a grayscale image."""
    plt.figure(figsize=figsize)
    plt.imshow(array, cmap="gray", interpolation="nearest")
    plt.axis("off")
    if title is not None:
        plt.title(f"{title} {array.shape}")
    if filename is not None:
        if "." not in filename:
            filename = filename + ".png"
        plt.savefig(f"images/{filename}", dpi=300)
