"""danielsinkin97@gmail.com"""

from typing import Optional

import matplotlib.pyplot as plt


def show_grayscale(
    array, title: Optional[str] = None, filename: Optional[str] = None
) -> None:
    plt.figure(figsize=(6, 6))
    plt.imshow(array, cmap="gray", interpolation="nearest")
    plt.axis("off")
    if title is not None:
        plt.title(f"{title} {array.shape}")
    if filename is not None:
        if "." not in filename:
            filename = filename + ".png"
        plt.savefig(f"images/{filename}")

    plt.show()
