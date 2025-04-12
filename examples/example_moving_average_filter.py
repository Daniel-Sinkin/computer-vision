#!/usr/bin/env python3
"""danielsinkin97@gmail.com"""

from typing import cast

import matplotlib.pyplot as plt
import numpy as np

from computer_vision.src.constants import FolderPath

FILTER = 1  # How far in each direction our filter goes, filter size is 2 * FILTER + 1


def generate_image() -> np.ndarray:
    return np.array(
        [
            [0.0] * 10,
            [0.0] * 10,
            [0.0] * 3 + [90.0] * 5 + [0.0] * 2,
            [0.0] * 3 + [90.0] * 5 + [0.0] * 2,
            [0.0] * 3 + [90.0] * 5 + [0.0] * 2,
            [0.0] * 3 + [90.0] + [0.0] + [90.0] * 3 + [0.0] * 2,
            [0.0] * 3 + [90.0] * 5 + [0.0] * 2,
            [0.0] * 10,
            [0.0] * 2 + [90.0] + [0.0] * 7,
            [0.0] * 10,
        ]
    )


def moving_average_filter(image: np.ndarray, filter_size: int) -> np.ndarray:
    filtered_image = np.zeros_like(image)
    image_x, image_y = image.shape

    for x in range(1, image_x - 1):
        for y in range(1, image_y - 1):
            filtered_image[x, y] = image[
                x - filter_size : x + filter_size + 1,
                y - filter_size : y + filter_size + 1,
            ].mean()
    return filtered_image


def map_range(value: float) -> float:
    """[0.0, 1.0] -> [0.2, 1.0]"""
    return 0.3 + 0.7 * (value / 100.0)


def plot(ax: np.ndarray, image: np.ndarray, title: str, annotate: bool = False) -> None:
    norm_img = map_range(image)
    ax.imshow(norm_img, cmap="gray", vmin=0.2, vmax=1.0)
    ax.set_title(title)
    ax.axis("off")
    if annotate:
        for (i, j), val in np.ndenumerate(image):
            ax.text(
                j, i, f"{val:.0f}", ha="center", va="center", color="black", fontsize=8
            )


def example_moving_average_filter() -> None:
    image = generate_image()
    filtered_image = moving_average_filter(image=image, filter_size=FILTER)

    for annotate in [True, False]:
        _, axs = plt.subplots(1, 2, figsize=(12, 6))
        axs = cast(np.ndarray, axs)

        plt.suptitle("Moving Average in 2D")

        plot(axs[0], image, "Original Image", annotate=annotate)
        plot(axs[1], filtered_image, "Filtered Image", annotate=annotate)

        plt.tight_layout()
        if annotate:
            filename = "moving_average_filter_annotated.png"
        else:
            filename = "moving_average_filter.png"

        plt.savefig(FolderPath.Images.joinpath(filename))

        plt.show()


if __name__ == "__main__":
    example_moving_average_filter()
