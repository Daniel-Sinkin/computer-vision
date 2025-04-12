#!/usr/bin/env python3
"""danielsinkin97@gmail.com"""

import matplotlib.pyplot as plt

from computer_vision.src.padding import PaddingType, apply_padding
from computer_vision.src.util_image import plot_grayscale
from computer_vision.util.images import load_image_as_array, rgb_to_grayscale

image = rgb_to_grayscale(
    load_image_as_array(
        "/Users/danielsinkin/GitHub_private/computer-vision/data/lion_downscaled.jpg"
    )
)


def apply_padding_and_plot(padding_size: int, padding_type: PaddingType):
    """Helper function that pads and plots in one go."""
    filename = f"padding_{padding_type}"
    plot_grayscale(
        apply_padding(image=image, pad=padding_size, padding_type=padding_type),
        title=filename,
        filename=filename,
    )
    plt.show()


def example_padding() -> None:
    """Displays and saves the different types of padding."""
    apply_padding_and_plot(5, "zero")
    apply_padding_and_plot(10, "wrap")
    apply_padding_and_plot(10, "clamp")
    apply_padding_and_plot(10, "mirror")


if __name__ == "__main__":
    example_padding()
