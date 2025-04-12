#!/usr/bin/env python3
"""danielsinkin97@gmail.com"""

import matplotlib.pyplot as plt
import numpy as np

from computer_vision.src.constants import FolderPath
from computer_vision.src.filter import FilterType, apply_filter, get_filter
from computer_vision.util.images import (
    load_image_as_array,
    plot_grayscale,
    rgb_to_grayscale,
)

image = rgb_to_grayscale(
    load_image_as_array(FolderPath.Data.joinpath("lion_downscaled.jpg"))
)


def finish_plot(show: bool = True) -> None:
    """Either shows the plot or just closes it, in either case the canvas is clear after this."""
    if show:
        plt.show()
    else:
        plt.close()


def get_example_image() -> np.ndarray:
    """Loads simple example image hardcoded based on the book."""
    # fmt: off
    return np.array(
        [
            [  45,  60,  98, 127, 132, 133, 137, 133],
            [  46,  65,  98, 123, 126, 128, 131, 133],
            [  47,  65,  96, 115, 119, 123, 135, 137],
            [  47,  63,  91, 107, 113, 122, 138, 134],
            [  50,  59,  80,  97, 110, 123, 133, 134],
            [  49,  53,  68,  83,  97, 113, 128, 133],
            [  50,  50,  58,  70,  84, 102, 116, 126],
            [  50,  50,  52,  58,  69,  86, 101, 120],
        ],
        dtype=np.float32,
    )
    # fmt: on


def get_my_filter() -> np.ndarray:
    """Returns example hardcoded smoothing filter from the book."""
    return np.array([[0.1, 0.1, 0.1], [0.1, 0.2, 0.1], [0.1, 0.1, 0.1]])


def filter_and_plot(filter_name: str, show=True) -> None:
    """Applies filter, plots and invokes plt.show()"""
    plot_grayscale(
        apply_filter(image, get_filter(filter_name)),
        title=filter_name,
        filename=filter_name,
    )
    finish_plot(show=show)


def example_filters(show: bool = True) -> None:
    """Applies different filters to images."""
    image_conv = apply_filter(image=get_example_image(), filter_=get_my_filter())

    plot_grayscale(get_example_image())
    finish_plot(show=show)
    plot_grayscale(image_conv)
    finish_plot(show=show)

    plot_grayscale(image, title="original")
    finish_plot(show=show)
    filter_and_plot(filter_name=FilterType.SOBEL_X, show=show)
    filter_and_plot(filter_name=FilterType.SOBEL_Y, show=show)
    filter_and_plot(filter_name=FilterType.LAPLACIAN, show=show)
    filter_and_plot(filter_name=FilterType.CORNER, show=show)
    filter_and_plot(filter_name=FilterType.BILINEAR, show=show)

    smoothed_image = apply_filter(image, get_filter(FilterType.GAUSS_5X5))
    plot_grayscale(
        apply_filter(smoothed_image, get_filter(FilterType.LAPLACIAN)),
        title="Smoothed (5x5) then Laplace",
    )
    finish_plot(show=show)
    plot_grayscale(
        apply_filter(smoothed_image, get_filter(FilterType.SOBEL_X)),
        title="Smoothed (5x5) then Sobel_x",
    )
    finish_plot(show=show)
    plot_grayscale(
        apply_filter(smoothed_image, get_filter(FilterType.SOBEL_Y)),
        title="Smoothed (5x5) then Sobel_y",
    )
    finish_plot(show=show)
    plot_grayscale(
        apply_filter(smoothed_image, get_filter(FilterType.CORNER)),
        title="Smoothed (5x5) then Corner",
    )
    finish_plot(show=show)

    for gaussian in [
        FilterType.GAUSS_3X3,
        FilterType.GAUSS_5X5,
        FilterType.GAUSS_7X7,
        FilterType.GAUSS_15X15,
    ]:
        filter_and_plot(filter_name=gaussian, show=show)


if __name__ == "__main__":
    example_filters(show=True)
