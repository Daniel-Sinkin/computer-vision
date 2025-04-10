"""danielsinkin97@gmail.com"""

import matplotlib.pyplot as plt
import numpy as np

from src.filter import FilterType, apply_filter, get_filter
from src.util_image import plot_grayscale
from util.image_to_np import load_image_as_array
from util.rbg_to_grayscale import rgb_to_grayscale

image = rgb_to_grayscale(
    load_image_as_array(
        "/Users/danielsinkin/GitHub_private/computer-vision/data/lion_downscaled.jpg"
    )
)


def get_my_image() -> np.ndarray:
    return np.array(
        [
            [45, 60, 98, 127, 132, 133, 137, 133],
            [46, 65, 98, 123, 126, 128, 131, 133],
            [47, 65, 96, 115, 119, 123, 135, 137],
            [47, 63, 91, 107, 113, 122, 138, 134],
            [50, 59, 80, 97, 110, 123, 133, 134],
            [49, 53, 68, 83, 97, 113, 128, 133],
            [50, 50, 58, 70, 84, 102, 116, 126],
            [50, 50, 52, 58, 69, 86, 101, 120],
        ],
        dtype=np.float32,
    )


def get_my_filter() -> np.ndarray:
    return np.array([[0.1, 0.1, 0.1], [0.1, 0.2, 0.1], [0.1, 0.1, 0.1]])


def filter_and_plot(filter_name):
    plot_grayscale(
        apply_filter(image, get_filter(filter_name)),
        title=filter_name,
        filename=filter_name,
    )
    plt.show()


def main() -> None:
    image_conv = apply_filter(image=get_my_image(), filter_=get_my_filter())

    plot_grayscale(get_my_image())
    plt.show()
    plot_grayscale(image_conv)
    plt.show()

    plot_grayscale(image, title="original")
    plt.show()
    filter_and_plot(filter_name=FilterType.SOBEL_X)
    filter_and_plot(filter_name=FilterType.SOBEL_Y)

    filter_and_plot(filter_name=FilterType.LAPLACIAN)

    filter_and_plot(filter_name=FilterType.CORNER)

    filter_and_plot(filter_name=FilterType.BILINEAR)

    smoothed_image = apply_filter(image, get_filter(FilterType.GAUSS_5X5))
    plot_grayscale(
        apply_filter(smoothed_image, get_filter(FilterType.LAPLACIAN)),
        title="Smoothed (5x5) then Laplace",
    )
    plt.show()
    plot_grayscale(
        apply_filter(smoothed_image, get_filter(FilterType.SOBEL_X)),
        title="Smoothed (5x5) then Sobel_x",
    )
    plt.show()
    plot_grayscale(
        apply_filter(smoothed_image, get_filter(FilterType.SOBEL_Y)),
        title="Smoothed (5x5) then Sobel_y",
    )
    plt.show()
    plot_grayscale(
        apply_filter(smoothed_image, get_filter(FilterType.CORNER)),
        title="Smoothed (5x5) then Corner",
    )
    plt.show()

    for gaussian in [
        FilterType.GAUSS_3X3,
        FilterType.GAUSS_5X5,
        FilterType.GAUSS_7X7,
        FilterType.GAUSS_15X15,
    ]:
        filter_and_plot(filter_name=gaussian)


if __name__ == "__main__":
    main()
