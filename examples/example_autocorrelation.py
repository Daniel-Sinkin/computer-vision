#!/usr/bin/env python3
"""danielsinkin97@gmail.com"""

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches

from computer_vision.src.constants import FolderPath
from computer_vision.src.filter import FilterType, apply_filter, get_filter
from computer_vision.src.util_image import plot_grayscale
from computer_vision.util.images import load_image_as_array, rgb_to_grayscale


def draw_centered_rect(x_center, y_center, width, height, **kwargs) -> None:
    """Rects are top left anchored in plt, this moves the anchor to the center before plotting."""
    x = x_center - width / 2
    y = y_center - height / 2
    rect = patches.Rectangle((x, y), width, height, **kwargs)
    plt.gca().add_patch(rect)


def _compute_error(
    image: np.ndarray,
    box_orig: np.ndarray,
    yp: int,
    xp: int,
    d: int,
    norm_factor: float,
) -> float:
    yp_slice = slice(yp - d, yp + d)
    xp_slice = slice(xp - d, xp + d)
    box_tmp = image[yp_slice, xp_slice]
    return norm_factor + np.linalg.norm(box_orig - box_tmp)


def compute_ac_error_surface(
    image: np.ndarray,
    y_orig: int,
    x_orig: int,
    d: int,
    max_offset_y: int = 5,
    max_offset_x: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the autocorrelation error surface and corresponding meshgrid."""
    dy_range = range(-max_offset_y, max_offset_y + 1)
    dx_range = range(-max_offset_x, max_offset_x + 1)

    z = np.empty((len(dy_range), len(dx_range)), dtype=np.float32)
    xs, ys = np.meshgrid(dx_range, dy_range)

    box_orig = image[y_orig - d : y_orig + d, x_orig - d : x_orig + d]
    norm_factor = 1.0 / ((2.0 * max_offset_x) * (2.0 * max_offset_y))

    for i, dy in enumerate(dy_range):
        yp = y_orig + dy
        for j, dx in enumerate(dx_range):
            xp = x_orig + dx
            z[i, j] = _compute_error(image, box_orig, yp, xp, d, norm_factor)

    return xs, ys, z


def plot_ac_error_surface(
    image: np.ndarray,
    y_orig: int,
    x_orig: int,
    d: int,
    max_offset_y: int = 5,
    max_offset_x: int = 5,
    filename: Optional[str] = None,
    show: bool = True,
) -> None:
    """Plot the autocorrelation error surface."""
    xs, ys, z = compute_ac_error_surface(
        image, y_orig, x_orig, d, max_offset_y, max_offset_x
    )

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(xs, ys, z, cmap="magma", edgecolor="none", antialiased=True)

    ax.set_xlabel("dx")
    ax.set_ylabel("dy")
    ax.set_title(
        "AC Error Surface\n"
        f"(y={y_orig}, x={x_orig}, "
        f"max_offset_y={max_offset_y}, max_offset_x={max_offset_x})"
    )

    if filename is not None:
        if "." not in filename:
            filename += ".png"
        plt.savefig(FolderPath.Images.joinpath(filename), dpi=300)

    if show:
        plt.show()
    else:
        plt.close()


def _plot_delta_boxes(x_orig, y_orig, d, dx, dy) -> None:
    y_offset, x_offset = y_orig + dy, x_orig + dx

    plt.scatter(x_orig, y_orig, c="red", marker="+", s=100, alpha=0.5)
    plt.scatter(x_offset, y_offset, c="blue", marker="+", s=100, alpha=0.5)
    plt.plot([x_orig, x_offset], [y_orig, y_offset], c="black", ls="--", zorder=0)

    draw_centered_rect(
        x_orig,
        y_orig,
        d,
        d,
        edgecolor="red",
        facecolor="none",
        linewidth=2,
        zorder=5,
        alpha=0.5,
    )
    draw_centered_rect(
        x_offset,
        y_offset,
        d,
        d,
        edgecolor="blue",
        facecolor="none",
        linewidth=2,
        zorder=6,
        alpha=0.5,
    )


def _plot_patch_comparison_boxes(
    image: np.ndarray,
    boxes: list[tuple[str, int, int]],
    d: int,
    show: bool,
    filename: Optional[str] = None,
) -> None:
    """
    Plot grayscale boxes from an image in a 2x2 layout for visual comparison.

    Args:
        image: Input grayscale image.
        boxes: A list of tuples like (title, y, x) for each box.
        d: Half-size of the box (box will be 2d x 2d).
        show: Whether to display the figure.
        filename: Optional filename to save the figure.
    """
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    fig.suptitle("Comparison of Image Patches", fontsize=16)

    for ax, (title, y, x) in zip(axes.flat, boxes):
        patch = image[y - d : y + d, x - d : x + d]
        ax.imshow(patch, cmap="gray")
        ax.set_title(f"{title}: (y={y}, x={x})")
        ax.axis("off")

    if filename:
        plt.savefig(FolderPath.Images.joinpath(filename), dpi=300)

    if show:
        plt.show()
    else:
        plt.close()


def example_autocorrelation(show: bool = True) -> None:
    """
    Shows two zoomed in image sections and compute autocorrelation for close neighbors, plotting
    the error surface.
    """
    image = rgb_to_grayscale(
        load_image_as_array(
            "/Users/danielsinkin/GitHub_private/computer-vision/data/lion_downscaled.jpg"
        )
    )
    image = apply_filter(image, filter_=get_filter(FilterType.GAUSS_3X3))

    y, x = 49, 120
    dy, dx = 1, 8

    y2, x2 = 39, 95
    dy2, dx2 = 5, 3

    d = 5

    xp, yp = x + dx, y + dy
    x2p, y2p = x2 + dx2, y2 + dy2

    plot_grayscale(image, figsize=(12, 12))

    _plot_delta_boxes(x_orig=x, y_orig=y, d=d, dx=dx, dy=dy)
    _plot_delta_boxes(x_orig=x2, y_orig=y2, d=d, dx=dx2, dy=dy2)

    plt.savefig(FolderPath.Images.joinpath("autocorrelation_lion.png"), dpi=300)
    if show:
        plt.show()
    else:
        plt.close()

    _plot_patch_comparison_boxes(
        image=image,
        boxes=[
            ("box1_orig", y, x),
            ("box1_offset", yp, xp),
            ("box2_orig", y2, x2),
            ("box2_offset", y2p, x2p),
        ],
        d=d,
        show=show,
        filename="autocorrelation_lion_boxes.png",
    )

    plot_ac_error_surface(
        image=image,
        y_orig=y,
        x_orig=x,
        d=d,
        max_offset_y=10,
        max_offset_x=10,
        filename="autocorrelation_error_surface1",
    )
    plot_ac_error_surface(
        image=image,
        y_orig=y2,
        x_orig=x2,
        d=d,
        max_offset_y=10,
        max_offset_x=10,
        filename="autocorrelation_error_surface2",
    )


if __name__ == "__main__":
    example_autocorrelation(show=True)
