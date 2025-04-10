"""danielsinkin97@gmail.com"""

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

from src.filter import FilterType, apply_filter, get_filter
from src.util_image import plot_grayscale
from util.image_to_np import load_image_as_array
from util.rbg_to_grayscale import rgb_to_grayscale


def draw_centered_rect(x_center, y_center, width, height, **kwargs) -> None:
    """Rects are top left anchored in plt, this moves the anchor to the center before plotting."""
    x = x_center - width / 2
    y = y_center - height / 2
    rect = patches.Rectangle((x, y), width, height, **kwargs)
    plt.gca().add_patch(rect)


def example_autocorrelation() -> None:
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
    x2p, y2p = x2 + dx, y2 + dy

    plot_grayscale(image, figsize=(12, 12))

    def plot_delta_boxes(x_orig, y_orig, dx, dy) -> None:
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

    plot_delta_boxes(x, y, dx, dy)
    plot_delta_boxes(x2, y2, dx2, dy2)

    plt.savefig("images/autocorrelation_lion.png", dpi=300)
    plt.show()

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    fig.suptitle("Comparison of Image Patches", fontsize=16)

    box1_orig = image[y - d : y + d, x - d : x + d]
    axes[0, 0].imshow(box1_orig, cmap="gray")
    axes[0, 0].set_title(f"box1_orig: (y={y}, x={x})")
    axes[0, 0].axis("off")

    box1_offset = image[yp - d : yp + d, xp - d : xp + d]
    axes[0, 1].imshow(box1_offset, cmap="gray")
    axes[0, 1].set_title(f"box1_offset: (y={yp}, x={xp})")
    axes[0, 1].axis("off")

    box2_orig = image[y2 - d : y2 + d, x2 - d : x2 + d]
    axes[1, 0].imshow(box2_orig, cmap="gray")
    axes[1, 0].set_title(f"box2_orig: (y={y2}, x={x2})")
    axes[1, 0].axis("off")

    box2_offset = image[y2p - d : y2p + d, x2p - d : x2p + d]
    axes[1, 1].imshow(box2_offset, cmap="gray")
    axes[1, 1].set_title(f"box2_offset: (y={y2p}, x={x2p})")
    axes[1, 1].axis("off")

    plt.savefig("images/autocorrelation_lion_boxes.png", dpi=300)
    plt.show()

    def plot_ac_error_surface(y_orig, x_orig, max_offset_y=5, max_offset_x=5) -> None:
        dy_range = range(-max_offset_y, max_offset_y + 1)
        dx_range = range(-max_offset_x, max_offset_x + 1)

        Z = np.empty((len(dy_range), len(dx_range)), dtype=np.float32)
        X, Y = np.meshgrid(dx_range, dy_range)

        box_orig = image[y_orig - d : y_orig + d, x - d : x + d]
        normalisation_factor = 1.0 / ((2.0 * max_offset_x) * (2.0 * max_offset_y))

        for i, dy in enumerate(dy_range):
            for j, dx in enumerate(dx_range):
                yp = y_orig + dy
                xp = x_orig + dx
                box_tmp = image[yp - d : yp + d, xp - d : xp + d]
                error = normalisation_factor + np.linalg.norm(box_orig - box_tmp)
                Z[i, j] = error

        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(X, Y, Z, cmap="magma", edgecolor="none", antialiased=True)

        ax.set_xlabel("dx")
        ax.set_ylabel("dy")
        ax.set_title(
            f"AC Error Surface\n(y={y_orig}, x={x_orig}, max_offset_y={max_offset_y}, max_offset_x={max_offset_x})"
        )

        plt.show()

    plot_ac_error_surface(y, x, 10, 10)
    plot_ac_error_surface(y2, x2, 10, 10)


if __name__ == "__main__":
    example_autocorrelation()
