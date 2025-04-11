"""danielsinkin97@gmail.com"""

import matplotlib.pyplot as plt
import numpy as np

from src.constants import FolderPath


def example_fourier_of_kernels() -> None:
    """Plots the fourier transform for a selection of separable kernels."""
    xs = np.linspace(0, 0.5, 100)

    kernels = [
        "1/3 [1, 1, 1]",
        "1/5 [1, 1, 1, 1, 1]",
        "1/4 [1, 2, 1]",
        "1/16 [1, 4, 6, 4, 1]",
        "1/2 [-1, 0, 1]",
        "1/2 [-1, 2, -1]",
    ]

    names = ["Box-3", "Box-5", "Linear", "Binomial", "Sobel", "Corner"]

    def func1(x):
        return 1 / 3 * (1 + 2 * np.cos(2 * np.pi * x))

    def func2(x):
        return 1 / 5 * (1 + 2 * np.cos(2 * np.pi * x) + 2 * np.cos(4 * np.pi * x))

    def func3(x):
        return 1 / 2 * (1 + np.cos(2 * np.pi * x))

    def func4(x):
        return 1 / 4 * (1 + np.cos(2 * np.pi * x)) ** 2

    def func5(x):
        return np.sin(2 * np.pi * x)

    def func6(x):
        return 1 / 2 * (1 - np.cos(2 * np.pi * x))

    funcs = [func1, func2, func3, func4, func5, func6]

    _, axs = plt.subplots(2, 3, figsize=(12, 6))
    for i, ax in enumerate(axs.flat):
        ax.axhline(y=0, color="black", lw=1, zorder=0)
        ax.plot(xs, funcs[i](xs), c="blue")
        ax.set_title(f"{names[i]}\n{kernels[i]}")
        ax.set_ylim(-0.4, 1.1)

    plt.suptitle("Fourier Transforms of separable kernels")

    plt.tight_layout()
    plt.savefig(FolderPath.Images.joinpath("example_fourier_of_kernels.png"), dpi=300)
    plt.show()


if __name__ == "__main__":
    example_fourier_of_kernels()
