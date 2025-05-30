#!/usr/bin/env python3
"""danielsinkin97@gmail.com"""

import matplotlib.pyplot as plt
import numpy as np

from computer_vision.src.constants import FolderPath


def example_fourier(show: bool = True) -> None:
    """Plots a function and its corresponding fourier transform"""
    n = 512
    x = np.linspace(0, 2 * np.pi, n, endpoint=False)

    f = 3 * np.exp(1j * x) + np.exp(3j * x) + 0.5 * np.exp(5j * x) + 2

    f_fourier = np.fft.fft(f)
    f_fourier_normalised = f_fourier / n

    f_fourier_shifted = np.fft.fftshift(f_fourier_normalised)

    dx = x[1] - x[0]
    freq = np.fft.fftfreq(n, d=dx)
    omega = freq * 2 * np.pi
    omega = np.fft.fftshift(omega)

    _, axs = plt.subplots(1, 2, figsize=(12, 5))

    axs[0].plot(x, f.real, label="Real part")
    axs[0].plot(x, f.imag, label="Imaginary part", linestyle="--")
    axs[0].set_title("Function f(x)")
    axs[0].set_xlabel("x")
    axs[0].set_ylabel("f(x)")
    axs[0].legend()
    axs[0].grid(True)

    axs[1].stem(omega, np.abs(f_fourier_shifted), basefmt=" ")
    axs[1].set_title("Fourier Transform |F(ω)|")
    axs[1].set_xlabel("Angular frequency ω")
    axs[1].set_ylabel("Magnitude")
    axs[1].set_xlim(-10, 10)
    axs[1].grid(True)

    plt.tight_layout()
    plt.savefig(FolderPath.Images.joinpath("example_fourier_basic.png"), dpi=300)
    if show:
        plt.show()
    else:
        plt.close()


if __name__ == "__main__":
    example_fourier(show=True)
