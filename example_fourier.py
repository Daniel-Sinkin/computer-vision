import matplotlib.pyplot as plt
import numpy as np

from src.constants import FolderPath


def main() -> None:
    # Number of sample points
    N = 512
    x = np.linspace(0, 2 * np.pi, N, endpoint=False)

    # Define the function f(x) = 3e^(ix) + e^(3ix) + 0.5e^(5ix) + 2
    f = 3 * np.exp(1j * x) + np.exp(3j * x) + 0.5 * np.exp(5j * x) + 2

    # Compute the Fourier transform using FFT
    F = np.fft.fft(f)
    # Normalize the FFT so that the peaks correspond to the coefficients
    F_normalized = F / N

    # Shift the zero-frequency component to the center for easier interpretation
    F_shifted = np.fft.fftshift(F_normalized)

    # Create the frequency array. The sampling spacing is:
    dx = x[1] - x[0]
    freq = np.fft.fftfreq(N, d=dx)
    # Convert frequency to angular frequency omega = 2pi * f
    omega = freq * 2 * np.pi
    omega = np.fft.fftshift(omega)

    # Create a 1x2 subplot
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # Plot f(x): both real and imaginary parts for clarity.
    axs[0].plot(x, f.real, label="Real part")
    axs[0].plot(x, f.imag, label="Imaginary part", linestyle="--")
    axs[0].set_title("Function f(x)")
    axs[0].set_xlabel("x")
    axs[0].set_ylabel("f(x)")
    axs[0].legend()
    axs[0].grid(True)

    # Plot the Fourier transform: use a stem plot for the spikes
    axs[1].stem(omega, np.abs(F_shifted), basefmt=" ")
    axs[1].set_title("Fourier Transform |F(ω)|")
    axs[1].set_xlabel("Angular frequency ω")
    axs[1].set_ylabel("Magnitude")
    axs[1].set_xlim(-10, 10)
    axs[1].grid(True)

    plt.tight_layout()
    plt.savefig(FolderPath.Images.joinpath("example_fourier_basic.png"), dpi=300)
    plt.show()


if __name__ == "__main__":
    main()
