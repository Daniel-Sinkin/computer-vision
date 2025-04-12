#!/usr/bin/env python3
"""danielsinkin97@gmail.com"""

from examples.example_autocorrelation import example_autocorrelation
from examples.example_autocorrelation_detection import example_autocorrelation_detection
from examples.example_filters import example_filters
from examples.example_fourier import example_fourier
from examples.example_fourier_of_kernels import example_fourier_of_kernels
from examples.example_moving_average_filter import example_moving_average_filter
from examples.example_padding import example_padding

examples = {
    "example_padding": example_padding,
    "example_filters": example_filters,
    "example_autocorrelation": example_autocorrelation,
    "example_moving_average_filter": example_moving_average_filter,
    "example_fourier": example_fourier,
    "example_autocorrelation_detection": example_autocorrelation_detection,
    "example_fourier_of_kernels": example_fourier_of_kernels,
}


def run_examples() -> None:
    for example_name, example_fn in examples.items():
        print(f"Running example '{example_name}':")
        example_fn(show=False)


if __name__ == "__main__":
    run_examples()
