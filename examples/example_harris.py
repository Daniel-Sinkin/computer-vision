import matplotlib.pyplot as plt

from computer_vision.util.images import (
    plot_grayscale,
    rgb_to_grayscale,
    load_image_as_array,
)
from computer_vision.src.filter import get_filter, apply_filter

image = rgb_to_grayscale(load_image_as_array("data/hummingbird.png"))

ix = apply_filter(image, get_filter("sobel_x"), pad_same_size=True)
iy = apply_filter(image, get_filter("sobel_y"), pad_same_size=True)

plot_grayscale(ix, title="I_x")
plot_grayscale(iy, title="I_y")

plot_grayscale(ix * iy, title="I_xI_y")

# Compute products of derivatives
ix2 = ix * ix
iy2 = iy * iy
ixy = ix * iy

# Apply Gaussian smoothing (structure tensor weights)
sxx = apply_filter(ix2, get_filter("gaussian_5x5"), pad_same_size=True)
syy = apply_filter(iy2, get_filter("gaussian_5x5"), pad_same_size=True)
sxy = apply_filter(ixy, get_filter("gaussian_5x5"), pad_same_size=True)

# Optional: visualize the second-moment components
plot_grayscale(sxx, title="S_xx")
plot_grayscale(syy, title="S_yy")
plot_grayscale(sxy, title="S_xy")

# Harris response
alpha = 0.05

det_M = sxx * syy - sxy * sxy
trace_M = sxx + syy

R = det_M - alpha * (trace_M**2)

plot_grayscale(R, title="Harris Response")

import numpy as np

p = 0.15
t = 0.3
plot_grayscale(
    p * image + (1.0 - p) * np.where(np.abs(R) > np.max(R) * t, 255.0, 0.0),
    title=f"Harris Response Thresholded\n{t=}, {p=} interpolate",
)
