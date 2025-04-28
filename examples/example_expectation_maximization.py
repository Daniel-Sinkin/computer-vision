"""danielsinkin97@gmail.com"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.mixture import GaussianMixture

from computer_vision.src.constants import FolderPath
from computer_vision.util.images import (
    load_image_as_array,
    rgb_to_grayscale,
)

image: np.ndarray = rgb_to_grayscale(load_image_as_array("data/fennec.png"))


# assume `image` is your (h, w) numpy array of gray levels in [0,255] or [0,1]
def plot_scikit() -> None:
    """Uses Sklean"""
    h, w = image.shape
    pixels = image.reshape(-1, 1)

    gm = GaussianMixture(
        n_components=2, covariance_type="full", tol=1e-4, max_iter=100, random_state=42
    )
    gm.fit(pixels)

    labels = gm.predict(pixels)
    segmented = labels.reshape(h, w)

    fg_label = np.argmax(gm.means_.ravel())
    mask = (segmented == fg_label).astype(np.uint8)

    _, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("Original")
    axes[1].imshow(segmented, cmap="gray")
    axes[1].set_title("GMM labels (0/1)")
    axes[2].imshow(mask, cmap="gray")
    axes[2].set_title("Binary mask")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.suptitle("Sklearn GaussianMixture")
    plt.savefig(
        FolderPath.Images.joinpath("example_expectaction_maximization_sklearn.png"),
        dpi=300,
    )
    plt.show()


def plot_custom() -> None:
    """Custom EM Algo implementation."""
    h, w = image.shape
    x = image.flatten()

    _rng = np.random.default_rng(0)

    mu = np.random.choice(x, size=2, replace=False)
    sigma2 = np.array([x.var(), x.var()])

    pi = np.array([0.5, 0.5])

    def sample_normal(x, mu, s2):
        return np.exp(-0.5 * (x - mu) ** 2 / s2) / np.sqrt(2 * np.pi * s2)

    for _ in range(250):
        p0 = pi[0] * sample_normal(x, mu[0], sigma2[0])
        p1 = pi[1] * sample_normal(x, mu[1], sigma2[1])
        denom = p0 + p1
        gamma0 = p0 / denom
        gamma1 = p1 / denom

        Nk = np.array([gamma0.sum(), gamma1.sum()])
        mu[0] = (gamma0 * x).sum() / Nk[0]
        mu[1] = (gamma0 * x).sum() / Nk[1]

        sigma2[0] = (gamma0 * (x - mu[0]) ** 2).sum() / Nk[0]
        sigma2[1] = (gamma1 * (x - mu[1]) ** 2).sum() / Nk[1]
        pi = Nk / Nk.sum()

    labels = (gamma1 > gamma0).astype(np.uint8)  # assume component 1 is foreground
    mask = labels.reshape(h, w)

    _, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("Original")
    axes[1].imshow(mask, cmap="gray")
    axes[1].set_title("EM segmentation")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.suptitle("Custom Expectation-Maximization Algorithm")
    plt.savefig(
        FolderPath.Images.joinpath("example_expectaction_maximization_custom_em.png"),
        dpi=300,
    )
    plt.show()


if __name__ == "__main__":
    plot_scikit()
    plot_custom()
