import numpy as np
import matplotlib.pyplot as plt


def softmax(x: np.ndarray) -> np.ndarray:
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)


def plot_softmax_scaling_example(
    logits: np.ndarray, top_k: int = 3, query_index: int = 0
) -> None:
    """
    Visualizes the effect of scaling attention logits by 1/√dₖ before applying softmax.

    This plot compares the unscaled softmax(x) with the scaled softmax(x / √dₖ) for a single query vector.
    The top-k largest and smallest values in softmax(x) are highlighted and annotated on both curves to show how the
    scaling operation smooths the output distribution, reducing peaks and elevating valleys.
    """
    d_k = logits.shape[1]
    softmax_x = softmax(logits)
    softmax_x_scaled = softmax(logits / np.sqrt(d_k))

    x = np.arange(d_k)
    y1 = softmax_x[query_index]
    y2 = softmax_x_scaled[query_index]

    top_indices = np.argsort(y1)[-top_k:][::-1]
    low_indices = np.argsort(y1)[:top_k]

    plt.figure(figsize=(10, 5))
    plt.plot(x, y1, label=r"$\mathrm{softmax}(x)$")
    plt.plot(x, y2, label=r"$\mathrm{softmax}\left(\frac{x}{\sqrt{d_k}}\right)$")

    for idx in top_indices:
        plt.plot(idx, y1[idx], "go")
        plt.plot(idx, y2[idx], "go")
        for y_val in [y1[idx], y2[idx]]:
            plt.annotate(
                f"{y_val:.3f}",
                (idx, y_val),
                textcoords="offset points",
                xytext=(5, 5),
                ha="left",
                color="white",
                fontsize=9,
                bbox=dict(facecolor="black", alpha=0.4, boxstyle="round,pad=0.3"),
            )

    for idx in low_indices:
        plt.plot(idx, y1[idx], "ro")
        plt.plot(idx, y2[idx], "ro")
        for y_val in [y1[idx], y2[idx]]:
            plt.annotate(
                f"{y_val:.3f}",
                (idx, y_val),
                textcoords="offset points",
                xytext=(-5, -5),
                ha="right",
                va="top",
                color="white",
                fontsize=9,
                bbox=dict(facecolor="black", alpha=0.4, boxstyle="round,pad=0.3"),
            )

    ticks = np.arange(0, d_k, 4)
    if d_k - 1 not in ticks:
        ticks = np.append(ticks, d_k - 1)
    plt.xticks(ticks=ticks)

    plt.title(f"Effect of Scaling on Softmax Output (d_k = {d_k})")
    plt.xlabel("Key Index")
    plt.ylabel("Attention Weight")
    plt.ticklabel_format(axis="y", style="plain")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    seed = 0x2025_07_13
    rng = np.random.default_rng(seed)
    n_rows = 1  # We only plot the first row anyway

    d_k = 64
    logits = rng.normal(loc=0.0, scale=1.0, size=(n_rows, d_k))
    plot_softmax_scaling_example(logits)

    d_k = 128
    logits = rng.normal(loc=0.0, scale=1.0, size=(n_rows, d_k))
    plot_softmax_scaling_example(logits)
