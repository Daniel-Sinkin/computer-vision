import numpy as np
import matplotlib.pyplot as plt
import numpy.typing
from typing import TypeVar

_rng = np.random.default_rng(0x2025_07_13)

# pylint:disable=invalid-name


T = TypeVar("T", bound=np.floating)
NDArrayT = numpy.typing.NDArray[T]


def softmax(x: NDArrayT[np.floating]) -> NDArrayT[np.floating]:
    """Computes softmax, applies the usual numerical stability trick."""
    es = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return es / np.sum(es, axis=-1, keepdims=True)


def plot_attention_weights(
    weights: NDArrayT[np.floating], title: str, filename: str
) -> None:
    """Plotting Helper"""
    seq_len = weights.shape[0]

    plt.figure(figsize=(9, 8))  # type: ignore
    plt.imshow(weights, cmap="viridis", aspect="auto")  # type: ignore
    plt.colorbar(label="Attention Weight")  # type: ignore
    plt.title(f"{title}")  # type: ignore
    plt.xlabel("Key index")  # type: ignore
    plt.ylabel("Query index")  # type: ignore
    ticks = np.arange(0, seq_len, 4)  # type: ignore
    if (seq_len - 1) not in ticks:  # type: ignore
        ticks = np.append(ticks, seq_len - 1)  # type: ignore
    plt.xticks(ticks)  # type: ignore
    plt.yticks(ticks)  # type: ignore
    plt.tight_layout()  # type: ignore
    plt.savefig(filename, dpi=300)  # type: ignore
    plt.show()  # type: ignore


def example_transformer() -> None:
    """
    Does a single single head self-attention step on random data.
    """
    d_in = 16
    d_out = 8
    d_q = d_out
    seq_len = 32
    temperature = 16

    X = _rng.normal(loc=0.0, scale=1.0, size=(seq_len, d_in)).astype(np.float32)
    W_K = _rng.normal(loc=0.0, scale=1.0, size=(d_in, d_out)).astype(np.float32)
    W_V = _rng.normal(loc=0.0, scale=1.0, size=(d_in, d_out)).astype(np.float32)
    W_Q = _rng.normal(loc=0.0, scale=1.0, size=(d_in, d_out)).astype(np.float32)

    query = np.matmul(X, W_Q)
    print("query = np.matmul(X, W_q)")
    print(f"{X.shape=},{W_Q.shape=} -> {query.shape=}")
    keys = np.matmul(X, W_K)
    print("keys = np.matmul(X, W_k)")
    print(f"{X.shape=},{W_K.shape=} -> {keys.shape=}")
    values = np.matmul(X, W_V)
    print("values = np.matmul(X, W_v)")
    print(f"{X.shape=},{W_V.shape=} -> {values.shape=}")
    print()
    similarities_noscale = np.matmul(query, keys.T)
    print("similarities (no scale) = np.matmul(query, keys.T)")
    print(f"{query.shape=},{keys.T.shape=} -> {similarities_noscale.shape=}")
    print(f"{similarities_noscale[:3, :3]} = {similarities_noscale[:3, :3]}")
    print()
    print(
        f"similarities = similarities_noscale / np.sqrt(d_q) = similarities_noscale / {np.sqrt(d_q):.2f}"
    )
    similarities = similarities_noscale / np.sqrt(d_q)
    print(f"similarities[:3,:3] = \n{similarities[:3,:3]}")
    print(f"query[2]=\n{query[2]}")
    print(f"keys[1]=\n{keys[1]}")
    print(f"np.dot(query[2], keys[1])={float(np.dot(query[2],keys[1])):.4f}")
    print(f"similarities_noscale[2, 1]={float(similarities_noscale[2, 1]):.4f}")
    print()
    similarities_masked = similarities.copy()
    # https://numpy.org/doc/2.2/reference/generated/numpy.tril_indices.html
    # Could also use https://numpy.org/doc/2.2/reference/generated/numpy.mask_indices.html#numpy.mask_indices
    rows, cols = np.tril_indices(seq_len, k=-1)
    similarities_masked[rows, cols] = -np.inf

    attention_weights = softmax(similarities)
    print(f"attention_weights[:3,:3]=\n{attention_weights[:3,:3]}")
    print(
        f"{np.max(attention_weights)=:.4f},"
        f"{np.min(attention_weights)=:.4f},"
        f"{np.sum(attention_weights)=:.4f}"
    )
    attention_weights_temp = softmax(similarities / temperature)
    print(
        f"Due to low dim this is still too spiky, dividing by {temperature} before softmax"
    )
    print(f"attention_weights_temp[:3,:3]=\n{attention_weights_temp[:3,:3]}")
    print(
        f"{np.max(attention_weights_temp)=:.4f},"
        f"{np.min(attention_weights_temp)=:.4f},"
        f"{np.sum(attention_weights_temp)=:.4f}"
    )

    attention_weights_temp_masked = softmax(similarities_masked / temperature)
    print(
        f"attention_weights_temp_masked[:3,:3]=\n{attention_weights_temp_masked[:3,:3]}"
    )
    print(
        f"{np.max(attention_weights_temp_masked)=:.4f},{np.min(attention_weights_temp_masked)=:.4f},{np.sum(attention_weights_temp_masked)=:.4f}"
    )

    plot_attention_weights(
        weights=attention_weights,
        title="Self-Attention Weights Heatmap",
        filename="example_transformer_attention_weights.png",
    )

    plot_attention_weights(
        weights=attention_weights_temp,
        title="Self-Attention Weights Heatmap (Temp)",
        filename="example_transformer_attention_weights_temp.png",
    )

    plot_attention_weights(
        weights=attention_weights_temp_masked,
        title="Self-Attention Weights Heatmap Masked (Temp)",
        filename="example_transformer_attention_weights_temp_masked.png",
    )


if __name__ == "__main__":
    example_transformer()
