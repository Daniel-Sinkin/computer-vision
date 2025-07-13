import numpy as np
import matplotlib.pyplot as plt

_rng = np.random.default_rng(0x2025_07_13)


def softmax(x: np.ndarray) -> np.ndarray:
    es = np.exp(x - np.max(x))
    return es / es.sum()


def example_transformer() -> None:
    d_in = 16
    d_out = 8
    d_q = d_out
    N = 32
    temp = 16

    X = _rng.normal(loc=0.0, scale=1.0, size=(N, d_in)).astype(np.float32)
    W_k = _rng.normal(loc=0.0, scale=1.0, size=(d_in, d_out)).astype(np.float32)
    W_v = _rng.normal(loc=0.0, scale=1.0, size=(d_in, d_out)).astype(np.float32)
    W_q = _rng.normal(loc=0.0, scale=1.0, size=(d_in, d_out)).astype(np.float32)

    query = np.matmul(X, W_q)
    print(f"query = np.matmul(X, W_q)")
    print(f"{X.shape=},{W_q.shape=} -> {query.shape=}")
    print()
    keys = np.matmul(X, W_k)
    print(f"keys = np.matmul(X, W_k)")
    print(f"{X.shape=},{W_k.shape=} -> {keys.shape=}")
    print()
    values = np.matmul(X, W_v)
    print(f"values = np.matmul(X, W_v)")
    print(f"{X.shape=},{W_v.shape=} -> {values.shape=}")
    print()
    similarities_noscale = np.matmul(query, keys.T)
    print(f"similarities (no scale) = np.matmul(query, keys.T)")
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
    for col in range(N):
        for row in range(col + 1, N):
            similarities_masked[row, col] = -np.inf

    attention_weights = softmax(similarities)
    print(f"attention_weights[:3,:3]=\n{attention_weights[:3,:3]}")
    print(
        f"{np.max(attention_weights)=:.4f},{np.min(attention_weights)=:.4f},{np.sum(attention_weights)=:.4f}"
    )
    attention_weights_temp = softmax(similarities / temp)
    print(f"Due to low dim this is still too spiky, dividing by {temp} before softmax")
    print(f"attention_weights_temp[:3,:3]=\n{attention_weights_temp[:3,:3]}")
    print(
        f"{np.max(attention_weights_temp)=:.4f},{np.min(attention_weights_temp)=:.4f},{np.sum(attention_weights_temp)=:.4f}"
    )

    attention_weights_temp_masked = softmax(similarities_masked / temp)
    print(
        f"attention_weights_temp_masked[:3,:3]=\n{attention_weights_temp_masked[:3,:3]}"
    )
    print(
        f"{np.max(attention_weights_temp_masked)=:.4f},{np.min(attention_weights_temp_masked)=:.4f},{np.sum(attention_weights_temp_masked)=:.4f}"
    )

    plt.figure(figsize=(9, 8))
    plt.imshow(attention_weights, cmap="viridis", aspect="auto")
    plt.colorbar(label="Attention Weight (temp adj. {temp=})")
    plt.title(f"Self-Attention Weights Heatmap\ntemperature adjusted {temp=}")
    plt.xlabel("Key index")
    plt.ylabel("Query index")
    ticks = np.arange(0, N, 4)
    if (N - 1) not in ticks:
        ticks = np.append(ticks, N - 1)

    plt.xticks(ticks)
    plt.yticks(ticks)
    plt.tight_layout()
    plt.savefig("example_transformer_attention_weights.png", dpi=300)
    plt.show()

    plt.figure(figsize=(9, 8))
    plt.imshow(attention_weights_temp, cmap="viridis", aspect="auto")
    plt.colorbar(label="Attention Weight (temp adj. {temp=})")
    plt.title(f"Self-Attention Weights Heatmap\ntemperature adjusted {temp=}")
    plt.xlabel("Key index")
    plt.ylabel("Query index")
    ticks = np.arange(0, N, 4)
    if (N - 1) not in ticks:
        ticks = np.append(ticks, N - 1)

    plt.xticks(ticks)
    plt.yticks(ticks)
    plt.tight_layout()
    plt.savefig("example_transformer_attention_weights_temp.png", dpi=300)
    plt.show()

    plt.figure(figsize=(9, 8))
    plt.imshow(attention_weights_temp_masked, cmap="viridis", aspect="auto")
    plt.colorbar(label="Attention Weight Masked (temp adj. {temp=})")
    plt.title(f"Self-Attention Weights Heatmap Masked\ntemperature adjusted {temp=}")
    plt.xlabel("Key index")
    plt.ylabel("Query index")
    ticks = np.arange(0, N, 4)
    if (N - 1) not in ticks:
        ticks = np.append(ticks, N - 1)

    plt.xticks(ticks)
    plt.yticks(ticks)
    plt.tight_layout()
    plt.savefig("example_transformer_attention_weights_temp_masked.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    example_transformer()
