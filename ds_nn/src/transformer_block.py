"""danielsinkin97@gmail.com"""

from typing import cast

from torch import Tensor
from torch import nn

from .masked_multi_head_self_attention import MaskedMultiHeadSelfAttention


class TransformerBlock(nn.Module):
    """Pre-norm Transformer block (MHSA -> FFN) with residual connections"""

    def __init__(
        self,
        d_model: int = 768,
        n_head: int = 12,
        d_ff: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()  # type: ignore

        self.ln_mhsa = nn.LayerNorm(d_model)
        self.ln_ff = nn.LayerNorm(d_model)

        self.mhsa = MaskedMultiHeadSelfAttention(d_model=d_model, n_head=n_head)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply MHSA and position-wise FFN (both with residual add & norm).
        """
        output = x + cast(Tensor, self.mhsa(self.ln_mhsa(x)))
        output = output + cast(Tensor, self.feed_forward(self.ln_ff(output)))
        return output
