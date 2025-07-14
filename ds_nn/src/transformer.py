"""danielsinkin97@gmail.com"""

from torch import Tensor
from torch import nn

from .transformer_block import TransformerBlock


class Transformer(nn.Module):
    """Stack of TransformerBlock for the encoder."""

    def __init__(
        self, d_model: int = 768, n_head: int = 12, d_ff: int = 2024, n_layer: int = 12
    ):
        super().__init__()  # type: ignore

        self.d_model = d_model
        self.n_head = n_head
        self.n_layer = n_layer
        self.d_ff = d_ff

        self.transformer_blocks = nn.Sequential(
            *(
                TransformerBlock(d_model=self.d_model, n_head=n_head, d_ff=self.d_ff)
                for _ in range(self.n_layer)
            )
        )

    def forward(self, x: Tensor) -> Tensor:
        """Run the input through every Transformer block in sequence."""
        return self.transformer_blocks(x)
