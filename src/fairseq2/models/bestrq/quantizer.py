from abc import ABC

import torch
from torch import nn
from torch.nn import Module, Parameter

from dataclasses import dataclass
from fairseq2.nn import Linear
from fairseq2.typing import DataType, Device


class RandomVectorQuantizer(Module, ABC):
    """Quantizes incoming data in a differentiable way."""

    input_dim: int
    output_dim: int
    num_codebook_entries: int
    random_projection: Linear
    code_book: Parameter
    num_updates: torch.Tensor
    normalize_input: bool

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_codebook_entries: int,
        normalize_input: bool,
        device: Device | None = None,
        dtype: DataType | None = None,
    ) -> None:
        """
        :param input_dim:
            The dimensionality of inputs.
        :param output_dim:
            The dimensionality of quantized outputs.
        """
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_codebook_entries = num_codebook_entries
        self.normalize_input = normalize_input

        self.random_projection = Linear(
            self.input_dim,
            self.output_dim,
            bias=False,
            device=device,
            dtype=dtype,
        )
        self.codebook = Parameter(
            torch.zeros(
                self.num_codebook_entries,
                self.output_dim,
                device=device,
                dtype=dtype,
            )
        )

        num_updates = torch.empty((), device=device, dtype=torch.int64)

        self.register_buffer("num_updates", num_updates)

        self.reset_parameters()

        # Freeze the random projection and code book
        self.random_projection.requires_grad = False
        self.codebook.requires_grad = False

    def reset_parameters(self) -> None:
        """Reset the parameters and buffers of the module."""
        nn.init.xavier_uniform_(self.random_projection.weight)
        
        torch.nn.init.normal_(self.codebook, mean=0.0, std=1.0)

        self.num_updates.zero_()

    def cosine_similarity(self, x1, x2, dim=1, eps=1e-8):
        # Normalize along the specified dimension
        x1_norm = x1 / (x1.norm(dim=dim, dtype=x1.dtype).clamp(min=eps).unsqueeze(dim))
        x2_norm = x2 / (x2.norm(dim=dim, dtype=x2.dtype).clamp(min=eps).unsqueeze(dim))
        # Compute dot product along the specified dimension
        return torch.sum(x1_norm * x2_norm, dim=dim, dtype=x1.dtype)

    def forward(self, x: torch.Tensor):
        bsz, tsz, fsz = x.shape

        if self.normalize_input:
            x = (x - x.mean(dim=[1, 2], keepdims=True)) / x.std(
                dim=[1, 2], keepdims=True
            )

        vector_output = self.random_projection(x).unsqueeze(-2)  # (N, T, quantized_dim)

        # Compute l2 norm targets and code vectors
        similarity_scores = self.cosine_similarity(
            vector_output, self.codebook, dim=-1
        )  # (N, T, num_codebook_entries)
        quantized_targets = torch.argmax(similarity_scores, dim=-1)  # N, T
        
        return RandomVectorQuantizerOutput(quantizer_output=vector_output.squeeze(), quantized_targets=quantized_targets, num_codebook_entries=self.num_codebook_entries)


@dataclass
class RandomVectorQuantizerOutput(ABC):
    """Holds the output of a vector quantizer."""

    quantizer_output: torch.Tensor
    """The vector output of the quantizer."""

    quantized_targets: torch.Tensor
    """The quantized target output."""
    
    num_codebook_entries: int
    """Number of entries in the codebook."""
    
    def compute_target_entropy(self):
        
        idxs, counts = torch.unique(self.quantized_targets, return_counts=True)
        counts = counts.float()
        probs = torch.zeros(self.num_codebook_entries, device=counts.device).scatter(
            0, idxs, counts
        )
        probs /= probs.sum()
        probs += 1e-10
        
        return -(probs * torch.log(probs)).sum().item()


class MultiRandomVectorQuantizer(Module, ABC):

    input_dim: int
    output_dim: int
    num_codebook_entries: int
    quantizers: nn.ModuleList  # list[RandomVectorQuantizer]
    num_updates: torch.Tensor
    normalize_input: bool

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_codebook_entries: int,
        num_quantizer: int,
        normalize_input: bool,
        device: Device | None = None,
        dtype: DataType | None = None,
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_codebook_entries = num_codebook_entries
        self.num_quantizer = num_quantizer
        self.normalize_input = normalize_input

        self.quantizers = nn.ModuleList([])
        for _ in range(num_quantizer):
            quantizer = RandomVectorQuantizer(
                input_dim=input_dim,
                output_dim=output_dim,
                num_codebook_entries=num_codebook_entries,
                normalize_input=normalize_input,
                device=device,
                dtype=dtype,
            )

            # Freeze it
            for param in quantizer.parameters():
                param.requires_grad = False
            self.quantizers.append(quantizer)

    def forward(self, x: torch.tensor):
        return MultiRandomVectorQuantizerOutput(
            quantizer_outputs=[q(x) for q in self.quantizers]
        )


@dataclass
class MultiRandomVectorQuantizerOutput(ABC):
    """Holds the output of a vector quantizer."""

    quantizer_outputs: list[RandomVectorQuantizerOutput]

    @property
    def quantized_targets(self):
        return torch.cat(
            [q.quantized_targets.unsqueeze(0) for q in self.quantizer_outputs], dim=0
        )

    @property
    def quantizer_output(self):
        return torch.cat(
            [q.quantizer_output.unsqueeze(0) for q in self.quantizer_outputs], dim=0
        )
    
    def compute_target_entropy(self):
        return torch.tensor([q.compute_target_entropy() for q in self.quantizer_outputs])
    
    @property
    def num_codebook_entries(self):
        assert len(set([q.num_codebook_entries for q in self.quantizer_outputs])) == 1
        return self.quantizer_outputs[0].num_codebook_entries
        