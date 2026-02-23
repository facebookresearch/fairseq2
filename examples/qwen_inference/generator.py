"""Text generator setup for Qwen inference."""

from torch.nn import Module

from fairseq2.data.tokenizers import Tokenizer
from fairseq2.gang import Gangs
from fairseq2.generation.sampling import SamplingSequenceGenerator, TopPSampler
from fairseq2.generation.text import TextCompleter
from fairseq2.logging import log


def create_text_completer(
    model: Module,
    tokenizer: Tokenizer,
    gangs: Gangs,
    max_gen_len: int,
    temperature: float,
    top_p: float,
    echo_prompt: bool = True,
    skip_special_tokens: bool = True,
) -> TextCompleter:
    """
    Create a TextCompleter for text generation.

    Args:
        model: Language model for generation
        tokenizer: Tokenizer for encoding/decoding text
        gangs: Gang abstraction for distributed execution
        max_gen_len: Maximum generation length in tokens
        temperature: Sampling temperature
        top_p: Top-p (nucleus) sampling parameter
        echo_prompt: Whether to include prompt in output
        skip_special_tokens: Whether to skip special tokens in output

    Returns:
        Configured TextCompleter
    """
    # Create a Top-P (nucleus) sampler
    sampler = TopPSampler(p=top_p)

    # Create a sampling-based sequence generator
    generator = SamplingSequenceGenerator(
        model=model,
        vocab_info=tokenizer.vocab_info,
        sampler=sampler,
        max_gen_len=max_gen_len,
        temperature=temperature,
        echo_prompt=echo_prompt,
    )

    # Create a TextCompleter
    text_completer = TextCompleter(
        generator=generator,
        tokenizer=tokenizer,
        skip_special_tokens=skip_special_tokens,
    )

    if gangs.root.rank == 0:
        log.info("Generator setup complete!")
        log.info("Max generation length: {}", max_gen_len)
        log.info("Temperature: {}", temperature)
        log.info("Top-p: {}", top_p)

    return text_completer
