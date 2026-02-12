# Example: How to add chat template support to Gemma3nTokenizer

@final
class Gemma3nTokenizer(Tokenizer):
    """Gemma3n tokenizer with chat template support."""

    def __init__(self, model: HuggingFaceTokenModel) -> None:
        self._model = model

    # ... existing methods ...

    def apply_chat_template(
        self,
        conversation: list[dict[str, str]],
        *,
        tokenize: bool = True,
        add_generation_prompt: bool = False,
    ) -> list[int] | str:
        """Apply Gemma3n chat template to format conversation.

        Args:
            conversation: List of messages with 'role' and 'content' keys.
                Roles can be 'user', 'assistant', or 'system'.
            tokenize: If True, return token IDs. If False, return formatted string.
            add_generation_prompt: If True, add prompt for model to continue.

        Returns:
            Token IDs if tokenize=True, formatted string otherwise.

        Example:
            >>> conversation = [
            ...     {"role": "user", "content": "Hello!"},
            ...     {"role": "assistant", "content": "Hi there!"},
            ... ]
            >>> tokenizer.apply_chat_template(conversation)
        """
        # Delegate to underlying HuggingFace model
        encoder = self.create_raw_encoder()
        return encoder.apply_chat_template(
            conversation,
            tokenize=tokenize,
            add_generation_prompt=add_generation_prompt,
        )

    @property
    def chat_template(self) -> str | None:
        """Get the current chat template (Jinja2 format)."""
        return self._model._tok.chat_template if hasattr(self._model._tok, "chat_template") else None
