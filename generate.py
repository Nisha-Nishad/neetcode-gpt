import torch
import torch.nn as nn
from torchtyping import TensorType

class Solution:
    @torch.no_grad()
    def generate(self, model, new_chars: int, context: TensorType[int], context_length: int, int_to_char: dict) -> str:
        generator = torch.manual_seed(0)
        initial_state = generator.get_state()
        result = []

        for _ in range(new_chars):
            # Crop to the model's maximum window
            context = context[:, -context_length:]

            # Forward pass -> logits of last position -> probabilities
            logits = model(context)
            probs = torch.softmax(logits[:, -1, :], dim=-1)

            # Sample (generator state reset, as required by the fixed code)
            generator.set_state(initial_state)
            next_token = torch.multinomial(probs, 1, generator=generator)

            # Append and record the character
            context = torch.cat((context, next_token), dim=1)
            result.append(int_to_char[next_token.item()])

        return "".join(result)