from typing import List, Tuple
from dataclasses import dataclass, field

import torch


@dataclass
class Batch:
    max_tokens: int
    prompts: List[str]

    def __post_init__(self):
        if not isinstance(self.max_tokens, int) or self.max_tokens <= 0:
            raise ValueError("max_tokens must be a positive integer")
        if not all(isinstance(prompt, str) for prompt in self.prompts):
            raise ValueError("All prompts must be strings")


def generate_batch_tokens_with_past(inputs, model):
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    last_logits = logits[:, -1, :]
    next_token_ids = last_logits.argmax(dim=1)
    return next_token_ids, outputs.past_key_values


def generate_batch(inputs, max_tokens, model, tokenizer):
    # create a list of tokens for every input in the batch
    generated_tokens = [[] for _ in range(inputs["input_ids"].shape[0])]

    attention_mask = inputs["attention_mask"]
    position_ids = attention_mask.long().cumsum(-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 1)

    next_inputs = {
        "position_ids": position_ids,
        **inputs
    }

    for _ in range(max_tokens):
        next_token_ids, past_key_values = generate_batch_tokens_with_past(next_inputs, model=model)
        next_inputs = {
            "input_ids": next_token_ids.reshape(-1, 1),
            "position_ids": next_inputs["position_ids"][:, -1].unsqueeze(-1) + 1,
            "attention_mask": torch.cat(
                [next_inputs["attention_mask"], torch.ones((next_token_ids.shape[0], 1))],
                dim=1
            ),
            "past_key_values": past_key_values
        }

        next_tokens = tokenizer.batch_decode(next_token_ids)
        for i, token in enumerate(next_tokens):
            generated_tokens[i].append(token)

    return ["".join(tokens) for tokens in generated_tokens]


def init_batch(sub_request_queue: List[Tuple[str, int]]) -> Batch:
    batch_prompts = [b[0] for b in sub_request_queue]
    max_tokens = max(b[1] for b in sub_request_queue)
    return Batch(max_tokens=max_tokens, prompts=batch_prompts)


def generate_next_token(batch: Batch, model, tokenizer):
    inputs = tokenizer(
        batch.prompts, padding=True, return_tensors="pt")
    return generate_batch(inputs, max_tokens=batch.max_tokens, model=model, tokenizer=tokenizer)



def merge_batches(cached_batch, new_batch):

    pass


def filter_batch(cached_batch):

    pass


def generate(model, tokenizer, request_queue):
    pass