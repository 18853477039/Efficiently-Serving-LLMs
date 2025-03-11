import torch

def generate(model, tokenizer, request_queue: [(str, int)]):
    """
    :param model: like GPT2LMHeadModel
    :param tokenizer:
    :param request_queue: [("The quick brown fox jumped over the", max_tokens)]
    :return: generated_tokens: [str]
    """
    generated_tokens = []
    for request in request_queue:
        prompt, max_tokens = request
        inputs = tokenizer(prompt, return_tensors="pt")
        next_inputs = inputs
        for _ in range(max_tokens):
            with torch.no_grad():
                outputs = model(**next_inputs)
            logits = outputs.logits
            last_logits = logits[0, -1, :]
            next_token_id = last_logits.argmax()
            next_inputs = {
                "input_ids": torch.cat(
                    [inputs["input_ids"], next_token_id.reshape(1, 1)],
                    dim=1
                ),
                "attention_mask": torch.cat(
                    [inputs["attention_mask"], torch.tensor([[1]])],
                    dim=1
                )
            }
            generated_tokens.append(tokenizer.decode(next_token_id))
    return "".join(generated_tokens),