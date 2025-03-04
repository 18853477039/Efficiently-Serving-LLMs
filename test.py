import torch

next_token_id = torch.tensor(5)
print(next_token_id)
reshaped_token = next_token_id.reshape(1, 1)
print(reshaped_token)