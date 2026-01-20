import torch
import time
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = torch.nn.Sequential(
    torch.nn.Linear(4096, 4096),
    torch.nn.ReLU(),
    torch.nn.Linear(4096, 4096),
    torch.nn.ReLU(),
    torch.nn.Linear(4096, 4096),
    torch.nn.Linear(4096, 1)
).to(device)

batch_size = 128
dummy_input = torch.randn(batch_size, 4096).to(device)

output = model(dummy_input)

if device.type == 'cuda':
    max_memory = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    print(f"GPU Naam: {torch.cuda.get_device_name(0)}")
    print(f"Max VRAM: {max_memory:.2f} MB")
else:
    print("Geen GPU gevonden")
