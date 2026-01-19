import torch
import time

model = torch.nn.Sequential(
    torch.nn.Linear(2, 1024),
    torch.nn.ReLU(),
    torch.nn.Dropout(p=0.5),
    torch.nn.Linear(1024, 1),
    torch.nn.Sigmoid()
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


dummy_input = torch.randn(1, 2).to(device)

print(f"Starting benchmark on {device}...")

for _ in range(10):
    _ = model(dummy_input)
if device.type == 'cuda': torch.cuda.synchronize()

start_time = time.time()
iterations = 1000  # We doen er wat meer voor een betere meting op GPU
for _ in range(iterations):
    _ = model(dummy_input)
    
if device.type == 'cuda': torch.cuda.synchronize()

end_time = time.time()

total_time = end_time - start_time
throughput = iterations / total_time

print(f"Total time for {iterations} iterations: {total_time:.4f} seconds")
print(f"Throughput: {throughput:.2f} img/s")

if device.type == 'cuda':
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
