import torch

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())

if torch.cuda.is_available():
    print("GPU Name:", torch.cuda.get_device_name(0))
    x = torch.rand(3, 3).to("cuda")
    print("Tensor on GPU:", x)
else:
    print("No GPU detected by PyTorch.")
