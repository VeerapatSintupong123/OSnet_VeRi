import torch
import torchreid

print("=== Library Versions ===")
print("torch version      :", torch.__version__)
print("torchreid version  :", torchreid.__version__)

print("\n=== CUDA Information ===")
print("CUDA available     :", torch.cuda.is_available())
print("CUDA version (PT)  :", torch.version.cuda)

if torch.cuda.is_available():
    print("GPU count          :", torch.cuda.device_count())
    print("Current GPU index  :", torch.cuda.current_device())
    print("GPU name           :", torch.cuda.get_device_name(0))
    print("Compute capability :", torch.cuda.get_device_capability(0))

# === Library Versions ===
# torch version      : 2.8.0+cu126
# torchreid version  : 0.2.5

# === CUDA Information ===
# CUDA available     : True
# CUDA version (PT)  : 12.6
# GPU count          : 1
# Current GPU index  : 0
# GPU name           : NVIDIA GeForce RTX 4050 Laptop GPU
# Compute capability : (8, 9)