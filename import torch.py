import torch
import tensorflow as tf

# --- 1. PyTorch MPS Check ---
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("PyTorch: Using MPS (Apple GPU) for acceleration! 🎉")
else:
    device = torch.device("cpu")
    print("PyTorch: Using CPU.")
    
x = torch.randn(2, 3, device=device)
print("Test Tensor device:", x.device)

# --- 2. TensorFlow Metal Check ---
print("TensorFlow Devices:", tf.config.list_physical_devices())

# Type this to exit the python interpreter:
exit()