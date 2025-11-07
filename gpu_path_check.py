import torch
import os
import sys

# --- 1. Find NVIDIA CUDA Installation Path ---
# This path is the default location where the CUDA Toolkit is installed.
# You might need to change '12.1' if you installed a slightly different version.
cuda_home = 'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.1'

if os.path.isdir(cuda_home):
    # --- 2. Add NVIDIA Binaries to the System PATH for this session ---
    os.environ['PATH'] = cuda_home + '/bin;' + os.environ['PATH']
    os.environ['PATH'] = cuda_home + '/lib/x64;' + os.environ['PATH']
    print(f"CUDA paths added: {cuda_home}")
else:
    print("Warning: CUDA Toolkit path not found in default location. Check installation.")
    sys.exit()

# --- 3. Final Check ---
print("\n--- Running Final PyTorch Check ---")
if torch.cuda.is_available():
    print(f"✅ SUCCESS: CUDA is available after path adjustment.")
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
else:
    print("❌ FAILURE: CUDA is still not available. Consult NVIDIA driver diagnostics.")