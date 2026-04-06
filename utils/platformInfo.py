import platform
import torch

def log_system_info():
    print("===== System Info =====")

    # CPU
    cpu_name = platform.processor()
    print(f"CPU: {cpu_name}")

    # OS (optional but useful)
    print(f"OS: {platform.system()} {platform.release()}")

    # GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"GPU: {gpu_name}")
        print(f"CUDA: Available")
    else:
        print("GPU: None")
        print("CUDA: Not available")

    print("========================\n")