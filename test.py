import torch
print("CUDA beschikbaar?:", torch.cuda.is_available())
print("Actieve GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Geen")
