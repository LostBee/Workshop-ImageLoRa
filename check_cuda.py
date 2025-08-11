# check_cuda.py
import torch            # add this line
# import platform       # optional – only if you want to print platform info

print("Torch:", torch.__version__)
print(" CUDA:", torch.version.cuda)
print(" GPU :", torch.cuda.get_device_name(0))
print(" OK? :", torch.cuda.is_available())
