from accelerate import Accelerator

acc = Accelerator()
print("Accelerate device:", acc.device)
