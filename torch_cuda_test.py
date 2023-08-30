import torch

print(f'device availability: {torch.cuda.is_available()}')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)



# #Additional Info when using cuda
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')
#
#
    max_memory_allocated = torch.cuda.max_memory_allocated()
    print(f"Maximum GPU memory allocated by PyTorch: {max_memory_allocated / 1024**3:.2f} GB")

x = torch.Tensor([1.0, -2.3]).to(device)
print(f'x (on device?) {x}')