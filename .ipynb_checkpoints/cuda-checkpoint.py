import torch

# Check for CUDA Availability
def check_for_CUDA():
    global device
    print(f"Is CUDA supported by this system? {torch.cuda.is_available()}")
    if torch.cuda.is_available() == False:
        device = torch.device("cpu")
        return
    else:
        print(f"CUDA version: {torch.version.cuda}")
        cuda_id = torch.cuda.current_device() # Storing ID of current CUDA device
        print(f"ID of current CUDA device: {torch.cuda.current_device()}")
        print(f"Name of current CUDA device: {torch.cuda.get_device_name(cuda_id)}")
        device = torch.device("cuda")

# When cuda is enabled, transfer a list of data in args to specified gpu with device_id
def transfer_to_GPU(args, device_id):
    if torch.cuda.is_available() == False:
        return
    print(f"Transferring data to cuda:{device_id}...")
    device = torch.device(f"cuda:{device_id}")
    for var in args:
        var.to(device)