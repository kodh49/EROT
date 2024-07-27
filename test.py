import torch

def print_tensor(loc: str) -> None:
    T = torch.load(loc)
    print(T)

print_tensor("sc100_2.pt")
print_tensor("wc100_2.pt")