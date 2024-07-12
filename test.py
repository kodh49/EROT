import torch

def print_tensor(loc: str) -> None:
    T = torch.load(loc)
    print(T)

print_tensor("./tests/testdata/C_wcou.pt")
print_tensor("./tests/testdata/C_scou.pt")