import torch
import tqdm
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

test_count = 10000000


def test1(loop_count):
    for loop in range(loop_count):
        print(f"Loop {loop} start!")
        for _ in tqdm.tqdm(range(test_count), "torch.zeros(device=device)"):
            x = torch.zeros(12, 12, 12, device=device)
        for _ in tqdm.tqdm(range(test_count), "torch.zeros().to(device)"):
            x = torch.zeros(12, 12, 12).to(device)
        print(f"Loop {loop} end!\n")


def test2(loop_count):
    test_vec = np.random.rand(12, 12, 12)
    for loop in range(loop_count):
        print(f"Loop {loop} start!")
        for _ in tqdm.tqdm(range(test_count), "torch.tensor(device=device)"):
            x = torch.tensor(test_vec, device=device)
            pass
        for _ in tqdm.tqdm(range(test_count), "torch.tensor().to(device)"):
            x = torch.tensor(test_vec).to(device)
        print(f"Loop {loop} end!\n")


test2(2)