import torch
import numpy as np
import time

# SMALL

# questionable
small_result = time.time() - time.time()

for x in range(1_000_000):
    x = torch.tensor(np.random.rand(1, 1, 10, 10))
    k = torch.tensor(np.random.rand(3, 1, 3, 3))

    start = time.time()
    torch.nn.functional.conv2d(x, k)
    end = time.time()
    small_result += end - start

print(f"Time elapsed for small array: {small_result}s")

# MEDIUM

medium_result = time.time() - time.time()

for x in range(10_000):
    x = torch.tensor(np.random.rand(1, 1, 100, 100))
    k = torch.tensor(np.random.rand(3, 1, 3, 3))

    start = time.time()
    torch.nn.functional.conv2d(x, k)
    end = time.time()
    medium_result += end - start

print(f"Time elapsed for medium array: {medium_result}s")

# LARGE

large_result = time.time() - time.time()

for x in range(10):
    x = torch.tensor(np.random.rand(1, 1, 1000, 1000))
    k = torch.tensor(np.random.rand(3, 1, 3, 3))

    start = time.time()
    torch.nn.functional.conv2d(x, k)
    end = time.time()
    large_result += end - start

print(f"Time elapsed for large array: {large_result}s")
