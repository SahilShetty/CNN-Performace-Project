import time

import torch
from torch import nn

import matplotlib.pyplot as plt

def single_batch_inference_benchmark(
        model: nn.Module,
        input: torch.Tensor,
        warmup_iterations: int = 10,
        iterations: int = 50
) -> tuple[float, float]:
    model.eval()

    with torch.inference_mode():
        # Warmup iterations
        for _ in range(warmup_iterations):
            model(input)

        start = time.perf_counter()

        # Benchmarking iterations
        for _ in range(iterations):
            model(input)

    end = time.perf_counter()

    time_taken =  end - start
    avg_latency_ms = time_taken / iterations * 1000

    batch_size = input.shape[0]
    avg_images_per_sec = batch_size * iterations / time_taken

    return avg_latency_ms, avg_images_per_sec


def varying_batch_inference_benchmark(
        model: nn.Module,
        warmup_iterations: int = 10,
        iterations: int = 50,
        diff_batch_sizes: int = 3):

    batch_sizes = [2 ** i for i in range(diff_batch_sizes)]
    diff_latency_ms = []
    diff_images_per_sec = []


    for batch_size in batch_sizes:
        input = torch.rand(batch_size, 3, 256, 256)
        avg_latency_ms, avg_images_per_sec = single_batch_inference_benchmark(model, input, warmup_iterations, iterations)
        diff_latency_ms.append(avg_latency_ms)
        diff_images_per_sec.append(avg_images_per_sec)

        print("Finished batch ", batch_size)

    plt.figure()
    plt.plot(batch_sizes, diff_latency_ms, color="red")
    plt.xlabel("Batch Size")
    plt.ylabel("Latency (ms)")

    plt.figure()
    plt.plot(batch_sizes, diff_images_per_sec, color="blue")
    plt.xlabel("Batch Size")
    plt.ylabel("Images per second")

    plt.show()

    


    
