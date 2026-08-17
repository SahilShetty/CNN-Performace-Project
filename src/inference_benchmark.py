import time

import torch
from torch import nn

import statistics
import matplotlib.pyplot as plt

from dataclasses import dataclass

@dataclass
class InferenceBenchmarkResult:
    batch_size: int
    warmup_iterations: int
    iterations: int
    latencies_ms: list[float]
    mean_latency_ms: float
    median_latency_ms: float
    stdev_latency_ms: float
    throughput_images_per_sec: float

def single_batch_inference_benchmark(
        model: nn.Module,
        inputs: torch.Tensor,
        warmup_iterations: int = 10,
        iterations: int = 50
) -> InferenceBenchmarkResult:
    model.eval()

    with torch.inference_mode():
        # Warmup iterations
        for _ in range(warmup_iterations):
            model(inputs)

        latencies_ms = []
        # Benchmarking iterations
        for _ in range(iterations):
            start = time.perf_counter()
            model(inputs)
            end = time.perf_counter()

            latencies_ms.append((end - start) * 1000)

    mean_latency_ms = statistics.mean(latencies_ms)
    median_latency_ms = statistics.median(latencies_ms)
    stdev_latency_ms = statistics.stdev(latencies_ms)

    batch_size = inputs.shape[0]
    throughput_images_per_sec = batch_size / mean_latency_ms * 1000

    return InferenceBenchmarkResult(batch_size, warmup_iterations, iterations,
                                    latencies_ms, mean_latency_ms, median_latency_ms,
                                    stdev_latency_ms, throughput_images_per_sec)


def varying_batch_inference_benchmark(
        model: nn.Module,
        warmup_iterations: int = 10,
        iterations: int = 50,
        diff_batch_sizes: int = 3):

    batch_sizes = [2 ** i for i in range(diff_batch_sizes)]
    diff_latency_ms = []
    diff_images_per_sec = []


    for batch_size in batch_sizes:
        inputs = torch.rand(batch_size, 3, 224, 224)
        result = single_batch_inference_benchmark(model, inputs, warmup_iterations, iterations)
        diff_latency_ms.append(result.mean_latency_ms)
        diff_images_per_sec.append(result.throughput_images_per_sec)

        print("Finished batch size", batch_size)

    plt.figure()
    plt.plot(batch_sizes, diff_latency_ms, color="red")
    plt.xlabel("Batch Size")
    plt.ylabel("Latency (ms)")

    plt.figure()
    plt.plot(batch_sizes, diff_images_per_sec, color="blue")
    plt.xlabel("Batch Size")
    plt.ylabel("Images per second")

    plt.show()


    


    
