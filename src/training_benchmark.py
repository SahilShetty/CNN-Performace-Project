from collections.abc import Iterator
from dataclasses import dataclass
from copy import deepcopy

import torch
from torch import nn

import matplotlib.pyplot as plt
from time import perf_counter
import statistics

@dataclass
class TrainingBenchmarkConfig:
    model: nn.Module
    loss_fn: torch.nn.Module
    optimizer: torch.optim.Optimizer
    warmup_iterations: int = 10
    iterations: int = 50
    num_batch_sizes: int = 5

    def __post_init__(self):
        if self.iterations < 2:
            raise ValueError("Number of iterations must be greater than 1")
        if self.warmup_iterations < 0:
            raise ValueError("Iterations must be at least 0")


@dataclass
class TimingStatistics:
    timings_ms: list[float]
    mean_ms: float
    median_ms: float
    stdev_ms: float

@dataclass
class TrainingBenchmarkResults:
    batch_size: int
    warmup_iterations: int
    iterations: int

    total_training_times: TimingStatistics
    dataloader_times: TimingStatistics
    forward_times: TimingStatistics
    loss_times: TimingStatistics
    zero_grad_times: TimingStatistics
    backward_times: TimingStatistics
    optimizer_times: TimingStatistics

    overhead_times: TimingStatistics

BatchIterator = Iterator[tuple[torch.Tensor, torch.Tensor]]

# helper function for dataloader
def next_batch(
        dataloader: torch.utils.data.DataLoader,
        iter_dataloader: BatchIterator
        ) -> tuple[torch.Tensor, torch.Tensor, BatchIterator]:
    try:
        (X, y) = next(iter_dataloader)
    except StopIteration:
        iter_dataloader = iter(dataloader)
        (X, y) = next(iter_dataloader)

    return (X, y, iter_dataloader)

# helper function for processing timings
def timings_processing(timings: list[float]) -> TimingStatistics:
    timings = [1000 * i for i in timings]

    mean_ms = statistics.mean(timings)
    median_ms = statistics.median(timings)
    stdev_ms = statistics.stdev(timings)

    return TimingStatistics(timings, mean_ms, median_ms, stdev_ms)
    

# using iterations and not epochs because benchmarking computation, not model accuracy, is the main goal
# if len(dataloader) == 5000 and we increase epochs by 3,
# that means 15000 - 5000 == 10000 more benchmark iterations
# we want to increase iterations by 1 and have the number of benchmark iterations increase by 1
def single_batch_training_benchmark(dataloader: torch.utils.data.DataLoader,
                                    config: TrainingBenchmarkConfig) -> TrainingBenchmarkResults:
    total_training_times = []
    dataloader_times = []
    forward_times = []
    loss_times = []
    zero_grad_times = []
    backward_times = []
    optimizer_times = []
    overhead_times = []

    config.model.train()

    iter_warmup_dataloader = iter(dataloader)
    for warmup_iteration in range(config.warmup_iterations):
        (X, y, iter_warmup_dataloader) = next_batch(dataloader, iter_warmup_dataloader)

        y_pred = config.model(X)
        loss = config.loss_fn(y_pred, y)
        config.optimizer.zero_grad()
        loss.backward()
        config.optimizer.step()

    # train_loss = 0

    iter_dataloader = iter(dataloader)
    for iteration in range(config.iterations):
        dataloader_time_start = perf_counter()
        (X, y, iter_dataloader) = next_batch(dataloader, iter_dataloader)
        dataloader_time_end = perf_counter()

        total_training_time_start = perf_counter()

        forward_time_start = perf_counter()
        y_pred = config.model(X)
        forward_time_end = perf_counter()

        loss_time_start = perf_counter()
        loss = config.loss_fn(y_pred, y)
        # train_loss += loss.item()
        loss_time_end = perf_counter()

        zero_grad_time_start = perf_counter()
        config.optimizer.zero_grad()
        zero_grad_time_end = perf_counter()

        backward_time_start = perf_counter()
        loss.backward()
        backward_time_end = perf_counter()

        optimizer_time_start = perf_counter()
        config.optimizer.step()
        optimizer_time_end = perf_counter()

        total_training_time_end = perf_counter()

        total_training_times.append(total_training_time_end - total_training_time_start)
        dataloader_times.append(dataloader_time_end - dataloader_time_start)
        forward_times.append(forward_time_end - forward_time_start)
        loss_times.append(loss_time_end - loss_time_start)
        zero_grad_times.append(zero_grad_time_end - zero_grad_time_start)
        backward_times.append(backward_time_end - backward_time_start)
        optimizer_times.append(optimizer_time_end - optimizer_time_start)

        if iteration % 50 == 0:
            print("Finished", iteration, "iterations")

    for i in range(len(total_training_times)):
        overhead_time = (total_training_times[i] -
                         forward_times[i] -
                         loss_times[i] -
                         zero_grad_times[i] -
                         backward_times[i] -
                         optimizer_times[i])
        overhead_times.append(overhead_time)

    batch_size = dataloader.batch_size

    return TrainingBenchmarkResults(batch_size, config.warmup_iterations, config.iterations,
                                    timings_processing(total_training_times),
                                    timings_processing(dataloader_times),
                                    timings_processing(forward_times),
                                    timings_processing(loss_times),
                                    timings_processing(zero_grad_times),
                                    timings_processing(backward_times),
                                    timings_processing(optimizer_times),
                                    timings_processing(overhead_times))

def varying_batch_training_benchmark(dataset: torch.utils.data.TensorDataset,
                                     config: TrainingBenchmarkConfig) -> list[TrainingBenchmarkResults]:
    batch_sizes = [2 ** i for i in range(config.num_batch_sizes)]
    results = []

    # original_config required because optimizer and weights of model don't reset automatically after each iteration of batch_sizes
    # in order to stay consistent, we must reset the optimizer and weights
    # deepcopy required because original_config = config makes it so both variables refer to same memory address
    original_config = deepcopy(config)

    for batch_size in batch_sizes:
        config = deepcopy(original_config)
        dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size)
        result = single_batch_training_benchmark(dataloader, config)
        results.append(result)

        print(f"Finished batch size {batch_size}: {result.total_training_times.mean_ms:.1f} ms total, {result.overhead_times.mean_ms} ms overhead")
    print("\n")

    return results

def plot_total_overhead(results: list[TrainingBenchmarkResults]):
    difrnt_batch_sizes = []
    difrnt_mean_total_times_ms = []
    difrnt_stdev_total_times_ms = []
    difrnt_mean_overhead_times_ms = []
    difrnt_stdev_overhead_times_ms = []

    for result in results:
        difrnt_batch_sizes.append(result.batch_size)
        difrnt_mean_total_times_ms.append(result.total_training_times.mean_ms)
        difrnt_stdev_total_times_ms.append(result.total_training_times.stdev_ms)
        difrnt_mean_overhead_times_ms.append(result.overhead_times.mean_ms)
        difrnt_stdev_overhead_times_ms.append(result.overhead_times.stdev_ms)

    plt.figure()
    plt.errorbar(
        difrnt_batch_sizes,
        difrnt_mean_total_times_ms,
        yerr=difrnt_stdev_total_times_ms,
        color="blue",
        ecolor="red",
        markeredgecolor="black",
        capsize=1.5,
        marker="o"
    )
    plt.xlabel("Batch Size")
    plt.ylabel("Total Training Time per Image (ms)")

    plt.figure()
    plt.errorbar(
        difrnt_batch_sizes,
        difrnt_mean_overhead_times_ms,
        yerr=difrnt_stdev_overhead_times_ms,
        color="blue",
        ecolor="red",
        markeredgecolor="black",
        capsize=1.5,
        marker="o"
    )
    plt.xlabel("Batch Size")
    plt.ylabel("Overhead Time per Image (ms)")

    plt.show()