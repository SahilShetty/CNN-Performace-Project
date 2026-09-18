from collections.abc import Iterator

import torch
from torch import nn

from time import perf_counter
import statistics

from dataclasses import dataclass, field

@dataclass
class TrainingBenchmarkConfig:
    model: nn.Module
    dataloader: torch.utils.data.DataLoader
    loss_fn: torch.nn.Module
    optimizer: torch.optim.Optimizer
    warmup_iterations: int = 10
    iterations: int = 100

@dataclass
class TrainingBenchmarkTimingResults:
    total_training_times: list[float]
    dataloader_times: list[float]
    forward_times: list[float]
    loss_times: list[float]
    zero_grad_times: list[float]
    backward_times: list[float]
    optimizer_times: list[float]

    overhead_times: list[float] = field(default_factory=list)

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

def processing_overhead_times(timing_results: TrainingBenchmarkTimingResults) -> None:
    overhead_times = []

    for i in range(len(timing_results.total_training_times)):
        overhead_time = (timing_results.total_training_times[i] -
                         timing_results.forward_times[i] -
                         timing_results.loss_times[i] -
                         timing_results.zero_grad_times[i] -
                         timing_results.backward_times[i] -
                         timing_results.optimizer_times[i])
        overhead_times.append(overhead_time)

    timing_results.overhead_times = overhead_times

# using iterations and not epochs because benchmarking computation, not model accuracy, is the main goal
# if len(dataloader) == 5000 and we increase epochs by 3,
# that means 15000 - 5000 == 10000 more benchmark iterations
# we want to increase iterations by 1 and have the number of benchmark iterations increase by 1
def training_loop(config: TrainingBenchmarkConfig) -> TrainingBenchmarkTimingResults:
    total_training_times = []
    dataloader_times = []
    forward_times = []
    loss_times = []
    zero_grad_times = []
    backward_times = []
    optimizer_times = []

    config.model.train()

    iter_warmup_dataloader = iter(config.dataloader)
    for warmup_iteration in range(config.warmup_iterations):
        (X, y, iter_warmup_dataloader) = next_batch(config.dataloader, iter_warmup_dataloader)

        y_pred = config.model(X)
        loss = config.loss_fn(y_pred, y)
        config.optimizer.zero_grad()
        loss.backward()
        config.optimizer.step()

    # train_loss = 0

    iter_dataloader = iter(config.dataloader)
    for iteration in range(config.iterations):
        dataloader_time_start = perf_counter()
        (X, y, iter_dataloader) = next_batch(config.dataloader, iter_dataloader)
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


        if iteration % 100 == 0:
            print("Done with iteration", iteration)

    timing_results = TrainingBenchmarkTimingResults(
        total_training_times,
        dataloader_times,
        forward_times,
        loss_times,
        zero_grad_times,
        backward_times,
        optimizer_times)
    
    processing_overhead_times(timing_results)

    return timing_results
