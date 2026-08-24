import torch
from torch import nn

from time import perf_counter
import statistics

from dataclasses import dataclass

@dataclass
class TrainingBenchmarkConfig:
    model: nn.Module
    dataloader: torch.utils.data.DataLoader
    loss_fn: torch.nn.Module
    optimizer: torch.optim.Optimizer
    warmup_iterations: int = 10
    iterations: int = 25

# using iterations and not epochs because benchmarking computation, not model accuracy, is the main goal
# if len(dataloader) == 5000 and we increase epochs by 3,
# that means 15000 - 5000 == 10000 more benchmark iterations
# we want to increase iterations by 1 and have the number of benchmark iterations increase by 1
def training_loop(config: TrainingBenchmarkConfig):
    total_training_times = []
    dataloader_times = []
    forward_times = []
    backward_times = []
    optimizer_times = []

    config.model.train()

    iter_warmup_dataloader = iter(config.dataloader)
    for warmup_iteration in range(config.warmup_iterations):
        try:
            (X, y) = next(iter_warmup_dataloader)
        except StopIteration:
            iter_warmup_dataloader = iter(config.dataloader)
            (X, y) = next(iter_warmup_dataloader)

        y_pred = config.model(X)
        loss = config.loss_fn(y_pred, y)
        config.optimizer.zero_grad()
        loss.backward()
        config.optimizer.step()

    train_loss = 0

    iter_dataloader = iter(config.dataloader)
    for iteration in range(config.iterations):
        dataloader_time_start = perf_counter()
        try:
            (X, y) = next(iter_dataloader)
        except StopIteration:
            iter_dataloader = iter(config.dataloader)
            (X, y) = next(iter_dataloader)

        dataloader_time_end = perf_counter()
        dataloader_times.append(dataloader_time_end - dataloader_time_start)

        total_training_time_start = perf_counter()

        forward_time_start = perf_counter()
        y_pred = config.model(X)
        forward_time_end = perf_counter()
        forward_times.append(forward_time_end - forward_time_start)

        loss = config.loss_fn(y_pred, y)
        train_loss += loss.item()

        config.optimizer.zero_grad()

        backward_time_start = perf_counter()
        loss.backward()
        backward_time_end = perf_counter()
        backward_times.append(backward_time_end - backward_time_start)

        optimizer_time_start = perf_counter()
        config.optimizer.step()
        optimizer_time_end = perf_counter()
        optimizer_times.append(optimizer_time_end - optimizer_time_start)

        total_training_time_end = perf_counter()
        total_training_times.append(total_training_time_end - total_training_time_start)

        if iteration % 100 == 0:
            print("Done with iteration", iteration)