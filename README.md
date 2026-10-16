# CNN Performance Benchmarking

A Python project for measuring how batch size affects convolutional neural network performance in PyTorch. It benchmarks inference latency and throughput, then breaks training time into individual operations to help identify where computation time is spent.

The current implementation focuses on CPU benchmarking, with ResNet-18 examples built using torchvision. It uses synthetic inputs so benchmarks can run without downloading a dataset or pretrained weights.

## Features

- Inference benchmarking with evaluation mode and `torch.inference_mode()`.
- Training benchmarks covering forward pass, loss calculation, gradient clearing, backward pass, and optimizer updates.
- Separate measurement of data loading and residual timing overhead.
- Configurable warmup iterations, measured iterations, and batch size sweeps.
- Raw timing samples, mean, median, and sample standard deviation.
- Matplotlib plots for inference latency and throughput, plus total training time and overhead.
- Model and optimizer state reset before each training batch size benchmark.

## Project files

| File | Purpose |
| --- | --- |
| `inference_benchmark.py` | Inference configuration, timing, result objects, and plotting functions. |
| `training_benchmark.py` | Training configuration, timing breakdowns, result objects, and plotting functions. |
| `resnet18_inference_benchmark.py` | Runnable ResNet-18 inference example. |
| `resnet18_training_benchmark.py` | Runnable ResNet-18 training example using cross-entropy loss and SGD. |

## Setup

From the project directory, create and activate a Python virtual environment:

```bash
python -m venv .venv
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

On macOS or Linux:

```bash
source .venv/bin/activate
```

Install the dependencies:

```bash
python -m pip install torch torchvision matplotlib
```

The scripts use CPU tensors by default. A GPU is not required.

## Run the benchmarks

Run each command from the directory containing the Python files.

### Inference

```bash
python resnet18_inference_benchmark.py
```

The supplied example benchmarks ResNet-18 with randomly initialized weights and random inputs of shape `(batch_size, 3, 224, 224)`. It uses 10 warmup iterations and 50 measured iterations for each batch size: 1, 2, 4, and 8.

The script prints throughput for each batch size and opens two plots:

- Mean inference latency per image, with standard deviation error bars.
- Throughput in images per second.

### Training

```bash
python resnet18_training_benchmark.py
```

The supplied example creates a synthetic dataset of 128 images and random labels, then benchmarks ResNet-18 with cross-entropy loss and SGD. It uses 10 warmup iterations and 50 measured iterations for each batch size: 1, 2, 4, 8, and 16.

The script prints mean total training time and overhead for each batch size, then opens plots of both metrics with standard deviation error bars. All training timings are in milliseconds per batch.

The synthetic data is intended to measure execution time. These runs do not evaluate model accuracy.

## What the metrics mean

### Inference

Each measured iteration times one forward pass over a complete batch. The elapsed time is divided by the batch size before it is stored:

```text
latency per image (ms) = batch forward-pass time (seconds) × 1000 / batch size
throughput (images/second) = 1000 / mean latency per image (ms)
```

`latencies_ms`, `mean_latency_ms`, `median_latency_ms`, and `stdev_latency_ms` therefore describe **average time per image within a batch**. This is an amortized measure: all images in the batch are processed together, so it does not represent the response time of an individual request within that batch.

Input creation and warmup are excluded from the measured inference timings.

### Training

Each measured iteration performs one training step on a batch:

| Result field | Measurement |
| --- | --- |
| `dataloader_times` | Time to retrieve the next batch, including restarting the iterator when needed. |
| `forward_times` | Model forward pass. |
| `loss_times` | Loss calculation. |
| `zero_grad_times` | Clearing gradients with `optimizer.zero_grad()`. |
| `backward_times` | Backpropagation with `loss.backward()`. |
| `optimizer_times` | Parameter update with `optimizer.step()`. |
| `total_training_times` | Training step from after batch retrieval through the optimizer update. |
| `overhead_times` | Total training time minus the five individually timed training operations. |

**Total training time excludes data loading.** The data loading timer runs separately, before the total training timer starts. In the supplied example, the dataset is already held in memory, so data loading measurements cover batch retrieval and collation rather than image files being read from disk.

Overhead is calculated for each iteration:

```text
overhead = total training time
           − (forward + loss + zero_grad + backward + optimizer)
```

This residual includes timer calls and Python execution between the timed operations. It is not a measurement of all framework overhead; work performed inside a timed operation remains part of that operation's measurement.

Each training metric is stored as a `TimingStatistics` object containing `timings_ms`, `mean_ms`, `median_ms`, and `stdev_ms`. These values are **per batch**, without division by batch size. To calculate amortized training time per image, divide a batch timing by the number of images in that batch.

## Configuration and results

Both configuration classes default to:

| Setting | Default | Meaning |
| --- | --- | --- |
| `warmup_iterations` | `10` | Iterations run before measurements begin. |
| `iterations` | `50` | Measured iterations per batch size. |
| `num_batch_sizes` | `5` | Number of batch sizes in the sweep: `2 ** i` for `i` from `0` to `num_batch_sizes - 1`. |

Use at least two measured iterations so a sample standard deviation can be calculated. Warmup iterations can be zero. The inference example overrides `num_batch_sizes` to `4`.

For example, customize an inference sweep:

```python
from torchvision.models import resnet18
from inference_benchmark import (
    InferenceBenchmarkConfig,
    varying_batch_inference_benchmark,
    plot_latency_images,
)

config = InferenceBenchmarkConfig(
    model=resnet18(weights=None),
    warmup_iterations=10,
    iterations=100,
    num_batch_sizes=5,
)

results = varying_batch_inference_benchmark(config)
plot_latency_images(results)

for result in results:
    print(
        f"Batch {result.batch_size}: "
        f"{result.mean_latency_ms:.2f} ms/image, "
        f"{result.throughput_images_per_sec:.2f} images/second"
    )
```

Edit `TrainingBenchmarkConfig` in the training example to change its iteration counts, loss function, or optimizer. The training sweep returns a list of `TrainingBenchmarkResults`; for example, `results[0].backward_times.mean_ms` gives the mean backward-pass time for the first batch size.

The sweep helpers use powers of two for batch sizes. For other batch sizes, use `single_batch_inference_benchmark(inputs, config)` with your own input tensor, or `single_batch_training_benchmark(dataloader, config)` with your own DataLoader.

## Benchmark methodology and current scope

- Inference uses evaluation mode and disables autograd. Training uses training mode and performs complete training steps during both warmup and measurement.
- Each training batch size starts from a deep copy of the same initial model and optimizer configuration. Warmup then updates that copy before measurement begins.
- Training is measured over a fixed number of iterations rather than epochs. The DataLoader iterator restarts when exhausted.
- The inference sweep uses fixed `3 × 224 × 224` inputs. Models requiring other shapes can use the single-batch function with suitable inputs.
- Timings use `time.perf_counter()` and currently target CPU execution. GPU benchmarking needs device handling and synchronization before these timings can be used reliably on CUDA.
- The example scripts display plots and print summaries. Results are also returned as Python dataclasses; automatic file export is not implemented.

Results depend on hardware, PyTorch configuration, CPU thread settings, and background activity. For comparisons, keep the environment consistent, record the hardware and software versions, and repeat runs. Error bars show variation across iterations within a run; they are not confidence intervals.

## Planned work

- CPU operator profiling with `torch.profiler` to investigate bottlenecks.
- Training throughput and peak memory measurements.
- GPU benchmarking with appropriate synchronization and CPU/GPU comparisons.
- Structured result export for comparisons across runs.
