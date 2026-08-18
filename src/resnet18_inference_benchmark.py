from torchvision.models import resnet18
from inference_benchmark import varying_batch_inference_benchmark, plot_latency_images, InferenceBenchmarkResult

model = resnet18(weights=None)

results = varying_batch_inference_benchmark(model, num_batch_sizes=7)
plot_latency_images(results)