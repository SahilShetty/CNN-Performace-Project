from torchvision.models import resnet18
from inference_benchmark import varying_batch_inference_benchmark, plot_latency_images, InferenceBenchmarkConfig

model = resnet18(weights=None)
config = InferenceBenchmarkConfig(model, num_batch_sizes=4)

results = varying_batch_inference_benchmark(config)
plot_latency_images(results)