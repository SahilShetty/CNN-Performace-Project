from torchvision.models import resnet18
from inference_benchmark import varying_batch_inference_benchmark

model = resnet18(weights=None)

varying_batch_inference_benchmark(model, diff_batch_sizes=5)