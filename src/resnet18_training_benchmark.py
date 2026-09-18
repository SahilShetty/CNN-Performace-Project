import torch
from torch import nn
from torchvision.models import resnet18
from training_benchmark import training_loop, processing_times, TrainingBenchmarkConfig
from dataclasses import asdict

images = torch.rand(128, 3, 224, 224) # (images, channel, height, width) for PyTorch images
labels = torch.randint(0, 100, (128,))
dataset = torch.utils.data.TensorDataset(images, labels)

model = resnet18(weights=None)
config = TrainingBenchmarkConfig(
    model,
    torch.utils.data.DataLoader(dataset=dataset, batch_size=4),
    nn.CrossEntropyLoss(),
    torch.optim.SGD(params=model.parameters(), lr=0.1))

results = training_loop(config)
