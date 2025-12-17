import torch
from echosampler.core import EchoSampler

torch.manual_seed(42)
logits = torch.randn(2, 50257) * 2  # Simulate batch=2, vocab=50257

sampler_reality = EchoSampler(dream_mode=False)
sampler_dream = EchoSampler(dream_mode=True)

print("Reality mode sample:", sampler_reality.sample(logits))
print("Dream mode sample:", sampler_dream.sample(logits))
