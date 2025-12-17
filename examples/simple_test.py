import torch
from echosampler.core import EchoSamplerProcessor

# Quick local smoke test using the HF-style LogitsProcessor API.
# We pass a dummy `input_ids` tensor to match the processors' signature.

torch.manual_seed(42)
logits = torch.randn(2, 50257) * 2  # Simulate batch=2, vocab=50257

processor_reality = EchoSamplerProcessor(dream_mode=False)
processor_dream = EchoSamplerProcessor(dream_mode=True)

batch = logits.size(0)
input_ids = torch.zeros((batch, 1), dtype=torch.long)

mod_logits_reality = processor_reality(input_ids, logits)
mod_logits_dream = processor_dream(input_ids, logits)

samples_reality = torch.multinomial(torch.softmax(mod_logits_reality, dim=-1), 1)
samples_dream = torch.multinomial(torch.softmax(mod_logits_dream, dim=-1), 1)

print("Reality mode sample:", samples_reality)
print("Dream mode sample:", samples_dream)
